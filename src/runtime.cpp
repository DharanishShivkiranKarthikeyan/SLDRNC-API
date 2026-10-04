// runtime.cpp -- kernel dispatch, thread pool, RNG, activations and the network forward pass.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>

#include "internal.hpp"

#if defined(__x86_64__) || defined(_M_X64)
#define SLDI_X86 1
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#include <emmintrin.h>
#endif
#endif

namespace sldi {

const Kernels* kernels_generic();
#if SLDI_X86
const Kernels* kernels_avx2();
static bool cpu_has_avx2_fma() {
#if defined(_MSC_VER)
    int r[4];
    __cpuid(r, 1);
    const bool fma = (r[2] >> 12) & 1, osx = (r[2] >> 27) & 1, avx = (r[2] >> 28) & 1;
    if (!(fma && osx && avx)) return false;
    if ((_xgetbv(0) & 6) != 6) return false;
    __cpuidex(r, 7, 0);
    return (r[1] >> 5) & 1;
#else
    unsigned a, b, c, d;
    if (!__get_cpuid(1, &a, &b, &c, &d)) return false;
    const bool fma = (c >> 12) & 1, osx = (c >> 27) & 1, avx = (c >> 28) & 1;
    if (!(fma && osx && avx)) return false;
    unsigned lo, hi;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    if ((lo & 6) != 6) return false;
    if (!__get_cpuid_count(7, 0, &a, &b, &c, &d)) return false;
    return (b >> 5) & 1;
#endif
}
#endif

static const Kernels* pick(const char** name) {
#if SLDI_X86
    if (cpu_has_avx2_fma()) { *name = "avx2-fma"; return kernels_avx2(); }
    *name = "sse2";
#elif defined(__aarch64__) || defined(_M_ARM64)
    *name = "neon";
#else
    *name = "generic";
#endif
    return kernels_generic();
}
static const char* g_kname = nullptr;
const Kernels& kernels() {
    static const Kernels* k = pick(&g_kname);
    return *k;
}
const char* kernel_name() { kernels(); return g_kname; }

// ------------------------------------------------------------------------------------------------ RNG
static uint64_t splitmix(uint64_t& x) {
    uint64_t z = (x += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
Rng::Rng(uint64_t seed) { for (auto& v : s) v = splitmix(seed); }
uint64_t Rng::next() {   // xoshiro256**
    const uint64_t r = ((s[1] * 5) << 7 | (s[1] * 5) >> 57) * 9;
    const uint64_t t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t;
    s[3] = (s[3] << 45) | (s[3] >> 19);
    return r;
}

// ------------------------------------------------------------------------------------------------ threads
int hw_threads() {
    const unsigned n = std::thread::hardware_concurrency();
    return n ? (int)n : 1;
}
namespace {
inline void cpu_relax() {
#if defined(SLDI_X86)
    _mm_pause();
#elif defined(__aarch64__)
    __asm__ __volatile__("yield");
#endif
}
// Workers and the caller spin for up to SPIN before sleeping: back-to-back calls (a fleet stepped in a loop, training
// steps) then cost no thread wake-ups, and an idle pool still sleeps.
struct Pool {
    static constexpr std::chrono::microseconds SPIN{200};
    std::vector<std::thread> th;
    std::mutex mu, call;                                  // mu: only for sleeping and waking
    std::condition_variable cv, done;
    std::atomic<uint64_t> job{0};                         // generation << 32 | parts: read in one load, so a consistent pair
    std::atomic<int> pending{0};
    std::atomic<bool> stop{false};
    int n = 0, parts = 0;                                 // written before `job`; stable while any part of the job runs
    void (*fn)(void*, int, int) = nullptr;
    void* ctx = nullptr;
    explicit Pool(int workers) {
        for (int i = 0; i < workers; i++) th.emplace_back([this, i] { run(i + 1); });
    }
    ~Pool() {
        { std::lock_guard<std::mutex> g(mu); stop = true; }
        cv.notify_all();
        for (auto& t : th) t.join();
    }
    void range(int part) {
        const int i0 = (int)((int64_t)n * part / parts), i1 = (int)((int64_t)n * (part + 1) / parts);
        if (i1 > i0) fn(ctx, i0, i1);
    }
    void run(int id) {
        uint32_t seen = 0;
        for (;;) {
            uint64_t j = job.load(std::memory_order_acquire);
            const auto t0 = std::chrono::steady_clock::now();
            for (unsigned spins = 1; (uint32_t)(j >> 32) == seen && !stop.load(std::memory_order_relaxed); spins++) {
                if ((spins & 255) == 0 && std::chrono::steady_clock::now() - t0 > SPIN) {
                    std::unique_lock<std::mutex> lk(mu);
                    cv.wait(lk, [&] { return stop.load() || (uint32_t)(job.load() >> 32) != seen; });
                }
                cpu_relax();
                j = job.load(std::memory_order_acquire);
            }
            if (stop.load()) return;
            seen = (uint32_t)(j >> 32);
            if (id >= (int)(uint32_t)j) continue;             // not part of this job
            range(id);
            if (pending.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                { std::lock_guard<std::mutex> g(mu); }        // the caller may be about to sleep on `done`
                done.notify_one();
            }
        }
    }
    void go(int p, int count, void (*f)(void*, int, int), void* c) {
        std::lock_guard<std::mutex> one(call);
        parts = p; n = count; fn = f; ctx = c;
        pending.store(p - 1, std::memory_order_relaxed);
        const uint32_t g = (uint32_t)(job.load(std::memory_order_relaxed) >> 32) + 1;
        job.store((uint64_t)g << 32 | (uint32_t)p, std::memory_order_release);
        { std::lock_guard<std::mutex> lk(mu); }           // a worker checking the job before it sleeps is now asleep
        cv.notify_all();
        range(0);
        const auto t0 = std::chrono::steady_clock::now();
        for (unsigned spins = 1; pending.load(std::memory_order_acquire) != 0; spins++) {
            if ((spins & 255) == 0 && std::chrono::steady_clock::now() - t0 > SPIN) {
                std::unique_lock<std::mutex> lk(mu);
                done.wait(lk, [&] { return pending.load(std::memory_order_acquire) == 0; });
                break;
            }
            cpu_relax();
        }
    }
};
Pool& pool() {
    static Pool p(std::max(0, hw_threads() - 1));
    return p;
}
}  // namespace
void parallel_for(int threads, int n, void (*fn)(void*, int, int), void* ctx) {
    int p = std::min(std::max(1, threads), std::min(n, hw_threads()));
    if (p <= 1) { if (n > 0) fn(ctx, 0, n); return; }
    pool().go(p, n, fn, ctx);
}

// ------------------------------------------------------------------------------------------------ network
float activate(int act, float z) {
    switch (act) {
        case SLDRNC_ACT_RELU: return z > 0 ? z : 0.0f;
        case SLDRNC_ACT_RELU2: { const float r = z > 0 ? z : 0.0f; return r * r; }
        case SLDRNC_ACT_LEAKY_RELU: return z > 0 ? z : 0.01f * z;
        case SLDRNC_ACT_TANH: return std::tanh(z);
        case SLDRNC_ACT_SIGMOID: return 1.0f / (1.0f + std::exp(-z));
        case SLDRNC_ACT_GELU: return 0.5f * z * (1.0f + std::tanh(0.7978845608f * (z + 0.044715f * z * z * z)));
        case SLDRNC_ACT_SILU: return z / (1.0f + std::exp(-z));
        default: return z;
    }
}
void activate_n(int act, float* z, int n) {
    switch (act) {                                    // the piecewise ones as plain loops the compiler can vectorise
        case SLDRNC_ACT_IDENTITY: return;
        case SLDRNC_ACT_RELU: for (int i = 0; i < n; i++) z[i] = z[i] > 0 ? z[i] : 0.0f; return;
        case SLDRNC_ACT_RELU2: for (int i = 0; i < n; i++) { const float r = z[i] > 0 ? z[i] : 0.0f; z[i] = r * r; } return;
        case SLDRNC_ACT_LEAKY_RELU: for (int i = 0; i < n; i++) z[i] = z[i] > 0 ? z[i] : 0.01f * z[i]; return;
        default: for (int i = 0; i < n; i++) z[i] = activate(act, z[i]);
    }
}
void set_layer(Layer& l, int in, int out, int act, const float* W, const float* b) {
    l.in = in; l.out = out; l.act = act;
    l.outp = (out + 15) & ~15;
    l.W.assign((size_t)in * l.outp, 0.0f);
    l.b.assign(l.outp, 0.0f);
    if (W) for (int i = 0; i < in; i++) std::memcpy(&l.W[(size_t)i * l.outp], W + (size_t)i * out, sizeof(float) * out);
    if (b) std::memcpy(l.b.data(), b, sizeof(float) * out);
}
int Net::max_width() const {
    int w = in();
    for (const auto& l : L) w = std::max(w, l.outp);
    return w;
}
int64_t Net::macs() const { int64_t m = 0; for (const auto& l : L) m += (int64_t)l.in * l.out; return m; }
int64_t Net::params() const { int64_t m = 0; for (const auto& l : L) m += (int64_t)l.in * l.out + l.out; return m; }
void Net::forward(const float* x, float* y, float* scratch) const {
    const int mw = max_width();
    float* a = scratch;
    float* b = scratch + mw;
    const float* cur = x;
    const Kernels& k = kernels();
    for (size_t li = 0; li < L.size(); li++) {
        const Layer& l = L[li];
        float* z = cur == a ? b : a;                 // never write into the buffer being read
        k.gemv(cur, l.in, l.W.data(), l.outp, l.b.data(), z);
        activate_n(l.act, z, l.out);
        for (int o = l.out; o < l.outp; o++) z[o] = 0.0f;
        cur = z;
    }
    std::memcpy(y, cur, sizeof(float) * out());
}

}  // namespace sldi
