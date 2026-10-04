// latency.cpp -- time the shipped library: one session, one tick at a time (step + observe), over a recorded stream.
//   latency <model.sldm> <stream.bin> [repeats] [logical processor, default 0, -1 = unpinned] [threads] [observe 1|0] [fleet robots]
// fleet robots > 0: a Fleet of that many robots stepped together on `threads` threads (0 = all) -> robot-ticks per second.
// With threads > 1: every thread runs its own sessions over the stream (many robots at once) -> robot-ticks per second.
// observe 0: step only, no measurements (e.g. a motion planner's rollouts, which never see the real outputs).
// stream.bin: int32 ticks, int32 input columns, int32 output columns, then ticks rows of inputs and of outputs (float64).
// Prints p50 / p99 / mean microseconds per tick (median over repeats of each statistic).
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

#include <sldrnc/sldrnc.hpp>
#if defined(_WIN32)
#include <windows.h>
static void pin(int lp) { SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << lp); SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST); }
#elif defined(__linux__)
#include <pthread.h>
#include <sched.h>
static void pin(int lp) { cpu_set_t c; CPU_ZERO(&c); CPU_SET(lp, &c); pthread_setaffinity_np(pthread_self(), sizeof c, &c); }
#else
static void pin(int) {}
#endif

int main(int argc, char** argv) {
    if (argc < 3) { std::printf("usage: latency <model.sldm> <stream.bin> [repeats]\n"); return 2; }
    const int reps = argc > 3 ? std::atoi(argv[3]) : 5;
    const int lp = argc > 4 ? std::atoi(argv[4]) : 0;   // logical processor to pin to (-1 = no pinning)
    if (lp >= 0) pin(lp);
    FILE* f = std::fopen(argv[2], "rb");
    if (!f) { std::printf("cannot open %s\n", argv[2]); return 1; }
    int hdr[3];
    if (std::fread(hdr, 4, 3, f) != 3) return 1;
    const size_t T = hdr[0], ni = hdr[1], no = hdr[2];
    std::vector<double> X(T * ni), Y(T * no);
    if (std::fread(X.data(), 8, X.size(), f) != X.size() || std::fread(Y.data(), 8, Y.size(), f) != Y.size()) return 1;
    std::fclose(f);
    sldrnc::Model model = sldrnc::Model::load(argv[1]);
    const sldrnc::Info info = model.info();
    if ((size_t)info.input_size != ni || (size_t)info.output_size != no) { std::printf("stream does not match the model\n"); return 1; }
    const int threads = argc > 5 ? std::atoi(argv[5]) : 1;
    const bool observe = argc > 6 ? std::atoi(argv[6]) != 0 : true;
    const long robots = argc > 7 ? std::atol(argv[7]) : 0;
    if (robots > 0) {                                // fleet: `robots` robots stepped together on `threads` threads for ~2 s
        sldrnc::Fleet fl = model.fleet((size_t)robots, true, true, 0, threads);
        std::vector<double> Xb[2], Yb[2], out((size_t)robots * no);   // two input blocks, alternated (they stay in cache)
        for (int b = 0; b < 2; b++) {
            Xb[b].resize((size_t)robots * ni);
            Yb[b].resize((size_t)robots * no);
            for (long r = 0; r < robots; r++) {
                const size_t t = ((size_t)b * 7919 + (size_t)r * 997) % T;
                std::copy(&X[t * ni], &X[t * ni] + ni, &Xb[b][(size_t)r * ni]);
                std::copy(&Y[t * no], &Y[t * no] + no, &Yb[b][(size_t)r * no]);
            }
        }
        for (int w = 0; w < 4; w++) fl.step(Xb[w & 1].data(), out.data());
        long long ticks = 0;
        const auto t0 = std::chrono::steady_clock::now();
        double sec = 0;
        for (int b = 0; sec < 2.0; b ^= 1) {
            fl.step(Xb[b].data(), out.data());
            if (observe) fl.observe(Yb[b].data());
            ticks += robots;
            sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        }
        std::printf("%s: fleet of %ld, %d threads%s: %.3g robot-ticks per second, %.3g outputs per second\n", argv[1], robots, threads,
                    observe ? "" : " (step only)", ticks / sec, ticks / sec * no);
        return 0;
    }
    if (threads > 1) {                               // throughput: independent sessions on all threads for ~2 s
        std::atomic<long long> total{0};
        std::vector<std::thread> th;
        const auto t0 = std::chrono::steady_clock::now();
        for (int k = 0; k < threads; k++)
            th.emplace_back([&, k] {
                pin(k);
                std::vector<double> yy(no);
                long long n = 0;
                double snk = 0;
                while (std::chrono::steady_clock::now() - t0 < std::chrono::seconds(2)) {
                    sldrnc::Session s = model.session();
                    for (size_t t = (size_t)k * 997 % T, c = 0; c < 8000; c++, t = (t + 1) % T) {
                        s.step(&X[t * ni], yy.data());
                        if (observe) s.observe(&Y[t * no]);
                        snk += yy[0];
                        n++;
                    }
                }
                total += n + (snk == 1.23e300 ? 1 : 0);
            });
        for (auto& t : th) t.join();
        const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("%s: %d threads, independent sessions%s: %.3g robot-ticks per second\n", argv[1], threads, observe ? "" : " (step only)",
                    total.load() / sec);
        return 0;
    }
    std::vector<double> p50, p99, mean;
    std::vector<double> y(no), dt(T);
    double sink = 0;
    for (int r = 0; r < reps + 1; r++) {             // first pass warms caches, not counted
        sldrnc::Session s = model.session();
        for (size_t t = 0; t < T; t++) {
            const auto a = std::chrono::steady_clock::now();
            s.step(&X[t * ni], y.data());
            if (observe) s.observe(&Y[t * no]);
            const auto b = std::chrono::steady_clock::now();
            dt[t] = std::chrono::duration<double, std::micro>(b - a).count();
            sink += y[0];
        }
        if (r == 0) continue;
        std::vector<double> d(dt);
        std::sort(d.begin(), d.end());
        double m = 0;
        for (double v : d) m += v;
        p50.push_back(d[T / 2]);
        p99.push_back(d[(size_t)(0.99 * (T - 1))]);
        mean.push_back(m / T);
    }
    auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
    std::printf("%s: %lld MACs/tick | per tick (%s): p50 %.2f us, p99 %.2f us, mean %.2f us | %zu ticks x %d repeats%s\n", argv[1],
                info.macs_per_tick, observe ? "step + observe" : "step only", med(p50), med(p99), med(mean), T, reps, sink == 1.234567e300 ? " " : "");
    return 0;
}
