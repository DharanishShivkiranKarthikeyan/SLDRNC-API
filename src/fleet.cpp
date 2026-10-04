// fleet.cpp -- many robots (or simulations, or candidate motions) stepped in one call.  Each robot keeps its own state
// (features, self-correction) exactly as a session does; the network runs for a chunk of robots at once, every weight
// load shared across 4 robots, or the table 8 robots at a time.  Chunks are handed out to the thread pool's workers on
// demand, so fast cores take more of them.  The outputs equal independent sessions bit for bit: how robots are grouped
// never changes a robot's own arithmetic.  Nothing allocates after fleet_init.
#include <algorithm>
#include <atomic>
#include <cstring>

#include "internal.hpp"

namespace sldi {

static const int MAX_CHUNK = 48;                          // robots per chunk: a multiple of 4 (network) and of 8 (table)

void fleet_init(Fleet& f, const Model& m, size_t n, bool use_corr, bool protect, int max_delay, int threads) {
    f.m = &m;
    f.n = n;
    f.use_corr = use_corr;
    f.protect = protect;
    f.max_delay_opt = max_delay;
    const int ring = m.corr.on ? (max_delay > 0 ? max_delay : std::max(1, m.corr.delay_ticks)) : 0;   // small by default
    f.rob.assign(n, Session());
    for (size_t r = 0; r < n; r++) session_init(f.rob[r], m, use_corr, protect, ring, true);
    f.threads = threads > 0 ? threads : hw_threads();
    // about 8 chunks per worker, so fast cores take more and the last chunk is short; a multiple of the batch (4 robots
    // per network pass, 8 per table block)
    const int unit = m.table.on() ? 8 : 4;
    const size_t want = n / ((size_t)f.threads * 8);
    f.chunk = (int)std::max<size_t>(unit, std::min<size_t>(MAX_CHUNK, want / unit * unit));
    f.parts = (int)std::max<size_t>(1, std::min<size_t>((size_t)f.threads, (n + f.chunk - 1) / f.chunk));
    f.in_mean.assign(m.in_mean.begin(), m.in_mean.end());
    f.in_inv.resize(m.in_std.size());
    for (size_t i = 0; i < m.in_std.size(); i++) f.in_inv[i] = 1.0 / (double)m.in_std[i];
    const int ldx = m.trunk_in() + 16, mw = m.net.max_width() + 16, ldo = std::max(m.table.on() ? m.table.hp : m.net.out(), 1) + 16;
    f.xin.assign(f.parts, std::vector<float>((size_t)f.chunk * ldx));
    f.act.assign(f.parts, std::vector<float>((size_t)2 * f.chunk * mw));
    f.out.assign(f.parts, std::vector<float>((size_t)f.chunk * ldo));
    f.xs.assign(f.parts, std::vector<float>(24));
    f.feat.assign(f.parts, std::vector<double>(std::max(m.nfeat(), 1)));
    f.aff.assign(f.parts, std::vector<double>((size_t)f.chunk * std::max(m.n_aff, 1)));
    if (m.table.on()) {                                   // the constants of a whole-row table step
        f.tio_col = m.aff_col;
        f.tio_mean.clear(); f.tio_inv.clear();
        for (int i : m.aff_idx) { f.tio_mean.push_back(f.in_mean[i]); f.tio_inv.push_back(f.in_inv[i]); }
        f.tio_omean.assign(m.out_mean.begin(), m.out_mean.end());
        f.tio_ostd.assign(m.out_std.begin(), m.out_std.end());
        f.tio.ldx = m.input_size();
        f.tio.no = m.output_size();
        f.tio.na = m.n_aff;
        f.tio.aff_col = f.tio_col.data();
        f.tio.aff_mean = f.tio_mean.data();
        f.tio.aff_inv = f.tio_inv.data();
        f.tio.out_mean = f.tio_omean.empty() ? nullptr : f.tio_omean.data();
        f.tio.out_std = f.tio_ostd.empty() ? nullptr : f.tio_ostd.data();
    }
}

struct StepCtx { Fleet* f; const double* X; double* Y; std::atomic<size_t> next; };

static void step_chunk(Fleet& f, int part, size_t c0, int cn, const double* X, double* Y) {
    const Model& m = *f.m;
    const Kernels& k = kernels();
    const int in = m.input_size(), no = m.output_size(), na = std::max(m.n_aff, 1);
    const int ldx = m.trunk_in() + 16, mw = m.net.max_width() + 16;
    float* xin = f.xin[part].data();
    float* act = f.act[part].data();
    float* out = f.out[part].data();
    double* aff = f.aff[part].data();
    double* feat = f.feat[part].data();
    if (m.table.on() && !m.corr.on && k.table_rows8) {   // stateless: whole rows in, whole rows out, 8 robots per call
        int q = 0;
        for (; q + 8 <= cn; q += 8) k.table_rows8(m.table, f.tio, X + (c0 + q) * in, Y + (c0 + q) * no, out);
        for (; q < cn; q++) {
            float x3[3];
            table_prepare(m, X + (c0 + q) * in, f.in_mean.data(), f.in_inv.data(), aff);
            table_inputs(m.table, X + (c0 + q) * in, x3);
            table_lookup(m.table, x3, out);
            combine_outputs(m, out, aff, Y + (c0 + q) * no);
        }
        return;
    }
    if (m.table.on()) {                                   // no robot state until the correction: prepare per robot
        for (int q = 0; q < cn; q++) table_prepare(m, X + (c0 + q) * in, f.in_mean.data(), f.in_inv.data(), aff + (size_t)q * na);
    } else {
        for (int q = 0; q < cn; q++)
            session_prepare(f.rob[c0 + q], X + (c0 + q) * in, feat, f.in_mean.data(), f.in_inv.data(), xin + (size_t)q * ldx, aff + (size_t)q * na);
    }
    const float* res;
    int ldr;
    if (m.table.on()) {
        const Table& t = m.table;
        float* xs = f.xs[part].data();
        int q = 0;
        for (; q + 8 <= cn; q += 8) {
            for (int l = 0; l < 8; l++) {
                float x3[3];
                table_inputs(t, X + (c0 + q + l) * in, x3);
                xs[l] = x3[0]; xs[8 + l] = x3[1]; xs[16 + l] = x3[2];
            }
            k.table8(t, xs, out + (size_t)q * t.hp);
        }
        for (; q < cn; q++) {
            float x3[3];
            table_inputs(t, X + (c0 + q) * in, x3);
            table_lookup(t, x3, out + (size_t)q * t.hp);
        }
        res = out;
        ldr = t.hp;
    } else {
        const float* cur = xin;
        int ldc = ldx;
        for (size_t li = 0; li < m.net.L.size(); li++) {
            const Layer& l = m.net.L[li];
            float* z = act + (size_t)(li & 1) * f.chunk * mw;
            k.gemv_rows(cur, cn, ldc, l.in, l.W.data(), l.outp, l.b.data(), z, mw);
            if (l.act != SLDRNC_ACT_IDENTITY)
                for (int q = 0; q < cn; q++) activate_n(l.act, z + (size_t)q * mw, l.out);
            cur = z;
            ldc = mw;
        }
        res = cur;
        ldr = ldc;
    }
    if (m.corr.on) {
        for (int q = 0; q < cn; q++)
            session_finish(f.rob[c0 + q], X + (c0 + q) * in, res + (size_t)q * ldr, aff + (size_t)q * na, Y + (c0 + q) * no);
    } else {                                              // without correction finish touches no robot state
        for (int q = 0; q < cn; q++) combine_outputs(m, res + (size_t)q * ldr, aff + (size_t)q * na, Y + (c0 + q) * no);
    }
}

static void step_part(void* p, int i0, int i1) {          // each part is one worker with its own scratch; chunks on demand
    auto& c = *(StepCtx*)p;
    Fleet& f = *c.f;
    for (int part = i0; part < i1; part++)
        for (;;) {
            const size_t c0 = c.next.fetch_add((size_t)f.chunk, std::memory_order_relaxed);
            if (c0 >= f.n) break;
            step_chunk(f, part, c0, (int)std::min<size_t>((size_t)f.chunk, f.n - c0), c.X, c.Y);
        }
}

void fleet_step(Fleet& f, const double* X, double* Y) {
    StepCtx c{&f, X, Y, {0}};
    if (f.parts == 1) step_part(&c, 0, 1);
    else parallel_for(f.threads, f.parts, step_part, &c);
}

struct ObsCtx { Fleet* f; const double* Y; int delay; std::atomic<size_t> next; };
static void observe_part(void* p, int i0, int i1) {
    auto& c = *(ObsCtx*)p;
    Fleet& f = *c.f;
    const int no = f.m->output_size();
    for (int part = i0; part < i1; part++)
        for (;;) {
            const size_t c0 = c.next.fetch_add((size_t)f.chunk, std::memory_order_relaxed);
            if (c0 >= f.n) break;
            const size_t c1 = std::min(f.n, c0 + (size_t)f.chunk);
            for (size_t r = c0; r < c1; r++) session_observe(f.rob[r], c.Y + r * no, c.delay);
        }
}

void fleet_observe(Fleet& f, const double* Y, int delay) {
    if (!f.m->corr.on) return;
    ObsCtx c{&f, Y, delay, {0}};
    if (f.parts == 1) observe_part(&c, 0, 1);
    else parallel_for(f.threads, f.parts, observe_part, &c);
}

}  // namespace sldi
