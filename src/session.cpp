// session.cpp -- per-robot run-time state: features (FULL), network or table, self-correction.  No allocation after init.
// A step has three stages -- prepare (features, network input, affine values), the network or table, finish (combine,
// scale, correct) -- so a fleet can run the middle stage for many robots at once with exactly the same arithmetic.
#include <algorithm>
#include <cmath>
#include <cstring>

#include "internal.hpp"

namespace sldi {

void session_init(Session& s, const Model& m, bool use_corr, bool protect, int max_delay, bool lean) {
    s.m = &m;
    s.use_corr = use_corr && m.corr.on;
    s.protect = protect;
    const int nf = m.nfeat(), no = m.output_size();
    // late measurements: keep the last ring_cap steps' regressors and standardised predictions
    s.max_delay_opt = max_delay;
    s.ring_cap = m.corr.on ? (max_delay > 0 ? max_delay : std::max(256, m.corr.delay_ticks)) : 0;
    s.n_steps = 0;
    s.hring.assign((size_t)s.ring_cap * no * 4, 0.0);
    s.oring.assign((size_t)s.ring_cap * no, 0.0);
    if (m.mode == SLDRNC_MODE_FULL) feat_init(m.schema, m.fs, s.fst);
    corr_init(m.corr, s.cst, no);
    s.o_std.assign(no, 0.0);
    s.c.assign(no, 0.0);
    s.h.assign((size_t)no * 4, 0.0);
    if (lean) return;
    const int net_out = m.table.on() ? m.table.hp : m.net.out();
    s.feat.assign(nf, 0.0);
    s.xin.assign(m.trunk_in() + 16, 0.0f);
    s.scratch.assign((size_t)2 * m.net.max_width() + 32, 0.0f);
    s.yout.assign(std::max(net_out, no) + 16, 0.0f);
    s.aff.assign(m.n_aff, 0.0);
    s.in_mean.assign(m.in_mean.begin(), m.in_mean.end());
    s.in_inv.resize(m.in_std.size());
    for (size_t i = 0; i < m.in_std.size(); i++) s.in_inv[i] = 1.0 / (double)m.in_std[i];
}

void table_prepare(const Model& m, const double* x, const double* mean, const double* inv, double* aff) {
    for (int k = 0; k < m.n_aff; k++) {                   // no history, and the affine features are exact copies of raw
        const int i = m.aff_idx[k];                       // input values; the table reads its own inputs raw
        aff[k] = (x[m.aff_col[k]] - mean[i]) * inv[i];
    }
}

void session_prepare(Session& s, const double* x, double* feat, const double* mean, const double* inv, float* xin, double* aff) {
    const Model& m = *s.m;
    const int nf = m.nfeat();
    const double* f = x;
    if (m.table.on()) { table_prepare(m, x, mean, inv, aff); return; }
    if (m.mode == SLDRNC_MODE_FULL) {
        feat_step(m.schema, m.fs, s.fst, x, feat);
        f = feat;
    }
    for (int k = 0; k < m.n_aff; k++) { const int i = m.aff_idx[k]; aff[k] = (f[i] - mean[i]) * inv[i]; }
    if (!m.trunk_idx.empty()) {
        const int nt = (int)m.trunk_idx.size();
        for (int t = 0; t < nt; t++) { const int i = m.trunk_idx[t]; xin[t] = (float)((f[i] - mean[i]) * inv[i]); }
    } else if (!m.in_mean.empty()) {
        for (int i = 0; i < nf; i++) xin[i] = (float)((f[i] - mean[i]) * inv[i]);   // in double: inputs far from their mean lose no precision
    } else {
        for (int i = 0; i < nf; i++) xin[i] = (float)f[i];
    }
}

void combine_outputs(const Model& m, const float* out, const double* aff, double* y) {
    const int no = m.output_size();
    for (int j = 0; j < no; j++) {
        double raw = out[j];
        for (int k = 0; k < m.n_aff; k++) raw += aff[k] * (double)out[(size_t)no * (k + 1) + j];   // outputs affine in a
        y[j] = m.out_mean.empty() ? raw : raw * m.out_std[j] + m.out_mean[j];
    }
}

void session_finish(Session& s, const double* x, const float* out, const double* aff, double* y) {
    const Model& m = *s.m;
    const int no = m.output_size();
    combine_outputs(m, out, aff, y);
    if (m.corr.on) {
        for (int j = 0; j < no; j++) s.o_std[j] = (y[j] - m.corr.mean[j]) / m.corr.sd[j];
        corr_regress(m.corr, s.o_std.data(), x, s.h.data(), no);
        const size_t slot = (size_t)(s.n_steps % (uint64_t)s.ring_cap);
        std::memcpy(&s.hring[slot * no * 4], s.h.data(), sizeof(double) * no * 4);
        std::memcpy(&s.oring[slot * no], s.o_std.data(), sizeof(double) * no);
        s.n_steps++;
        s.cst.primed = true;
        if (s.use_corr) {
            corr_apply(m.corr, s.cst, s.h.data(), s.c.data(), no, s.protect);
            for (int j = 0; j < no; j++) y[j] += s.c[j] * m.corr.sd[j];
        }
    }
}

void table_inputs(const Table& t, const double* x, float* x3) {
    for (int d = 0; d < 3; d++) x3[d] = t.col[d] >= 0 ? (float)x[t.col[d]] : 0.0f;
}

void session_step(Session& s, const double* x, double* y) {
    const Model& m = *s.m;
    session_prepare(s, x, s.feat.data(), s.in_mean.data(), s.in_inv.data(), s.xin.data(), s.aff.data());
    if (m.table.on()) {
        float x3[3];
        table_inputs(m.table, x, x3);
        table_lookup(m.table, x3, s.yout.data());
    } else {
        m.net.forward(s.xin.data(), s.yout.data(), s.scratch.data());
    }
    session_finish(s, x, s.yout.data(), s.aff.data(), y);
}

void session_observe(Session& s, const double* y_meas, int delay) {
    const Model& m = *s.m;
    if (!m.corr.on || s.n_steps < (uint64_t)delay) return;   // nothing predicted yet for that tick
    const int no = m.output_size();
    const size_t slot = (size_t)((s.n_steps - (uint64_t)delay) % (uint64_t)s.ring_cap);
    const double* h = &s.hring[slot * no * 4];
    const double* o = &s.oring[slot * no];
    double* r = s.c.data();                               // reused: the applied correction is no longer needed
    for (int j = 0; j < no; j++) r[j] = (y_meas[j] - m.corr.mean[j]) / m.corr.sd[j] - o[j];
    const unsigned char* valid = s.protect ? corr_screen(m.corr, s.cst, y_meas, no) : nullptr;
    corr_update(m.corr, s.cst, h, r, no, s.protect, valid);
}

}  // namespace sldi
