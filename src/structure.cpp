// structure.cpp -- structured FULL models (outputs affine in chosen channels) and tables (a model without history whose
// non-affine inputs are at most 3 raw values, compiled into a grid of its network's outputs).
//
// table_lookup is the reference arithmetic for tables: kernels_table_avx2.cpp repeats it 8 robots at a time and must
// give the same bits, so this file is compiled without floating-point contraction and every fused step is a std::fma.
#include <algorithm>
#include <cmath>
#include <cstring>

#include "internal.hpp"

namespace sldi {

static const double TWO_PI = 6.283185307179586476925;

void setup_structure(Model& m) {
    m.trunk_idx.clear();
    m.aff_idx.clear();
    m.aff_col.clear();
    m.n_aff = 0;
    if (m.mode != SLDRNC_MODE_FULL) return;
    bool any = false;
    for (const auto& c : m.schema.ch) any = any || c.affine;
    if (!any) return;
    std::vector<char> is_aff(m.fs.nf, 0);
    for (size_t k = 0; k < m.schema.ch.size(); k++) {
        const Channel& c = m.schema.ch[k];
        if (!c.affine) continue;
        for (int i = 0; i < c.size; i++) {                // a channel's current values are its first features
            const int idx = m.fs.ch_feat[k] + i;
            is_aff[idx] = 1;
            m.aff_idx.push_back(idx);
            m.aff_col.push_back(c.offset + i);
        }
    }
    for (int i = 0; i < m.fs.nf; i++) if (!is_aff[i]) m.trunk_idx.push_back(i);
    m.n_aff = (int)m.aff_idx.size();
}

int64_t Model::macs() const {
    const int64_t combine = (int64_t)output_size() * n_aff;
    if (table.on()) return 4 * (int64_t)table.hp + combine;     // 4 nodes per lookup
    return net.macs() + combine;
}

// ------------------------------------------------------------------------------------------------ lookup
void table_lookup(const Table& t, const float* x3, float* heads) {
    float f[3];
    int s[3];
    int n0 = 0;
    for (int d = 0; d < 3; d++) {
        const float x = x3[d];
        if (t.periodic[d]) {
            float w = (x - t.lo[d]) * t.scale[d];
            const float q = std::floor(w * t.inv_n[d]);
            const float wrap = t.nfl[d] * q;
            w = w - wrap;                                 // into [0, n)
            const float wf = std::floor(w);
            int k = (int)wf;
            if (k > t.n[d] - 1) k -= t.n[d];
            if (k < 0) k += t.n[d];
            int k1 = k + 1;
            if (k1 == t.n[d]) k1 = 0;
            f[d] = w - wf;
            n0 += k * t.stride[d];
            s[d] = (k1 - k) * t.stride[d];
        } else {
            float u = (x - t.lo[d]) * t.scale[d];
            u = u > 0.0f ? u : 0.0f;                      // = _mm256_max_ps(u, 0)
            u = u < t.umax[d] ? u : t.umax[d];            // = _mm256_min_ps(u, umax)
            const float uf = std::floor(u);
            f[d] = u - uf;
            n0 += (int)uf * t.stride[d];
            s[d] = t.stride[d];
        }
    }
    // Kuhn simplex: walk the cell's corners in order of decreasing fraction
    auto cs = [&](int a, int b) {
        if (f[a] < f[b]) { std::swap(f[a], f[b]); std::swap(s[a], s[b]); }
    };
    cs(0, 1); cs(1, 2); cs(0, 1);
    const int n1 = n0 + s[0], n2 = n1 + s[1], n3 = n2 + s[2];
    const float w0 = 1.0f - f[0], w1 = f[0] - f[1], w2 = f[1] - f[2], w3 = f[2];
    const int hp = t.hp;
    const float *T0 = &t.T[(size_t)n0 * hp], *T1 = &t.T[(size_t)n1 * hp], *T2 = &t.T[(size_t)n2 * hp], *T3 = &t.T[(size_t)n3 * hp];
    for (int h = 0; h < hp; h++) {
        float a = w0 * T0[h];
        a = std::fma(w1, T1[h], a);
        a = std::fma(w2, T2[h], a);
        heads[h] = std::fma(w3, T3[h], a);
    }
}

// the portable 8-robot form (CPUs without AVX2): one lookup at a time
void table8_scalar(const Table& t, const float* xs, float* heads) {
    for (int l = 0; l < 8; l++) {
        const float x3[3] = {xs[l], xs[8 + l], xs[16 + l]};
        table_lookup(t, x3, heads + (size_t)l * t.hp);
    }
}

// ------------------------------------------------------------------------------------------------ compile
struct FillCtx { const Model* m; Table* t; double step[3]; };
static void fill_task(void* p, int i0, int i1) {
    auto& c = *(FillCtx*)p;
    const Model& m = *c.m;
    Table& t = *c.t;
    const int W = m.schema.width, nf = m.fs.nf, H = m.net.out();
    std::vector<double> x(W, 0.0), feat(nf), mean(m.in_mean.begin(), m.in_mean.end()), inv(nf);
    for (int i = 0; i < nf; i++) inv[i] = 1.0 / (double)m.in_std[i];
    std::vector<float> xin(m.trunk_in() + 16), out(H + 16), scratch((size_t)2 * m.net.max_width() + 32);
    FeatState st;
    feat_init(m.schema, m.fs, st);
    for (int node = i0; node < i1; node++) {
        const int idx[3] = {node / t.stride[0], (node / t.stride[1]) % t.n[1], node % t.n[2]};
        for (int d = 0; d < t.dims; d++)
            x[t.col[d]] = (double)t.lo[d] + idx[d] * (t.periodic[d] ? TWO_PI / t.n[d] : c.step[d]);
        feat_step(m.schema, m.fs, st, x.data(), feat.data());
        if (m.trunk_idx.empty())
            for (int i = 0; i < nf; i++) xin[i] = (float)((feat[i] - mean[i]) * inv[i]);
        else
            for (size_t k = 0; k < m.trunk_idx.size(); k++) { const int i = m.trunk_idx[k]; xin[k] = (float)((feat[i] - mean[i]) * inv[i]); }
        m.net.forward(xin.data(), out.data(), scratch.data());
        std::memcpy(&t.T[(size_t)node * t.hp], out.data(), sizeof(float) * H);
    }
}

std::unique_ptr<Model> table_compile(const Model& src, int n_grid, const int* grid) {
    if (src.mode != SLDRNC_MODE_FULL) fail(SLDRNC_ERROR_STATE, "only FULL models can be compiled into a table");
    if (src.schema.history) fail(SLDRNC_ERROR_STATE, "a table needs a model trained without history (turn the schema's history off)");
    if (src.table.on()) fail(SLDRNC_ERROR_STATE, "this model is already a table");
    std::vector<int> cols, per;
    for (const auto& c : src.schema.ch) {
        if (c.affine) continue;
        for (int i = 0; i < c.size; i++) { cols.push_back(c.offset + i); per.push_back(c.kind == SLDRNC_INPUT_ANGLE ? 1 : 0); }
    }
    if (cols.empty() || cols.size() > 3)
        fail(SLDRNC_ERROR_STATE, "a table needs 1 to 3 non-affine input values; this model has " + std::to_string(cols.size()) +
                                     " (declare the other channels affine, or keep the network)");
    if (n_grid != 0 && n_grid != (int)cols.size())
        fail(SLDRNC_ERROR_ARGUMENT, "give one grid size per table dimension (" + std::to_string(cols.size()) + ")");
    if (src.raw_lo.size() != (size_t)src.schema.width) fail(SLDRNC_ERROR_STATE, "the model has no recorded input ranges (retrain it with this version)");
    auto m = std::make_unique<Model>(src);
    Table& t = m->table;
    t.dims = (int)cols.size();
    FillCtx fc{m.get(), &t, {0, 0, 0}};
    for (int d = 0; d < 3; d++) {
        if (d >= t.dims) {                                // unused: 2 identical layers, coordinate 0
            t.n[d] = 2; t.col[d] = -1; t.periodic[d] = 0; t.lo[d] = 0.0f; t.scale[d] = 1.0f;
            t.umax[d] = std::nextafter(1.0f, 0.0f); t.nfl[d] = 2.0f; t.inv_n[d] = 0.5f;
            continue;
        }
        const int n = grid && grid[d] > 0 ? grid[d] : (per[d] ? 128 : 64);   // defaults: the PMSM grid study (round 15)
        if (n < (per[d] ? 4 : 2) || n > 4096) fail(SLDRNC_ERROR_ARGUMENT, "grid size " + std::to_string(n) + " out of range (2..4096; angles 4..4096)");
        t.n[d] = n;
        t.col[d] = cols[d];
        t.periodic[d] = per[d];
        t.nfl[d] = (float)n;
        t.inv_n[d] = (float)(1.0 / n);
        if (per[d]) {                                     // one turn from the training range's start: the network also sees
            t.lo[d] = m->raw_lo[cols[d]];                 // the raw angle, so evaluate it where it was trained
            t.scale[d] = (float)(n / TWO_PI);
        } else {
            double lo = m->raw_lo[cols[d]], hi = m->raw_hi[cols[d]];
            if (!(hi > lo)) hi = lo + 1e-6;
            fc.step[d] = (hi - lo) / (n - 1);
            t.lo[d] = (float)lo;
            t.scale[d] = (float)(1.0 / fc.step[d]);
            t.umax[d] = std::nextafter((float)(n - 1), 0.0f);
        }
    }
    t.stride[2] = 1;
    t.stride[1] = t.n[2];
    t.stride[0] = t.n[1] * t.n[2];
    const int H = m->net.out();
    t.hp = (H + 7) / 8 * 8;
    const double bytes = (double)t.nodes() * t.hp * 4;
    if (bytes > 2e9) fail(SLDRNC_ERROR_ARGUMENT, "the table would need " + std::to_string((long long)(bytes / 1e6)) + " MB; use smaller grids");
    t.T.assign(t.nodes() * t.hp, 0.0f);
    parallel_for(hw_threads(), (int)t.nodes(), fill_task, &fc);
    return m;
}

}  // namespace sldi
