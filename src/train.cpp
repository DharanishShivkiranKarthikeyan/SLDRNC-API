// train.cpp -- FULL-mode training (features -> network -> self-correction) and CORRECTION-mode fitting.
// Optimiser: Adam with a one-cycle schedule (learning rate and beta1), global gradient-norm clipping at 1, decoupled
// weight decay on weights, best validation checkpoint every 100 steps.  Network: 2 hidden layers, squared ReLU.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

#include "internal.hpp"

namespace sldi {

static const double PI = 3.14159265358979323846;
static double cosann(double a, double b, double p) { return b + (a - b) / 2.0 * (std::cos(PI * p) + 1.0); }
static void onecycle(int step, int total, double max_lr, double& lr, double& b1) {
    const double init = max_lr / 25.0, mn = init / 3e3, e1 = 0.15 * total - 1.0, e2 = total - 1.0;
    if (step <= e1) { const double p = step / e1; lr = cosann(init, max_lr, p); b1 = cosann(0.95, 0.85, p); return; }
    const double p = (step - e1) / (e2 - e1);
    lr = cosann(max_lr, mn, p);
    b1 = cosann(0.85, 0.95, p);
}

static void check_finite(const double* a, size_t n, const char* what) {
    for (size_t i = 0; i < n; i++)
        if (!std::isfinite(a[i])) fail(SLDRNC_ERROR_DATA, std::string(what) + " contains NaN or infinity at element " + std::to_string(i));
}

// ------------------------------------------------------------------------------------------------ batch forward
struct BatchFwdCtx { const Net* net; const float* X; int ldx; float* Y; int ldy; int n; };
static void batch_fwd_task(void* p, int r0, int r1) {
    auto& c = *(BatchFwdCtx*)p;
    const Net& net = *c.net;
    const int mw = net.max_width(), CH = 64;
    r1 = std::min(r1, c.n);
    std::vector<float> a((size_t)CH * mw), b((size_t)CH * mw);
    const Kernels& k = kernels();
    for (int s = r0; s < r1; s += CH) {
        const int n = std::min(CH, r1 - s);
        const float* cur = c.X + (size_t)s * c.ldx;
        int ldc = c.ldx;
        float* bufs[2] = {a.data(), b.data()};
        for (size_t li = 0; li < net.L.size(); li++) {
            const Layer& l = net.L[li];
            float* z = bufs[li & 1];
            k.gemm_rows(cur, n, ldc, l.in, l.W.data(), l.outp, l.b.data(), z, l.outp);
            for (int r = 0; r < n; r++) activate_n(l.act, z + (size_t)r * l.outp, l.out);
            cur = z;
            ldc = l.outp;
        }
        for (int r = 0; r < n; r++) std::memcpy(c.Y + (size_t)(s + r) * c.ldy, cur + (size_t)r * ldc, sizeof(float) * net.out());
    }
}
static void batch_forward(const Net& net, const float* X, int n, int ldx, float* Y, int ldy, int threads) {
    BatchFwdCtx c{&net, X, ldx, Y, ldy, n};
    parallel_for(threads, (n + 63) / 64, [](void* p, int i0, int i1) { batch_fwd_task(p, i0 * 64, i1 * 64); }, &c);
}

// nmse over standardised predictions / targets (rows x n)
static double nmse_std(const float* P, const float* Y, size_t T, int n) {
    double tot = 0;
    for (int j = 0; j < n; j++) {
        double m = 0, m2 = 0, se = 0;
        for (size_t t = 0; t < T; t++) {
            const double y = Y[t * n + j], e = (double)P[t * n + j] - y;
            m += y; m2 += y * y; se += e * e;
        }
        const double var = m2 / T - (m / T) * (m / T);
        tot += se / T / std::max(var, 1e-12);
    }
    return tot / n;
}

// ------------------------------------------------------------------------------------------------ training step
struct Grads { std::vector<std::vector<float>> dW, db; };
struct StepCtx {
    const Net* net;
    const float* Xb; int nf;
    const float* Yb; int no;
    int B;
    float *Z1, *H1, *Z2, *H2, *O, *D3, *D2, *D1;   // row-major, padded widths
    Grads* g;
    const float* Ab; int na;                     // structured models: the batch's affine values (B x na); na = 0: plain
};
static void step_rows(void* p, int c0, int c1) {   // forward + backward to the pre-activation grads, rows [4c0, 4c1)
    auto& c = *(StepCtx*)p;
    const Net& net = *c.net;
    const Layer &l1 = net.L[0], &l2 = net.L[1], &l3 = net.L[2];
    const int r0 = c0 * 4, r1 = std::min(c.B, c1 * 4), n = r1 - r0;
    if (n <= 0) return;
    const Kernels& k = kernels();
    const int w1 = l1.outp, w2 = l2.outp, w3 = l3.outp;
    float* Z1 = c.Z1 + (size_t)r0 * w1; float* H1 = c.H1 + (size_t)r0 * w1;
    float* Z2 = c.Z2 + (size_t)r0 * w2; float* H2 = c.H2 + (size_t)r0 * w2;
    float* O = c.O + (size_t)r0 * w3;   float* D3 = c.D3 + (size_t)r0 * w3;
    float* D2 = c.D2 + (size_t)r0 * w2; float* D1 = c.D1 + (size_t)r0 * w1;
    k.gemm_rows(c.Xb + (size_t)r0 * c.nf, n, c.nf, c.nf, l1.W.data(), w1, l1.b.data(), Z1, w1);
    for (size_t i = 0; i < (size_t)n * w1; i++) { const float r = Z1[i] > 0 ? Z1[i] : 0.0f; H1[i] = r * r; }
    k.gemm_rows(H1, n, w1, l1.out, l2.W.data(), w2, l2.b.data(), Z2, w2);
    for (size_t i = 0; i < (size_t)n * w2; i++) { const float r = Z2[i] > 0 ? Z2[i] : 0.0f; H2[i] = r * r; }
    k.gemm_rows(H2, n, w2, l2.out, l3.W.data(), w3, l3.b.data(), O, w3);
    const float sc = 2.0f / (float)((size_t)c.B * c.no);
    if (c.na == 0) {
        for (int r = 0; r < n; r++)
            for (int o = 0; o < w3; o++) D3[(size_t)r * w3 + o] = o < c.no ? (O[(size_t)r * w3 + o] - c.Yb[(size_t)(r0 + r) * c.no + o]) * sc : 0.0f;
    } else {                                      // prediction_j = O_j + sum_k a_k O_(no (k+1) + j)
        for (int r = 0; r < n; r++) {
            const float* o = O + (size_t)r * w3;
            const float* a = c.Ab + (size_t)(r0 + r) * c.na;
            const float* y = c.Yb + (size_t)(r0 + r) * c.no;
            float* d = D3 + (size_t)r * w3;
            for (int q = 0; q < w3; q++) d[q] = 0.0f;
            for (int j = 0; j < c.no; j++) {
                float pj = o[j];
                for (int k = 0; k < c.na; k++) pj += a[k] * o[(size_t)c.no * (k + 1) + j];
                const float e = (pj - y[j]) * sc;
                d[j] = e;
                for (int k = 0; k < c.na; k++) d[(size_t)c.no * (k + 1) + j] = e * a[k];
            }
        }
    }
    k.back_input(D3, n, w3, l3.out, l3.W.data(), w3, l3.in, D2, w2);
    for (size_t i = 0; i < (size_t)n * w2; i++) D2[i] *= Z2[i] > 0 ? 2.0f * Z2[i] : 0.0f;
    k.back_input(D2, n, w2, l2.out, l2.W.data(), w2, l2.in, D1, w1);
    for (size_t i = 0; i < (size_t)n * w1; i++) D1[i] *= Z1[i] > 0 ? 2.0f * Z1[i] : 0.0f;
}
static void step_wgrads(void* p, int i0, int i1) {   // index space: [0,w2) -> dW3 rows, [w2, w2+w1) -> dW2, then dW1
    auto& c = *(StepCtx*)p;
    const Net& net = *c.net;
    const Layer &l1 = net.L[0], &l2 = net.L[1], &l3 = net.L[2];
    const Kernels& k = kernels();
    const int n2 = l3.in, n1 = l2.in;
    auto run = [&](int lo, int hi, int base, const float* X, int ldx, const float* D, int ldd, int out, std::vector<float>& dW, int ldw) {
        const int a = std::max(lo, base), b = std::min(hi, base + (int)(dW.size() / ldw));
        if (b <= a) return;
        std::memset(&dW[(size_t)(a - base) * ldw], 0, sizeof(float) * (size_t)(b - a) * ldw);
        k.outer_acc(X, c.B, ldx, a - base, b - base, D, ldd, out, dW.data(), ldw);
    };
    run(i0, i1, 0, c.H2, l2.outp, c.D3, l3.outp, l3.out, c.g->dW[2], l3.outp);
    run(i0, i1, n2, c.H1, l1.outp, c.D2, l2.outp, l2.out, c.g->dW[1], l2.outp);
    run(i0, i1, n2 + n1, c.Xb, c.nf, c.D1, l1.outp, l1.out, c.g->dW[0], l1.outp);
}

// ------------------------------------------------------------------------------------------------ FULL
std::unique_ptr<Model> train_full(const Schema& sc, const std::vector<Recording>& trs, const std::vector<Recording>& vas,
                                  const sldrnc_train_options& opt, sldrnc_report& rep) {
    using clk = std::chrono::steady_clock;
    const auto t_start = clk::now();
    if (sc.ch.empty()) fail(SLDRNC_ERROR_ARGUMENT, "schema has no input channels");
    if (sc.n_out <= 0) fail(SLDRNC_ERROR_ARGUMENT, "schema has no outputs (call sldrnc_schema_set_outputs)");
    if (trs.empty() || vas.empty()) fail(SLDRNC_ERROR_ARGUMENT, "need at least one training and one validation recording");
    const int B = opt.batch > 0 ? opt.batch : 512, steps = opt.steps > 0 ? opt.steps : 12000;
    // the first WU ticks of every recording are warm-up: the slow input features have not settled yet (they start from
    // the first sample).  They are fed through the features but never trained on, scored or used for tuning.
    const size_t WU = (size_t)std::ceil(warmup_seconds(sc) * sc.rate);
    const int W = sc.width, no = sc.n_out;
    // recordings are laid end to end (row r of the k-th recording = global row off_k + r); `use` lists the global rows
    // after each recording's warm-up, `use_starts` the positions in `use` where a recording begins
    struct Rows { size_t T = 0; std::vector<size_t> off, use, use_starts; };
    auto layout = [&](const std::vector<Recording>& rs, const char* what) {
        Rows L;
        for (size_t k = 0; k < rs.size(); k++) {
            const Recording& r = rs[k];
            const std::string name = rs.size() == 1 ? std::string(what) : std::string(what) + "[" + std::to_string(k) + "]";
            if (!r.X || !r.Y || r.T == 0) fail(SLDRNC_ERROR_ARGUMENT, name + " is empty");
            check_finite(r.X, r.T * W, ("X_" + name).c_str());
            check_finite(r.Y, r.T * no, ("Y_" + name).c_str());
            L.off.push_back(L.T);
            if (r.T > WU) {
                L.use_starts.push_back(L.use.size());
                for (size_t t = WU; t < r.T; t++) L.use.push_back(L.T + t);
            }
            L.T += r.T;
        }
        return L;
    };
    const Rows Lt = layout(trs, "train"), Lv = layout(vas, "val");
    const size_t Nt = Lt.use.size(), Nv = Lv.use.size();
    if (Nt < (size_t)std::max(B, 1000))
        fail(SLDRNC_ERROR_DATA, "training data too short: need at least " + std::to_string(std::max(B, 1000)) + " ticks after the " + std::to_string(WU) +
                                    "-tick warm-up at the start of each recording (have " + std::to_string(Nt) + ")");
    if (Nv < 200)
        fail(SLDRNC_ERROR_DATA, "validation data too short: need at least 200 ticks after the " + std::to_string(WU) +
                                    "-tick warm-up at the start of each recording (have " + std::to_string(Nv) + ")");
    const int threads = opt.threads > 0 ? opt.threads : hw_threads();
    auto m = std::make_unique<Model>();
    m->mode = SLDRNC_MODE_FULL;
    m->rate = sc.rate;
    m->schema = sc;
    m->fs = make_featspec(sc);
    setup_structure(*m);
    const int nf = m->fs.nf, na = m->n_aff, nt = m->trunk_in(), H = no * (1 + na);   // H: the network's outputs
    // features (float), each recording from a fresh state, then standardise with TRAIN statistics
    auto featurise = [&](const std::vector<Recording>& rs, const Rows& L, std::vector<float>& F) {
        F.resize(L.T * nf);
        std::vector<double> f(nf);
        for (size_t k = 0; k < rs.size(); k++) {
            FeatState st;
            feat_init(sc, m->fs, st);
            for (size_t t = 0; t < rs[k].T; t++) {
                feat_step(sc, m->fs, st, rs[k].X + t * W, f.data());
                float* row = &F[(L.off[k] + t) * nf];
                for (int i = 0; i < nf; i++) row[i] = (float)f[i];
            }
        }
    };
    auto yrow = [&](const std::vector<Recording>& rs, const Rows& L, size_t g) {   // measured outputs of global row g
        const size_t k = (size_t)(std::upper_bound(L.off.begin(), L.off.end(), g) - L.off.begin()) - 1;
        return rs[k].Y + (g - L.off[k]) * no;
    };
    auto xrow = [&](const std::vector<Recording>& rs, const Rows& L, size_t g) {
        const size_t k = (size_t)(std::upper_bound(L.off.begin(), L.off.end(), g) - L.off.begin()) - 1;
        return rs[k].X + (g - L.off[k]) * W;
    };
    m->raw_lo.assign(W, 0.0f);                    // the training range of every raw input (tables are built over it)
    m->raw_hi.assign(W, 0.0f);
    {
        std::vector<double> lo(W, std::numeric_limits<double>::infinity()), hi(W, -std::numeric_limits<double>::infinity());
        for (size_t g : Lt.use) {
            const double* x = xrow(trs, Lt, g);
            for (int col = 0; col < W; col++) { lo[col] = std::min(lo[col], x[col]); hi[col] = std::max(hi[col], x[col]); }
        }
        for (int col = 0; col < W; col++) { m->raw_lo[col] = (float)lo[col]; m->raw_hi[col] = (float)hi[col]; }
    }
    // structured models: the network sees the trunk features; the affine values are kept apart
    auto split = [&](const float* frow, float* trunk, float* a) {
        for (int t = 0; t < nt; t++) trunk[t] = frow[m->trunk_idx[t]];
        for (int k = 0; k < na; k++) a[k] = frow[m->aff_idx[k]];
    };
    // standardised predictions (rows x no) of a network over rows of trunk features F (ld nt) and affine values A
    auto predict_std = [&](const Net& n, const float* F, const float* A, size_t rows, float* P) {
        if (na == 0) { batch_forward(n, F, (int)rows, nt, P, no, threads); return; }
        std::vector<float> Ho(rows * H);
        batch_forward(n, F, (int)rows, nt, Ho.data(), H, threads);
        for (size_t r = 0; r < rows; r++)
            for (int j = 0; j < no; j++) {
                float pj = Ho[r * H + j];
                for (int k = 0; k < na; k++) pj += A[r * na + k] * Ho[r * H + (size_t)no * (k + 1) + j];
                P[r * no + j] = pj;
            }
    };
    std::vector<float> Ft, Fv;
    featurise(trs, Lt, Ft);
    featurise(vas, Lv, Fv);
    m->in_mean.assign(nf, 0.0f);
    m->in_std.assign(nf, 1.0f);
    for (int i = 0; i < nf; i++) {
        double s = 0, s2 = 0;
        for (size_t g : Lt.use) { const double v = Ft[g * nf + i]; s += v; s2 += v * v; }
        const double cnt = (double)Nt, mu = s / cnt, var = std::max(0.0, s2 / cnt - mu * mu), sd = std::sqrt(var);
        m->in_mean[i] = (float)mu;
        m->in_std[i] = sd < 1e-6 ? 1.0f : (float)sd;
    }
    auto standardise = [&](std::vector<float>& F, size_t T) {
        for (size_t t = 0; t < T; t++) for (int i = 0; i < nf; i++) F[t * nf + i] = (F[t * nf + i] - m->in_mean[i]) / m->in_std[i];
    };
    standardise(Ft, Lt.T);
    standardise(Fv, Lv.T);
    m->out_mean.assign(no, 0.0f);
    m->out_std.assign(no, 1.0f);
    for (int j = 0; j < no; j++) {
        double s = 0, s2 = 0;
        for (size_t g : Lt.use) { const double y = yrow(trs, Lt, g)[j]; s += y; s2 += y * y; }
        const double cnt = (double)Nt, mu = s / cnt, sd = std::sqrt(std::max(0.0, s2 / cnt - mu * mu));
        if (sd < 1e-12) fail(SLDRNC_ERROR_DATA, "output " + std::to_string(j) + " is constant in the training data");
        m->out_mean[j] = (float)mu;
        m->out_std[j] = (float)sd;
    }
    std::vector<float> Ys(Lt.T * no), Yvs(Nv * no), Fvu(Nv * nf);   // validation: usable rows only, in order
    for (size_t k = 0; k < trs.size(); k++)
        for (size_t t = 0; t < trs[k].T; t++)
            for (int j = 0; j < no; j++) Ys[(Lt.off[k] + t) * no + j] = (float)((trs[k].Y[t * no + j] - m->out_mean[j]) / m->out_std[j]);
    for (size_t u = 0; u < Nv; u++) {
        const double* y = yrow(vas, Lv, Lv.use[u]);
        for (int j = 0; j < no; j++) Yvs[u * no + j] = (float)((y[j] - m->out_mean[j]) / m->out_std[j]);
        std::memcpy(&Fvu[u * nf], &Fv[Lv.use[u] * nf], sizeof(float) * nf);
    }
    std::vector<float> Fvt, Av;                   // structured models: validation trunk features and affine values
    if (na > 0) {
        Fvt.resize(Nv * nt);
        Av.resize(Nv * na);
        for (size_t u = 0; u < Nv; u++) split(&Fvu[u * nf], &Fvt[u * nt], &Av[u * na]);
    }
    const float* Fval = na > 0 ? Fvt.data() : Fvu.data();
    // network
    const int width = opt.size == SLDRNC_SIZE_SMALL ? 64 : opt.size == SLDRNC_SIZE_LARGE ? 256 : 128;
    const int dims[4] = {nt, width, width, H};
    Rng rng(opt.seed * 0x9E3779B97F4A7C15ull + 12345);
    m->net.L.resize(3);
    for (int l = 0; l < 3; l++) {
        std::vector<float> Wl((size_t)dims[l] * dims[l + 1]), bl(dims[l + 1]);
        const double bd = 1.0 / std::sqrt((double)dims[l]);
        for (auto& v : Wl) v = (float)((rng.uniform() * 2 - 1) * bd);
        for (auto& v : bl) v = (float)((rng.uniform() * 2 - 1) * bd);
        set_layer(m->net.L[l], dims[l], dims[l + 1], l < 2 ? SLDRNC_ACT_RELU2 : SLDRNC_ACT_IDENTITY, Wl.data(), bl.data());
    }
    Net& net = m->net;
    const int w1 = net.L[0].outp, w2 = net.L[1].outp, w3 = net.L[2].outp;
    std::vector<float> Xb((size_t)B * nt), Ab((size_t)B * std::max(na, 1)), Yb((size_t)B * no), Z1((size_t)B * w1), H1((size_t)B * w1), Z2((size_t)B * w2), H2((size_t)B * w2),
        O((size_t)B * w3), D3((size_t)B * w3), D2((size_t)B * w2), D1((size_t)B * w1);
    Grads g;
    std::vector<std::vector<float>> m1, m2, bm1, bm2;
    for (auto& l : net.L) {
        g.dW.emplace_back(l.W.size(), 0.0f); g.db.emplace_back(l.b.size(), 0.0f);
        m1.emplace_back(l.W.size(), 0.0f); m2.emplace_back(l.W.size(), 0.0f);
        bm1.emplace_back(l.b.size(), 0.0f); bm2.emplace_back(l.b.size(), 0.0f);
    }
    StepCtx ctx{&net, Xb.data(), nt, Yb.data(), no, B, Z1.data(), H1.data(), Z2.data(), H2.data(), O.data(), D3.data(), D2.data(), D1.data(), &g,
                Ab.data(), na};
    const double max_lr = opt.learning_rate > 0 ? opt.learning_rate : 5e-3;
    const double wd = opt.weight_decay >= 0 ? opt.weight_decay : 0.2;
    std::vector<float> Pv(Nv * no);
    double best = std::numeric_limits<double>::infinity();
    Net best_net = net;
    Rng brng(opt.seed + 777);
    const int wtasks = net.L[2].in + net.L[1].in + net.L[0].in;
    for (int s = 0; s < steps; s++) {
        for (int r = 0; r < B; r++) {
            const size_t i = Lt.use[(size_t)(brng.next() % Nt)];
            if (na == 0) std::memcpy(&Xb[(size_t)r * nf], &Ft[i * nf], sizeof(float) * nf);
            else split(&Ft[i * nf], &Xb[(size_t)r * nt], &Ab[(size_t)r * na]);
            std::memcpy(&Yb[(size_t)r * no], &Ys[i * no], sizeof(float) * no);
        }
        parallel_for(threads, (B + 3) / 4, step_rows, &ctx);
        parallel_for(threads, wtasks, step_wgrads, &ctx);
        const float* Dl[3] = {D1.data(), D2.data(), D3.data()};
        for (int l = 0; l < 3; l++) {
            const int wp = net.L[l].outp;
            std::fill(g.db[l].begin(), g.db[l].end(), 0.0f);
            for (int r = 0; r < B; r++) for (int o = 0; o < wp; o++) g.db[l][o] += Dl[l][(size_t)r * wp + o];
        }
        double gn2 = 0;
        for (int l = 0; l < 3; l++) {
            for (float v : g.dW[l]) gn2 += (double)v * v;
            for (float v : g.db[l]) gn2 += (double)v * v;
        }
        const double gn = std::sqrt(gn2);
        const float gscale = gn > 1.0 ? (float)(1.0 / gn) : 1.0f;
        double lr, b1;
        onecycle(s, steps, max_lr, lr, b1);
        const int t = s + 1;
        const double bc1 = 1.0 - std::pow(b1, t), bc2s = std::sqrt(1.0 - std::pow(0.999, t));
        const float fb1 = (float)b1, fstep = (float)(lr / bc1), fbc2 = (float)bc2s, decay = (float)(1.0 - lr * wd);
        for (int l = 0; l < 3; l++) {
            auto upd = [&](std::vector<float>& p, std::vector<float>& gr, std::vector<float>& a1, std::vector<float>& a2, bool is_w) {
                for (size_t i = 0; i < p.size(); i++) {
                    const float q = gr[i] * gscale;
                    a1[i] = a1[i] * fb1 + (1.0f - fb1) * q;
                    a2[i] = a2[i] * 0.999f + 0.001f * q * q;
                    if (is_w && wd > 0) p[i] *= decay;
                    p[i] -= fstep * (a1[i] / (std::sqrt(a2[i]) / fbc2 + 1e-8f));
                }
            };
            upd(net.L[l].W, g.dW[l], m1[l], m2[l], true);
            upd(net.L[l].b, g.db[l], bm1[l], bm2[l], false);
        }
        if (s % 100 == 99 || s == steps - 1) {
            predict_std(net, Fval, Av.data(), Nv, Pv.data());
            const double v = nmse_std(Pv.data(), Yvs.data(), Nv, no);
            if (std::isfinite(v) && v < best) { best = v; best_net = net; }
            if (opt.verbose && (s % 1000 == 999 || s == steps - 1))
                std::printf("sldrnc: step %d/%d  validation error %.4f (best %.4f)\n", s + 1, steps, v, best);
        }
    }
    if (!std::isfinite(best)) fail(SLDRNC_ERROR_DATA, "training diverged (no finite validation error); try a lower learning rate");
    m->net = best_net;
    // reports
    {
        std::vector<float> Fsub, Asub, Ysub;
        for (size_t u = 0; u < Nt; u += 10) {
            const size_t t = Lt.use[u];
            if (na == 0) {
                Fsub.insert(Fsub.end(), &Ft[t * nf], &Ft[t * nf] + nf);
            } else {
                Fsub.resize(Fsub.size() + nt);
                Asub.resize(Asub.size() + na);
                split(&Ft[t * nf], &Fsub[Fsub.size() - nt], &Asub[Asub.size() - na]);
            }
            Ysub.insert(Ysub.end(), &Ys[t * no], &Ys[t * no] + no);
        }
        const size_t ns = Ysub.size() / no;
        std::vector<float> Ps(ns * no);
        predict_std(m->net, Fsub.data(), Asub.data(), ns, Ps.data());
        rep.train_nmse = nmse_std(Ps.data(), Ysub.data(), ns, no);
    }
    predict_std(m->net, Fval, Av.data(), Nv, Pv.data());
    rep.val_nmse = nmse_std(Pv.data(), Yvs.data(), Nv, no);
    rep.val_nmse_corrected = std::numeric_limits<double>::quiet_NaN();
    rep.correction_time_ms = 0;
    // self-correction
    Correction& c = m->corr;
    c.mean.assign(m->out_mean.begin(), m->out_mean.end());
    c.sd.assign(m->out_std.begin(), m->out_std.end());
    c.vel_idx = sc.vel_idx.empty() ? std::vector<int>(no, -1) : sc.vel_idx;
    c.vel_sd.assign(no, 1.0);
    for (int j = 0; j < no; j++) {
        const int vi = c.vel_idx[j];
        if (vi < 0) continue;
        double s = 0, s2 = 0;
        for (size_t g : Lt.use) { const double v = xrow(trs, Lt, g)[vi]; s += v; s2 += v * v; }
        const double cnt = (double)Nt, sd = std::sqrt(std::max(0.0, s2 / cnt - (s / cnt) * (s / cnt)));
        if (sd < 1e-9) c.vel_idx[j] = -1; else c.vel_sd[j] = sd;
    }
    if (opt.fit_correction) {
        std::vector<double> o(Nv * no), y(Nv * no), Xu(Nv * W);
        for (size_t i = 0; i < Nv * no; i++) { o[i] = Pv[i]; y[i] = Yvs[i]; }
        for (size_t u = 0; u < Nv; u++) std::memcpy(&Xu[u * W], xrow(vas, Lv, Lv.use[u]), sizeof(double) * W);
        rep.val_nmse_corrected = corr_tune(c, o, y, Xu.data(), W, Nv, no, sc.rate, opt.correction_delay_ticks, &Lv.use_starts);
        rep.correction_time_ms = c.tau_ticks * 1000.0 / sc.rate;
        rep.correction_delay_ticks = c.delay_ticks;
        c.on = true;
    }
    rep.macs_per_tick = m->macs();
    rep.steps = steps;
    rep.seconds = std::chrono::duration<double>(clk::now() - t_start).count();
    return m;
}

// ------------------------------------------------------------------------------------------------ CORRECTION
// standardised predictions / targets for tuning, in the correction's units; returns the uncorrected nmse
static double standardise(const double* yp, const double* Y, size_t T, int no, const Correction& c, std::vector<double>& o, std::vector<double>& y) {
    o.assign(T * no, 0.0);
    y.assign(T * no, 0.0);
    double base = 0;
    for (int j = 0; j < no; j++) {
        double se = 0, ym = 0, ym2 = 0;
        for (size_t t = 0; t < T; t++) {
            o[t * no + j] = (yp[t * no + j] - c.mean[j]) / c.sd[j];
            y[t * no + j] = (Y[t * no + j] - c.mean[j]) / c.sd[j];
            const double e = y[t * no + j] - o[t * no + j];
            se += e * e;
            ym += y[t * no + j]; ym2 += y[t * no + j] * y[t * no + j];
        }
        base += se / T / std::max(1e-12, ym2 / T - (ym / T) * (ym / T));
    }
    return base / no;
}

// the model's own predictions over a recording, exactly as a session computes them, without correction
static std::vector<double> predict_uncorrected(const Model& m, const double* X, size_t T) {
    const int nin = m.input_size(), no = m.output_size();
    std::vector<double> yp(T * no);
    Session s;
    session_init(s, m, false, false);
    for (size_t t = 0; t < T; t++) session_step(s, X + t * nin, &yp[t * no]);
    return yp;
}

void add_correction(Model& m, const double* X, const double* Y, size_t T, double rate, const int* vel_idx, int delay, sldrnc_report& rep) {
    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    if (m.mode == SLDRNC_MODE_FULL) fail(SLDRNC_ERROR_STATE, "FULL models are trained with self-correction already");
    if (!(rate > 0)) fail(SLDRNC_ERROR_ARGUMENT, "rate_hz must be positive");
    if (T < 200) fail(SLDRNC_ERROR_DATA, "correction data too short: need at least 200 ticks");
    const int nin = m.input_size(), no = m.output_size();
    check_finite(X, T * nin, "X"); check_finite(Y, T * no, "Y");
    Model raw = m;                            // the model's own predictions, without any earlier correction
    raw.corr = Correction();
    const std::vector<double> yp = predict_uncorrected(raw, X, T);
    Correction c;
    c.mean.assign(no, 0.0);
    c.sd.assign(no, 1.0);
    for (int j = 0; j < no; j++) {
        double s = 0, s2 = 0;
        for (size_t t = 0; t < T; t++) { s += Y[t * no + j]; s2 += Y[t * no + j] * Y[t * no + j]; }
        const double mu = s / T, sd = std::sqrt(std::max(0.0, s2 / T - mu * mu));
        if (sd < 1e-12) fail(SLDRNC_ERROR_DATA, "output " + std::to_string(j) + " is constant in the correction data");
        c.mean[j] = mu; c.sd[j] = sd;
    }
    c.vel_idx.assign(no, -1);
    c.vel_sd.assign(no, 1.0);
    for (int j = 0; j < no; j++) {
        if (!vel_idx || vel_idx[j] < 0) continue;
        if (vel_idx[j] >= nin) fail(SLDRNC_ERROR_ARGUMENT, "velocity_index[" + std::to_string(j) + "] is outside the input row");
        double s = 0, s2 = 0;
        for (size_t t = 0; t < T; t++) { const double v = X[t * nin + vel_idx[j]]; s += v; s2 += v * v; }
        const double sd = std::sqrt(std::max(0.0, s2 / T - (s / T) * (s / T)));
        if (sd >= 1e-9) { c.vel_idx[j] = vel_idx[j]; c.vel_sd[j] = sd; }
    }
    std::vector<double> o, y;
    rep.val_nmse = standardise(yp.data(), Y, T, no, c, o, y);
    rep.train_nmse = std::numeric_limits<double>::quiet_NaN();
    rep.val_nmse_corrected = corr_tune(c, o, y, X, nin, T, no, rate, delay);
    rep.correction_time_ms = c.tau_ticks * 1000.0 / rate;
    rep.correction_delay_ticks = c.delay_ticks;
    c.on = true;
    m.corr = c;
    m.mode = SLDRNC_MODE_CORRECTION;
    m.rate = rate;
    rep.macs_per_tick = m.macs();
    rep.steps = 0;
    rep.seconds = std::chrono::duration<double>(clk::now() - t0).count();
}

// ------------------------------------------------------------------------------------------------ re-tune for a delay
std::unique_ptr<Model> tune_correction(const Model& src, const double* X, const double* Y, size_t T, int delay, sldrnc_report& rep) {
    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    if (!src.corr.on) fail(SLDRNC_ERROR_STATE, "the model has no self-correction to tune (for a RAW model use add_correction)");
    const int nin = src.input_size(), no = src.output_size();
    check_finite(X, T * nin, "X"); check_finite(Y, T * no, "Y");
    const size_t skip = src.mode == SLDRNC_MODE_FULL ? (size_t)std::ceil(warmup_seconds(src.schema) * src.rate) : 0;   // FULL: warm-up
    if (T < skip + 200) fail(SLDRNC_ERROR_DATA, "tuning data too short: need at least " + std::to_string(skip + 200) + " ticks");
    auto m = std::make_unique<Model>(src);
    const std::vector<double> yp = predict_uncorrected(*m, X, T);
    Correction c = src.corr;                  // keeps the output / velocity scales; only the time constant is re-tuned
    std::vector<double> o, y;
    rep.val_nmse = standardise(yp.data() + skip * no, Y + skip * no, T - skip, no, c, o, y);
    rep.train_nmse = std::numeric_limits<double>::quiet_NaN();
    rep.val_nmse_corrected = corr_tune(c, o, y, X + skip * nin, nin, T - skip, no, src.rate, delay);
    rep.correction_time_ms = c.tau_ticks * 1000.0 / src.rate;
    rep.correction_delay_ticks = c.delay_ticks;
    m->corr = c;
    rep.macs_per_tick = m->macs();
    rep.steps = 0;
    rep.seconds = std::chrono::duration<double>(clk::now() - t0).count();
    return m;
}

}  // namespace sldi
