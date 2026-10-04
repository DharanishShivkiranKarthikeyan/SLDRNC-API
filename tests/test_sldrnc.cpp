// test_sldrnc.cpp -- tests of the SLD-RNC library through its public C++ API (and the C API for error paths).
// Uses a simulated two-joint machine: torques from inertia, friction, gravity, a 20-ms actuator lag and a slow drift.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "sldrnc/sldrnc.hpp"

static int g_fail = 0, g_pass = 0;
#define CHECK(cond, ...)                                                       \
    do {                                                                       \
        if (cond) { g_pass++; } else {                                         \
            g_fail++; std::printf("FAIL %s:%d: ", __FILE__, __LINE__);         \
            std::printf(__VA_ARGS__); std::printf("\n");                       \
        }                                                                      \
    } while (0)

// ------------------------------------------------------------------------------------------------ simulated machine
struct Sim { std::vector<double> X, Y; size_t T; };   // X: q(2) qd(2) per tick; Y: torque(2)
static Sim simulate(size_t T, unsigned seed, double drift_amp, double t0 = 0) {
    Sim s; s.T = T; s.X.resize(T * 4); s.Y.resize(T * 2);
    double lag[2] = {0, 0};
    unsigned st = seed * 2654435761u + 1;
    auto noise = [&]() { st = st * 1664525u + 1013904223u; return ((st >> 8) / 16777216.0 - 0.5) * 0.02; };
    for (size_t t = 0; t < T; t++) {
        const double tt = t0 + t * 1e-3;
        for (int j = 0; j < 2; j++) {
            // motion: a sum of three fixed-frequency sinusoids, so velocity and acceleration are exact derivatives
            const double amp[3] = {0.5, 0.3, 0.15}, frq[3] = {1.3 + 0.4 * j, 2.9 + 0.4 * j, 4.7 + 0.4 * j};
            double q = 0, qd = 0, qdd = 0;
            for (int k = 0; k < 3; k++) {
                const double ph = frq[k] * tt + 1.7 * k + j;
                q += amp[k] * std::sin(ph); qd += amp[k] * frq[k] * std::cos(ph); qdd -= amp[k] * frq[k] * frq[k] * std::sin(ph);
            }
            const double tau = 1.5 * qdd + 0.4 * qd + 0.8 * std::tanh(qd / 0.05) + 3.0 * std::sin(q) + drift_amp * std::sin(0.2 * tt + j);
            lag[j] += (tau - lag[j]) / 20.0;                       // 20-ms actuator lag: needs history
            s.X[t * 4 + j] = q;
            s.X[t * 4 + 2 + j] = qd;
            s.Y[t * 2 + j] = lag[j] + noise();
        }
    }
    return s;
}

// ------------------------------------------------------------------------------------------------ tiny ONNX writer
struct PBW {
    std::vector<unsigned char> b;
    void varint(unsigned long long v) { while (v >= 0x80) { b.push_back((unsigned char)(v | 0x80)); v >>= 7; } b.push_back((unsigned char)v); }
    void key(int f, int wt) { varint((unsigned long long)f << 3 | wt); }
    void bytes(int f, const void* p, size_t n) { key(f, 2); varint(n); b.insert(b.end(), (const unsigned char*)p, (const unsigned char*)p + n); }
    void str(int f, const std::string& s) { bytes(f, s.data(), s.size()); }
    void msg(int f, const PBW& m) { bytes(f, m.b.data(), m.b.size()); }
    void vint(int f, long long v) { key(f, 0); varint((unsigned long long)v); }
};
static PBW tensor(const std::string& name, const std::vector<long long>& dims, const std::vector<float>& v) {
    PBW t;
    for (auto d : dims) t.vint(1, d);
    t.vint(2, 1);
    t.str(8, name);
    t.bytes(9, v.data(), v.size() * 4);
    return t;
}
static PBW node(const std::string& op, const std::vector<std::string>& in, const std::string& out) {
    PBW n;
    for (const auto& s : in) n.str(1, s);
    n.str(2, out);
    n.str(4, op);
    return n;
}
static PBW value_info(const std::string& name) { PBW v; v.str(1, name); return v; }

// ------------------------------------------------------------------------------------------------ tests
static double rms(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0;
    for (size_t i = 0; i < a.size(); i++) s += (a[i] - b[i]) * (a[i] - b[i]);
    return std::sqrt(s / a.size());
}

int main() {
    std::printf("SLD-RNC %s tests\n", sldrnc::version().c_str());
    // ---------------- FULL mode
    const Sim tr = simulate(60000, 1, 0.3), va = simulate(10000, 2, 0.3, 60.0), te = simulate(10000, 3, 0.3, 70.0);
    sldrnc::Schema schema(1000.0);
    schema.add("q", sldrnc::Input::Position, 2).add("qd", sldrnc::Input::Velocity, 2).outputs(2, "qd");
    CHECK(schema.width() == 4 && schema.column("qd") == 2, "schema layout");
    sldrnc::TrainOptions opt;
    opt.size = sldrnc::Size::Small;
    opt.steps = 3000;
    opt.seed = 7;
    sldrnc::Report rep;
    sldrnc::Model full = sldrnc::Model::train(schema, {tr.X, 4}, {tr.Y, 2}, {va.X, 4}, {va.Y, 2}, opt, &rep);
    std::printf("  FULL: train %.4f  val %.4f  val corrected %.4f  (tau %.0f ms, %.1f s, %lld MACs)\n", rep.train_error, rep.validation_error,
                rep.validation_error_corrected, rep.correction_time_ms, rep.seconds, rep.macs_per_tick);
    CHECK(rep.validation_error < 0.05, "FULL model should learn the simulated machine (val %.4f)", rep.validation_error);
    CHECK(rep.validation_error_corrected <= rep.validation_error, "self-correction should not hurt on validation");
    const sldrnc::Info fi = full.info();
    CHECK(fi.mode == sldrnc::Mode::Full && fi.input_size == 4 && fi.output_size == 2 && fi.has_correction, "FULL info");
    sldrnc::Data Xte(te.X, 4), Yte(te.Y, 2);
    const auto r_full = full.run(Xte, &Yte);
    const auto r_full_nc = full.run(Xte, &Yte, 1, 0, false);
    std::printf("  FULL test: %.4f corrected, %.4f without correction\n", r_full.error, r_full_nc.error);
    CHECK(r_full.error < r_full_nc.error, "correction should help on drifting test data");

    // several recordings: the training log split in two, each with its own warm-up
    {
        const size_t half = tr.T / 2;
        std::vector<double> X1(tr.X.begin(), tr.X.begin() + half * 4), X2(tr.X.begin() + half * 4, tr.X.end());
        std::vector<double> Y1(tr.Y.begin(), tr.Y.begin() + half * 2), Y2(tr.Y.begin() + half * 2, tr.Y.end());
        sldrnc::Report mrep;
        const sldrnc::Model multi = sldrnc::Model::train(schema, {{X1, 4}, {X2, 4}}, {{Y1, 2}, {Y2, 2}}, {{va.X, 4}}, {{va.Y, 2}}, opt, &mrep);
        std::printf("  FULL from 2 recordings: val %.4f, corrected %.4f\n", mrep.validation_error, mrep.validation_error_corrected);
        CHECK(mrep.validation_error < 0.05 && mrep.validation_error_corrected <= mrep.validation_error, "training from two recordings works");
        bool threw = false;
        try { sldrnc::Model::train(schema, {{X1, 4}, {X2, 4}}, {{Y1, 2}}, {{va.X, 4}}, {{va.Y, 2}}, opt); } catch (const sldrnc::Error& e) { threw = e.code() == SLDRNC_ERROR_ARGUMENT; }
        CHECK(threw, "one Y recording per X recording, else SLDRNC_ERROR_ARGUMENT");
        const std::vector<double> Xshort(100 * 4, 0.0), Yshort(100 * 2, 0.0);
        threw = false;
        try { sldrnc::Model::train(schema, {{Xshort, 4}}, {{Yshort, 2}}, {{va.X, 4}}, {{va.Y, 2}}, opt); } catch (const sldrnc::Error& e) { threw = e.code() == SLDRNC_ERROR_DATA; }
        CHECK(threw, "recordings shorter than the warm-up leave no training data -> SLDRNC_ERROR_DATA");
    }

    // ANGLE inputs: each joint angle plus its sine and cosine
    {
        sldrnc::Schema sa(1000.0);
        sa.add("q", sldrnc::Input::Angle, 2).add("qd", sldrnc::Input::Velocity, 2).outputs(2, "qd");
        sldrnc::Report arep;
        const sldrnc::Model am = sldrnc::Model::train(sa, {tr.X, 4}, {tr.Y, 2}, {va.X, 4}, {va.Y, 2}, opt, &arep);
        std::printf("  FULL with ANGLE: val %.4f (POSITION %.4f), %lld MACs (POSITION %lld)\n", arep.validation_error, rep.validation_error,
                    arep.macs_per_tick, rep.macs_per_tick);
        CHECK(arep.validation_error < 0.05, "FULL with ANGLE inputs learns the simulated machine (val %.4f)", arep.validation_error);
        CHECK(arep.macs_per_tick - rep.macs_per_tick == 2 * 2 * 64, "ANGLE adds a sine and a cosine per joint (2 joints x 2 features x 64 units)");
        am.save("sldrnc_test_angle.sldm");
        const sldrnc::Model back = sldrnc::Model::load("sldrnc_test_angle.sldm");
        std::remove("sldrnc_test_angle.sldm");
        CHECK(rms(back.run(Xte, &Yte).predictions, am.run(Xte, &Yte).predictions) == 0.0, "an ANGLE model saves and loads unchanged");
    }

    // fleets: many robots at once, the same numbers as separate sessions (dense FULL model with self-correction)
    {
        const int R = 7, T = 1500;                                  // 7: not a multiple of the 3-row or 8-robot blocks
        for (int threads : {1, 0}) {
            sldrnc::Fleet fl = full.fleet(R, true, true, 0, threads);
            std::vector<sldrnc::Session> ss;
            for (int r = 0; r < R; r++) ss.push_back(full.session());
            std::vector<double> X(R * 4), Y(R * 2), Ym(R * 2), y(2);
            bool same = true;
            for (int t = 0; t < T; t++) {
                for (int r = 0; r < R; r++) {
                    std::copy(&te.X[(size_t)(t + 300 * r) * 4], &te.X[(size_t)(t + 300 * r) * 4] + 4, &X[r * 4]);
                    std::copy(&te.Y[(size_t)(t + 300 * r) * 2], &te.Y[(size_t)(t + 300 * r) * 2] + 2, &Ym[r * 2]);
                }
                fl.step(X.data(), Y.data());
                for (int r = 0; r < R; r++) {
                    ss[r].step(&X[r * 4], y.data());
                    same = same && y[0] == Y[r * 2] && y[1] == Y[r * 2 + 1];
                    ss[r].observe(&Ym[r * 2]);
                }
                fl.observe(Ym.data());
            }
            CHECK(same, "a fleet equals separate sessions bit for bit (threads %d)", threads);
        }
    }

    // structured models: no history, outputs affine in one input, and the table form
    {
        auto make = [](size_t n, unsigned seed, std::vector<double>& X, std::vector<double>& Y) {
            X.resize(n * 3); Y.resize(n * 2);
            unsigned st = seed;
            auto u = [&]() { st = st * 1664525u + 1013904223u; return (st >> 8) / 16777216.0; };
            for (size_t i = 0; i < n; i++) {                       // unordered samples: x0 an angle, x1 affine, x2 linear
                const double a = (u() * 2 - 1) * 3.14159, v = u() * 2 - 1, g = u() * 2 - 1;
                X[i * 3] = a; X[i * 3 + 1] = v; X[i * 3 + 2] = g;
                Y[i * 2] = std::sin(a) * (1 + 0.5 * g) + v * std::cos(a);
                Y[i * 2 + 1] = 0.5 * std::cos(2 * a) * g - v * std::sin(a);
            }
        };
        std::vector<double> Xs, Ys, Xv, Yv, Xt, Yt;
        make(30000, 11, Xs, Ys); make(4000, 12, Xv, Yv); make(4000, 13, Xt, Yt);
        sldrnc::Schema sm(1000.0);
        sm.add("angle", sldrnc::Input::Angle, 1).add("v", sldrnc::Input::Generic, 1).add("g", sldrnc::Input::Generic, 1).outputs(2);
        sm.history(false).affine("v");
        sldrnc::TrainOptions so;
        so.size = sldrnc::Size::Small; so.steps = 3000; so.weight_decay = 0.0; so.fit_correction = false; so.seed = 2;
        sldrnc::Report sr;
        const sldrnc::Model net = sldrnc::Model::train(sm, {Xs, 3}, {Ys, 2}, {Xv, 3}, {Yv, 2}, so, &sr);
        const sldrnc::Info ni = net.info();
        CHECK(!ni.history && ni.affine_inputs == 1 && ni.warmup_seconds == 0.0 && ni.table_bytes == 0, "structured model info");
        sldrnc::Data Xtd(Xt, 3), Ytd(Yt, 2);
        const auto rn = net.run(Xtd, &Ytd, 1, 0, false);
        const sldrnc::Model tab = net.compile_table({256, 64});
        const auto rt = tab.run(Xtd, &Ytd, 1, 0, false);
        std::printf("  structured (no history, affine): val %.5f, test %.5f; table %lld bytes, test %.5f\n", sr.validation_error, rn.error,
                    tab.info().table_bytes, rt.error);
        CHECK(sr.validation_error < 0.01, "a no-history model with an affine input learns an affine-shaped function (val %.5f)", sr.validation_error);
        CHECK(tab.info().table_bytes > 0 && rt.error < rn.error * 1.5 + 1e-4, "the table stays close to its network (%.5f vs %.5f)", rt.error, rn.error);
        // table: fleet (8-robot vector lookups) == sessions (scalar lookups), bit for bit
        const int R = 21;
        sldrnc::Fleet tf = tab.fleet(R);
        std::vector<double> X(R * 3), Y(R * 2), y(2);
        bool same = true;
        for (int t = 0; t < 100; t++) {
            for (int r = 0; r < R; r++) std::copy(&Xt[(size_t)(t * R + r) * 3], &Xt[(size_t)(t * R + r) * 3] + 3, &X[r * 3]);
            tf.step(X.data(), Y.data());
            for (int r = 0; r < R; r++) {
                sldrnc::Session s1 = tab.session(false);
                s1.step(&X[r * 3], y.data());
                same = same && y[0] == Y[r * 2] && y[1] == Y[r * 2 + 1];
            }
        }
        CHECK(same, "a table fleet equals separate table sessions bit for bit");
        tab.save("sldrnc_test_table.sldm");
        const sldrnc::Model tb = sldrnc::Model::load("sldrnc_test_table.sldm");
        std::remove("sldrnc_test_table.sldm");
        CHECK(rms(tb.run(Xtd).predictions, tab.run(Xtd).predictions) == 0.0, "a table saves and loads unchanged");
        bool st = false;
        try { full.compile_table(); } catch (const sldrnc::Error& e) { st = e.code() == SLDRNC_ERROR_STATE; }
        CHECK(st, "compile_table on a model with history -> SLDRNC_ERROR_STATE");
        bool arg = false;
        try { sldrnc::Schema q(1000.0); q.add("o", sldrnc::Input::Quaternion, 4).add("x", sldrnc::Input::Generic, 1).affine("o"); }
        catch (const sldrnc::Error& e) { arg = e.code() == SLDRNC_ERROR_ARGUMENT; }
        CHECK(arg, "a QUATERNION channel cannot be affine");
        arg = false;
        try { sldrnc::Schema q(1000.0); q.add("x", sldrnc::Input::Generic, 1).affine("x"); } catch (const sldrnc::Error& e) { arg = e.code() == SLDRNC_ERROR_ARGUMENT; }
        CHECK(arg, "at least one channel must stay non-affine");
    }

    // session == run (delay 1)
    {
        sldrnc::Session s = full.session();
        std::vector<double> y(2), pred(te.T * 2);
        for (size_t t = 0; t < te.T; t++) { s.step(&te.X[t * 4], y.data()); pred[t * 2] = y[0]; pred[t * 2 + 1] = y[1]; s.observe(&te.Y[t * 2]); }
        CHECK(rms(pred, r_full.predictions) == 0.0, "session step/observe must equal model.run exactly (rms %.3g)", rms(pred, r_full.predictions));
        s.reset();
        s.step(&te.X[0], y.data());
        CHECK(y[0] == r_full.predictions[0], "reset restores the initial state");
    }
    // save / load round trip
    {
        full.save("sldrnc_test_model.sldm");
        sldrnc::Model again = sldrnc::Model::load("sldrnc_test_model.sldm");
        const auto r2 = again.run(Xte, &Yte);
        CHECK(rms(r2.predictions, r_full.predictions) == 0.0, "loaded model must predict identically");
        std::ifstream f("sldrnc_test_model.sldm", std::ios::binary);
        std::vector<char> bytes((std::istreambuf_iterator<char>(f)), {});
        f.close();
        bytes[bytes.size() / 2] ^= 0x5A;
        std::ofstream g("sldrnc_test_bad.sldm", std::ios::binary);
        g.write(bytes.data(), (std::streamsize)bytes.size());
        g.close();
        bool threw = false;
        try { sldrnc::Model::load("sldrnc_test_bad.sldm"); } catch (const sldrnc::Error& e) { threw = e.code() == SLDRNC_ERROR_FORMAT; }
        CHECK(threw, "a damaged model file must be rejected with SLDRNC_ERROR_FORMAT");
        std::remove("sldrnc_test_model.sldm");
        std::remove("sldrnc_test_bad.sldm");
    }

    // ---------------- RAW mode: a user's own network (here: a linear fit on [q, qd, sin q, tanh(qd/0.05)])
    auto user_features = [](const Sim& s) {
        std::vector<double> F(s.T * 8);
        for (size_t t = 0; t < s.T; t++) for (int j = 0; j < 2; j++) {
            const double q = s.X[t * 4 + j], qd = s.X[t * 4 + 2 + j];
            F[t * 8 + j] = q; F[t * 8 + 2 + j] = qd; F[t * 8 + 4 + j] = std::sin(q); F[t * 8 + 6 + j] = std::tanh(qd / 0.05);
        }
        return F;
    };
    const auto Ftr = user_features(tr), Fva = user_features(va), Fte = user_features(te);
    // least squares per output (normal equations, 9 x 9 with bias)
    std::vector<float> W(8 * 2), b(2);
    for (int j = 0; j < 2; j++) {
        double A[9][10] = {};
        for (size_t t = 0; t < tr.T; t++) {
            double z[9];
            for (int i = 0; i < 8; i++) z[i] = Ftr[t * 8 + i];
            z[8] = 1;
            for (int a = 0; a < 9; a++) { for (int c = 0; c < 9; c++) A[a][c] += z[a] * z[c]; A[a][9] += z[a] * tr.Y[t * 2 + j]; }
        }
        for (int c = 0; c < 9; c++) {
            int p = c;
            for (int r = c + 1; r < 9; r++) if (std::fabs(A[r][c]) > std::fabs(A[p][c])) p = r;
            for (int k = 0; k < 10; k++) std::swap(A[c][k], A[p][k]);
            for (int r = 0; r < 9; r++) if (r != c) { const double f = A[r][c] / A[c][c]; for (int k = 0; k < 10; k++) A[r][k] -= f * A[c][k]; }
        }
        for (int i = 0; i < 8; i++) W[i * 2 + j] = (float)(A[i][9] / A[i][i]);
        b[j] = (float)(A[8][9] / A[8][8]);
    }
    sldrnc::Model raw = sldrnc::Model::from_layers({{8, 2, sldrnc::Activation::Identity, W, b}});
    sldrnc::Data Fte_d(Fte, 8);
    const auto r_raw = raw.run(Fte_d, &Yte);
    sldrnc::Model corr = sldrnc::Model::from_layers({{8, 2, sldrnc::Activation::Identity, W, b}});
    const sldrnc::Report crep = corr.add_correction({Fva, 8}, {va.Y, 2}, 1000.0, {2, 3});
    const auto r_corr = corr.run(Fte_d, &Yte);
    std::printf("  RAW test %.4f | CORRECTION test %.4f (tau %.0f ms; validation %.4f -> %.4f)\n", r_raw.error, r_corr.error, crep.correction_time_ms,
                crep.validation_error, crep.validation_error_corrected);
    CHECK(corr.info().mode == sldrnc::Mode::Correction, "add_correction turns a RAW model into CORRECTION mode");
    CHECK(r_corr.error < 0.5 * r_raw.error, "correction should at least halve the error of a lagging user model (%.4f vs %.4f)", r_corr.error, r_raw.error);
    bool threw = false;
    try { full.add_correction(Xte, Yte, 1000.0); } catch (const sldrnc::Error& e) { threw = e.code() == SLDRNC_ERROR_STATE; }
    CHECK(threw, "add_correction on a FULL model must fail with SLDRNC_ERROR_STATE");
    {   // re-tuning a CORRECTION model for delay 1 on its own tuning data reproduces it; delays reach add_correction
        const sldrnc::Model again = corr.tune_correction({Fva, 8}, {va.Y, 2}, 1);
        CHECK(rms(again.run(Fte_d, &Yte).predictions, r_corr.predictions) == 0.0, "tune_correction(delay 1) on the same data reproduces add_correction");
        sldrnc::Model corr20 = sldrnc::Model::from_layers({{8, 2, sldrnc::Activation::Identity, W, b}});
        corr20.add_correction({Fva, 8}, {va.Y, 2}, 1000.0, {2, 3}, 20);
        CHECK(corr20.info().correction_delay_ticks == 20, "add_correction(..., delay_ticks) records the delay");
        bool st = false;
        try { raw.tune_correction(Fte_d, Yte, 5); } catch (const sldrnc::Error& e) { st = e.code() == SLDRNC_ERROR_STATE; }
        CHECK(st, "tune_correction on a RAW model (no correction) -> SLDRNC_ERROR_STATE");
    }

    // ---------------- sensor fault: protection keeps the corrected model sane
    {
        std::vector<double> Ybad = te.Y;
        for (size_t t = 4000; t < 6000; t++) Ybad[t * 2] = 0.0;      // output 0's sensor stuck at zero for 2 s
        auto run_against_truth = [&](bool protect) {
            sldrnc::Session s = full.session(true, protect);
            std::vector<double> y(2);
            double se = 0;
            for (size_t t = 0; t < te.T; t++) {
                s.step(&te.X[t * 4], y.data());
                if (t >= 4000 && t < 9000) se += (y[0] - te.Y[t * 2]) * (y[0] - te.Y[t * 2]);
                s.observe(&Ybad[t * 2]);
            }
            return std::sqrt(se / 5000);
        };
        const double unprot = run_against_truth(false), prot = run_against_truth(true);
        std::printf("  sensor fault: RMS error over fault + 3 s: protected %.3f, unprotected %.3f\n", prot, unprot);
        CHECK(prot < unprot, "protection must reduce the damage of a stuck sensor");
    }

    // ---------------- measurement delay: tune for it; late measurements through a session
    {
        const int D = 50;
        sldrnc::Report drep;
        const sldrnc::Model full50 = full.tune_correction({va.X, 4}, {va.Y, 2}, D, &drep);
        const auto r1_at50 = full.run(Xte, &Yte, D), r50_at50 = full50.run(Xte, &Yte, D);
        std::printf("  delay %d: tuned for 1 -> %.4f, tuned for %d -> %.4f (tau %.0f -> %.0f ms)\n", D, r1_at50.error, D, r50_at50.error,
                    full.info().correction_time_ms, drep.correction_time_ms);
        CHECK(full50.info().correction_delay_ticks == D && drep.correction_delay_ticks == D && full.info().correction_delay_ticks == 1,
              "tune_correction returns a copy tuned for the new delay and leaves the original alone");
        CHECK(r50_at50.error <= 1.10 * r1_at50.error, "tuning for a delay should not be clearly worse at that delay (%.4f vs %.4f)", r50_at50.error,
              r1_at50.error);
        sldrnc::Session s = full50.session();
        std::vector<double> y(2), pred(te.T * 2);
        for (size_t t = 0; t < te.T; t++) {
            s.step(&te.X[t * 4], y.data());
            pred[t * 2] = y[0]; pred[t * 2 + 1] = y[1];
            if (t + 1 >= (size_t)D) s.observe(&te.Y[(t + 1 - D) * 2], D);    // the measurement of tick t + 1 - D arrives now
        }
        CHECK(rms(pred, r50_at50.predictions) == 0.0, "session observe(y, delay) must equal model.run(delay) exactly (rms %.3g)",
              rms(pred, r50_at50.predictions));
        full50.save("sldrnc_test_d50.sldm");
        const sldrnc::Model back = sldrnc::Model::load("sldrnc_test_d50.sldm");
        std::remove("sldrnc_test_d50.sldm");
        CHECK(back.info().correction_delay_ticks == D && rms(back.run(Xte, &Yte, D).predictions, r50_at50.predictions) == 0.0,
              "save / load keeps the tuned delay");
        sldrnc::Session short_mem = full.session(true, true, 10);
        short_mem.step(&te.X[0], y.data());
        bool refused = false;
        try { short_mem.observe(&te.Y[0], 11); } catch (const sldrnc::Error& e) { refused = e.code() == SLDRNC_ERROR_ARGUMENT; }
        CHECK(refused, "a delay beyond the session's max_delay_ticks -> SLDRNC_ERROR_ARGUMENT");
    }

    // ---------------- activations and from_layers arithmetic
    {
        const sldrnc::Activation acts[] = {sldrnc::Activation::Identity, sldrnc::Activation::Relu, sldrnc::Activation::Relu2, sldrnc::Activation::LeakyRelu,
                                           sldrnc::Activation::Tanh, sldrnc::Activation::Sigmoid, sldrnc::Activation::Gelu, sldrnc::Activation::Silu};
        for (auto a : acts) {
            std::vector<float> W1 = {0.5f, -1.0f, 2.0f, 0.25f, -0.75f, 1.5f}, b1 = {0.1f, -0.2f, 0.3f};   // 2 x 3
            std::vector<float> W2 = {1.0f, -2.0f, 0.5f}, b2 = {0.05f};                                     // 3 x 1
            sldrnc::Model m = sldrnc::Model::from_layers({{2, 3, a, W1, b1}, {3, 1, sldrnc::Activation::Identity, W2, b2}}, {1.0f, -1.0f}, {2.0f, 4.0f}, {10.0f}, {3.0f});
            const double x[2] = {2.0, 3.0};
            double y = 0;
            sldrnc::Session s = m.session();
            s.step(x, &y);
            const double xn[2] = {(2.0 - 1.0) / 2.0, (3.0 + 1.0) / 4.0};
            double out = b2[0];
            for (int o = 0; o < 3; o++) {
                double z = b1[o] + xn[0] * W1[o] + xn[1] * W1[3 + o], h;
                switch (a) {
                    case sldrnc::Activation::Relu: h = z > 0 ? z : 0; break;
                    case sldrnc::Activation::Relu2: h = z > 0 ? z * z : 0; break;
                    case sldrnc::Activation::LeakyRelu: h = z > 0 ? z : 0.01 * z; break;
                    case sldrnc::Activation::Tanh: h = std::tanh(z); break;
                    case sldrnc::Activation::Sigmoid: h = 1 / (1 + std::exp(-z)); break;
                    case sldrnc::Activation::Gelu: h = 0.5 * z * (1 + std::tanh(0.7978845608 * (z + 0.044715 * z * z * z))); break;
                    case sldrnc::Activation::Silu: h = z / (1 + std::exp(-z)); break;
                    default: h = z;
                }
                out += h * W2[o];
            }
            out = out * 3.0 + 10.0;
            CHECK(std::fabs(y - out) < 1e-5 * (1 + std::fabs(out)), "from_layers activation %d: %.7f vs %.7f", (int)a, y, out);
        }
    }

    // ---------------- ONNX import: (x - mean) / std -> Gemm(transB) -> Relu -> x*x -> MatMul -> Add -> Mul(scale)
    {
        const std::vector<float> mean = {0.5f, -1.0f}, inv_std = {2.0f, 0.25f};
        const std::vector<float> Wt = {0.3f, -0.6f, 1.2f, 0.4f, -0.2f, 0.9f};   // Gemm B with transB=1: shape (3, 2)
        const std::vector<float> b1 = {0.1f, 0.2f, -0.3f}, W2 = {0.7f, -1.1f, 0.5f}, b2 = {0.25f}, scale = {4.0f};
        PBW g;
        g.msg(1, node("Sub", {"x", "mean"}, "a"));
        g.msg(1, node("Mul", {"a", "inv_std"}, "bq"));
        PBW gem = node("Gemm", {"bq", "Wt", "b1"}, "h");
        PBW at; at.str(1, "transB"); at.vint(3, 1); at.vint(20, 2);
        gem.msg(5, at);
        g.msg(1, gem);
        g.msg(1, node("Relu", {"h"}, "r"));
        g.msg(1, node("Mul", {"r", "r"}, "r2"));
        g.msg(1, node("MatMul", {"r2", "W2"}, "m"));
        g.msg(1, node("Add", {"m", "b2"}, "o"));
        g.msg(1, node("Mul", {"o", "scale"}, "y"));
        g.msg(5, tensor("mean", {2}, mean)); g.msg(5, tensor("inv_std", {2}, inv_std));
        g.msg(5, tensor("Wt", {3, 2}, Wt)); g.msg(5, tensor("b1", {3}, b1));
        g.msg(5, tensor("W2", {3, 1}, W2)); g.msg(5, tensor("b2", {1}, b2)); g.msg(5, tensor("scale", {1}, scale));
        g.msg(11, value_info("x"));
        g.msg(12, value_info("y"));
        PBW model;
        model.vint(1, 8);
        model.msg(7, g);
        sldrnc::Model m = sldrnc::Model::from_onnx_bytes(model.b);
        const double x[2] = {1.5, 2.0};
        double y = 0;
        sldrnc::Session s = m.session();
        s.step(x, &y);
        const double xn[2] = {(1.5 - 0.5) * 2.0, (2.0 + 1.0) * 0.25};
        double out = b2[0];
        for (int o = 0; o < 3; o++) {
            const double z = b1[o] + xn[0] * Wt[o * 2] + xn[1] * Wt[o * 2 + 1];
            const double r = z > 0 ? z : 0;
            out += r * r * W2[o];
        }
        out *= 4.0;
        CHECK(std::fabs(y - out) < 1e-5 * (1 + std::fabs(out)), "ONNX import: %.7f vs %.7f", y, out);
        CHECK(m.info().mode == sldrnc::Mode::Raw && m.info().input_size == 2, "ONNX model info");
        // unsupported op is reported clearly
        PBW g2;
        g2.msg(1, node("Conv", {"x", "W"}, "y"));
        g2.msg(5, tensor("W", {1, 1}, {1.0f}));
        g2.msg(11, value_info("x"));
        g2.msg(12, value_info("y"));
        PBW m2; m2.msg(7, g2);
        sldrnc_model* bad = nullptr;
        const sldrnc_status st = sldrnc_model_import_onnx_memory(m2.b.data(), m2.b.size(), &bad);
        CHECK(st == SLDRNC_ERROR_UNSUPPORTED && std::strstr(sldrnc_last_error(), "Conv"), "unsupported ONNX op -> SLDRNC_ERROR_UNSUPPORTED naming the op");
    }

    // ---------------- C API argument checks
    {
        sldrnc_model* m = nullptr;
        CHECK(sldrnc_model_load(nullptr, &m) == SLDRNC_ERROR_ARGUMENT && std::strlen(sldrnc_last_error()) > 0, "NULL path -> ARGUMENT with a message");
        CHECK(sldrnc_model_load("definitely_missing_file.sldm", &m) == SLDRNC_ERROR_IO, "missing file -> IO");
        sldrnc_schema* sc = nullptr;
        CHECK(sldrnc_schema_create(-1.0, &sc) == SLDRNC_ERROR_ARGUMENT, "negative rate -> ARGUMENT");
        CHECK(sldrnc_schema_create(1000.0, &sc) == SLDRNC_OK, "schema create");
        CHECK(sldrnc_schema_add_input(sc, "quat", SLDRNC_INPUT_QUATERNION, 3) == SLDRNC_ERROR_ARGUMENT, "quaternion must have 4 values");
        sldrnc_schema_free(sc);
        double nanrow[4] = {NAN, 0, 0, 0};
        std::vector<double> Xn = tr.X;
        Xn[17] = NAN;
        bool threw_data = false;
        try { sldrnc::Model::train(schema, {Xn, 4}, {tr.Y, 2}, {va.X, 4}, {va.Y, 2}, opt); } catch (const sldrnc::Error& e) { threw_data = e.code() == SLDRNC_ERROR_DATA; }
        CHECK(threw_data, "NaN in training data -> SLDRNC_ERROR_DATA");
        (void)nanrow;
    }

    std::printf("%s: %d passed, %d failed\n", g_fail ? "FAIL" : "PASS", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
