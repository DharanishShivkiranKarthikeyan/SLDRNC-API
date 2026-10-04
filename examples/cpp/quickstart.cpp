// quickstart.cpp -- SLD-RNC in five minutes (C++).
//
// A simulated two-joint machine stands in for your robot: the example logs joint positions / velocities and the
// measured joint torques, then uses the three modes and compares them on data recorded later:
//   1. FULL        train an SLD-RNC model from the logs
//   2. CORRECTION  keep "your" model and add SLD-RNC self-correction
//   3. RAW         run "your" model as-is
// and finally run the best model the way a 1 kHz control loop would.
//
// Build: cmake -S . -B build && cmake --build build --config Release   ->   build/sldrnc_example
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include <sldrnc/sldrnc.hpp>

// ---- a stand-in robot: two joints, inertia + friction + gravity + a 20-ms actuator lag + slow drift --------------
struct Log { std::vector<double> X, Y; size_t ticks; };     // X rows: q0 q1 qd0 qd1;  Y rows: torque0 torque1
Log record(size_t ticks, double start_s, unsigned seed) {
    Log L{std::vector<double>(ticks * 4), std::vector<double>(ticks * 2), ticks};
    double lag[2] = {0, 0};
    unsigned st = seed;
    for (size_t t = 0; t < ticks; t++) {
        const double s = start_s + t * 1e-3;
        for (int j = 0; j < 2; j++) {
            const double amp[3] = {0.5, 0.3, 0.15}, frq[3] = {1.3 + 0.4 * j, 2.9 + 0.4 * j, 4.7 + 0.4 * j};
            double q = 0, qd = 0, qdd = 0;                            // sum of sinusoids: exact derivatives
            for (int k = 0; k < 3; k++) {
                const double ph = frq[k] * s + 1.7 * k + j;
                q += amp[k] * std::sin(ph); qd += amp[k] * frq[k] * std::cos(ph); qdd -= amp[k] * frq[k] * frq[k] * std::sin(ph);
            }
            const double torque = 1.5 * qdd + 0.4 * qd + 0.8 * std::tanh(qd / 0.05) + 3.0 * std::sin(q) + 0.3 * std::sin(0.2 * s + j);
            lag[j] += (torque - lag[j]) / 20.0;
            st = st * 1664525u + 1013904223u;
            L.X[t * 4 + j] = q;
            L.X[t * 4 + 2 + j] = qd;
            L.Y[t * 2 + j] = lag[j] + ((st >> 8) / 16777216.0 - 0.5) * 0.02;
        }
    }
    return L;
}

// ---- "your own model": a linear least-squares fit on hand-made features (stands in for any network you already have)
std::vector<double> my_features(const Log& L) {
    std::vector<double> F(L.ticks * 6);
    for (size_t t = 0; t < L.ticks; t++)
        for (int j = 0; j < 2; j++) {
            F[t * 6 + j] = L.X[t * 4 + 2 + j];                          // velocity
            F[t * 6 + 2 + j] = std::sin(L.X[t * 4 + j]);                // gravity-like term
            F[t * 6 + 4 + j] = std::tanh(L.X[t * 4 + 2 + j] / 0.05);    // friction-like term
        }
    return F;
}
sldrnc::DenseLayer fit_my_model(const std::vector<double>& F, const std::vector<double>& Y, size_t ticks) {
    sldrnc::DenseLayer layer{6, 2, sldrnc::Activation::Identity, std::vector<float>(12), std::vector<float>(2)};
    for (int j = 0; j < 2; j++) {
        double A[7][8] = {};
        for (size_t t = 0; t < ticks; t++) {
            double z[7] = {F[t * 6], F[t * 6 + 1], F[t * 6 + 2], F[t * 6 + 3], F[t * 6 + 4], F[t * 6 + 5], 1.0};
            for (int a = 0; a < 7; a++) { for (int c = 0; c < 7; c++) A[a][c] += z[a] * z[c]; A[a][7] += z[a] * Y[t * 2 + j]; }
        }
        for (int c = 0; c < 7; c++)
            for (int r = 0; r < 7; r++)
                if (r != c) { const double f = A[r][c] / A[c][c]; for (int k = 0; k < 8; k++) A[r][k] -= f * A[c][k]; }
        for (int i = 0; i < 6; i++) layer.W[i * 2 + j] = (float)(A[i][7] / A[i][i]);
        layer.b[j] = (float)(A[6][7] / A[6][6]);
    }
    return layer;
}

int main() {
    std::printf("SLD-RNC %s\n\n", sldrnc::version().c_str());
    const Log train = record(60000, 0.0, 1), val = record(10000, 60.0, 2), test = record(20000, 70.0, 3);   // 60 s / 10 s / 20 s

    // ---- 1. FULL: describe the signals of one tick, then train
    sldrnc::Schema schema(1000.0);                                  // the log and the control loop run at 1 kHz
    schema.add("q", sldrnc::Input::Position, 2)
          .add("qd", sldrnc::Input::Velocity, 2)
          .outputs(2, "qd");                                        // output j is the torque of the joint with velocity qd[j]
    sldrnc::TrainOptions options;
    options.size = sldrnc::Size::Small;
    options.steps = 4000;                                           // the default (12000) is better; this keeps the demo quick
    sldrnc::Report report;
    sldrnc::Model full = sldrnc::Model::train(schema, {train.X, 4}, {train.Y, 2}, {val.X, 4}, {val.Y, 2}, options, &report);
    std::printf("trained FULL model in %.1f s: validation error %.4f, %.4f with self-correction\n", report.seconds, report.validation_error,
                report.validation_error_corrected);

    // ---- 3. RAW: bring your own model (from_layers here; Model::from_onnx("file.onnx") works the same way)
    const std::vector<double> F_train = my_features(train), F_val = my_features(val), F_test = my_features(test);
    const sldrnc::DenseLayer mine = fit_my_model(F_train, train.Y, train.ticks);
    sldrnc::Model raw = sldrnc::Model::from_layers({mine});

    // ---- 2. CORRECTION: the same model plus self-correction, tuned on validation data
    sldrnc::Model corrected = sldrnc::Model::from_layers({mine});
    corrected.add_correction({F_val, 6}, {val.Y, 2}, 1000.0, {0, 1});   // columns 0, 1 of my features are the joint velocities

    // ---- compare on the later test recording (error: 0 = perfect, 1 = no better than the average), after a 2-s warm-up
    sldrnc::Data Yt(test.Y, 2);
    std::printf("\n%-34s %s\n", "mode", "test error");
    std::printf("%-34s %.4f\n", "RAW (your model as-is)", raw.run({F_test, 6}, &Yt, 1, 2000).error);
    std::printf("%-34s %.4f\n", "CORRECTION (your model + SLD-RNC)", corrected.run({F_test, 6}, &Yt, 1, 2000).error);
    std::printf("%-34s %.4f\n", "FULL, without self-correction", full.run({test.X, 4}, &Yt, 1, 2000, false).error);
    std::printf("%-34s %.4f\n", "FULL", full.run({test.X, 4}, &Yt, 1, 2000).error);

    // ---- run time: one session per robot; step() every tick, observe() when the measurement arrives
    full.save("robot.sldm");
    sldrnc::Model model = sldrnc::Model::load("robot.sldm");
    sldrnc::Session robot = model.session();
    std::vector<double> torque(2);
    const auto t0 = std::chrono::steady_clock::now();
    for (size_t t = 0; t < test.ticks; t++) {
        robot.step(&test.X[t * 4], torque.data());                  // -> feed-forward torque for this tick
        robot.observe(&test.Y[t * 2]);                              // measured torque for the same tick
    }
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / test.ticks;
    std::printf("\nreal-time loop: %.2f us per tick (step + observe), %zu ticks\n", us, test.ticks);

    // ---- many robots: a fleet steps them all in one call (same outputs as separate sessions, one call per tick)
    const size_t N = 1000, T = 500;
    sldrnc::Fleet fleet = model.fleet(N);
    std::vector<double> X(N * 4), Y(N * 2), U(N * 2);
    const auto f0 = std::chrono::steady_clock::now();
    for (size_t t = 0; t < T; t++) {
        for (size_t r = 0; r < N; r++) {                            // robot r replays the test recording from its own place
            const size_t k = (r * 17 % (test.ticks - T)) + t;
            for (int c = 0; c < 4; c++) X[r * 4 + c] = test.X[k * 4 + c];
            for (int c = 0; c < 2; c++) Y[r * 2 + c] = test.Y[k * 2 + c];
        }
        fleet.step(X.data(), U.data());
        fleet.observe(Y.data());
    }
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - f0).count();
    std::printf("fleet of %zu robots: %.3g robot-ticks per second (step + observe, including filling the inputs)\n", N, N * T / sec);
    std::remove("robot.sldm");
    return 0;
}
