// sldrnc.hpp -- SLD-RNC (Self-correcting Learned Dynamics for Real-time Neural Control) C++ API (header-only, C++17).  A thin, safe layer over sldrnc.h: RAII handles, exceptions,
// enums and std::vector.  Link against the sldrnc shared library.
//
//   #include <sldrnc/sldrnc.hpp>
//
//   // 1. FULL: train an SLD-RNC model from your robot's logs
//   sldrnc::Schema schema(1000.0);                                   // 1 kHz
//   schema.add("q", sldrnc::Input::Position, 12)
//         .add("qd", sldrnc::Input::Velocity, 12)
//         .add("imu_gyro", sldrnc::Input::Gyro, 3)
//         .outputs(12, "qd");                                        // output j belongs to the joint of qd[j]
//   sldrnc::Report rep;
//   sldrnc::Model model = sldrnc::Model::train(schema, X_train, Y_train, X_val, Y_val, {}, &rep);
//
//   // 3. RAW: your own network      2. CORRECTION: your network + self-correction
//   sldrnc::Model mine = sldrnc::Model::from_onnx("my_net.onnx");
//   mine.add_correction(X_val, Y_val, 1000.0, velocity_columns);
//
//   // run time (any mode)
//   sldrnc::Session robot = model.session();
//   robot.step(x, y);            // predict this tick
//   robot.observe(y_measured);   // feed the measurement (drives self-correction)
//
//   // a copy tuned for measurements 50 ticks late (e.g. a planner predicting 50 ms ahead at 1 kHz)
//   sldrnc::Model planner = model.tune_correction(X_val, Y_val, 50);
#pragma once
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "sldrnc.h"

namespace sldrnc {

// ------------------------------------------------------------------------------------------------ basics
class Error : public std::runtime_error {
   public:
    Error(sldrnc_status code, const std::string& msg) : std::runtime_error(msg), code_(code) {}
    sldrnc_status code() const { return code_; }

   private:
    sldrnc_status code_;
};
inline void check(sldrnc_status s) {
    if (s != SLDRNC_OK) throw Error(s, sldrnc_last_error());
}
inline std::string version() { return sldrnc_version(); }

enum class Mode { Full = SLDRNC_MODE_FULL, Correction = SLDRNC_MODE_CORRECTION, Raw = SLDRNC_MODE_RAW };
enum class Input {
    Position = SLDRNC_INPUT_POSITION, Velocity = SLDRNC_INPUT_VELOCITY, Acceleration = SLDRNC_INPUT_ACCELERATION,
    Gyro = SLDRNC_INPUT_GYRO, Accel = SLDRNC_INPUT_ACCEL, Quaternion = SLDRNC_INPUT_QUATERNION, Generic = SLDRNC_INPUT_GENERIC,
    Angle = SLDRNC_INPUT_ANGLE        // rotary joint angles (rad); the model also sees their sine and cosine
};
enum class Activation {
    Identity = SLDRNC_ACT_IDENTITY, Relu = SLDRNC_ACT_RELU, Relu2 = SLDRNC_ACT_RELU2, LeakyRelu = SLDRNC_ACT_LEAKY_RELU,
    Tanh = SLDRNC_ACT_TANH, Sigmoid = SLDRNC_ACT_SIGMOID, Gelu = SLDRNC_ACT_GELU, Silu = SLDRNC_ACT_SILU
};
enum class Size { Small = SLDRNC_SIZE_SMALL, Medium = SLDRNC_SIZE_MEDIUM, Large = SLDRNC_SIZE_LARGE };

// A non-owning view of a recording: `ticks` rows of `columns` doubles, row-major.
struct Data {
    const double* data = nullptr;
    std::size_t ticks = 0;
    std::size_t columns = 0;
    Data() = default;
    Data(const double* d, std::size_t t, std::size_t c) : data(d), ticks(t), columns(c) {}
    Data(const std::vector<double>& v, std::size_t c) : data(v.data()), ticks(c ? v.size() / c : 0), columns(c) {
        if (c == 0 || v.size() % c) throw Error(SLDRNC_ERROR_ARGUMENT, "vector size is not a multiple of the column count");
    }
};

// ------------------------------------------------------------------------------------------------ schema
// Describes one tick of raw signals for FULL training: channels in the order of the input row.
class Schema {
   public:
    explicit Schema(double rate_hz) {
        sldrnc_schema* s = nullptr;
        check(sldrnc_schema_create(rate_hz, &s));
        h_.reset(s, sldrnc_schema_free);
    }
    Schema& add(const std::string& name, Input kind, int size) {
        check(sldrnc_schema_add_input(h_.get(), name.c_str(), (sldrnc_input_kind)kind, size));
        return *this;
    }
    // n outputs; optionally the velocity column of each output's joint (-1 = none)
    Schema& outputs(int n, const std::vector<int>& velocity_index = {}) {
        if (!velocity_index.empty() && (int)velocity_index.size() != n) throw Error(SLDRNC_ERROR_ARGUMENT, "velocity_index needs one entry per output");
        check(sldrnc_schema_set_outputs(h_.get(), n, velocity_index.empty() ? nullptr : velocity_index.data()));
        return *this;
    }
    // n outputs, output j belonging to the joint whose velocity is element j of channel `velocity_channel`
    Schema& outputs(int n, const std::string& velocity_channel) {
        const int c = column(velocity_channel);
        if (c < 0) throw Error(SLDRNC_ERROR_ARGUMENT, "no channel named '" + velocity_channel + "'");
        std::vector<int> v(n);
        for (int j = 0; j < n; j++) v[j] = c + j;
        return outputs(n, v);
    }
    // false: the outputs depend only on the current inputs (current values only, no warm-up, rows may be unordered)
    Schema& history(bool on) {
        check(sldrnc_schema_set_history(h_.get(), on ? 1 : 0));
        return *this;
    }
    // the outputs are affine (linear plus offset) in this channel's current values
    Schema& affine(const std::string& name, bool on = true) {
        check(sldrnc_schema_set_affine(h_.get(), name.c_str(), on ? 1 : 0));
        return *this;
    }
    int column(const std::string& name) const { return sldrnc_schema_column(h_.get(), name.c_str()); }
    int width() const { return sldrnc_schema_input_size(h_.get()); }
    int output_count() const { return sldrnc_schema_output_size(h_.get()); }
    const sldrnc_schema* handle() const { return h_.get(); }

   private:
    std::shared_ptr<sldrnc_schema> h_;
};

// ------------------------------------------------------------------------------------------------ options / results
struct TrainOptions {
    Size size = Size::Medium;
    int steps = 12000;
    int batch = 512;
    double learning_rate = 5e-3;
    double weight_decay = 0.2;
    unsigned long long seed = 0;
    int threads = 0;                 // 0 = all cores
    bool fit_correction = true;      // also tune self-correction on the validation data
    bool verbose = false;
    int correction_delay_ticks = 1;  // tune self-correction for measurements this many ticks late (1 = the next tick)
    sldrnc_train_options c() const {
        sldrnc_train_options o;
        sldrnc_train_options_default(&o);
        o.size = (sldrnc_size)size; o.steps = steps; o.batch = batch; o.learning_rate = learning_rate; o.weight_decay = weight_decay;
        o.seed = seed; o.threads = threads; o.fit_correction = fit_correction ? 1 : 0; o.verbose = verbose ? 1 : 0;
        o.correction_delay_ticks = correction_delay_ticks;
        return o;
    }
};
// Errors are normalised MSE (0 = perfect, 1 = no better than the average).
struct Report {
    double train_error = 0, validation_error = 0, validation_error_corrected = 0, correction_time_ms = 0, seconds = 0;
    long long macs_per_tick = 0;
    int steps = 0;
    int correction_delay_ticks = 0;  // the measurement delay self-correction was tuned for (0 = not fitted)
    static Report from(const sldrnc_report& r) {
        return {r.train_nmse, r.val_nmse, r.val_nmse_corrected, r.correction_time_ms, r.seconds, (long long)r.macs_per_tick, r.steps,
                r.correction_delay_ticks};
    }
};
struct Info {
    Mode mode;
    int input_size, output_size;
    bool has_correction;
    double rate_hz, correction_time_ms;
    long long macs_per_tick, parameters;
    double warmup_seconds;            // FULL: a fresh session is fully accurate after this much data
    int correction_delay_ticks;       // the measurement delay self-correction was tuned for (0 = no correction)
    bool history;                     // FULL: uses the recent past (false: current inputs only)
    int affine_inputs;                // FULL: input values the outputs are affine in
    long long table_bytes;            // > 0: the model is a compiled table of this size
};
struct DenseLayer {
    int in = 0, out = 0;
    Activation activation = Activation::Identity;
    std::vector<float> W;            // in x out, row-major (W[i * out + o])
    std::vector<float> b;            // out
};
struct RunResult {
    std::vector<double> predictions; // ticks x outputs
    double error = 0;                // NaN if no targets were given
    std::vector<double> error_per_output;
};

class Model;

// ------------------------------------------------------------------------------------------------ fleet
// Many robots stepped together: row r of X / Y belongs to robot r.  Same numbers as separate sessions, in one call (faster for larger networks).
class Fleet {
   public:
    void step(const double* X, double* Y) { check(sldrnc_fleet_step(h_.get(), X, Y)); }
    std::vector<double> step(const std::vector<double>& X) {
        if (X.size() != robots_ * (std::size_t)in_) throw Error(SLDRNC_ERROR_ARGUMENT, "X must hold robots x input size values");
        std::vector<double> Y(robots_ * (std::size_t)out_);
        step(X.data(), Y.data());
        return Y;
    }
    void observe(const double* Y_measured, int delay_ticks = 1) {
        check(delay_ticks == 1 ? sldrnc_fleet_observe(h_.get(), Y_measured) : sldrnc_fleet_observe_delayed(h_.get(), Y_measured, delay_ticks));
    }
    void reset() { check(sldrnc_fleet_reset(h_.get())); }
    std::size_t robots() const { return robots_; }
    int input_size() const { return in_; }
    int output_size() const { return out_; }

   private:
    friend class Model;
    std::shared_ptr<sldrnc_model> model_;
    std::shared_ptr<sldrnc_fleet> h_;
    std::size_t robots_ = 0;
    int in_ = 0, out_ = 0;
};

// ------------------------------------------------------------------------------------------------ session
// Run-time state for one robot.  Not thread-safe; create one per robot / thread.  Keeps its model alive.
class Session {
   public:
    // predict this tick (x: one input row; y: output_size values)
    void step(const double* x, double* y) { check(sldrnc_session_step(h_.get(), x, y)); }
    std::vector<double> step(const std::vector<double>& x) {
        std::vector<double> y(out_);
        if ((int)x.size() != in_) throw Error(SLDRNC_ERROR_ARGUMENT, "input row has " + std::to_string(x.size()) + " values, expected " + std::to_string(in_));
        step(x.data(), y.data());
        return y;
    }
    // feed measured outputs: delay_ticks = 1 for the tick just predicted, D for the tick predicted D - 1 steps before it
    void observe(const double* y_measured, int delay_ticks = 1) {
        check(delay_ticks == 1 ? sldrnc_session_observe(h_.get(), y_measured) : sldrnc_session_observe_delayed(h_.get(), y_measured, delay_ticks));
    }
    void observe(const std::vector<double>& y_measured, int delay_ticks = 1) {
        if ((int)y_measured.size() != out_) throw Error(SLDRNC_ERROR_ARGUMENT, "measurement has the wrong number of values");
        observe(y_measured.data(), delay_ticks);
    }
    void reset() { check(sldrnc_session_reset(h_.get())); }
    int input_size() const { return in_; }
    int output_size() const { return out_; }

   private:
    friend class Model;
    std::shared_ptr<sldrnc_model> model_;
    std::shared_ptr<sldrnc_session> h_;
    int in_ = 0, out_ = 0;
};

// ------------------------------------------------------------------------------------------------ model
class Model {
   public:
    // MODE 1 -- FULL: train an SLD-RNC model.  X: raw signal rows (schema width); Y: measured targets.
    static Model train(const Schema& schema, const Data& X_train, const Data& Y_train, const Data& X_val, const Data& Y_val,
                       const TrainOptions& options = {}, Report* report = nullptr) {
        if (X_train.ticks != Y_train.ticks || X_val.ticks != Y_val.ticks) throw Error(SLDRNC_ERROR_ARGUMENT, "X and Y must have the same number of ticks");
        if ((int)X_train.columns != schema.width() || (int)X_val.columns != schema.width())
            throw Error(SLDRNC_ERROR_ARGUMENT, "X must have " + std::to_string(schema.width()) + " columns (the schema width)");
        if (schema.output_count() <= 0) throw Error(SLDRNC_ERROR_ARGUMENT, "call schema.outputs(...) before training");
        if ((int)Y_train.columns != schema.output_count() || (int)Y_val.columns != schema.output_count())
            throw Error(SLDRNC_ERROR_ARGUMENT, "Y must have " + std::to_string(schema.output_count()) + " columns (the schema outputs)");
        const sldrnc_train_options o = options.c();
        sldrnc_report r;
        sldrnc_model* m = nullptr;
        check(sldrnc_train_full(schema.handle(), X_train.data, Y_train.data, X_train.ticks, X_val.data, Y_val.data, X_val.ticks, &o, &m, &r));
        if (report) *report = Report::from(r);
        return Model(m);
    }
    // MODE 1 -- FULL, from several recordings (e.g. one per log file).  Element k of X_train goes with element k of
    // Y_train.  Each recording is one continuous stretch with its own 2-s warm-up; never join separate recordings.
    static Model train(const Schema& schema, const std::vector<Data>& X_train, const std::vector<Data>& Y_train, const std::vector<Data>& X_val,
                       const std::vector<Data>& Y_val, const TrainOptions& options = {}, Report* report = nullptr) {
        if (X_train.size() != Y_train.size() || X_val.size() != Y_val.size() || X_train.empty() || X_val.empty())
            throw Error(SLDRNC_ERROR_ARGUMENT, "give one Y recording per X recording, and at least one of each for training and validation");
        if (schema.output_count() <= 0) throw Error(SLDRNC_ERROR_ARGUMENT, "call schema.outputs(...) before training");
        auto unpack = [&](const std::vector<Data>& X, const std::vector<Data>& Y, std::vector<const double*>& px, std::vector<const double*>& py,
                          std::vector<std::size_t>& n) {
            for (std::size_t k = 0; k < X.size(); k++) {
                if (X[k].ticks != Y[k].ticks) throw Error(SLDRNC_ERROR_ARGUMENT, "recording " + std::to_string(k) + ": X and Y must have the same number of ticks");
                if ((int)X[k].columns != schema.width()) throw Error(SLDRNC_ERROR_ARGUMENT, "X must have " + std::to_string(schema.width()) + " columns (the schema width)");
                if ((int)Y[k].columns != schema.output_count()) throw Error(SLDRNC_ERROR_ARGUMENT, "Y must have " + std::to_string(schema.output_count()) + " columns (the schema outputs)");
                px.push_back(X[k].data); py.push_back(Y[k].data); n.push_back(X[k].ticks);
            }
        };
        std::vector<const double*> xt, yt, xv, yv;
        std::vector<std::size_t> nt, nv;
        unpack(X_train, Y_train, xt, yt, nt);
        unpack(X_val, Y_val, xv, yv, nv);
        const sldrnc_train_options o = options.c();
        sldrnc_report r;
        sldrnc_model* m = nullptr;
        check(sldrnc_train_full_multi(schema.handle(), xt.size(), xt.data(), yt.data(), nt.data(), xv.size(), xv.data(), yv.data(), nv.data(), &o, &m, &r));
        if (report) *report = Report::from(r);
        return Model(m);
    }
    // MODE 3 -- RAW: your own fully connected network
    static Model from_layers(const std::vector<DenseLayer>& layers, const std::vector<float>& input_mean = {}, const std::vector<float>& input_std = {},
                             const std::vector<float>& output_mean = {}, const std::vector<float>& output_std = {}) {
        std::vector<sldrnc_layer> L;
        for (const auto& l : layers) {
            if (l.W.size() != (size_t)l.in * l.out || (!l.b.empty() && l.b.size() != (size_t)l.out))
                throw Error(SLDRNC_ERROR_ARGUMENT, "layer weights do not match in x out");
            L.push_back({l.in, l.out, (sldrnc_activation)l.activation, l.W.data(), l.b.empty() ? nullptr : l.b.data()});
        }
        auto p = [](const std::vector<float>& v) { return v.empty() ? nullptr : v.data(); };
        sldrnc_model* m = nullptr;
        check(sldrnc_model_from_layers((int)L.size(), L.data(), p(input_mean), p(input_std), p(output_mean), p(output_std), &m));
        return Model(m);
    }
    // MODE 3 -- RAW: import an ONNX file
    static Model from_onnx(const std::string& path) {
        sldrnc_model* m = nullptr;
        check(sldrnc_model_import_onnx(path.c_str(), &m));
        return Model(m);
    }
    static Model from_onnx_bytes(const std::vector<unsigned char>& bytes) {
        sldrnc_model* m = nullptr;
        check(sldrnc_model_import_onnx_memory(bytes.data(), bytes.size(), &m));
        return Model(m);
    }
    static Model load(const std::string& path) {
        sldrnc_model* m = nullptr;
        check(sldrnc_model_load(path.c_str(), &m));
        return Model(m);
    }
    // MODE 2 -- CORRECTION: add self-correction to this (RAW) model, tuned on validation data for measurements
    // delay_ticks late (1 = the next tick).  Call before creating sessions from this model.
    Report add_correction(const Data& X, const Data& Y, double rate_hz, const std::vector<int>& velocity_index = {}, int delay_ticks = 1) {
        if (X.ticks != Y.ticks) throw Error(SLDRNC_ERROR_ARGUMENT, "X and Y must have the same number of ticks");
        if ((int)X.columns != info().input_size) throw Error(SLDRNC_ERROR_ARGUMENT, "X must have the model's input width");
        if ((int)Y.columns != info().output_size) throw Error(SLDRNC_ERROR_ARGUMENT, "Y must have one column per model output");
        if (!velocity_index.empty() && (int)velocity_index.size() != info().output_size) throw Error(SLDRNC_ERROR_ARGUMENT, "velocity_index needs one entry per output");
        sldrnc_report r;
        check(sldrnc_model_add_correction(h_.get(), X.data, Y.data, X.ticks, rate_hz, velocity_index.empty() ? nullptr : velocity_index.data(), delay_ticks, &r));
        return Report::from(r);
    }
    // A copy of this FULL / CORRECTION model with self-correction re-tuned for measurements delay_ticks late, on
    // validation data (same rows as the model takes).  This model is unchanged.
    Model tune_correction(const Data& X, const Data& Y, int delay_ticks, Report* report = nullptr) const {
        if (X.ticks != Y.ticks) throw Error(SLDRNC_ERROR_ARGUMENT, "X and Y must have the same number of ticks");
        const Info i = info();
        if ((int)X.columns != i.input_size) throw Error(SLDRNC_ERROR_ARGUMENT, "X must have the model's input width");
        if ((int)Y.columns != i.output_size) throw Error(SLDRNC_ERROR_ARGUMENT, "Y must have one column per model output");
        sldrnc_report r;
        sldrnc_model* m = nullptr;
        check(sldrnc_model_tune_correction(h_.get(), X.data, Y.data, X.ticks, delay_ticks, &m, &r));
        if (report) *report = Report::from(r);
        return Model(m);
    }
    void save(const std::string& path) const { check(sldrnc_model_save(h_.get(), path.c_str())); }
    Info info() const {
        sldrnc_model_info i;
        check(sldrnc_model_get_info(h_.get(), &i));
        return {(Mode)i.mode, i.input_size, i.output_size, i.has_correction != 0, i.rate_hz, i.correction_time_ms, (long long)i.macs_per_tick, (long long)i.parameters, i.warmup_seconds,
                i.correction_delay_ticks, i.history != 0, i.affine_inputs, (long long)i.table_bytes};
    }
    // A copy compiled into a lookup table (FULL, no history, 1 to 3 non-affine input values): the fastest form.
    // grid: nodes per table dimension in input order (empty: 64 per value, 128 per ANGLE value).
    Model compile_table(const std::vector<int>& grid = {}) const {
        sldrnc_model* m = nullptr;
        check(sldrnc_model_compile_table(h_.get(), (int)grid.size(), grid.empty() ? nullptr : grid.data(), &m));
        return Model(m);
    }
    // Run-time state for many robots stepped together.  threads: 0 = all cores.
    Fleet fleet(std::size_t robots, bool use_correction = true, bool protect = true, int max_delay_ticks = 0, int threads = 0) const {
        sldrnc_fleet_options o{use_correction ? 1 : 0, protect ? 1 : 0, max_delay_ticks, threads};
        sldrnc_fleet* f = nullptr;
        check(sldrnc_fleet_create(h_.get(), robots, &o, &f));
        Fleet out;
        out.model_ = h_;
        out.h_.reset(f, sldrnc_fleet_free);
        out.robots_ = robots;
        const Info i = info();
        out.in_ = i.input_size;
        out.out_ = i.output_size;
        return out;
    }
    // use_correction: apply self-correction (if the model has it).  protect: ignore implausible measurements and bound
    // the correction.  max_delay_ticks: the largest delay observe() will accept (0 = max(256, the model's tuned delay)).
    Session session(bool use_correction = true, bool protect = true, int max_delay_ticks = 0) const {
        sldrnc_session_options o{use_correction ? 1 : 0, protect ? 1 : 0, max_delay_ticks};
        sldrnc_session* s = nullptr;
        check(sldrnc_session_create(h_.get(), &o, &s));
        Session out;
        out.model_ = h_;
        out.h_.reset(s, sldrnc_session_free);
        const Info i = info();
        out.in_ = i.input_size;
        out.out_ = i.output_size;
        return out;
    }
    // Replay a recording (step every tick, observe each row of Y with delay delay_ticks) and score it from tick
    // score_from on.
    RunResult run(const Data& X, const Data* Y = nullptr, int delay_ticks = 1, std::size_t score_from = 0, bool use_correction = true,
                  bool protect = true) const {
        const Info i = info();
        if ((int)X.columns != i.input_size) throw Error(SLDRNC_ERROR_ARGUMENT, "X must have the model's input width");
        RunResult r;
        r.predictions.resize(X.ticks * i.output_size);
        r.error_per_output.resize(i.output_size);
        sldrnc_session_options o{use_correction ? 1 : 0, protect ? 1 : 0, 0};
        check(sldrnc_model_run(h_.get(), &o, X.data, Y ? Y->data : nullptr, X.ticks, delay_ticks, score_from, r.predictions.data(), &r.error,
                               r.error_per_output.data()));
        return r;
    }
    const sldrnc_model* handle() const { return h_.get(); }

   private:
    explicit Model(sldrnc_model* m) : h_(m, sldrnc_model_free) {}
    std::shared_ptr<sldrnc_model> h_;
};

}  // namespace sldrnc
