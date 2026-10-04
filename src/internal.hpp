// internal.hpp -- SLD-RNC library internals (not installed; never exposed through the public headers).
#pragma once
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "sldrnc/sldrnc.h"

namespace sldi {

// Thrown inside the library, turned into a status code at the C boundary.
struct Error : std::runtime_error {
    sldrnc_status code;
    Error(sldrnc_status c, const std::string& m) : std::runtime_error(m), code(c) {}
};
[[noreturn]] inline void fail(sldrnc_status c, const std::string& m) { throw Error(c, m); }

// ------------------------------------------------------------------------------------------------ schema
struct Channel {
    std::string name;
    int kind = 0, size = 0, offset = 0;
    bool affine = false;                       // the outputs are affine in this channel's current values
};
struct Schema {
    double rate = 0;
    std::vector<Channel> ch;
    int width = 0;
    int n_out = 0;
    std::vector<int> vel_idx;                  // per output: column of the raw row, -1 = none
    bool history = true;                       // false: instantaneous features only (no warm-up; rows may be unordered)
};

// ------------------------------------------------------------------------------------------------ features
// Time constants and lags of the input recipe, converted to ticks for the schema's rate.
struct FeatSpec {
    int nf = 0;
    int lag[4] = {0, 0, 0, 0};
    int max_lag = 0;
    double bank_a[4] = {0, 0, 0, 0};           // exponential-average coefficients
    double gyro_a = 0;
    double rate = 0;
    bool history = true;
    std::vector<int> ch_feat;                  // first feature of each channel; its current values come first
};
FeatSpec make_featspec(const Schema& s);
// FULL models: seconds of data a fresh session needs before its slowest input features have settled
inline double warmup_seconds(const Schema& s) { return s.history ? 2.0 : 0.0; }

struct FeatState {
    bool started = false;
    int pos = 0;                               // ring position
    std::vector<double> mem;                   // all per-channel state, laid out by layout()
};
void feat_init(const Schema& s, const FeatSpec& f, FeatState& st);
void feat_step(const Schema& s, const FeatSpec& f, FeatState& st, const double* x, double* out);

// ------------------------------------------------------------------------------------------------ network
struct Layer {
    int in = 0, out = 0, outp = 0;             // outp = out padded to a multiple of 16
    int act = 0;
    std::vector<float> W;                      // in x outp, input-major, zero padded
    std::vector<float> b;                      // outp
};
struct Net {
    std::vector<Layer> L;
    int in() const { return L.empty() ? 0 : L.front().in; }
    int out() const { return L.empty() ? 0 : L.back().out; }
    int max_width() const;
    int64_t macs() const;
    int64_t params() const;
    void forward(const float* x, float* y, float* scratch) const;   // scratch: 2 * max_width floats
};
void set_layer(Layer& l, int in, int out, int act, const float* W, const float* b);   // W in x out
float activate(int act, float z);
void activate_n(int act, float* z, int n);       // z[i] = activate(act, z[i]), the same bits, one loop per activation

// ------------------------------------------------------------------------------------------------ tables
// A FULL model without history whose non-affine inputs are at most 3 raw values, compiled into a grid of its network's
// outputs.  Lookup: Kuhn-simplex (4-node) interpolation in the grid cell; ANGLE dimensions wrap around 2 pi.  Unused
// dimensions have 2 identical node layers and coordinate 0.
struct Table {
    int dims = 0;                              // real dimensions (1..3); 0 = no table
    int n[3] = {2, 2, 2};                      // nodes per dimension
    int col[3] = {-1, -1, -1};                 // raw input column of each dimension (-1 = unused)
    int periodic[3] = {0, 0, 0};
    float lo[3] = {0, 0, 0};                   // linear: u = (x - lo) * scale, clamped to [0, umax]
    float scale[3] = {1, 1, 1};                // periodic: w = (x - lo) * scale (scale = n / 2 pi), wrapped into [0, n)
    float umax[3] = {0, 0, 0};
    float nfl[3] = {2, 2, 2}, inv_n[3] = {0.5f, 0.5f, 0.5f};
    int stride[3] = {4, 2, 1};                 // node strides
    int hp = 0;                                // values per node: the network's outputs, padded to a multiple of 8
    std::vector<float> T;                      // nodes x hp
    bool on() const { return dims > 0; }
    size_t nodes() const { return (size_t)n[0] * n[1] * n[2]; }
};
// scalar lookup for one robot: x3 = the raw values of the table dimensions (0 for unused ones); heads: hp floats
void table_lookup(const Table& t, const float* x3, float* heads);

// kernels (dispatched to AVX2 / FMA where the CPU has it)
// what a table model without correction needs besides the table, for whole rows in and out (a fleet builds it once)
struct TableIO {
    int ldx = 0, no = 0, na = 0;                       // input row width, outputs, affine values
    const int* aff_col = nullptr;                      // raw input column of each affine value
    const double* aff_mean = nullptr;                  // its standardisation (mean, reciprocal std), in double
    const double* aff_inv = nullptr;
    const double* out_mean = nullptr;                  // output scaling; NULL = none
    const double* out_std = nullptr;
};
struct Kernels {
    void (*gemv)(const float* x, int in, const float* W, int ldw, const float* b, float* z);
    void (*gemm_rows)(const float* X, int n, int ldx, int in, const float* W, int ldw, const float* b, float* Z, int ldz);
    void (*outer_acc)(const float* X, int n, int ldx, int j0, int j1, const float* D, int ldd, int out, float* dW, int ldw);
    void (*back_input)(const float* D, int n, int ldd, int out, const float* W, int ldw, int in, float* dX, int ldx);
    // n rows, each exactly as gemv would compute it (bit for bit), sharing every weight load across rows
    void (*gemv_rows)(const float* X, int n, int ldx, int in, const float* W, int ldw, const float* b, float* Z, int ldz);
    // 8 robots: xs = 3 x 8 raw values (dimension-major), heads = 8 x hp; exactly as table_lookup per robot
    void (*table8)(const Table& t, const float* xs, float* heads);
    // 8 robots of a table model without correction, rows in (X, io.ldx wide) to rows out (Y, io.no wide), exactly as a
    // session computes them; heads: 8 x hp scratch.  NULL when the kernel set has none (the fleet then goes per robot)
    void (*table_rows8)(const Table& t, const TableIO& io, const double* X, double* Y, float* heads);
};
const Kernels& kernels();
const char* kernel_name();

// ------------------------------------------------------------------------------------------------ correction
struct Correction {
    bool on = false;
    double tau_ticks = 0;
    std::vector<int> vel_idx;                  // per output, column of the input row (-1 none)
    std::vector<double> vel_sd;                // per output
    std::vector<double> res_sd;                // per output, residual std (standardised units) -> protection gate
    std::vector<double> mean, sd;              // per output: correction works on (y - mean) / sd
    double gate = 10.0;                        // protection: implausible = |innovation| >= gate x its RMS on validation data
    double clamp = 5.0;                        // protection: |correction| <= clamp x the output's standard deviation
    int frozen_ticks = 20;                     // protection: a measurement repeating exactly this long is a frozen sensor
    int delay_ticks = 1;                       // the measurement delay tau was tuned for (1 = next tick)
};
struct CorrState {
    std::vector<double> th, P;                 // n x 4, n x 16
    std::vector<double> h, o;                  // last step's regressors (n x 4) and standardised predictions (n)
    std::vector<int> rejected;                 // consecutive rejected measurements per output
    std::vector<double> last_meas;             // previous raw measurement per output (frozen-sensor check)
    std::vector<int> same;                     // consecutive exact repeats per output
    std::vector<unsigned char> valid;          // scratch: measurement usable this tick
    bool primed = false;
};
void corr_init(const Correction& c, CorrState& s, int n);
void corr_regress(const Correction& c, const double* o_std, const double* x, double* h, int n);
void corr_apply(const Correction& c, const CorrState& s, const double* h, double* corr, int n, bool protect);
void corr_update(const Correction& c, CorrState& s, const double* h, const double* r, int n, bool protect, const unsigned char* valid = nullptr);
// frozen-sensor check on the raw measurements; fills s.valid and returns it
const unsigned char* corr_screen(const Correction& c, CorrState& s, const double* y_raw, int n);
// tune tau for measurements `delay` ticks late on a stream of standardised predictions / targets (and input rows for
// velocities); sets c.delay_ticks; returns the nmse at that delay
// starts (optional): rows where a new recording begins (the filter restarts there)
double corr_tune(Correction& c, const std::vector<double>& o_std, const std::vector<double>& y_std, const double* X, int width,
                 size_t T, int n, double rate, int delay, const std::vector<size_t>* starts = nullptr);

// ------------------------------------------------------------------------------------------------ model
struct Model {
    int mode = SLDRNC_MODE_RAW;
    double rate = 0;
    Schema schema;                             // FULL only
    FeatSpec fs;                               // FULL only
    std::vector<float> in_mean, in_std;        // FULL: feature standardisation; RAW: optional input normalisation
    Net net;
    std::vector<float> out_mean, out_std;      // optional output scaling (FULL: always)
    Correction corr;
    // structured FULL models: outputs = o_0 + sum_k a_k o_k (standardised), the network giving o = [o_0, o_1, ...] from
    // the trunk features and a = the affine channels' current values (standardised features)
    int n_aff = 0;
    std::vector<int> trunk_idx, aff_idx;       // feature indices (empty trunk_idx = every feature)
    std::vector<int> aff_col;                  // the raw input column of each affine value (its feature is an exact copy)
    std::vector<float> raw_lo, raw_hi;         // FULL: training range of every raw input column (tables)
    Table table;
    int input_size() const { return mode == SLDRNC_MODE_FULL ? schema.width : net.in(); }
    int output_size() const { return mode == SLDRNC_MODE_FULL ? schema.n_out : net.out(); }
    int nfeat() const { return mode == SLDRNC_MODE_FULL ? fs.nf : net.in(); }
    int trunk_in() const { return trunk_idx.empty() ? nfeat() : (int)trunk_idx.size(); }
    int64_t macs() const;                      // per prediction
};
// FULL: trunk / affine feature indices from the schema's affine channels
void setup_structure(Model& m);
// a copy of a FULL model (history off, at most 3 non-affine raw inputs) with its network compiled into a table
std::unique_ptr<Model> table_compile(const Model& m, int n_grid, const int* grid);

struct Session {
    const Model* m = nullptr;
    bool use_corr = true, protect = true;
    FeatState fst;
    CorrState cst;
    std::vector<double> feat;                  // nfeat
    std::vector<float> xin, scratch, yout;
    std::vector<double> o_std, c, h, aff;
    std::vector<double> in_mean, in_inv;      // input normalisation in double, reciprocal precomputed
    // the last ring_cap steps' regressors and standardised predictions, for late measurements
    int max_delay_opt = 0;                     // as requested at creation (0 = automatic)
    int ring_cap = 0;
    uint64_t n_steps = 0;
    std::vector<double> hring, oring;
};
// max_delay: the largest measurement delay observe() will accept (0 = max(256, the model's tuned delay)).  lean: per-robot
// state only (a fleet supplies the scratch space and the normalisation constants)
void session_init(Session& s, const Model& m, bool use_corr, bool protect, int max_delay = 0, bool lean = false);
// predict; also records the standardised prediction and regressors for observe()
void session_step(Session& s, const double* x, double* y);
// a measurement of the step `delay` - 1 steps before the latest (1 <= delay <= s.ring_cap)
void session_observe(Session& s, const double* y_meas, int delay = 1);
// the stages of a step (a fleet runs the middle one, the network or the table, for many robots at once):
//   prepare: features (into feat), the network's input row xin (trunk features, normalised) and the affine values
//   finish:  out = the network's (or the table's) outputs -> combined, scaled, corrected -> y
void session_prepare(Session& s, const double* x, double* feat, const double* mean, const double* inv, float* xin, double* aff);
void session_finish(Session& s, const double* x, const float* out, const double* aff, double* y);
// prepare for a table model, which needs no robot state (no history): the standardised affine values
void table_prepare(const Model& m, const double* x, const double* mean, const double* inv, double* aff);
// the first part of finish, which needs no robot state: network / table outputs + affine values -> scaled outputs
void combine_outputs(const Model& m, const float* out, const double* aff, double* y);
// x3 of a table model: the raw values of the table dimensions
void table_inputs(const Table& t, const double* x, float* x3);

// ------------------------------------------------------------------------------------------------ fleets
struct Fleet {
    const Model* m = nullptr;
    size_t n = 0;
    int threads = 1, parts = 1, chunk = 48;    // parts: workers, each with its own scratch; chunk: robots per work item
    bool use_corr = true, protect = true;
    int max_delay_opt = 0;
    std::vector<Session> rob;                  // per-robot state (lean sessions)
    std::vector<double> in_mean, in_inv;
    std::vector<std::vector<float>> xin, act, out, xs;      // per part scratch
    std::vector<std::vector<double>> feat, aff;
    TableIO tio;                               // a table model's whole-row constants (pointing into the vectors below)
    std::vector<int> tio_col;
    std::vector<double> tio_mean, tio_inv, tio_omean, tio_ostd;
};
void fleet_init(Fleet& f, const Model& m, size_t n, bool use_corr, bool protect, int max_delay, int threads);
void fleet_step(Fleet& f, const double* X, double* Y);
void fleet_observe(Fleet& f, const double* Y, int delay);

// training / building.  A recording is one continuous stretch of rows: X (T x schema width) and Y (T x outputs).
struct Recording { const double* X; const double* Y; size_t T; };
std::unique_ptr<Model> train_full(const Schema& s, const std::vector<Recording>& train, const std::vector<Recording>& val,
                                  const sldrnc_train_options& o, sldrnc_report& rep);
void add_correction(Model& m, const double* X, const double* Y, size_t T, double rate, const int* vel_idx, int delay, sldrnc_report& rep);
std::unique_ptr<Model> tune_correction(const Model& m, const double* X, const double* Y, size_t T, int delay, sldrnc_report& rep);

// files
void save_model(const Model& m, std::vector<uint8_t>& out);
std::unique_ptr<Model> load_model(const uint8_t* p, size_t n);
std::unique_ptr<Model> import_onnx(const uint8_t* p, size_t n);

// threads
int hw_threads();
void parallel_for(int threads, int n, void (*fn)(void* ctx, int i0, int i1), void* ctx);

// small deterministic RNG (identical on every platform)
struct Rng {
    uint64_t s[4];
    explicit Rng(uint64_t seed);
    uint64_t next();
    double uniform() { return (double)(next() >> 11) * (1.0 / 9007199254740992.0); }
};

}  // namespace sldi
