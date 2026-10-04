# C++ API reference

SLD-RNC: Self-correcting Learned Dynamics for Real-time Neural Control.

```cpp
#include <sldrnc/sldrnc.hpp>
```

A header-only C++17 layer over the [C interface](c.md). It adds RAII handles, exceptions, enums and `std::vector`.
Link against the `sldrnc` shared library:

```cmake
find_package(sldrnc REQUIRED)                       # -DCMAKE_PREFIX_PATH=<package root>
target_link_libraries(my_app PRIVATE sldrnc::sldrnc)
```

**Conventions**
- Everything is in namespace `sldrnc`.
- Data is row-major `double`, one row per tick, passed as a [`Data`](#data) view.
- Every failure throws `sldrnc::Error`, whose `code()` is an `sldrnc_status` and `what()` a readable message.
- Errors are normalised mean squared errors: 0 = perfect, 1 = no better than always predicting the average.

Contents: [basics](#basics) · [Data](#data) · [Schema](#schema) · [TrainOptions](#trainoptions) · [Model](#model) ·
[Session](#session) · [result types](#result-types) · [ownership and threads](#ownership-and-threads) ·
[complete example](#complete-example)

---

## Basics

| | |
|---|---|
| `std::string version()` | library version, e.g. `"1.0.0"` |
| `class Error : std::runtime_error` | `sldrnc_status code() const`; `what()` is the message |
| `void check(sldrnc_status)` | throws `Error` for anything but `SLDRNC_OK` (useful when mixing in C calls) |

| enum class | values |
|---|---|
| `Mode` | `Full`, `Correction`, `Raw` |
| `Input` | `Angle` (rotary joint angles, rad; sine and cosine added), `Position` (used as they are), `Velocity`, `Acceleration`, `Gyro` (3), `Accel` (3), `Quaternion` (w, x, y, z), `Generic` |
| `Activation` | `Identity`, `Relu`, `Relu2` (squared ReLU), `LeakyRelu` (0.01), `Tanh`, `Sigmoid`, `Gelu` (tanh form), `Silu` |
| `Size` | `Small` (fastest), `Medium`, `Large` (most accurate) |

The meaning of each `Input` kind is in the [Python reference](python.md#enums) and in [Concepts](concepts.md#schemas-full-mode).

---

## Data

```cpp
struct Data {
    const double* data; std::size_t ticks; std::size_t columns;
    Data(const double* d, std::size_t ticks, std::size_t columns);
    Data(const std::vector<double>& v, std::size_t columns);   // ticks = v.size() / columns
};
```

A non-owning view of a recording: `ticks` rows of `columns` doubles, row-major. The memory must outlive the call it
is passed to. The vector constructor throws if `v.size()` is not a multiple of `columns`.

---

## Schema

Describes one tick of raw signals for FULL training. Cheap to copy (copies share one schema).

| member | |
|---|---|
| `explicit Schema(double rate_hz)` | ticks per second of your logs and control loop |
| `Schema& add(const std::string& name, Input kind, int size)` | append a channel; chainable |
| `Schema& outputs(int n, const std::string& velocity_channel)` | `n` outputs; output `j` belongs to the joint whose velocity is element `j` of that channel |
| `Schema& outputs(int n, const std::vector<int>& velocity_index = {})` | `n` outputs, with explicit velocity columns (`-1` = none) |
| `Schema& history(bool on)` | `false`: the model sees the current inputs only (memoryless targets such as a current controller): no warm-up, unordered training rows allowed, and the model can be compiled into a table. Default `true` |
| `Schema& affine(const std::string& name, bool on = true)` | the outputs are affine in this channel's current values, with coefficients learnt from the other inputs (torques in accelerations; voltages in speed and current references). Not for `Quaternion`; at least one channel must stay non-affine |
| `int column(const std::string& name) const` | first column of a channel, `-1` if absent |
| `int width() const` | columns of the input row |
| `int output_count() const` | number of outputs (0 until `outputs`) |

```cpp
sldrnc::Schema schema(1000.0);
schema.add("q", sldrnc::Input::Position, 12)
      .add("qd", sldrnc::Input::Velocity, 12)
      .add("orientation", sldrnc::Input::Quaternion, 4)
      .add("gyro", sldrnc::Input::Gyro, 3)
      .outputs(12, "qd");

sldrnc::Schema drive(20000.0);                    // a 20 kHz motor drive: memoryless, voltages affine in 3 inputs
drive.history(false)
     .add("i_dq", sldrnc::Input::Generic, 2).add("angle", sldrnc::Input::Angle, 1)
     .add("speed", sldrnc::Input::Velocity, 1).add("i_dq_ref", sldrnc::Input::Generic, 2)
     .outputs(2).affine("speed").affine("i_dq_ref");
```

What `history` and `affine` do, and when to use them: [python.md](python.md#schema).

---

## TrainOptions

| field | default | |
|---|---|---|
| `Size size` | `Size::Medium` | model size |
| `int steps` | 12000 | optimisation steps |
| `int batch` | 512 | ticks per step |
| `double learning_rate` | 5e-3 | peak learning rate |
| `double weight_decay` | 0.2 | regularisation |
| `unsigned long long seed` | 0 | same seed + same thread count → same model |
| `int threads` | 0 | worker threads, 0 = all cores |
| `bool fit_correction` | `true` | tune self-correction on the validation data |
| `bool verbose` | `false` | print progress |
| `int correction_delay_ticks` | 1 | tune self-correction for measurements this late ([Measurement delay](concepts.md#measurement-delay)) |

---

## Model

A trained or imported model. A `Model` object is a shared handle: copying it is cheap, and copies refer to the same
model.

### Creating

| | mode |
|---|---|
| `static Model train(const Schema&, const Data& X_train, const Data& Y_train, const Data& X_val, const Data& Y_val, const TrainOptions& = {}, Report* = nullptr)` | FULL |
| `static Model train(const Schema&, const std::vector<Data>& X_train, const std::vector<Data>& Y_train, const std::vector<Data>& X_val, const std::vector<Data>& Y_val, const TrainOptions& = {}, Report* = nullptr)` | FULL, several recordings |
| `static Model from_onnx(const std::string& path)` | RAW |
| `static Model from_onnx_bytes(const std::vector<unsigned char>& bytes)` | RAW |
| `static Model from_layers(const std::vector<DenseLayer>& layers, input_mean = {}, input_std = {}, output_mean = {}, output_std = {})` | RAW |
| `static Model load(const std::string& path)` | any |

- **`train`**: `X_*` must have `schema.width()` columns and `Y_*` `schema.output_count()`. Each recording is one
  continuous stretch at the schema rate; validation should be recorded after the training data. The first 2 s of
  each recording are warm-up. Needs at least 1000 training ticks and 200 validation ticks after the warm-ups.
- **Several recordings** (one per log file): pass `std::vector<Data>`, element *k* of `X_train` going with element
  *k* of `Y_train`, e.g. `Model::train(schema, {{Xa, 24}, {Xb, 24}}, {{Ya, 12}, {Yb, 12}}, {{Xv, 24}}, {{Yv, 12}})`.
  Never join separate recordings into one array.
- **`from_layers`**: each `DenseLayer{in, out, activation, W, b}` has `W` of `in × out` floats, row-major
  (`W[i * out + o]`), and `b` of `out` floats (empty = zeros). The optional vectors normalise inputs
  (`(x − mean) / std`) and scale outputs (`y · std + mean`).
- **`from_onnx`**: see [supported ONNX operations](model_formats.md#networks-sld-rnc-can-run-raw--correction).

### Self-correction

| | |
|---|---|
| `Report add_correction(const Data& X, const Data& Y, double rate_hz, const std::vector<int>& velocity_index = {}, int delay_ticks = 1)` | **CORRECTION mode**: adds self-correction to this RAW model, in place. Call before creating sessions |
| `Model tune_correction(const Data& X, const Data& Y, int delay_ticks, Report* report = nullptr) const` | a **copy** of this FULL / CORRECTION model, re-tuned for another measurement delay; this model is unchanged |

`X` holds validation rows in the model's own input format, and `Y` the measured outputs. `velocity_index` is the
column of `X` holding each output's joint velocity (`-1` = none). At least 200 ticks (FULL: 2 s + 200) and
4 × `delay_ticks` are needed.

### Using

| | |
|---|---|
| `Session session(bool use_correction = true, bool protect = true, int max_delay_ticks = 0) const` | run-time state for one robot |
| `Fleet fleet(std::size_t robots, bool use_correction = true, bool protect = true, int max_delay_ticks = 0, int threads = 0) const` | run-time state for many robots stepped together ([Fleet](#fleet)); `threads`: 0 = all cores, 1 = the calling thread only |
| `Model compile_table(const std::vector<int>& grid = {}) const` | a copy compiled into a lookup table: FULL, history off, 1 to 3 non-affine input values. `grid`: nodes per dimension (default 64, 128 for angles). See [tables](python.md#tables) |
| `RunResult run(const Data& X, const Data* Y = nullptr, int delay_ticks = 1, std::size_t score_from = 0, bool use_correction = true, bool protect = true) const` | replay a recording through a fresh session; observes each row of `Y` with delay `delay_ticks` and scores rows `>= score_from` |
| `void save(const std::string& path) const` | write a `.sldm` file |
| `Info info() const` | what the model is |
| `const sldrnc_model* handle() const` | the C handle, for mixing with C calls |

`max_delay_ticks` is the largest delay `Session::observe` will accept: 0 means 256 or the model's tuned delay,
whichever is larger. For FULL models, score from at least 2 s of ticks.

---

## Session

| | |
|---|---|
| `void step(const double* x, double* y)` | predict: `x` has `input_size()` values, `y` receives `output_size()` |
| `std::vector<double> step(const std::vector<double>& x)` | the same, returning a new vector (checks the size) |
| `void observe(const double* y_measured, int delay_ticks = 1)` | feed measured outputs |
| `void observe(const std::vector<double>& y_measured, int delay_ticks = 1)` | the same, checking the size |
| `void reset()` | forget history and correction, keeping the options |
| `int input_size() const`, `int output_size() const` | |

With `delay_ticks = 1`, `y_measured` belongs to the tick just predicted. A late measurement, of the tick predicted
`D − 1` steps before the latest, takes `delay_ticks = D`; feed them in tick order. Skipping `observe` is fine, and NaN
values are skipped. `step` and `observe` with a pointer never allocate, lock or block.

---

## Fleet

Many robots of one model, stepped in one call. Row `r` of every array belongs to robot `r`; each robot keeps its own
history and self-correction, and the outputs equal separate sessions bit for bit.

| | |
|---|---|
| `void step(const double* X, double* Y)` | `X`: `robots() × input_size()` values, row-major; `Y` receives `robots() × output_size()` |
| `std::vector<double> step(const std::vector<double>& X)` | the same, returning a new vector (checks the size) |
| `void observe(const double* Y_measured, int delay_ticks = 1)` | every robot's measured outputs; NaN values are skipped |
| `void reset()` | forget every robot's history and correction |
| `std::size_t robots() const`, `int input_size() const`, `int output_size() const` | |

```cpp
sldrnc::Fleet fleet = model.fleet(4096);           // at start-up
std::vector<double> X(4096 * fleet.input_size()), U(4096 * fleet.output_size());
fleet.step(X.data(), U.data());                    // every tick: all 4,096 robots
fleet.observe(Y_measured.data());
```

A fleet is used from one thread at a time and runs its own workers (the library's thread pool) inside `step` and
`observe`. Pointer calls never allocate.

---

## Result types

```cpp
struct Report {          // train / add_correction / tune_correction
    double train_error, validation_error, validation_error_corrected, correction_time_ms, seconds;
    long long macs_per_tick; int steps; int correction_delay_ticks;
};
struct Info {            // Model::info()
    Mode mode; int input_size, output_size; bool has_correction;
    double rate_hz, correction_time_ms; long long macs_per_tick, parameters;
    double warmup_seconds; int correction_delay_ticks;
    bool history; int affine_inputs; long long table_bytes;
};
struct RunResult {       // Model::run()
    std::vector<double> predictions;       // ticks x outputs
    double error;                          // NaN without targets
    std::vector<double> error_per_output;
};
```

The meaning of each field is the same as in the [Python reference](python.md#result-types).

---

## Ownership and threads

- A `Model` can be shared by any number of threads and sessions: it is read-only once built. The exception is
  `add_correction`, which changes the model, so call it before any session exists.
- A `Session` holds a reference to its model, so the model stays alive while the session exists. A session is **not**
  thread-safe: use it from one thread at a time. One session per robot.
- A `Fleet` likewise keeps its model alive and is used from one thread at a time.
- Create sessions and load models outside the real-time path. After that, `step` and `observe` (the pointer
  overloads) are real-time safe.
- Exceptions come only from invalid use (wrong sizes, NULL pointers, a delay beyond `max_delay_ticks`), never from
  valid real-time calls.

---

## Complete example

[`examples/cpp/quickstart.cpp`](../examples/cpp/quickstart.cpp) uses all three modes on a simulated machine and runs
a control loop. The core of it:

```cpp
#include <sldrnc/sldrnc.hpp>

int main() {
    // logs: X rows = q0 q1 qd0 qd1, Y rows = measured torques (see the example for a simulator)
    sldrnc::Schema schema(1000.0);
    schema.add("q", sldrnc::Input::Position, 2).add("qd", sldrnc::Input::Velocity, 2).outputs(2, "qd");

    sldrnc::TrainOptions opt;
    opt.size = sldrnc::Size::Small;
    sldrnc::Report rep;
    sldrnc::Model model = sldrnc::Model::train(schema, {X_train, 4}, {Y_train, 2}, {X_val, 4}, {Y_val, 2}, opt, &rep);
    model.save("robot.sldm");

    sldrnc::Session robot = model.session();
    std::vector<double> torque(2);
    for (;;) {                                         // 1 kHz control loop
        robot.step(read_sensors(), torque.data());     // feed-forward torque
        apply(torque);
        robot.observe(read_torque_sensors());          // self-correction
    }
}
```
