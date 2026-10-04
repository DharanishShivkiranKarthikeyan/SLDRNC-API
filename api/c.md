# C API reference

SLD-RNC: Self-correcting Learned Dynamics for Real-time Neural Control.

```c
#include <sldrnc/sldrnc.h>
```

The C interface is the library's stable boundary: the C++ and Python APIs, and any other language (Rust, C#, Julia,
...), sit on top of it.

**Conventions**
- Every function that can fail returns `sldrnc_status`; `SLDRNC_OK` (0) means success. On failure,
  `sldrnc_last_error()` returns a readable message for the calling thread. Nothing throws, aborts or prints, except
  `verbose` training.
- **Handles** (`sldrnc_schema*`, `sldrnc_model*`, `sldrnc_session*`) are opaque. Each is created by one function and
  freed by its `*_free` function. `*_free(NULL)` is a no-op.
- **Arrays** are row-major `double`, one row per tick. A function never keeps a pointer you pass after it returns.
- **Option structs**: fill them with their `*_default()` function first, then change the fields you need.
- **Handles and reports** returned through `out` / `report` are written only on success. `report` arguments may be
  `NULL`.

Contents: [status and version](#status-and-version) · [enums](#enums) · [schema](#schema) · [FULL
training](#full-training) · [RAW models](#raw-models) · [self-correction](#self-correction) · [model
files](#model-files-and-info) · [sessions](#sessions) · [offline runs](#offline-runs) · [threads and real
time](#threads-and-real-time) · [ABI and versions](#abi-and-versions) · [example](#example)

---

## Status and version

```c
const char* sldrnc_last_error(void);   /* message of the last failing call on this thread; "" if none */
const char* sldrnc_version(void);      /* "1.0.0" */
#define SLDRNC_VERSION_MAJOR 1  /* also _MINOR, _PATCH */
```

| `sldrnc_status` | value | meaning |
|---|---|---|
| `SLDRNC_OK` | 0 | success |
| `SLDRNC_ERROR_ARGUMENT` | 1 | a pointer was NULL, a size did not match, a value was out of range |
| `SLDRNC_ERROR_IO` | 2 | a file could not be opened, read or written |
| `SLDRNC_ERROR_FORMAT` | 3 | a model file is damaged, from a newer version, or not an SLD-RNC / ONNX file |
| `SLDRNC_ERROR_UNSUPPORTED` | 4 | an ONNX model uses an operation SLD-RNC doesn't run (the message names it) |
| `SLDRNC_ERROR_STATE` | 5 | the call doesn't apply to this model or session |
| `SLDRNC_ERROR_DATA` | 6 | training / tuning data is too short, constant, or contains NaN / infinity |
| `SLDRNC_ERROR_INTERNAL` | 7 | a bug (or out of memory); please report the message |

The message pointer stays valid until the next failing call on the same thread.

---

## Enums

| enum | values |
|---|---|
| `sldrnc_mode` | `SLDRNC_MODE_FULL` (0), `SLDRNC_MODE_CORRECTION` (1), `SLDRNC_MODE_RAW` (2) |
| `sldrnc_input_kind` | `SLDRNC_INPUT_POSITION` (0), `_VELOCITY` (1), `_ACCELERATION` (2), `_GYRO` (3, 3 values), `_ACCEL` (4, 3 values), `_QUATERNION` (5, exactly 4 values: w, x, y, z), `_GENERIC` (6), `_ANGLE` (7, rotary joint angles in rad: the model also sees their sine and cosine) |
| `sldrnc_activation` | `SLDRNC_ACT_IDENTITY` (0), `_RELU` (1), `_RELU2` (2, squared ReLU), `_LEAKY_RELU` (3, slope 0.01), `_TANH` (4), `_SIGMOID` (5), `_GELU` (6, tanh form), `_SILU` (7) |
| `sldrnc_size` | `SLDRNC_SIZE_SMALL` (0), `_MEDIUM` (1), `_LARGE` (2) |

---

## Schema

Describes one tick of raw signals for FULL training.

```c
sldrnc_status sldrnc_schema_create(double rate_hz, sldrnc_schema** out);
sldrnc_status sldrnc_schema_add_input(sldrnc_schema* schema, const char* name, sldrnc_input_kind kind, int size);
sldrnc_status sldrnc_schema_set_outputs(sldrnc_schema* schema, int n_outputs, const int* velocity_index);
sldrnc_status sldrnc_schema_set_history(sldrnc_schema* schema, int on);
sldrnc_status sldrnc_schema_set_affine(sldrnc_schema* schema, const char* name, int on);
int           sldrnc_schema_input_size(const sldrnc_schema* schema);
int           sldrnc_schema_output_size(const sldrnc_schema* schema);
int           sldrnc_schema_column(const sldrnc_schema* schema, const char* name);
void          sldrnc_schema_free(sldrnc_schema* schema);
```

| function | |
|---|---|
| `schema_create` | `rate_hz` > 0: ticks per second of your logs and control loop |
| `schema_add_input` | append a channel of `size` values; channels form the input row in the order added. Names must be unique |
| `schema_set_outputs` | declare `n_outputs`. `velocity_index` (may be NULL): `n_outputs` ints, the input column holding each output's joint velocity, `-1` = none. Add the inputs first |
| `schema_set_history` | `on = 0`: the model sees the current inputs only (memoryless targets such as a current controller): no warm-up, unordered training rows allowed, and the model can be compiled into a table. Default 1 |
| `schema_set_affine` | `on = 1`: the outputs are affine in channel `name`'s current values, with coefficients learnt from the other inputs. Not for `QUATERNION` channels; at least one channel must stay non-affine (else `SLDRNC_ERROR_ARGUMENT`) |
| `schema_input_size` | width of the input row (0 for NULL) |
| `schema_output_size` | number of outputs (0 until set) |
| `schema_column` | first column of channel `name`, `-1` if absent |

---

## FULL training

```c
typedef struct sldrnc_train_options {
    sldrnc_size size;              /* default SLDRNC_SIZE_MEDIUM */
    int steps;                     /* default 12000 */
    int batch;                     /* default 512 */
    double learning_rate;          /* default 5e-3 */
    double weight_decay;           /* default 0.2 */
    uint64_t seed;                 /* default 0 */
    int threads;                   /* default 0 = all cores */
    int fit_correction;            /* default 1 */
    int verbose;                   /* default 0 */
    int correction_delay_ticks;    /* default 1 */
} sldrnc_train_options;
void sldrnc_train_options_default(sldrnc_train_options* options);

typedef struct sldrnc_report {
    double train_nmse, val_nmse, val_nmse_corrected, correction_time_ms, seconds;
    int64_t macs_per_tick;
    int steps;
    int correction_delay_ticks;
} sldrnc_report;

sldrnc_status sldrnc_train_full(const sldrnc_schema* schema,
                                const double* X_train, const double* Y_train, size_t ticks_train,
                                const double* X_val, const double* Y_val, size_t ticks_val,
                                const sldrnc_train_options* options, sldrnc_model** out, sldrnc_report* report);

sldrnc_status sldrnc_train_full_multi(const sldrnc_schema* schema,
                                      size_t n_train, const double* const* X_train, const double* const* Y_train,
                                      const size_t* ticks_train,
                                      size_t n_val, const double* const* X_val, const double* const* Y_val,
                                      const size_t* ticks_val,
                                      const sldrnc_train_options* options, sldrnc_model** out, sldrnc_report* report);
```

- `X_*`: `ticks × sldrnc_schema_input_size(schema)` raw signals; `Y_*`: `ticks × n_outputs` measured outputs.
- `sldrnc_train_full_multi` takes several recordings (one per log file): recording *k* is `X_train[k]` /
  `Y_train[k]` with `ticks_train[k]` rows. Each has its own warm-up. Never join separate recordings into one array.
  `sldrnc_train_full` is the one-recording case.
- Each recording is one continuous stretch at the schema rate. Validation picks the best checkpoint and tunes
  self-correction; use a stretch recorded after the training data.
- The first 2 s of each recording are warm-up: not trained on or scored. Minimum: 1000 training ticks and 200
  validation ticks after the warm-ups.
- `options` may be NULL (defaults). The same `seed` and `threads` give the same model, bit for bit.
- `correction_delay_ticks`: tune self-correction for measurements this late ([Measurement delay](concepts.md#measurement-delay)).

**`sldrnc_report`** (every field is set on success):

| field | |
|---|---|
| `train_nmse` | error of the model alone on a sample of the training data (NaN for correction tuning) |
| `val_nmse` | error of the model alone on the validation / tuning data |
| `val_nmse_corrected` | error with self-correction at the tuned delay (NaN if not fitted) |
| `correction_time_ms` | the correction's time constant chosen on the data |
| `seconds` | wall time |
| `macs_per_tick` | multiply-accumulates per prediction |
| `steps` | optimisation steps run (0 for correction tuning) |
| `correction_delay_ticks` | the delay self-correction was tuned for (0 if not fitted) |

Errors are normalised MSE: the mean over outputs of MSE / variance. 0 = perfect, 1 = no better than the average.

---

## RAW models

```c
typedef struct sldrnc_layer {
    int in, out;
    sldrnc_activation activation;
    const float* W;                /* in x out, row-major: W[i * out + o] */
    const float* b;                /* out values, or NULL = zeros */
} sldrnc_layer;

sldrnc_status sldrnc_model_from_layers(int n_layers, const sldrnc_layer* layers,
                                       const float* input_mean, const float* input_std,
                                       const float* output_mean, const float* output_std, sldrnc_model** out);
sldrnc_status sldrnc_model_import_onnx(const char* path, sldrnc_model** out);
sldrnc_status sldrnc_model_import_onnx_memory(const void* data, size_t bytes, sldrnc_model** out);
```

- **`from_layers`**: 1–64 layers, each `in` matching the previous `out`. Weights are copied. The normalisation
  arrays are optional, but give each pair together: `input_mean` / `input_std` (first layer's `in` values) apply
  `(x − mean) / std`; `output_mean` / `output_std` (last layer's `out` values) apply `y · std + mean`.
- **`import_onnx`**: a feed-forward network; see [supported operations](model_formats.md#networks-sld-rnc-can-run-raw--correction).
  An unsupported operation gives `SLDRNC_ERROR_UNSUPPORTED`, naming the operation.

---

## Self-correction

```c
sldrnc_status sldrnc_model_add_correction(sldrnc_model* model, const double* X, const double* Y, size_t ticks,
                                          double rate_hz, const int* velocity_index, int delay_ticks,
                                          sldrnc_report* report);
sldrnc_status sldrnc_model_tune_correction(const sldrnc_model* model, const double* X, const double* Y, size_t ticks,
                                           int delay_ticks, sldrnc_model** out, sldrnc_report* report);
```

| function | |
|---|---|
| `model_add_correction` | **CORRECTION mode.** Adds self-correction to a RAW model, tuned on validation data, **in place**. `X`: `ticks × input_size` (your network's inputs); `Y`: `ticks × output_size`. `velocity_index` (may be NULL): per output, a column of `X` with that joint's velocity, `-1` = none. `delay_ticks` ≥ 1: the measurement delay to tune for. On a FULL model: `SLDRNC_ERROR_STATE`. Call before creating sessions from the model |
| `model_tune_correction` | a **new** model (in `*out`; free it with `sldrnc_model_free`): a copy of a FULL or CORRECTION model with self-correction re-tuned for `delay_ticks`. The original is unchanged and may be in use. `X`, `Y` as that model takes them; for FULL the first 2 s are warm-up. Without correction: `SLDRNC_ERROR_STATE` |

At least 200 ticks (FULL: 2 s + 200) and 4 × `delay_ticks` are needed.

---

## Model files and info

```c
sldrnc_status sldrnc_model_load(const char* path, sldrnc_model** out);
sldrnc_status sldrnc_model_save(const sldrnc_model* model, const char* path);
void          sldrnc_model_free(sldrnc_model* model);

typedef struct sldrnc_model_info {
    sldrnc_mode mode;
    int input_size, output_size, has_correction;
    double rate_hz, correction_time_ms;
    int64_t macs_per_tick, parameters;
    double warmup_seconds;
    int correction_delay_ticks;
    int history;                   /* FULL: 1 = uses the recent past; 0 = current inputs only */
    int affine_inputs;             /* FULL: input values the outputs are affine in (0 = none) */
    int64_t table_bytes;           /* > 0: the model is a compiled table of this size */
} sldrnc_model_info;
sldrnc_status sldrnc_model_get_info(const sldrnc_model* model, sldrnc_model_info* info);

sldrnc_status sldrnc_model_compile_table(const sldrnc_model* model, int n_grid, const int* grid, sldrnc_model** out);
```

- `.sldm` files hold any mode with everything needed to run. They are portable across platforms and checksummed. A
  damaged file, or one from a newer format, gives `SLDRNC_ERROR_FORMAT`. Older files load.
- `info`: `rate_hz` is 0 for RAW models. `correction_time_ms` and `correction_delay_ticks` are 0 without correction.
  `warmup_seconds` is 2 for FULL models with history (the data a fresh session needs before full accuracy), else 0.
- `model_compile_table`: a **new** model (free it with `sldrnc_model_free`) holding a FULL model trained without
  history, whose non-affine inputs are 1 to 3 values, compiled into a lookup table: the fastest form. `grid`:
  `n_grid` node counts, one per table dimension in input order; `n_grid = 0` and `grid = NULL` give the defaults (64
  per value, 128 per `ANGLE` value). `ANGLE` values wrap around; the others cover the training range and are clamped
  outside it. A model with history, a RAW / CORRECTION model or more than 3 non-affine values gives
  `SLDRNC_ERROR_STATE`. See [tables](python.md#tables).

---

## Sessions

```c
typedef struct sldrnc_session_options {
    int use_correction;            /* default 1: apply self-correction if the model has it */
    int protect;                   /* default 1: ignore implausible measurements, bound the correction */
    int max_delay_ticks;           /* default 0: 256 or the model's tuned delay, whichever is larger */
} sldrnc_session_options;
void sldrnc_session_options_default(sldrnc_session_options* options);

sldrnc_status sldrnc_session_create(const sldrnc_model* model, const sldrnc_session_options* options, sldrnc_session** out);
sldrnc_status sldrnc_session_step(sldrnc_session* session, const double* x, double* y);
sldrnc_status sldrnc_session_observe(sldrnc_session* session, const double* y_measured);
sldrnc_status sldrnc_session_observe_delayed(sldrnc_session* session, const double* y_measured, int delay_ticks);
sldrnc_status sldrnc_session_reset(sldrnc_session* session);
void          sldrnc_session_free(sldrnc_session* session);
```

| function | |
|---|---|
| `session_create` | run-time state for one robot. `options` may be NULL (defaults). `max_delay_ticks` (0–1000000) reserves history for late measurements. The model must outlive the session |
| `session_step` | predict: `x` = one input row (`input_size`), `y` receives `output_size` values |
| `session_observe` | the measured outputs of the tick just predicted; they correct the next step on. NaN values are skipped; skipping calls is fine |
| `session_observe_delayed` | a late measurement: the outputs of the tick predicted `delay_ticks − 1` steps before the latest (`delay_ticks = 1` is `session_observe`). Feed them in tick order. More than `max_delay_ticks`: `SLDRNC_ERROR_ARGUMENT`. Older than the session: ignored |
| `session_reset` | forget history and correction, keeping the options |

Protection ignores a frozen sensor (an exact repeat lasting 20 ms) and isolated spikes, and limits the correction to
5 standard deviations of each output ([details](concepts.md#protection)).

---

## Fleets

```c
typedef struct sldrnc_fleet_options {
    int use_correction;            /* default 1 */
    int protect;                   /* default 1 */
    int max_delay_ticks;           /* default 0: the model's tuned delay (memory is reserved per robot) */
    int threads;                   /* default 0: all cores */
} sldrnc_fleet_options;
void sldrnc_fleet_options_default(sldrnc_fleet_options* options);

sldrnc_status sldrnc_fleet_create(const sldrnc_model* model, size_t robots, const sldrnc_fleet_options* options,
                                  sldrnc_fleet** out);
sldrnc_status sldrnc_fleet_step(sldrnc_fleet* fleet, const double* X, double* Y);
sldrnc_status sldrnc_fleet_observe(sldrnc_fleet* fleet, const double* Y_measured);
sldrnc_status sldrnc_fleet_observe_delayed(sldrnc_fleet* fleet, const double* Y_measured, int delay_ticks);
sldrnc_status sldrnc_fleet_reset(sldrnc_fleet* fleet);
size_t        sldrnc_fleet_size(const sldrnc_fleet* fleet);
void          sldrnc_fleet_free(sldrnc_fleet* fleet);
```

| function | |
|---|---|
| `fleet_create` | run-time state for `robots` robots of one model. `options` may be NULL. Unlike a session, the default `max_delay_ticks` reserves only the model's tuned delay per robot (memory scales with the fleet); raise it to feed later measurements. The model must outlive the fleet |
| `fleet_step` | `X`: `robots × input_size` values, row-major (row `r` = robot `r`); `Y` receives `robots × output_size`. Equal, bit for bit, to stepping each robot's own session |
| `fleet_observe`, `fleet_observe_delayed` | every robot's measured outputs (`robots × output_size`), as `session_observe` / `session_observe_delayed`. NaN values are skipped, so a robot without a measurement can pass NaN |
| `fleet_reset` | forget every robot's history and correction |
| `fleet_size` | number of robots (0 for NULL) |

A fleet belongs to one thread at a time. With `threads = 1`, `fleet_step` runs on the calling thread and is
real-time safe like `session_step`. With more threads it uses the library's worker pool: built for throughput, not for
hard deadlines.

---

## Offline runs

```c
sldrnc_status sldrnc_model_run(const sldrnc_model* model, const sldrnc_session_options* options,
                               const double* X, const double* Y, size_t ticks, int delay_ticks, size_t score_from,
                               double* Y_pred, double* nmse, double* nmse_per_output);
```

This replays a recording through a fresh session. It steps every row of `X` and, if `Y` is given (may be NULL),
observes each row of `Y` with delay `delay_ticks` (≥ 1).

| output | |
|---|---|
| `Y_pred` | `ticks × output_size` predictions (may be NULL) |
| `nmse` | the error over ticks `>= score_from` (may be NULL; NaN without `Y`) |
| `nmse_per_output` | per output, `output_size` values (may be NULL) |

`score_from` must be less than `ticks`; for FULL models use at least 2 s of ticks. A non-finite prediction (from
non-finite input) gives `SLDRNC_ERROR_DATA`, naming the tick.

---

## Threads and real time

- **Models** are read-only once built, so share them across threads freely. The one exception is
  `sldrnc_model_add_correction`, which changes the model: call it before creating sessions.
- **Sessions** and **fleets** belong to one thread at a time.
- **Real-time safe:** `sldrnc_session_step`, `sldrnc_session_observe` and `sldrnc_session_observe_delayed` never
  allocate memory (except to store an error message on invalid arguments), take locks, perform I/O or block. Load
  models and create sessions outside the real-time path.
- Training uses `threads` worker threads, and returns only when they have finished.
- Error messages are per thread.

---

## ABI and versions

- The library exports only `sldrnc_*` symbols with C linkage. On Windows link `sldrnc.lib` and ship `sldrnc.dll`; on
  Linux `libsldrnc.so` (glibc 2.17+); on macOS `libsldrnc.dylib` (macOS 11+, install name `@rpath/libsldrnc.dylib`).
- Within one major version, functions keep their signatures and structs keep their layouts. New features arrive as
  new functions. A change to an existing struct or signature means a new major version.
- Use the header that came with the library. At start-up, a program can check that `sldrnc_version()` starts with
  the `SLDRNC_VERSION_MAJOR` it was compiled against.
- `.sldm` model files: a newer library loads older files; an older library refuses newer files with
  `SLDRNC_ERROR_FORMAT`.

---

## Example

```c
#include <stdio.h>
#include <sldrnc/sldrnc.h>

int main(void) {
    sldrnc_model* model = NULL;
    if (sldrnc_model_load("robot.sldm", &model) != SLDRNC_OK) {
        fprintf(stderr, "load failed: %s\n", sldrnc_last_error());
        return 1;
    }
    sldrnc_model_info info;
    sldrnc_model_get_info(model, &info);

    sldrnc_session* robot = NULL;
    if (sldrnc_session_create(model, NULL, &robot) != SLDRNC_OK) {
        fprintf(stderr, "%s\n", sldrnc_last_error());
        sldrnc_model_free(model);
        return 1;
    }
    double x[64], y[32], y_measured[32];       /* sized for info.input_size / info.output_size */
    for (;;) {                                 /* control loop */
        read_sensors(x);
        sldrnc_session_step(robot, x, y);      /* feed-forward outputs for this tick */
        apply(y);
        read_measured(y_measured);
        sldrnc_session_observe(robot, y_measured);
    }
    sldrnc_session_free(robot);
    sldrnc_model_free(model);
    return 0;
}
```
