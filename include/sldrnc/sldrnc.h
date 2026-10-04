/*
 * sldrnc.h -- SLD-RNC (Self-correcting Learned Dynamics for Real-time Neural Control): fast learned dynamics for
 * robots and machines.  Public C interface.
 *
 * Everything in the library is reached through this header.  It is plain C (usable from C, C++, Python via
 * ctypes, Rust, C#, ...), uses opaque handles, never throws, and reports errors as status codes plus a
 * human-readable message from sldrnc_last_error().
 *
 * THREE WAYS TO USE IT
 *   1. FULL        sldrnc_train_full()           train an SLD-RNC model from your robot's logs (most accurate)
 *   2. CORRECTION  sldrnc_model_add_correction() keep your own network, add SLD-RNC self-correction on top
 *   3. RAW         sldrnc_model_import_onnx() / sldrnc_model_from_layers()   run your own network as-is
 *
 * Every model, whatever its mode, is used the same way at run time:
 *   sldrnc_session_create() -> per tick: sldrnc_session_step() (predict), then sldrnc_session_observe()
 *   (feed the measured values, which drives self-correction) -> sldrnc_session_free().
 * Many robots at once (fleets, simulations, candidate motions): sldrnc_fleet_create() and sldrnc_fleet_step(), which
 * give exactly the numbers of separate sessions, faster.
 *
 * MEASUREMENT DELAY
 *   A measurement's delay is how many steps after its own tick it reaches the session: delay 1 = the measurement of
 *   tick t is observed right after step t and corrects step t+1 on (the normal control loop); delay D = it arrives
 *   right after step t+D-1 (late sensors, or a planner predicting D ticks ahead).  Self-correction is tuned for one
 *   delay (default 1; correction_delay_ticks / delay_ticks below) and works best when used at that delay.
 *
 * DATA LAYOUT
 *   All arrays are row-major double precision: one row per tick.  For FULL models a row is the raw signal vector
 *   described by the schema (channels concatenated in the order they were added).  For RAW / CORRECTION models a
 *   row is your network's own input vector.
 *
 * THREADING
 *   Models are read-only after creation and may be shared by any number of threads.  A session belongs to one
 *   thread at a time.  sldrnc_session_step() and sldrnc_session_observe() never allocate memory, lock, or block,
 *   so they are safe inside real-time control loops.
 */
#ifndef SLDRNC_H
#define SLDRNC_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(SLDRNC_BUILD)
#    define SLDRNC_API __declspec(dllexport)
#  else
#    define SLDRNC_API __declspec(dllimport)
#  endif
#else
#  define SLDRNC_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define SLDRNC_VERSION_MAJOR 1
#define SLDRNC_VERSION_MINOR 0
#define SLDRNC_VERSION_PATCH 0

/* ------------------------------------------------------------------------------------------------ status */
typedef enum sldrnc_status {
    SLDRNC_OK = 0,
    SLDRNC_ERROR_ARGUMENT = 1,     /* a pointer was NULL, a size did not match, a value was out of range */
    SLDRNC_ERROR_IO = 2,           /* a file could not be opened, read or written */
    SLDRNC_ERROR_FORMAT = 3,       /* a model file is damaged, from a newer version, or not an SLD-RNC / ONNX file */
    SLDRNC_ERROR_UNSUPPORTED = 4,  /* the model uses an operation SLD-RNC does not run (see the model formats doc) */
    SLDRNC_ERROR_STATE = 5,        /* the call does not apply to this model or session (e.g. correction on a FULL model) */
    SLDRNC_ERROR_DATA = 6,         /* training / tuning data is too short, constant, or contains NaN / infinity */
    SLDRNC_ERROR_INTERNAL = 7      /* a bug; please report it with the message */
} sldrnc_status;

/* Message for the last failing call on this thread ("" if none).  Valid until the next failing call. */
SLDRNC_API const char* sldrnc_last_error(void);
/* "1.0.0" */
SLDRNC_API const char* sldrnc_version(void);

/* ------------------------------------------------------------------------------------------------ enums */
typedef enum sldrnc_mode {
    SLDRNC_MODE_FULL = 0,          /* trained by SLD-RNC from your logs */
    SLDRNC_MODE_CORRECTION = 1,    /* your network + SLD-RNC self-correction */
    SLDRNC_MODE_RAW = 2            /* your network, run as-is */
} sldrnc_mode;

/* What a schema input channel measures.  The kind tells FULL training how to use the signal. */
typedef enum sldrnc_input_kind {
    SLDRNC_INPUT_POSITION = 0,     /* joint positions, used as they are (e.g. linear joints, in m) */
    SLDRNC_INPUT_VELOCITY = 1,     /* joint velocities (rad/s or m/s) */
    SLDRNC_INPUT_ACCELERATION = 2, /* measured joint accelerations, if you have them (otherwise they are derived) */
    SLDRNC_INPUT_GYRO = 3,         /* angular velocity from an IMU, 3 values */
    SLDRNC_INPUT_ACCEL = 4,        /* linear acceleration / specific force from an IMU, 3 values */
    SLDRNC_INPUT_QUATERNION = 5,   /* orientation quaternion (w, x, y, z), 4 values */
    SLDRNC_INPUT_GENERIC = 6,      /* anything else: commands, pressures, temperatures, ... */
    SLDRNC_INPUT_ANGLE = 7         /* rotary joint angles (rad): the model also sees their sine and cosine */
} sldrnc_input_kind;

/* Activations accepted in imported networks (RAW / CORRECTION). */
typedef enum sldrnc_activation {
    SLDRNC_ACT_IDENTITY = 0,
    SLDRNC_ACT_RELU = 1,
    SLDRNC_ACT_RELU2 = 2,          /* max(x, 0)^2 */
    SLDRNC_ACT_LEAKY_RELU = 3,     /* slope 0.01 for x < 0 */
    SLDRNC_ACT_TANH = 4,
    SLDRNC_ACT_SIGMOID = 5,
    SLDRNC_ACT_GELU = 6,           /* tanh approximation */
    SLDRNC_ACT_SILU = 7            /* x * sigmoid(x), a.k.a. swish */
} sldrnc_activation;

/* Model size for FULL training.  Measured on a 12-joint quadruped at 1 kHz (laptop core, one robot): */
typedef enum sldrnc_size {
    SLDRNC_SIZE_SMALL = 0,         /* fastest (~2 us per tick); almost as accurate as LARGE */
    SLDRNC_SIZE_MEDIUM = 1,        /* ~3 us per tick */
    SLDRNC_SIZE_LARGE = 2          /* ~6 us per tick; most accurate */
} sldrnc_size;

/* ------------------------------------------------------------------------------------------------ handles */
typedef struct sldrnc_schema_s sldrnc_schema;     /* describes one tick of raw signals (FULL mode) */
typedef struct sldrnc_model_s sldrnc_model;       /* a trained / imported model; read-only, shareable */
typedef struct sldrnc_session_s sldrnc_session;   /* run-time state for one robot / one stream */
typedef struct sldrnc_fleet_s sldrnc_fleet;       /* run-time state for many robots, stepped together */

/* ------------------------------------------------------------------------------------------------ schema */
/* rate_hz: how many ticks per second your data and control loop run at (e.g. 1000). */
SLDRNC_API sldrnc_status sldrnc_schema_create(double rate_hz, sldrnc_schema** out);
/* Append a channel of `size` values.  Channels form the raw input row in the order they are added. */
SLDRNC_API sldrnc_status sldrnc_schema_add_input(sldrnc_schema* schema, const char* name, sldrnc_input_kind kind, int size);
/* Declare the predicted outputs.  velocity_index (optional, may be NULL) gives, for each output, the column of the
   raw input row holding the velocity of the joint that output belongs to (-1 = none).  Self-correction uses it to
   follow friction; without it, correction still tracks offsets and gains. */
SLDRNC_API sldrnc_status sldrnc_schema_set_outputs(sldrnc_schema* schema, int n_outputs, const int* velocity_index);
/* Total width of the raw input row, and the column where channel `name` starts (-1 if absent). */
SLDRNC_API int sldrnc_schema_input_size(const sldrnc_schema* schema);
SLDRNC_API int sldrnc_schema_output_size(const sldrnc_schema* schema);   /* 0 until set_outputs */
SLDRNC_API int sldrnc_schema_column(const sldrnc_schema* schema, const char* name);
/* Structure options (both off by default):
   history 0: the outputs depend only on the current inputs (e.g. a motor's current controller).  Features are the
     current values only, there is no warm-up, and training rows may be unordered samples rather than recordings.
   affine (per channel): the outputs are affine (linear plus offset) in this channel's current values, with
     coefficients the model learns as functions of the other inputs.  E.g. joint torques are affine in joint
     accelerations; a current controller's voltages are affine in speed and the current references.  Not for
     QUATERNION channels; at least one channel must stay non-affine. */
SLDRNC_API sldrnc_status sldrnc_schema_set_history(sldrnc_schema* schema, int on);
SLDRNC_API sldrnc_status sldrnc_schema_set_affine(sldrnc_schema* schema, const char* name, int on);
SLDRNC_API void sldrnc_schema_free(sldrnc_schema* schema);

/* ------------------------------------------------------------------------------------------------ training */
typedef struct sldrnc_train_options {
    sldrnc_size size;              /* FULL: model size (default MEDIUM) */
    int steps;                     /* FULL: optimisation steps (default 12000) */
    int batch;                     /* FULL: ticks per step (default 512) */
    double learning_rate;          /* FULL: peak learning rate (default 5e-3) */
    double weight_decay;           /* FULL: regularisation (default 0.2) */
    uint64_t seed;                 /* random seed; the same seed and thread count give the same model */
    int threads;                   /* worker threads, 0 = all cores */
    int fit_correction;            /* 1 = also tune self-correction on the validation data (default 1) */
    int verbose;                   /* 1 = print progress to stdout */
    int correction_delay_ticks;    /* tune self-correction for measurements this late (default 1; see MEASUREMENT DELAY) */
} sldrnc_train_options;

/* Fill with the defaults listed above. */
SLDRNC_API void sldrnc_train_options_default(sldrnc_train_options* options);

typedef struct sldrnc_report {
    double train_nmse;             /* error on (a sample of) the training data, model alone */
    double val_nmse;               /* error on the validation data, model alone */
    double val_nmse_corrected;     /* error on the validation data with self-correction (NaN if not fitted) */
    double correction_time_ms;     /* the correction's time constant chosen on the validation data */
    double seconds;                /* wall time */
    int64_t macs_per_tick;         /* multiply-accumulates per prediction */
    int steps;                     /* optimisation steps run */
    int correction_delay_ticks;    /* the measurement delay self-correction was tuned for (0 if not fitted) */
} sldrnc_report;
/* Errors are normalised mean squared errors: mean over outputs of MSE / variance.  0 = perfect, 1 = no better than
   always predicting the average. */

/* MODE 1 -- FULL.  X_*: T x schema input size raw signals; Y_*: T x n_outputs measured targets (e.g. joint
   torques).  Each recording must be one continuous stretch at the schema rate.  Validation data picks the best
   checkpoint and tunes self-correction; use a stretch recorded after the training data.  The first 2 s of each
   recording are warm-up (the slow input features settle) and are not trained on or scored.  report may be NULL. */
SLDRNC_API sldrnc_status sldrnc_train_full(const sldrnc_schema* schema,
                                           const double* X_train, const double* Y_train, size_t ticks_train,
                                           const double* X_val, const double* Y_val, size_t ticks_val,
                                           const sldrnc_train_options* options, sldrnc_model** out, sldrnc_report* report);

/* MODE 1 -- FULL, from several recordings (e.g. one per log file): n_train training recordings, recording k being
   X_train[k] / Y_train[k] with ticks_train[k] rows, and likewise n_val validation recordings.  Each recording is one
   continuous stretch; its first 2 s are warm-up.  Never join separate recordings into one array: the jump between
   them would be learnt as if it were real motion.  Otherwise as sldrnc_train_full. */
SLDRNC_API sldrnc_status sldrnc_train_full_multi(const sldrnc_schema* schema,
                                                 size_t n_train, const double* const* X_train, const double* const* Y_train,
                                                 const size_t* ticks_train,
                                                 size_t n_val, const double* const* X_val, const double* const* Y_val,
                                                 const size_t* ticks_val,
                                                 const sldrnc_train_options* options, sldrnc_model** out, sldrnc_report* report);

/* MODE 3 -- RAW: describe your own fully connected network layer by layer.  W is in x out, row-major (input-major:
   W[i * out + o]); b has `out` values.  Optional (NULL = none): input_mean / input_std applied as (x - mean) / std
   before the first layer; output_mean / output_std applied as y * std + mean after the last. */
typedef struct sldrnc_layer {
    int in, out;
    sldrnc_activation activation;
    const float* W;
    const float* b;
} sldrnc_layer;
SLDRNC_API sldrnc_status sldrnc_model_from_layers(int n_layers, const sldrnc_layer* layers,
                                                  const float* input_mean, const float* input_std,
                                                  const float* output_mean, const float* output_std, sldrnc_model** out);
/* MODE 3 -- RAW: import an ONNX file (a feed-forward network: Gemm / MatMul / Add / Sub / Mul / Div / Relu /
   LeakyRelu / Tanh / Sigmoid / Pow(2) / Identity / Flatten, float32 or float64 weights). */
SLDRNC_API sldrnc_status sldrnc_model_import_onnx(const char* path, sldrnc_model** out);
SLDRNC_API sldrnc_status sldrnc_model_import_onnx_memory(const void* data, size_t bytes, sldrnc_model** out);

/* MODE 2 -- CORRECTION: add self-correction to a RAW model, tuned on validation data.  X: T x the model's input
   size (your network's own inputs); Y: T x outputs, measured.  velocity_index (optional): per output, a column of X
   holding that joint's velocity (-1 = none).  rate_hz: tick rate of the data.  delay_ticks: the measurement delay to
   tune for (1 = the normal case).  Turns the model into CORRECTION mode in place; call it before creating sessions
   from this model.  report may be NULL. */
SLDRNC_API sldrnc_status sldrnc_model_add_correction(sldrnc_model* model, const double* X, const double* Y, size_t ticks,
                                                     double rate_hz, const int* velocity_index, int delay_ticks,
                                                     sldrnc_report* report);

/* Re-tune the self-correction of a FULL or CORRECTION model for another measurement delay, on validation data (X, Y
   as for that model: raw signal rows for FULL, your network's inputs for CORRECTION; for FULL the first 2 s are
   warm-up and not scored).  Returns a NEW model in *out; the original is unchanged, so it may be in use.  E.g. keep
   one copy tuned for delay 1 for the control loop and one tuned for 50 for a planner.  report may be NULL. */
SLDRNC_API sldrnc_status sldrnc_model_tune_correction(const sldrnc_model* model, const double* X, const double* Y, size_t ticks,
                                                      int delay_ticks, sldrnc_model** out, sldrnc_report* report);

/* Compile a FULL model trained without history, whose non-affine inputs are 1 to 3 values, into a lookup table: the
   fastest form (about 5 x 10^8 outputs per second across a fleet on a 12-thread laptop).  grid: n_grid node counts, one per table
   dimension in input order (NULL: 64 per value, 128 per ANGLE value); ANGLE values wrap around, the others cover the
   training range and are clamped outside it.  Returns a NEW model in *out; the original is unchanged. */
SLDRNC_API sldrnc_status sldrnc_model_compile_table(const sldrnc_model* model, int n_grid, const int* grid, sldrnc_model** out);

/* ------------------------------------------------------------------------------------------------ models */
SLDRNC_API sldrnc_status sldrnc_model_load(const char* path, sldrnc_model** out);
SLDRNC_API sldrnc_status sldrnc_model_save(const sldrnc_model* model, const char* path);
SLDRNC_API void sldrnc_model_free(sldrnc_model* model);

typedef struct sldrnc_model_info {
    sldrnc_mode mode;
    int input_size;                /* width of one input row */
    int output_size;
    int has_correction;
    double rate_hz;                /* 0 for RAW models */
    double correction_time_ms;     /* 0 if no correction */
    int64_t macs_per_tick;
    int64_t parameters;
    double warmup_seconds;         /* FULL: a fresh session is fully accurate after this much data (2 s); else 0 */
    int correction_delay_ticks;    /* the measurement delay self-correction was tuned for; 0 if no correction */
    int history;                   /* FULL: 1 = uses the recent past (needs warm-up); 0 = current inputs only */
    int affine_inputs;             /* FULL: number of input values the outputs are affine in (0 = none) */
    int64_t table_bytes;           /* > 0: the model is a compiled table of this size */
} sldrnc_model_info;
SLDRNC_API sldrnc_status sldrnc_model_get_info(const sldrnc_model* model, sldrnc_model_info* info);

/* ------------------------------------------------------------------------------------------------ sessions */
typedef struct sldrnc_session_options {
    int use_correction;            /* 1 (default) = apply self-correction if the model has it */
    int protect;                   /* 1 (default) = ignore implausible measurements (e.g. a failed sensor) and
                                      bound the correction; 0 = trust every measurement */
    int max_delay_ticks;           /* the largest delay sldrnc_session_observe_delayed() will accept; memory is
                                      reserved for it at creation.  0 (default) = 256 or the model's tuned delay,
                                      whichever is larger */
} sldrnc_session_options;
SLDRNC_API void sldrnc_session_options_default(sldrnc_session_options* options);

SLDRNC_API sldrnc_status sldrnc_session_create(const sldrnc_model* model, const sldrnc_session_options* options /* NULL = defaults */,
                                               sldrnc_session** out);
/* Predict this tick's outputs from this tick's input row.  Real-time safe. */
SLDRNC_API sldrnc_status sldrnc_session_step(sldrnc_session* session, const double* x, double* y);
/* Feed the measured outputs for the tick just predicted (delay 1).  Self-correction uses them from the next step on.
   Skipping observe() for some ticks is fine: the correction holds.  NaN values are skipped.  Real-time safe. */
SLDRNC_API sldrnc_status sldrnc_session_observe(sldrnc_session* session, const double* y_measured);
/* Feed a late measurement: the measured outputs of the tick predicted delay_ticks - 1 steps before the latest step
   (delay_ticks = 1 is sldrnc_session_observe).  Feed late measurements in tick order.  delay_ticks may not exceed the
   session's max_delay_ticks; a measurement older than the session itself is ignored.  Real-time safe. */
SLDRNC_API sldrnc_status sldrnc_session_observe_delayed(sldrnc_session* session, const double* y_measured, int delay_ticks);
/* Forget all history and correction (e.g. after the robot was switched off). */
SLDRNC_API sldrnc_status sldrnc_session_reset(sldrnc_session* session);
SLDRNC_API void sldrnc_session_free(sldrnc_session* session);

/* ------------------------------------------------------------------------------------------------ fleets */
/* Many robots stepped together: robot r's input row is row r of X (robots x input size), its outputs row r of Y.
   Each robot keeps its own history and self-correction, exactly as a session would, and the outputs equal those of
   separate sessions bit for bit; the network runs for many robots at once on several threads, sharing every weight
   load.  With threads = 1 a step is real-time safe like a session step; with more threads it uses the library's
   thread pool (built for throughput, not for hard deadlines). */
typedef struct sldrnc_fleet_options {
    int use_correction;            /* default 1 */
    int protect;                   /* default 1 */
    int max_delay_ticks;           /* default 0: the model's tuned delay (memory is reserved per robot) */
    int threads;                   /* default 0: all cores */
} sldrnc_fleet_options;
SLDRNC_API void sldrnc_fleet_options_default(sldrnc_fleet_options* options);
SLDRNC_API sldrnc_status sldrnc_fleet_create(const sldrnc_model* model, size_t robots, const sldrnc_fleet_options* options /* NULL = defaults */,
                                             sldrnc_fleet** out);
SLDRNC_API sldrnc_status sldrnc_fleet_step(sldrnc_fleet* fleet, const double* X, double* Y);
/* measured outputs of every robot (robots x outputs), for the tick just predicted / delay_ticks late */
SLDRNC_API sldrnc_status sldrnc_fleet_observe(sldrnc_fleet* fleet, const double* Y_measured);
SLDRNC_API sldrnc_status sldrnc_fleet_observe_delayed(sldrnc_fleet* fleet, const double* Y_measured, int delay_ticks);
SLDRNC_API sldrnc_status sldrnc_fleet_reset(sldrnc_fleet* fleet);
SLDRNC_API size_t sldrnc_fleet_size(const sldrnc_fleet* fleet);
SLDRNC_API void sldrnc_fleet_free(sldrnc_fleet* fleet);

/* ------------------------------------------------------------------------------------------------ offline */
/* Run a recording through a fresh session: step every tick and, if Y is given, observe each measurement with delay
   `delay_ticks` (1 = the normal real-time case; larger values test late measurements or predicting further ahead).
   Y_pred (T x outputs, may be NULL) receives the predictions.  If Y is given, nmse (may be NULL) receives the error
   over ticks >= score_from, and nmse_per_output (may be NULL, `outputs` values) the error of each output. */
SLDRNC_API sldrnc_status sldrnc_model_run(const sldrnc_model* model, const sldrnc_session_options* options,
                                          const double* X, const double* Y, size_t ticks, int delay_ticks, size_t score_from,
                                          double* Y_pred, double* nmse, double* nmse_per_output);

#ifdef __cplusplus
}
#endif
#endif /* SLDRNC_H */
