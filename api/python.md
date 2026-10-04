# Python API reference

SLD-RNC: Self-correcting Learned Dynamics for Real-time Neural Control.

```python
import sldrnc
```

The Python package is a thin layer over the C library (through `ctypes`). It needs NumPy and nothing else. The
compiled library ships inside the package. To use a different build, set `SLDRNC_LIBRARY=/path/to/library` before the
import.

**Conventions**
- Data is NumPy arrays, one row per tick: **2-D, `ticks × columns`**. Any float dtype is accepted; arrays are
  converted to contiguous float64.
- Every failure raises `sldrnc.Error`, with a `code` and a readable message. Nothing fails silently.
- Errors are normalised mean squared errors: 0 = perfect, 1 = no better than always predicting the average.
- Long calls (`train`, `add_correction`, `tune_correction`, `run`) release the GIL, so other Python threads keep running.

Contents: [module](#module) · [Schema](#schema) · [train](#train) · [Model](#model) · [Session](#session) ·
[Fleet](#fleet) · [tables](#tables) · [result types](#result-types) · [enums](#enums) · [Error](#error) · [performance notes](#performance-notes)

---

## Module

| | |
|---|---|
| `sldrnc.version() -> str` | library version, e.g. `"1.0.0"` (also `sldrnc.__version__`) |
| `sldrnc.train(...)` | FULL mode: [train a model](#train) |
| `sldrnc.Schema`, `sldrnc.Model`, `sldrnc.Session` | the three main objects |
| `sldrnc.Fleet` | [many robots stepped together](#fleet) |
| `sldrnc.Dense`, `sldrnc.Report`, `sldrnc.Info`, `sldrnc.RunResult` | [helper and result types](#result-types) |
| `sldrnc.Mode`, `sldrnc.Input`, `sldrnc.Activation`, `sldrnc.Size` | [enums](#enums) |
| `sldrnc.Error` | [the exception](#error) |

---

## Schema

```python
schema = sldrnc.Schema(rate_hz=1000)
```

Describes one tick of raw signals for FULL training: which channels the input row holds, in which order, and which
outputs are predicted. Not needed for RAW / CORRECTION models.

| member | |
|---|---|
| `Schema(rate_hz, history=True)` | `rate_hz`: ticks per second of your logs and your control loop (> 0). `history`: see below |
| `add(name, kind, size) -> Schema` | append a channel of `size` values. `kind` is an [`Input`](#enums). Names must be unique. Returns the schema, so calls chain |
| `outputs(n, velocity=None, velocity_index=None) -> Schema` | declare `n` predicted outputs (see below) |
| `affine(*names) -> Schema` | the outputs are affine in these channels' current values (see below) |
| `column(name) -> int` | first column of channel `name` in the input row, `-1` if there is none |
| `width` | number of columns of the input row (sum of channel sizes) |
| `output_count` | number of outputs (0 until `outputs()` is called) |
| `rate_hz` | the rate given at construction |

**Input row.** Channels are concatenated in the order they were added. With
`add("q", POSITION, 12).add("qd", VELOCITY, 12)`, columns 0–11 are `q` and 12–23 are `qd`.

**Outputs and velocities.** Self-correction follows friction better when it knows which joint velocity belongs to
each output. Either:
- `outputs(12, velocity="qd")`: output `j` belongs to the joint whose velocity is element `j` of channel `"qd"`;
- `outputs(3, velocity_index=[12, 13, -1])`: one input column per output, `-1` = none.

Without either, correction still tracks offsets and gains. A `QUATERNION` channel must have exactly 4 values
(w, x, y, z).

**`history=False`: memoryless maps.** By default a FULL model uses the recent past of every signal, which is what
makes it accurate on real machines (friction, hydraulics, sensor lag). Some targets depend only on the current
inputs, such as a motor drive's current controller. For those, `history=False` gives a model that sees the current
values only: no warm-up (`info.warmup_seconds` is 0), any tick can be predicted on its own, and the training rows may
be unordered samples rather than recordings. Only a model without history can be compiled into a [table](#tables).

**`affine(...)`: outputs linear in some inputs.** Many machines are linear in some of their inputs, with
coefficients that depend on the others: joint torques are linear in the joint accelerations (the coefficients depend
on the arm's pose), and a current controller's voltages are linear in the speed and the current references (the
coefficients depend on the currents and the rotor angle). Declaring such channels affine builds that structure into
the model. It usually makes the model more accurate away from the training data, and for tables it keeps those
channels out of the grid. Rules: a `QUATERNION` channel can't be affine, and at least one channel must stay
non-affine. With history on, the affine channels' past values still reach the rest of the model.

```python
schema = sldrnc.Schema(rate_hz=20000, history=False)          # a 20 kHz motor drive
schema.add("i_dq", sldrnc.Input.GENERIC, 2).add("angle", sldrnc.Input.ANGLE, 1)
schema.add("speed", sldrnc.Input.VELOCITY, 1).add("i_dq_ref", sldrnc.Input.GENERIC, 2)
schema.outputs(2).affine("speed", "i_dq_ref")                  # the two voltages
```

---

## train

```python
model, report = sldrnc.train(schema, X_train, Y_train, X_val, Y_val, size="small")
```

**FULL mode.** Trains an SLD-RNC model on your logs, picks the best checkpoint on the validation data, and tunes
self-correction on it.

| parameter | default | |
|---|---|---|
| `schema` | — | a `Schema` with outputs declared |
| `X_train`, `Y_train` | — | `ticks × schema.width` raw signals and `ticks × schema.output_count` measured outputs; or **lists** of such arrays, one per recording |
| `X_val`, `Y_val` | — | the same for validation: recorded **after** the training data |
| `size` | `"medium"` | `"small"`, `"medium"`, `"large"` or a [`Size`](#enums) |
| `steps` | 12000 | optimisation steps |
| `batch` | 512 | ticks per step |
| `learning_rate` | 5e-3 | peak learning rate |
| `weight_decay` | 0.2 | regularisation |
| `seed` | 0 | the same seed and thread count give the same model, bit for bit |
| `threads` | 0 | worker threads; 0 = all cores |
| `fit_correction` | `True` | tune self-correction on the validation data |
| `verbose` | `False` | print progress |
| `correction_delay_ticks` | 1 | tune self-correction for measurements this many ticks late (see [Measurement delay](concepts.md#measurement-delay)) |

Returns `(Model, Report)`.

**Several recordings.** Logs usually come as several files. Pass them as lists, element *k* of `X_train` going with
element *k* of `Y_train`:

```python
model, report = sldrnc.train(schema, [X_a, X_b, X_c], [Y_a, Y_b, Y_c], [X_v], [Y_v], size="small")
```

Never join separate recordings into one array: the jump between them would be learnt as if it were real motion.

**Data requirements.** Each recording is one continuous stretch at the schema rate, with no gaps. The first 2 s of
each recording are warm-up: they feed the model's input processing but are not trained on or scored, so recordings
shorter than 2 s add nothing. Training needs at least 1000 ticks after the warm-ups, and validation at least 200.
NaN or infinity anywhere raises `Error` with code `DATA`. The logs must cover the conditions the model will meet
(speeds, loads, ranges of motion): a FULL model is only reliable inside them.

**Without history.** With `Schema(..., history=False)`, each row is one independent sample: the arrays may be
single arrays or lists, there is no warm-up, and rows need not be in time order.

**Time.** On a laptop, about 4 minutes of 1 kHz data from a 12-joint robot trains in 15 s (small), about 30 s
(medium) and about 2 minutes (large).

---

## Model

A trained (FULL) or imported (RAW / CORRECTION) model. Read-only once built, except for `add_correction`. Any number
of threads and sessions can share one model.

### Creating a model

| | mode | |
|---|---|---|
| `sldrnc.train(...)` | FULL | [above](#train) |
| `Model.from_onnx(path_or_bytes)` | RAW | import an ONNX file, given as a path or as the file's bytes ([what is supported](model_formats.md)) |
| `Model.from_torch(module, input_size)` | RAW | a PyTorch feed-forward module, via an in-memory ONNX export. Needs `torch` |
| `Model.from_layers(layers, input_mean=None, input_std=None, output_mean=None, output_std=None)` | RAW | a list of `Dense` layers (see below) |
| `Model.load(path)` | any | a saved `.sldm` file |

**`from_layers`.** Each `sldrnc.Dense(W, b=None, activation="identity")` is one fully connected layer: `W` has shape
`(in, out)` (so `y = x @ W + b`), `b` has `out` values, and `activation` is an [`Activation`](#enums) or its name
(`"relu"`, `"tanh"`, ...). Optional normalisation: `x → (x − input_mean) / input_std` before the first layer and
`y → y · output_std + output_mean` after the last.

```python
mine = sldrnc.Model.from_layers([
    sldrnc.Dense(W1, b1, "relu"),         # W1: (in, hidden)
    sldrnc.Dense(W2, b2),                 # W2: (hidden, out), identity
], input_mean=mu, input_std=sd)
```

### Self-correction

| method | |
|---|---|
| `add_correction(X, Y, rate_hz, velocity_index=None, delay_ticks=1) -> Report` | **CORRECTION mode.** Adds self-correction to this RAW model, tuned on validation data, in place. Call it before creating sessions |
| `tune_correction(X, Y, delay_ticks) -> (Model, Report)` | returns a **copy** of this FULL / CORRECTION model with self-correction re-tuned for another measurement delay. This model is unchanged |

- `X`: validation input rows, in the model's own input format (raw signals for FULL, your network's inputs for
  RAW / CORRECTION). `Y`: the measured outputs, `ticks × output_size`.
- `rate_hz`: tick rate of the data.
- `velocity_index`: per output, the column of `X` holding that joint's velocity (`-1` = none). Optional.
- `delay_ticks`: the measurement delay to tune for (1 = the normal case).
- At least 200 ticks (FULL: 2 s + 200), and at least 4 × `delay_ticks`.

```python
corrected = sldrnc.Model.from_onnx("my_net.onnx")
corrected.add_correction(X_val, Y_val, rate_hz=1000, velocity_index=list(range(12, 24)))

planner, rep = model.tune_correction(X_val, Y_val, delay_ticks=50)   # a copy for 50-ms-ahead predictions
```

### Running

| method | |
|---|---|
| `session(correction=True, protect=True, max_delay_ticks=0) -> Session` | run-time state for one robot (see [Session](#session)) |
| `fleet(robots, correction=True, protect=True, max_delay_ticks=0, threads=0) -> Fleet` | run-time state for many robots, stepped together (see [Fleet](#fleet)) |
| `compile_table(grid=None) -> Model` | a copy compiled into a lookup table (see [tables](#tables)) |
| `run(X, Y=None, delay_ticks=1, score_from=0, correction=True, protect=True) -> RunResult` | replay a whole recording through a fresh session |

`session` parameters:
- `correction`: apply self-correction (if the model has it).
- `protect`: ignore implausible measurements, such as a failed sensor, and bound the correction ([details](concepts.md#protection)).
- `max_delay_ticks`: the largest delay `observe()` will accept. 0 = 256 or the model's tuned delay, whichever is
  larger.

`run` steps every row of `X`. If `Y` is given, it also observes each row of `Y` with delay `delay_ticks`, and scores
the predictions of rows `>= score_from`. For FULL models, use `score_from` of at least 2 s of ticks (the warm-up).

### Saving and inspecting

| | |
|---|---|
| `save(path)` | write a `.sldm` file (any mode) |
| `info -> Info` | what the model is ([fields](#result-types)) |

---

## Session

Run-time state for one robot. Create one per robot, or per independent stream. **Not thread-safe**: use a session
from one thread at a time. A session keeps its model alive.

| method | |
|---|---|
| `step(x) -> ndarray` | predict this tick's outputs from this tick's input row (`input_size` values). Returns a new array |
| `step_into(x, out)` | the same, allocation-free: `x` and `out` must be contiguous float64 arrays of the right sizes |
| `observe(y_measured, delay_ticks=1)` | feed measured outputs (see below) |
| `reset()` | forget all history and correction, keeping the options |

**`observe`.** With `delay_ticks=1`, `y_measured` is the measurement of the tick just predicted; it corrects the
next `step` on. A late measurement, of the tick predicted `D − 1` steps before the latest, takes `delay_ticks=D`.
Feed late measurements in tick order. Skipping `observe` is fine (the correction holds), and NaN values are skipped.

```python
robot = model.session()
out = np.empty(model.info.output_size)
while running:
    robot.step_into(read_sensors(), out)   # feed-forward for this tick
    apply(out)
    robot.observe(read_torques())          # measured values for the same tick
```

---

## Fleet

```python
fleet = model.fleet(4096)              # 4,096 robots
U = fleet.step(X)                      # X: 4096 x input_size  ->  U: 4096 x output_size
fleet.observe(Y_measured)              # 4096 x output_size
```

Many robots of the same model stepped in one call: a fleet, a simulation with many copies of the machine, or a
planner trying many candidate motions at once. Row `r` of every array belongs to robot `r`. Each robot keeps its own
history and self-correction, exactly like its own session, and **the outputs equal separate sessions bit for bit**.
Robots are computed in blocks that share every weight load, spread over all cores. How much that buys depends on
where the time goes (12 laptop threads, against the same model in independent sessions): **1.3–1.8× more robot-ticks
per second for medium and large networks**, about the same for small models whose time goes mostly into their own
input processing and self-correction, and **about 5 × 10⁸ outputs per second for a table**. Very large fleets of
models with history are limited by memory: each robot keeps a few kilobytes of recent history.

| member | |
|---|---|
| `Model.fleet(robots, correction=True, protect=True, max_delay_ticks=0, threads=0)` | create; the options are those of [`session`](#running). `threads`: worker threads, 0 = all cores, 1 = the calling thread only |
| `step(X) -> ndarray` | predict every robot's outputs for this tick. `X`: `robots × input_size`. Returns `robots × output_size` |
| `step_into(X, out)` | the same, allocation-free: contiguous float64 arrays of the right shapes |
| `observe(Y_measured, delay_ticks=1)` | every robot's measured outputs (`robots × output_size`), as in [`Session.observe`](#session). NaN values are skipped, so a robot without a measurement this tick can pass NaN |
| `reset()` | forget every robot's history and correction |
| `robots` | number of robots |

A fleet is used from one thread at a time; it runs its own worker threads inside `step` and `observe`. Its memory
is a few hundred bytes to a few kilobytes per robot, plus scratch per worker thread.

---

## Tables

```python
table = model.compile_table()          # or compile_table([32, 32, 64])
table.save("drive_table.sldm")
```

A FULL model **without history** whose non-affine channels hold **1 to 3 values** in total can be compiled into a
lookup table: the network is evaluated once on a grid over the training range, and each prediction then blends four
grid nodes instead of running the network. This is the fastest form SLD-RNC has: a few dozen operations per
prediction, whatever the network's size. The table is an ordinary `Model`: save it, load it, run it in sessions or
fleets, with or without self-correction.

- `grid`: nodes per table dimension, in input order (one entry per non-affine value). Default: 64 per value, 128 per
  `ANGLE` value. More nodes = closer to the network and more memory (`info.table_bytes`: 16.8 MB for the motor
  drive's default 64 × 64 × 128 table).
- `ANGLE` values wrap around (the table is periodic in them); other values are clamped to the range seen in
  training. The affine channels are not in the table: they enter exactly, as in the network.
- The table's error is the network's plus the interpolation error. Check it on your data: for the motor drive, the
  default grid added 0.5–1.4 % to the network's error; a coarser 32 × 32 × 64 grid added 4–10 %, mostly from too
  few angle nodes.
- Raises `Error.STATE` for a model with history, a RAW / CORRECTION model or more than 3 non-affine values, and
  `Error.ARGUMENT` for a bad grid or a table over 2 GB.

---

## Result types

**`Report`**: returned by `train`, `add_correction` and `tune_correction`.

| field | |
|---|---|
| `train_error` | model alone, on a sample of the training data (NaN for `add_correction` / `tune_correction`) |
| `validation_error` | model alone, on the validation / tuning data |
| `validation_error_corrected` | with self-correction, at the tuned delay (NaN if not fitted) |
| `correction_time_ms` | the correction's time constant chosen on the data |
| `correction_delay_ticks` | the measurement delay it was tuned for (0 if not fitted) |
| `seconds` | wall time |
| `macs_per_tick` | multiply-accumulates per prediction |
| `steps` | optimisation steps run (0 for correction tuning) |

**`Info`**: `model.info`.

| field | |
|---|---|
| `mode` | `Mode.FULL`, `Mode.CORRECTION` or `Mode.RAW` |
| `input_size`, `output_size` | width of an input row and number of outputs |
| `has_correction` | the model has self-correction |
| `rate_hz` | tick rate (0 for RAW) |
| `correction_time_ms`, `correction_delay_ticks` | the correction's time constant and tuned delay (0 if none) |
| `macs_per_tick`, `parameters` | cost per prediction and network size |
| `warmup_seconds` | FULL: a fresh session is fully accurate after this much data (2 s; 0 without history); 0 otherwise |
| `history` | FULL: the model uses the recent past (`False`: current inputs only) |
| `affine_inputs` | FULL: number of input values the outputs are affine in (0 = none) |
| `table_bytes` | > 0: the model is a compiled table of this size |

**`RunResult`**: returned by `run`. `predictions` (`ticks × outputs`), `error` (NaN without `Y`) and
`error_per_output`.

---

## Enums

| enum | values |
|---|---|
| `Mode` | `FULL`, `CORRECTION`, `RAW` |
| `Input` | `ANGLE` (rotary joint angles in rad; the model also sees their sine and cosine), `POSITION` (positions used as they are, e.g. linear joints in m), `VELOCITY` (joint velocities), `ACCELERATION` (measured joint accelerations, if you have them), `GYRO` (IMU angular velocity, 3), `ACCEL` (IMU linear acceleration, 3), `QUATERNION` (orientation w, x, y, z), `GENERIC` (anything else: commands, pressures, temperatures) |
| `Activation` | `IDENTITY`, `RELU`, `RELU2` (squared ReLU), `LEAKY_RELU` (slope 0.01), `TANH`, `SIGMOID`, `GELU` (tanh form), `SILU` |
| `Size` | `SMALL` (fastest), `MEDIUM`, `LARGE` (most accurate) |

---

## Error

```python
try:
    model = sldrnc.Model.load("robot.sldm")
except sldrnc.Error as e:
    print(e.code, e)
```

| `e.code` | meaning |
|---|---|
| `Error.ARGUMENT` | a shape, size or value is wrong (the message says which) |
| `Error.IO` | a file can't be read or written |
| `Error.FORMAT` | not a model file, damaged, or from a newer library version |
| `Error.UNSUPPORTED` | the ONNX graph uses something SLD-RNC doesn't run; the message names it |
| `Error.STATE` | the call doesn't apply (e.g. `add_correction` on a FULL model, `tune_correction` without correction, `compile_table` on a model with history) |
| `Error.DATA` | data too short, constant, or containing NaN / infinity |
| `Error.INTERNAL` | a bug: please report the message |

If the compiled library can't be found, the import raises `OSError` and lists the places it looked.

---

## Performance notes

- One `step` + `observe` from Python costs about 10 µs on the test laptop, almost all of it Python overhead (the
  library itself takes about 2 µs). That's fine for logging, analysis and soft real time. For a hard real-time
  loop, call the library from C or C++.
- `step_into` avoids an allocation per tick.
- `run()` replays a whole recording in C, at full library speed.
- For many robots, one `Fleet.step_into` call replaces thousands of `step` calls: the Python overhead is paid once
  per tick, not once per robot. The quick-start example's 1,000-robot fleet runs 2–3.5 million robot-ticks per
  second from Python, against about 0.1 million for a loop over sessions.
