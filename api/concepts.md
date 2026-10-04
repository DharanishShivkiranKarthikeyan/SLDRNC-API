# Concepts

SLD-RNC stands for **Self-correcting Learned Dynamics for Real-time Neural Control**: a learned model of a machine's dynamics that corrects
itself from the machine's own measurements, and runs inside the control loop.

## What SLD-RNC predicts

A **model** maps what a machine senses at each control tick (joint positions, velocities, IMU readings, ...) to the
values you want to predict, usually the efforts the joints need (torques or forces). It runs one tick at a time, in a
**session**, so it can be used inside a control loop: for feed-forward torques, for detecting collisions (measured
minus predicted), or for simulating many robots and candidate motions.

## The three modes

**FULL: SLD-RNC trains the model.** You provide logs: recordings of raw signals `X` and the measured outputs `Y` at a
fixed rate, plus later recordings for validation. SLD-RNC builds its own model, picks the best checkpoint on the
validation data, and tunes self-correction there too. Training takes seconds to minutes on a laptop CPU.

**CORRECTION: your model plus SLD-RNC self-correction.** You keep a network you already have and give SLD-RNC some
validation data. SLD-RNC adds an online correction that learns, from the measured outputs, how your model is wrong
right now.

**RAW: your model, as-is.** SLD-RNC runs your network with no dependencies, at microsecond latency. There's no
accuracy change; this mode is for portability and speed.

What each mode bought on the HyQ quadruped (test error, lower is better): see the table in the
[API readme](readme.md#3-three-ways-to-use-it).

## Choosing a mode

| your situation | mode | what I measured |
|---|---|---|
| logs of a robot or machine with joint or force sensing, at robot rates (hundreds of Hz to a few kHz) | **FULL** | quadruped: the most accurate and the fastest option (nMSE 0.020–0.021 with self-correction, against 0.118–0.119 for a standard network). Robot arm, with joint angles as `ANGLE` and torques affine in the accelerations: at least as accurate as a standard network 6× its size (0.0135–0.0139 against 0.0134–0.0165), and 0.0085–0.0087 with self-correction |
| a memoryless map at a high rate, such as a 20 kHz motor current controller, with samples covering the operating range | **FULL** with `history=False` and the right `affine` channels, then `compile_table()` | motor drive: 1.32 V rms (large model) against 1.71–1.75 V for a standard 512-wide network at a quarter of its computation; as a table, 1.34 V at about 0.1 µs per prediction |
| a network you trust, and measurements of what it predicts | **CORRECTION** | robot arm: a large network plus self-correction reached 0.0088–0.0101; quadruped: about 3–4× better than the network alone |
| a network you must keep as it is, or no measurements | **RAW** | runs it unchanged (same error), with no dependencies |

FULL learns from your data, so it only knows the conditions the data contains. On the motor drive, FULL models
trained only on simulated runs at low speeds failed completely on recorded runs at high speeds. Trained on samples
that also cover the whole operating range (currents, speeds, references), the memoryless FULL model above beat the
standard network on the recorded runs. Inside the conditions of the simulated runs the 512-wide network stayed
slightly ahead (1.54–1.55 V against 1.61–1.63 V).

## Schemas (FULL mode)

A schema describes one tick of raw signals: a list of named **channels**, each with a **kind** and a size, plus the
number of outputs and the tick rate.

| kind | what to put there |
|---|---|
| `ANGLE` | rotary joint angles in radians: the model also sees their sine and cosine |
| `POSITION` | positions used as they are: linear joints (m), or angles you don't want treated as angles |
| `VELOCITY` | joint velocities |
| `ACCELERATION` | measured joint accelerations, only if you have them (SLD-RNC derives them otherwise) |
| `GYRO` | IMU angular velocity (3) |
| `ACCEL` | IMU linear acceleration / specific force (3) |
| `QUATERNION` | orientation, (w, x, y, z) |
| `GENERIC` | anything else: commands, pressures, temperatures, motor currents, ... |

The input row is the channels concatenated in the order you added them. If you added `q` (12) then `qd` (12), columns
0–11 are `q` and 12–23 are `qd`; `schema.column("qd")` returns 12.

The **kind matters**: SLD-RNC treats each kind differently when it builds its model. Declare signals truthfully
rather than as `GENERIC` where a better kind applies.

**`ANGLE` or `POSITION` for rotary joints?** Use `ANGLE` where gravity acting on the links shapes the torques, as in
robot arms. On the SARCOS arm it lowered the small model's error by 7 %, the medium's by 2 % and the large's by less
than 1 %, for 0.1–0.3 µs more per tick. On the quadruped it made no consistent difference (−7 % to +7 % across
sizes, about the spread between training runs), so `POSITION` is fine there.

`outputs(n, velocity="qd")` says there are `n` outputs and that output `j` belongs to the joint whose velocity is
element `j` of channel `qd`. Self-correction uses this to follow friction changes. If your outputs aren't joint
efforts, leave it out.

**Data requirements**

- Each recording is one continuous stretch at the schema rate (no gaps). Several recordings (one per log file) are
  passed as lists; never join them into one array, as that creates false jumps.
- The logs must cover the conditions the model will meet: speeds, loads, ranges of motion.
- Validation data should be recorded after the training data, as it will be used later in deployment.
- No NaN or infinity in training data (the call fails with `DATA` and the position of the first bad value).
- As a guide, a few minutes of varied motion at 1 kHz was enough for a quadruped.
- Each recording needs more than 2 s: the first 2 s are warm-up.

## Sessions and real-time use

Create one **session** per robot (or per independent stream). Each tick:

1. `step(x)`: predict this tick's outputs from this tick's input row.
2. `observe(y_measured)`: when the measured outputs for that tick arrive, feed them in. Self-correction uses them from
   the next `step` on.

`step` and `observe` never allocate memory, lock, or block, so they are safe in a real-time thread. Models are
read-only and can be shared by many sessions and threads; a session belongs to one thread at a time.

If a measurement is missing, skip `observe` (the correction holds), or pass NaN for the missing values.

**Warm-up.** FULL models also use slow summaries of the recent past, which need about 2 s of data to settle
(`model.info.warmup_seconds`). Predictions in the first 2 s of a fresh session are less accurate. Start the session
before you need it, or feed it the last 2 s of sensor data. When scoring recordings with `run()`, use
`score_from` of at least 2 s. Training handles this automatically: the first 2 s of each recording are never
trained on or scored. Models without history (`Schema(..., history=False)`) need no warm-up.

## Many robots: fleets

When one program runs many robots of the same model (a fleet, a simulation with thousands of copies, a planner
trying many candidate motions), use a **fleet** instead of many sessions: `model.fleet(N)` holds N robots, and one
`step` call predicts all of them from an `N × input_size` array. Each robot keeps its own history and
self-correction, and its outputs are identical, bit for bit, to what its own session would give. The robots are
computed in blocks that share every load of the model's weights, spread over all cores, with work handed out so
faster cores take more.

What a fleet buys depends on where the time goes. Measured on a 12-thread laptop against the same model in
independent sessions:

| model | sessions | fleet | |
|---|---|---|---|
| motor drive, large network (69k MACs) | 1.5 M robot-ticks/s | 2.7 M | 1.8× |
| a 512-wide network (267k MACs) | 0.33 M | 0.55 M | 1.65× |
| robot arm, medium, with self-correction | 1.6 M | 2.05 M | 1.27× |
| quadruped, small, with self-correction (1,024 robots) | 2.4–2.5 M | 2.4–2.7 M | ≈ 1× |
| motor drive table | — | 2.6–3.2 × 10⁸ (5–6 × 10⁸ outputs/s) | |

Small models spend most of their time on each robot's own input processing and self-correction, which can't be
shared, so a fleet runs them at about session speed (it still saves the per-call overhead of thousands of `step`
calls, which matters most from Python). Each robot of a model with history keeps a few kilobytes of state, so very
large fleets of such models are limited by memory: the quadruped ran at 1.6 M robot-ticks/s with 4,096 robots.

- Robots that don't move this tick can still be stepped (their history then includes that tick), or keep them in
  their own fleet.
- A robot with no measurement this tick takes NaN in its row of `observe`; its correction holds.
- `threads=1` keeps a fleet on the calling thread, real-time safe like a session; more threads use the library's
  worker pool and are meant for throughput.

## Memoryless targets, affine inputs and tables

Two schema options describe the structure of what you predict. They apply to FULL models.

**History off.** By default a FULL model uses the recent past of every signal: real machines have friction,
hydraulics and sensor lag, and the past is what reveals them. Some targets have no such memory: a motor drive's
current controller computes its voltages from the present currents, angle, speed and references alone.
`Schema(rate_hz, history=False)` builds a model of the current inputs only. It needs no warm-up, and training rows
may be unordered samples, for example a grid or random draws over the operating range rather than recordings.

**Affine inputs.** Many targets are linear in some inputs, with coefficients that depend on the others. Joint
torques are linear in joint accelerations, with an inertia that depends on the pose; a current controller's voltages
are linear in speed and in the current references, with coefficients that depend on the currents and rotor angle.
`schema.affine("qdd")` builds this into the model: the network learns the coefficients from the other inputs, and the
affine inputs enter exactly. This tends to help most away from the training data, where an unstructured network
has to guess how the output scales.

**Tables.** A model with history off whose remaining (non-affine) inputs are at most 3 values can be compiled into a
**lookup table** (`model.compile_table()`): the network is evaluated once on a grid over the training range, and each
prediction then blends four grid points. Its cost no longer depends on the network's size: it is the fastest form
SLD-RNC has. It adds a small interpolation error; check the table's error on your data before switching to it.

Together these give the motor drive's best model: history off; the network sees the two currents and the rotor
angle; the voltages are affine in speed and the two current references; and it compiles into a 64 × 64 × 128 table
(the default grid).

## Self-correction

Models are never perfect: friction changes as actuators warm up, a payload is added, or the robot stands still in a
way it rarely did in training. If the machine measures what the model predicts (joint torque or force sensors, motor
currents), self-correction uses those measurements to remove most of the current error within a fraction of a second.

How much it helps depends on how old the measurements are when the prediction is made:

- **Next tick (the normal control loop).** The correction knows the error up to the previous tick, and it removes
  most of it.
- **Further ahead** (for example 50 ms for a motion planner), or **late measurements**. The correction helps less,
  because the error changes in the meantime.

## Measurement delay

A measurement's **delay** is how many steps after its own tick it reaches the session. Delay 1 means the
measurement of tick *t* is observed right after step *t* and corrects step *t + 1* on: the normal control loop. Delay
50 at 1 kHz means it arrives 50 ms later, which is also the situation of a planner predicting 50 ms ahead.

**Tune for the delay you will run at.** Self-correction is tuned for one delay:

| | how |
|---|---|
| FULL, while training | `correction_delay_ticks=50` (Python `train`, C++ `TrainOptions`, C `sldrnc_train_options`) |
| CORRECTION | `add_correction(..., delay_ticks=50)` |
| an existing model | `tune_correction(X_val, Y_val, delay_ticks=50)` returns a re-tuned **copy**, in seconds |

**Feed late measurements as late measurements.** `observe(y, delay_ticks=D)` pairs `y` with the tick it belongs to:
the one predicted `D − 1` steps before the latest. A session keeps the history needed for delays up to
`max_delay_ticks` (default: 256, or the model's tuned delay if larger). `run(X, Y, delay_ticks=D)` replays a
recording the same way, so it shows what to expect.

**Keep one copy per delay.** A copy tuned for 50 ms was 51–163 % worse than the next-tick tuning when used in the
normal next-tick loop. Use the next-tick model for control and a 50 ms copy for the planner; both can run side by
side from the same training.

**What tuning for a delay buys** (quadruped, predictions 50 ms ahead, FULL models; error with the next-tick tuning →
with the 50 ms tuning):

| test stretch | error | change |
|---|---|---|
| later in the same recording (trotting) | 0.043–0.049 → 0.037–0.041 | 14–20 % better |
| a recording never seen, slower gait | 0.031–0.037 → 0.026–0.031 | 15–19 % better |
| a recording never seen, faster gait | 0.043–0.046 → 0.040–0.048 | 8 % better to 7 % worse |
| mostly standing still, after trotting | 0.018–0.023 → 0.026–0.032 | 36–57 % worse |

Tuned for a delay, the correction becomes slower (time constants of 1–5 s instead of 0.1 s): it averages out more
noise, but it follows a change of conditions more slowly. So tune on validation data that contains the conditions
you expect, including changes such as stopping and starting, and check the result with `run(..., delay_ticks=D)` on
data that wasn't used for tuning. At delays up to 20 ms, tuning for the delay made no reliable difference on the
quadruped (at 10 ms it was 6–7 % worse); at 50–200 ms it was 14–26 % better on the trotting stretch.

## Protection

With `protect=True` (the default), a session distrusts measurements that look like a failed sensor:

- A measurement that repeats exactly for 20 ms is treated as a frozen sensor and ignored until it changes. Real
  measurements essentially never repeat exactly; the quadruped's longest run was 5 ticks in 12 minutes of data.
- A measurement implausibly far from what the corrected model expects (a spike) is skipped, but for at most 20 ms
  in a row. A deviation that persists is accepted as real, because a stopped robot or a new payload really does
  change the loads.
- The correction is bounded to ±5 standard deviations of each output.

Measured on the quadruped, with one force sensor stuck at zero for 2 s (error over the fault and the 5 s after):

| | error |
|---|---|
| no self-correction | 0.121 |
| self-correction, unprotected (chases the broken sensor) | 0.135 |
| **self-correction, protected** | **0.066** |

On clean data, protection costs about 1 % (0.0400 vs 0.0396), and about 6 % on a recording the model never saw.
Turn it off (`protect=False`) only if your measurements are always valid.

## Reading the errors

Reports and `run()` give a **normalised mean squared error** per output: MSE divided by the variance of that output
over the scored data, averaged over outputs.

- 0 means perfect; 1 means no better than always predicting the average.
- 0.02 means the remaining error has 2 % of the signal's variance, roughly 14 % of its standard deviation.

`run(...).error_per_output` shows which outputs are hard.

## Sizes (FULL mode)

`SMALL`, `MEDIUM` and `LARGE` trade speed for accuracy. On the quadruped, trained on one recording, all three were
within 10 % of each other (with self-correction: 0.020 / 0.020 / 0.021); trained on the other recording, `MEDIUM` and
`LARGE` were about 15 % better than `SMALL` (0.016 vs 0.019). `SMALL` is 3.6× faster than `LARGE` (1.7 vs 6.1 µs per
tick) and trains in 15 s instead of 130 s. Start with `SMALL`; try `MEDIUM` when accuracy matters more than speed.
