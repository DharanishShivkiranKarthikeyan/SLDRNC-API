**SLD-RNC: Self-correcting Learned Dynamics for Real-time Neural Control**

Fast learned dynamics for legged robots, robot arms and motor drives

**Dharanish Karthikeyan** · 4 October 2026

Every number here was measured, except where it says *estimate*.

---

## In short

SLD-RNC, short for **Self-correcting Learned Dynamics for Real-time Neural Control**, predicts what a machine needs
(joint forces, joint torques, motor voltages) from the machine's own sensors, fast enough to run inside the control
loop. It ships as a library for Windows, Linux and macOS, with C, C++ and Python interfaces.

- **Quadruped robot (real hydraulic robot, 12 joints, 1 kHz):** **5.8× lower error** than a standard neural network,
  and **8.2× lower** on a recording it never saw, made at a different walking speed. Most of that comes from
  correcting itself online from the robot's force sensors, and it stays safe when one of those sensors fails. One
  control tick takes **1.7 µs** on one laptop core.
- **Robot arm (SARCOS benchmark, 7 joints):** at least as accurate as a standard network with **6× more computation**,
  at **2.5 µs** per tick instead of 12–14 µs. With self-correction, **1.7× lower error** than that network.
- **Motor drive (20 kHz current controller):** **23 % lower error** than a standard network on recorded runs. Compiled
  into a lookup table it takes **about 0.1 µs** per prediction, and one laptop computes **510–630 million** control
  voltages per second across a fleet of motors.
- **Many robots at once:** one call steps thousands of robots, with exactly the same results as running each on its
  own.
- It can also add self-correction to **a network you already have**: 2.7–4.4× lower error on the quadruped.

---

## Capabilities at a glance

**How errors are measured** (lower is better for both):
- **nMSE**, *normalised mean squared error*, for the robots: the mean squared prediction error divided by the
  variance of the true signal, averaged over joints. 0 is perfect, and 1 is no better than always predicting the
  average. An nMSE of 0.02 means the remaining error's typical size is about 14 % of the signal's.
- **Volts rms** for the motor drive: the root-mean-square difference between the predicted voltages and those of
  the exact controller.

The **baseline** is a conventional dense neural network trained on the same data. Ranges cover two independently
trained copies of each model; the "× lower" factors compare their averages. MACs are the multiply-accumulate
operations per prediction. All speeds are on one 15 W laptop processor (Intel Core i7-1255U), with no GPU.

| | **Quadruped robot** (HyQ) | **Robot arm** (SARCOS) | **Motor drive** (PMSM) |
|---|---|---|---|
| task | 12 joint forces, 1 kHz loop | 7 joint torques, 100 Hz | 2 control voltages, 20 kHz loop |
| error measure | nMSE | nMSE (the benchmark's score) | volts rms |
| **baseline** network | 104k MACs · 3.9 µs per tick | 276k MACs · 12–14 µs | 267k MACs · 12.5 µs |
| **SLD-RNC** model | FULL small · **24k MACs** | FULL medium · **46k MACs** | FULL large, as a lookup table · **38 MACs** |
| error: baseline → SLD-RNC | 0.118–0.119 → **0.020–0.021** with self-correction (**5.8× lower**); 0.070–0.073 without (1.7× lower) | 0.0134–0.0165 → **0.0085–0.0087** with self-correction (**1.7× lower**); 0.0135–0.0139 without (8 % lower) | 1.71–1.75 V → **1.34 V** (**23 % lower**) |
| error on a recording it never saw: baseline → SLD-RNC | 0.161–0.186 → **0.021** (**8.2× lower**) | — | — |
| time per prediction, one core | **1.7 µs** (2.3× faster) | **2.5 µs** (5× faster) | **≈ 0.1 µs** (about 125× faster) |
| control loop it could run, one core | **590 kHz** | **400 kHz** | **≈ 10 MHz** |
| share of each control tick | **0.17 %** | **0.03 %** | **0.2 %** |
| many robots at once, 12 threads | **2.4–2.7 million** robot-ticks per second | **2.05 million** | **260–320 million** |
| predictions per second, 12 threads | **29–32 million** | **14 million** | **510–630 million** |
| robots served in real time, one laptop | **2,400–2,700** quadrupeds at 1 kHz | **20,000** arms at 100 Hz | **13,000–16,000** motors at 20 kHz |
| model file | **100 KB** (baseline 421 KB) | **187 KB** (baseline 1.1 MB) | **17 MB** table, or 278 KB as a network (baseline 1.1 MB) |
| also | self-corrects online; halves the error when a force sensor fails; trains in 15 s | trains in 40 s | as a network instead of a table: 1.32 V at 2.9 µs |

The 12-thread figures are for a cool laptop: under sustained full load this 15 W chip slows down by up to a third.

---

## 1. Ways to use SLD-RNC

| mode | you bring | you get |
|---|---|---|
| **FULL** | recorded data from your machine: its sensor signals and the measured forces (or voltages) | a model trained by SLD-RNC, with self-correction. The most accurate and the fastest option |
| **CORRECTION** | a neural network you have already trained, plus about half a minute of recorded data | your network with SLD-RNC self-correction added |
| **RAW** | a neural network you have already trained | your network, run by the SLD-RNC engine with no other software needed |

- **Self-correction can be switched on or off** at run time in FULL and CORRECTION modes.
- **One robot or many.** Each robot runs in its own *session*, or thousands run together in a *fleet*: one call per
  tick for all of them, with exactly the same results as separate sessions. For larger models a fleet is 1.3–1.8×
  faster than the same robots in sessions; for small models it runs at about the same speed.
- **Lookup tables.** A model of a memoryless task (one whose output depends only on the present inputs, such as a
  motor's current controller) can be compiled into a lookup table: the fastest form, about 0.1 µs per prediction.
- **Formats.** Networks come in as ONNX files (which PyTorch, TensorFlow and most other tools can export) or as
  plain layer weights. Models are saved in SLD-RNC's own protected file format (`.sldm`).
- **Training time.** On four minutes of 1 kHz robot logs: 15 s (small), about 30 s (medium) and about 2 minutes
  (large) on the laptop. Adding self-correction to an existing network takes a few seconds.

---

## 2. Quadruped robot (HyQ)

**What was tested.** HyQ is a hydraulic quadruped built by the Italian Institute of Technology, and its public
dataset records the robot trotting around a lab. SLD-RNC predicts all 12 joint efforts, 1,000 times per second, from
the robot's joint encoders and motion sensors. The predictions are scored against the efforts the robot's force
sensors actually measured.

Two recordings were used: "Trot in Lab 2" (344 s, about 0.2 m/s) and "Trot in Lab 1" (374 s, faster, about 0.3 m/s).
Models were trained on one recording and tested on a later stretch of it, and on the whole of the other recording.
The "standard neural network" is a conventional network of the kind normally used for this task, trained on the
same data.

### Accuracy of each mode

Error is nMSE (0 = perfect, 1 = no better than predicting the average). Each cell shows the error **without → with**
self-correction.

| mode | same recording (Lab 2), held-out stretch | trained on Lab 2, scored on all of Lab 1 | trained on Lab 1, scored on all of Lab 2 |
|---|---|---|---|
| **RAW**: standard network, as-is | 0.118–0.119 | 0.161–0.186 | 0.160–0.163 |
| **CORRECTION**: standard network + SLD-RNC self-correction | 0.118–0.119 → **0.040–0.044** | 0.161–0.186 → **0.040–0.045** | 0.160–0.163 → **0.037** |
| **FULL**, small | 0.070–0.073 → **0.020–0.021** | 0.079–0.083 → **0.021** | 0.048–0.054 → **0.019** |
| **FULL**, medium | 0.072–0.076 → **0.020** | 0.077–0.079 → **0.020** | 0.045 → **0.016** |
| **FULL**, large | 0.068–0.070 → **0.021** | 0.077–0.085 → **0.019–0.020** | 0.043–0.046 → **0.016** |

Ranges cover two independently trained copies of each model. All numbers were measured with the released library.

What the table says:
- **FULL beats the standard network before any correction**: 1.6–1.7× lower error on the same recording, and
  1.9–3.7× on the recording it never saw.
- **Self-correction is the biggest single step.** It cuts the standard network's error 2.7–4.4× (CORRECTION) and
  SLD-RNC's own model's error a further 2.5–4.2× (FULL). With it, FULL's error is about half of CORRECTION's.
- **Size matters little once corrected.** Trained on Lab 2, small, medium and large end up within 10 % of each other;
  trained on Lab 1, medium and large are about 15 % better than small. The small model is 3.6× faster than the large
  one.
- **Direction matters.** A model trained on the faster, more varied trot transfers better (0.043–0.054 uncorrected on
  the slower recording) than the other way round (0.077–0.085).

More detail:
- **Physical units.** With self-correction, FULL's typical errors are about 0.8–1 Nm on the hip joints and 46–60 N on
  the hydraulic leg actuators. That's about 1 % of the sensors' range. CORRECTION reaches 1.2–1.5 Nm and 66–75 N;
  the standard network alone is at 2–3 Nm and 76–110 N.
- **Planning ahead.** Self-correction normally uses the force measured one millisecond earlier. If the prediction
  must be made 50 ms ahead, as a motion planner would need, FULL's error is 0.031–0.049 and CORRECTION's is
  0.084–0.110. A copy of the model tuned for 50 ms does better while the robot keeps trotting: 0.037–0.041 later in
  the same recording and 0.026–0.031 on the slower unseen recording (14–20 % lower), and about the same (8 % better to
  7 % worse) on the faster one. Where the robot comes to a stop it is worse (0.026–0.032 instead of 0.018–0.023): tuned
  for late measurements, the correction adapts more slowly.
- **Bigger standard networks do not catch up.** Networks up to 43× larger than the standard one were no more
  accurate.

### Hardest moments

Both recordings end with the robot standing still, a situation that barely appears in training. With all four feet
on the ground, how the weight is shared between the legs cannot be worked out from the joint angles alone, so every
model is less accurate there without self-correction. Lab 1's own held-out stretch is mostly standing: the standard
network scores 0.29 on it and FULL without self-correction 0.20–0.28. With self-correction, FULL scores 0.010–0.014
and CORRECTION 0.024.

### Sensor failure

One leg force sensor was made to read zero for 2 seconds, and accuracy (nMSE) was scored against the true forces:

| error during the failure and the 5 s after | FULL | CORRECTION (standard network) |
|---|---|---|
| no self-correction | 0.060–0.066 | 0.121–0.130 |
| naive self-correction (trusts the broken sensor) | 0.100–0.103 (**worse** than none) | 0.135 (**worse** than none) |
| **SLD-RNC protected self-correction** | **0.031–0.034** | **0.066–0.067** |

The protection costs little when every sensor is healthy: under 1 % extra error on three of the four test stretches,
and 6–13 % on the fourth (the Lab 1 recording, for models trained on Lab 2).

### Speed

One robot, one control tick, from raw sensor readings to corrected joint efforts, on one laptop processor core; and
many robots at once on all 12 threads:

| | time per tick (typical) | slowest 1 % of ticks | share of a 1 kHz control loop | many robots at once (12 threads) |
|---|---|---|---|---|
| RAW: standard network | 3.9 µs | 4.8 µs | 0.4 % | 1.4 million per second |
| CORRECTION: standard network + self-correction | 4.3 µs | 5.1 µs | 0.4 % | 1.3 million per second |
| **FULL, small** | **1.7 µs** | **1.9 µs** | **0.17 %** | **2.4–2.7 million per second** |
| FULL, medium | 3.0 µs | 3.4 µs | 0.3 % | 1.5 million per second |
| FULL, large | 6.1 µs | 7.8 µs | 0.6 % | 0.9 million per second |

- **Many robots** are 1,024 robots in one fleet, each with its own self-correction. The figures are for a cool
  laptop; under sustained full load this 15 W chip slows down by up to a third. The slowest 1 % of ticks depends on
  what else the laptop is doing.
- **Other cores.** On the laptop's power-saving cores the small model takes 4.1 µs.
- **Planning budget.** A motion planner that tries 1,000 candidate motions, 50 steps each, 50 times a second needs
  2.5 million evaluations per second. Running predictions only, a fleet of the small model delivers 3.5 million per
  second on the laptop, almost 40 % above that budget.
- **Microcontroller.** The self-correction stage also runs on an 8-bit, 16 MHz Arduino-class chip, at 1.2 ms per
  joint using 1.5 KB of memory (measured in a cycle-accurate simulation of the chip).

---

## 3. Robot arm (SARCOS benchmark)

SARCOS is a standard benchmark: a 7-joint hydraulic arm, predicting joint torques from joint positions, velocities
and accelerations at 100 Hz. Models were trained on the benchmark's recording with eight stretches held out, and are
scored on those held-out stretches with the benchmark's error measure (nMSE). The standard network is a conventional
dense network trained on the same data (two independently trained copies).

For the arm, SLD-RNC's models use two options from its interface: the joint positions are declared as rotary angles,
and the torques as linear in the joint accelerations, which is how rigid-body physics behaves.

| | error (nMSE) | with self-correction | computation per tick | time per tick |
|---|---|---|---|---|
| standard network, run as-is (RAW) | 0.0134–0.0165 | — | 276k operations | 12–14 µs |
| standard network + self-correction (CORRECTION) | — | 0.0088–0.0101 | 276k operations | 13 µs |
| SLD-RNC FULL, small | 0.0143–0.0161 | 0.0094–0.0104 | 19k operations | **1.4 µs** |
| **SLD-RNC FULL, medium** | **0.0135–0.0139** | **0.0085–0.0087** | 46k operations | **2.5 µs** |
| SLD-RNC FULL, large | **0.0129** | **0.0079–0.0083** | 125k operations | 5.7 µs |

- **Without self-correction** (the benchmark's convention), SLD-RNC's medium model is at least as accurate as the
  standard network (8 % lower on average) with 6× less computation and a fifth of the time per tick. The large model
  is 14 % more accurate than the standard network, with less than half its computation.
- **With self-correction**, using the torque measured one tick (10 ms) earlier, every model gets 35–38 % more
  accurate. SLD-RNC's medium model with self-correction (0.0085–0.0087) beats the standard network with
  self-correction (0.0088–0.0101) and is 5× faster.
- **Many arms or candidate motions.** On 12 threads a fleet of the medium model evaluates 2.05 million robot-ticks
  per second with self-correction, and the small model 3.1 million: 1,000 candidate 30-step motions in 15 ms
  (medium) or 10 ms (small). The standard network manages 0.5 million per second as a fleet.

---

## 4. Motor drive (permanent-magnet motor current control)

The task is to reproduce a motor's current controller, 20,000 times per second: from the measured currents, speed,
current references and rotor angle, compute the two voltages to apply. It is scored on recorded runs of the motor
against the exact controller (volts rms), and on simulated runs that stay inside the training conditions.

This task has no memory: the right voltages depend only on the present inputs. SLD-RNC's models for it are trained
on samples that cover the motor's whole operating range, with two options from its interface (current inputs only,
and voltages linear in the speed and the current references). The trained model can then be compiled into a lookup
table.

| | voltage error, recorded runs | simulated runs | computation per prediction | time per prediction, one core | many motors (12 threads) |
|---|---|---|---|---|---|
| standard network, 512 wide (RAW) | 1.71–1.75 V | **1.54–1.55 V** | 267k operations | 12.5 µs | 0.55 million per second |
| standard network, 64 wide (RAW) | 2.65–2.73 V | — | 4.7k operations | 0.25 µs | 26 million per second |
| SLD-RNC FULL, small | 1.76–1.79 V | 2.26–2.42 V | 4.9k operations | 0.30 µs | 23 million per second |
| SLD-RNC FULL, medium | 1.46–1.50 V | 1.80–1.98 V | 18k operations | 0.9 µs | 7.3 million per second |
| SLD-RNC FULL, large | **1.32 V** | 1.61–1.63 V | 69k operations | 2.9 µs | 2.7 million per second |
| **SLD-RNC FULL, large, as a lookup table** | **1.34 V** | 1.70–1.72 V | **38 operations** | **≈ 0.1 µs** | **260–320 million per second** |

- **On the recorded runs**, SLD-RNC's large model is 23 % more accurate than the 512-wide network with a quarter of
  its computation, and its lookup table keeps almost all of that (1.34 V) at about 0.1 µs per prediction. On one
  core, a fleet of motors takes 14 ns per motor.
- **On the simulated runs**, inside the training conditions of every model here, the 512-wide network stays the
  most accurate (1.54–1.55 V against 1.61–1.63 V).
- **Many motors.** The table runs 260–320 million motor-ticks per second on 12 threads (65,536 motors per call, 510–630
  million voltages per second); with 4,096 motors per call, 190–240 million. That is 13,000–16,000 motors at 20 kHz on
  one laptop.
- **What the data must cover.** Models trained only on simulated runs at low speeds failed completely on the recorded
  runs, which reach speeds those runs never did. With samples covering the whole operating range, they did not.
- **Your own network** can still be run as-is in RAW mode, with exactly its own accuracy (first two rows).

---

## 5. Using the library

- **Languages:** C++, Python (with NumPy) and plain C. Other languages can call the C interface.
- **Platforms:** Windows x64 and Linux x64 (any distribution from 2014 on) are built and tested. Linux ARM64 (e.g.
  Jetson, Raspberry Pi 4/5) and macOS (Intel and Apple Silicon, macOS 11 or later) are built but not yet run on
  that hardware.
- **Models in:** your recorded data (FULL), or your own network as an ONNX file, a PyTorch model or plain layer
  weights (RAW and CORRECTION).
- **Models out:** `.sldm` files, SLD-RNC's protected model format, portable across platforms.
- **Dependencies:** none. The C++ runtime is built in; the Python package needs only NumPy.
- **Shipped as:** one zip per platform, with headers, the library, a CMake package, the Python package, examples and
  the documentation.

**Documentation.** The full API documentation ships inside every package, in the `api/` folder. Start with
**`api/readme.md`**: installation, the three modes, quick starts in C, C++ and Python, and the rules that apply
everywhere. From there:
- `api/python.md`, `api/cpp.md`, `api/c.md`: every function, parameter and error, per language;
- `api/concepts.md`: how sessions, fleets, lookup tables, self-correction and measurement delays work, and which
  mode suits which task;
- `api/model_formats.md`: which networks can be imported, and what comes out.

---

## 6. Limits

- **Recorded data only.** Accuracy was measured against recorded robot data; SLD-RNC has not yet driven a physical
  robot. Closed-loop control was tested only for the motor drive, in simulation.
- **One quadruped.** Results come from two recordings of the same robot. On a second quadruped dataset (MIT Mini
  Cheetah) no method I tested, mine included, did well. That robot has no joint-force sensors, so the "true"
  torques are only estimates.
- **Self-correction needs force sensing** at the joints.
- **FULL learns what the data contains.** The data must cover the speeds, loads and motions the model will meet. On
  the motor drive, inside the conditions of the simulated runs, the large standard network remained more accurate.
- **Fleets speed up larger models only.** For small models, most of the time goes into each robot's own input
  processing and self-correction, and a fleet runs at about the speed of separate sessions. Very large fleets of
  models that keep recent history are limited by memory.
- **Lookup tables** are for memoryless tasks whose main inputs are at most three values; they add a small error
  (0.5–1.4 % on the motor drive) and take more memory (17 MB there).
- **Planning ahead needs representative tuning data.** Self-correction tuned for predictions 50 ms ahead was 14–20 %
  more accurate while conditions stayed like the tuning data, and up to 57 % less accurate after the robot stopped.
- **One test machine.** Speeds were measured on one laptop processor. The ARM and macOS builds have not been run on
  their hardware, and embedded robot computers and real microcontroller boards have not been measured.

---

## 7. Test hardware

| | |
|---|---|
| processor | Intel Core i7-1255U laptop chip (15 W): 10 cores (2 performance, 8 power-saving), 12 threads, AVX2 |
| memory | 32 GB |
| operating systems | Windows 11 (MSVC compiler); Ubuntu 24.04 under WSL2 on the same laptop (GCC 13) |
| microcontroller | ATmega328P, 16 MHz, 2 KB RAM (cycle-accurate simulation of the compiled program) |
| GPU | none used |

All single-robot speeds are medians of repeated runs on one performance core, unless "power-saving cores" or
"12 threads" is stated. Microsecond timings of the lookup table are at the limit of the timer's resolution (0.1 µs).
