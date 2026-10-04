# SLD-RNC API

**SLD-RNC** (Self-correcting Learned Dynamics for Real-time Neural Control) predicts the forces a machine needs (joint
torques, actuator forces, motor voltages, ...) from its own sensors, fast enough to run inside the control loop. It
also corrects itself online from the measured values. For a 12-joint quadruped, one control tick takes about 2 µs on
one laptop core.

This folder documents the API. Start here, then go to the reference for your language.

| page | contents |
|---|---|
| **readme.md** (this page) | install, the three modes, a quick start in each language, the rules that apply everywhere |
| [python.md](python.md) | Python reference: every class, function, parameter and error |
| [cpp.md](cpp.md) | C++ reference |
| [c.md](c.md) | C reference: every function, struct field and enum value; ABI rules |
| [concepts.md](concepts.md) | how the pieces work: schemas, sessions, fleets, tables, self-correction, measurement delay, protection, reading errors |
| [model_formats.md](model_formats.md) | which models go in (ONNX, PyTorch, layer weights, your logs) and what comes out (`.sldm`) |

---

## 1. What's in the package

```
sldrnc-1.0.0-<platform>/
├── README.md                 short overview
├── LICENSE.txt               licence of the API: headers, Python package, examples, docs
├── LICENSE-BINARY.txt        licence of the compiled library
├── api/                      this documentation
├── include/sldrnc/
│   ├── sldrnc.h              C interface (everything goes through it)
│   └── sldrnc.hpp            C++ interface (header-only, C++17)
├── bin/sldrnc.dll            Windows: the library
├── lib/
│   ├── sldrnc.lib            Windows: import library to link against
│   ├── libsldrnc.so          Linux: the library
│   ├── libsldrnc.dylib       macOS: the library
│   └── cmake/sldrnc/         CMake package files (find_package)
├── python/                   pip-installable Python package (with the library inside)
└── examples/
    ├── cpp/                  quickstart.cpp + CMakeLists.txt
    └── python/               quickstart.py
```

Each zip holds the library for one platform only.

---

## 2. Install

### C++ with CMake

```cmake
cmake_minimum_required(VERSION 3.16)
project(my_robot CXX)
find_package(sldrnc REQUIRED)                         # configure with -DCMAKE_PREFIX_PATH=/path/to/sldrnc-1.0.0-<platform>
add_executable(my_robot main.cpp)
target_link_libraries(my_robot PRIVATE sldrnc::sldrnc)
```

The `sldrnc::sldrnc` target carries the include path and the library. On Windows, copy `bin/sldrnc.dll` next to
your executable. The example's `CMakeLists.txt` shows how to do it automatically.

### C or C++ without CMake

| platform | compile | link | at run time |
|---|---|---|---|
| Windows (MSVC) | `/I <pkg>\include /std:c++17` | `<pkg>\lib\sldrnc.lib` | `sldrnc.dll` next to the `.exe` (or on `PATH`) |
| Linux (GCC / Clang) | `-I<pkg>/include -std=c++17` | `-L<pkg>/lib -lsldrnc -Wl,-rpath,'$ORIGIN'` | `libsldrnc.so` next to the program (or on `LD_LIBRARY_PATH`) |
| macOS (Clang) | `-I<pkg>/include -std=c++17` | `-L<pkg>/lib -lsldrnc -Wl,-rpath,@executable_path` | `libsldrnc.dylib` next to the program |

From plain C, include `sldrnc.h` only; it is C99.

The library has no dependencies beyond the operating system: the C++ runtime is linked in, and on Linux it needs
only glibc 2.17 or later.

### Python

```bash
pip install ./python          # from the unzipped package; needs Python 3.8+ and NumPy
```

```python
import sldrnc
print(sldrnc.version())
```

To use the package without installing it, put `python/` on `PYTHONPATH`.

### Other languages

Rust, C#, Julia, Go and others can call the C interface directly: opaque handles, plain structs and status codes
([c.md](c.md)).

---

## 3. Three ways to use it

| mode | you bring | you get | API |
|---|---|---|---|
| **FULL** | logs from your machine: sensor signals + measured outputs | a model trained by SLD-RNC, with self-correction. The most accurate and the fastest | `sldrnc.train` / `Model::train` / `sldrnc_train_full` |
| **CORRECTION** | your own trained network + about half a minute of recorded data | your network with SLD-RNC self-correction on top | `add_correction` |
| **RAW** | your own trained network | your network, run by the SLD-RNC engine with no other dependencies | `from_onnx`, `from_layers`, `from_torch` (Python) |

Self-correction can be switched off per session, so FULL and CORRECTION models can also run uncorrected.

Every model runs one robot per **session**, or many robots at once in a **fleet** (same outputs bit for bit, one call
per tick, up to 1.8× the throughput of sessions for larger networks). A FULL model of a memoryless map (history off) with at most 3 non-affine input values can also be
compiled into a **lookup table**, the fastest form. See [concepts.md](concepts.md#many-robots-fleets).

**Measured on a real hydraulic quadruped** (HyQ, 12 joints, 1 kHz; error = nMSE, the normalised mean squared error:
0 is perfect, 1 is no better than predicting the average; ranges over two training seeds):

| mode | held-out end of the training recording | a different recording, never seen | time per tick |
|---|---|---|---|
| RAW: a standard network | 0.118–0.119 | 0.160–0.186 | 3.9 µs |
| CORRECTION: the same network + self-correction | 0.040–0.044 | 0.037–0.045 | 4.3 µs |
| FULL (small), self-correction off | 0.070–0.073 | 0.048–0.083 | — |
| **FULL (small)** | **0.020–0.021** | **0.019–0.021** | **1.7 µs** |

Times are for one laptop core (Intel Core i7-1255U). RAW and CORRECTION times depend on the size of your network.

Which mode fits which task (a robot arm, a 20 kHz motor drive, ...), with measurements: see
[concepts.md](concepts.md#choosing-a-mode).

---

## 4. How it fits together

```
           FULL                 RAW                       CORRECTION
  Schema + logs ──train──►  Model ◄──from_onnx / from_layers──  your network
                              │         └──add_correction──► (CORRECTION model)
                              │
              save / load ◄───┤──► info
                              │
                 session()  or  fleet(N)      one robot, or N robots stepped together
                              │
             every tick:  step(x) ──► y        predict (a fleet: N rows at once)
                          observe(y_measured)  self-correct (when the measurement arrives)

  memoryless FULL model ──compile_table──► table Model (same API, the fastest form)
```

- **Schema** (FULL only): which signals one tick holds, and which outputs to predict.
- **Model**: read-only once built, saved to and loaded from `.sldm` files, and shared by any number of sessions and
  threads.
- **Session**: the state of one robot (recent history and current correction). `step` and `observe` never allocate,
  lock or block.
- **Fleet**: the state of N robots, stepped in one call. Outputs equal N separate sessions bit for bit.

---

## 5. Quick start

**Python**

```python
import numpy as np
import sldrnc

schema = sldrnc.Schema(rate_hz=1000)
schema.add("q", sldrnc.Input.POSITION, 12).add("qd", sldrnc.Input.VELOCITY, 12)
schema.add("gyro", sldrnc.Input.GYRO, 3).add("accel", sldrnc.Input.ACCEL, 3)
schema.outputs(12, velocity="qd")                     # 12 joint torques

model, report = sldrnc.train(schema, X_train, Y_train, X_val, Y_val, size="small")
print(report.validation_error, report.validation_error_corrected)
model.save("robot.sldm")

robot = sldrnc.Model.load("robot.sldm").session()
torque = robot.step(x)                                # every tick
robot.observe(torque_measured)
```

**C++**

```cpp
#include <sldrnc/sldrnc.hpp>

sldrnc::Schema schema(1000.0);
schema.add("q", sldrnc::Input::Position, 12).add("qd", sldrnc::Input::Velocity, 12).outputs(12, "qd");
sldrnc::Model model = sldrnc::Model::train(schema, {X_train, 24}, {Y_train, 12}, {X_val, 24}, {Y_val, 12});

sldrnc::Session robot = model.session();
robot.step(x, torque);                                // every tick
robot.observe(torque_measured);
```

**C**

```c
#include <sldrnc/sldrnc.h>

sldrnc_model* model = NULL;
sldrnc_session* robot = NULL;
if (sldrnc_model_load("robot.sldm", &model) != SLDRNC_OK) { puts(sldrnc_last_error()); return 1; }
sldrnc_session_create(model, NULL, &robot);
sldrnc_session_step(robot, x, torque);                /* every tick */
sldrnc_session_observe(robot, torque_measured);
sldrnc_session_free(robot);
sldrnc_model_free(model);
```

**Bring your own network**

```python
mine = sldrnc.Model.from_onnx("my_net.onnx")                        # RAW
mine.add_correction(X_val, Y_val, rate_hz=1000,                     # CORRECTION
                    velocity_index=list(range(12, 24)))
```

**Many robots at once**

```python
fleet = model.fleet(4096)                         # 4,096 robots of one model
U = fleet.step(X)                                 # X: 4096 x input_size -> U: 4096 x outputs
fleet.observe(Y_measured)
```

**A 20 kHz motor drive: memoryless, structured, compiled to a table**

```python
schema = sldrnc.Schema(rate_hz=20000, history=False)
schema.add("i_dq", sldrnc.Input.GENERIC, 2).add("angle", sldrnc.Input.ANGLE, 1)
schema.add("speed", sldrnc.Input.VELOCITY, 1).add("i_dq_ref", sldrnc.Input.GENERIC, 2)
schema.outputs(2).affine("speed", "i_dq_ref")
model, report = sldrnc.train(schema, X_samples, V_samples, X_val, V_val, size="medium", fit_correction=False)
table = model.compile_table()                     # same API; the fastest form
```

Complete, runnable programs (with a simulated robot) are in `examples/`.

---

## 6. Rules that apply everywhere

**Data.** One row per control tick, as row-major `double` (NumPy 2-D arrays in Python). For FULL models, a row is the
raw signals in the schema's channel order. For RAW / CORRECTION models, a row is your network's own input vector. A
recording must be one continuous stretch at a fixed rate. FULL training accepts several recordings (one per log file)
as lists; never join separate recordings into one array.

**Warm-up.** A fresh FULL session needs about 2 s of data before it is fully accurate (`info.warmup_seconds`). When
scoring a recording, skip those ticks (`score_from`). Models trained without history need no warm-up.

**Measurement delay.** A measurement observed right after the tick it belongs to has delay 1 (the normal control
loop). If measurements arrive later, or if you predict ahead (a planner), tune the correction for that delay
(`correction_delay_ticks` when training, `delay_ticks` in `add_correction`, or a re-tuned copy from
`tune_correction`) and feed late measurements with `observe(y, delay_ticks)`. See
[concepts.md](concepts.md#measurement-delay).

**Threads.** Models can be shared by any number of threads. A session or fleet belongs to one thread at a time (a
fleet runs its own worker threads inside `step`). Call `add_correction` (which changes a model) before creating
sessions from it.

**Real time.** Load models and create sessions at start-up. After that, `step` and `observe` never allocate memory,
take locks, do I/O or block.

**Errors.** C functions return a status code and `sldrnc_last_error()` explains it; C++ throws `sldrnc::Error`;
Python raises `sldrnc.Error`. The codes are the same everywhere:

| code | meaning |
|---|---|
| `ARGUMENT` | a pointer, size or value is wrong (the message says which) |
| `IO` | a file can't be read or written |
| `FORMAT` | not a model file, damaged, or from a newer version |
| `UNSUPPORTED` | the ONNX model uses an operation SLD-RNC doesn't run (the message names it) |
| `STATE` | the call doesn't apply here (e.g. `add_correction` on a FULL model, `compile_table` on a model with history) |
| `DATA` | data too short, constant, or containing NaN / infinity |
| `INTERNAL` | a bug: please report the message |

---

## 7. Model formats

| in | | out | |
|---|---|---|---|
| your logs | FULL training | `.sldm` file | any mode; portable across platforms; checksummed and stored scrambled |
| ONNX file | feed-forward networks: Gemm / MatMul, common activations | predictions | one tick (`step`) or a whole recording (`run`) |
| PyTorch | via ONNX (`Model.from_torch` in Python) | reports | training / tuning summaries, errors per output |
| layer weights | `from_layers`: matrices, biases, activations | | |

Details and the list of supported ONNX operations: [model_formats.md](model_formats.md).

---

## 8. Platforms

| platform | library | status |
|---|---|---|
| Windows x64 | `bin/sldrnc.dll` + `lib/sldrnc.lib` | built and tested |
| Linux x64, glibc 2.17+ (practically every distribution since 2014) | `lib/libsldrnc.so` | built and tested |
| Linux ARM64 (Jetson, Raspberry Pi 4/5 with a 64-bit OS) | `lib/libsldrnc.so` | built; not yet run on that hardware |
| macOS 11+, Intel and Apple Silicon | `lib/libsldrnc.dylib` | built; not yet run on that hardware |

x86-64 builds pick AVX2 / FMA code at run time when the processor has it, and portable code otherwise.

---

## 9. Versions and compatibility

- Version 1.0.0. Within major version 1, functions keep their signatures and structs keep their layouts; new
  features arrive as new functions.
- Use the headers that came with the library. `sldrnc_version()` reports the library's version at run time.
- `.sldm` files from older versions load in newer ones. A file from a newer version is refused with `FORMAT`, never
  misread.

---

## 10. License

The API (headers, Python package, examples and this documentation) is under the terms in `LICENSE.txt`. The compiled
library is under the terms in `LICENSE-BINARY.txt`. Models you train with SLD-RNC, and their predictions, are yours.
