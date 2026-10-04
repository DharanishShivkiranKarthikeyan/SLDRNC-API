# SLD-RNC

**Fast learned dynamics for robots and machines.** SLD-RNC (Self-correcting Learned Dynamics for Real-time Neural
Control) predicts the forces (torques, efforts, voltages, ...) a
machine needs from its own sensors, fast enough to run inside the control loop. It also corrects itself online from
the measured values.

It is one small shared library with a C interface, plus a C++ API and a Python package on top of it.

| | |
|---|---|
| Platforms | Windows x64, Linux x64 / ARM64 (glibc 2.17+), macOS x64 / Apple Silicon (11+) |
| Languages | C, C++17 (header-only wrapper), Python 3.8+ (NumPy) |
| Dependencies | none (the Python package needs NumPy) |
| Speed | ~2 µs per control tick for a 12-joint quadruped on one laptop core |

## Three ways to use it

| mode | you bring | you get | when to use it |
|---|---|---|---|
| **FULL** | logs from your machine (sensor signals + measured outputs) | a model trained by SLD-RNC, with self-correction | best accuracy; the recommended path |
| **CORRECTION** | your own trained network (ONNX / PyTorch / layers) + a little validation data | your network plus SLD-RNC self-correction | you must keep your model, but want much of the accuracy gain |
| **RAW** | your own trained network | your network, run by the SLD-RNC engine | you only want fast, dependency-free inference |

Measured on a real hydraulic quadruped (HyQ, 12 joints, 1 kHz; lower is better, 0 = perfect):

| mode | held-out test (same recording) | a different recording, never seen | the other way round | time per tick (one laptop core) |
|---|---|---|---|---|
| RAW: your network as-is | 0.118–0.119 | 0.161–0.186 | 0.160–0.163 | 3.9 µs |
| CORRECTION: your network + SLD-RNC self-correction | **0.040–0.044** | **0.040–0.045** | **0.037** | 4.3 µs |
| FULL (small), self-correction off | 0.070–0.073 | 0.079–0.083 | 0.048–0.054 | — |
| **FULL (small)** | **0.020–0.021** | **0.021** | **0.019** | **1.7 µs** |

- "Your network" here is a standard dense network (2 × 256, 104k multiply-adds) trained on the same data. RAW and
  CORRECTION timings depend on your network's size.
- Trained on Trot in Lab 2 and scored on its held-out end and on all of Trot in Lab 1; "the other way round" trains on
  Lab 1 and scores all of Lab 2. Ranges cover two training seeds. Medium and large FULL models: see
  [Concepts](api/concepts.md#sizes-full-mode).
- Self-correction uses each measurement from the previous millisecond; when predicting 50 ms ahead, FULL scores
  0.043–0.045 and CORRECTION 0.095–0.110 on the held-out test.
- Reproduce with `benchmarks/hyq_modes.py` (add `train=lab1` for the other way round) and `benchmarks/summarise_modes.py`.

## Install

**Python**

```bash
pip install ./python          # the compiled library for your platform is inside the package
```

**C / C++ (CMake)**

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
cmake --install build --prefix /your/prefix   # then: find_package(sldrnc) ; target_link_libraries(app sldrnc::sldrnc)
```

Prebuilt libraries are in `dist/<platform>/`. Include `include/sldrnc/sldrnc.hpp` (C++) or `sldrnc.h` (C), and link
`sldrnc`.

## Quickstart (Python)

```python
import sldrnc

# describe one tick of your logs
schema = sldrnc.Schema(rate_hz=1000)
schema.add("q", sldrnc.Input.POSITION, 12)        # joint positions
schema.add("qd", sldrnc.Input.VELOCITY, 12)       # joint velocities
schema.add("imu_gyro", sldrnc.Input.GYRO, 3)
schema.add("imu_accel", sldrnc.Input.ACCEL, 3)
schema.outputs(12, velocity="qd")                 # 12 torques; output j belongs to the joint of qd[j]

# FULL: train (X: ticks x schema.width, Y: ticks x 12 measured torques)
model, report = sldrnc.train(schema, X_train, Y_train, X_val, Y_val, size="small")
model.save("robot.sldm")

# run time: one session per robot
robot = sldrnc.Model.load("robot.sldm").session()
torque = robot.step(x)            # every tick: predict
robot.observe(torque_measured)    # when the measurement arrives: self-correct
```

Bring your own network instead:

```python
mine = sldrnc.Model.from_onnx("my_net.onnx")                   # RAW
report = mine.add_correction(X_val, Y_val, rate_hz=1000,       # CORRECTION
                             velocity_index=[12 + j for j in range(12)])
```

## Quickstart (C++)

```cpp
#include <sldrnc/sldrnc.hpp>

sldrnc::Schema schema(1000.0);
schema.add("q", sldrnc::Input::Position, 12)
      .add("qd", sldrnc::Input::Velocity, 12)
      .outputs(12, "qd");

sldrnc::Report report;
sldrnc::Model model = sldrnc::Model::train(schema, {X_train, schema.width()}, {Y_train, 12},
                                           {X_val, schema.width()}, {Y_val, 12}, {}, &report);

sldrnc::Session robot = model.session();
robot.step(x, torque);            // every tick
robot.observe(torque_measured);   // drives self-correction
```

Complete, runnable programs: [`examples/cpp/quickstart.cpp`](examples/cpp/quickstart.cpp) and
[`examples/python/quickstart.py`](examples/python/quickstart.py).

## Documentation

The API documentation lives in [`api/`](api/readme.md) and ships inside every package:

- [API readme](api/readme.md): install, the three modes, quick starts in C, C++ and Python, the rules that apply
  everywhere
- References: [Python](api/python.md), [C++](api/cpp.md), [C](api/c.md)
- [Concepts](api/concepts.md): schemas, sessions, fleets, tables, self-correction, measurement delay, protection,
  reading the errors
- [Model formats](api/model_formats.md): what you can import (ONNX, PyTorch, layers) and what you get out (`.sldm`)
- [Building, platforms and packaging](docs/BUILDING.md) (for maintainers; not shipped)

## Shipping

`python scripts/package.py` builds one zip per platform in `packages/` (headers, library, CMake package, Python
package, examples, `api/` docs and licences; no library source). The API is under `LICENSE.txt` (MIT) and the compiled
library under `LICENSE-BINARY.txt`, which is a draft to complete before shipping.

## Testing

```bash
ctest --test-dir build -C Release      # C++ tests
python tests/test_python.py            # Python tests
python benchmarks/hyq_modes.py         # the quadruped benchmark above (needs the HyQ dataset)
```
