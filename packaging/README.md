# SLD-RNC @VERSION@ (@PLATFORM@)

**SLD-RNC** (Self-correcting Learned Dynamics for Real-time Neural Control) predicts the forces a machine needs from
its own sensors, fast enough to run inside the control loop, and corrects itself online from the measured values.

**Start here: [api/readme.md](api/readme.md)**, which covers installation, the three modes, quick starts in C, C++
and Python, and links to the full references.

| | |
|---|---|
| C / C++ | headers in `include/sldrnc/`, library in `@LIBDIR@`, CMake package in `lib/cmake/sldrnc/` |
| Python | `pip install ./python` (Python 3.8+, NumPy) |
| examples | `examples/cpp/` (build with CMake) and `examples/python/` |
| platform | @PLATFORM_LONG@ |

Quick check after unzipping:

```bash
pip install ./python
python examples/python/quickstart.py
```

The API (headers, Python package, examples, docs) is licensed under `LICENSE.txt`, and the compiled library under
`LICENSE-BINARY.txt`.
