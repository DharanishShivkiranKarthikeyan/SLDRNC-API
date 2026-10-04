# Building and platforms

## Prebuilt libraries (`dist/`)

| platform | file | built with | tested here |
|---|---|---|---|
| Windows x64 | `windows-x64/sldrnc.dll` (+ `sldrnc.lib` import library) | MSVC 14.51, CMake + Ninja | yes: C++ and Python test suites |
| Linux x64 | `linux-x64/libsldrnc.so` | Zig 0.16 (clang), glibc 2.17 target | yes: C++ and Python test suites (Ubuntu 24.04) |
| Linux ARM64 | `linux-arm64/libsldrnc.so` | Zig 0.16, glibc 2.17 target | no (cross-compiled; inspected only) |
| macOS x64 | `macos-x64/libsldrnc.dylib` | Zig 0.16, macOS 11 target | no (cross-compiled; inspected only) |
| macOS ARM64 | `macos-arm64/libsldrnc.dylib` | Zig 0.16, macOS 11 target | no (cross-compiled; inspected only) |

The same libraries are inside the Python package (`python/sldrnc/_native/<platform>/`). Build on a Mac, or use the
CI workflow, for macOS libraries tested on real hardware.

Each library exports only the C interface (`sldrnc_*`); everything else is hidden and stripped. Every build carries
its own C++ runtime: the Windows DLL needs only `KERNEL32.dll` (no Visual C++ Redistributable; CMake option
`SLDRNC_STATIC_RUNTIME`, default ON), and the Linux library needs only the system C library (glibc 2.17 or newer:
practically every distribution since 2014). The Linux library's soname is `libsldrnc.so`; the macOS install name is
`@rpath/libsldrnc.dylib`.

On x86-64 the library checks the CPU at start-up and uses AVX2 + FMA kernels when available, with SSE2 otherwise.
On ARM64 it uses NEON.

## Building from source

Requires CMake 3.16+ and a C++17 compiler: MSVC 2019+, GCC 9+, Clang 10+ or AppleClang 12+.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release
```

| script | |
|---|---|
| `scripts/build_windows.ps1 [-Test]` | Windows build with Visual Studio 2022+; copies the DLL into `dist/` and the Python package |
| `scripts/build_cross.py [targets]` | cross-compile Linux x64 / ARM64 and macOS x64 / ARM64 with Zig (`pip install ziglang`) |
| `scripts/stage_python.py` | copy every library in `dist/` into the Python package |
| `scripts/package.py [platforms]` | build the shipping zips, `packages/sldrnc-<version>-<platform>.zip` |
| `.github/workflows/sldrnc.yml` (repository root) | builds and tests on Windows, Linux and macOS (Intel and Apple Silicon) runners |

## Packages (what ships)

`scripts/package.py` makes one zip per platform from `dist/`: `README.md`, the two licences, `api/` (all user
documentation), `include/sldrnc/`, the library (`bin/sldrnc.dll` + `lib/sldrnc.lib`, or `lib/libsldrnc.so`, or
`lib/libsldrnc.dylib`), `lib/cmake/sldrnc/` (`find_package(sldrnc)`; sources in `packaging/`), `python/` (pip
package with that platform's library) and `examples/`. It warns while the licences still hold placeholders.

Release checklist: build Windows (`build_windows.ps1 -Test`) and the cross targets (`build_cross.py`), run the
Linux tests in WSL or CI, `stage_python.py`, `package.py`, then unzip each package on its platform and build
`examples/cpp` with CMake and run `examples/python/quickstart.py`.

Long paths: on Windows, build the packaged example from a short folder; CMake's compiler checks fail when the build
folder's path exceeds about 250 characters.

## Python package

```bash
pip install ./python
python -c "import sldrnc; print(sldrnc.version())"
```

The package is pure Python (`ctypes`) around the compiled library, so one package serves every Python version.
Set `SLDRNC_LIBRARY` to load a library from elsewhere.
