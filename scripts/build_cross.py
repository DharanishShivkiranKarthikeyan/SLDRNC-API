"""build_cross.py -- cross-compile the SLD-RNC shared library with Zig (pip install ziglang) for platforms this machine
cannot build natively: Linux x86-64 / ARM64 against glibc 2.17 (runs on practically every distribution since 2014) and
macOS x86-64 / ARM64 (macOS 11+).  Same sources and flags as CMakeLists.txt.

  python scripts/build_cross.py [linux-x64 linux-arm64 macos-x64 macos-arm64]
Output: dist/<platform>/ and python/sldrnc/_native/<platform>/.
"""
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGETS = {
    "linux-x64": ("x86_64-linux-gnu.2.17", "libsldrnc.so", True),
    "linux-arm64": ("aarch64-linux-gnu.2.17", "libsldrnc.so", False),
    "macos-x64": ("x86_64-macos.11.0", "libsldrnc.dylib", True),
    "macos-arm64": ("aarch64-macos.11.0", "libsldrnc.dylib", False),
}
SOURCES = ["capi", "correction", "features", "kernels_generic", "model_io", "onnx_import", "runtime", "session", "train", "fleet",
           "structure"]


def zig(*args):
    cmd = [sys.executable, "-m", "ziglang", *args]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if r.returncode:
        sys.exit(" ".join(cmd) + "\n" + r.stdout + r.stderr)
    return r.stdout + r.stderr


def build(name):
    target, libname, x86 = TARGETS[name]
    obj = os.path.join(ROOT, "build", "cross", name)
    os.makedirs(obj, exist_ok=True)
    common = ["-target", target, "-O2", "-std=c++17", "-fPIC", "-fvisibility=hidden", "-fvisibility-inlines-hidden", "-DSLDRNC_BUILD",
              "-DNDEBUG", "-Iinclude"]
    objs = []
    for s in SOURCES + (["kernels_avx2", "kernels_table_avx2"] if x86 else []):
        if s in ("structure", "kernels_table_avx2"):       # tables: scalar and vector lookups must round identically
            extra = ["-ffp-contract=off"] + (["-mavx2", "-mfma"] if s == "kernels_table_avx2" else [])
        else:
            extra = ["-ffp-contract=fast"] if s.startswith("kernels") else []
            if s == "kernels_avx2":
                extra += ["-mavx2", "-mfma"]
        o = os.path.join(obj, s + ".o")
        zig("c++", *common, *extra, "-c", os.path.join("src", s + ".cpp"), "-o", o)
        objs.append(o)
    out_dir = os.path.join(ROOT, "dist", name)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, libname)
    link = ["c++", "-target", target, "-shared", *objs, "-o", out]
    if "linux" in name:
        link += ["-Wl,--version-script=src/exports.map", "-Wl,-soname," + libname, "-s"]
    else:
        link += ["-Wl,-exported_symbols_list,src/exports_macos.txt", "-install_name", "@rpath/" + libname]
    zig(*link)
    nat = os.path.join(ROOT, "python", "sldrnc", "_native", name)
    os.makedirs(nat, exist_ok=True)
    shutil.copy2(out, nat)
    print("%-12s %s  (%d KB)" % (name, os.path.relpath(out, ROOT), os.path.getsize(out) // 1024))


if __name__ == "__main__":
    for n in sys.argv[1:] or list(TARGETS):
        build(n)
