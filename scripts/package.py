"""package.py -- build the zips to ship: one per platform that has a library in dist/<platform>/.

  python scripts/package.py [platform ...]          ->  packages/sldrnc-<version>-<platform>.zip

Each zip holds: README.md, LICENSE.txt, LICENSE-BINARY.txt, api/ (the documentation), include/sldrnc/ (C and C++
headers), the library (Windows: bin/sldrnc.dll + lib/sldrnc.lib; Linux: lib/libsldrnc.so; macOS: lib/libsldrnc.dylib),
lib/cmake/sldrnc/ (find_package), python/ (pip-installable, with that platform's library inside) and examples/.
No source code of the library goes in.  Build the libraries first (scripts/build_windows.ps1, scripts/build_cross.py).
"""
import os
import re
import shutil
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIST = os.path.join(ROOT, "dist")
OUT = os.path.join(ROOT, "packages")
STAGE = os.path.join(ROOT, "build", "package")
PLATFORMS = {   # platform: (files in dist/<platform>/ -> path in the package, description)
    "windows-x64": ({"sldrnc.dll": "bin/sldrnc.dll", "sldrnc.lib": "lib/sldrnc.lib"}, "Windows 10 / 11, x86-64", "bin/"),
    "linux-x64": ({"libsldrnc.so": "lib/libsldrnc.so"}, "Linux x86-64, glibc 2.17 or later", "lib/"),
    "linux-arm64": ({"libsldrnc.so": "lib/libsldrnc.so"}, "Linux ARM64 (aarch64), glibc 2.17 or later", "lib/"),
    "macos-x64": ({"libsldrnc.dylib": "lib/libsldrnc.dylib"}, "macOS 11 or later, Intel", "lib/"),
    "macos-arm64": ({"libsldrnc.dylib": "lib/libsldrnc.dylib"}, "macOS 11 or later, Apple Silicon", "lib/"),
}


def version():
    h = open(os.path.join(ROOT, "include", "sldrnc", "sldrnc.h"), encoding="utf-8").read()
    return ".".join(re.search(r"#define SLDRNC_VERSION_%s (\d+)" % k, h).group(1) for k in ("MAJOR", "MINOR", "PATCH"))


def put(src, dst_rel, stage, text_subs=None):
    dst = os.path.join(stage, dst_rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if text_subs:
        s = open(src, encoding="utf-8").read()
        for k, v in text_subs.items():
            s = s.replace(k, v)
        open(dst, "w", encoding="utf-8", newline="\n").write(s)
    else:
        shutil.copy2(src, dst)


def build(plat, ver):
    files, long_name, libdir = PLATFORMS[plat]
    name = "sldrnc-%s-%s" % (ver, plat)
    stage = os.path.join(STAGE, name)
    shutil.rmtree(stage, ignore_errors=True)
    major = ver.split(".")[0]
    put(os.path.join(ROOT, "packaging", "README.md"), "README.md", stage,
        {"@VERSION@": ver, "@PLATFORM@": plat, "@PLATFORM_LONG@": long_name, "@LIBDIR@": "`%s`" % libdir})
    for f in ("LICENSE.txt", "LICENSE-BINARY.txt"):
        put(os.path.join(ROOT, f), f, stage)
    for f in sorted(os.listdir(os.path.join(ROOT, "api"))):
        put(os.path.join(ROOT, "api", f), "api/" + f, stage)
    for f in ("sldrnc.h", "sldrnc.hpp"):
        put(os.path.join(ROOT, "include", "sldrnc", f), "include/sldrnc/" + f, stage)
    for src, dst in files.items():
        p = os.path.join(DIST, plat, src)
        if not os.path.exists(p):
            sys.exit("missing %s: build the library for %s first" % (p, plat))
        put(p, dst, stage)
    put(os.path.join(ROOT, "packaging", "sldrncConfig.cmake"), "lib/cmake/sldrnc/sldrncConfig.cmake", stage)
    put(os.path.join(ROOT, "packaging", "sldrncConfigVersion.cmake.in"), "lib/cmake/sldrnc/sldrncConfigVersion.cmake", stage,
        {"@VERSION@": ver, "@MAJOR@": major})
    # Python package with this platform's library only
    put(os.path.join(ROOT, "python", "pyproject.toml"), "python/pyproject.toml", stage)
    put(os.path.join(ROOT, "python", "README.md"), "python/README.md", stage)
    for f in ("__init__.py", "_capi.py"):
        put(os.path.join(ROOT, "python", "sldrnc", f), "python/sldrnc/" + f, stage)
    lib = [s for s in files if not s.endswith(".lib")][0]
    put(os.path.join(DIST, plat, lib), "python/sldrnc/_native/%s/%s" % (plat, lib), stage)
    # examples
    put(os.path.join(ROOT, "examples", "cpp", "quickstart.cpp"), "examples/cpp/quickstart.cpp", stage)
    put(os.path.join(ROOT, "examples", "cpp", "CMakeLists.txt"), "examples/cpp/CMakeLists.txt", stage)
    put(os.path.join(ROOT, "examples", "python", "quickstart.py"), "examples/python/quickstart.py", stage)
    # zip, in a fixed order, everything under one top-level folder
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, name + ".zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for dirpath, dirnames, filenames in os.walk(stage):
            dirnames.sort()
            for f in sorted(filenames):
                p = os.path.join(dirpath, f)
                z.write(p, os.path.join(name, os.path.relpath(p, stage)).replace(os.sep, "/"))
    print("%-30s %5d KB" % (os.path.relpath(out, ROOT), os.path.getsize(out) // 1024))


def main():
    ver = version()
    plats = sys.argv[1:] or [p for p in PLATFORMS if os.path.isdir(os.path.join(DIST, p))]
    for p in plats:
        build(p, ver)
    for f in ("LICENSE.txt", "LICENSE-BINARY.txt"):
        s = open(os.path.join(ROOT, f), encoding="utf-8").read()
        if "[COPYRIGHT HOLDER]" in s or "[DRAFT" in s:
            print("WARNING: %s still has placeholders ([COPYRIGHT HOLDER] / [DRAFT ...]); fill them in before shipping" % f)


if __name__ == "__main__":
    main()
