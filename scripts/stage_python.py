"""stage_python.py -- copy the prebuilt libraries from dist/<platform>/ into the Python package
(python/sldrnc/_native/<platform>/) so `pip install ./python` ships them.

  python scripts/stage_python.py
"""
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIST = os.path.join(ROOT, "dist")
NATIVE = os.path.join(ROOT, "python", "sldrnc", "_native")

for plat in sorted(os.listdir(DIST)):
    src = os.path.join(DIST, plat)
    for name in ("sldrnc.dll", "libsldrnc.so", "libsldrnc.dylib"):
        p = os.path.join(src, name)
        if os.path.exists(p):
            os.makedirs(os.path.join(NATIVE, plat), exist_ok=True)
            shutil.copy2(p, os.path.join(NATIVE, plat, name))
            print("staged", plat, name)
