"""regress_all.py -- replay every saved model (results/*.sldm) on its recorded stream with whatever library is loaded,
for a bit-for-bit comparison between two builds.

  SLDRNC_LIBRARY=<old dll> python benchmarks/regress_all.py old.npz
  SLDRNC_LIBRARY=<new dll> python benchmarks/regress_all.py new.npz old.npz     # also compares, exactly

Each model runs 4,000 ticks of its stream: run() with measurements (correction on), run() without correction, a
session step/observe replay, and a fleet of 7 robots (each its own offset into the stream).
"""
import glob
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(LIB, "python"))
import sldrnc  # noqa: E402

RES = os.path.join(LIB, "results")
STREAMS = ["hyq_stream_raw.bin", "hyq_stream_feat.bin", "sarcos_stream.bin", "pmsm_stream.bin", "pmsm_heads_stream.bin"]
T = 4000


def read_stream(path):
    with open(path, "rb") as f:
        n, ni, no = struct.unpack("<3i", f.read(12))
        X = np.frombuffer(f.read(8 * n * ni), np.float64).reshape(n, ni)
        Y = np.frombuffer(f.read(8 * n * no), np.float64).reshape(n, no)
    return X, Y


def main():
    streams = {s: read_stream(os.path.join(RES, s)) for s in STREAMS if os.path.exists(os.path.join(RES, s))}
    out = {}
    for path in sorted(glob.glob(os.path.join(RES, "*.sldm"))):
        name = os.path.basename(path)[:-5]
        m = sldrnc.Model.load(path)
        fam = name.split("_")[0]
        cand = [s for s in streams if s.startswith(fam) and streams[s][0].shape[1] == m.info.input_size and streams[s][1].shape[1] == m.info.output_size]
        if not cand:
            print("%-44s no matching stream, skipped" % name)
            continue
        X, Y = streams[cand[0]]
        X, Y = np.ascontiguousarray(X[:T]), np.ascontiguousarray(Y[:T])
        out[name + "|run"] = m.run(X, Y).predictions
        out[name + "|nocorr"] = m.run(X, Y, correction=False).predictions
        s = m.session()
        P = np.empty((T, m.info.output_size))
        for t in range(T):
            s.step_into(X[t], P[t])
            s.observe(Y[t])
        out[name + "|session"] = P
        fl = m.fleet(7, threads=0)
        off = np.arange(7) * 311
        F = np.empty((600, 7, m.info.output_size))
        for t in range(600):
            F[t] = fl.step(X[off + t])
            fl.observe(Y[off + t])
        out[name + "|fleet"] = F
        print("%-44s %s" % (name, cand[0]), flush=True)
    np.savez(sys.argv[1], **out)
    if len(sys.argv) > 2:
        ref = np.load(sys.argv[2])
        diff = [k for k in out if k not in ref or not np.array_equal(out[k], ref[k], equal_nan=True)]
        print("%d arrays compared: %s" % (len(out), "IDENTICAL" if not diff else "DIFFERENT: " + ", ".join(diff[:20])))


if __name__ == "__main__":
    main()
