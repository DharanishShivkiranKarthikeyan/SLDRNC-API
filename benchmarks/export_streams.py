"""export_streams.py -- write the HyQ test stretch for benchmarks/latency.cpp, and save the CORRECTION model.
  results/hyq_stream_raw.bin   raw signals (FULL models' input rows) + measured efforts
  results/hyq_stream_feat.bin  the baseline network's own input rows + measured efforts (RAW / CORRECTION models)
  results/hyq_correction_s0.sldm, results/hyq_raw_s0.sldm
"""
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(LIB, "python"))
import hyq_modes as H  # noqa: E402
import sldrnc  # noqa: E402


def write(path, X, Y):
    with open(path, "wb") as f:
        f.write(struct.pack("<iii", len(X), X.shape[1], Y.shape[1]))
        f.write(np.ascontiguousarray(X, "<f8").tobytes())
        f.write(np.ascontiguousarray(Y, "<f8").tobytes())


X2, Y2 = H.raw_rows("trot_in_lab_2")
T = len(X2)
n, g = T // 10, 50
te = slice(8 * n + g, T - g)
va = slice(7 * n + g, 8 * n - g)
out = os.path.join(LIB, "results")
os.makedirs(out, exist_ok=True)
write(os.path.join(out, "hyq_stream_raw.bin"), X2[te], Y2[te])
F2 = H.baseline_inputs("prep9.npz")
write(os.path.join(out, "hyq_stream_feat.bin"), F2[te], Y2[te])
ob, _ = H.baseline_onnx("ab_ref_s0")
raw = sldrnc.Model.from_onnx(ob)
raw.save(os.path.join(out, "hyq_raw_s0.sldm"))
raw.add_correction(F2[va], Y2[va], rate_hz=1000, velocity_index=[12 + j for j in range(12)])
raw.save(os.path.join(out, "hyq_correction_s0.sldm"))
print("written", len(X2[te]), "ticks")
