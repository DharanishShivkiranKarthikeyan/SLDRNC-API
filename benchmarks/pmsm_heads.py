"""pmsm_heads.py -- round 15 (vnet/PLAN.md), bars H2 and T1: an SLD-RNC model for the PMSM motor drive, and its table.

The model: FULL, history off (the controller is a memoryless map), outputs affine in the electrical speed and the two
current references, the network seeing the currents i_d, i_q and the rotor angle (ANGLE) -- the research build's
"heads" form, now built by the library.  Trained on the research training set (vnet/killA/census.pmsm_train_data: 4
simulated runs + operating-box samples; validation: 60k box samples) at the research budget (24,000 steps, batch 2,048,
no weight decay).  Scored like round 13: W2 = the recorded runs 20-23 (rms volts against ctrl_exact), SIM = 4 simulated
5-s runs after 2 s.  Each model is also compiled into a 32 x 32 x 64 table and scored the same way.

  python benchmarks/pmsm_heads.py [steps=24000] [sizes=small,medium,large]
Writes results/pmsm_heads.json, results/pmsm_heads_*.sldm and results/pmsm_heads_stream.bin (W2 run 20, for latency).
"""
import json
import os
import struct
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.dirname(HERE)
REPO = os.path.dirname(LIB)
sys.path.insert(0, os.path.join(LIB, "python"))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "vnet", "killA"))
import pmsm_plant as P  # noqa: E402
import sldrnc  # noqa: E402

kw = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
STEPS = int(kw.get("steps", 24000))
SIZES = kw.get("sizes", "small,medium,large").split(",")
RES = os.path.join(LIB, "results")
OUT = os.path.join(RES, "pmsm_heads.json")
WARM = 40000


def rows(Xn):
    """controller input (normalised x / X_SCALE, 7 columns) -> i_d, i_q, angle (rad), speed, i_d_ref, i_q_ref"""
    th = np.arctan2(Xn[:, 5], Xn[:, 6])
    return np.ascontiguousarray(np.stack([Xn[:, 0], Xn[:, 1], th, Xn[:, 2], Xn[:, 3], Xn[:, 4]], 1))


def rms(pred, true):
    return float(np.sqrt(np.mean((np.concatenate(pred) - np.concatenate(true)) ** 2)))


def write_stream(path, X, Y):
    with open(path, "wb") as f:
        f.write(struct.pack("<3i", len(X), X.shape[1], Y.shape[1]))
        f.write(np.ascontiguousarray(X, np.float64).tobytes())
        f.write(np.ascontiguousarray(Y, np.float64).tobytes())


def main():
    t_all = time.time()
    import census as C
    Xm, Ym, Xv, Yv = C.pmsm_train_data()
    Xtr, Ytr = rows(Xm), np.ascontiguousarray(Ym * P.Y_SCALE)
    Xva, Yva = rows(Xv), np.ascontiguousarray(Yv * P.Y_SCALE)
    z = np.load(os.path.join(REPO, "results", "pmsm_ssm_data_hs.npz"))
    W2X = [rows(z["U%d" % j][:, :7].astype(np.float64)) for j in range(20, 24)]
    W2Y = [P.ctrl_exact(z["U%d" % j][:, :7].astype(np.float64) * P.X_SCALE) for j in range(20, 24)]
    sim = np.load(os.path.join(RES, "pmsm_sim_runs.npz"))
    SX = [rows(sim["X%d" % s][WARM:]) for s in (300, 301, 302, 303)]
    SY = [sim["V%d" % s][WARM:] for s in (300, 301, 302, 303)]
    print("training set %d rows, validation %d" % (len(Xtr), len(Xva)), flush=True)

    schema = sldrnc.Schema(rate_hz=20000, history=False)
    schema.add("i_dq", sldrnc.Input.GENERIC, 2).add("angle", sldrnc.Input.ANGLE, 1)
    schema.add("speed", sldrnc.Input.VELOCITY, 1).add("i_dq_ref", sldrnc.Input.GENERIC, 2)
    schema.outputs(2).affine("speed", "i_dq_ref")
    res = {"steps": STEPS, "batch": 2048, "rows": []}
    for size in SIZES:
        for seed in (0, 1):
            t0 = time.time()
            m, rep = sldrnc.train(schema, Xtr, Ytr, Xva, Yva, size=size, steps=STEPS, batch=2048, weight_decay=0.0, seed=seed,
                                  fit_correction=False)
            row = {"size": size, "seed": seed, "seconds": rep.seconds, "macs": m.info.macs_per_tick, "val": rep.validation_error,
                   "w2_volts": rms([m.run(x).predictions for x in W2X], W2Y), "sim_volts": rms([m.run(x).predictions for x in SX], SY)}
            m.save(os.path.join(RES, "pmsm_heads_%s_s%d.sldm" % (size, seed)))
            tab = m.compile_table([32, 32, 64])
            row["table_w2_volts"] = rms([tab.run(x).predictions for x in W2X], W2Y)
            row["table_sim_volts"] = rms([tab.run(x).predictions for x in SX], SY)
            row["table_bytes"] = tab.info.table_bytes
            tab.save(os.path.join(RES, "pmsm_heads_%s_s%d_table.sldm" % (size, seed)))
            res["rows"].append(row)
            print("heads %-6s seed %d: W2 %.4f V, SIM %.4f V | table W2 %.4f V, SIM %.4f V | val %.5f | %.0f s, %d MACs" % (
                size, seed, row["w2_volts"], row["sim_volts"], row["table_w2_volts"], row["table_sim_volts"], row["val"], time.time() - t0,
                row["macs"]), flush=True)
            json.dump(res, open(OUT, "w"), indent=1)
    write_stream(os.path.join(RES, "pmsm_heads_stream.bin"), W2X[0], W2Y[0])
    print("done in %.0f s" % (time.time() - t_all))


if __name__ == "__main__":
    main()
