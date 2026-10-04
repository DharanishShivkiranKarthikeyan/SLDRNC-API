"""sarcos_library.py -- round 13 (vnet/PLAN.md): SARCOS inverse dynamics measured with the library.

Protocol = rounds 2 / 6 (vnet/round2/S4_sarcos): one 100 Hz stream of 44,484 rows, 8 validation and 8 test blocks of
556 rows with 100-row guards.  BLK100 (primary) = per-joint MSE / variance over the 8 x 556 test-block rows, mean over
joints.  The library trains on the stream minus the blocks and guards, as 17 separate recordings; each test block is
replayed from 300 rows before it (the first 200 rows of any replay are warm-up) and only its own rows are scored.

  RAW   round 2's dense w512 on raw inputs, seeds 0 and 1 (the whitepaper's "standard network"), imported with
        from_layers; each must reproduce its own research error (bar M1) before anything else is reported
  FULL  sldrnc.train, sizes small / medium / large x seeds 0, 1; scored without self-correction (the benchmark
        convention) and with it (measurement of the previous tick)

  python benchmarks/sarcos_library.py [steps=12000] [angle=1] [affine=1]
      angle=1: q declared as ANGLE (round 14); affine=1: the torques affine in qdd (round 15)
Writes results/sarcos_library.json, results/sarcos_*.sldm and results/sarcos_stream.bin (for benchmarks/latency.cpp).
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
sys.path.insert(0, os.path.join(REPO, "vnet", "round2", "S4_sarcos"))
import sarcos as S4  # noqa: E402
import sldrnc  # noqa: E402

kw = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
STEPS = int(kw.get("steps", 12000))
ANGLE = kw.get("angle", "0") == "1"
AFFINE = kw.get("affine", "0") == "1"           # round 15: torques affine in the measured accelerations
TAG = ("angle_" if ANGLE else "") + ("affine_" if AFFINE else "")
RES = os.path.join(LIB, "results")
OUT = os.path.join(RES, "sarcos_library%s.json" % ("_" + TAG.rstrip("_") if TAG else ""))
NQ, BLOCK, GUARD, CTX = 7, S4.BLOCK, S4.GUARD, 300
RESEARCH_W512 = {0: 0.0165, 1: 0.0134}   # round 2's dense w512 per seed, BLK100 (vnet/round2/S4_sarcos/results.txt)


def write_stream(path, X, Y):
    with open(path, "wb") as f:
        f.write(struct.pack("<3i", len(X), X.shape[1], Y.shape[1]))
        f.write(np.ascontiguousarray(X, np.float64).tobytes())
        f.write(np.ascontiguousarray(Y, np.float64).tobytes())


def blk100(pred_blocks, true_blocks):
    """vnet/round6/id6.nmse: per-joint MSE over the pooled block rows / variance of those rows, mean over joints."""
    yh, y = np.concatenate(pred_blocks), np.concatenate(true_blocks)
    return float(np.mean(np.mean((yh - y) ** 2, 0) / y.var(0)))


def correction_mode():
    """CORRECTION (added after the pre-registration, labelled so): the imported w512 + self-correction tuned on the
    validation recordings (joined; RAW models have no history), scored like FULL (replay from 300 rows before each
    test block).  Appends to results/sarcos_library.json."""
    sp = S4.load_split()
    St, _ = S4.raw()
    X, Y = np.ascontiguousarray(St[:, :3 * NQ]), np.ascontiguousarray(St[:, 3 * NQ:])
    Xva = np.concatenate([X[v - CTX:v + BLOCK] for v in sp.val_blocks])
    Yva = np.concatenate([Y[v - CTX:v + BLOCK] for v in sp.val_blocks])
    true_blocks = [Y[t:t + BLOCK] for t in sp.test_blocks]
    res = json.load(open(OUT))
    res["correction_w512"] = []
    for rs in (0, 1):
        m = sldrnc.Model.load(os.path.join(RES, "sarcos_raw_w512_s%d.sldm" % rs))
        rep = m.add_correction(Xva, Yva, rate_hz=100, velocity_index=list(range(NQ, 2 * NQ)))
        e = blk100([m.run(X[t - CTX:t + BLOCK], Y[t - CTX:t + BLOCK], 1, CTX).predictions[CTX:] for t in sp.test_blocks], true_blocks)
        res["correction_w512"].append({"seed": rs, "blk100_corrected": e, "tau_ms": rep.correction_time_ms})
        print("CORRECTION w512 seed %d: BLK100 %.5f with self-correction (tau %.0f ms)" % (rs, e, rep.correction_time_ms), flush=True)
    json.dump(res, open(OUT, "w"), indent=1)


def main():
    t_all = time.time()
    if "correction" in sys.argv[1:]:
        correction_mode()
        return
    sp = S4.load_split()
    St, _ = S4.raw()
    X, Y = np.ascontiguousarray(St[:, :3 * NQ]), np.ascontiguousarray(St[:, 3 * NQ:])
    n = len(X)
    held = np.zeros(n, bool)                                     # blocks + guards, never trained on
    for b in list(sp.test_blocks) + list(sp.val_blocks):
        held[max(0, b - GUARD):min(n, b + BLOCK + GUARD)] = True
    edges = np.flatnonzero(np.diff(np.concatenate([[1], held.astype(int), [1]])))
    segs = [(edges[i], edges[i + 1]) for i in range(0, len(edges), 2)]     # runs of not-held rows
    Xtr, Ytr = [X[a:b] for a, b in segs], [Y[a:b] for a, b in segs]
    Xva = [X[v - CTX:v + BLOCK] for v in sp.val_blocks]
    Yva = [Y[v - CTX:v + BLOCK] for v in sp.val_blocks]
    true_blocks = [Y[t:t + BLOCK] for t in sp.test_blocks]
    res = {"steps": STEPS, "train_recordings": len(segs), "train_rows": int(sum(b - a for a, b in segs)), "rows": []}
    print("SARCOS: %d training recordings, %d rows; %d validation, %d test blocks" % (len(segs), res["train_rows"], len(Xva), len(true_blocks)))

    # ---------------- RAW: round 2's w512 (both seeds), imported (bar M1)
    res["raw_w512"] = []
    for rs in (0, 1):
        z = np.load(os.path.join(REPO, "vnet", "round2", "S4_sarcos", "ckpt_relu2_w512_s%d.npz" % rs), allow_pickle=True)
        Wk = [z["relu2_w512_s%d|W%d" % (rs, i)] for i in range(3)]
        bk = [z["relu2_w512_s%d|b%d" % (rs, i)] for i in range(3)]
        raw = sldrnc.Model.from_layers([sldrnc.Dense(Wk[0], bk[0], "relu2"), sldrnc.Dense(Wk[1], bk[1], "relu2"), sldrnc.Dense(Wk[2], bk[2])],
                                       input_mean=sp.mu, input_std=sp.sd, output_mean=sp.ym, output_std=sp.ys)

        def numpy_w512(Xb):
            h = (Xb - sp.mu) / sp.sd
            for i in range(3):
                h = h @ Wk[i] + bk[i]
                if i < 2:
                    h = np.maximum(h, 0) ** 2
            return h * sp.ys + sp.ym
        e_raw = blk100([raw.run(X[t:t + BLOCK]).predictions for t in sp.test_blocks], true_blocks)
        e_np = blk100([numpy_w512(X[t:t + BLOCK]) for t in sp.test_blocks], true_blocks)
        ref = RESEARCH_W512[rs]
        m1 = abs(e_raw - ref) <= 0.01 * ref + 5e-5                       # research values are quoted to 3 significant digits
        res["raw_w512"].append({"seed": rs, "blk100": e_raw, "numpy_blk100": e_np, "research": ref, "M1_pass": bool(m1),
                                "macs": raw.info.macs_per_tick})
        print("RAW w512 seed %d: BLK100 %.5f (numpy %.5f, research %.4f) -> M1 %s" % (rs, e_raw, e_np, ref, "PASS" if m1 else "FAIL"), flush=True)
        raw.save(os.path.join(RES, "sarcos_raw_w512_s%d.sldm" % rs))
        if not m1:
            json.dump(res, open(OUT, "w"), indent=1)
            sys.exit("M1 failed: the RAW import does not reproduce the research error; stopping (protocol mismatch)")

    # ---------------- FULL
    schema = sldrnc.Schema(rate_hz=100)
    schema.add("q", sldrnc.Input.ANGLE if ANGLE else sldrnc.Input.POSITION, NQ)
    schema.add("qd", sldrnc.Input.VELOCITY, NQ).add("qdd", sldrnc.Input.ACCELERATION, NQ)
    schema.outputs(NQ, velocity="qd")
    if AFFINE:
        schema.affine("qdd")
    for size in ("small", "medium", "large"):
        for seed in (0, 1):
            t0 = time.time()
            model, rep = sldrnc.train(schema, Xtr, Ytr, Xva, Yva, size=size, steps=STEPS, seed=seed)
            row = {"size": size, "seed": seed, "seconds": rep.seconds, "macs": rep.macs_per_tick, "val": rep.validation_error,
                   "val_corrected": rep.validation_error_corrected, "tau_ms": rep.correction_time_ms}
            for tag, corr in (("blk100", False), ("blk100_corrected", True)):
                preds = [model.run(X[t - CTX:t + BLOCK], Y[t - CTX:t + BLOCK], 1, CTX, correction=corr).predictions[CTX:] for t in sp.test_blocks]
                row[tag] = blk100(preds, true_blocks)
            res["rows"].append(row)
            model.save(os.path.join(RES, "sarcos_full_%s%s_s%d.sldm" % (TAG, size, seed)))
            print("FULL %-6s seed %d: BLK100 %.5f, with self-correction %.5f | val %.4f | %.0f s, %d MACs" % (
                size, seed, row["blk100"], row["blk100_corrected"], row["val"], time.time() - t0, row["macs"]), flush=True)
            json.dump(res, open(OUT, "w"), indent=1)
    write_stream(os.path.join(RES, "sarcos_stream.bin"), X[:20000], Y[:20000])
    print("done in %.0f s" % (time.time() - t_all))


if __name__ == "__main__":
    main()
