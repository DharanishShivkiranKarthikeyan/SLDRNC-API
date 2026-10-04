"""pmsm_library.py -- round 13 (vnet/PLAN.md): the PMSM motor drive (20 kHz current control) measured with the library.

Task (vnet W2): predict the exact current controller's voltages (v_d, v_q) from its 7 normalised inputs (i_d, i_q,
electrical speed, i_d_ref, i_q_ref, sin and cos of the rotor angle).  Error = rms volts against ctrl_exact.

  RAW   round 2 / K3's converged dense w512 (seeds 0, 1), imported with from_layers; each must reproduce its research
        W2 error (bar M1) before anything else is reported
  FULL  sldrnc.train on closed-loop runs from the research simulator (scripts/pmsm_patch_probe.scenario, research
        training settings): 6 runs of 5 s for training (seeds 202-207), 2 for validation (208, 209); small / medium /
        large x seeds 0, 1; no self-correction (a controller has no measurement of its own output)
  scored  (a) W2: the recorded 1-s runs 20-23, each from a cold session, all 80,000 ticks (the research metric)
          (b) SIM: 4 simulated 5-s runs (seeds 300-303) after the 2-s warm-up

  python benchmarks/pmsm_library.py [steps=12000]      (or: simulate -- only generate the cached runs)
Simulated runs are cached in results/pmsm_sim_runs.npz.  Writes results/pmsm_library.json, results/pmsm_*.sldm and
results/pmsm_stream.bin (W2 run 20, for benchmarks/latency.cpp).
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
import pmsm_plant as P  # noqa: E402
import sldrnc  # noqa: E402

kw = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
STEPS = int(kw.get("steps", 12000))
RES = os.path.join(LIB, "results")
OUT = os.path.join(RES, "pmsm_library.json")
CACHE = os.path.join(RES, "pmsm_sim_runs.npz")
RATE, RUN_S, WARM = 20000.0, 5.0, 40000
TRAIN_SEEDS, VAL_SEEDS, TEST_SEEDS = (202, 203, 204, 205, 206, 207), (208, 209), (300, 301, 302, 303)
RESEARCH_W512 = {0: 1.7506, 1: 1.7113}            # vnet/round2/K3_baselines/results.txt, W2 rms volts per seed


def write_stream(path, X, Y):
    with open(path, "wb") as f:
        f.write(struct.pack("<3i", len(X), X.shape[1], Y.shape[1]))
        f.write(np.ascontiguousarray(X, np.float64).tobytes())
        f.write(np.ascontiguousarray(Y, np.float64).tobytes())


def sim_runs():
    """closed-loop runs from the research simulator: normalised controller inputs and the exact controller's volts"""
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        return {int(k[1:]): (z[k], z["V" + k[1:]]) for k in z.files if k.startswith("X")}
    import pmsm_patch_probe as Q
    out, store = {}, {}
    for sd in TRAIN_SEEDS + VAL_SEEDS + TEST_SEEDS:
        t0 = time.time()
        sc = Q.scenario(RATE, RUN_S, 400.0, 300, seed=sd)
        out[sd] = (sc["X"] / P.X_SCALE, sc["V"])
        store["X%d" % sd], store["V%d" % sd] = out[sd]
        print("  simulated run %d (%.0f s)" % (sd, time.time() - t0), flush=True)
    np.savez(CACHE, **store)
    return out


def rms(pred, true):
    return float(np.sqrt(np.mean((np.concatenate(pred) - np.concatenate(true)) ** 2)))


RESEARCH_SIZES = {(64, 0): 2.6468, (64, 1): 2.7324, (128, 0): 1.9645, (128, 1): 1.9970, (256, 0): 1.8422, (256, 1): 1.9066}


def raw_sizes():
    """(added after the pre-registration) RAW imports of K3's smaller converged nets: a speed / accuracy menu for RAW
    mode.  Each is checked against its research W2 error like M1.  Appends to results/pmsm_library.json."""
    z = np.load(os.path.join(REPO, "results", "pmsm_ssm_data_hs.npz"))
    W2X = [z["U%d" % j][:, :7].astype(np.float64) for j in range(20, 24)]
    W2Y = [P.ctrl_exact(x * P.X_SCALE) for x in W2X]
    n = np.load(os.path.join(REPO, "vnet", "round2", "K3_baselines", "nets_4x_b.npz"), allow_pickle=True)
    res = json.load(open(OUT))
    res["raw_sizes"] = []
    for (w, rs), ref in RESEARCH_SIZES.items():
        key = "relu2_w%d_s%d_4x" % (w, rs)
        Wk = [n["%s|W%d" % (key, i)] for i in range(3)]
        bk = [n["%s|b%d" % (key, i)] for i in range(3)]
        m = sldrnc.Model.from_layers([sldrnc.Dense(Wk[0], bk[0], "relu2"), sldrnc.Dense(Wk[1], bk[1], "relu2"), sldrnc.Dense(Wk[2], bk[2])],
                                     output_mean=np.zeros(2), output_std=np.asarray(P.Y_SCALE, np.float64))
        e = rms([m.run(x).predictions for x in W2X], W2Y)
        ok = abs(e - ref) <= 0.01 * ref
        res["raw_sizes"].append({"width": w, "seed": rs, "w2_volts": e, "research": ref, "match": bool(ok), "macs": m.info.macs_per_tick})
        print("RAW w%-3d seed %d: W2 %.4f V (research %.4f) %s, %d MACs" % (w, rs, e, ref, "match" if ok else "MISMATCH", m.info.macs_per_tick), flush=True)
        if rs == 0:
            m.save(os.path.join(RES, "pmsm_raw_w%d_s0.sldm" % w))
    json.dump(res, open(OUT, "w"), indent=1)


def main():
    t_all = time.time()
    if "simulate" in sys.argv[1:]:          # only generate (and cache) the simulated runs
        sim_runs()
        return
    if "raw_sizes" in sys.argv[1:]:
        raw_sizes()
        return
    z = np.load(os.path.join(REPO, "results", "pmsm_ssm_data_hs.npz"))
    W2X = [z["U%d" % j][:, :7].astype(np.float64) for j in range(20, 24)]
    W2Y = [P.ctrl_exact(x * P.X_SCALE) for x in W2X]                # volts
    runs = sim_runs()
    Xtr, Ytr = [runs[s][0] for s in TRAIN_SEEDS], [runs[s][1] for s in TRAIN_SEEDS]
    Xva, Yva = [runs[s][0] for s in VAL_SEEDS], [runs[s][1] for s in VAL_SEEDS]
    simX, simY = [runs[s][0] for s in TEST_SEEDS], [runs[s][1] for s in TEST_SEEDS]
    res = {"steps": STEPS, "rows": [], "raw_w512": []}

    def score(model, cold_runs=True):
        a = rms([model.run(x).predictions for x in W2X], W2Y)
        b = rms([model.run(x).predictions[WARM:] for x in simX], [y[WARM:] for y in simY])
        return a, b

    # ---------------- RAW: K3's converged w512 (bar M1)
    for rs, f in ((0, "nets_4x_a.npz"), (1, "nets_4x_c.npz")):
        n = np.load(os.path.join(REPO, "vnet", "round2", "K3_baselines", f), allow_pickle=True)
        key = "relu2_w512_s%d_4x" % rs
        Wk = [n["%s|W%d" % (key, i)] for i in range(3)]
        bk = [n["%s|b%d" % (key, i)] for i in range(3)]
        raw = sldrnc.Model.from_layers([sldrnc.Dense(Wk[0], bk[0], "relu2"), sldrnc.Dense(Wk[1], bk[1], "relu2"), sldrnc.Dense(Wk[2], bk[2])],
                                       output_mean=np.zeros(2), output_std=np.asarray(P.Y_SCALE, np.float64))
        e_w2, e_sim = score(raw)
        ref = RESEARCH_W512[rs]
        m1 = abs(e_w2 - ref) <= 0.01 * ref
        res["raw_w512"].append({"seed": rs, "w2_volts": e_w2, "sim_volts": e_sim, "research": ref, "M1_pass": bool(m1), "macs": raw.info.macs_per_tick})
        print("RAW w512 seed %d: W2 %.4f V (research %.4f) | SIM %.4f V -> M1 %s" % (rs, e_w2, ref, e_sim, "PASS" if m1 else "FAIL"), flush=True)
        raw.save(os.path.join(RES, "pmsm_raw_w512_s%d.sldm" % rs))
        if not m1:
            json.dump(res, open(OUT, "w"), indent=1)
            sys.exit("M1 failed: the RAW import does not reproduce the research error; stopping (protocol mismatch)")

    # ---------------- FULL
    schema = sldrnc.Schema(rate_hz=RATE)
    schema.add("i_dq", sldrnc.Input.GENERIC, 2).add("speed", sldrnc.Input.VELOCITY, 1)
    schema.add("i_dq_ref", sldrnc.Input.GENERIC, 2).add("angle_sin_cos", sldrnc.Input.GENERIC, 2)
    schema.outputs(2)
    for size in ("small", "medium", "large"):
        for seed in (0, 1):
            t0 = time.time()
            model, rep = sldrnc.train(schema, Xtr, Ytr, Xva, Yva, size=size, steps=STEPS, seed=seed, fit_correction=False)
            e_w2, e_sim = score(model)
            row = {"size": size, "seed": seed, "w2_volts": e_w2, "sim_volts": e_sim, "val": rep.validation_error, "seconds": rep.seconds,
                   "macs": rep.macs_per_tick}
            res["rows"].append(row)
            model.save(os.path.join(RES, "pmsm_full_%s_s%d.sldm" % (size, seed)))
            print("FULL %-6s seed %d: W2 %.4f V (cold 1-s runs) | SIM %.4f V (after warm-up) | val %.4f | %.0f s, %d MACs" % (
                size, seed, e_w2, e_sim, rep.validation_error, time.time() - t0, row["macs"]), flush=True)
            json.dump(res, open(OUT, "w"), indent=1)
    write_stream(os.path.join(RES, "pmsm_stream.bin"), W2X[0], W2Y[0])
    print("done in %.0f s" % (time.time() - t_all))


if __name__ == "__main__":
    main()
