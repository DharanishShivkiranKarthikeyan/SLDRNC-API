"""hyq_delay.py -- round 12: self-correction tuned for the measurement delay, measured with the library on HyQ.

For every saved model of hyq_modes.py (FULL small / medium / large x 2 seeds, both training directions) and the
CORRECTION baseline (rebuilt from ONNX), re-tune the correction for delay 50 on the same validation chunk
(Model.tune_correction) and score at D = 50 against the delay-1 tuning; check the pre-registered bars (vnet/PLAN.md,
round 12); record the delay curve for FULL small; and retrain FULL small seed 0 to check that delay-1 training
still reproduces the saved round-11 model exactly.  Writes results/hyq_delay.json.

  python benchmarks/hyq_delay.py
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = sys.argv[:1]                     # hyq_modes parses sys.argv at import
sys.path.insert(0, HERE)
import hyq_modes as H  # noqa: E402
import sldrnc  # noqa: E402

RES = os.path.join(H.LIB, "results")
OUT = os.path.join(RES, "hyq_delay.json")
D = 50
# per-delay-tuned counterparts from rounds 9 / 10 (vnet/PLAN.md round 12)
COUNTER = {
    "lab2_test": {"small": [0.0381, 0.0383], "medium": [0.0366], "large": [0.0395, 0.0388], "correction": [0.0751]},
    "lab2_to_lab1": {"small": [0.048, 0.047], "medium": [0.0427, 0.046], "large": [0.0411, 0.0438], "correction": [0.097, 0.1021]},
    "lab1_to_lab2": {"small": [0.0319, 0.0293], "medium": [0.0269, 0.0269], "large": [0.0269, 0.0263], "correction": [0.0687, 0.0698]},
}


def split(rec_tr, rec_x):
    X2, Y2 = H.raw_rows(rec_tr)
    X1, Y1 = H.raw_rows(rec_x)
    T = len(X2)
    n, g = T // 10, 50
    return dict(X2=X2, Y2=Y2, X1=X1, Y1=Y1, tr=slice(g, 7 * n - g), va=slice(7 * n + g, 8 * n - g), te=slice(8 * n + g, T - g),
                te_from=(9 * n + g) - (8 * n + g), x1=slice(g, len(X1) - g))


def main():
    t_all = time.time()
    res = {"delay": D, "rows": [], "curve": [], "bars": {}}
    for direction, tag, rec_tr, rec_x, prep_tr, prep_x, keys in (
            ("lab2", "", "trot_in_lab_2", "trot_in_lab_1", "prep9.npz", "prep9_lab1_xfer.npz", ("ab_ref_s0", "ab_ref_s1")),
            ("lab1", "_lab1", "trot_in_lab_1", "trot_in_lab_2", "prep9_trot_in_lab_1.npz", "prep9_lab2_xfer.npz", ("L1_ref_s0", "L1_ref_s1"))):
        S = split(rec_tr, rec_x)
        test_name, other_name = ("lab2_test", "lab2_to_lab1") if direction == "lab2" else ("lab1_test", "lab1_to_lab2")
        models = []
        for size in ("small", "medium", "large"):
            for seed in (0, 1):
                m = sldrnc.Model.load(os.path.join(RES, "hyq_full_%s%s_s%d.sldm" % (size, tag, seed)))
                models.append((size, seed, m, S["X2"], S["X1"]))
        H.PREP_TR = prep_tr                                       # baseline_onnx reads the training prep file
        F2, F1 = H.baseline_inputs(prep_tr), H.baseline_inputs(prep_x)
        for seed, key in enumerate(keys):
            m = sldrnc.Model.from_onnx(H.baseline_onnx(key)[0])
            m.add_correction(F2[S["va"]], S["Y2"][S["va"]], rate_hz=1000, velocity_index=[12 + j for j in range(12)])
            models.append(("correction", seed, m, F2, F1))
        for size, seed, m1, Xtr_rec, Xx_rec in models:
            t0 = time.time()
            m50, rep = m1.tune_correction(Xtr_rec[S["va"]], S["Y2"][S["va"]], delay_ticks=D)
            row = {"direction": direction, "model": size, "seed": seed, "tau1_ms": m1.info.correction_time_ms, "tau50_ms": rep.correction_time_ms}
            for name, Xs, Ys, start in ((test_name, Xtr_rec[S["te"]], S["Y2"][S["te"]], S["te_from"]),
                                        (other_name, Xx_rec[S["x1"]], S["Y1"][S["x1"]], 2000)):
                row[name] = {"t1_d50": m1.run(Xs, Ys, D, start).error, "t50_d50": m50.run(Xs, Ys, D, start).error,
                             "t1_d1": m1.run(Xs, Ys, 1, start).error, "t50_d1": m50.run(Xs, Ys, 1, start).error}
            res["rows"].append(row)
            print("%s %-10s s%d tau %4.0f -> %4.0f ms | %s D50: %.4f -> %.4f | %s D50: %.4f -> %.4f   (%.0f s)" % (
                direction, size, seed, row["tau1_ms"], row["tau50_ms"], test_name, row[test_name]["t1_d50"], row[test_name]["t50_d50"],
                other_name, row[other_name]["t1_d50"], row[other_name]["t50_d50"], time.time() - t0), flush=True)
            with open(OUT, "w") as f:
                json.dump(res, f, indent=1)
        if direction == "lab2":
            # delay curve, FULL small, Lab 2 test
            for seed in (0, 1):
                m1 = [m for s_, sd, m, _, _ in models if s_ == "small" and sd == seed][0]
                Xs, Ys = S["X2"][S["te"]], S["Y2"][S["te"]]
                for d in (1, 5, 10, 20, 50, 100, 200):
                    md, rep = m1.tune_correction(S["X2"][S["va"]], S["Y2"][S["va"]], delay_ticks=d)
                    pt = {"seed": seed, "delay": d, "tuned_for_1": m1.run(Xs, Ys, d, S["te_from"]).error,
                          "tuned_for_delay": md.run(Xs, Ys, d, S["te_from"]).error, "tau_ms": rep.correction_time_ms}
                    res["curve"].append(pt)
                    print("  curve small s%d D=%3d: tuned for 1 %.4f | tuned for D %.4f (tau %.0f ms)" % (seed, d, pt["tuned_for_1"], pt["tuned_for_delay"], pt["tau_ms"]), flush=True)
            # B1: delay-1 training reproduces the saved round-11 model
            sc = H.schema()
            m_new, _ = sldrnc.train(sc, S["X2"][S["tr"]], S["Y2"][S["tr"]], S["X2"][S["va"]], S["Y2"][S["va"]], size="small", steps=H.STEPS, seed=0)
            m_old = sldrnc.Model.load(os.path.join(RES, "hyq_full_small_s0.sldm"))
            Xs, Ys = S["X2"][S["te"]], S["Y2"][S["te"]]
            same = bool(np.array_equal(m_new.run(Xs, Ys, 1, S["te_from"]).predictions, m_old.run(Xs, Ys, 1, S["te_from"]).predictions))
            res["bars"]["B1_retrain_identical"] = same
            print("B1: retrained FULL small s0 (delay 1) predicts identically to the saved round-11 model:", same, flush=True)
    # ---- bars
    b2, b3 = [], []
    for stretch, counter in COUNTER.items():
        for model, ref in counter.items():
            rows = [r for r in res["rows"] if r["model"] == model and stretch in r]
            lib = float(np.mean([r[stretch]["t50_d50"] for r in rows]))
            bar = 1.10 * float(np.mean(ref))
            b2.append({"stretch": stretch, "model": model, "library": lib, "counterpart": float(np.mean(ref)), "pass": lib <= bar})
    for r in res["rows"]:
        for stretch in [k for k in r if isinstance(r[k], dict)]:
            v = r[stretch]
            b3.append({"direction": r["direction"], "model": r["model"], "seed": r["seed"], "stretch": stretch, "ratio": v["t50_d50"] / v["t1_d50"],
                       "pass": v["t50_d50"] <= 1.02 * v["t1_d50"]})
    res["bars"]["B2"] = b2
    res["bars"]["B3"] = b3
    print("\nB2 (tuned for 50, D = 50, two-seed mean <= 1.10 x the round 9/10 per-delay counterpart):")
    for x in b2:
        print("  %-13s %-10s library %.4f vs %.4f  -> %s" % (x["stretch"], x["model"], x["library"], x["counterpart"], "PASS" if x["pass"] else "FAIL"))
    print("B3 (tuned for 50 never > 2 %% worse than tuned for 1 at D = 50): %d / %d pass; worst ratio %.3f" % (
        sum(x["pass"] for x in b3), len(b3), max(x["ratio"] for x in b3)))
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    print("done in %.0f s" % (time.time() - t_all))


if __name__ == "__main__":
    main()
