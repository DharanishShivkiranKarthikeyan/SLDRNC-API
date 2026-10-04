"""regress_saved.py -- score every saved HyQ model with whatever library is loaded, to check that a library change
leaves results unchanged bit for bit.

  python benchmarks/regress_saved.py out.json              # record (set SLDRNC_LIBRARY to pick a build)
  python benchmarks/regress_saved.py out.json ref.json     # record and compare exactly with ref.json

Scores, for each saved model (results/hyq_*.sldm, from hyq_modes.py / export_streams.py): run() at D = 1 with and
without correction and protection, run() at D = 50, and one session-driven replay (step / observe from Python).
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv, ARGS = sys.argv[:1], sys.argv[1:]      # hyq_modes parses sys.argv at import
sys.path.insert(0, HERE)
import hyq_modes as H  # noqa: E402
import sldrnc  # noqa: E402

RES = os.path.join(H.LIB, "results")


def stretches(rec_tr, rec_x):
    X2, Y2 = H.raw_rows(rec_tr)
    X1, Y1 = H.raw_rows(rec_x)
    T = len(X2)
    n, g = T // 10, 50
    te = slice(8 * n + g, T - g)
    return (X2[te], Y2[te], (9 * n + g) - (8 * n + g)), (X1[g:len(X1) - g], Y1[g:len(Y1) - g], 2000), te, slice(g, len(X1) - g)


def score(model, Xs, Ys, start):
    out = {}
    if model.info.has_correction:
        out["d1"] = model.run(Xs, Ys, 1, start).error
        out["d1_unprotected"] = model.run(Xs, Ys, 1, start, protect=False).error
        out["d50"] = model.run(Xs, Ys, 50, start).error
    out["uncorrected"] = model.run(Xs, Ys, 1, start, correction=False).error
    return out


def main():
    res = {"library": sldrnc._capi.path if hasattr(sldrnc._capi, "path") else "", "rows": {}}
    for tag, rec_tr, rec_x, prep_tr, prep_x in (("", "trot_in_lab_2", "trot_in_lab_1", "prep9.npz", "prep9_lab1_xfer.npz"),
                                                 ("_lab1", "trot_in_lab_1", "trot_in_lab_2", "prep9_trot_in_lab_1.npz", "prep9_lab2_xfer.npz")):
        (Xt, Yt, st), (Xo, Yo, so), te, xo = stretches(rec_tr, rec_x)
        for size in ("small", "medium", "large"):
            for seed in (0, 1):
                name = "hyq_full_%s%s_s%d" % (size, tag, seed)
                m = sldrnc.Model.load(os.path.join(RES, name + ".sldm"))
                res["rows"][name] = {"test": score(m, Xt, Yt, st), "other": score(m, Xo, Yo, so)}
                print(name, res["rows"][name]["test"], flush=True)
        if not tag:
            F2, F1 = H.baseline_inputs(prep_tr), H.baseline_inputs(prep_x)
            for name in ("hyq_raw_s0", "hyq_correction_s0"):
                m = sldrnc.Model.load(os.path.join(RES, name + ".sldm"))
                res["rows"][name] = {"test": score(m, F2[te], Yt, st), "other": score(m, F1[xo], Yo, so)}
                print(name, res["rows"][name]["test"], flush=True)
            # session-driven replay (step / observe from Python), FULL small seed 0, Lab 2 test
            m = sldrnc.Model.load(os.path.join(RES, "hyq_full_small_s0.sldm"))
            s = m.session()
            P = np.empty_like(Yt)
            for t in range(len(Xt)):
                P[t] = s.step(Xt[t])
                s.observe(Yt[t])
            res["rows"]["session_small_s0"] = {"sum": float(P.sum()), "first": P[st].tolist(), "last": P[-1].tolist()}
    with open(ARGS[0], "w") as f:
        json.dump(res, f, indent=1)
    if len(ARGS) > 1:
        ref = json.load(open(ARGS[1]))["rows"]
        bad = 0

        def walk(a, b, path):
            nonlocal bad
            if isinstance(a, dict):
                for k in b:
                    walk(a.get(k), b[k], path + "/" + k)
            elif a != b:
                bad += 1
                print("DIFFERS", path, a, b)
        walk(res["rows"], ref, "")
        print("compared with %s: %s" % (ARGS[1], "IDENTICAL" if bad == 0 else "%d values differ" % bad))
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
