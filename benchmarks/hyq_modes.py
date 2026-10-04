"""hyq_modes.py -- accuracy of the three SLD-RNC modes on the HyQ quadruped, measured with the library itself.

  RAW         "your" network = the dense baseline (w256 + 4-tick history) trained in vnet/round9, exported to ONNX and
              imported with Model.from_onnx (its inputs: that network's own features, computed outside SLD-RNC)
  CORRECTION  the same imported network + Model.add_correction tuned on the validation chunk
  FULL        sldrnc.train on raw signals (generic schema: joint positions / velocities, orientation, two IMUs)

Scored on Trot in Lab 2's held-out test chunk (models trained on chunks 0-6, validated on 7) and on the whole of
Trot in Lab 1 (never seen).  Needs the HyQ data (data/hyq/...) and the round-9 baseline weights; writes results/hyq_modes.json.

  python benchmarks/hyq_modes.py [steps=12000] [train=lab2|lab1] [angle=1]

angle=1 (round 14): the joint positions are declared as ANGLE (value, sine, cosine); only FULL is run (RAW and
CORRECTION do not use the schema); writes results/hyq_modes[_lab1]_angle.json.

train=lab1 reverses the roles: train / validate / test on Trot in Lab 1, score the whole of Trot in Lab 2 as the unseen
recording (baseline: the Lab-1-trained round-9 networks); writes results/hyq_modes_lab1.json.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.dirname(HERE)
REPO = os.path.dirname(LIB)
sys.path.insert(0, os.path.join(LIB, "python"))
sys.path.insert(0, os.path.join(REPO, "vnet", "round9"))
import sldrnc  # noqa: E402

kw = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
STEPS = int(kw.get("steps", 12000))
LAB1 = kw.get("train", "lab2") == "lab1"
OUT = os.path.join(LIB, "results", "hyq_modes_lab1.json" if LAB1 else "hyq_modes.json")
# (training recording, unseen recording, prep file of the training recording, prep file of the unseen one, baselines)
REC_TR, REC_X = ("trot_in_lab_1", "trot_in_lab_2") if LAB1 else ("trot_in_lab_2", "trot_in_lab_1")
PREP_TR, PREP_X = ("prep9_trot_in_lab_1.npz", "prep9_lab2_xfer.npz") if LAB1 else ("prep9.npz", "prep9_lab1_xfer.npz")
BASELINES = ("L1_ref_s0", "L1_ref_s1") if LAB1 else ("ab_ref_s0", "ab_ref_s1")
TAG = "_lab1" if LAB1 else ""
ANGLE = kw.get("angle", "0") == "1"
if ANGLE:
    OUT = OUT.replace(".json", "_angle.json")
    TAG += "_angle"


def raw_rows(rec):
    import prep9
    S = prep9.raw(rec)
    X = np.concatenate([S[:, 3:15], S[:, 15:27], S[:, 39:43], S[:, 43:46], S[:, 46:49], S[:, 49:52], S[:, 52:55]], 1)
    return np.ascontiguousarray(X), np.ascontiguousarray(S[:, 27:39])


def schema():
    s = sldrnc.Schema(rate_hz=1000)
    s.add("q", sldrnc.Input.ANGLE if ANGLE else sldrnc.Input.POSITION, 12).add("qd", sldrnc.Input.VELOCITY, 12).add("orientation", sldrnc.Input.QUATERNION, 4)
    s.add("gyro", sldrnc.Input.GYRO, 3).add("accel", sldrnc.Input.ACCEL, 3).add("gyro2", sldrnc.Input.GYRO, 3).add("accel2", sldrnc.Input.ACCEL, 3)
    s.outputs(12, velocity="qd")
    return s


def baseline_onnx(key):
    """the round-9 dense baseline as an ONNX file: (x - mu) / sd -> 3 Gemm (ReLU^2 between) -> * ys + ym"""
    from onnx import TensorProto, helper, numpy_helper
    z = np.load(os.path.join(REPO, "vnet", "round9", PREP_TR))
    mu = np.concatenate([z["mu|base"], z["mu|hist4"]]).astype(np.float32)
    sd = np.concatenate([z["sd|base"], z["sd|hist4"]]).astype(np.float32)
    w = np.load(os.path.join(REPO, "vnet", "round9", "runs", key + ".npz"))
    inits = [numpy_helper.from_array(mu, "mu"), numpy_helper.from_array(sd, "sd"), numpy_helper.from_array(np.array(2.0, np.float32), "two"),
             numpy_helper.from_array(z["ys"].astype(np.float32), "ys"), numpy_helper.from_array(z["ym"].astype(np.float32), "ym")]
    nodes = [helper.make_node("Sub", ["x", "mu"], ["a"]), helper.make_node("Div", ["a", "sd"], ["h0"])]
    prev = "h0"
    for i in range(3):
        inits += [numpy_helper.from_array(w["W%d" % i].astype(np.float32), "W%d" % i), numpy_helper.from_array(w["b%d" % i].astype(np.float32), "b%d" % i)]
        nodes.append(helper.make_node("Gemm", [prev, "W%d" % i, "b%d" % i], ["g%d" % i]))
        prev = "g%d" % i
        if i < 2:
            nodes += [helper.make_node("Relu", [prev], ["r%d" % i]), helper.make_node("Pow", ["r%d" % i, "two"], ["p%d" % i])]
            prev = "p%d" % i
    nodes += [helper.make_node("Mul", [prev, "ys"], ["s"]), helper.make_node("Add", ["s", "ym"], ["y"])]
    g = helper.make_graph(nodes, "baseline", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 140])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 12])], inits)
    return helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)]).SerializeToString(), w


def baseline_inputs(prep_file):
    z = np.load(os.path.join(REPO, "vnet", "round9", prep_file))
    return np.concatenate([z["g|base"], z["g|hist4"]], 1).astype(np.float64)


def main():
    t_all = time.time()
    X2, Y2 = raw_rows(REC_TR)       # 2 = the training recording, 1 = the unseen one (names from the default direction)
    X1, Y1 = raw_rows(REC_X)
    T = len(X2)
    n, g = T // 10, 50
    tr = slice(g, 7 * n - g)
    va = slice(7 * n + g, 8 * n - g)
    te_run = slice(8 * n + g, T - g)            # replay from the guard chunk, score the test chunk
    te_from = (9 * n + g) - (8 * n + g)
    x1 = slice(g, len(X1) - g)
    res = {"steps": STEPS, "rows": []}

    def score(model, X_te, X_x1, label, seed):
        row = {"mode": label, "seed": seed}
        for name, Xs, Ys, start in (("test", X_te, Y2[te_run], te_from), ("other_recording", X_x1, Y1[x1], 2000)):
            row[name + "_corrected"] = model.run(Xs, Ys, 1, start).error if model.info.has_correction else None
            row[name + "_uncorrected"] = model.run(Xs, Ys, 1, start, correction=False).error
            row[name + "_d50"] = model.run(Xs, Ys, 50, start).error if model.info.has_correction else None
            if model.info.has_correction:
                row[name + "_corrected_unprotected"] = model.run(Xs, Ys, 1, start, protect=False).error
            if model.info.has_correction:   # physical units, corrected, D = 1
                p = model.run(Xs, Ys, 1, start).predictions[start:]
                e = np.sqrt(((p - Ys[start:]) ** 2).mean(0))
                row[name + "_rms_haa_Nm"] = float(e[[0, 3, 6, 9]].mean())
                row[name + "_rms_fe_N"] = float(e[[1, 2, 4, 5, 7, 8, 10, 11]].mean())
        if model.info.has_correction:   # round-9 fault: left-front knee force stuck at 0 N for 2 s, 12 s into the test chunk
            Xs, Ys = X_te, Y2[te_run]
            k0, k1 = te_from + 12000, te_from + 14000
            Ybad = Ys.copy()
            Ybad[k0:k1, 2] = 0.0
            var = Ys[te_from:].var(0)
            for tag, corr_on, prot in (("fault_nocorrection", False, True), ("fault_unprotected", True, False), ("fault_protected", True, True)):
                sess = model.session(correction=corr_on, protect=prot)
                P = np.empty_like(Ys)
                for t in range(len(Xs)):
                    P[t] = sess.step(Xs[t])
                    sess.observe(Ybad[t])
                w = slice(k0, k1 + 5000)
                row[tag] = float(np.mean(((P[w] - Ys[w]) ** 2).mean(0) / var))
        res["rows"].append(row)
        print("  %-30s seed %d | test %s / %.4f | other recording %s / %.4f   (corrected / uncorrected)" % (
            label, seed, "%.4f" % row["test_corrected"] if row["test_corrected"] is not None else "  -   ", row["test_uncorrected"],
            "%.4f" % row["other_recording_corrected"] if row["other_recording_corrected"] is not None else "  -   ", row["other_recording_uncorrected"]), flush=True)
        with open(OUT, "w") as f:
            json.dump(res, f, indent=1)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    # ---------------- RAW and CORRECTION: the user's own network (round-9 baseline), imported from ONNX
    F2, F1 = baseline_inputs(PREP_TR), baseline_inputs(PREP_X)
    for seed, key in enumerate(BASELINES if not ANGLE else ()):
        onnx_bytes, w = baseline_onnx(key)
        raw = sldrnc.Model.from_onnx(onnx_bytes)
        # check: the imported network reproduces the numpy forward pass
        z = np.load(os.path.join(REPO, "vnet", "round9", PREP_TR))
        mu = np.concatenate([z["mu|base"], z["mu|hist4"]]); sd = np.concatenate([z["sd|base"], z["sd|hist4"]])
        xs = F2[va][:2000]
        h = ((xs - mu) / sd).astype(np.float32)
        for i in range(3):
            h = h @ w["W%d" % i] + w["b%d" % i]
            if i < 2:
                h = np.maximum(h, 0) ** 2
        ref = h * z["ys"] + z["ym"]
        dev = float((np.abs(raw.run(xs).predictions - ref) / z["ys"]).max())        # per output, in units of that output's std
        res.setdefault("onnx_import_max_dev_in_output_std", []).append(dev)
        print("RAW %s: ONNX import vs numpy, max deviation %.2e output std" % (key, dev))
        score(raw, F2[te_run], F1[x1], "RAW (your network)", seed)
        corr = sldrnc.Model.from_onnx(onnx_bytes)
        rep = corr.add_correction(F2[va], Y2[va], rate_hz=1000, velocity_index=[12 + j for j in range(12)])
        print("CORRECTION %s: validation %.4f -> %.4f, time constant %.0f ms" % (key, rep.validation_error, rep.validation_error_corrected, rep.correction_time_ms))
        score(corr, F2[te_run], F1[x1], "CORRECTION (your network + SLD-RNC)", seed)
    # ---------------- FULL
    sc = schema()
    for size in ("small", "medium", "large"):
        for seed in (0, 1):
            t0 = time.time()
            model, rep = sldrnc.train(sc, X2[tr], Y2[tr], X2[va], Y2[va], size=size, steps=STEPS, seed=seed)
            print("FULL %s seed %d: trained in %.0f s; train %.4f, validation %.4f -> %.4f corrected (%.0f ms), %d MACs" % (
                size, seed, time.time() - t0, rep.train_error, rep.validation_error, rep.validation_error_corrected, rep.correction_time_ms, rep.macs_per_tick))
            res.setdefault("full_reports", []).append(dict(size=size, seed=seed, train=rep.train_error, val=rep.validation_error,
                                                           val_corrected=rep.validation_error_corrected, seconds=rep.seconds, macs=rep.macs_per_tick))
            score(model, X2[te_run], X1[x1], "FULL %s" % size, seed)
            model.save(os.path.join(LIB, "results", "hyq_full_%s%s_s%d.sldm" % (size, TAG, seed)))
    print("done in %.0f s" % (time.time() - t_all))


if __name__ == "__main__":
    main()
