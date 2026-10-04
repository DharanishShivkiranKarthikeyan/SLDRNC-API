"""Tests of the SLD-RNC Python API (run: python tests/test_python.py, or pytest tests/test_python.py).
Needs the native library in python/sldrnc/_native/<platform>/ (the build scripts put it there) or SLDRNC_LIBRARY."""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
import sldrnc  # noqa: E402


def simulate(T, t0, seed, drift=0.3):
    """two joints: inertia + friction + gravity + a 20-ms actuator lag + slow drift (X: q0 q1 qd0 qd1; Y: torques)."""
    t = t0 + np.arange(T) * 1e-3
    X, Y = np.zeros((T, 4)), np.zeros((T, 2))
    rng = np.random.default_rng(seed)
    for j in range(2):
        amp, frq = np.array([0.5, 0.3, 0.15]), np.array([1.3, 2.9, 4.7]) + 0.4 * j
        ph = frq[None, :] * t[:, None] + 1.7 * np.arange(3)[None, :] + j
        q = (amp * np.sin(ph)).sum(1)
        qd = (amp * frq * np.cos(ph)).sum(1)
        qdd = -(amp * frq ** 2 * np.sin(ph)).sum(1)
        tau = 1.5 * qdd + 0.4 * qd + 0.8 * np.tanh(qd / 0.05) + 3.0 * np.sin(q) + drift * np.sin(0.2 * t + j)
        lag = np.zeros(T)
        acc = 0.0
        for k in range(T):
            acc += (tau[k] - acc) / 20.0
            lag[k] = acc
        X[:, j], X[:, 2 + j], Y[:, j] = q, qd, lag + rng.uniform(-0.01, 0.01, T)
    return X, Y


TR, VA, TE = simulate(60000, 0.0, 1), simulate(10000, 60.0, 2), simulate(10000, 70.0, 3)


def make_schema():
    s = sldrnc.Schema(rate_hz=1000)
    s.add("q", sldrnc.Input.POSITION, 2).add("qd", sldrnc.Input.VELOCITY, 2).outputs(2, velocity="qd")
    return s


_FULL = {}


def full_model():
    if "m" not in _FULL:
        _FULL["m"] = sldrnc.train(make_schema(), *TR, *VA, size="small", steps=3000, seed=3)
    return _FULL["m"]


def test_full_mode():
    model, rep = full_model()
    assert rep.validation_error < 0.05, rep
    assert rep.validation_error_corrected <= rep.validation_error
    info = model.info
    assert info.mode == sldrnc.Mode.FULL and info.input_size == 4 and info.output_size == 2 and info.has_correction
    r = model.run(*TE)
    r_nc = model.run(*TE, correction=False)
    assert r.error < r_nc.error
    print("  FULL: val %.4f (corrected %.4f); test %.4f corrected, %.4f without" % (rep.validation_error, rep.validation_error_corrected, r.error, r_nc.error))


def test_session_equals_run():
    model, _ = full_model()
    X, Y = TE
    s = model.session()
    pred = np.array([(lambda y: (s.observe(Y[t]), y)[1])(s.step(X[t])) for t in range(len(X))])
    assert np.array_equal(pred, model.run(X, Y).predictions)


def test_save_load():
    model, _ = full_model()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.sldm")
        model.save(p)
        again = sldrnc.Model.load(p)
        assert np.array_equal(again.run(*TE).predictions, model.run(*TE).predictions)
        raw = bytearray(open(p, "rb").read())
        raw[len(raw) // 2] ^= 0x5A
        open(p, "wb").write(raw)
        try:
            sldrnc.Model.load(p)
            assert False, "damaged file accepted"
        except sldrnc.Error as e:
            assert e.code == sldrnc.Error.FORMAT


def user_features(X):
    return np.concatenate([X[:, 2:4], np.sin(X[:, 0:2]), np.tanh(X[:, 2:4] / 0.05)], 1)


def test_raw_and_correction():
    F = user_features(TR[0])
    A = np.concatenate([F, np.ones((len(F), 1))], 1)
    coef, *_ = np.linalg.lstsq(A, TR[1], rcond=None)
    layer = sldrnc.Dense(W=coef[:-1], b=coef[-1], activation="identity")
    raw = sldrnc.Model.from_layers([layer])
    F_te = user_features(TE[0])
    expected = F_te @ coef[:-1] + coef[-1]
    assert np.allclose(raw.run(F_te).predictions, expected, atol=1e-4)
    corr = sldrnc.Model.from_layers([layer])
    rep = corr.add_correction(user_features(VA[0]), VA[1], rate_hz=1000, velocity_index=[0, 1])
    e_raw = raw.run(user_features(TE[0]), TE[1]).error
    e_corr = corr.run(user_features(TE[0]), TE[1]).error
    print("  RAW test %.4f -> CORRECTION %.4f (validation %.4f -> %.4f, %.0f ms)" % (e_raw, e_corr, rep.validation_error, rep.validation_error_corrected, rep.correction_time_ms))
    assert corr.info.mode == sldrnc.Mode.CORRECTION and e_corr < 0.5 * e_raw


def test_onnx_import():
    try:
        import onnx
        from onnx import TensorProto, helper, numpy_helper
    except ImportError:
        print("  (onnx not installed: ONNX test skipped)")
        return
    rng = np.random.default_rng(0)
    mean, std = rng.normal(size=5).astype(np.float32), rng.uniform(0.5, 2, 5).astype(np.float32)
    W1, b1 = rng.normal(size=(16, 5)).astype(np.float32), rng.normal(size=16).astype(np.float32)   # Gemm transB=1
    W2, b2 = rng.normal(size=(16, 3)).astype(np.float32), rng.normal(size=3).astype(np.float32)
    inits = [numpy_helper.from_array(a, n) for a, n in ((mean, "mean"), (std, "std"), (W1, "W1"), (b1, "b1"), (W2, "W2"), (b2, "b2"),
                                                        (np.array(2.0, np.float32), "two"))]
    nodes = [helper.make_node("Sub", ["x", "mean"], ["a"]), helper.make_node("Div", ["a", "std"], ["n"]),
             helper.make_node("Gemm", ["n", "W1", "b1"], ["h"], transB=1), helper.make_node("Relu", ["h"], ["r"]),
             helper.make_node("Pow", ["r", "two"], ["r2"]), helper.make_node("MatMul", ["r2", "W2"], ["m"]),
             helper.make_node("Add", ["m", "b2"], ["y"])]
    g = helper.make_graph(nodes, "net", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 5])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])], inits)
    model_bytes = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)]).SerializeToString()
    m = sldrnc.Model.from_onnx(model_bytes)
    x = rng.normal(size=(200, 5))
    ref = np.maximum((x - mean) / std @ W1.T + b1, 0) ** 2 @ W2 + b2
    got = m.run(x).predictions
    assert m.info.mode == sldrnc.Mode.RAW and np.abs(got - ref).max() < 1e-3 * (1 + np.abs(ref).max()), np.abs(got - ref).max()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "net.onnx")
        open(p, "wb").write(model_bytes)
        assert np.array_equal(sldrnc.Model.from_onnx(p).run(x).predictions, got)
    print("  ONNX import: max |difference| vs numpy %.2e" % np.abs(got - ref).max())


def test_sensor_fault_protection():
    model, _ = full_model()
    X, Y = TE
    Ybad = Y.copy()
    Ybad[4000:6000, 0] = 0.0
    out = {}
    for protect in (True, False):
        s = model.session(protect=protect)
        se = 0.0
        for t in range(len(X)):
            y = s.step(X[t])
            if 4000 <= t < 9000:
                se += (y[0] - Y[t, 0]) ** 2
            s.observe(Ybad[t])
        out[protect] = np.sqrt(se / 5000)
    print("  stuck sensor: RMS error protected %.3f, unprotected %.3f" % (out[True], out[False]))
    assert out[True] < out[False]


def test_several_recordings():
    X, Y = TR
    half = len(X) // 2
    model, rep = sldrnc.train(make_schema(), [X[:half], X[half:]], [Y[:half], Y[half:]], [VA[0]], [VA[1]], size="small", steps=1500,
                              seed=3)
    assert rep.validation_error < 0.05 and rep.validation_error_corrected <= rep.validation_error, rep
    try:
        sldrnc.train(make_schema(), [X[:half], X[half:]], [Y[:half]], *VA, steps=10)
        assert False
    except sldrnc.Error as e:
        assert e.code == sldrnc.Error.ARGUMENT
    print("  2 recordings: validation %.4f (corrected %.4f)" % (rep.validation_error, rep.validation_error_corrected))


def test_angle_inputs():
    s = sldrnc.Schema(rate_hz=1000)
    s.add("q", sldrnc.Input.ANGLE, 2).add("qd", sldrnc.Input.VELOCITY, 2).outputs(2, velocity="qd")
    model, rep = sldrnc.train(s, *TR, *VA, size="small", steps=1500, seed=3)
    assert rep.validation_error < 0.05 and model.info.input_size == 4, rep
    try:
        sldrnc.Schema(1000).add("x", 8, 1)
        assert False
    except sldrnc.Error as e:
        assert e.code == sldrnc.Error.ARGUMENT
    print("  ANGLE inputs: validation %.4f" % rep.validation_error)


def test_fleet_equals_sessions():
    model, _ = full_model()
    X, Y = TE
    R = 5
    fl = model.fleet(R)
    ss = [model.session() for _ in range(R)]
    for t in range(500):
        Xr = np.stack([X[t + 400 * r] for r in range(R)])
        Yr = np.stack([Y[t + 400 * r] for r in range(R)])
        out = fl.step(Xr)
        ref = np.stack([s.step(Xr[r]) for r, s in enumerate(ss)])
        assert np.array_equal(out, ref), "fleet differs from sessions at tick %d" % t
        fl.observe(Yr)
        for r, s in enumerate(ss):
            s.observe(Yr[r])


def test_structured_and_table():
    rng = np.random.default_rng(5)

    def make(n):
        a, v, g = rng.uniform(-3.14, 3.14, n), rng.uniform(-1, 1, n), rng.uniform(-1, 1, n)
        return np.stack([a, v, g], 1), np.stack([np.sin(a) * (1 + 0.5 * g) + v * np.cos(a), 0.5 * np.cos(2 * a) * g - v * np.sin(a)], 1)
    Xs, Ys = make(20000)
    Xv, Yv = make(3000)
    Xt, Yt = make(3000)
    s = sldrnc.Schema(rate_hz=1000, history=False)
    s.add("angle", sldrnc.Input.ANGLE, 1).add("v", sldrnc.Input.GENERIC, 1).add("g", sldrnc.Input.GENERIC, 1).outputs(2)
    s.affine("v")
    model, rep = sldrnc.train(s, Xs, Ys, Xv, Yv, size="small", steps=2000, weight_decay=0, fit_correction=False)
    info = model.info
    assert not info.history and info.affine_inputs == 1 and info.warmup_seconds == 0
    table = model.compile_table([256, 64])
    e_net, e_tab = model.run(Xt, Yt).error, table.run(Xt, Yt).error
    assert table.info.table_bytes > 0 and e_tab < 1.5 * e_net + 1e-4, (e_net, e_tab)
    fl = table.fleet(len(Xt))
    assert np.array_equal(fl.step(Xt), table.run(Xt).predictions)
    print("  structured: validation %.5f, test %.5f; table test %.5f" % (rep.validation_error, e_net, e_tab))


def test_measurement_delay():
    model, _ = full_model()
    D = 50
    planner, rep = model.tune_correction(*VA, delay_ticks=D)
    assert planner.info.correction_delay_ticks == D and rep.correction_delay_ticks == D
    assert model.info.correction_delay_ticks == 1, "tune_correction must leave the original model unchanged"
    X, Y = TE
    ref = planner.run(X, Y, delay_ticks=D)
    s = planner.session()
    P = np.empty_like(Y)
    for t in range(len(X)):
        P[t] = s.step(X[t])
        if t + 1 >= D:
            s.observe(Y[t + 1 - D], delay_ticks=D)      # the measurement of tick t + 1 - D arrives now
    assert np.array_equal(P, ref.predictions), "session observe(y, delay) must equal run(delay) exactly"
    with tempfile.TemporaryDirectory() as d:
        planner.save(os.path.join(d, "p.sldm"))
        assert sldrnc.Model.load(os.path.join(d, "p.sldm")).info.correction_delay_ticks == D
    short = model.session(max_delay_ticks=10)
    short.step(X[0])
    try:
        short.observe(Y[0], delay_ticks=11)
        assert False
    except sldrnc.Error as e:
        assert e.code == sldrnc.Error.ARGUMENT
    print("  delay %d: tuned for 1 -> %.4f, tuned for %d -> %.4f" % (D, model.run(X, Y, delay_ticks=D).error, D, ref.error))


def test_errors():
    model, _ = full_model()
    try:
        model.run(np.zeros((10, 3)))
        assert False
    except sldrnc.Error as e:
        assert e.code == sldrnc.Error.ARGUMENT
    try:
        sldrnc.Model.load("does_not_exist.sldm")
        assert False
    except sldrnc.Error as e:
        assert e.code == sldrnc.Error.IO
    try:
        sldrnc.Schema(1000).add("quat", sldrnc.Input.QUATERNION, 3)
        assert False
    except sldrnc.Error as e:
        assert e.code == sldrnc.Error.ARGUMENT and "4" in str(e)
    try:
        model.add_correction(*VA, rate_hz=1000)
        assert False
    except sldrnc.Error as e:
        assert e.code in (sldrnc.Error.STATE, sldrnc.Error.ARGUMENT)


if __name__ == "__main__":
    print("SLD-RNC", sldrnc.version(), "Python tests")
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok  ", name)
            except Exception as e:  # noqa: BLE001
                fails += 1
                print("FAIL", name, type(e).__name__, e)
    print("PASS" if not fails else "FAIL: %d test(s)" % fails)
    sys.exit(1 if fails else 0)
