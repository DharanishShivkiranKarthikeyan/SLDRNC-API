"""SLD-RNC in five minutes (Python).

A simulated two-joint machine stands in for your robot: the example logs joint positions / velocities and the
measured joint torques, then compares the three modes on data recorded later:
  1. FULL        train an SLD-RNC model from the logs
  2. CORRECTION  keep "your" model and add SLD-RNC self-correction
  3. RAW         run "your" model as-is
and finally run the model the way a 1 kHz control loop would.

    python examples/python/quickstart.py
"""
import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))   # not needed after pip install
import sldrnc  # noqa: E402


def record(seconds, start, seed):
    """stand-in robot: two joints, inertia + friction + gravity + a 20-ms actuator lag + slow drift.
    Returns X (ticks x 4: q0, q1, qd0, qd1) and Y (ticks x 2: measured torques), at 1 kHz."""
    T = int(seconds * 1000)
    t = start + np.arange(T) / 1000.0
    X, Y = np.zeros((T, 4)), np.zeros((T, 2))
    rng = np.random.default_rng(seed)
    for j in range(2):
        amp, frq = np.array([0.5, 0.3, 0.15]), np.array([1.3, 2.9, 4.7]) + 0.4 * j
        ph = frq * t[:, None] + 1.7 * np.arange(3) + j
        q, qd, qdd = (amp * np.sin(ph)).sum(1), (amp * frq * np.cos(ph)).sum(1), -(amp * frq ** 2 * np.sin(ph)).sum(1)
        torque = 1.5 * qdd + 0.4 * qd + 0.8 * np.tanh(qd / 0.05) + 3.0 * np.sin(q) + 0.3 * np.sin(0.2 * t + j)
        lagged = np.zeros(T)
        for k in range(1, T):
            lagged[k] = lagged[k - 1] + (torque[k] - lagged[k - 1]) / 20.0
        X[:, j], X[:, 2 + j], Y[:, j] = q, qd, lagged + rng.uniform(-0.01, 0.01, T)
    return X, Y


def main():
    print("SLD-RNC", sldrnc.version())
    X_train, Y_train = record(60, 0, 1)        # 60 s of logs to learn from
    X_val, Y_val = record(10, 60, 2)           # 10 s recorded afterwards, to pick and tune
    X_test, Y_test = record(20, 70, 3)         # 20 s recorded later still, to score

    # ---- 1. FULL: describe the signals of one tick, then train
    schema = sldrnc.Schema(rate_hz=1000)
    schema.add("q", sldrnc.Input.POSITION, 2)
    schema.add("qd", sldrnc.Input.VELOCITY, 2)
    schema.outputs(2, velocity="qd")           # output j is the torque of the joint whose velocity is qd[j]
    full, report = sldrnc.train(schema, X_train, Y_train, X_val, Y_val, size="small", steps=4000)   # default steps: 12000
    print("trained FULL model in %.1f s: validation error %.4f, %.4f with self-correction"
          % (report.seconds, report.validation_error, report.validation_error_corrected))

    # ---- 3. RAW: bring your own model.  Here: a least-squares fit on hand-made features; from_onnx() / from_torch()
    #         work the same way for real networks.
    def my_features(X):
        return np.concatenate([X[:, 2:4], np.sin(X[:, 0:2]), np.tanh(X[:, 2:4] / 0.05)], 1)

    A = np.concatenate([my_features(X_train), np.ones((len(X_train), 1))], 1)
    coef, *_ = np.linalg.lstsq(A, Y_train, rcond=None)
    my_layer = sldrnc.Dense(W=coef[:-1], b=coef[-1], activation="identity")
    raw = sldrnc.Model.from_layers([my_layer])

    # ---- 2. CORRECTION: the same model plus self-correction, tuned on validation data
    corrected = sldrnc.Model.from_layers([my_layer])
    corrected.add_correction(my_features(X_val), Y_val, rate_hz=1000, velocity_index=[0, 1])   # columns 0, 1 = joint velocities

    # ---- compare on the later recording (error: 0 = perfect, 1 = no better than the average), after a 2-s warm-up
    print("\n%-34s %s" % ("mode", "test error"))
    print("%-34s %.4f" % ("RAW (your model as-is)", raw.run(my_features(X_test), Y_test, score_from=2000).error))
    print("%-34s %.4f" % ("CORRECTION (your model + SLD-RNC)", corrected.run(my_features(X_test), Y_test, score_from=2000).error))
    print("%-34s %.4f" % ("FULL, without self-correction", full.run(X_test, Y_test, score_from=2000, correction=False).error))
    print("%-34s %.4f" % ("FULL", full.run(X_test, Y_test, score_from=2000).error))

    # ---- run time: one session per robot; step() every tick, observe() when the measurement arrives
    with tempfile.TemporaryDirectory() as d:
        full.save(os.path.join(d, "robot.sldm"))
        model = sldrnc.Model.load(os.path.join(d, "robot.sldm"))
    robot = model.session()
    out = np.empty(2)
    t0 = time.perf_counter()
    for t in range(len(X_test)):
        robot.step_into(X_test[t], out)        # feed-forward torque for this tick (allocation-free)
        robot.observe(Y_test[t])               # measured torque for the same tick
    us = (time.perf_counter() - t0) / len(X_test) * 1e6
    print("\nreal-time loop from Python: %.1f us per tick (step + observe, including Python overhead)" % us)

    # ---- many robots: a fleet steps them all in one call (same outputs as separate sessions, one call per tick)
    N = 1000
    fleet = model.fleet(N)
    starts = np.arange(N) * 17 % (len(X_test) - 500)          # each robot replays the test recording from its own place
    Uf = np.empty((N, 2))
    t0 = time.perf_counter()
    for t in range(500):
        fleet.step_into(np.ascontiguousarray(X_test[starts + t]), Uf)
        fleet.observe(Y_test[starts + t])
    rate = N * 500 / (time.perf_counter() - t0)
    print("fleet of %d robots from Python: %.2g robot-ticks per second (step + observe)" % (N, rate))


if __name__ == "__main__":
    main()
