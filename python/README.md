# sldrnc (Python)

Python API of **SLD-RNC** (Self-correcting Learned Dynamics for Real-time Neural Control). It predicts the forces a
machine needs from its own sensors, fast enough to run inside the control loop, and corrects itself online from the
measured values.

```python
import sldrnc

schema = sldrnc.Schema(rate_hz=1000)
schema.add("q", sldrnc.Input.POSITION, 12).add("qd", sldrnc.Input.VELOCITY, 12)
schema.outputs(12, velocity="qd")
model, report = sldrnc.train(schema, X_train, Y_train, X_val, Y_val, size="small")   # FULL mode

robot = model.session()
torque = robot.step(x)              # every tick
robot.observe(torque_measured)      # self-correction
```

The compiled library for the platform is inside the package; NumPy is the only dependency. The full documentation
is in the `api/` folder of the SLD-RNC package (`api/python.md` for this API).

The Python source is licensed under the API licence (MIT, `LICENSE.txt`), and the compiled library inside it under the
SLD-RNC binary licence (`LICENSE-BINARY.txt`).
