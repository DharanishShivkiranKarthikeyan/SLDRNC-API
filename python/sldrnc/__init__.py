"""SLD-RNC (Self-correcting Learned Dynamics for Real-time Neural Control) -- fast learned dynamics for robots and
machines (Python API).

Three ways to use it:

    import sldrnc

    # 1. FULL: train an SLD-RNC model from your robot's logs (most accurate)
    schema = sldrnc.Schema(rate_hz=1000)
    schema.add("q", sldrnc.Input.POSITION, 12)
    schema.add("qd", sldrnc.Input.VELOCITY, 12)
    schema.outputs(12, velocity="qd")              # output j belongs to the joint whose velocity is qd[j]
    model, report = sldrnc.train(schema, X_train, Y_train, X_val, Y_val, size="small")

    # 3. RAW: run your own network as-is            (from_onnx / from_layers / from_torch)
    mine = sldrnc.Model.from_onnx("my_net.onnx")

    # 2. CORRECTION: your network + SLD-RNC self-correction, tuned on validation data
    report = mine.add_correction(X_val, Y_val, rate_hz=1000, velocity_index=[...])

    # run time (any mode): one session per robot
    robot = model.session()
    y = robot.step(x)                # predict this tick
    robot.observe(y_measured)        # feed the measurement (drives self-correction)

    # a copy tuned for measurements 50 ticks late (e.g. a planner predicting 50 ms ahead at 1 kHz)
    planner, report = model.tune_correction(X_val, Y_val, delay_ticks=50)

Arrays are NumPy, one row per tick.  FULL models take raw signal rows described by the schema; RAW / CORRECTION models
take your network's own input rows.
"""
import ctypes as _C
import enum as _enum
import os as _os
from dataclasses import dataclass as _dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as _np

from . import _capi

__all__ = ["Error", "Mode", "Input", "Activation", "Size", "Schema", "Dense", "Report", "Info", "RunResult", "Model", "Session",
           "Fleet", "train", "version"]
__version__ = _capi.sldrnc_version().decode()


class Error(Exception):
    """Raised when the library reports a problem.  `code` is the sldrnc_status value (see ERROR_* names below)."""
    ARGUMENT, IO, FORMAT, UNSUPPORTED, STATE, DATA, INTERNAL = 1, 2, 3, 4, 5, 6, 7

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


def _check(status: int):
    if status != 0:
        raise Error(status, _capi.sldrnc_last_error().decode(errors="replace"))


def version() -> str:
    return __version__


class Mode(_enum.IntEnum):
    FULL = 0          # trained by SLD-RNC from your logs
    CORRECTION = 1    # your network + SLD-RNC self-correction
    RAW = 2           # your network, run as-is


class Input(_enum.IntEnum):
    """What a schema channel measures (tells FULL training how to use it)."""
    POSITION = 0
    VELOCITY = 1
    ACCELERATION = 2
    GYRO = 3
    ACCEL = 4
    QUATERNION = 5    # w, x, y, z
    GENERIC = 6
    ANGLE = 7         # rotary joint angles in rad: the model also sees their sine and cosine


class Activation(_enum.IntEnum):
    IDENTITY = 0
    RELU = 1
    RELU2 = 2         # max(x, 0) ** 2
    LEAKY_RELU = 3    # slope 0.01
    TANH = 4
    SIGMOID = 5
    GELU = 6          # tanh approximation
    SILU = 7


class Size(_enum.IntEnum):
    SMALL = 0
    MEDIUM = 1
    LARGE = 2


def _rows(a, name, cols=None) -> _np.ndarray:
    a = _np.ascontiguousarray(a, dtype=_np.float64)
    if a.ndim == 1:
        a = a.reshape(-1, 1) if cols == 1 else a.reshape(1, -1)
    if a.ndim != 2:
        raise Error(Error.ARGUMENT, "%s must be a 2-D array (ticks x columns)" % name)
    if cols is not None and a.shape[1] != cols:
        raise Error(Error.ARGUMENT, "%s has %d columns, expected %d" % (name, a.shape[1], cols))
    return a


def _dp(a):
    return a.ctypes.data_as(_capi.c_double_p)


def _fp(a):
    return None if a is None else a.ctypes.data_as(_capi.c_float_p)


# ------------------------------------------------------------------------------------------------ schema
class Schema:
    """Describes one tick of raw signals for FULL training.  Channels form the input row in the order they are added."""

    def __init__(self, rate_hz: float, history: bool = True):
        """history=False: the outputs depend only on the current inputs (e.g. a motor's current controller): current values
        only, no warm-up, and training rows may be unordered samples."""
        h = _C.c_void_p()
        _check(_capi.sldrnc_schema_create(float(rate_hz), _C.byref(h)))
        self._h = h
        self.rate_hz = float(rate_hz)
        if not history:
            _check(_capi.sldrnc_schema_set_history(h, 0))
        self.history = bool(history)

    def __del__(self):
        h, self._h = getattr(self, "_h", None), None
        if h:
            _capi.sldrnc_schema_free(h)

    def add(self, name: str, kind: Union[Input, int], size: int) -> "Schema":
        _check(_capi.sldrnc_schema_add_input(self._h, name.encode(), int(kind), int(size)))
        return self

    def outputs(self, n: int, velocity: Optional[str] = None, velocity_index: Optional[Sequence[int]] = None) -> "Schema":
        """Declare n outputs.  velocity="qd": output j belongs to the joint whose velocity is element j of channel "qd".
        Or give velocity_index explicitly (one input column per output, -1 = none)."""
        if velocity is not None:
            c = self.column(velocity)
            if c < 0:
                raise Error(Error.ARGUMENT, "no channel named %r" % velocity)
            velocity_index = [c + j for j in range(n)]
        arr = None
        if velocity_index is not None:
            if len(velocity_index) != n:
                raise Error(Error.ARGUMENT, "velocity_index needs one entry per output")
            arr = (_C.c_int * n)(*[int(v) for v in velocity_index])
        _check(_capi.sldrnc_schema_set_outputs(self._h, int(n), arr))
        return self

    def affine(self, *names: str) -> "Schema":
        """The outputs are affine (linear plus offset) in these channels' current values, with coefficients the model
        learns from the other inputs.  E.g. joint torques are affine in joint accelerations; a current controller's
        voltages are affine in speed and the current references."""
        for n in names:
            _check(_capi.sldrnc_schema_set_affine(self._h, n.encode(), 1))
        return self

    def column(self, name: str) -> int:
        return _capi.sldrnc_schema_column(self._h, name.encode())

    @property
    def width(self) -> int:
        return _capi.sldrnc_schema_input_size(self._h)

    @property
    def output_count(self) -> int:
        return _capi.sldrnc_schema_output_size(self._h)


# ------------------------------------------------------------------------------------------------ results
@_dataclass
class Report:
    """Training / tuning summary.  Errors are normalised MSE: 0 = perfect, 1 = no better than the average."""
    train_error: float
    validation_error: float
    validation_error_corrected: float
    correction_time_ms: float
    seconds: float
    macs_per_tick: int
    steps: int
    correction_delay_ticks: int     # the measurement delay self-correction was tuned for (0 = not fitted)

    @staticmethod
    def _from(r: "_capi.Report") -> "Report":
        return Report(r.train_nmse, r.val_nmse, r.val_nmse_corrected, r.correction_time_ms, r.seconds, int(r.macs_per_tick), int(r.steps),
                      int(r.correction_delay_ticks))


@_dataclass
class Info:
    mode: Mode
    input_size: int
    output_size: int
    has_correction: bool
    rate_hz: float
    correction_time_ms: float
    macs_per_tick: int
    parameters: int
    warmup_seconds: float           # FULL: a fresh session is fully accurate after this much data
    correction_delay_ticks: int     # the measurement delay self-correction was tuned for (0 = no correction)
    history: bool                   # FULL: uses the recent past (False: current inputs only)
    affine_inputs: int              # FULL: input values the outputs are affine in (0 = none)
    table_bytes: int                # > 0: the model is a compiled table of this size


@_dataclass
class RunResult:
    predictions: _np.ndarray          # ticks x outputs
    error: float                      # NaN if no targets were given
    error_per_output: _np.ndarray


@_dataclass
class Dense:
    """One fully connected layer for Model.from_layers: W is (in, out); b is (out,)."""
    W: _np.ndarray
    b: Optional[_np.ndarray] = None
    activation: Union[Activation, str, int] = Activation.IDENTITY


def _act(a) -> int:
    if isinstance(a, str):
        try:
            return int(Activation[a.upper()])
        except KeyError:
            raise Error(Error.ARGUMENT, "unknown activation %r (use one of %s)" % (a, ", ".join(x.name.lower() for x in Activation)))
    return int(a)


# ------------------------------------------------------------------------------------------------ session
class Session:
    """Run-time state for one robot.  Not thread-safe: one session per robot / thread."""

    def __init__(self, model: "Model", correction: bool, protect: bool, max_delay_ticks: int = 0):
        self._model = model                      # keeps the model alive
        o = _capi.SessionOptions(1 if correction else 0, 1 if protect else 0, int(max_delay_ticks))
        h = _C.c_void_p()
        _check(_capi.sldrnc_session_create(model._h, _C.byref(o), _C.byref(h)))
        self._h = h
        info = model.info
        self._nin, self._nout = info.input_size, info.output_size
        self._y = _np.zeros(self._nout)

    def __del__(self):
        h, self._h = getattr(self, "_h", None), None
        if h:
            _capi.sldrnc_session_free(h)

    def step(self, x) -> _np.ndarray:
        """Predict this tick's outputs from this tick's input row.  Returns a new array."""
        x = _np.ascontiguousarray(x, dtype=_np.float64)
        if x.size != self._nin:
            raise Error(Error.ARGUMENT, "input row has %d values, expected %d" % (x.size, self._nin))
        y = _np.empty(self._nout)
        _check(_capi.sldrnc_session_step(self._h, _dp(x), _dp(y)))
        return y

    def step_into(self, x: _np.ndarray, out: _np.ndarray) -> None:
        """Allocation-free variant: x and out must be contiguous float64 arrays of the right size."""
        _check(_capi.sldrnc_session_step(self._h, _dp(x), _dp(out)))

    def observe(self, y_measured, delay_ticks: int = 1) -> None:
        """Feed measured outputs.  delay_ticks = 1: the tick just predicted.  delay_ticks = D: the tick predicted D - 1
        steps before the latest (a late measurement; feed them in tick order).  NaN values are skipped."""
        y = _np.ascontiguousarray(y_measured, dtype=_np.float64)
        if y.size != self._nout:
            raise Error(Error.ARGUMENT, "measurement has %d values, expected %d" % (y.size, self._nout))
        if delay_ticks == 1:
            _check(_capi.sldrnc_session_observe(self._h, _dp(y)))
        else:
            _check(_capi.sldrnc_session_observe_delayed(self._h, _dp(y), int(delay_ticks)))

    def reset(self) -> None:
        _check(_capi.sldrnc_session_reset(self._h))


# ------------------------------------------------------------------------------------------------ fleet
class Fleet:
    """Many robots stepped together.  Row r of the inputs belongs to robot r.  Each robot keeps its own history and
    self-correction; the outputs equal separate sessions bit for bit."""

    def __init__(self, model: "Model", robots: int, correction: bool, protect: bool, max_delay_ticks: int, threads: int):
        self._model = model                      # keeps the model alive
        o = _capi.FleetOptions(1 if correction else 0, 1 if protect else 0, int(max_delay_ticks), int(threads))
        h = _C.c_void_p()
        _check(_capi.sldrnc_fleet_create(model._h, int(robots), _C.byref(o), _C.byref(h)))
        self._h = h
        info = model.info
        self.robots = int(robots)
        self._nin, self._nout = info.input_size, info.output_size

    def __del__(self):
        h, self._h = getattr(self, "_h", None), None
        if h:
            _capi.sldrnc_fleet_free(h)

    def step(self, X) -> _np.ndarray:
        """X: robots x input size.  Returns robots x outputs."""
        X = _rows(X, "X", self._nin)
        if len(X) != self.robots:
            raise Error(Error.ARGUMENT, "X has %d rows, the fleet has %d robots" % (len(X), self.robots))
        Y = _np.empty((self.robots, self._nout))
        _check(_capi.sldrnc_fleet_step(self._h, _dp(X), _dp(Y)))
        return Y

    def step_into(self, X: _np.ndarray, out: _np.ndarray) -> None:
        """Allocation-free variant: X (robots x inputs) and out (robots x outputs) must be contiguous float64."""
        _check(_capi.sldrnc_fleet_step(self._h, _dp(X), _dp(out)))

    def observe(self, Y_measured, delay_ticks: int = 1) -> None:
        """Measured outputs of every robot (robots x outputs), for the tick just predicted (or delay_ticks late)."""
        Y = _rows(Y_measured, "Y", self._nout)
        if len(Y) != self.robots:
            raise Error(Error.ARGUMENT, "Y has %d rows, the fleet has %d robots" % (len(Y), self.robots))
        if delay_ticks == 1:
            _check(_capi.sldrnc_fleet_observe(self._h, _dp(Y)))
        else:
            _check(_capi.sldrnc_fleet_observe_delayed(self._h, _dp(Y), int(delay_ticks)))

    def reset(self) -> None:
        _check(_capi.sldrnc_fleet_reset(self._h))


# ------------------------------------------------------------------------------------------------ model
class Model:
    """A trained (FULL) or imported (RAW / CORRECTION) model.  Read-only after creation; safe to share across threads."""

    def __init__(self, handle):
        self._h = handle

    def __del__(self):
        h, self._h = getattr(self, "_h", None), None
        if h:
            _capi.sldrnc_model_free(h)

    # ---- construction
    @staticmethod
    def from_layers(layers: Sequence[Dense], input_mean=None, input_std=None, output_mean=None, output_std=None) -> "Model":
        """RAW mode: your own fully connected network.  Optional normalisation: x -> (x - input_mean) / input_std before the
        first layer, y -> y * output_std + output_mean after the last."""
        keep, L = [], (_capi.Layer * len(layers))()
        prev = None
        for i, d in enumerate(layers):
            W = _np.ascontiguousarray(d.W, dtype=_np.float32)
            if W.ndim != 2:
                raise Error(Error.ARGUMENT, "layer %d: W must be 2-D (in, out)" % i)
            b = None if d.b is None else _np.ascontiguousarray(d.b, dtype=_np.float32).reshape(-1)
            if b is not None and b.size != W.shape[1]:
                raise Error(Error.ARGUMENT, "layer %d: b has %d values, W has %d outputs" % (i, b.size, W.shape[1]))
            if prev is not None and W.shape[0] != prev:
                raise Error(Error.ARGUMENT, "layer %d expects %d inputs but the previous layer has %d outputs" % (i, W.shape[0], prev))
            prev = W.shape[1]
            keep += [W, b]
            L[i] = _capi.Layer(W.shape[0], W.shape[1], _act(d.activation), _fp(W), _fp(b))
        vec = lambda v: None if v is None else _np.ascontiguousarray(v, dtype=_np.float32).reshape(-1)
        im, isd, om, osd = vec(input_mean), vec(input_std), vec(output_mean), vec(output_std)
        h = _C.c_void_p()
        _check(_capi.sldrnc_model_from_layers(len(layers), L, _fp(im), _fp(isd), _fp(om), _fp(osd), _C.byref(h)))
        return Model(h)

    @staticmethod
    def from_onnx(path_or_bytes: Union[str, bytes, "_os.PathLike"]) -> "Model":
        """RAW mode: import an ONNX feed-forward network (file path or the file's bytes)."""
        h = _C.c_void_p()
        if isinstance(path_or_bytes, (bytes, bytearray)):
            buf = (_C.c_char * len(path_or_bytes)).from_buffer_copy(path_or_bytes)
            _check(_capi.sldrnc_model_import_onnx_memory(buf, len(path_or_bytes), _C.byref(h)))
        else:
            _check(_capi.sldrnc_model_import_onnx(_os.fspath(path_or_bytes).encode(), _C.byref(h)))
        return Model(h)

    @staticmethod
    def from_torch(module, input_size: int) -> "Model":
        """RAW mode: a PyTorch feed-forward module (Linear layers + supported activations), via an in-memory ONNX export.
        Requires `torch` (and its ONNX exporter)."""
        import io
        import torch  # noqa: F401  (optional dependency)
        buf = io.BytesIO()
        module = module.eval()
        torch.onnx.export(module, torch.zeros(1, int(input_size)), buf, input_names=["x"], output_names=["y"], opset_version=17,
                          dynamo=False)
        return Model.from_onnx(buf.getvalue())

    @staticmethod
    def load(path) -> "Model":
        h = _C.c_void_p()
        _check(_capi.sldrnc_model_load(_os.fspath(path).encode(), _C.byref(h)))
        return Model(h)

    # ---- CORRECTION
    def add_correction(self, X, Y, rate_hz: float, velocity_index: Optional[Sequence[int]] = None, delay_ticks: int = 1) -> Report:
        """CORRECTION mode: add self-correction to this RAW model, tuned on validation data (X: the model's input rows,
        Y: measured outputs).  velocity_index: per output, the X column with that joint's velocity (-1 = none).
        delay_ticks: tune for measurements this many ticks late (1 = the next tick).  Call before creating sessions."""
        info = self.info
        X = _rows(X, "X", info.input_size)
        Y = _rows(Y, "Y", info.output_size)
        if len(X) != len(Y):
            raise Error(Error.ARGUMENT, "X and Y must have the same number of rows")
        vi = None
        if velocity_index is not None:
            if len(velocity_index) != info.output_size:
                raise Error(Error.ARGUMENT, "velocity_index needs one entry per output")
            vi = (_C.c_int * len(velocity_index))(*[int(v) for v in velocity_index])
        r = _capi.Report()
        _check(_capi.sldrnc_model_add_correction(self._h, _dp(X), _dp(Y), len(X), float(rate_hz), vi, int(delay_ticks), _C.byref(r)))
        return Report._from(r)

    def tune_correction(self, X, Y, delay_ticks: int) -> Tuple["Model", Report]:
        """A copy of this FULL / CORRECTION model with self-correction re-tuned for measurements delay_ticks late, on
        validation data (the same rows the model takes; for FULL the first 2 s are warm-up).  This model is unchanged,
        so it can stay in use: e.g. one copy for the control loop (delay 1) and one for a planner (delay 50)."""
        info = self.info
        X = _rows(X, "X", info.input_size)
        Y = _rows(Y, "Y", info.output_size)
        if len(X) != len(Y):
            raise Error(Error.ARGUMENT, "X and Y must have the same number of rows")
        h = _C.c_void_p()
        r = _capi.Report()
        _check(_capi.sldrnc_model_tune_correction(self._h, _dp(X), _dp(Y), len(X), int(delay_ticks), _C.byref(h), _C.byref(r)))
        return Model(h), Report._from(r)

    # ---- use
    def save(self, path) -> None:
        _check(_capi.sldrnc_model_save(self._h, _os.fspath(path).encode()))

    @property
    def info(self) -> Info:
        i = _capi.ModelInfo()
        _check(_capi.sldrnc_model_get_info(self._h, _C.byref(i)))
        return Info(Mode(i.mode), i.input_size, i.output_size, bool(i.has_correction), i.rate_hz, i.correction_time_ms,
                    int(i.macs_per_tick), int(i.parameters), i.warmup_seconds, int(i.correction_delay_ticks), bool(i.history),
                    int(i.affine_inputs), int(i.table_bytes))

    def compile_table(self, grid: Optional[Sequence[int]] = None) -> "Model":
        """A copy of this model compiled into a lookup table: the fastest form.  The model must be FULL, trained without
        history, with 1 to 3 non-affine input values.  grid: nodes per table dimension, in input order (default 64 per
        value, 128 per ANGLE value)."""
        h = _C.c_void_p()
        g = None if grid is None else (_C.c_int * len(grid))(*[int(v) for v in grid])
        _check(_capi.sldrnc_model_compile_table(self._h, 0 if grid is None else len(grid), g, _C.byref(h)))
        return Model(h)

    def fleet(self, robots: int, correction: bool = True, protect: bool = True, max_delay_ticks: int = 0, threads: int = 0) -> "Fleet":
        """Run-time state for many robots stepped together (fleets, simulations, candidate motions): the same numbers as
        separate sessions, in one call (faster for larger networks).  threads: 0 = all cores."""
        return Fleet(self, robots, correction, protect, max_delay_ticks, threads)

    def session(self, correction: bool = True, protect: bool = True, max_delay_ticks: int = 0) -> Session:
        """A run-time session for one robot.  correction: apply self-correction (if the model has it).  protect: ignore
        implausible measurements (e.g. a failed sensor) and bound the correction.  max_delay_ticks: the largest delay
        observe() will accept (0 = max(256, the model's tuned delay))."""
        return Session(self, correction, protect, max_delay_ticks)

    def run(self, X, Y=None, delay_ticks: int = 1, score_from: int = 0, correction: bool = True, protect: bool = True) -> RunResult:
        """Replay a recording through a fresh session: predict every tick and, if Y is given, observe each row with delay
        `delay_ticks` (1 = normal real time; e.g. 50 at 1 kHz = predicting 50 ms ahead).  Scores ticks >= score_from."""
        info = self.info
        X = _rows(X, "X", info.input_size)
        Yc = None if Y is None else _rows(Y, "Y", info.output_size)
        if Yc is not None and len(Yc) != len(X):
            raise Error(Error.ARGUMENT, "X and Y must have the same number of rows")
        pred = _np.empty((len(X), info.output_size))
        err = _C.c_double()
        per = _np.empty(info.output_size)
        o = _capi.SessionOptions(1 if correction else 0, 1 if protect else 0, 0)
        _check(_capi.sldrnc_model_run(self._h, _C.byref(o), _dp(X), None if Yc is None else _dp(Yc), len(X), int(delay_ticks),
                                      int(score_from), _dp(pred), _C.byref(err), _dp(per)))
        return RunResult(pred, float(err.value), per)


# ------------------------------------------------------------------------------------------------ FULL
def train(schema: Schema, X_train, Y_train, X_val, Y_val, size: Union[Size, str] = Size.MEDIUM, steps: int = 12000, batch: int = 512,
          learning_rate: float = 5e-3, weight_decay: float = 0.2, seed: int = 0, threads: int = 0, fit_correction: bool = True,
          verbose: bool = False, correction_delay_ticks: int = 1) -> Tuple[Model, Report]:
    """FULL mode: train an SLD-RNC model.  X_*: raw signal rows (schema.width columns); Y_*: measured targets.  Each
    recording must be one continuous stretch at the schema rate; use validation data recorded after the training data.
    Several recordings (e.g. one per log file): pass lists of arrays, X_train=[X1, X2, ...], Y_train=[Y1, Y2, ...]
    (likewise for validation); never join separate recordings into one array.
    correction_delay_ticks: tune self-correction for measurements this many ticks late (1 = the next tick)."""
    if schema.output_count <= 0:
        raise Error(Error.ARGUMENT, "call schema.outputs(...) before training")
    n_out = schema.output_count

    def recordings(X, Y, name):
        many = isinstance(X, (list, tuple)) and len(X) > 0 and all(_np.ndim(x) == 2 for x in X)
        Xs, Ys = (list(X), list(Y)) if many else ([X], [Y])
        if len(Xs) != len(Ys):
            raise Error(Error.ARGUMENT, "%s: give one Y recording per X recording" % name)
        Xs = [_rows(x, "X_" + name, schema.width) for x in Xs]
        Ys = [_rows(y, "Y_" + name, n_out) for y in Ys]
        for x, y in zip(Xs, Ys):
            if len(x) != len(y):
                raise Error(Error.ARGUMENT, "X and Y must have the same number of rows")
        return Xs, Ys

    Xt, Yt = recordings(X_train, Y_train, "train")
    Xv, Yv = recordings(X_val, Y_val, "val")
    if isinstance(size, str):
        size = Size[size.upper()]
    o = _capi.TrainOptions()
    _capi.sldrnc_train_options_default(_C.byref(o))
    o.size, o.steps, o.batch, o.learning_rate, o.weight_decay = int(size), int(steps), int(batch), float(learning_rate), float(weight_decay)
    o.seed, o.threads, o.fit_correction, o.verbose = int(seed), int(threads), 1 if fit_correction else 0, 1 if verbose else 0
    o.correction_delay_ticks = int(correction_delay_ticks)
    h = _C.c_void_p()
    r = _capi.Report()
    ptrs = lambda arrs: (_capi.c_double_p * len(arrs))(*[_dp(a) for a in arrs])
    lens = lambda arrs: (_C.c_size_t * len(arrs))(*[len(a) for a in arrs])
    _check(_capi.sldrnc_train_full_multi(schema._h, len(Xt), ptrs(Xt), ptrs(Yt), lens(Xt), len(Xv), ptrs(Xv), ptrs(Yv), lens(Xv),
                                         _C.byref(o), _C.byref(h), _C.byref(r)))
    return Model(h), Report._from(r)
