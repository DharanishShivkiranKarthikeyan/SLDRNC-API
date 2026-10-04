"""Low-level ctypes binding to the SLD-RNC C interface (sldrnc.h).  Use the high-level API in sldrnc/__init__.py."""
import ctypes as C
import os
import platform
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def _platform_tag():
    m = platform.machine().lower()
    arch = "arm64" if m in ("arm64", "aarch64") else "x64"
    if sys.platform.startswith("win"):
        return "windows-" + arch, "sldrnc.dll"
    if sys.platform == "darwin":
        return "macos-" + arch, "libsldrnc.dylib"
    return "linux-" + arch, "libsldrnc.so"


def _load():
    override = os.environ.get("SLDRNC_LIBRARY")
    if override and not os.path.exists(override):          # never fall back silently to another build
        raise OSError("SLDRNC_LIBRARY is set to %s, which does not exist" % override)
    tag, name = _platform_tag()
    candidates = [override] if override else []
    candidates += [os.path.join(_HERE, "_native", tag, name), os.path.join(_HERE, name)]
    for path in candidates:
        if path and os.path.exists(path):
            return C.CDLL(path)
    raise OSError("SLD-RNC native library not found for %s (looked in: %s). Set SLDRNC_LIBRARY to its path."
                  % (tag, ", ".join(p for p in candidates if p)))


lib = _load()

c_double_p = C.POINTER(C.c_double)
c_float_p = C.POINTER(C.c_float)
c_int_p = C.POINTER(C.c_int)


class TrainOptions(C.Structure):
    _fields_ = [("size", C.c_int), ("steps", C.c_int), ("batch", C.c_int), ("learning_rate", C.c_double),
                ("weight_decay", C.c_double), ("seed", C.c_uint64), ("threads", C.c_int), ("fit_correction", C.c_int),
                ("verbose", C.c_int), ("correction_delay_ticks", C.c_int)]


class Report(C.Structure):
    _fields_ = [("train_nmse", C.c_double), ("val_nmse", C.c_double), ("val_nmse_corrected", C.c_double),
                ("correction_time_ms", C.c_double), ("seconds", C.c_double), ("macs_per_tick", C.c_int64), ("steps", C.c_int),
                ("correction_delay_ticks", C.c_int)]


class Layer(C.Structure):
    _fields_ = [("in_", C.c_int), ("out", C.c_int), ("activation", C.c_int), ("W", c_float_p), ("b", c_float_p)]


class ModelInfo(C.Structure):
    _fields_ = [("mode", C.c_int), ("input_size", C.c_int), ("output_size", C.c_int), ("has_correction", C.c_int),
                ("rate_hz", C.c_double), ("correction_time_ms", C.c_double), ("macs_per_tick", C.c_int64), ("parameters", C.c_int64),
                ("warmup_seconds", C.c_double), ("correction_delay_ticks", C.c_int), ("history", C.c_int), ("affine_inputs", C.c_int),
                ("table_bytes", C.c_int64)]


class SessionOptions(C.Structure):
    _fields_ = [("use_correction", C.c_int), ("protect", C.c_int), ("max_delay_ticks", C.c_int)]


class FleetOptions(C.Structure):
    _fields_ = [("use_correction", C.c_int), ("protect", C.c_int), ("max_delay_ticks", C.c_int), ("threads", C.c_int)]


def _sig(name, res, *args):
    f = getattr(lib, name)
    f.restype = res
    f.argtypes = list(args)
    return f


vp = C.c_void_p
st = C.c_int
sldrnc_last_error = _sig("sldrnc_last_error", C.c_char_p)
sldrnc_version = _sig("sldrnc_version", C.c_char_p)
sldrnc_schema_create = _sig("sldrnc_schema_create", st, C.c_double, C.POINTER(vp))
sldrnc_schema_add_input = _sig("sldrnc_schema_add_input", st, vp, C.c_char_p, C.c_int, C.c_int)
sldrnc_schema_set_outputs = _sig("sldrnc_schema_set_outputs", st, vp, C.c_int, c_int_p)
sldrnc_schema_input_size = _sig("sldrnc_schema_input_size", C.c_int, vp)
sldrnc_schema_output_size = _sig("sldrnc_schema_output_size", C.c_int, vp)
sldrnc_schema_column = _sig("sldrnc_schema_column", C.c_int, vp, C.c_char_p)
sldrnc_schema_free = _sig("sldrnc_schema_free", None, vp)
sldrnc_schema_set_history = _sig("sldrnc_schema_set_history", st, vp, C.c_int)
sldrnc_schema_set_affine = _sig("sldrnc_schema_set_affine", st, vp, C.c_char_p, C.c_int)
sldrnc_model_compile_table = _sig("sldrnc_model_compile_table", st, vp, C.c_int, c_int_p, C.POINTER(vp))
sldrnc_fleet_create = _sig("sldrnc_fleet_create", st, vp, C.c_size_t, C.POINTER(FleetOptions), C.POINTER(vp))
sldrnc_fleet_step = _sig("sldrnc_fleet_step", st, vp, c_double_p, c_double_p)
sldrnc_fleet_observe = _sig("sldrnc_fleet_observe", st, vp, c_double_p)
sldrnc_fleet_observe_delayed = _sig("sldrnc_fleet_observe_delayed", st, vp, c_double_p, C.c_int)
sldrnc_fleet_reset = _sig("sldrnc_fleet_reset", st, vp)
sldrnc_fleet_size = _sig("sldrnc_fleet_size", C.c_size_t, vp)
sldrnc_fleet_free = _sig("sldrnc_fleet_free", None, vp)
sldrnc_train_options_default = _sig("sldrnc_train_options_default", None, C.POINTER(TrainOptions))
sldrnc_train_full = _sig("sldrnc_train_full", st, vp, c_double_p, c_double_p, C.c_size_t, c_double_p, c_double_p, C.c_size_t,
                         C.POINTER(TrainOptions), C.POINTER(vp), C.POINTER(Report))
c_double_pp = C.POINTER(c_double_p)
sldrnc_train_full_multi = _sig("sldrnc_train_full_multi", st, vp, C.c_size_t, c_double_pp, c_double_pp, C.POINTER(C.c_size_t),
                               C.c_size_t, c_double_pp, c_double_pp, C.POINTER(C.c_size_t), C.POINTER(TrainOptions), C.POINTER(vp),
                               C.POINTER(Report))
sldrnc_model_from_layers = _sig("sldrnc_model_from_layers", st, C.c_int, C.POINTER(Layer), c_float_p, c_float_p, c_float_p, c_float_p,
                                C.POINTER(vp))
sldrnc_model_import_onnx = _sig("sldrnc_model_import_onnx", st, C.c_char_p, C.POINTER(vp))
sldrnc_model_import_onnx_memory = _sig("sldrnc_model_import_onnx_memory", st, C.c_void_p, C.c_size_t, C.POINTER(vp))
sldrnc_model_add_correction = _sig("sldrnc_model_add_correction", st, vp, c_double_p, c_double_p, C.c_size_t, C.c_double, c_int_p,
                                   C.c_int, C.POINTER(Report))
sldrnc_model_tune_correction = _sig("sldrnc_model_tune_correction", st, vp, c_double_p, c_double_p, C.c_size_t, C.c_int, C.POINTER(vp),
                                    C.POINTER(Report))
sldrnc_model_load = _sig("sldrnc_model_load", st, C.c_char_p, C.POINTER(vp))
sldrnc_model_save = _sig("sldrnc_model_save", st, vp, C.c_char_p)
sldrnc_model_free = _sig("sldrnc_model_free", None, vp)
sldrnc_model_get_info = _sig("sldrnc_model_get_info", st, vp, C.POINTER(ModelInfo))
sldrnc_session_options_default = _sig("sldrnc_session_options_default", None, C.POINTER(SessionOptions))
sldrnc_session_create = _sig("sldrnc_session_create", st, vp, C.POINTER(SessionOptions), C.POINTER(vp))
sldrnc_session_step = _sig("sldrnc_session_step", st, vp, c_double_p, c_double_p)
sldrnc_session_observe = _sig("sldrnc_session_observe", st, vp, c_double_p)
sldrnc_session_observe_delayed = _sig("sldrnc_session_observe_delayed", st, vp, c_double_p, C.c_int)
sldrnc_session_reset = _sig("sldrnc_session_reset", st, vp)
sldrnc_session_free = _sig("sldrnc_session_free", None, vp)
sldrnc_model_run = _sig("sldrnc_model_run", st, vp, C.POINTER(SessionOptions), c_double_p, c_double_p, C.c_size_t, C.c_int, C.c_size_t,
                        c_double_p, c_double_p, c_double_p)
