# Model formats

SLD-RNC: Self-correcting Learned Dynamics for Real-time Neural Control.

## What goes in

| source | how | modes |
|---|---|---|
| **your logs** | arrays of raw signals + measured outputs, described by a `Schema` | FULL |
| **ONNX file** | `Model.from_onnx(path)` / `from_onnx_bytes` / `sldrnc_model_import_onnx` | RAW, then CORRECTION |
| **PyTorch module** | `Model.from_torch(module, input_size)` (exports to ONNX in memory) | RAW, then CORRECTION |
| **Keras / TensorFlow** | convert with `tf2onnx`, then `from_onnx` | RAW, then CORRECTION |
| **any framework, by hand** | `Model.from_layers([...])`: weight matrices, biases, activations | RAW, then CORRECTION |
| **SLD-RNC model file** | `Model.load("x.sldm")` | any |

### Networks SLD-RNC can run (RAW / CORRECTION)

Fully connected (feed-forward) networks with one input vector and one output vector. That covers the usual
inverse-dynamics, actuator and residual models.

**ONNX operations supported**

| category | operations |
|---|---|
| layers | `Gemm` (any `alpha`, `beta`, `transB`), `MatMul` by constant weights |
| constants | `Add`, `Sub`, `Mul`, `Div` with constants (input normalisation, biases, output scaling are folded or kept exact) |
| activations | `Relu`, `LeakyRelu` (alpha 0.01), `Tanh`, `Sigmoid`, `Gelu` (`approximate="tanh"`) |
| patterns | squared ReLU (`Relu` then `x*x` or `Pow(x, 2)`); SiLU / swish (`x * Sigmoid(x)`) |
| shape-only | `Identity`, `Dropout`, `Flatten`, `Reshape`, `Squeeze`, `Cast`, `Constant` |
| data types | float32 and float64 weights stored inside the file |

**Not supported** (the import fails with `UNSUPPORTED`, naming the operation):
- convolutions, recurrent layers (LSTM / GRU), attention;
- branching graphs, multiple inputs or outputs;
- weights in external data files, float16.

**`from_layers` activations:** identity, relu, relu2, leaky_relu, tanh, sigmoid, gelu, silu.

**Input rows for RAW / CORRECTION models** are your network's own input vectors, exactly as it was trained. If it
expects features you compute (history, filtered signals), compute them before `step()`.

## What comes out

| output | what it is |
|---|---|
| **`.sldm` file** | `model.save(path)`: every mode, everything needed to run, including self-correction settings. Versioned, checksummed, and stored scrambled (the weights aren't in plain view) |
| **table `.sldm`** | `model.compile_table()` then `save`: a FULL model without history, compiled into a lookup table (the fastest form). Loads and runs like any model |
| **predictions** | `session.step()` (one tick), or `model.run()` (`ticks × outputs` array) |
| **reports** | `Report` (training / tuning summary) and `RunResult.error` / `error_per_output` |

`.sldm` files are portable between platforms (Windows, Linux, macOS; x86 and ARM). A file from a newer library
version is refused with a clear message rather than misread.

FULL models are saved only as `.sldm`. They can't be exported to ONNX, because they include their input processing
and self-correction.
