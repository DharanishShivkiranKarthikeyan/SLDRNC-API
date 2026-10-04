// onnx_import.cpp -- import a feed-forward ONNX network (RAW mode) without any dependency: a minimal protobuf reader
// plus a pass that folds the graph into dense layers.
//
// Supported: Gemm, MatMul (activations x constant weights), Add / Sub / Mul / Div with constants (folded into the
// neighbouring layer, e.g. input normalisation or output scaling), Relu, LeakyRelu (alpha 0.01), Tanh, Sigmoid,
// Gelu (approximate="tanh"), x*sigmoid(x) (SiLU), Relu followed by x*x or Pow(x, 2) (squared ReLU), Identity, Dropout,
// Flatten, Reshape, Cast, Constant.  The graph must be a single chain from one input to one output.
#include <cmath>
#include <cstring>
#include <map>
#include <set>

#include "internal.hpp"

namespace sldi {
namespace {

// ------------------------------------------------------------------------------------------------ protobuf
struct PB {
    const uint8_t* p; size_t n, i = 0;
    bool more() const { return i < n; }
    uint64_t varint() {
        uint64_t v = 0; int s = 0;
        for (;;) {
            if (i >= n || s > 63) fail(SLDRNC_ERROR_FORMAT, "ONNX file is damaged (varint)");
            const uint8_t b = p[i++];
            v |= (uint64_t)(b & 0x7F) << s;
            if (!(b & 0x80)) return v;
            s += 7;
        }
    }
    void key(int& field, int& wt) { const uint64_t k = varint(); field = (int)(k >> 3); wt = (int)(k & 7); }
    PB sub() {
        const uint64_t len = varint();
        if (len > n - i) fail(SLDRNC_ERROR_FORMAT, "ONNX file is damaged (length)");
        PB s{p + i, (size_t)len};
        i += (size_t)len;
        return s;
    }
    std::string str() { PB s = sub(); return std::string((const char*)s.p, s.n); }
    uint32_t f32bits() { if (i + 4 > n) fail(SLDRNC_ERROR_FORMAT, "ONNX file is damaged"); uint32_t v; std::memcpy(&v, p + i, 4); i += 4; return v; }
    uint64_t f64bits() { if (i + 8 > n) fail(SLDRNC_ERROR_FORMAT, "ONNX file is damaged"); uint64_t v; std::memcpy(&v, p + i, 8); i += 8; return v; }
    void skip(int wt) {
        switch (wt) {
            case 0: varint(); break;
            case 1: f64bits(); break;
            case 2: sub(); break;
            case 5: f32bits(); break;
            default: fail(SLDRNC_ERROR_FORMAT, "ONNX file is damaged (wire type)");
        }
    }
};

struct Tensor {
    std::vector<int64_t> dims;
    std::vector<double> v;
    int64_t numel() const { int64_t k = 1; for (auto d : dims) k *= d; return k; }
};
Tensor parse_tensor(PB pb, std::string* name) {
    Tensor t;
    int dtype = 0;
    std::vector<uint8_t> raw;
    std::vector<double> fl;
    while (pb.more()) {
        int f, wt;
        pb.key(f, wt);
        if (f == 1) {
            if (wt == 2) { PB s = pb.sub(); while (s.more()) t.dims.push_back((int64_t)s.varint()); }
            else t.dims.push_back((int64_t)pb.varint());
        } else if (f == 2) dtype = (int)pb.varint();
        else if (f == 4) {   // float_data
            if (wt == 2) { PB s = pb.sub(); while (s.more()) { const uint32_t b = s.f32bits(); float x; std::memcpy(&x, &b, 4); fl.push_back(x); } }
            else { const uint32_t b = pb.f32bits(); float x; std::memcpy(&x, &b, 4); fl.push_back(x); }
        } else if (f == 7) {   // int64_data
            if (wt == 2) { PB s = pb.sub(); while (s.more()) fl.push_back((double)(int64_t)s.varint()); }
            else fl.push_back((double)(int64_t)pb.varint());
        } else if (f == 10) {   // double_data
            if (wt == 2) { PB s = pb.sub(); while (s.more()) { const uint64_t b = s.f64bits(); double x; std::memcpy(&x, &b, 8); fl.push_back(x); } }
            else { const uint64_t b = pb.f64bits(); double x; std::memcpy(&x, &b, 8); fl.push_back(x); }
        } else if (f == 8 && name) *name = pb.str();
        else if (f == 9) { PB s = pb.sub(); raw.assign(s.p, s.p + s.n); }
        else if (f == 14 && pb.varint() == 1) fail(SLDRNC_ERROR_UNSUPPORTED, "ONNX weights stored in external files are not supported; re-export with weights inside the file");
        else pb.skip(wt);
    }
    if (!raw.empty()) {
        if (dtype == 1) { fl.resize(raw.size() / 4); for (size_t k = 0; k < fl.size(); k++) { float x; std::memcpy(&x, &raw[k * 4], 4); fl[k] = x; } }
        else if (dtype == 11) { fl.resize(raw.size() / 8); for (size_t k = 0; k < fl.size(); k++) { double x; std::memcpy(&x, &raw[k * 8], 8); fl[k] = x; } }
        else if (dtype == 7) { fl.resize(raw.size() / 8); for (size_t k = 0; k < fl.size(); k++) { int64_t x; std::memcpy(&x, &raw[k * 8], 8); fl[k] = (double)x; } }
        else fail(SLDRNC_ERROR_UNSUPPORTED, "ONNX tensor data type " + std::to_string(dtype) + " is not supported (use float32 or float64)");
    } else if (dtype != 0 && dtype != 1 && dtype != 7 && dtype != 11 && !fl.empty()) {
        fail(SLDRNC_ERROR_UNSUPPORTED, "ONNX tensor data type " + std::to_string(dtype) + " is not supported (use float32 or float64)");
    }
    t.v = std::move(fl);
    if (t.dims.empty() && t.v.size() > 1) t.dims.push_back((int64_t)t.v.size());
    return t;
}

struct Attr { std::string name; double f = 0; int64_t i = 0; std::string s; bool has_t = false; Tensor t; };
struct Node { std::string op; std::vector<std::string> in, out; std::vector<Attr> attr; };

Attr parse_attr(PB pb) {
    Attr a;
    while (pb.more()) {
        int f, wt;
        pb.key(f, wt);
        if (f == 1) a.name = pb.str();
        else if (f == 2) { const uint32_t b = pb.f32bits(); float x; std::memcpy(&x, &b, 4); a.f = x; }
        else if (f == 3) a.i = (int64_t)pb.varint();
        else if (f == 4) a.s = pb.str();
        else if (f == 5) { a.t = parse_tensor(pb.sub(), nullptr); a.has_t = true; }
        else pb.skip(wt);
    }
    return a;
}
Node parse_node(PB pb) {
    Node nd;
    while (pb.more()) {
        int f, wt;
        pb.key(f, wt);
        if (f == 1) nd.in.push_back(pb.str());
        else if (f == 2) nd.out.push_back(pb.str());
        else if (f == 4) nd.op = pb.str();
        else if (f == 5) nd.attr.push_back(parse_attr(pb.sub()));
        else pb.skip(wt);
    }
    return nd;
}
std::string value_info_name(PB pb) {
    std::string name;
    while (pb.more()) { int f, wt; pb.key(f, wt); if (f == 1) name = pb.str(); else pb.skip(wt); }
    return name;
}
const Attr* find_attr(const Node& n, const char* name) {
    for (const auto& a : n.attr) if (a.name == name) return &a;
    return nullptr;
}

// ------------------------------------------------------------------------------------------------ folding
struct Build {
    std::vector<int> in_dim, out_dim, act;
    std::vector<std::vector<double>> W, b;      // W: in x out
    bool open = false;                          // last layer still linear (no activation yet)
    int width = -1;                             // current chain width (-1 unknown = network input)
    std::vector<double> ps, pt;                 // pending elementwise affine x -> x * ps + pt on the chain
    bool pending = false;
    int input_width = -1;
    std::vector<double> in_scale, in_shift;    // input normalisation before the first layer (kept exact, not folded)

    static std::vector<double> bcast(const Tensor& c, int k, const char* what) {
        if (c.v.size() == 1) return std::vector<double>(k, c.v[0]);
        if ((int)c.v.size() != k) fail(SLDRNC_ERROR_UNSUPPORTED, std::string("ONNX ") + what + ": constant has " + std::to_string(c.v.size()) +
                                                                         " values, expected 1 or " + std::to_string(k));
        return c.v;
    }
    // apply x -> x * s + t to the chain
    void affine(const Tensor* s, const Tensor* t, const char* what) {
        if (open) {
            const int out = out_dim.back(), in = in_dim.back();
            if (s) { const auto sv = bcast(*s, out, what); for (int i = 0; i < in; i++) for (int o = 0; o < out; o++) W.back()[(size_t)i * out + o] *= sv[o]; for (int o = 0; o < out; o++) b.back()[o] *= sv[o]; }
            if (t) { const auto tv = bcast(*t, out, what); for (int o = 0; o < out; o++) b.back()[o] += tv[o]; }
            return;
        }
        int k = width;
        if (k < 0) {
            const Tensor* c = s ? s : t;
            if (c->v.size() == 1) fail(SLDRNC_ERROR_UNSUPPORTED, std::string("ONNX ") + what + " with a scalar before the first layer: the input width is unknown");
            k = (int)c->v.size();
            input_width = width = k;
        }
        if (!pending) { ps.assign(k, 1.0); pt.assign(k, 0.0); pending = true; }
        if (s) { const auto sv = bcast(*s, k, what); for (int i = 0; i < k; i++) { ps[i] *= sv[i]; pt[i] *= sv[i]; } }
        if (t) { const auto tv = bcast(*t, k, what); for (int i = 0; i < k; i++) pt[i] += tv[i]; }
    }
    void linear(const Tensor& Wt, bool transB, double alpha, const Tensor* bias, double beta) {
        if (Wt.dims.size() != 2) fail(SLDRNC_ERROR_UNSUPPORTED, "ONNX weight matrix must be 2-D");
        const int r = (int)Wt.dims[0], c = (int)Wt.dims[1];
        const int in = transB ? c : r, out = transB ? r : c;
        if (width >= 0 && width != in) fail(SLDRNC_ERROR_UNSUPPORTED, "ONNX layer expects " + std::to_string(in) + " inputs but receives " + std::to_string(width));
        if (open) { act.back() = SLDRNC_ACT_IDENTITY; open = false; }   // two linear maps in a row: keep both
        std::vector<double> Wn((size_t)in * out), bn(out, 0.0);
        for (int i = 0; i < in; i++) for (int o = 0; o < out; o++) Wn[(size_t)i * out + o] = alpha * (transB ? Wt.v[(size_t)o * c + i] : Wt.v[(size_t)i * c + o]);
        if (bias) { const auto bv = bcast(*bias, out, "Gemm bias"); for (int o = 0; o < out; o++) bn[o] = beta * bv[o]; }
        if (pending) {
            bool invertible = true;
            for (double v : ps) invertible = invertible && v != 0.0;
            if (W.empty() && invertible) {
                // before the first layer: keep it as input normalisation applied in double precision (folding a large mean
                // into float32 weights cancels catastrophically for inputs far from zero)
                in_scale = ps; in_shift = pt;
            } else {   // fold: (x*s + t) W = x (diag(s) W) + t W
                for (int o = 0; o < out; o++) { double acc = 0; for (int i = 0; i < in; i++) acc += pt[i] * Wn[(size_t)i * out + o]; bn[o] += acc; }
                for (int i = 0; i < in; i++) for (int o = 0; o < out; o++) Wn[(size_t)i * out + o] *= ps[i];
            }
            pending = false;
        }
        if (input_width < 0) input_width = in;
        in_dim.push_back(in); out_dim.push_back(out); act.push_back(SLDRNC_ACT_IDENTITY);
        W.push_back(std::move(Wn)); b.push_back(std::move(bn));
        open = true;
        width = out;
    }
    void activation(int a, const char* op) {
        if (pending) fail(SLDRNC_ERROR_UNSUPPORTED, std::string("ONNX ") + op + " after a scaling that cannot be folded into a layer");
        const bool derived = a == SLDRNC_ACT_RELU2 || a == SLDRNC_ACT_SILU;   // only valid as the second half of a pattern
        if (open && derived) fail(SLDRNC_ERROR_UNSUPPORTED, std::string("ONNX ") + op + " is supported only after Relu (x*x, Pow 2) or Sigmoid (x*sigmoid(x))");
        if (!open) {
            if (!act.empty() && act.back() == SLDRNC_ACT_RELU && a == SLDRNC_ACT_RELU2) { act.back() = SLDRNC_ACT_RELU2; return; }
            if (!act.empty() && act.back() == SLDRNC_ACT_SIGMOID && a == SLDRNC_ACT_SILU) { act.back() = SLDRNC_ACT_SILU; return; }
            fail(SLDRNC_ERROR_UNSUPPORTED, std::string("ONNX ") + op + " is not directly after a layer");
        }
        act.back() = a;
        open = false;
    }
};

}  // namespace

std::unique_ptr<Model> import_onnx(const uint8_t* p, size_t n) {
    PB model{p, n};
    PB graph{nullptr, 0};
    bool have_graph = false;
    while (model.more()) {
        int f, wt;
        model.key(f, wt);
        if (f == 7 && wt == 2) { graph = model.sub(); have_graph = true; }
        else model.skip(wt);
    }
    if (!have_graph) fail(SLDRNC_ERROR_FORMAT, "not an ONNX model (no graph found)");
    std::vector<Node> nodes;
    std::map<std::string, Tensor> consts;
    std::vector<std::string> inputs, outputs;
    while (graph.more()) {
        int f, wt;
        graph.key(f, wt);
        if (f == 1) nodes.push_back(parse_node(graph.sub()));
        else if (f == 5) { std::string name; Tensor t = parse_tensor(graph.sub(), &name); consts[name] = std::move(t); }
        else if (f == 11) inputs.push_back(value_info_name(graph.sub()));
        else if (f == 12) outputs.push_back(value_info_name(graph.sub()));
        else graph.skip(wt);
    }
    std::vector<std::string> real_inputs;
    for (const auto& s : inputs) if (!consts.count(s)) real_inputs.push_back(s);
    if (real_inputs.size() != 1 || outputs.size() != 1)
        fail(SLDRNC_ERROR_UNSUPPORTED, "ONNX graph must have exactly one input and one output (has " + std::to_string(real_inputs.size()) + " / " +
                                           std::to_string(outputs.size()) + ")");
    Build bd;
    std::string chain = real_inputs[0], before_sigmoid;
    auto is_const = [&](const std::string& s) { return consts.count(s) > 0; };
    for (const Node& nd : nodes) {
        const std::string& op = nd.op;
        if (op == "Constant") {
            const Attr* a = find_attr(nd, "value");
            if (!a || !a->has_t) {
                const Attr* fv = find_attr(nd, "value_float");
                if (!fv) fail(SLDRNC_ERROR_UNSUPPORTED, "ONNX Constant without a tensor value");
                Tensor t; t.v = {fv->f}; consts[nd.out[0]] = t;
            } else consts[nd.out[0]] = a->t;
            continue;
        }
        // which input is the chain?
        int ci = -1;
        for (size_t k = 0; k < nd.in.size(); k++) if (nd.in[k] == chain) { ci = (int)k; break; }
        if (ci < 0) {
            if (op == "Shape" || op == "Gather" || op == "Unsqueeze" || op == "Concat") continue;   // shape bookkeeping
            fail(SLDRNC_ERROR_UNSUPPORTED, "ONNX node '" + op + "' is not on the input-to-output chain (branching graphs are not supported)");
        }
        auto cst = [&](const std::string& s) -> const Tensor& {
            if (!is_const(s)) fail(SLDRNC_ERROR_UNSUPPORTED, "ONNX " + op + ": the second operand must be a constant (weights / biases / scales)");
            return consts[s];
        };
        if (op == "Gemm") {
            const Attr* ta = find_attr(nd, "transA");
            if (ta && ta->i) fail(SLDRNC_ERROR_UNSUPPORTED, "ONNX Gemm with transA=1 is not supported");
            const Attr* tb = find_attr(nd, "transB");
            const Attr* al = find_attr(nd, "alpha");
            const Attr* be = find_attr(nd, "beta");
            const Tensor* bias = nd.in.size() > 2 && !nd.in[2].empty() ? &cst(nd.in[2]) : nullptr;
            bd.linear(cst(nd.in[1]), tb && tb->i, al ? al->f : 1.0, bias, be ? be->f : 1.0);
        } else if (op == "MatMul") {
            if (ci != 0) fail(SLDRNC_ERROR_UNSUPPORTED, "ONNX MatMul must multiply the activations by the weights (x @ W)");
            bd.linear(cst(nd.in[1]), false, 1.0, nullptr, 1.0);
        } else if (op == "Add") {
            bd.affine(nullptr, &cst(nd.in[1 - ci]), "Add");
        } else if (op == "Sub") {
            Tensor t = cst(nd.in[1 - ci]);
            if (ci == 0) { for (auto& v : t.v) v = -v; bd.affine(nullptr, &t, "Sub"); }
            else { Tensor m1; m1.v = {-1.0}; bd.affine(&m1, &t, "Sub"); }     // c - x
        } else if (op == "Mul") {
            if (nd.in[0] == nd.in[1]) bd.activation(SLDRNC_ACT_RELU2, "Mul(x, x)");
            else if (!before_sigmoid.empty() && (nd.in[0] == before_sigmoid || nd.in[1] == before_sigmoid)) bd.activation(SLDRNC_ACT_SILU, "x*sigmoid(x)");
            else bd.affine(&cst(nd.in[1 - ci]), nullptr, "Mul");
        } else if (op == "Div") {
            if (ci != 0) fail(SLDRNC_ERROR_UNSUPPORTED, "ONNX Div of a constant by the activations is not supported");
            Tensor t = cst(nd.in[1]);
            for (auto& v : t.v) { if (v == 0) fail(SLDRNC_ERROR_UNSUPPORTED, "ONNX Div by zero"); v = 1.0 / v; }
            bd.affine(&t, nullptr, "Div");
        } else if (op == "Pow") {
            const Tensor& e = cst(nd.in[1]);
            if (e.v.size() != 1 || e.v[0] != 2.0) fail(SLDRNC_ERROR_UNSUPPORTED, "ONNX Pow is supported only with exponent 2 after Relu");
            bd.activation(SLDRNC_ACT_RELU2, "Pow");
        } else if (op == "Relu") bd.activation(SLDRNC_ACT_RELU, "Relu");
        else if (op == "LeakyRelu") {
            const Attr* a = find_attr(nd, "alpha");
            if (a && std::fabs(a->f - 0.01) > 1e-6) fail(SLDRNC_ERROR_UNSUPPORTED, "ONNX LeakyRelu is supported with alpha 0.01 only");
            bd.activation(SLDRNC_ACT_LEAKY_RELU, "LeakyRelu");
        } else if (op == "Tanh") bd.activation(SLDRNC_ACT_TANH, "Tanh");
        else if (op == "Sigmoid") {
            before_sigmoid = chain;
            bd.activation(SLDRNC_ACT_SIGMOID, "Sigmoid");
            chain = nd.out[0];
            continue;
        } else if (op == "Gelu") {
            const Attr* a = find_attr(nd, "approximate");
            if (!a || a->s != "tanh") fail(SLDRNC_ERROR_UNSUPPORTED, "ONNX Gelu is supported with approximate=\"tanh\" only");
            bd.activation(SLDRNC_ACT_GELU, "Gelu");
        } else if (op == "Identity" || op == "Dropout" || op == "Flatten" || op == "Reshape" || op == "Cast" || op == "Squeeze") {
            // shape-only on a vector input
        } else {
            fail(SLDRNC_ERROR_UNSUPPORTED, "ONNX operation '" + op + "' is not supported (supported: Gemm, MatMul, Add, Sub, Mul, Div, "
                                         "Relu, LeakyRelu, Tanh, Sigmoid, Gelu(tanh), Pow(2), Identity, Dropout, Flatten, Reshape, Cast)");
        }
        before_sigmoid = op == "Mul" ? std::string() : before_sigmoid;
        chain = nd.out[0];
    }
    if (chain != outputs[0]) fail(SLDRNC_ERROR_UNSUPPORTED, "ONNX graph output is not the end of the layer chain");
    if (bd.W.empty()) fail(SLDRNC_ERROR_UNSUPPORTED, "ONNX graph has no dense layers");
    auto to_float = [](const std::vector<double>& v) {
        std::vector<float> f(v.size());
        for (size_t i = 0; i < v.size(); i++) f[i] = (float)v[i];
        return f;
    };
    auto m = std::make_unique<Model>();
    m->mode = SLDRNC_MODE_RAW;
    m->net.L.resize(bd.W.size());
    for (size_t l = 0; l < bd.W.size(); l++) {
        const std::vector<float> Wf = to_float(bd.W[l]), bf = to_float(bd.b[l]);
        set_layer(m->net.L[l], bd.in_dim[l], bd.out_dim[l], bd.act[l], Wf.data(), bf.data());
    }
    if (!bd.in_scale.empty()) {   // x -> x * s + t  ==  (x - mean) / std  with  std = 1 / s, mean = -t / s
        m->in_std.resize(bd.in_scale.size());
        m->in_mean.resize(bd.in_scale.size());
        for (size_t i = 0; i < bd.in_scale.size(); i++) { m->in_std[i] = (float)(1.0 / bd.in_scale[i]); m->in_mean[i] = (float)(-bd.in_shift[i] / bd.in_scale[i]); }
    }
    if (bd.pending) {   // trailing scaling -> output scaling
        m->out_std = to_float(bd.ps);
        m->out_mean = to_float(bd.pt);
        if ((int)m->out_std.size() != m->net.out()) fail(SLDRNC_ERROR_UNSUPPORTED, "ONNX trailing scaling does not match the output width");
    }
    return m;
}

}  // namespace sldi
