// model_io.cpp -- the .sldm model file.
//   header:  "SLDRNC" 0x01 0x00 | u32 format version | u32 flags | 12-byte nonce | u64 payload bytes
//   payload: the model fields, ChaCha20-scrambled (key built into the library, fresh nonce per file)
//   trailer: u32 CRC-32 of the unscrambled payload
// The scrambling keeps weights and structure out of plain sight; it is not a substitute for licence protection.
#include <cstring>
#include <random>

#include "internal.hpp"

namespace sldi {

static const uint8_t MAGIC[8] = {'S', 'L', 'D', 'R', 'N', 'C', 1, 0};
static const uint32_t FORMAT = 3;   // 2: + the correction's tuned delay; 3: + history / affine channels, input ranges, table.
                                    // Older files still load (delay 1, history on, no affine channels, no table).

// ------------------------------------------------------------------------------------------------ CRC-32, ChaCha20
struct CrcTable {
    uint32_t t[256];
    CrcTable() {
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t c = i;
            for (int k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            t[i] = c;
        }
    }
};
static uint32_t crc32(const uint8_t* p, size_t n) {
    static const CrcTable table;                               // thread-safe one-time initialisation
    const uint32_t* tab = table.t;
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; i++) c = tab[(c ^ p[i]) & 0xFF] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}
static inline uint32_t rotl(uint32_t v, int c) { return (v << c) | (v >> (32 - c)); }
static void chacha_block(const uint32_t key[8], uint32_t counter, const uint32_t nonce[3], uint8_t out[64]) {
    uint32_t s[16] = {0x61707865, 0x3320646e, 0x79622d32, 0x6b206574, key[0], key[1], key[2], key[3], key[4], key[5], key[6], key[7],
                      counter, nonce[0], nonce[1], nonce[2]};
    uint32_t x[16];
    std::memcpy(x, s, sizeof x);
    auto qr = [&](int a, int b, int c, int d) {
        x[a] += x[b]; x[d] = rotl(x[d] ^ x[a], 16);
        x[c] += x[d]; x[b] = rotl(x[b] ^ x[c], 12);
        x[a] += x[b]; x[d] = rotl(x[d] ^ x[a], 8);
        x[c] += x[d]; x[b] = rotl(x[b] ^ x[c], 7);
    };
    for (int i = 0; i < 10; i++) {
        qr(0, 4, 8, 12); qr(1, 5, 9, 13); qr(2, 6, 10, 14); qr(3, 7, 11, 15);
        qr(0, 5, 10, 15); qr(1, 6, 11, 12); qr(2, 7, 8, 13); qr(3, 4, 9, 14);
    }
    for (int i = 0; i < 16; i++) {
        const uint32_t v = x[i] + s[i];
        out[4 * i] = (uint8_t)v; out[4 * i + 1] = (uint8_t)(v >> 8); out[4 * i + 2] = (uint8_t)(v >> 16); out[4 * i + 3] = (uint8_t)(v >> 24);
    }
}
static void scramble(uint8_t* p, size_t n, const uint8_t nonce_b[12]) {
    uint32_t key[8];
    uint64_t z = 0x5D1F2A7C93E40B68ull;                       // built-in key material
    for (int i = 0; i < 8; i++) { z = z * 6364136223846793005ull + 1442695040888963407ull; key[i] = (uint32_t)(z >> 29); }
    uint32_t nonce[3];
    std::memcpy(nonce, nonce_b, 12);
    uint8_t ks[64];
    for (size_t off = 0, blk = 1; off < n; off += 64, blk++) {
        chacha_block(key, (uint32_t)blk, nonce, ks);
        const size_t m = n - off < 64 ? n - off : 64;
        for (size_t i = 0; i < m; i++) p[off + i] ^= ks[i];
    }
}

// ------------------------------------------------------------------------------------------------ serialisation
struct W {
    std::vector<uint8_t> b;
    void raw(const void* p, size_t n) { const uint8_t* q = (const uint8_t*)p; b.insert(b.end(), q, q + n); }
    void u32(uint32_t v) { raw(&v, 4); }
    void i32(int32_t v) { raw(&v, 4); }
    void f64(double v) { raw(&v, 8); }
    void str(const std::string& s) { u32((uint32_t)s.size()); raw(s.data(), s.size()); }
    void vf(const std::vector<float>& v) { u32((uint32_t)v.size()); raw(v.data(), v.size() * 4); }
    void vd(const std::vector<double>& v) { u32((uint32_t)v.size()); raw(v.data(), v.size() * 8); }
    void vi(const std::vector<int>& v) { u32((uint32_t)v.size()); for (int x : v) i32(x); }
};
struct R {
    const uint8_t* p; size_t n, i = 0;
    void raw(void* d, size_t k) {
        if (i + k > n) fail(SLDRNC_ERROR_FORMAT, "model file is truncated or damaged");
        std::memcpy(d, p + i, k); i += k;
    }
    uint32_t u32() { uint32_t v; raw(&v, 4); return v; }
    int32_t i32() { int32_t v; raw(&v, 4); return v; }
    double f64() { double v; raw(&v, 8); return v; }
    uint32_t count(size_t elem) { const uint32_t c = u32(); if ((size_t)c * elem > n - i) fail(SLDRNC_ERROR_FORMAT, "model file is damaged"); return c; }
    std::string str() { const uint32_t c = count(1); std::string s(c, '\0'); raw(&s[0], c); return s; }
    std::vector<float> vf() { const uint32_t c = count(4); std::vector<float> v(c); raw(v.data(), (size_t)c * 4); return v; }
    std::vector<double> vd() { const uint32_t c = count(8); std::vector<double> v(c); raw(v.data(), (size_t)c * 8); return v; }
    std::vector<int> vi() { const uint32_t c = count(4); std::vector<int> v(c); for (auto& x : v) x = i32(); return v; }
};

void save_model(const Model& m, std::vector<uint8_t>& out) {
    W w;
    w.i32(m.mode);
    w.f64(m.rate);
    w.f64(m.schema.rate);
    w.u32((uint32_t)m.schema.ch.size());
    for (const auto& c : m.schema.ch) { w.str(c.name); w.i32(c.kind); w.i32(c.size); }
    w.i32(m.schema.n_out);
    w.vi(m.schema.vel_idx);
    w.vf(m.in_mean); w.vf(m.in_std);
    w.u32((uint32_t)m.net.L.size());
    for (const auto& l : m.net.L) {
        w.i32(l.in); w.i32(l.out); w.i32(l.act);
        std::vector<float> Wu((size_t)l.in * l.out);
        for (int i = 0; i < l.in; i++) std::memcpy(&Wu[(size_t)i * l.out], &l.W[(size_t)i * l.outp], sizeof(float) * l.out);
        w.vf(Wu);
        w.vf(std::vector<float>(l.b.begin(), l.b.begin() + l.out));
    }
    w.vf(m.out_mean); w.vf(m.out_std);
    const Correction& c = m.corr;
    w.i32(c.on ? 1 : 0); w.f64(c.tau_ticks); w.vi(c.vel_idx); w.vd(c.vel_sd); w.vd(c.res_sd); w.vd(c.mean); w.vd(c.sd);
    w.f64(c.gate); w.f64(c.clamp); w.i32(c.frozen_ticks);
    w.i32(c.delay_ticks);
    // format 3
    w.i32(m.schema.history ? 1 : 0);
    std::vector<int> aff;
    for (const auto& ch : m.schema.ch) aff.push_back(ch.affine ? 1 : 0);
    w.vi(aff);
    w.vf(m.raw_lo); w.vf(m.raw_hi);
    const Table& t = m.table;
    w.i32(t.dims);
    if (t.dims > 0) {
        for (int d = 0; d < 3; d++) {
            w.i32(t.n[d]); w.i32(t.col[d]); w.i32(t.periodic[d]); w.i32(t.stride[d]);
            w.f64(t.lo[d]); w.f64(t.scale[d]); w.f64(t.umax[d]); w.f64(t.nfl[d]); w.f64(t.inv_n[d]);
        }
        w.i32(t.hp);
        w.vf(t.T);
    }
    uint8_t nonce[12];
    std::random_device rd;
    for (int i = 0; i < 12; i += 4) { const uint32_t v = rd(); std::memcpy(nonce + i, &v, 4); }
    const uint32_t crc = crc32(w.b.data(), w.b.size());
    std::vector<uint8_t> payload = w.b;
    scramble(payload.data(), payload.size(), nonce);
    out.clear();
    out.insert(out.end(), MAGIC, MAGIC + 8);
    auto put = [&](const void* p, size_t n) { const uint8_t* q = (const uint8_t*)p; out.insert(out.end(), q, q + n); };
    const uint32_t fmt = FORMAT, flags = 0;
    const uint64_t pn = payload.size();
    put(&fmt, 4); put(&flags, 4); put(nonce, 12); put(&pn, 8);
    out.insert(out.end(), payload.begin(), payload.end());
    put(&crc, 4);
}

std::unique_ptr<Model> load_model(const uint8_t* p, size_t n) {
    if (n < 40 || std::memcmp(p, MAGIC, 8) != 0) fail(SLDRNC_ERROR_FORMAT, "not an SLD-RNC model file");
    uint32_t fmt;
    std::memcpy(&fmt, p + 8, 4);
    if (fmt > FORMAT) fail(SLDRNC_ERROR_FORMAT, "model file is from a newer SLD-RNC version (format " + std::to_string(fmt) + ")");
    uint8_t nonce[12];
    std::memcpy(nonce, p + 16, 12);
    uint64_t pn;
    std::memcpy(&pn, p + 28, 8);
    if (36 + pn + 4 != n) fail(SLDRNC_ERROR_FORMAT, "model file is truncated or damaged");
    std::vector<uint8_t> pl(p + 36, p + 36 + pn);
    scramble(pl.data(), pl.size(), nonce);
    uint32_t crc;
    std::memcpy(&crc, p + 36 + pn, 4);
    if (crc32(pl.data(), pl.size()) != crc) fail(SLDRNC_ERROR_FORMAT, "model file is damaged (checksum mismatch)");
    R r{pl.data(), pl.size()};
    auto m = std::make_unique<Model>();
    m->mode = r.i32();
    m->rate = r.f64();
    m->schema.rate = r.f64();
    const uint32_t nch = r.count(12);
    int off = 0;
    for (uint32_t i = 0; i < nch; i++) {
        Channel c;
        c.name = r.str(); c.kind = r.i32(); c.size = r.i32(); c.offset = off;
        if (c.kind < SLDRNC_INPUT_POSITION || c.kind > SLDRNC_INPUT_ANGLE)
            fail(SLDRNC_ERROR_FORMAT, "model file uses an input kind this library version does not know (" + std::to_string(c.kind) + "); it needs a newer version");
        if (c.size <= 0) fail(SLDRNC_ERROR_FORMAT, "model file is damaged (channel size)");
        off += c.size;
        m->schema.ch.push_back(c);
    }
    m->schema.width = off;
    m->schema.n_out = r.i32();
    m->schema.vel_idx = r.vi();
    m->in_mean = r.vf(); m->in_std = r.vf();
    const uint32_t nl = r.count(12);
    m->net.L.resize(nl);
    for (uint32_t i = 0; i < nl; i++) {
        const int in = r.i32(), out = r.i32(), act = r.i32();
        const std::vector<float> Wu = r.vf(), b = r.vf();
        if (in <= 0 || out <= 0 || Wu.size() != (size_t)in * out || b.size() != (size_t)out) fail(SLDRNC_ERROR_FORMAT, "model file is damaged (layer shape)");
        set_layer(m->net.L[i], in, out, act, Wu.data(), b.data());
    }
    m->out_mean = r.vf(); m->out_std = r.vf();
    Correction& c = m->corr;
    c.on = r.i32() != 0; c.tau_ticks = r.f64(); c.vel_idx = r.vi(); c.vel_sd = r.vd(); c.res_sd = r.vd(); c.mean = r.vd(); c.sd = r.vd();
    c.gate = r.f64(); c.clamp = r.f64(); c.frozen_ticks = r.i32();
    c.delay_ticks = fmt >= 2 ? r.i32() : 1;
    if (c.delay_ticks < 1) fail(SLDRNC_ERROR_FORMAT, "model file is damaged (correction delay)");
    if (fmt >= 3) {
        m->schema.history = r.i32() != 0;
        const std::vector<int> aff = r.vi();
        if (aff.size() != m->schema.ch.size()) fail(SLDRNC_ERROR_FORMAT, "model file is damaged (affine channels)");
        for (size_t k = 0; k < aff.size(); k++) m->schema.ch[k].affine = aff[k] != 0;
        m->raw_lo = r.vf(); m->raw_hi = r.vf();
        Table& t = m->table;
        t.dims = r.i32();
        if (t.dims < 0 || t.dims > 3) fail(SLDRNC_ERROR_FORMAT, "model file is damaged (table)");
        if (t.dims > 0) {
            for (int d = 0; d < 3; d++) {
                t.n[d] = r.i32(); t.col[d] = r.i32(); t.periodic[d] = r.i32(); t.stride[d] = r.i32();
                t.lo[d] = (float)r.f64(); t.scale[d] = (float)r.f64(); t.umax[d] = (float)r.f64(); t.nfl[d] = (float)r.f64(); t.inv_n[d] = (float)r.f64();
                if (t.n[d] < 2 || t.col[d] < -1 || t.col[d] >= m->schema.width) fail(SLDRNC_ERROR_FORMAT, "model file is damaged (table)");
            }
            t.hp = r.i32();
            t.T = r.vf();
            if (t.hp < 8 || t.hp % 8 || t.stride[2] != 1 || t.stride[1] != t.n[2] || t.stride[0] != t.n[1] * t.n[2] ||
                t.T.size() != t.nodes() * (size_t)t.hp)
                fail(SLDRNC_ERROR_FORMAT, "model file is damaged (table)");
        }
    }
    if (m->mode == SLDRNC_MODE_FULL) m->fs = make_featspec(m->schema);
    setup_structure(*m);
    return m;
}

}  // namespace sldi
