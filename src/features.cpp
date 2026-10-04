// features.cpp -- FULL mode's input recipe: causal features from the raw signals of each tick.
// Per channel kind (n = channel size):
//   POSITION      value                                                                    n
//   ANGLE         value, sine, cosine (rotary joints, radians)                             3 n
//   VELOCITY      value, derived acceleration, both at t-1..t-4, value at t-8/16/32/64 ms,
//                 exponential averages at 10/40/160/640 ms                                  18 n
//   ACCELERATION  value, and at t-1..t-4                                                     5 n
//   GYRO          value, derived angular acceleration (5-ms smoothed), averages              6 n
//   ACCEL         value, averages                                                            5 n
//   QUATERNION    gravity direction (3), roll, pitch                                         5
//   GENERIC       value, averages                                                            5 n
// History before the first tick repeats the first sample; averages start at the first sample.
// Without history (Schema::history false) every kind gives its current value only (ANGLE: value, sine, cosine;
// QUATERNION: its 5 derived values), so rows can be unordered samples and there is no warm-up.
#include <algorithm>
#include <cmath>
#include <cstring>

#include "internal.hpp"

namespace sldi {

static const double LAG_MS[4] = {8, 16, 32, 64};
static const double BANK_MS[4] = {10, 40, 160, 640};

FeatSpec make_featspec(const Schema& s) {
    FeatSpec f;
    f.rate = s.rate;
    for (int k = 0; k < 4; k++) {
        f.lag[k] = std::max(1, (int)std::lround(LAG_MS[k] * s.rate / 1000.0));
        f.max_lag = std::max(f.max_lag, f.lag[k]);
        f.bank_a[k] = 1.0 / std::max(1.0, BANK_MS[k] * s.rate / 1000.0);
    }
    f.gyro_a = 1.0 / std::max(1.0, 5.0 * s.rate / 1000.0);
    f.history = s.history;
    int nf = 0;
    for (const auto& c : s.ch) {
        f.ch_feat.push_back(nf);
        if (!s.history) {
            nf += c.kind == SLDRNC_INPUT_ANGLE ? 3 * c.size : c.kind == SLDRNC_INPUT_QUATERNION ? 5 : c.size;
            continue;
        }
        switch (c.kind) {
            case SLDRNC_INPUT_POSITION: nf += c.size; break;
            case SLDRNC_INPUT_ANGLE: nf += 3 * c.size; break;
            case SLDRNC_INPUT_VELOCITY: nf += 18 * c.size; break;
            case SLDRNC_INPUT_ACCELERATION: nf += 5 * c.size; break;
            case SLDRNC_INPUT_GYRO: nf += 6 * c.size; break;
            case SLDRNC_INPUT_ACCEL: nf += 5 * c.size; break;
            case SLDRNC_INPUT_QUATERNION: nf += 5; break;
            default: nf += 5 * c.size; break;
        }
    }
    f.nf = nf;
    return f;
}

// state words per channel
static int ring_depth(const FeatSpec& f) { return std::max(f.max_lag, 4); }
static size_t state_words(const Channel& c, const FeatSpec& f) {
    if (!f.history) return 0;
    const size_t n = c.size;
    switch (c.kind) {
        case SLDRNC_INPUT_VELOCITY: return n * ring_depth(f) + 4 * n + n + 4 * n;   // v ring, acc ring, prev v, bank
        case SLDRNC_INPUT_ACCELERATION: return 4 * n;
        case SLDRNC_INPUT_GYRO: return n + n + 4 * n;                                // smoothed, prev smoothed, bank
        case SLDRNC_INPUT_ACCEL: return 4 * n;
        case SLDRNC_INPUT_GENERIC: return 4 * n;
        default: return 0;
    }
}

void feat_init(const Schema& s, const FeatSpec& f, FeatState& st) {
    size_t w = 0;
    for (const auto& c : s.ch) w += state_words(c, f);
    st.mem.assign(w, 0.0);
    st.started = false;
    st.pos = 0;
}

static inline void ewma(double a, bool first, double x, double& z, double& y) {
    if (first) z = (1.0 - a) * x;
    y = a * x + z;
    z = (1.0 - a) * y;
}

static void feat_instant(const Schema& s, const double* x, double* o) {
    for (const auto& c : s.ch) {
        const double* v = x + c.offset;
        const int n = c.size;
        if (c.kind == SLDRNC_INPUT_ANGLE) {
            for (int i = 0; i < n; i++) *o++ = v[i];
            for (int i = 0; i < n; i++) *o++ = std::sin(v[i]);
            for (int i = 0; i < n; i++) *o++ = std::cos(v[i]);
        } else if (c.kind == SLDRNC_INPUT_QUATERNION) {
            const double w = v[0], qx = v[1], qy = v[2], qz = v[3];
            *o++ = 2 * (qx * qz - w * qy);
            *o++ = 2 * (qy * qz + w * qx);
            *o++ = 1 - 2 * (qx * qx + qy * qy);
            *o++ = std::atan2(2 * (w * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy));
            double sp = 2 * (w * qy - qz * qx);
            sp = sp < -1 ? -1 : (sp > 1 ? 1 : sp);
            *o++ = std::asin(sp);
        } else {
            for (int i = 0; i < n; i++) *o++ = v[i];
        }
    }
}

void feat_step(const Schema& s, const FeatSpec& f, FeatState& st, const double* x, double* out) {
    if (!f.history) {
        feat_instant(s, x, out);
        st.started = true;
        return;
    }
    const bool first = !st.started;
    const int D = ring_depth(f);
    const uint32_t t = (uint32_t)st.pos;
    auto slot = [&](int k, int depth) { return (int)((t + (uint32_t)depth * 1024u - (uint32_t)k) % (uint32_t)depth); };
    double* m = st.mem.data();
    double* o = out;
    for (const auto& c : s.ch) {
        const double* v = x + c.offset;
        const int n = c.size;
        switch (c.kind) {
            case SLDRNC_INPUT_POSITION:
                for (int i = 0; i < n; i++) *o++ = v[i];
                break;
            case SLDRNC_INPUT_ANGLE:
                for (int i = 0; i < n; i++) *o++ = v[i];
                for (int i = 0; i < n; i++) *o++ = std::sin(v[i]);
                for (int i = 0; i < n; i++) *o++ = std::cos(v[i]);
                break;
            case SLDRNC_INPUT_VELOCITY: {
                double* ring = m;
                double* aring = ring + (size_t)n * D;
                double* prev = aring + 4 * n;
                double* bank = prev + n;
                if (first)
                    for (int k = 0; k < D; k++) for (int i = 0; i < n; i++) ring[(size_t)k * n + i] = v[i];
                double* acc = o + n;
                for (int i = 0; i < n; i++) { o[i] = v[i]; acc[i] = first ? 0.0 : (v[i] - prev[i]) * f.rate; }
                if (first) for (int k = 0; k < 4; k++) for (int i = 0; i < n; i++) aring[k * n + i] = acc[i];
                o += 2 * n;
                for (int k = 1; k <= 4; k++) {
                    const double* rv = ring + (size_t)slot(k, D) * n;
                    const double* ra = aring + (size_t)slot(k, 4) * n;
                    for (int i = 0; i < n; i++) *o++ = rv[i];
                    for (int i = 0; i < n; i++) *o++ = ra[i];
                }
                for (int k = 0; k < 4; k++) {
                    const double* rv = ring + (size_t)slot(f.lag[k], D) * n;
                    for (int i = 0; i < n; i++) *o++ = rv[i];
                }
                for (int k = 0; k < 4; k++)
                    for (int i = 0; i < n; i++) ewma(f.bank_a[k], first, v[i], bank[k * n + i], *o++);
                // state update
                double* wv = ring + (size_t)slot(0, D) * n;
                double* wa = aring + (size_t)slot(0, 4) * n;
                for (int i = 0; i < n; i++) { wv[i] = v[i]; wa[i] = acc[i]; prev[i] = v[i]; }
                m += state_words(c, f);
                break;
            }
            case SLDRNC_INPUT_ACCELERATION: {
                double* ring = m;
                if (first) for (int k = 0; k < 4; k++) for (int i = 0; i < n; i++) ring[k * n + i] = v[i];
                for (int i = 0; i < n; i++) *o++ = v[i];
                for (int k = 1; k <= 4; k++) {
                    const double* r = ring + (size_t)slot(k, 4) * n;
                    for (int i = 0; i < n; i++) *o++ = r[i];
                }
                double* w = ring + (size_t)slot(0, 4) * n;
                for (int i = 0; i < n; i++) w[i] = v[i];
                m += state_words(c, f);
                break;
            }
            case SLDRNC_INPUT_GYRO: {
                double* sm = m;
                double* prev = sm + n;
                double* bank = prev + n;
                for (int i = 0; i < n; i++) *o++ = v[i];
                for (int i = 0; i < n; i++) {
                    double ys;
                    double z = sm[i];
                    ewma(f.gyro_a, first, v[i], z, ys);
                    sm[i] = z;
                    *o++ = first ? 0.0 : (ys - prev[i]) * f.rate;
                    prev[i] = ys;
                }
                for (int k = 0; k < 4; k++)
                    for (int i = 0; i < n; i++) ewma(f.bank_a[k], first, v[i], bank[k * n + i], *o++);
                m += state_words(c, f);
                break;
            }
            case SLDRNC_INPUT_QUATERNION: {
                const double w = v[0], qx = v[1], qy = v[2], qz = v[3];
                *o++ = 2 * (qx * qz - w * qy);
                *o++ = 2 * (qy * qz + w * qx);
                *o++ = 1 - 2 * (qx * qx + qy * qy);
                *o++ = std::atan2(2 * (w * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy));
                double sp = 2 * (w * qy - qz * qx);
                sp = sp < -1 ? -1 : (sp > 1 ? 1 : sp);
                *o++ = std::asin(sp);
                break;
            }
            default: {   // ACCEL and GENERIC: value + averages
                double* bank = m;
                for (int i = 0; i < n; i++) *o++ = v[i];
                for (int k = 0; k < 4; k++)
                    for (int i = 0; i < n; i++) ewma(f.bank_a[k], first, v[i], bank[k * n + i], *o++);
                m += state_words(c, f);
                break;
            }
        }
    }
    st.started = true;
    st.pos = (int)((t + 1u) % (uint32_t)(D * 4 * 1024));
}

}  // namespace sldi
