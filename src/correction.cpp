// correction.cpp -- self-correction from measured outputs (per output: a small Kalman filter on four slowly varying
// terms: offset, Coulomb-like and viscous-like friction from the joint's velocity, and gain on the prediction), plus the
// protection filter (innovation gate + magnitude clamp).  Works in standardised output units.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

#include "internal.hpp"

namespace sldi {

void corr_init(const Correction& c, CorrState& s, int n) {
    s.th.assign((size_t)n * 4, 0.0);
    s.P.assign((size_t)n * 16, 0.0);
    for (int j = 0; j < n; j++) for (int k = 0; k < 4; k++) s.P[(size_t)j * 16 + k * 5] = 0.1;
    s.h.assign((size_t)n * 4, 0.0);
    s.o.assign(n, 0.0);
    s.rejected.assign(n, 0);
    s.last_meas.assign(n, std::numeric_limits<double>::quiet_NaN());
    s.same.assign(n, 0);
    s.valid.assign(n, 1);
    s.primed = false;
    (void)c;
}

void corr_regress(const Correction& c, const double* o_std, const double* x, double* h, int n) {
    for (int j = 0; j < n; j++) {
        double* hj = h + 4 * j;
        hj[0] = 1.0;
        const int vi = c.vel_idx.empty() ? -1 : c.vel_idx[j];
        if (vi >= 0) {
            const double v = x[vi], sd = c.vel_sd[j];
            hj[1] = std::tanh(v / (0.14 * sd));
            hj[2] = v / sd;
        } else {
            hj[1] = hj[2] = 0.0;
        }
        hj[3] = o_std[j];
    }
}

void corr_apply(const Correction& c, const CorrState& s, const double* h, double* corr, int n, bool protect) {
    for (int j = 0; j < n; j++) {
        const double* hj = h + 4 * j;
        const double* t = &s.th[(size_t)j * 4];
        double v = hj[0] * t[0] + hj[1] * t[1] + hj[2] * t[2] + hj[3] * t[3];
        if (protect) v = std::min(c.clamp, std::max(-c.clamp, v));
        corr[j] = v;
    }
}

const unsigned char* corr_screen(const Correction& c, CorrState& s, const double* y_raw, int n) {
    for (int j = 0; j < n; j++) {
        s.same[j] = y_raw[j] == s.last_meas[j] ? s.same[j] + 1 : 0;
        s.last_meas[j] = y_raw[j];
        s.valid[j] = s.same[j] < c.frozen_ticks ? 1 : 0;   // repeating exactly for too long: frozen sensor, ignore
    }
    return s.valid.data();
}

void corr_update(const Correction& c, CorrState& s, const double* h, const double* r, int n, bool protect, const unsigned char* valid) {
    const double q = 1.0 / (c.tau_ticks * c.tau_ticks);
    for (int j = 0; j < n; j++) {
        const double* hj = h + 4 * j;
        double* t = &s.th[(size_t)j * 4];
        double* P = &s.P[(size_t)j * 16];
        const double pred = hj[0] * t[0] + hj[1] * t[1] + hj[2] * t[2] + hj[3] * t[3];
        const double inn = r[j] - pred;
        if (!std::isfinite(inn)) continue;
        if (protect && valid && !valid[j]) continue;
        if (protect && !c.res_sd.empty()) {
            // a spike: skip implausible measurements, but at most frozen_ticks in a row -- a deviation that persists is real
            // (a frozen sensor is caught separately by corr_screen).  Rules tried and rejected on HyQ: reset-and-reacquire
            // after 100 ms of rejections (+94 % error on an unseen recording), a 6x gate (+7-14 %).
            if (std::fabs(inn) >= c.gate * c.res_sd[j]) {
                if (++s.rejected[j] < c.frozen_ticks) continue;
            } else {
                s.rejected[j] = 0;
            }
        }
        for (int k = 0; k < 4; k++) P[k * 5] += q;
        double Ph[4];
        for (int k = 0; k < 4; k++) Ph[k] = P[k * 4 + 0] * hj[0] + P[k * 4 + 1] * hj[1] + P[k * 4 + 2] * hj[2] + P[k * 4 + 3] * hj[3];
        const double S = hj[0] * Ph[0] + hj[1] * Ph[1] + hj[2] * Ph[2] + hj[3] * Ph[3] + 1.0;
        for (int k = 0; k < 4; k++) {
            const double K = Ph[k] / S;
            t[k] += K * inn;
            for (int l = 0; l < 4; l++) P[k * 4 + l] -= K * Ph[l];
        }
    }
}

// nmse of a replay (no protection) for the current tau, with each measurement used `D` ticks after its own tick:
// at tick t the prediction is corrected with what was learnt from ticks 0 .. t - D.  The filter's own update sequence
// (and so its innovations, which set the protection gate) does not depend on D; only the staleness of the applied
// correction does.  D = 1 runs exactly the original next-tick replay.  starts (optional): rows where a new recording
// begins; the filter restarts there and never uses a measurement from an earlier recording.  *used: measurements used.
static double replay(const Correction& c, const std::vector<double>& o, const std::vector<double>& y, const double* X, int width,
                     size_t T, int n, int D, std::vector<double>* innov = nullptr, const std::vector<size_t>* starts = nullptr,
                     size_t* used = nullptr) {
    CorrState s;
    corr_init(c, s, n);
    std::vector<double> h((size_t)n * 4), cor(n), r(n), se(n, 0.0), m(n, 0.0), m2(n, 0.0), hk((size_t)n * 4), pk(n);
    size_t seg = 0, next = 0, n_used = 0;                        // current recording's first row; index of the next start
    if (starts) while (next < starts->size() && (*starts)[next] == 0) next++;
    for (size_t t = 0; t < T; t++) {
        if (starts && next < starts->size() && (*starts)[next] == t) {
            corr_init(c, s, n);
            seg = t;
            next++;
        }
        corr_regress(c, &o[t * n], X + t * width, h.data(), n);
        corr_apply(c, s, h.data(), cor.data(), n, false);
        for (int j = 0; j < n; j++) {
            const double e = o[t * n + j] + cor[j] - y[t * n + j];
            se[j] += e * e;
            m[j] += y[t * n + j];
            m2[j] += y[t * n + j] * y[t * n + j];
        }
        if (D == 1) {
            for (int j = 0; j < n; j++) {
                r[j] = y[t * n + j] - o[t * n + j];
                if (innov) (*innov)[t * n + j] = r[j] - (cor[j]);
            }
            corr_update(c, s, h.data(), r.data(), n, false);
            n_used++;
        } else if (t + 1 >= seg + (size_t)D) {       // the measurement of tick k (same recording) arrives now
            const size_t k = t + 1 - D;
            corr_regress(c, &o[k * n], X + k * width, hk.data(), n);
            corr_apply(c, s, hk.data(), pk.data(), n, false);
            for (int j = 0; j < n; j++) {
                r[j] = y[k * n + j] - o[k * n + j];
                if (innov) (*innov)[k * n + j] = r[j] - pk[j];
            }
            corr_update(c, s, hk.data(), r.data(), n, false);
            n_used++;
        }
    }
    if (used) *used = n_used;
    double tot = 0;
    for (int j = 0; j < n; j++) {
        const double var = m2[j] / T - (m[j] / T) * (m[j] / T);
        tot += se[j] / T / std::max(var, 1e-12);
    }
    return tot / n;
}

double corr_tune(Correction& c, const std::vector<double>& o, const std::vector<double>& y, const double* X, int width, size_t T, int n,
                 double rate, int delay, const std::vector<size_t>* starts) {
    static const double TAU_MS[] = {100, 250, 500, 1000, 2000, 5000};
    if (delay < 1 || (size_t)delay * 4 > T) fail(SLDRNC_ERROR_DATA, "tuning data too short for a delay of " + std::to_string(delay) + " ticks (need at least " + std::to_string(4 * (size_t)delay) + " ticks)");
    double best = std::numeric_limits<double>::infinity(), best_tau = 0;
    for (double ms : TAU_MS) {
        c.tau_ticks = std::max(1.0, ms * rate / 1000.0);
        const double v = replay(c, o, y, X, width, T, n, delay, nullptr, starts);
        if (v < best) { best = v; best_tau = c.tau_ticks; }
    }
    c.tau_ticks = best_tau;
    c.delay_ticks = delay;
    c.frozen_ticks = std::max(10, (int)std::lround(0.02 * rate));     // 20 ms
    // protection gate scale: the RMS of the tuned filter's innovations on this data.  (A median-based scale was tried
    // and rejected: robot innovations are heavy-tailed, so it flagged real load changes as faults and cost 26 % on HyQ.)
    std::vector<double> inn(T * (size_t)n, 0.0);                     // rows whose measurement was never used stay 0
    size_t cnt = 0;
    replay(c, o, y, X, width, T, n, delay, &inn, starts, &cnt);
    if (cnt == 0) fail(SLDRNC_ERROR_DATA, "tuning data too short for a delay of " + std::to_string(delay) + " ticks");
    c.res_sd.assign(n, 0.0);
    for (int j = 0; j < n; j++) {
        double s2 = 0;
        for (size_t t = 0; t < T; t++) s2 += inn[t * n + j] * inn[t * n + j];
        c.res_sd[j] = std::max(1e-6, std::sqrt(s2 / cnt));
    }
    return best;
}

}  // namespace sldi
