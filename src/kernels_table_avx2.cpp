// kernels_table_avx2.cpp -- table lookups for 8 robots at once (AVX2 + FMA).  Repeats structure.cpp's table_lookup
// operation for operation, so a fleet and a session give the same bits; compiled without floating-point contraction.
// After the research build's round-5 kernel: each node's values are whole vectors, 4 nodes per robot (Kuhn simplex).
#include <immintrin.h>

#include "internal.hpp"

namespace sldi {

void table8_avx2(const Table& t, const float* xs, float* heads) {
    const __m256 zero = _mm256_setzero_ps(), one = _mm256_set1_ps(1.0f);
    __m256 f[3];
    __m256i s[3];
    __m256i n0 = _mm256_setzero_si256();
    for (int d = 0; d < 3; d++) {
        const __m256 x = _mm256_loadu_ps(xs + 8 * d);
        const __m256i stride = _mm256_set1_epi32(t.stride[d]);
        if (t.periodic[d]) {
            __m256 w = _mm256_mul_ps(_mm256_sub_ps(x, _mm256_set1_ps(t.lo[d])), _mm256_set1_ps(t.scale[d]));
            const __m256 q = _mm256_floor_ps(_mm256_mul_ps(w, _mm256_set1_ps(t.inv_n[d])));
            const __m256 wrap = _mm256_mul_ps(_mm256_set1_ps(t.nfl[d]), q);
            w = _mm256_sub_ps(w, wrap);
            const __m256 wf = _mm256_floor_ps(w);
            const __m256i N = _mm256_set1_epi32(t.n[d]);
            __m256i k = _mm256_cvttps_epi32(wf);
            k = _mm256_sub_epi32(k, _mm256_and_si256(_mm256_cmpgt_epi32(k, _mm256_set1_epi32(t.n[d] - 1)), N));
            k = _mm256_add_epi32(k, _mm256_and_si256(_mm256_cmpgt_epi32(_mm256_setzero_si256(), k), N));
            __m256i k1 = _mm256_add_epi32(k, _mm256_set1_epi32(1));
            k1 = _mm256_andnot_si256(_mm256_cmpeq_epi32(k1, N), k1);
            f[d] = _mm256_sub_ps(w, wf);
            n0 = _mm256_add_epi32(n0, _mm256_mullo_epi32(k, stride));
            s[d] = _mm256_mullo_epi32(_mm256_sub_epi32(k1, k), stride);
        } else {
            __m256 u = _mm256_mul_ps(_mm256_sub_ps(x, _mm256_set1_ps(t.lo[d])), _mm256_set1_ps(t.scale[d]));
            u = _mm256_max_ps(u, zero);
            u = _mm256_min_ps(u, _mm256_set1_ps(t.umax[d]));
            const __m256 uf = _mm256_floor_ps(u);
            f[d] = _mm256_sub_ps(u, uf);
            n0 = _mm256_add_epi32(n0, _mm256_mullo_epi32(_mm256_cvttps_epi32(uf), stride));
            s[d] = stride;
        }
    }
    auto cs = [&](int a, int b) {
        const __m256 m = _mm256_cmp_ps(f[a], f[b], _CMP_LT_OQ);
        const __m256 fa = _mm256_blendv_ps(f[a], f[b], m), fb = _mm256_blendv_ps(f[b], f[a], m);
        const __m256i mi = _mm256_castps_si256(m);
        const __m256i sa = _mm256_blendv_epi8(s[a], s[b], mi), sb = _mm256_blendv_epi8(s[b], s[a], mi);
        f[a] = fa; f[b] = fb; s[a] = sa; s[b] = sb;
    };
    cs(0, 1); cs(1, 2); cs(0, 1);
    const __m256i n1 = _mm256_add_epi32(n0, s[0]), n2 = _mm256_add_epi32(n1, s[1]), n3 = _mm256_add_epi32(n2, s[2]);
    alignas(32) int nodes[4][8];
    alignas(32) float wts[4][8];
    _mm256_store_si256((__m256i*)nodes[0], n0); _mm256_store_si256((__m256i*)nodes[1], n1);
    _mm256_store_si256((__m256i*)nodes[2], n2); _mm256_store_si256((__m256i*)nodes[3], n3);
    _mm256_store_ps(wts[0], _mm256_sub_ps(one, f[0])); _mm256_store_ps(wts[1], _mm256_sub_ps(f[0], f[1]));
    _mm256_store_ps(wts[2], _mm256_sub_ps(f[1], f[2])); _mm256_store_ps(wts[3], f[2]);
    const int hp = t.hp;
    const float* T = t.T.data();
    for (int l = 0; l < 8; l++) {
        const float *T0 = T + (size_t)nodes[0][l] * hp, *T1 = T + (size_t)nodes[1][l] * hp, *T2 = T + (size_t)nodes[2][l] * hp,
                    *T3 = T + (size_t)nodes[3][l] * hp;
        const __m256 w0 = _mm256_set1_ps(wts[0][l]), w1 = _mm256_set1_ps(wts[1][l]), w2 = _mm256_set1_ps(wts[2][l]), w3 = _mm256_set1_ps(wts[3][l]);
        float* out = heads + (size_t)l * hp;
        for (int c = 0; c < hp; c += 8) {
            __m256 a = _mm256_mul_ps(w0, _mm256_loadu_ps(T0 + c));
            a = _mm256_fmadd_ps(w1, _mm256_loadu_ps(T1 + c), a);
            a = _mm256_fmadd_ps(w2, _mm256_loadu_ps(T2 + c), a);
            _mm256_storeu_ps(out + c, _mm256_fmadd_ps(w3, _mm256_loadu_ps(T3 + c), a));
        }
    }
}

// 8 robots of a table model without correction, rows in to rows out.  The affine values and the output combination
// run 4 robots per vector in double, with the operations of session.cpp's table_prepare and combine_outputs in their
// order (subtract, multiply; multiply, add; no contraction), so every robot gets a session's bits.
void table_rows8_avx2(const Table& t, const TableIO& io, const double* X, double* Y, float* heads) {
    alignas(32) float xs[24];
    for (int d = 0; d < 3; d++) {
        const int c = t.col[d];
        for (int l = 0; l < 8; l++) xs[d * 8 + l] = c >= 0 ? (float)X[(size_t)l * io.ldx + c] : 0.0f;
    }
    table8_avx2(t, xs, heads);
    const int hp = t.hp, no = io.no, na = io.na;
    for (int h = 0; h < 8; h += 4) {
        const double *x0 = X + (size_t)h * io.ldx, *x1 = x0 + io.ldx, *x2 = x1 + io.ldx, *x3 = x2 + io.ldx;
        const float *h0 = heads + (size_t)h * hp, *h1 = h0 + hp, *h2 = h1 + hp, *h3 = h2 + hp;
        double* y0 = Y + (size_t)h * no;
        if (no <= 8) {                                    // each affine value once: k outer, outputs inner (each output
            __m256d raw[8];                               // still adds its terms in k order)
            for (int j = 0; j < no; j++) raw[j] = _mm256_cvtps_pd(_mm_set_ps(h3[j], h2[j], h1[j], h0[j]));
            for (int k = 0; k < na; k++) {
                const int c = io.aff_col[k];
                const __m256d a = _mm256_mul_pd(_mm256_sub_pd(_mm256_set_pd(x3[c], x2[c], x1[c], x0[c]), _mm256_set1_pd(io.aff_mean[k])),
                                                _mm256_set1_pd(io.aff_inv[k]));
                for (int j = 0, e = no * (k + 1); j < no; j++, e++)
                    raw[j] = _mm256_add_pd(raw[j], _mm256_mul_pd(a, _mm256_cvtps_pd(_mm_set_ps(h3[e], h2[e], h1[e], h0[e]))));
            }
            for (int j = 0; j < no; j++) {
                __m256d v = raw[j];
                if (io.out_std) v = _mm256_add_pd(_mm256_mul_pd(v, _mm256_set1_pd(io.out_std[j])), _mm256_set1_pd(io.out_mean[j]));
                alignas(32) double r[4];
                _mm256_store_pd(r, v);
                y0[j] = r[0]; y0[no + j] = r[1]; y0[2 * no + j] = r[2]; y0[3 * no + j] = r[3];
            }
            continue;
        }
        for (int j = 0; j < no; j++) {
            __m256d raw = _mm256_cvtps_pd(_mm_set_ps(h3[j], h2[j], h1[j], h0[j]));
            for (int k = 0; k < na; k++) {
                const int c = io.aff_col[k], e = no * (k + 1) + j;
                const __m256d a = _mm256_mul_pd(_mm256_sub_pd(_mm256_set_pd(x3[c], x2[c], x1[c], x0[c]), _mm256_set1_pd(io.aff_mean[k])),
                                                _mm256_set1_pd(io.aff_inv[k]));
                raw = _mm256_add_pd(raw, _mm256_mul_pd(a, _mm256_cvtps_pd(_mm_set_ps(h3[e], h2[e], h1[e], h0[e]))));
            }
            if (io.out_std) raw = _mm256_add_pd(_mm256_mul_pd(raw, _mm256_set1_pd(io.out_std[j])), _mm256_set1_pd(io.out_mean[j]));
            alignas(32) double r[4];
            _mm256_store_pd(r, raw);
            y0[j] = r[0]; y0[no + j] = r[1]; y0[2 * no + j] = r[2]; y0[3 * no + j] = r[3];
        }
    }
}

}  // namespace sldi
