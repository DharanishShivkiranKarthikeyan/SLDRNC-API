// kernels_avx2.cpp -- the kernels compiled with AVX2 + FMA (this file only; the build adds the flags).
// Selected at run time when the CPU supports AVX2 and FMA.  The single-row forward pass (the per-tick path) is
// hand-written with intrinsics and register blocking: compilers do not reliably keep the accumulators in registers.
#include <immintrin.h>

#include "internal.hpp"

namespace sldi {
namespace kavx2 {
#include "kernels.inc"

// z[o] = b[o] + sum_j x[j] W[j * ldw + o]; ldw is a multiple of 16.  Up to 64 outputs per pass in 8 accumulators.
static void gemv_reg(const float* x, int in, const float* W, int ldw, const float* b, float* z) {
    int o0 = 0;
    for (; o0 + 64 <= ldw; o0 += 64) {
        __m256 a0 = _mm256_loadu_ps(b + o0), a1 = _mm256_loadu_ps(b + o0 + 8), a2 = _mm256_loadu_ps(b + o0 + 16), a3 = _mm256_loadu_ps(b + o0 + 24);
        __m256 a4 = _mm256_loadu_ps(b + o0 + 32), a5 = _mm256_loadu_ps(b + o0 + 40), a6 = _mm256_loadu_ps(b + o0 + 48), a7 = _mm256_loadu_ps(b + o0 + 56);
        const float* w = W + o0;
        for (int j = 0; j < in; j++, w += ldw) {
            const __m256 v = _mm256_broadcast_ss(x + j);
            a0 = _mm256_fmadd_ps(v, _mm256_loadu_ps(w), a0);
            a1 = _mm256_fmadd_ps(v, _mm256_loadu_ps(w + 8), a1);
            a2 = _mm256_fmadd_ps(v, _mm256_loadu_ps(w + 16), a2);
            a3 = _mm256_fmadd_ps(v, _mm256_loadu_ps(w + 24), a3);
            a4 = _mm256_fmadd_ps(v, _mm256_loadu_ps(w + 32), a4);
            a5 = _mm256_fmadd_ps(v, _mm256_loadu_ps(w + 40), a5);
            a6 = _mm256_fmadd_ps(v, _mm256_loadu_ps(w + 48), a6);
            a7 = _mm256_fmadd_ps(v, _mm256_loadu_ps(w + 56), a7);
        }
        _mm256_storeu_ps(z + o0, a0); _mm256_storeu_ps(z + o0 + 8, a1); _mm256_storeu_ps(z + o0 + 16, a2); _mm256_storeu_ps(z + o0 + 24, a3);
        _mm256_storeu_ps(z + o0 + 32, a4); _mm256_storeu_ps(z + o0 + 40, a5); _mm256_storeu_ps(z + o0 + 48, a6); _mm256_storeu_ps(z + o0 + 56, a7);
    }
    for (; o0 < ldw; o0 += 16) {                       // remaining 16-wide blocks, two independent sums each
        __m256 a0 = _mm256_loadu_ps(b + o0), a1 = _mm256_loadu_ps(b + o0 + 8);
        __m256 c0 = _mm256_setzero_ps(), c1 = _mm256_setzero_ps();
        const float* w = W + o0;
        int j = 0;
        for (; j + 2 <= in; j += 2, w += 2 * (size_t)ldw) {
            const __m256 v0 = _mm256_broadcast_ss(x + j), v1 = _mm256_broadcast_ss(x + j + 1);
            a0 = _mm256_fmadd_ps(v0, _mm256_loadu_ps(w), a0);
            a1 = _mm256_fmadd_ps(v0, _mm256_loadu_ps(w + 8), a1);
            c0 = _mm256_fmadd_ps(v1, _mm256_loadu_ps(w + ldw), c0);
            c1 = _mm256_fmadd_ps(v1, _mm256_loadu_ps(w + ldw + 8), c1);
        }
        for (; j < in; j++, w += ldw) {
            const __m256 v = _mm256_broadcast_ss(x + j);
            a0 = _mm256_fmadd_ps(v, _mm256_loadu_ps(w), a0);
            a1 = _mm256_fmadd_ps(v, _mm256_loadu_ps(w + 8), a1);
        }
        _mm256_storeu_ps(z + o0, _mm256_add_ps(a0, c0));
        _mm256_storeu_ps(z + o0 + 8, _mm256_add_ps(a1, c1));
    }
}
// n rows of gemv_reg, bit for bit: every output accumulates in gemv_reg's order (the 64-wide blocks one running FMA sum
// from the bias, the remaining 16-wide blocks even and odd inputs in two sums added at the end).  Four rows share each
// weight load in the 64-wide blocks (4 rows x 16 outputs: 8 accumulators, 14 registers live, so nothing spills; 3 x 32
// needed 19 and ran slower than one row at a time), two rows in the even / odd blocks.
static void gemv_rows(const float* X, int n, int ldx, int in, const float* W, int ldw, const float* b, float* Z, int ldz) {
    const int n64 = ldw / 64 * 64;
    int r = 0;
    for (; r + 4 <= n; r += 4) {
        const float* x0 = X + (size_t)r * ldx;
        const float* x1 = x0 + ldx;
        const float* x2 = x1 + ldx;
        const float* x3 = x2 + ldx;
        float* z0 = Z + (size_t)r * ldz;
        float* z1 = z0 + ldz;
        float* z2 = z1 + ldz;
        float* z3 = z2 + ldz;
        for (int o0 = 0; o0 < n64; o0 += 16) {
            __m256 a00 = _mm256_loadu_ps(b + o0), a01 = _mm256_loadu_ps(b + o0 + 8);
            __m256 a10 = a00, a11 = a01, a20 = a00, a21 = a01, a30 = a00, a31 = a01;
            const float* w = W + o0;
            for (int j = 0; j < in; j++, w += ldw) {
                const __m256 w0 = _mm256_loadu_ps(w), w1 = _mm256_loadu_ps(w + 8);
                __m256 v = _mm256_broadcast_ss(x0 + j);
                a00 = _mm256_fmadd_ps(v, w0, a00); a01 = _mm256_fmadd_ps(v, w1, a01);
                v = _mm256_broadcast_ss(x1 + j);
                a10 = _mm256_fmadd_ps(v, w0, a10); a11 = _mm256_fmadd_ps(v, w1, a11);
                v = _mm256_broadcast_ss(x2 + j);
                a20 = _mm256_fmadd_ps(v, w0, a20); a21 = _mm256_fmadd_ps(v, w1, a21);
                v = _mm256_broadcast_ss(x3 + j);
                a30 = _mm256_fmadd_ps(v, w0, a30); a31 = _mm256_fmadd_ps(v, w1, a31);
            }
            _mm256_storeu_ps(z0 + o0, a00); _mm256_storeu_ps(z0 + o0 + 8, a01);
            _mm256_storeu_ps(z1 + o0, a10); _mm256_storeu_ps(z1 + o0 + 8, a11);
            _mm256_storeu_ps(z2 + o0, a20); _mm256_storeu_ps(z2 + o0 + 8, a21);
            _mm256_storeu_ps(z3 + o0, a30); _mm256_storeu_ps(z3 + o0 + 8, a31);
        }
        for (int o0 = n64; o0 < ldw; o0 += 16)
            for (int h = 0; h < 4; h += 2) {
                const float* y0 = X + (size_t)(r + h) * ldx;
                const float* y1 = y0 + ldx;
                const __m256 b0 = _mm256_loadu_ps(b + o0), b1 = _mm256_loadu_ps(b + o0 + 8), zero = _mm256_setzero_ps();
                __m256 a00 = b0, a01 = b1, c00 = zero, c01 = zero, a10 = b0, a11 = b1, c10 = zero, c11 = zero;
                const float* w = W + o0;
                int j = 0;
                for (; j + 2 <= in; j += 2, w += 2 * (size_t)ldw) {
                    const __m256 w0 = _mm256_loadu_ps(w), w1 = _mm256_loadu_ps(w + 8), u0 = _mm256_loadu_ps(w + ldw), u1 = _mm256_loadu_ps(w + ldw + 8);
                    __m256 v = _mm256_broadcast_ss(y0 + j), e = _mm256_broadcast_ss(y0 + j + 1);
                    a00 = _mm256_fmadd_ps(v, w0, a00); a01 = _mm256_fmadd_ps(v, w1, a01); c00 = _mm256_fmadd_ps(e, u0, c00); c01 = _mm256_fmadd_ps(e, u1, c01);
                    v = _mm256_broadcast_ss(y1 + j); e = _mm256_broadcast_ss(y1 + j + 1);
                    a10 = _mm256_fmadd_ps(v, w0, a10); a11 = _mm256_fmadd_ps(v, w1, a11); c10 = _mm256_fmadd_ps(e, u0, c10); c11 = _mm256_fmadd_ps(e, u1, c11);
                }
                for (; j < in; j++, w += ldw) {
                    const __m256 w0 = _mm256_loadu_ps(w), w1 = _mm256_loadu_ps(w + 8);
                    __m256 v = _mm256_broadcast_ss(y0 + j);
                    a00 = _mm256_fmadd_ps(v, w0, a00); a01 = _mm256_fmadd_ps(v, w1, a01);
                    v = _mm256_broadcast_ss(y1 + j);
                    a10 = _mm256_fmadd_ps(v, w0, a10); a11 = _mm256_fmadd_ps(v, w1, a11);
                }
                float* q0 = Z + (size_t)(r + h) * ldz;
                float* q1 = q0 + ldz;
                _mm256_storeu_ps(q0 + o0, _mm256_add_ps(a00, c00)); _mm256_storeu_ps(q0 + o0 + 8, _mm256_add_ps(a01, c01));
                _mm256_storeu_ps(q1 + o0, _mm256_add_ps(a10, c10)); _mm256_storeu_ps(q1 + o0 + 8, _mm256_add_ps(a11, c11));
            }
    }
    for (; r < n; r++) gemv_reg(X + (size_t)r * ldx, in, W, ldw, b, Z + (size_t)r * ldz);
}
}  // namespace kavx2
void table8_avx2(const Table& t, const float* xs, float* heads);   // kernels_table_avx2.cpp
void table_rows8_avx2(const Table& t, const TableIO& io, const double* X, double* Y, float* heads);
const Kernels* kernels_avx2() {
    static const Kernels k = {kavx2::gemv_reg, kavx2::gemm_rows, kavx2::outer_acc, kavx2::back_input, kavx2::gemv_rows, table8_avx2,
                              table_rows8_avx2};
    return &k;
}
}  // namespace sldi
