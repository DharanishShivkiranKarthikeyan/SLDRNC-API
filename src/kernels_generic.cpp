// kernels_generic.cpp -- the kernels for the baseline instruction set of the target (SSE2 on x86-64, NEON on ARM64).
#include "internal.hpp"

namespace sldi {
namespace kgen {
#include "kernels.inc"

// n rows of gemv, one at a time (the same arithmetic as a session's forward pass)
static void gemv_rows(const float* X, int n, int ldx, int in, const float* W, int ldw, const float* b, float* Z, int ldz) {
    for (int r = 0; r < n; r++) gemv(X + (size_t)r * ldx, in, W, ldw, b, Z + (size_t)r * ldz);
}
}  // namespace kgen
void table8_scalar(const Table& t, const float* xs, float* heads);   // structure.cpp
const Kernels* kernels_generic() {
    static const Kernels k = {kgen::gemv, kgen::gemm_rows, kgen::outer_acc, kgen::back_input, kgen::gemv_rows, table8_scalar, nullptr};
    return &k;
}
}  // namespace sldi
