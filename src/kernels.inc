// kernels.inc -- dense-layer kernels, written as plain loops that compilers vectorise (SSE / AVX2 / NEON).
// Included twice: by kernels_generic.cpp (baseline ISA) and kernels_avx2.cpp (built with AVX2 + FMA), each inside
// its own namespace.  Output dimensions are padded to multiples of 16 (zero weights), so blocks never run short.

// z[o] = b[o] + sum_j x[j] W[j * ldw + o]   (o < ldw)
static void gemv(const float* x, int in, const float* W, int ldw, const float* b, float* z) {
    for (int o0 = 0; o0 < ldw; o0 += 16) {
        float acc[16];
        for (int u = 0; u < 16; u++) acc[u] = b[o0 + u];
        const float* w = W + o0;
        for (int j = 0; j < in; j++, w += ldw) {
            const float a = x[j];
            for (int u = 0; u < 16; u++) acc[u] += a * w[u];
        }
        for (int u = 0; u < 16; u++) z[o0 + u] = acc[u];
    }
}

// Z (n x ldz) = X (n x in, row stride ldx) W + b, four rows at a time so each weight block is loaded once per 4 rows
static void gemm_rows(const float* X, int n, int ldx, int in, const float* W, int ldw, const float* b, float* Z, int ldz) {
    int r = 0;
    for (; r + 4 <= n; r += 4) {
        const float* x0 = X + (size_t)r * ldx;
        const float* x1 = x0 + ldx;
        const float* x2 = x1 + ldx;
        const float* x3 = x2 + ldx;
        for (int o0 = 0; o0 < ldw; o0 += 16) {
            float a0[16], a1[16], a2[16], a3[16];
            for (int u = 0; u < 16; u++) a0[u] = a1[u] = a2[u] = a3[u] = b[o0 + u];
            const float* w = W + o0;
            for (int j = 0; j < in; j++, w += ldw) {
                const float v0 = x0[j], v1 = x1[j], v2 = x2[j], v3 = x3[j];
                for (int u = 0; u < 16; u++) {
                    a0[u] += v0 * w[u];
                    a1[u] += v1 * w[u];
                    a2[u] += v2 * w[u];
                    a3[u] += v3 * w[u];
                }
            }
            float* z = Z + (size_t)r * ldz + o0;
            for (int u = 0; u < 16; u++) { z[u] = a0[u]; z[ldz + u] = a1[u]; z[2 * ldz + u] = a2[u]; z[3 * ldz + u] = a3[u]; }
        }
    }
    for (; r < n; r++) gemv(X + (size_t)r * ldx, in, W, ldw, b, Z + (size_t)r * ldz);
}

// dW[j][o] += sum_r X[r][j] D[r][o]  for j in [j0, j1), o < ldd (D zero padded), four j at a time
static void outer_acc(const float* X, int n, int ldx, int j0, int j1, const float* D, int ldd, int out, float* dW, int ldw) {
    (void)out;
    int j = j0;
    for (; j + 4 <= j1; j += 4) {
        float* w0 = dW + (size_t)j * ldw;
        float* w1 = w0 + ldw;
        float* w2 = w1 + ldw;
        float* w3 = w2 + ldw;
        for (int r = 0; r < n; r++) {
            const float* xr = X + (size_t)r * ldx + j;
            const float v0 = xr[0], v1 = xr[1], v2 = xr[2], v3 = xr[3];
            if (v0 == 0.0f && v1 == 0.0f && v2 == 0.0f && v3 == 0.0f) continue;   // ReLU zeros are common
            const float* d = D + (size_t)r * ldd;
            for (int o = 0; o < ldd; o++) {
                const float dv = d[o];
                w0[o] += v0 * dv;
                w1[o] += v1 * dv;
                w2[o] += v2 * dv;
                w3[o] += v3 * dv;
            }
        }
    }
    for (; j < j1; j++) {
        float* w0 = dW + (size_t)j * ldw;
        for (int r = 0; r < n; r++) {
            const float v0 = X[(size_t)r * ldx + j];
            if (v0 == 0.0f) continue;
            const float* d = D + (size_t)r * ldd;
            for (int o = 0; o < ldd; o++) w0[o] += v0 * d[o];
        }
    }
}

// dX[r][j] = sum_o D[r][o] W[j][o]   (j < in)
static void back_input(const float* D, int n, int ldd, int out, const float* W, int ldw, int in, float* dX, int ldx) {
    (void)out;
    for (int r = 0; r < n; r++) {
        const float* d = D + (size_t)r * ldd;
        float* dx = dX + (size_t)r * ldx;
        for (int j = 0; j < in; j++) {
            const float* w = W + (size_t)j * ldw;
            float acc[16] = {0};
            for (int o0 = 0; o0 < ldd; o0 += 16)
                for (int u = 0; u < 16; u++) acc[u] += d[o0 + u] * w[o0 + u];
            float s = 0;
            for (int u = 0; u < 16; u++) s += acc[u];
            dx[j] = s;
        }
    }
}
