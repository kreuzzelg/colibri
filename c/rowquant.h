#ifndef COLIBRI_ROWQUANT_H
#define COLIBRI_ROWQUANT_H
/* Symmetric per-row int8 quantization for dense f32 matrices, as consumed by
 * qwen36's matmul_q.  Its own header so tests/test_rowquant.c can link the very
 * kernel the engine runs; qwen36.c carries a main and cannot be linked into a
 * test, and quant.h is a different world (its own matmul/matmul_q would clash).
 *
 * NOT quant.h's quantize_rows(bits=8): that one divides by s and floors the
 * scale at 1e-8, and lands on different bytes.  x*(1/s) and x/s round apart on
 * ~1.7e-6 of f16 weights -- rare, but over 291 matrices a certainty.  The dense
 * int8 tier is bound to these exact rounding decisions. */
#include <stdint.h>
#include <math.h>

static void quantize_rows_sym8(const float *w, int8_t *q, float *scale, int O, int I){
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const float *r = w + (int64_t)o*I; float am = 0.f;
        for (int i = 0; i < I; i++) { float a = fabsf(r[i]); if (a > am) am = a; }
        float s = am > 1e-12f ? am/127.f : 1.f; scale[o] = s; float inv = 1.f/s;
        int8_t *d = q + (int64_t)o*I;
        for (int i = 0; i < I; i++) { int v = (int)lrintf(r[i]*inv); if (v>127) v=127; if (v<-127) v=-127; d[i] = (int8_t)v; }
    }
}

#endif /* COLIBRI_ROWQUANT_H */
