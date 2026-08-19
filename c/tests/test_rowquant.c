/* ============================================================================
 * test_rowquant.c — standalone unit tests for the symmetric per-row int8
 * quantizer (rowquant.h: quantize_rows_sym8) used by qwen36's dense weights.
 * No model, no weights: it links the SAME rowquant.h and st.h the engine links.
 *
 * This is the on-box proof for the one assumption the fused f16->int8 load path
 * rests on: quantizing a tensor in CHUNKS OF WHOLE ROWS produces byte-identical
 * output to quantizing it in one go.  That is what lets model_init read a matrix
 * through a small scratch buffer instead of materialising it whole in f32.
 *
 * Required properties:
 *   P1 CHUNKED == WHOLE — quantizing row blocks separately yields the same int8
 *      bytes and the same per-row scales as one whole-tensor call.
 *   P2 ROWS ARE INDIVISIBLE — a chunk boundary that falls INSIDE a row changes
 *      the scale, and therefore the bytes.  This is the negative control: it is
 *      why the chunk loop must step in rows, not in elements.
 *   P3 ZERO ROW — a row of all zeros gets scale 1.0 and quantizes to zeros
 *      (the amax > 1e-12 guard), not a division by zero.
 *   P4 MOVE IS A MOVE — quantize_rows_sym8 reproduces, bit for bit, the loop it
 *      was extracted from.  A verbatim copy of that loop lives below as the
 *      reference, so the extraction cannot silently become a rewrite.  The
 *      sample carries a planted row for this: see plant_tie_row.
 *   P5 F16 BLOCK CONVERSION — f16_to_f32_n agrees with the scalar f16_to_f32 on
 *      finite inputs including subnormals (this model has 8.87M of them).
 *      sNaN is excluded on purpose: vcvtph2ps quiets it, see st.h.
 * Exit 0 = all pass.
 * ==========================================================================*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "../rowquant.h"
#include "../st.h"

static int fails = 0;
#define CHECK(cond,msg) do{ if(!(cond)){ printf("  FAIL: %s\n", msg); fails++; } \
                            else printf("  ok:   %s\n", msg); }while(0)

/* Deliberately awkward shape: I is neither a power of two nor a multiple of 8,
 * so both the SIMD body and the scalar tail of f16_to_f32_n are exercised. */
enum { O = 7, I = 133, N = O * I };

/* The loop quantize_rows_sym8 was extracted from (qwen36.c qdw_register).
 * Kept verbatim: P4 compares against it, so any drift shows up as a failure. */
static void reference_sym8(const float *W, int8_t *q, float *sc, int o_count, int i_len){
    for (int o = 0; o < o_count; o++) {
        const float *r = W + (int64_t)o*i_len; float am = 0.f;
        for (int i = 0; i < i_len; i++) { float a = fabsf(r[i]); if (a > am) am = a; }
        float s = am > 1e-12f ? am/127.f : 1.f; sc[o] = s; float inv = 1.f/s;
        int8_t *d = q + (int64_t)o*i_len;
        for (int i = 0; i < i_len; i++) { int v = (int)lrintf(r[i]*inv); if (v>127) v=127; if (v<-127) v=-127; d[i] = (int8_t)v; }
    }
}

static uint32_t rng_state = 0x9E3779B9u;
static uint32_t rng_next(void){ rng_state = rng_state*1664525u + 1013904223u; return rng_state; }

/* A finite f16: exponent 0..30 (30 excludes Inf/NaN), 0 gives subnormals. */
static uint16_t rand_f16(void){
    uint32_t r = rng_next();
    uint16_t sign = (uint16_t)((r >> 31) << 15);
    uint16_t exp  = (uint16_t)((r >> 13) % 31);
    uint16_t mant = (uint16_t)(r & 0x3FF);
    return (uint16_t)(sign | (uint16_t)(exp << 10) | mant);
}

/* The rewrite that matters most is r[i]/s instead of r[i]*(1/s): the reciprocal
 * rounds first, so the two round apart wherever the quotient lands within ~1 ulp
 * of a half-unit boundary.  On random f16 weights that is 1.7e-6 per element --
 * a few hundred values miss it 99.8% of the time, so P4 on a random sample alone
 * would not catch the one thing the dense path cannot survive.  Hence a planted
 * row: search f16 space for a row max and values that provably round apart under
 * the two forms.  Fills row[0..len-1], returns the row max, 0 if nothing found. */
static float plant_tie_row(uint16_t *row, int len){
    for (uint16_t a = 0x0400; a < 0x7C00; a++) {
        float am = f16_to_f32(a), s = am/127.f, inv = 1.f/s;
        int n = 0;
        for (uint32_t bits = 0; bits < 0x10000 && n < len-1; bits++) {
            uint16_t h = (uint16_t)bits;
            if ((h & 0x7C00) == 0x7C00) continue;                 /* Inf/NaN */
            float x = f16_to_f32(h);
            if (fabsf(x) > am) continue;                          /* would move the max */
            if (lrintf(x*inv) != lrintf(x/s)) row[1 + n++] = h;
        }
        if (n) {
            row[0] = a;
            for (int i = 1+n; i < len; i++) row[i] = row[1 + (i-1-n) % n];
            return am;
        }
    }
    return 0.f;
}

int main(void){
    printf("test_rowquant\n");

    static uint16_t h[N];
    static float w[N];
    for (int i = 0; i < N; i++) h[i] = rand_f16();
    /* One row of exact zeros for P3, and one row whose two halves have very
     * different magnitudes, so P2 has something to detect. */
    for (int i = 0; i < I; i++) h[3*I + i] = 0;
    for (int i = 0; i < I; i++) h[5*I + i] = (i < I/2) ? 0x6400 /* 1024.0 */ : 0x1400 /* ~0.00061 */;
    float tie_max = plant_tie_row(h + 6*I, I);
    f16_to_f32_n(h, w, N);

    static int8_t q_whole[N], q_chunk[N], q_ref[N];
    static float  sc_whole[O], sc_chunk[O], sc_ref[O];

    /* ---- P1: chunked over whole rows == one whole-tensor call --------------- */
    quantize_rows_sym8(w, q_whole, sc_whole, O, I);

    memset(q_chunk, 0x5A, sizeof q_chunk);
    const int chunks[] = { 3, 1, 2, 1 };            /* 7 rows, uneven blocks */
    int r0 = 0;
    for (unsigned c = 0; c < sizeof chunks / sizeof *chunks; c++) {
        int rows = chunks[c];
        quantize_rows_sym8(w + (int64_t)r0*I, q_chunk + (int64_t)r0*I, sc_chunk + r0, rows, I);
        r0 += rows;
    }
    CHECK(r0 == O, "the chunk sizes cover every row exactly once");
    CHECK(memcmp(q_whole, q_chunk, sizeof q_whole) == 0,
          "P1 chunking over whole rows leaves every int8 byte identical");
    int scales_equal = 1;
    for (int o = 0; o < O; o++) if (sc_whole[o] != sc_chunk[o]) scales_equal = 0;
    CHECK(scales_equal, "P1 chunking over whole rows leaves every row scale identical");

    /* ---- P2: a boundary inside a row changes the result --------------------- */
    /* Reinterpreting each row as two half-rows is exactly what a chunk loop that
     * stepped in elements instead of rows would do to row 5. */
    enum { HALF = I / 2 };
    static int8_t q_half[O * 2 * HALF];
    static float  sc_half[O * 2];
    quantize_rows_sym8(w, q_half, sc_half, O * 2, HALF);
    CHECK(sc_half[5*2] != sc_whole[5],
          "P2 a chunk boundary inside a row yields a different scale for that row");

    /* ---- P3: an all-zero row ----------------------------------------------- */
    int zero_row_clean = 1;
    for (int i = 0; i < I; i++) if (q_whole[3*I + i] != 0) zero_row_clean = 0;
    CHECK(sc_whole[3] == 1.f, "P3 an all-zero row gets scale 1.0");
    CHECK(zero_row_clean,    "P3 an all-zero row quantizes to all zeros");

    /* ---- P4: the extracted kernel matches the loop it came from ------------- */
    reference_sym8(w, q_ref, sc_ref, O, I);
    CHECK(tie_max > 0.f,
          "P4 the sample carries a row where x*(1/s) and x/s round apart");
    CHECK(memcmp(q_whole, q_ref, sizeof q_whole) == 0,
          "P4 quantize_rows_sym8 matches the original loop byte for byte");
    int ref_scales_equal = 1;
    for (int o = 0; o < O; o++) if (sc_whole[o] != sc_ref[o]) ref_scales_equal = 0;
    CHECK(ref_scales_equal, "P4 quantize_rows_sym8 matches the original loop on every scale");

    /* ---- P5: block f16 conversion == scalar, subnormals included ------------ */
    int conv_equal = 1, subnormals = 0;
    for (int i = 0; i < N; i++) {
        float ref = f16_to_f32(h[i]);
        if (memcmp(&ref, &w[i], sizeof ref) != 0) conv_equal = 0;
        if ((h[i] & 0x7C00) == 0 && (h[i] & 0x03FF) != 0) subnormals++;
    }
    CHECK(subnormals > 0, "P5 the sample actually contains subnormal f16 values");
    CHECK(conv_equal, "P5 f16_to_f32_n agrees with scalar f16_to_f32 on every element");

    printf("\n%s  (%d failure%s)\n", fails?"TEST FAIL":"ALL PASS", fails, fails==1?"":"s");
    return fails ? 1 : 0;
}
