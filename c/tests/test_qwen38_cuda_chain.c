/* COLI_CUDA_CHAIN=1: Qwen3.8's layers as one chain on the card give the oracle's tokens.
 *
 * The tiny FP8 fixture (make qwen38-tiny-fp8-generate: 4 layers, two of them
 * DeltaNet, the PLE layer, QSA with its indexer, hidden 32) runs the real engine on
 * the fake CUDA tier with the whole dense trunk placed and computed by the fake
 * (fake_dense_compute), the chain's ops answered by the fake's host-side table
 * (tests/qwen36_fake_cuda.h, the kernels' arithmetic): once with the chain on, once
 * with a frame that fails mid-run (the state rebuilt on the CPU, the run finishing
 * there), once without the chain. Every run must stay within the oracle's limits
 * (ref.json), the chain runs must have set up and run every layer on the device. */
#define main qwen38_main_unused
#include "../qwen38.c"
#undef main

#ifdef _WIN32
#undef setenv
#undef unsetenv
#define unsetenv(name) _putenv_s(name, "")
#endif

#include "qwen36_fake_cuda.h"

#include "../qwen36_tier.c"

static int t_fails;
static void tk(int ok, const char *what) { if (ok) { printf("  ok   %s\n", what); return; } printf("  FAIL %s\n", what); t_fails++; }

int main(void) {
    setenv("SNAP", "./qwen38_tiny_fp8", 1);
    setenv("OMP_NUM_THREADS", "2", 1);
    setenv("NOSTREAM", "1", 1);
    setenv("USAGE_SAVE", "0", 1);
    setenv("COLI_CUDA", "1", 1);
    setenv("COLI_GPUS", "0", 1);
    setenv("QT_NO_WARMSTART", "1", 1);
    setenv("QT_UPLOAD_SYNC", "1", 1);
    setenv("HEAT_FILE", "", 1);
    setenv("Q38_TRUNK_GPU", "1", 1);
    setenv("Q38_TRUNK_MIN_KB", "0", 1);          /* every dense matrix of the fixture is offered */
    setenv("COLI_PLACE", "auto", 1);
    setenv("CUDA_EXPERT_GB", "0.5", 1);
    setenv("Q38_MTP", "0", 1);                     /* no draft: the run stays plain */
    fake_dense_compute = 1;
    char *argv[] = { (char *)"qwen38", (char *)"1", (char *)"8", (char *)"./qwen38_tiny_fp8/ref.json" };

    printf(" the chain on the fake card\n");
    setenv("COLI_CUDA_CHAIN", "1", 1);
    fake_chain_frames = 0; fake_chain_fail_at = 0;
    int rc = qwen38_main_unused(4, argv);
    tk(rc == 0, "engine with every layer on the chain stays within the oracle's limits");
    tk(fake_chain_frames > 8, "the forwards ran through the chain's frames");
    qt_shutdown();

    printf(" a frame that fails mid-run\n");
    fake_chain_frames = 0; fake_chain_fail_at = 30;   /* past the first forward's frames: a decode token's frame fails */
    rc = qwen38_main_unused(4, argv);
    tk(rc == 0, "the state rebuilt on the CPU, the run within the oracle's limits");
    tk(fake_chain_frames >= 30, "the fault fired");
    fake_chain_fail_at = 0;
    qt_shutdown();

    printf(" the same without the chain\n");
    setenv("COLI_CUDA_CHAIN", "0", 1);
    fake_chain_frames = 0;
    rc = qwen38_main_unused(4, argv);
    tk(rc == 0, "the per-matrix run stays within the oracle's limits");
    tk(fake_chain_frames == 0, "no frame reached the card");
    qt_shutdown();

    if (t_fails) { printf("test_qwen38_cuda_chain: %d failure(s)\n", t_fails); return 1; }
    printf("OK test_qwen38_cuda_chain: Qwen3.8's chain on the card gives the oracle's tokens\n");
    return 0;
}
