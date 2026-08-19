/* ============================================================================
 * test_affinity.c — standalone unit tests for fast-core selection
 * (affinity.h: coli_select_fast_cores)
 *
 * qwen36's expert matmuls run under `#pragma omp parallel for schedule(static)`,
 * which splits the output rows evenly across the team. An even split means the
 * SLOWEST thread sets the pace of every matmul: the barrier at the end of the
 * region waits for it. Two kinds of logical CPU are therefore actively harmful
 * to add to the team, and both were measured on a Core Ultra 7 155H running
 * 400 decode tokens:
 *
 *   an SMT sibling  — 12 threads on 6 physical cores: 5.07 tok/s vs 5.85
 *   a slower class  — 6 P-cores + 2 E-cores:          4.84 tok/s vs 5.85
 *                     all 16 physical cores:          4.06 tok/s vs 5.85
 *
 * So the team wants one thread per PHYSICAL core, and only cores of the
 * FASTEST class. That is what this selection computes. It is deliberately a
 * pure function over a topology table: the sysfs reader that fills that table
 * is untestable without the machine it runs on, but the policy is not, and the
 * policy is where the mistakes live.
 *
 *   P1 REAL HYBRID    — the recorded Meteor Lake topology (6 P-cores with SMT,
 *                       8 E-cores, 2 low-power E-cores in the SoC tile) must
 *                       yield exactly the six cpus that measured fastest.
 *   P2 HOMOGENEOUS    — a 16C/32T part with one frequency must keep all 16
 *                       physical cores. This is the case omp_tune.h already
 *                       handles today; the class filter must not regress it.
 *   P3 DEGENERATE     — a single core must not select an empty team.
 *   P4 BIG.LITTLE     — a two-tier ARM part must drop the little cores.
 *   P5 NO CPUFREQ     — when no frequency is readable (some VMs expose no
 *                       cpufreq driver) every physical core is kept, i.e. we
 *                       fall back to plain physical-core sizing rather than
 *                       selecting nothing.
 *   P6 MULTI-SOCKET   — core ids repeat per package. Deduplication keyed on
 *                       core id alone would collapse two sockets into one.
 *
 * Exit 0 = all pass.
 * ==========================================================================*/
#define _GNU_SOURCE   /* affinity.h reaches sched_setaffinity; must precede libc */
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "../affinity.h"

static int fails = 0;
#define CHECK(cond, msg) do { \
    if (cond) printf("  ok   %s\n", msg); \
    else { printf("  FAIL %s\n", msg); fails++; } \
} while (0)

/* Compares the selection against an expected cpu list, order included: the
   caller feeds the result straight to a cpu mask, so a stable ascending order
   is part of the contract rather than an accident of the implementation. */
static int selects(const ColiCpu *cpus, int n, const int *want, int nwant)
{
    int got[64];
    int ngot = coli_select_fast_cores(cpus, n, got, 64);
    if (ngot != nwant) {
        printf("       (selected %d cpus, expected %d)\n", ngot, nwant);
        return 0;
    }
    for (int i = 0; i < ngot; i++) {
        if (got[i] != want[i]) {
            printf("       (cpu %d at slot %d, expected %d)\n", got[i], i, want[i]);
            return 0;
        }
    }
    return 1;
}

/* Recorded verbatim from /sys/devices/system/cpu on the Core Ultra 7 155H the
   measurements above were taken on. Columns: cpu, core_id, package, max kHz.
   Note the SMT pairing is not adjacent -- cpu0 pairs with cpu5 -- so a
   selection that assumed "every other cpu" would pick two siblings of the same
   core here and still look plausible. */
static const ColiCpu meteor_lake[] = {
    {  0, 16, 0, 4500000 }, {  1,  8, 0, 4800000 }, {  2,  8, 0, 4800000 },
    {  3, 12, 0, 4800000 }, {  4, 12, 0, 4800000 }, {  5, 16, 0, 4500000 },
    {  6, 20, 0, 4500000 }, {  7, 20, 0, 4500000 }, {  8, 24, 0, 4500000 },
    {  9, 24, 0, 4500000 }, { 10, 28, 0, 4500000 }, { 11, 28, 0, 4500000 },
    { 12,  0, 0, 3800000 }, { 13,  1, 0, 3800000 }, { 14,  2, 0, 3800000 },
    { 15,  3, 0, 3800000 }, { 16,  4, 0, 3800000 }, { 17,  5, 0, 3800000 },
    { 18,  6, 0, 3800000 }, { 19,  7, 0, 3800000 },
    { 20, 32, 0, 2500000 }, { 21, 33, 0, 2500000 },
};

int main(void)
{
    printf("test_affinity: fast-core selection for the OpenMP team\n");

    /* ---- P1: the topology the measurements were taken on ------------------ */
    {
        /* The six cpus that ran at 5.85 tok/s: one per physical P-core. */
        static const int want[] = { 0, 1, 3, 6, 8, 10 };
        CHECK(selects(meteor_lake, 22, want, 6),
              "P1 Meteor Lake selects the six physical P-cores");
    }

    /* ---- P2: homogeneous part must keep every physical core --------------- */
    {
        ColiCpu zen3[32];
        int want[16];
        for (int core = 0; core < 16; core++) {
            /* SMT siblings are core and core+16, as Zen enumerates them. */
            zen3[core]      = (ColiCpu){ core,      core, 0, 3400000 };
            zen3[core + 16] = (ColiCpu){ core + 16, core, 0, 3400000 };
            want[core] = core;
        }
        CHECK(selects(zen3, 32, want, 16),
              "P2 homogeneous 16C/32T keeps all 16 physical cores");
    }

    /* ---- P3: degenerate topologies ---------------------------------------- */
    {
        static const ColiCpu one[] = { { 0, 0, 0, 2000000 } };
        static const int want[] = { 0 };
        CHECK(selects(one, 1, want, 1), "P3 a single core selects itself");
    }

    /* ---- P4: two-tier ARM ------------------------------------------------- */
    {
        static const ColiCpu big_little[] = {
            { 0, 0, 0, 2800000 }, { 1, 1, 0, 2800000 },
            { 2, 2, 0, 2800000 }, { 3, 3, 0, 2800000 },
            { 4, 4, 0, 1800000 }, { 5, 5, 0, 1800000 },
            { 6, 6, 0, 1800000 }, { 7, 7, 0, 1800000 },
        };
        static const int want[] = { 0, 1, 2, 3 };
        CHECK(selects(big_little, 8, want, 4),
              "P4 big.LITTLE drops the little cores");
    }

    /* ---- P5: no cpufreq at all -------------------------------------------- */
    {
        static const ColiCpu novfreq[] = {
            { 0, 0, 0, 0 }, { 1, 0, 0, 0 },
            { 2, 1, 0, 0 }, { 3, 1, 0, 0 },
        };
        static const int want[] = { 0, 2 };
        CHECK(selects(novfreq, 4, want, 2),
              "P5 unknown frequencies fall back to physical-core sizing");
    }

    /* ---- P6: core ids repeat across packages ------------------------------ */
    {
        static const ColiCpu dual[] = {
            { 0, 0, 0, 3000000 }, { 1, 1, 0, 3000000 },
            { 2, 0, 1, 3000000 }, { 3, 1, 1, 3000000 },
        };
        static const int want[] = { 0, 1, 2, 3 };
        CHECK(selects(dual, 4, want, 4),
              "P6 two sockets with equal core ids stay four distinct cores");
    }

    /* ---- P7: fast_hint overrides an inverted khz signal -------------------
     * Same physical layout as P1 (Meteor Lake), but khz is deliberately
     * swapped -- P-cores read as slow, E-cores read as fast -- while
     * fast_hint (as filled from /sys/devices/cpu_core|cpu_atom/cpus) still
     * marks the P-cores fast. The selection must follow fast_hint, not khz,
     * proving the cpu_core/cpu_atom signal takes precedence when present. */
    {
        static const ColiCpu hinted[] = {
            {  0, 16, 0,  100000, COLI_HINT_FAST }, {  1,  8, 0,  100000, COLI_HINT_FAST },
            {  2,  8, 0,  100000, COLI_HINT_FAST }, {  3, 12, 0,  100000, COLI_HINT_FAST },
            {  4, 12, 0,  100000, COLI_HINT_FAST }, {  5, 16, 0,  100000, COLI_HINT_FAST },
            {  6, 20, 0,  100000, COLI_HINT_FAST }, {  7, 20, 0,  100000, COLI_HINT_FAST },
            {  8, 24, 0,  100000, COLI_HINT_FAST }, {  9, 24, 0,  100000, COLI_HINT_FAST },
            { 10, 28, 0,  100000, COLI_HINT_FAST }, { 11, 28, 0,  100000, COLI_HINT_FAST },
            { 12,  0, 0, 9999999, COLI_HINT_SLOW }, { 13,  1, 0, 9999999, COLI_HINT_SLOW },
            { 14,  2, 0, 9999999, COLI_HINT_SLOW }, { 15,  3, 0, 9999999, COLI_HINT_SLOW },
            { 16,  4, 0, 9999999, COLI_HINT_SLOW }, { 17,  5, 0, 9999999, COLI_HINT_SLOW },
            { 18,  6, 0, 9999999, COLI_HINT_SLOW }, { 19,  7, 0, 9999999, COLI_HINT_SLOW },
            { 20, 32, 0, 9999999, COLI_HINT_SLOW }, { 21, 33, 0, 9999999, COLI_HINT_SLOW },
        };
        static const int want[] = { 0, 1, 3, 6, 8, 10 };
        CHECK(selects(hinted, 22, want, 6),
              "P7 fast_hint overrides an inverted khz signal (cpu_core/cpu_atom)");
    }

    /* ---- coli_parse_cpu_list: Linux cpumap-list string parsing ------------ */
    {
        unsigned char set[32];
        memset(set, 0, sizeof set);
        int n = coli_parse_cpu_list("0-11", set, 32);
        int ok = (n == 12);
        for (int c = 0; c <= 11 && ok; c++) if (!set[c]) ok = 0;
        for (int c = 12; c < 32 && ok; c++) if (set[c]) ok = 0;
        CHECK(ok, "parse_cpu_list \"0-11\" sets cpus 0..11");
    }
    {
        unsigned char set[32];
        memset(set, 0, sizeof set);
        int n = coli_parse_cpu_list("12-21", set, 32);
        int ok = (n == 10);
        for (int c = 12; c <= 21 && ok; c++) if (!set[c]) ok = 0;
        for (int c = 0; c < 12 && ok; c++) if (set[c]) ok = 0;
        CHECK(ok, "parse_cpu_list \"12-21\" sets cpus 12..21");
    }
    {
        unsigned char set[8];
        memset(set, 0, sizeof set);
        int n = coli_parse_cpu_list("", set, 8);
        int ok = (n == 0);
        for (int c = 0; c < 8 && ok; c++) if (set[c]) ok = 0;
        CHECK(ok, "parse_cpu_list empty string sets nothing");
    }
    {
        unsigned char set[8];
        memset(set, 0, sizeof set);
        int n = coli_parse_cpu_list("5", set, 8);
        CHECK(n == 1 && set[5] && !set[0] && !set[4] && !set[6],
              "parse_cpu_list single cpu \"5\"");
    }
    {
        unsigned char set[32];
        memset(set, 0, sizeof set);
        int n = coli_parse_cpu_list("0-5,12-19\n", set, 32);
        int ok = (n == 14);
        for (int c = 0; c <= 5 && ok; c++) if (!set[c]) ok = 0;
        for (int c = 12; c <= 19 && ok; c++) if (!set[c]) ok = 0;
        for (int c = 6; c <= 11 && ok; c++) if (set[c]) ok = 0;
        CHECK(ok, "parse_cpu_list handles comma-separated ranges and trailing newline");
    }
    {
        /* Regression: the range-expansion loop must stop at max_cpu instead of
         * iterating out to hi, or a huge/overflowing sysfs value would hang. */
        unsigned char set[8];
        memset(set, 0, sizeof set);
        int n = coli_parse_cpu_list("5-2000000000", set, 8);
        int ok = (n == 3) && set[5] && set[6] && set[7] && !set[0] && !set[4];
        CHECK(ok, "parse_cpu_list bounds the range loop at max_cpu");
    }

    /* ---- coli_read_sysfs_str: truncation must not look like success ------- */
    {
        char path[] = "/tmp/coli_affinity_test_XXXXXX";
        int fd = mkstemp(path);
        ssize_t n = write(fd, "0-11", 4); (void)n;
        close(fd);
        char buf[5];   /* exactly as long as the content: fits without truncation */
        int ok = coli_read_sysfs_str(path, buf, sizeof buf) && strcmp(buf, "0-11") == 0;
        unlink(path);
        CHECK(ok, "read_sysfs_str reads a file that exactly fits the buffer");
    }
    {
        char path[] = "/tmp/coli_affinity_test_XXXXXX";
        int fd = mkstemp(path);
        ssize_t n = write(fd, "0-11,12-21\n", 11); (void)n;
        close(fd);
        char buf[5];   /* shorter than the content: must report failure, not a cut string */
        int ok = !coli_read_sysfs_str(path, buf, sizeof buf);
        unlink(path);
        CHECK(ok, "read_sysfs_str rejects a read that would truncate");
    }

    printf("\n%s (%d failure%s)\n", fails ? "TEST FAIL" : "ALL PASS",
           fails, fails == 1 ? "" : "s");
    return fails ? 1 : 0;
}
