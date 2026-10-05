/* gguf.h — GGUF v3 container reader (header-only, pure C, no model knowledge).
 *
 * Phase 1 of docs/gguf/ARCHITECTURE.md: index one GGUF file or a split set
 * (`<stem>-00001-of-0000N.gguf`) the way st.h indexes safetensors shards —
 * headers only, never the tensor payload — and hand the engine
 * (name, ggml type, shape, file, absolute offset, byte size) per tensor plus a
 * typed key/value store for the metadata.
 *
 * Posture: the file is UNTRUSTED input (it comes from a mirror). Every length,
 * count, offset and dimension is bounded and overflow-checked before it reaches
 * a malloc or a pread; any violation names the key/tensor and the rule. The
 * parsing functions RETURN -1 with a message in GgufSet.err (so a test can
 * exercise every rule in-process); gguf_open_set_or_die() is the exit(1)
 * wrapper the engine uses, mirroring every other st.h reader.
 *
 * Nothing here computes on weights: the ggml type table only carries the
 * block geometry (elements per block, bytes per block) needed to size and
 * bounds-check tensors. Kernels arrive with gq.h (phase 2).
 *
 * Spec: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
 * Block sizes: ggml/src/ggml-common.h (llama.cpp). */
#ifndef GGUF_H
#define GGUF_H
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>
#include "compat.h"

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#error "gguf.h reads little-endian GGUF files with native loads; big-endian hosts are not supported"
#endif

/* ---- limits (NFR-4) --------------------------------------------------------- */
#define GGUF_MAGIC          0x46554747u     /* "GGUF" as a little-endian u32 */
#define GGUF_VERSION        3
#define GGUF_MAX_SPLITS     512
#define GGUF_MAX_STR        (64u << 10)     /* keys, tensor names, string values, tokens */
#define GGUF_MAX_TENSORS    (1 << 20)
#define GGUF_MAX_KV         (1 << 16)
#define GGUF_MAX_ARR        (1 << 24)       /* elements per array (a 155k vocab is far below) */
#define GGUF_MAX_META       (1ll << 30)     /* bytes retained from one part's metadata */
#define GGUF_MAX_ALIGN      (1 << 20)
#define GGUF_DEFAULT_ALIGN  32
#define GGUF_MAX_DIMS       4
#define GGUF_ERR_LEN        512

/* ---- metadata value types ----------------------------------------------------- */
enum { GGUF_T_U8 = 0, GGUF_T_I8, GGUF_T_U16, GGUF_T_I16, GGUF_T_U32, GGUF_T_I32, GGUF_T_F32,
       GGUF_T_BOOL, GGUF_T_STR, GGUF_T_ARR, GGUF_T_U64, GGUF_T_I64, GGUF_T_F64, GGUF_T_COUNT };
static const int gguf_vt_size[GGUF_T_COUNT] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
static const char *const gguf_vt_name[GGUF_T_COUNT] = {"u8", "i8", "u16", "i16", "u32", "i32", "f32",
                                                        "bool", "str", "arr", "u64", "i64", "f64"};

/* ---- ggml tensor types: block geometry only ------------------------------------ */
enum {
    GGML_TYPE_F32 = 0, GGML_TYPE_F16 = 1, GGML_TYPE_Q4_0 = 2, GGML_TYPE_Q4_1 = 3,
    GGML_TYPE_Q5_0 = 6, GGML_TYPE_Q5_1 = 7, GGML_TYPE_Q8_0 = 8, GGML_TYPE_Q8_1 = 9,
    GGML_TYPE_Q2_K = 10, GGML_TYPE_Q3_K = 11, GGML_TYPE_Q4_K = 12, GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14, GGML_TYPE_Q8_K = 15, GGML_TYPE_IQ2_XXS = 16, GGML_TYPE_IQ2_XS = 17,
    GGML_TYPE_IQ3_XXS = 18, GGML_TYPE_IQ1_S = 19, GGML_TYPE_IQ4_NL = 20, GGML_TYPE_IQ3_S = 21,
    GGML_TYPE_IQ2_S = 22, GGML_TYPE_IQ4_XS = 23, GGML_TYPE_I8 = 24, GGML_TYPE_I16 = 25,
    GGML_TYPE_I32 = 26, GGML_TYPE_I64 = 27, GGML_TYPE_F64 = 28, GGML_TYPE_IQ1_M = 29,
    GGML_TYPE_BF16 = 30, GGML_TYPE_TQ1_0 = 34, GGML_TYPE_TQ2_0 = 35, GGML_TYPE_MXFP4 = 39,
    GGML_TYPE_COUNT = 40
};
typedef struct { const char *name; int block; int tsize; } GgmlTypeInfo;
/* {name, elements per block, bytes per block}; ids 4, 5, 31..33, 36..38 were removed
 * upstream and stay unnamed (= unknown) here. */
static const GgmlTypeInfo gguf_types[GGML_TYPE_COUNT] = {
    [GGML_TYPE_F32]     = {"F32",     1,   4}, [GGML_TYPE_F16]     = {"F16",     1,   2},
    [GGML_TYPE_Q4_0]    = {"Q4_0",    32,  18}, [GGML_TYPE_Q4_1]   = {"Q4_1",    32,  20},
    [GGML_TYPE_Q5_0]    = {"Q5_0",    32,  22}, [GGML_TYPE_Q5_1]   = {"Q5_1",    32,  24},
    [GGML_TYPE_Q8_0]    = {"Q8_0",    32,  34}, [GGML_TYPE_Q8_1]   = {"Q8_1",    32,  36},
    [GGML_TYPE_Q2_K]    = {"Q2_K",    256, 84}, [GGML_TYPE_Q3_K]   = {"Q3_K",    256, 110},
    [GGML_TYPE_Q4_K]    = {"Q4_K",    256, 144}, [GGML_TYPE_Q5_K]  = {"Q5_K",    256, 176},
    [GGML_TYPE_Q6_K]    = {"Q6_K",    256, 210}, [GGML_TYPE_Q8_K]  = {"Q8_K",    256, 292},
    [GGML_TYPE_IQ2_XXS] = {"IQ2_XXS", 256, 66}, [GGML_TYPE_IQ2_XS] = {"IQ2_XS",  256, 74},
    [GGML_TYPE_IQ3_XXS] = {"IQ3_XXS", 256, 98}, [GGML_TYPE_IQ1_S]  = {"IQ1_S",   256, 50},
    [GGML_TYPE_IQ4_NL]  = {"IQ4_NL",  32,  18}, [GGML_TYPE_IQ3_S]  = {"IQ3_S",   256, 110},
    [GGML_TYPE_IQ2_S]   = {"IQ2_S",   256, 82}, [GGML_TYPE_IQ4_XS] = {"IQ4_XS",  256, 136},
    [GGML_TYPE_I8]      = {"I8",      1,   1}, [GGML_TYPE_I16]     = {"I16",     1,   2},
    [GGML_TYPE_I32]     = {"I32",     1,   4}, [GGML_TYPE_I64]     = {"I64",     1,   8},
    [GGML_TYPE_F64]     = {"F64",     1,   8}, [GGML_TYPE_IQ1_M]   = {"IQ1_M",   256, 56},
    [GGML_TYPE_BF16]    = {"BF16",    1,   2}, [GGML_TYPE_TQ1_0]   = {"TQ1_0",   256, 54},
    [GGML_TYPE_TQ2_0]   = {"TQ2_0",   256, 66}, [GGML_TYPE_MXFP4]  = {"MXFP4",   32,  17},
};
static inline int gguf_type_known(int t) { return t >= 0 && t < GGML_TYPE_COUNT && gguf_types[t].name != NULL; }
static inline const char *gguf_type_name(int t) { return gguf_type_known(t) ? gguf_types[t].name : "unknown"; }
/* bytes of ONE row of ne0 elements; -1 if the type is unknown or ne0 is not a
 * whole number of blocks (llama.cpp refuses such tensors too). */
static inline int64_t gguf_row_size(int t, int64_t ne0) {
    if (!gguf_type_known(t) || ne0 < 0 || ne0 % gguf_types[t].block) return -1;
    return ne0 / gguf_types[t].block * (int64_t)gguf_types[t].tsize;
}
static inline double gguf_bits_per_weight(int t) {
    return gguf_type_known(t) ? 8.0 * gguf_types[t].tsize / gguf_types[t].block : 0.0;
}

/* ---- data structures ---------------------------------------------------------- */
typedef struct {
    char   *path;
    int     fd, dfd;            /* buffered fd + O_DIRECT twin (-1 if none) */
    int     mfd, mdfd;          /* dual-SSD mirror replica, -1 = absent */
    int64_t size;               /* file size */
    int64_t data_off;           /* start of the tensor data section (aligned) */
    int64_t align;              /* general.alignment of THIS part */
    int64_t n_tensors, n_kv;
    int     split_no, split_count, split_tensors;   /* -1 when absent */
} GgufFile;

/* Offsets instead of pointers: the arenas grow with realloc while parsing. */
typedef struct {
    int64_t key;                /* sarena offset of the NUL-terminated key */
    int     type;               /* GGUF_T_* */
    int     atype;              /* array element type, -1 for scalars */
    int64_t n;                  /* elements (1 for scalars) */
    int64_t data;               /* scalar / scalar array: barena offset of raw LE bytes
                                 * string: sarena offset
                                 * string array: iarena offset of n sarena offsets
                                 * nested array: -1 (parsed, bounds-checked, not retained) */
    int     file;
} GgufKV;

typedef struct {
    int64_t name;               /* sarena offset */
    int     type;               /* ggml type id (may be unknown: nbytes == -1) */
    int     n_dims;
    int64_t ne[GGUF_MAX_DIMS];  /* unused dims are 1 */
    int64_t nbytes;             /* ne[1]*ne[2]*ne[3] * row_size(type, ne[0]); -1 if unknown type */
    int64_t rel_off;            /* offset relative to the data section */
    int64_t off;                /* ABSOLUTE offset in files[file] */
    int     file;
} GgufTensor;

typedef struct {
    GgufFile    files[GGUF_MAX_SPLITS]; int nfiles;
    GgufKV     *kv;  int64_t nkv, kvcap;
    GgufTensor *t;   int64_t nt, tcap;
    int        *hidx; int64_t hcap;             /* open-addressing name index, as st.h */
    char       *sarena; int64_t slen, scap;     /* NUL-terminated strings */
    uint8_t    *barena; int64_t blen, bcap;     /* raw little-endian scalar bytes */
    int64_t    *iarena; int64_t ilen, icap;     /* string-array element offsets */
    int         nmirror;
    char        err[GGUF_ERR_LEN];
} GgufSet;

#define GGUF_FAIL(G, ...) do { snprintf((G)->err, GGUF_ERR_LEN, __VA_ARGS__); return -1; } while (0)

/* ---- little-endian decode (native loads: see the #error above) ----------------- */
static inline uint16_t gguf_le16(const void *p) { uint16_t v; memcpy(&v, p, 2); return v; }
static inline uint32_t gguf_le32(const void *p) { uint32_t v; memcpy(&v, p, 4); return v; }
static inline uint64_t gguf_le64(const void *p) { uint64_t v; memcpy(&v, p, 8); return v; }

/* ---- arenas -------------------------------------------------------------------- */
static int gguf_arena_grow(GgufSet *G, void **base, int64_t *cap, int64_t need, size_t elem) {
    if (need <= *cap) return 0;
    int64_t nc = *cap ? *cap : 4096;
    while (nc < need) { if (nc > (INT64_MAX >> 2)) GGUF_FAIL(G, "metadata arena overflow"); nc <<= 1; }
    void *nb = realloc(*base, (size_t)nc * elem);
    if (!nb) GGUF_FAIL(G, "out of memory growing metadata arena to %lld bytes", (long long)nc * (long long)elem);
    *base = nb; *cap = nc; return 0;
}
static int gguf_sarena_reserve(GgufSet *G, int64_t n) { return gguf_arena_grow(G, (void **)&G->sarena, &G->scap, G->slen + n, 1); }
static int gguf_barena_reserve(GgufSet *G, int64_t n) { return gguf_arena_grow(G, (void **)&G->barena, &G->bcap, G->blen + n, 1); }
static int gguf_iarena_reserve(GgufSet *G, int64_t n) { return gguf_arena_grow(G, (void **)&G->iarena, &G->icap, G->ilen + n, sizeof(int64_t)); }
static inline const char *gguf_str(const GgufSet *G, int64_t off) { return G->sarena + off; }

/* ---- buffered read cursor over one part ---------------------------------------- */
typedef struct {
    int fd; int64_t size, pos;
    uint8_t *buf; int64_t boff, blen, bcap;
    const char *path;
} GgufCur;
#define GGUF_CUR_WINDOW (1 << 20)

static int gguf_cur_fill(GgufSet *G, GgufCur *c, int64_t n) {
    if (n < 0 || c->pos < 0 || n > c->size - c->pos)
        GGUF_FAIL(G, "%s: metadata runs past the end of the file (need %lld bytes at %lld, file is %lld)",
                  c->path, (long long)n, (long long)c->pos, (long long)c->size);
    if (c->pos >= c->boff && c->pos + n <= c->boff + c->blen) return 0;
    int64_t want = n > GGUF_CUR_WINDOW ? n : GGUF_CUR_WINDOW;
    if (want > c->size - c->pos) want = c->size - c->pos;
    if (want > c->bcap) {
        uint8_t *nb = realloc(c->buf, (size_t)want);
        if (!nb) GGUF_FAIL(G, "%s: out of memory for a %lld-byte metadata window", c->path, (long long)want);
        c->buf = nb; c->bcap = want;
    }
    int64_t got = 0;
    while (got < want) {
        ssize_t r = pread(c->fd, c->buf + got, (size_t)(want - got), c->pos + got);
        if (r < 0) { if (errno == EINTR) continue; GGUF_FAIL(G, "%s: %s while reading metadata at %lld", c->path, strerror(errno), (long long)(c->pos + got)); }
        if (r == 0) GGUF_FAIL(G, "%s: short read in metadata at %lld — truncated file?", c->path, (long long)(c->pos + got));
        got += r;
    }
    c->boff = c->pos; c->blen = got;
    return 0;
}
static int gguf_cur_read(GgufSet *G, GgufCur *c, void *dst, int64_t n) {
    if (gguf_cur_fill(G, c, n)) return -1;
    memcpy(dst, c->buf + (c->pos - c->boff), (size_t)n);
    c->pos += n; return 0;
}
static int gguf_cur_u32(GgufSet *G, GgufCur *c, uint32_t *v) { uint8_t b[4]; if (gguf_cur_read(G, c, b, 4)) return -1; *v = gguf_le32(b); return 0; }
static int gguf_cur_u64(GgufSet *G, GgufCur *c, uint64_t *v) { uint8_t b[8]; if (gguf_cur_read(G, c, b, 8)) return -1; *v = gguf_le64(b); return 0; }
/* read a GGUF string into the string arena; *off receives its sarena offset */
static int gguf_cur_str(GgufSet *G, GgufCur *c, const char *what, int64_t *off) {
    uint64_t len;
    if (gguf_cur_u64(G, c, &len)) return -1;
    if (len > GGUF_MAX_STR) GGUF_FAIL(G, "%s: %s is %llu bytes long (limit %u)", c->path, what, (unsigned long long)len, GGUF_MAX_STR);
    if (gguf_sarena_reserve(G, (int64_t)len + 1)) return -1;
    if (gguf_cur_read(G, c, G->sarena + G->slen, (int64_t)len)) return -1;
    if (memchr(G->sarena + G->slen, 0, (size_t)len)) GGUF_FAIL(G, "%s: %s contains a NUL byte", c->path, what);
    *off = G->slen; G->slen += (int64_t)len; G->sarena[G->slen++] = 0;
    return 0;
}

/* ---- value parsing -------------------------------------------------------------- */
/* Parses one value of `type` at the cursor. retain=0 parses and bounds-checks but
 * rolls the arenas back (parts > 0 of a split set repeat the whole metadata; we keep
 * one copy). depth guards nested arrays (spec allows them; nothing in the wild
 * nests deeper than one level). Fills kv->{atype,n,data}. */
static int gguf_parse_value(GgufSet *G, GgufCur *c, const char *key, int type, int depth, GgufKV *kv, int retain) {
    if (type < 0 || type >= GGUF_T_COUNT) GGUF_FAIL(G, "%s: key %s has unknown value type %d", c->path, key, type);
    kv->type = type; kv->atype = -1; kv->n = 1; kv->data = -1;
    if (type == GGUF_T_STR) {
        int64_t so; if (gguf_cur_str(G, c, key, &so)) return -1;
        kv->data = so; return 0;
    }
    if (type != GGUF_T_ARR) {
        int sz = gguf_vt_size[type];
        if (gguf_barena_reserve(G, sz)) return -1;
        if (gguf_cur_read(G, c, G->barena + G->blen, sz)) return -1;
        kv->data = G->blen; G->blen += sz; return 0;
    }
    uint32_t at; uint64_t n;
    if (gguf_cur_u32(G, c, &at) || gguf_cur_u64(G, c, &n)) return -1;
    if (at >= GGUF_T_COUNT) GGUF_FAIL(G, "%s: array %s has unknown element type %u", c->path, key, at);
    if (n > GGUF_MAX_ARR) GGUF_FAIL(G, "%s: array %s has %llu elements (limit %d)", c->path, key, (unsigned long long)n, GGUF_MAX_ARR);
    kv->atype = (int)at; kv->n = (int64_t)n;
    if (at == GGUF_T_ARR) {                       /* nested: validate, do not retain */
        if (depth >= 1) GGUF_FAIL(G, "%s: array %s nests deeper than 2 levels", c->path, key);
        for (uint64_t i = 0; i < n; i++) {        /* each element is itself an array value */
            GgufKV inner;
            int64_t s0 = G->slen, b0 = G->blen, i0 = G->ilen;
            if (gguf_parse_value(G, c, key, GGUF_T_ARR, depth + 1, &inner, 0)) return -1;
            G->slen = s0; G->blen = b0; G->ilen = i0;
        }
        kv->data = -1; return 0;
    }
    if (at == GGUF_T_STR) {
        if (gguf_iarena_reserve(G, (int64_t)n)) return -1;
        int64_t base = G->ilen; G->ilen += (int64_t)n;
        for (uint64_t i = 0; i < n; i++) {
            int64_t so; if (gguf_cur_str(G, c, key, &so)) return -1;
            G->iarena[base + (int64_t)i] = so;
            if (G->slen > GGUF_MAX_META) GGUF_FAIL(G, "%s: metadata exceeds %lld bytes", c->path, (long long)GGUF_MAX_META);
        }
        kv->data = base; return 0;
    }
    int64_t sz = (int64_t)gguf_vt_size[at] * (int64_t)n;
    if (G->blen + sz > GGUF_MAX_META) GGUF_FAIL(G, "%s: metadata exceeds %lld bytes", c->path, (long long)GGUF_MAX_META);
    if (gguf_barena_reserve(G, sz)) return -1;
    if (gguf_cur_read(G, c, G->barena + G->blen, sz)) return -1;
    kv->data = G->blen; G->blen += sz;
    (void)retain;
    return 0;
}

/* ---- typed KV access ------------------------------------------------------------- */
static const GgufKV *gguf_kv(const GgufSet *G, const char *key) {
    for (int64_t i = 0; i < G->nkv; i++) if (!strcmp(gguf_str(G, G->kv[i].key), key)) return &G->kv[i];
    return NULL;
}
static const GgufKV *gguf_kv_in_file(const GgufSet *G, int file, const char *key) {
    for (int64_t i = 0; i < G->nkv; i++) if (G->kv[i].file == file && !strcmp(gguf_str(G, G->kv[i].key), key)) return &G->kv[i];
    return NULL;
}
static int gguf_decode_i64(const GgufSet *G, int type, int64_t boff, int64_t *out) {
    const uint8_t *p = G->barena + boff;
    switch (type) {
        case GGUF_T_U8:   *out = p[0]; return 1;
        case GGUF_T_I8:   *out = (int8_t)p[0]; return 1;
        case GGUF_T_BOOL: *out = p[0] != 0; return 1;
        case GGUF_T_U16:  *out = gguf_le16(p); return 1;
        case GGUF_T_I16:  *out = (int16_t)gguf_le16(p); return 1;
        case GGUF_T_U32:  *out = gguf_le32(p); return 1;
        case GGUF_T_I32:  *out = (int32_t)gguf_le32(p); return 1;
        case GGUF_T_U64:  { uint64_t u = gguf_le64(p); if (u > (uint64_t)INT64_MAX) return 0; *out = (int64_t)u; return 1; }
        case GGUF_T_I64:  *out = (int64_t)gguf_le64(p); return 1;
        default: return 0;
    }
}
static int gguf_decode_f64(const GgufSet *G, int type, int64_t boff, double *out) {
    const uint8_t *p = G->barena + boff;
    if (type == GGUF_T_F32) { uint32_t u = gguf_le32(p); float f; memcpy(&f, &u, 4); *out = f; return 1; }
    if (type == GGUF_T_F64) { uint64_t u = gguf_le64(p); double d; memcpy(&d, &u, 8); *out = d; return 1; }
    int64_t i; if (gguf_decode_i64(G, type, boff, &i)) { *out = (double)i; return 1; }
    return 0;
}
/* 1 if present and integer-typed (bool included); value in *out. */
static int gguf_kv_i64(const GgufSet *G, const char *key, int64_t *out) {
    const GgufKV *k = gguf_kv(G, key);
    return (k && k->type != GGUF_T_ARR && k->type != GGUF_T_STR) ? gguf_decode_i64(G, k->type, k->data, out) : 0;
}
static int64_t gguf_kv_i64_or(const GgufSet *G, const char *key, int64_t dflt) { int64_t v; return gguf_kv_i64(G, key, &v) ? v : dflt; }
static int gguf_kv_f64(const GgufSet *G, const char *key, double *out) {
    const GgufKV *k = gguf_kv(G, key);
    return (k && k->type != GGUF_T_ARR && k->type != GGUF_T_STR) ? gguf_decode_f64(G, k->type, k->data, out) : 0;
}
static const char *gguf_kv_str(const GgufSet *G, const char *key) {
    const GgufKV *k = gguf_kv(G, key);
    return (k && k->type == GGUF_T_STR) ? gguf_str(G, k->data) : NULL;
}
static int64_t gguf_kv_arr_len(const GgufSet *G, const char *key) {
    const GgufKV *k = gguf_kv(G, key);
    return (k && k->type == GGUF_T_ARR) ? k->n : -1;
}
static const char *gguf_kv_arr_str(const GgufSet *G, const char *key, int64_t i) {
    const GgufKV *k = gguf_kv(G, key);
    if (!k || k->type != GGUF_T_ARR || k->atype != GGUF_T_STR || i < 0 || i >= k->n) return NULL;
    return gguf_str(G, G->iarena[k->data + i]);
}
static int gguf_kv_arr_i64(const GgufSet *G, const char *key, int64_t i, int64_t *out) {
    const GgufKV *k = gguf_kv(G, key);
    if (!k || k->type != GGUF_T_ARR || k->atype == GGUF_T_STR || k->atype == GGUF_T_ARR || i < 0 || i >= k->n) return 0;
    return gguf_decode_i64(G, k->atype, k->data + i * gguf_vt_size[k->atype], out);
}
static int gguf_kv_arr_f64(const GgufSet *G, const char *key, int64_t i, double *out) {
    const GgufKV *k = gguf_kv(G, key);
    if (!k || k->type != GGUF_T_ARR || k->atype == GGUF_T_STR || k->atype == GGUF_T_ARR || i < 0 || i >= k->n) return 0;
    return gguf_decode_f64(G, k->atype, k->data + i * gguf_vt_size[k->atype], out);
}

/* ---- name index ------------------------------------------------------------------- */
static uint64_t gguf_hash(const char *s) {
    uint64_t h = 1469598103934665603ULL;
    while (*s) { h ^= (unsigned char)*s++; h *= 1099511628211ULL; }
    return h;
}
static int gguf_build_index(GgufSet *G) {
    int64_t cap = 16; while (cap < G->nt * 2) cap <<= 1;
    free(G->hidx); G->hidx = malloc((size_t)cap * sizeof(int)); G->hcap = cap;
    if (!G->hidx) GGUF_FAIL(G, "out of memory for the tensor index");
    for (int64_t i = 0; i < cap; i++) G->hidx[i] = -1;
    for (int64_t i = 0; i < G->nt; i++) {
        const char *nm = gguf_str(G, G->t[i].name);
        uint64_t h = gguf_hash(nm) & (uint64_t)(cap - 1);
        while (G->hidx[h] >= 0) {
            if (!strcmp(gguf_str(G, G->t[G->hidx[h]].name), nm))
                GGUF_FAIL(G, "%s: duplicate tensor name %s (also in %s)", G->files[G->t[i].file].path, nm, G->files[G->t[G->hidx[h]].file].path);
            h = (h + 1) & (uint64_t)(cap - 1);
        }
        G->hidx[h] = (int)i;
    }
    return 0;
}
static GgufTensor *gguf_find(GgufSet *G, const char *name) {
    if (!G->hidx) return NULL;
    uint64_t h = gguf_hash(name) & (uint64_t)(G->hcap - 1);
    while (G->hidx[h] >= 0) {
        GgufTensor *t = &G->t[G->hidx[h]];
        if (!strcmp(gguf_str(G, t->name), name)) return t;
        h = (h + 1) & (uint64_t)(G->hcap - 1);
    }
    return NULL;
}
static int gguf_has(GgufSet *G, const char *name) { return gguf_find(G, name) != NULL; }
static inline const char *gguf_tensor_name(const GgufSet *G, const GgufTensor *t) { return gguf_str(G, t->name); }

/* ---- one part ---------------------------------------------------------------------- */
static int gguf_parse_file(GgufSet *G, int fi, int retain_kv) {
    GgufFile *F = &G->files[fi];
    F->fd = open(F->path, COMPAT_O_RDONLY);
    if (F->fd < 0) GGUF_FAIL(G, "%s: %s", F->path, strerror(errno));
    F->size = lseek(F->fd, 0, SEEK_END);
    if (F->size < 0) GGUF_FAIL(G, "%s: cannot determine file size", F->path);
    F->dfd = F->mfd = F->mdfd = -1;
    F->split_no = F->split_count = F->split_tensors = -1;
    GgufCur c = {F->fd, F->size, 0, NULL, 0, 0, 0, F->path};
    int rc = -1;
    uint32_t magic, version; uint64_t nt, nkv;
    if (gguf_cur_u32(G, &c, &magic)) goto done;
    if (magic != GGUF_MAGIC) { snprintf(G->err, GGUF_ERR_LEN, "%s: not a GGUF file (magic 0x%08x)", F->path, magic); goto done; }
    if (gguf_cur_u32(G, &c, &version)) goto done;
    if (version != GGUF_VERSION) { snprintf(G->err, GGUF_ERR_LEN, "%s: GGUF version %u is not supported (need %d)", F->path, version, GGUF_VERSION); goto done; }
    if (gguf_cur_u64(G, &c, &nt) || gguf_cur_u64(G, &c, &nkv)) goto done;
    if (nt > GGUF_MAX_TENSORS) { snprintf(G->err, GGUF_ERR_LEN, "%s: %llu tensors declared (limit %d)", F->path, (unsigned long long)nt, GGUF_MAX_TENSORS); goto done; }
    if (nkv > GGUF_MAX_KV) { snprintf(G->err, GGUF_ERR_LEN, "%s: %llu metadata keys declared (limit %d)", F->path, (unsigned long long)nkv, GGUF_MAX_KV); goto done; }
    if (G->nt + (int64_t)nt > GGUF_MAX_TENSORS) { snprintf(G->err, GGUF_ERR_LEN, "split set declares more than %d tensors", GGUF_MAX_TENSORS); goto done; }
    F->n_tensors = (int64_t)nt; F->n_kv = (int64_t)nkv;

    /* key/value pairs */
    for (uint64_t i = 0; i < nkv; i++) {
        int64_t s0 = G->slen, b0 = G->blen, i0 = G->ilen;
        int64_t koff; if (gguf_cur_str(G, &c, "metadata key", &koff)) goto done;
        uint32_t vt; if (gguf_cur_u32(G, &c, &vt)) goto done;
        const char *key = gguf_str(G, koff);
        GgufKV kv; memset(&kv, 0, sizeof kv); kv.key = koff; kv.file = fi;
        if (gguf_parse_value(G, &c, key, (int)vt, 0, &kv, retain_kv)) goto done;
        /* parts > 0 repeat the model metadata: keep only what describes THIS part */
        int keep = retain_kv || !strncmp(key, "split.", 6) || !strcmp(key, "general.alignment");
        if (!keep) { G->slen = s0; G->blen = b0; G->ilen = i0; continue; }
        if (gguf_arena_grow(G, (void **)&G->kv, &G->kvcap, G->nkv + 1, sizeof(GgufKV))) goto done;
        G->kv[G->nkv++] = kv;
        if (G->slen + G->blen > GGUF_MAX_META) { snprintf(G->err, GGUF_ERR_LEN, "%s: metadata exceeds %lld bytes", F->path, (long long)GGUF_MAX_META); goto done; }
    }
    /* alignment: this part's own key (default 32), power of two */
    { const GgufKV *ak = gguf_kv_in_file(G, fi, "general.alignment"); int64_t al = GGUF_DEFAULT_ALIGN;
      if (ak && !(ak->type != GGUF_T_ARR && ak->type != GGUF_T_STR && gguf_decode_i64(G, ak->type, ak->data, &al)))
          { snprintf(G->err, GGUF_ERR_LEN, "%s: general.alignment is not an integer", F->path); goto done; }
      if (al <= 0 || al > GGUF_MAX_ALIGN || (al & (al - 1)))
          { snprintf(G->err, GGUF_ERR_LEN, "%s: general.alignment=%lld is not a power of two in [1,%d]", F->path, (long long)al, GGUF_MAX_ALIGN); goto done; }
      F->align = al; }
    { int64_t v;
      const GgufKV *k;
      if ((k = gguf_kv_in_file(G, fi, "split.no")) && k->type != GGUF_T_ARR && k->type != GGUF_T_STR && gguf_decode_i64(G, k->type, k->data, &v)) F->split_no = (int)v;
      if ((k = gguf_kv_in_file(G, fi, "split.count")) && k->type != GGUF_T_ARR && k->type != GGUF_T_STR && gguf_decode_i64(G, k->type, k->data, &v)) F->split_count = (int)v;
      if ((k = gguf_kv_in_file(G, fi, "split.tensors.count")) && k->type != GGUF_T_ARR && k->type != GGUF_T_STR && gguf_decode_i64(G, k->type, k->data, &v)) F->split_tensors = (int)v; }

    /* tensor table */
    int64_t t0 = G->nt;
    if (gguf_arena_grow(G, (void **)&G->t, &G->tcap, G->nt + (int64_t)nt, sizeof(GgufTensor))) goto done;
    for (uint64_t i = 0; i < nt; i++) {
        GgufTensor *T = &G->t[G->nt]; memset(T, 0, sizeof *T); T->file = fi;
        if (gguf_cur_str(G, &c, "tensor name", &T->name)) goto done;
        const char *nm = gguf_str(G, T->name);
        uint32_t nd; if (gguf_cur_u32(G, &c, &nd)) goto done;
        if (nd < 1 || nd > GGUF_MAX_DIMS) { snprintf(G->err, GGUF_ERR_LEN, "%s: tensor %s has %u dimensions (1..%d)", F->path, nm, nd, GGUF_MAX_DIMS); goto done; }
        T->n_dims = (int)nd;
        for (int d = 0; d < GGUF_MAX_DIMS; d++) T->ne[d] = 1;
        for (uint32_t d = 0; d < nd; d++) {
            uint64_t e; if (gguf_cur_u64(G, &c, &e)) goto done;
            if (e < 1 || e > (uint64_t)INT64_MAX) { snprintf(G->err, GGUF_ERR_LEN, "%s: tensor %s has dimension %u = %llu", F->path, nm, d, (unsigned long long)e); goto done; }
            T->ne[d] = (int64_t)e;
        }
        uint32_t ty; uint64_t off;
        if (gguf_cur_u32(G, &c, &ty) || gguf_cur_u64(G, &c, &off)) goto done;
        if (ty > INT32_MAX) { snprintf(G->err, GGUF_ERR_LEN, "%s: tensor %s has type id %u", F->path, nm, ty); goto done; }
        T->type = (int)ty;
        if (off > (uint64_t)INT64_MAX) { snprintf(G->err, GGUF_ERR_LEN, "%s: tensor %s offset %llu out of range", F->path, nm, (unsigned long long)off); goto done; }
        T->rel_off = (int64_t)off;
        if (gguf_type_known(T->type)) {
            int64_t row = gguf_row_size(T->type, T->ne[0]);
            if (row < 0) { snprintf(G->err, GGUF_ERR_LEN, "%s: tensor %s: ne[0]=%lld is not a multiple of the %s block (%d)", F->path, nm, (long long)T->ne[0], gguf_type_name(T->type), gguf_types[T->type].block); goto done; }
            int64_t rows, nb;
            if (__builtin_mul_overflow(T->ne[1], T->ne[2], &rows) || __builtin_mul_overflow(rows, T->ne[3], &rows) ||
                __builtin_mul_overflow(rows, row, &nb)) { snprintf(G->err, GGUF_ERR_LEN, "%s: tensor %s: shape overflows 64 bits", F->path, nm); goto done; }
            T->nbytes = nb;
        } else T->nbytes = -1;                        /* indexed so doctor can list it; refused on use */
        G->nt++;
    }
    /* data section */
    { int64_t p = c.pos, al = F->align;
      F->data_off = (p + al - 1) / al * al;
      if (F->data_off > F->size) { snprintf(G->err, GGUF_ERR_LEN, "%s: tensor data section starts past the end of the file", F->path); goto done; }
      int64_t dsz = F->size - F->data_off;
      for (int64_t i = t0; i < G->nt; i++) {
          GgufTensor *T = &G->t[i]; const char *nm = gguf_str(G, T->name);
          if (T->rel_off % al) { snprintf(G->err, GGUF_ERR_LEN, "%s: tensor %s offset %lld is not %lld-byte aligned", F->path, nm, (long long)T->rel_off, (long long)al); goto done; }
          if (T->rel_off > dsz || (T->nbytes >= 0 && T->nbytes > dsz - T->rel_off)) {
              snprintf(G->err, GGUF_ERR_LEN, "%s: tensor %s (%lld bytes at data+%lld) runs past the end of the file (%lld data bytes)",
                       F->path, nm, (long long)T->nbytes, (long long)T->rel_off, (long long)dsz); goto done; }
          T->off = F->data_off + T->rel_off;
      } }
#ifdef O_DIRECT
    F->dfd = open(F->path, COMPAT_O_RDONLY | O_DIRECT);
#elif defined(__APPLE__) || defined(_WIN32)
    F->dfd = compat_open_direct(F->path);
#endif
    rc = 0;
done:
    free(c.buf);
    return rc;
}

/* ---- file discovery ---------------------------------------------------------------- */
static const char *gguf_basename(const char *p) {
    const char *b = strrchr(p, '/');
#ifdef _WIN32
    const char *b2 = strrchr(p, '\\'); if (b2 && (!b || b2 > b)) b = b2;
#endif
    return b ? b + 1 : p;
}
/* "<stem>-00002-of-00009.gguf" -> stem length, no, count; returns 1 if it is a split name */
static int gguf_split_name(const char *base, int *stem_len, int *no, int *count) {
    size_t n = strlen(base);
    if (n < 5 + 15 || strcmp(base + n - 5, ".gguf")) return 0;
    const char *p = base + n - 5 - 15;             /* "-NNNNN-of-NNNNN" (15 chars) */
    if (p[0] != '-' || strncmp(p + 6, "-of-", 4)) return 0;
    for (int i = 1; i <= 5; i++) if (p[i] < '0' || p[i] > '9' || p[i + 9] < '0' || p[i + 9] > '9') return 0;
    *no = atoi(p + 1); *count = atoi(p + 10); *stem_len = (int)(p - base);
    return *count >= 1 && *no >= 1 && *no <= *count;
}
static int gguf_is_gguf_name(const char *base) { size_t n = strlen(base); return n > 5 && !strcmp(base + n - 5, ".gguf"); }

static int gguf_add_file(GgufSet *G, const char *path) {
    if (G->nfiles >= GGUF_MAX_SPLITS) GGUF_FAIL(G, "more than %d GGUF parts", GGUF_MAX_SPLITS);
    GgufFile *F = &G->files[G->nfiles]; memset(F, 0, sizeof *F);
    F->fd = F->dfd = F->mfd = F->mdfd = -1;
    F->path = strdup(path); if (!F->path) GGUF_FAIL(G, "out of memory");
    G->nfiles++; return 0;
}
/* out = dir/base, or 0 if it would not fit (a path that long cannot exist anyway) */
static int gguf_join(char *out, size_t outn, const char *dir, const char *base) {
    size_t ld = strlen(dir), lb = strlen(base);
    if (ld + 1 + lb + 1 > outn) return 0;
    memcpy(out, dir, ld); out[ld] = '/'; memcpy(out + ld + 1, base, lb); out[ld + 1 + lb] = 0;
    return 1;
}
/* search `base` in dir, then in each ';'/','-separated extra dir */
static int gguf_locate(const char *dir, const char *extra_dirs, const char *base, char *out, size_t outn) {
    struct stat st;
    if (gguf_join(out, outn, dir, base) && !stat(out, &st) && S_ISREG(st.st_mode)) return 1;
    if (!extra_dirs || !*extra_dirs) return 0;
    char buf[4096]; snprintf(buf, sizeof buf, "%s", extra_dirs);
    char *p = buf;
    while (p && *p) {
        char *sep = p; while (*sep && *sep != ';' && *sep != ',') sep++;
        int last = (*sep == 0); *sep = 0;
        if (*p && gguf_join(out, outn, p, base) && !stat(out, &st) && S_ISREG(st.st_mode)) return 1;
        p = last ? NULL : sep + 1;
    }
    return 0;
}

/* Indexes `path`: a single .gguf, the first (or any) part of a split set, or a
 * directory holding exactly one model's parts. extra_dirs (COLI_MODEL_DIRS) is a
 * search path for parts that live on other drives. Returns 0, or -1 with G->err. */
static int gguf_open_set(GgufSet *G, const char *path, const char *extra_dirs) {
    memset(G, 0, sizeof *G);
    struct stat st;
    if (stat(path, &st)) GGUF_FAIL(G, "%s: %s", path, strerror(errno));
    char dir[2048]; char first[2048];
    if (S_ISDIR(st.st_mode)) {
        DIR *d = opendir(path); struct dirent *e;
        if (!d) GGUF_FAIL(G, "%s: %s", path, strerror(errno));
        char stem[1024] = ""; int nstem = 0, ambiguous = 0; first[0] = 0;
        while ((e = readdir(d))) {
            if (!gguf_is_gguf_name(e->d_name)) continue;
            int sl, no, cnt; char s[1024];
            if (gguf_split_name(e->d_name, &sl, &no, &cnt)) snprintf(s, sizeof s, "%.*s", sl, e->d_name);
            else snprintf(s, sizeof s, "%s", e->d_name);
            if (!nstem) { snprintf(stem, sizeof stem, "%s", s); snprintf(first, sizeof first, "%s/%s", path, e->d_name); nstem = 1; }
            else if (strcmp(stem, s)) ambiguous = 1;
            else if (strcmp(e->d_name, gguf_basename(first)) < 0) snprintf(first, sizeof first, "%s/%s", path, e->d_name);
        }
        closedir(d);
        if (!nstem) GGUF_FAIL(G, "%s: no .gguf file in this directory", path);
        if (ambiguous) GGUF_FAIL(G, "%s: several GGUF models in one directory — pass the model file itself", path);
        snprintf(dir, sizeof dir, "%s", path);
    } else {
        snprintf(first, sizeof first, "%s", path);
        const char *b = gguf_basename(path);           /* directory = everything before the basename */
        if (b == path) snprintf(dir, sizeof dir, ".");
        else if (b - path == 1) snprintf(dir, sizeof dir, "%c", path[0]);   /* "/model.gguf" */
        else snprintf(dir, sizeof dir, "%.*s", (int)(b - path - 1), path);
    }
    const char *base = gguf_basename(first);
    int sl, no, cnt;
    if (gguf_split_name(base, &sl, &no, &cnt)) {
        if (cnt > GGUF_MAX_SPLITS) GGUF_FAIL(G, "%s: %d parts (limit %d)", first, cnt, GGUF_MAX_SPLITS);
        for (int k = 1; k <= cnt; k++) {
            char want[1100], found[2048];
            snprintf(want, sizeof want, "%.*s-%05d-of-%05d.gguf", sl, base, k, cnt);
            if (!gguf_locate(dir, extra_dirs, want, found, sizeof found))
                GGUF_FAIL(G, "%.160s: part %d of %d (%.120s) not found in %.120s%s%.60s", first, k, cnt, want, dir,
                          extra_dirs && *extra_dirs ? " or " : "", extra_dirs && *extra_dirs ? extra_dirs : "");
            if (gguf_add_file(G, found)) return -1;
        }
    } else if (gguf_add_file(G, first)) return -1;

    for (int fi = 0; fi < G->nfiles; fi++) if (gguf_parse_file(G, fi, fi == 0)) return -1;

    /* split consistency */
    if (G->nfiles > 1 || G->files[0].split_count > 1) {
        for (int fi = 0; fi < G->nfiles; fi++) {
            GgufFile *F = &G->files[fi];
            if (F->split_count != G->nfiles)
                GGUF_FAIL(G, "%s: split.count=%d but %d part file(s) found", F->path, F->split_count, G->nfiles);
            if (F->split_no != fi)
                GGUF_FAIL(G, "%s: split.no=%d, expected %d from its file name", F->path, F->split_no, fi);
        }
        if (G->files[0].split_tensors >= 0 && G->files[0].split_tensors != G->nt)
            GGUF_FAIL(G, "%s: split.tensors.count=%d but the parts hold %lld tensors", G->files[0].path, G->files[0].split_tensors, (long long)G->nt);
    }
    return gguf_build_index(G);
}
static void gguf_open_set_or_die(GgufSet *G, const char *path, const char *extra_dirs) {
    if (gguf_open_set(G, path, extra_dirs)) { fprintf(stderr, "gguf: %s\n", G->err); exit(1); }
}
static void gguf_close(GgufSet *G) {
    for (int i = 0; i < G->nfiles; i++) {
        GgufFile *F = &G->files[i];
        if (F->fd >= 0) close(F->fd);
        if (F->dfd >= 0) close(F->dfd);
        if (F->mfd >= 0) close(F->mfd);
        if (F->mdfd >= 0) close(F->mdfd);
        free(F->path);
    }
    free(G->kv); free(G->t); free(G->hidx); free(G->sarena); free(G->barena); free(G->iarena);
    memset(G, 0, sizeof *G);
}

/* ---- dual-SSD mirror (FR-7) --------------------------------------------------------- */
/* Registers <dir>/<basename> as a read replica of every part. Accepted only if the
 * size and the whole metadata region [0, data_off) are byte-identical: then every
 * (off, nbytes) is valid on either copy. Missing/divergent parts stay on the
 * primary (partial mirrors are fine). Returns the number of accepted parts. */
static int gguf_mirror_init(GgufSet *G, const char *dir) {
    for (int i = 0; i < G->nfiles; i++) {
        GgufFile *F = &G->files[i];
        if (F->mfd >= 0) { close(F->mfd); F->mfd = -1; }
        if (F->mdfd >= 0) { close(F->mdfd); F->mdfd = -1; }
    }
    G->nmirror = 0;
    for (int i = 0; i < G->nfiles; i++) {
        GgufFile *F = &G->files[i];
        char mp[2048]; snprintf(mp, sizeof mp, "%s/%s", dir, gguf_basename(F->path));
        int mfd = open(mp, COMPAT_O_RDONLY);
        if (mfd < 0) continue;
        if (lseek(mfd, 0, SEEK_END) != F->size) {
            fprintf(stderr, "[MIRROR] %s: size differs from the primary copy — file skipped\n", mp); close(mfd); continue; }
        int ok = 1; int64_t pos = 0; const int64_t CH = 1 << 20;
        char *a = malloc((size_t)CH), *b = malloc((size_t)CH);
        if (!a || !b) ok = 0;
        while (ok && pos < F->data_off) {
            int64_t n = F->data_off - pos; if (n > CH) n = CH;
            if (pread(F->fd, a, (size_t)n, pos) != (ssize_t)n || pread(mfd, b, (size_t)n, pos) != (ssize_t)n || memcmp(a, b, (size_t)n)) ok = 0;
            pos += n;
        }
        free(a); free(b);
        if (!ok) { fprintf(stderr, "[MIRROR] %s: header differs from the primary copy — file skipped\n", mp); close(mfd); continue; }
        F->mfd = mfd;
#ifdef O_DIRECT
        F->mdfd = open(mp, COMPAT_O_RDONLY | O_DIRECT);
#elif defined(__APPLE__) || defined(_WIN32)
        F->mdfd = compat_open_direct(mp);
#endif
        G->nmirror++;
    }
    return G->nmirror;
}
static inline int gguf_fd_rep(const GgufSet *G, int file, int rep) { return rep ? G->files[file].mfd : G->files[file].fd; }
static inline int gguf_direct_fd_rep(const GgufSet *G, int file, int rep) { return rep ? G->files[file].mdfd : G->files[file].dfd; }

/* ---- summaries ----------------------------------------------------------------------- */
/* bytes per known type, for the startup line / inspect; returns total known bytes */
static int64_t gguf_type_mix(const GgufSet *G, int64_t bytes[GGML_TYPE_COUNT], int64_t count[GGML_TYPE_COUNT]) {
    int64_t tot = 0;
    for (int i = 0; i < GGML_TYPE_COUNT; i++) { bytes[i] = 0; count[i] = 0; }
    for (int64_t i = 0; i < G->nt; i++) {
        const GgufTensor *T = &G->t[i];
        if (gguf_type_known(T->type)) { bytes[T->type] += T->nbytes; count[T->type]++; tot += T->nbytes; }
    }
    return tot;
}
/* "glm-dsa · GLM-5.2 · 9 parts · 1495 tensors · Q4_K 71% Q6_K 27% F32 2%" */
static void gguf_describe(const GgufSet *G, char *out, size_t n) {
    int64_t by[GGML_TYPE_COUNT], ct[GGML_TYPE_COUNT]; int64_t tot = gguf_type_mix(G, by, ct);
    const char *arch = gguf_kv_str(G, "general.architecture"), *name = gguf_kv_str(G, "general.name");
    int w = snprintf(out, n, "%s · %s · %d part%s · %lld tensors", arch ? arch : "?", name ? name : "?",
                     G->nfiles, G->nfiles == 1 ? "" : "s", (long long)G->nt);
    for (int k = 0; k < 4 && w < (int)n; k++) {              /* top-4 types by bytes */
        int best = -1; for (int i = 0; i < GGML_TYPE_COUNT; i++) if (by[i] > 0 && (best < 0 || by[i] > by[best])) best = i;
        if (best < 0) break;
        w += snprintf(out + w, n - (size_t)w, "%s%s %.0f%%", k ? " " : " · ", gguf_type_name(best), tot ? 100.0 * (double)by[best] / (double)tot : 0.0);
        by[best] = 0;
    }
}
/* Machine-readable dump (one record per line) used by the C/Python cross-check:
 *   F <idx> <path> <size> <data_off> <align> <split_no> <split_count>
 *   K <key> <type> <atype> <n>
 *   T <name> <type> <n_dims> <ne0> <ne1> <ne2> <ne3> <file> <off> <nbytes>   (with tensors=1) */
static void gguf_dump(const GgufSet *G, FILE *f, int tensors) {
    for (int i = 0; i < G->nfiles; i++) {
        const GgufFile *F = &G->files[i];
        fprintf(f, "F %d %s %lld %lld %lld %d %d\n", i, F->path, (long long)F->size, (long long)F->data_off, (long long)F->align, F->split_no, F->split_count);
    }
    for (int64_t i = 0; i < G->nkv; i++) {
        const GgufKV *k = &G->kv[i];
        fprintf(f, "K %s %s %s %lld", gguf_str(G, k->key), gguf_vt_name[k->type], k->atype >= 0 ? gguf_vt_name[k->atype] : "-", (long long)k->n);
        if (k->type == GGUF_T_STR) fprintf(f, " %s", gguf_str(G, k->data));
        else if (k->type != GGUF_T_ARR) { double d; int64_t v; if (gguf_decode_i64(G, k->type, k->data, &v)) fprintf(f, " %lld", (long long)v); else if (gguf_decode_f64(G, k->type, k->data, &d)) fprintf(f, " %.9g", d); }
        fputc('\n', f);
    }
    if (tensors) for (int64_t i = 0; i < G->nt; i++) {
        const GgufTensor *T = &G->t[i];
        fprintf(f, "T %s %d %d %lld %lld %lld %lld %d %lld %lld\n", gguf_str(G, T->name), T->type, T->n_dims,
                (long long)T->ne[0], (long long)T->ne[1], (long long)T->ne[2], (long long)T->ne[3], T->file, (long long)T->off, (long long)T->nbytes);
    }
}

#endif /* GGUF_H */
