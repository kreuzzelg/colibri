/* gguf.h tests: a tiny in-test GGUF writer produces valid files (alignment 32 and
 * 64, a 3-part split set) and one malformed file per validation rule in
 * ARCHITECTURE.md §4.3; a bit-flip sweep over the whole metadata region checks
 * that no corruption crashes the parser (it must accept or refuse with a
 * message); the dual-SSD mirror acceptance rule is exercised too.
 *
 * Also a CLI for the C/Python cross-check in tests/test_ggufinfo.py:
 *     tests/test_gguf <model.gguf | dir>     -> gguf_dump() to stdout */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/stat.h>
#include <unistd.h>

#include "../gguf.h"

static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); fails++; } } while (0)
#define CHECK_ERR(G, needle) do { if (!strstr((G)->err, needle)) { printf("FAIL %s:%d: error '%s' lacks '%s'\n", __FILE__, __LINE__, (G)->err, needle); fails++; } } while (0)

/* ---- growable byte buffer ------------------------------------------------- */
typedef struct { uint8_t *b; size_t n, cap; } Buf;
static void bput(Buf *B, const void *p, size_t n) {
    if (B->n + n > B->cap) { B->cap = (B->n + n) * 2 + 64; B->b = realloc(B->b, B->cap); if (!B->b) { perror("realloc"); exit(2); } }
    memcpy(B->b + B->n, p, n); B->n += n;
}
static void bu8(Buf *B, uint8_t v) { bput(B, &v, 1); }
static void bu16(Buf *B, uint16_t v) { bput(B, &v, 2); }
static void bu32(Buf *B, uint32_t v) { bput(B, &v, 4); }
static void bu64(Buf *B, uint64_t v) { bput(B, &v, 8); }
static void bi32(Buf *B, int32_t v) { bput(B, &v, 4); }
static void bf32(Buf *B, float v) { bput(B, &v, 4); }
static void bf64(Buf *B, double v) { bput(B, &v, 8); }
static void bstrn(Buf *B, const char *s, size_t n) { bu64(B, n); bput(B, s, n); }
static void bstr(Buf *B, const char *s) { bstrn(B, s, strlen(s)); }
static void bfree(Buf *B) { free(B->b); memset(B, 0, sizeof *B); }

/* ---- KV helpers (written into a KV buffer; the caller counts them) -------- */
static void kv_str(Buf *K, const char *k, const char *v) { bstr(K, k); bu32(K, GGUF_T_STR); bstr(K, v); }
static void kv_u32(Buf *K, const char *k, uint32_t v) { bstr(K, k); bu32(K, GGUF_T_U32); bu32(K, v); }
static void kv_u16(Buf *K, const char *k, uint16_t v) { bstr(K, k); bu32(K, GGUF_T_U16); bu16(K, v); }
static void kv_i32(Buf *K, const char *k, int32_t v) { bstr(K, k); bu32(K, GGUF_T_I32); bi32(K, v); }
static void kv_u8(Buf *K, const char *k, uint8_t v) { bstr(K, k); bu32(K, GGUF_T_U8); bu8(K, v); }
static void kv_bool(Buf *K, const char *k, int v) { bstr(K, k); bu32(K, GGUF_T_BOOL); bu8(K, (uint8_t)v); }
static void kv_f32(Buf *K, const char *k, float v) { bstr(K, k); bu32(K, GGUF_T_F32); bf32(K, v); }
static void kv_f64(Buf *K, const char *k, double v) { bstr(K, k); bu32(K, GGUF_T_F64); bf64(K, v); }
static void kv_u64(Buf *K, const char *k, uint64_t v) { bstr(K, k); bu32(K, GGUF_T_U64); bu64(K, v); }
static void kv_i64(Buf *K, const char *k, int64_t v) { bstr(K, k); bu32(K, GGUF_T_I64); bput(K, &v, 8); }

/* ---- tensor table + data section ------------------------------------------ */
typedef struct { Buf tab, data; int64_t n; int64_t align; } Tensors;
static uint8_t pat(const char *name, int64_t i) { return (uint8_t)((unsigned)name[0] * 7u + (unsigned)i * 13u + 3u); }
/* nbytes<0: size from the type table. off<0: next aligned slot. Returns the relative offset used. */
static int64_t tadd(Tensors *T, const char *name, int type, int nd, const int64_t *ne, int64_t nbytes, int64_t off) {
    if (nbytes < 0) { int64_t rows = 1; for (int d = 1; d < nd; d++) rows *= ne[d]; nbytes = rows * gguf_row_size(type, ne[0]); }
    if (off < 0) { size_t pad = (size_t)((T->align - (int64_t)T->data.n % T->align) % T->align); for (size_t i = 0; i < pad; i++) bu8(&T->data, 0); off = (int64_t)T->data.n; }
    if (off == (int64_t)T->data.n) for (int64_t i = 0; i < nbytes; i++) bu8(&T->data, pat(name, i));
    bstr(&T->tab, name); bu32(&T->tab, (uint32_t)nd);
    for (int d = 0; d < nd; d++) bu64(&T->tab, (uint64_t)ne[d]);
    bu32(&T->tab, (uint32_t)type); bu64(&T->tab, (uint64_t)off);
    T->n++;
    return off;
}
static void tinit(Tensors *T, int64_t align) { memset(T, 0, sizeof *T); T->align = align; }
static void tfree(Tensors *T) { bfree(&T->tab); bfree(&T->data); }

static void write_file(const char *path, const Buf *B) {
    FILE *f = fopen(path, "wb"); if (!f) { perror(path); exit(2); }
    if (B->n && fwrite(B->b, 1, B->n, f) != B->n) { perror(path); exit(2); }
    fclose(f);
}
/* assemble header + kv + tensor table (padded to align) + data */
static void assemble(Buf *out, uint32_t magic, uint32_t version, uint64_t nt_decl, uint64_t nkv_decl,
                     const Buf *kv, const Tensors *T) {
    memset(out, 0, sizeof *out);
    bu32(out, magic); bu32(out, version); bu64(out, nt_decl); bu64(out, nkv_decl);
    bput(out, kv->b, kv->n); bput(out, T->tab.b, T->tab.n);
    while ((int64_t)out->n % T->align) bu8(out, 0);
    bput(out, T->data.b, T->data.n);
}

/* the "demo" model: every KV type, mixed tensor types, one unknown type */
static int demo_kv(Buf *K, int64_t align, int with_split, int split_no, int split_count, int split_tensors) {
    int n = 0;
    kv_str(K, "general.architecture", "demo"); n++;
    kv_str(K, "general.name", "gguf fixture"); n++;
    if (align != GGUF_DEFAULT_ALIGN) { kv_u32(K, "general.alignment", (uint32_t)align); n++; }
    kv_u8(K, "test.u8", 200); kv_u16(K, "test.u16", 60000); kv_u32(K, "test.u32", 4000000000u); n += 3;
    kv_i32(K, "test.i32", -70000); kv_f32(K, "test.f32", 1.5f); kv_bool(K, "test.bool", 1); n += 3;
    kv_u64(K, "test.u64", 1ull << 40); kv_i64(K, "test.i64", -(1ll << 40)); kv_f64(K, "test.f64", 2.25); n += 3;
    bstr(K, "test.arr_i32"); bu32(K, GGUF_T_ARR); bu32(K, GGUF_T_I32); bu64(K, 4); bi32(K, -1); bi32(K, 0); bi32(K, 1); bi32(K, 2); n++;
    bstr(K, "test.arr_str"); bu32(K, GGUF_T_ARR); bu32(K, GGUF_T_STR); bu64(K, 3); bstr(K, "a"); bstr(K, "bb"); bstr(K, "ccc"); n++;
    bstr(K, "test.arr_f32"); bu32(K, GGUF_T_ARR); bu32(K, GGUF_T_F32); bu64(K, 2); bf32(K, 0.5f); bf32(K, -0.5f); n++;
    /* nested: [[1,2] as u8, ["x"] as str] */
    bstr(K, "test.arr_nested"); bu32(K, GGUF_T_ARR); bu32(K, GGUF_T_ARR); bu64(K, 2);
    bu32(K, GGUF_T_U8); bu64(K, 2); bu8(K, 1); bu8(K, 2);
    bu32(K, GGUF_T_STR); bu64(K, 1); bstr(K, "x"); n++;
    kv_str(K, "test.empty_str", ""); n++;
    bstr(K, "test.empty_arr"); bu32(K, GGUF_T_ARR); bu32(K, GGUF_T_I32); bu64(K, 0); n++;
    kv_u32(K, "demo.block_count", 2); n++;
    bstr(K, "tokenizer.ggml.tokens"); bu32(K, GGUF_T_ARR); bu32(K, GGUF_T_STR); bu64(K, 4); bstr(K, "<s>"); bstr(K, "a"); bstr(K, "b"); bstr(K, "c"); n++;
    if (with_split) { kv_u16(K, "split.no", (uint16_t)split_no); kv_u16(K, "split.count", (uint16_t)split_count); kv_i32(K, "split.tensors.count", split_tensors); n += 3; }
    return n;
}
static void demo_tensors(Tensors *T) {
    int64_t ne1[4] = {4, 4}, ne2[4] = {256, 3, 2}, ne3[4] = {256, 2, 2}, ne4[4] = {4}, ne5[4] = {64, 2}, ne6[4] = {32};
    tadd(T, "token_embd.weight", GGML_TYPE_F32, 2, ne1, -1, -1);
    tadd(T, "blk.0.ffn_gate_exps.weight", GGML_TYPE_Q4_K, 3, ne2, -1, -1);
    tadd(T, "blk.0.ffn_down_exps.weight", GGML_TYPE_Q6_K, 3, ne3, -1, -1);
    tadd(T, "blk.0.attn_norm.weight", GGML_TYPE_F16, 1, ne4, -1, -1);
    tadd(T, "blk.1.attn_q_a.weight", GGML_TYPE_Q8_0, 2, ne5, -1, -1);
    tadd(T, "blk.1.odd.weight", 99, 1, ne6, 40, -1);          /* unknown type, explicit payload size */
}
static void write_demo(const char *path, int64_t align) {
    Buf K = {0}, F = {0}; Tensors T; tinit(&T, align);
    int nkv = demo_kv(&K, align, 0, 0, 0, 0); demo_tensors(&T);
    assemble(&F, GGUF_MAGIC, GGUF_VERSION, (uint64_t)T.n, (uint64_t)nkv, &K, &T);
    write_file(path, &F); bfree(&K); bfree(&F); tfree(&T);
}

/* ---- filesystem helpers --------------------------------------------------- */
static const char *TMP = "tests/tmp_gguf";
static void mkd(const char *d) {
#ifdef _WIN32
    mkdir(d);
#else
    mkdir(d, 0755);
#endif
}
static void copy_file(const char *src, const char *dst) {
    FILE *a = fopen(src, "rb"), *b = fopen(dst, "wb"); if (!a || !b) { perror("copy"); exit(2); }
    char buf[65536]; size_t n; while ((n = fread(buf, 1, sizeof buf, a)) > 0) fwrite(buf, 1, n, b);
    fclose(a); fclose(b);
}
static Buf slurp(const char *path) {
    Buf B = {0}; FILE *f = fopen(path, "rb"); if (!f) { perror(path); exit(2); }
    char buf[65536]; size_t n; while ((n = fread(buf, 1, sizeof buf, f)) > 0) bput(&B, buf, n);
    fclose(f); return B;
}
/* a GNU/BSD extension with that name is absent on MinGW: a tiny portable byte search */
static uint8_t *find_bytes(uint8_t *hay, size_t n, const char *needle, size_t m) {
    for (size_t i = 0; m <= n && i + m <= n; i++) if (!memcmp(hay + i, needle, m)) return hay + i;
    return NULL;
}
static char *P(const char *name) { static char p[8][1024]; static int i = 0; char *o = p[i++ & 7]; snprintf(o, 1024, "%s/%s", TMP, name); return o; }

/* ---- tests ----------------------------------------------------------------- */
static void test_valid(int64_t align) {
    const char *path = P(align == 32 ? "demo32.gguf" : "demo64.gguf");
    write_demo(path, align);
    GgufSet G; int rc = gguf_open_set(&G, path, NULL);
    if (rc) printf("  err: %s\n", G.err);
    CHECK(rc == 0);
    CHECK(G.nfiles == 1 && G.nt == 6);
    CHECK(G.files[0].align == align && G.files[0].data_off % align == 0 && G.files[0].split_count == -1);
    /* kv access */
    int64_t v; double d;
    CHECK(gguf_kv_i64(&G, "test.u8", &v) && v == 200);
    CHECK(gguf_kv_i64(&G, "test.u16", &v) && v == 60000);
    CHECK(gguf_kv_i64(&G, "test.u32", &v) && v == 4000000000ll);
    CHECK(gguf_kv_i64(&G, "test.i32", &v) && v == -70000);
    CHECK(gguf_kv_i64(&G, "test.bool", &v) && v == 1);
    CHECK(gguf_kv_i64(&G, "test.u64", &v) && v == (1ll << 40));
    CHECK(gguf_kv_i64(&G, "test.i64", &v) && v == -(1ll << 40));
    CHECK(gguf_kv_f64(&G, "test.f32", &d) && d == 1.5);
    CHECK(gguf_kv_f64(&G, "test.f64", &d) && d == 2.25);
    CHECK(gguf_kv_f64(&G, "test.u8", &d) && d == 200.0);          /* ints read as doubles too */
    CHECK(!gguf_kv_i64(&G, "test.f32", &v));                       /* floats are not ints */
    CHECK(!gguf_kv_i64(&G, "general.name", &v));
    CHECK(gguf_kv_i64_or(&G, "nope", 7) == 7);
    const char *s = gguf_kv_str(&G, "general.architecture"); CHECK(s && !strcmp(s, "demo"));
    s = gguf_kv_str(&G, "test.empty_str"); CHECK(s && !*s);
    CHECK(gguf_kv_str(&G, "test.u8") == NULL);
    CHECK(gguf_kv_arr_len(&G, "test.arr_i32") == 4 && gguf_kv_arr_len(&G, "test.empty_arr") == 0 && gguf_kv_arr_len(&G, "test.u8") == -1);
    CHECK(gguf_kv_arr_i64(&G, "test.arr_i32", 0, &v) && v == -1);
    CHECK(gguf_kv_arr_i64(&G, "test.arr_i32", 3, &v) && v == 2);
    CHECK(!gguf_kv_arr_i64(&G, "test.arr_i32", 4, &v));
    CHECK(gguf_kv_arr_f64(&G, "test.arr_f32", 1, &d) && d == -0.5);
    CHECK(gguf_kv_arr_len(&G, "test.arr_str") == 3);
    s = gguf_kv_arr_str(&G, "test.arr_str", 2); CHECK(s && !strcmp(s, "ccc"));
    CHECK(gguf_kv_arr_str(&G, "test.arr_str", 3) == NULL);
    CHECK(gguf_kv_arr_str(&G, "tokenizer.ggml.tokens", 0) && !strcmp(gguf_kv_arr_str(&G, "tokenizer.ggml.tokens", 0), "<s>"));
    const GgufKV *nk = gguf_kv(&G, "test.arr_nested"); CHECK(nk && nk->type == GGUF_T_ARR && nk->atype == GGUF_T_ARR && nk->n == 2 && nk->data == -1);
    /* tensors */
    GgufTensor *t = gguf_find(&G, "blk.0.ffn_gate_exps.weight");
    CHECK(t && t->type == GGML_TYPE_Q4_K && t->n_dims == 3 && t->ne[0] == 256 && t->ne[1] == 3 && t->ne[2] == 2 && t->ne[3] == 1);
    CHECK(t && t->nbytes == 3 * 2 * 144 && t->off == G.files[0].data_off + t->rel_off && t->rel_off % align == 0);
    t = gguf_find(&G, "blk.0.ffn_down_exps.weight"); CHECK(t && t->nbytes == 2 * 2 * 210);
    t = gguf_find(&G, "blk.1.attn_q_a.weight"); CHECK(t && t->nbytes == 2 * 2 * 34);   /* 64 elems = 2 blocks per row */
    t = gguf_find(&G, "blk.0.attn_norm.weight"); CHECK(t && t->nbytes == 8 && t->n_dims == 1);
    t = gguf_find(&G, "blk.1.odd.weight"); CHECK(t && t->type == 99 && t->nbytes == -1 && !gguf_type_known(t->type));
    CHECK(gguf_find(&G, "missing") == NULL && gguf_has(&G, "token_embd.weight"));
    /* the offsets land on the bytes we wrote */
    t = gguf_find(&G, "token_embd.weight");
    if (t) { uint8_t b[64]; CHECK(pread(G.files[0].fd, b, 64, t->off) == 64); int ok = 1; for (int i = 0; i < 64; i++) if (b[i] != pat("token_embd.weight", i)) ok = 0; CHECK(ok); }
    /* row-size table sanity */
    CHECK(gguf_row_size(GGML_TYPE_Q4_K, 512) == 288 && gguf_row_size(GGML_TYPE_Q4_K, 300) == -1 && gguf_row_size(GGML_TYPE_F32, 7) == 28 && gguf_row_size(99, 32) == -1);
    CHECK(gguf_bits_per_weight(GGML_TYPE_Q4_K) == 4.5 && gguf_bits_per_weight(GGML_TYPE_Q6_K) > 6.56 && gguf_bits_per_weight(GGML_TYPE_Q6_K) < 6.57);
    char desc[256]; gguf_describe(&G, desc, sizeof desc);
    CHECK(strstr(desc, "demo") && strstr(desc, "6 tensors") && strstr(desc, "Q4_K"));
    gguf_close(&G);
}

static void test_split(void) {
    const char *dir = P("split"); mkd(dir);
    /* 3 parts: tensors round-robin, part 0 carries the model metadata */
    Tensors all; tinit(&all, 32); demo_tensors(&all); tfree(&all);
    char paths[3][1024];
    for (int part = 0; part < 3; part++) {
        Buf K = {0}, F = {0}; Tensors T; tinit(&T, 32);
        int nkv;
        if (part == 0) nkv = demo_kv(&K, 32, 1, 0, 3, 6);
        else { nkv = 0; kv_u16(&K, "split.no", (uint16_t)part); kv_u16(&K, "split.count", 3); kv_i32(&K, "split.tensors.count", 6); nkv = 3;
               kv_str(&K, "general.architecture", "demo"); nkv++; }     /* repeated metadata must be dropped */
        int64_t ne1[4] = {4, 4}, ne2[4] = {256, 3, 2}, ne3[4] = {256, 2, 2}, ne4[4] = {4}, ne5[4] = {64, 2}, ne6[4] = {32};
        if (part == 0) { tadd(&T, "token_embd.weight", GGML_TYPE_F32, 2, ne1, -1, -1); tadd(&T, "blk.0.attn_norm.weight", GGML_TYPE_F16, 1, ne4, -1, -1); }
        if (part == 1) { tadd(&T, "blk.0.ffn_gate_exps.weight", GGML_TYPE_Q4_K, 3, ne2, -1, -1); tadd(&T, "blk.1.attn_q_a.weight", GGML_TYPE_Q8_0, 2, ne5, -1, -1); }
        if (part == 2) { tadd(&T, "blk.0.ffn_down_exps.weight", GGML_TYPE_Q6_K, 3, ne3, -1, -1); tadd(&T, "blk.1.odd.weight", 99, 1, ne6, 40, -1); }
        assemble(&F, GGUF_MAGIC, GGUF_VERSION, (uint64_t)T.n, (uint64_t)nkv, &K, &T);
        snprintf(paths[part], sizeof paths[part], "%s/demo-%05d-of-00003.gguf", dir, part + 1);
        write_file(paths[part], &F); bfree(&K); bfree(&F); tfree(&T);
    }
    GgufSet G; int rc;
    rc = gguf_open_set(&G, dir, NULL); if (rc) printf("  err: %s\n", G.err);
    CHECK(rc == 0 && G.nfiles == 3 && G.nt == 6);
    CHECK(G.files[0].split_no == 0 && G.files[2].split_no == 2 && G.files[1].split_count == 3);
    GgufTensor *t = gguf_find(&G, "blk.0.ffn_down_exps.weight"); CHECK(t && t->file == 2 && t->off == G.files[2].data_off + t->rel_off);
    t = gguf_find(&G, "token_embd.weight"); CHECK(t && t->file == 0);
    CHECK(gguf_kv_str(&G, "general.architecture") && !strcmp(gguf_kv_str(&G, "general.architecture"), "demo"));
    int arch_copies = 0; for (int64_t i = 0; i < G.nkv; i++) if (!strcmp(gguf_str(&G, G.kv[i].key), "general.architecture")) arch_copies++;
    CHECK(arch_copies == 1);                                   /* part 1's repeated key was dropped */
    gguf_close(&G);
    /* open by any part name -> same set */
    rc = gguf_open_set(&G, paths[1], NULL); CHECK(rc == 0 && G.nfiles == 3 && G.nt == 6); if (!rc) gguf_close(&G);
    /* parts on another "drive": move part 3 away and point COLI_MODEL_DIRS-style search at it */
    const char *dir2 = P("split_drive2"); mkd(dir2);
    char moved[1100]; snprintf(moved, sizeof moved, "%s/demo-00003-of-00003.gguf", dir2);
    CHECK(rename(paths[2], moved) == 0);
    rc = gguf_open_set(&G, paths[0], NULL); CHECK(rc < 0); CHECK_ERR(&G, "part 3 of 3");
    rc = gguf_open_set(&G, paths[0], dir2); CHECK(rc == 0 && G.nfiles == 3); if (!rc) { CHECK(!strcmp(G.files[2].path, moved)); gguf_close(&G); }
    CHECK(rename(moved, paths[2]) == 0); rmdir(dir2);
    /* a second model in the same directory -> ambiguous */
    write_demo(P("split/other.gguf"), 32);
    rc = gguf_open_set(&G, dir, NULL); CHECK(rc < 0); CHECK_ERR(&G, "several GGUF models");
    unlink(P("split/other.gguf"));
    /* wrong split.no in part 2 */
    { Buf B = slurp(paths[1]); /* find "split.no" value: key then type u32 then u16 value */
      uint8_t *p = find_bytes(B.b, B.n, "split.no", 8); CHECK(p != NULL);
      if (p) { uint16_t bad = 5; memcpy(p + 8 + 4, &bad, 2); write_file(paths[1], &B); }
      rc = gguf_open_set(&G, dir, NULL); CHECK(rc < 0); CHECK_ERR(&G, "split.no=5");
      if (p) { uint16_t good = 1; memcpy(p + 8 + 4, &good, 2); write_file(paths[1], &B); }
      bfree(&B); }
    /* wrong split.tensors.count in part 1 */
    { Buf B = slurp(paths[0]);
      uint8_t *p = find_bytes(B.b, B.n, "split.tensors.count", 19); CHECK(p != NULL);
      if (p) { int32_t bad = 7; memcpy(p + 19 + 4, &bad, 4); write_file(paths[0], &B); }
      rc = gguf_open_set(&G, dir, NULL); CHECK(rc < 0); CHECK_ERR(&G, "split.tensors.count=7");
      if (p) { int32_t good = 6; memcpy(p + 19 + 4, &good, 4); write_file(paths[0], &B); }
      bfree(&B); }
    for (int i = 0; i < 3; i++) unlink(paths[i]);
    rmdir(dir);
}

/* one malformed file per rule */
static void expect_fail(const char *label, const Buf *F, const char *needle) {
    const char *path = P("bad.gguf"); write_file(path, F);
    GgufSet G; int rc = gguf_open_set(&G, path, NULL);
    if (rc == 0) { printf("FAIL %s: accepted\n", label); fails++; gguf_close(&G); return; }
    if (!strstr(G.err, needle)) { printf("FAIL %s: error '%s' lacks '%s'\n", label, G.err, needle); fails++; }
    unlink(path);
}
static void test_malformed(void) {
    Buf K = {0}, F = {0}; Tensors T;
    int64_t ne2[4] = {256, 3, 2}, ne1[4] = {4, 4};
#define FRESH() do { bfree(&K); bfree(&F); tinit(&T, 32); } while (0)
    tinit(&T, 32);
    /* magic */
    FRESH(); int n = demo_kv(&K, 32, 0, 0, 0, 0); demo_tensors(&T); assemble(&F, 0x58554747u, 3, (uint64_t)T.n, (uint64_t)n, &K, &T); tfree(&T); expect_fail("magic", &F, "not a GGUF");
    /* version */
    FRESH(); n = demo_kv(&K, 32, 0, 0, 0, 0); demo_tensors(&T); assemble(&F, GGUF_MAGIC, 2, (uint64_t)T.n, (uint64_t)n, &K, &T); tfree(&T); expect_fail("version", &F, "version 2");
    /* absurd counts */
    FRESH(); n = demo_kv(&K, 32, 0, 0, 0, 0); demo_tensors(&T); assemble(&F, GGUF_MAGIC, 3, 1ull << 40, (uint64_t)n, &K, &T); tfree(&T); expect_fail("n_tensors", &F, "limit");
    FRESH(); n = demo_kv(&K, 32, 0, 0, 0, 0); demo_tensors(&T); assemble(&F, GGUF_MAGIC, 3, (uint64_t)T.n, 1ull << 40, &K, &T); tfree(&T); expect_fail("n_kv", &F, "limit");
    /* declared more kv than present -> runs into the tensor table / EOF */
    FRESH(); n = demo_kv(&K, 32, 0, 0, 0, 0); demo_tensors(&T); assemble(&F, GGUF_MAGIC, 3, (uint64_t)T.n, (uint64_t)n + 50, &K, &T); tfree(&T); expect_fail("kv overrun", &F, ":");
    /* key too long */
    FRESH(); { char *huge = malloc(GGUF_MAX_STR + 2); memset(huge, 'k', GGUF_MAX_STR + 1); bstrn(&K, huge, GGUF_MAX_STR + 1); bu32(&K, GGUF_T_U8); bu8(&K, 1); free(huge); }
    demo_tensors(&T); assemble(&F, GGUF_MAGIC, 3, (uint64_t)T.n, 1, &K, &T); tfree(&T); expect_fail("key length", &F, "limit");
    /* unknown value type */
    FRESH(); bstr(&K, "x"); bu32(&K, 42); bu8(&K, 0); demo_tensors(&T); assemble(&F, GGUF_MAGIC, 3, (uint64_t)T.n, 1, &K, &T); tfree(&T); expect_fail("value type", &F, "unknown value type");
    /* unknown array element type */
    FRESH(); bstr(&K, "x"); bu32(&K, GGUF_T_ARR); bu32(&K, 42); bu64(&K, 0); demo_tensors(&T); assemble(&F, GGUF_MAGIC, 3, (uint64_t)T.n, 1, &K, &T); tfree(&T); expect_fail("array type", &F, "unknown element type");
    /* array too long */
    FRESH(); bstr(&K, "x"); bu32(&K, GGUF_T_ARR); bu32(&K, GGUF_T_U8); bu64(&K, (uint64_t)GGUF_MAX_ARR + 1); demo_tensors(&T); assemble(&F, GGUF_MAGIC, 3, (uint64_t)T.n, 1, &K, &T); tfree(&T); expect_fail("array length", &F, "limit");
    /* nested too deep: [[[1]]] */
    FRESH(); bstr(&K, "x"); bu32(&K, GGUF_T_ARR); bu32(&K, GGUF_T_ARR); bu64(&K, 1); bu32(&K, GGUF_T_ARR); bu64(&K, 1); bu32(&K, GGUF_T_U8); bu64(&K, 1); bu8(&K, 1);
    demo_tensors(&T); assemble(&F, GGUF_MAGIC, 3, (uint64_t)T.n, 1, &K, &T); tfree(&T); expect_fail("nesting", &F, "nests deeper");
    /* NUL inside a string */
    FRESH(); bstrn(&K, "ab\0c", 4); bu32(&K, GGUF_T_U8); bu8(&K, 1); demo_tensors(&T); assemble(&F, GGUF_MAGIC, 3, (uint64_t)T.n, 1, &K, &T); tfree(&T); expect_fail("nul", &F, "NUL");
    /* alignment not a power of two / not an integer / zero */
    FRESH(); kv_u32(&K, "general.alignment", 48); demo_tensors(&T); assemble(&F, GGUF_MAGIC, 3, (uint64_t)T.n, 1, &K, &T); tfree(&T); expect_fail("align 48", &F, "power of two");
    FRESH(); kv_u32(&K, "general.alignment", 0); demo_tensors(&T); assemble(&F, GGUF_MAGIC, 3, (uint64_t)T.n, 1, &K, &T); tfree(&T); expect_fail("align 0", &F, "power of two");
    FRESH(); kv_str(&K, "general.alignment", "32"); demo_tensors(&T); assemble(&F, GGUF_MAGIC, 3, (uint64_t)T.n, 1, &K, &T); tfree(&T); expect_fail("align str", &F, "not an integer");
    /* tensor: 5 dims */
    FRESH(); n = demo_kv(&K, 32, 0, 0, 0, 0); bstr(&T.tab, "t"); bu32(&T.tab, 5); for (int i = 0; i < 5; i++) bu64(&T.tab, 2); bu32(&T.tab, GGML_TYPE_F32); bu64(&T.tab, 0); T.n = 1;
    assemble(&F, GGUF_MAGIC, 3, 1, (uint64_t)n, &K, &T); tfree(&T); expect_fail("dims", &F, "dimensions");
    /* tensor: zero dimension */
    FRESH(); n = demo_kv(&K, 32, 0, 0, 0, 0); { int64_t z[4] = {0, 4}; tadd(&T, "t", GGML_TYPE_F32, 2, z, 0, 0); }
    assemble(&F, GGUF_MAGIC, 3, 1, (uint64_t)n, &K, &T); tfree(&T); expect_fail("zero dim", &F, "dimension");
    /* tensor: ne0 not a multiple of the block */
    FRESH(); n = demo_kv(&K, 32, 0, 0, 0, 0); { int64_t z[4] = {100, 2}; tadd(&T, "t", GGML_TYPE_Q4_K, 2, z, 64, -1); }
    assemble(&F, GGUF_MAGIC, 3, 1, (uint64_t)n, &K, &T); tfree(&T); expect_fail("block", &F, "not a multiple");
    /* tensor: shape overflow */
    FRESH(); n = demo_kv(&K, 32, 0, 0, 0, 0); { int64_t z[4] = {256, 1ll << 40, 1ll << 40}; tadd(&T, "t", GGML_TYPE_Q4_K, 3, z, 0, 0); }
    assemble(&F, GGUF_MAGIC, 3, 1, (uint64_t)n, &K, &T); tfree(&T); expect_fail("overflow", &F, "overflows");
    /* tensor: misaligned offset */
    FRESH(); n = demo_kv(&K, 32, 0, 0, 0, 0); tadd(&T, "a", GGML_TYPE_F32, 2, ne1, -1, -1); tadd(&T, "b", GGML_TYPE_F32, 2, ne1, -1, 16);
    assemble(&F, GGUF_MAGIC, 3, 2, (uint64_t)n, &K, &T); tfree(&T); expect_fail("misaligned", &F, "aligned");
    /* tensor: past EOF */
    FRESH(); n = demo_kv(&K, 32, 0, 0, 0, 0); tadd(&T, "a", GGML_TYPE_Q4_K, 3, ne2, -1, 1 << 20);
    assemble(&F, GGUF_MAGIC, 3, 1, (uint64_t)n, &K, &T); tfree(&T); expect_fail("eof", &F, "runs past");
    /* tensor: known type but data truncated by one byte */
    FRESH(); n = demo_kv(&K, 32, 0, 0, 0, 0); tadd(&T, "a", GGML_TYPE_Q4_K, 3, ne2, -1, -1);
    assemble(&F, GGUF_MAGIC, 3, 1, (uint64_t)n, &K, &T); tfree(&T); F.n -= 1; expect_fail("short data", &F, "runs past");
    /* duplicate tensor name */
    FRESH(); n = demo_kv(&K, 32, 0, 0, 0, 0); tadd(&T, "a", GGML_TYPE_F32, 2, ne1, -1, -1); tadd(&T, "a", GGML_TYPE_F32, 2, ne1, -1, -1);
    assemble(&F, GGUF_MAGIC, 3, 2, (uint64_t)n, &K, &T); tfree(&T); expect_fail("dup", &F, "duplicate");
    /* truncated in the middle of the metadata */
    FRESH(); n = demo_kv(&K, 32, 0, 0, 0, 0); demo_tensors(&T); assemble(&F, GGUF_MAGIC, 3, (uint64_t)T.n, (uint64_t)n, &K, &T); tfree(&T); F.n = 60; expect_fail("truncated", &F, "end of the file");
    /* empty file / tiny file */
    FRESH(); expect_fail("empty", &F, "end of the file");
    /* not a file at all */
    { GgufSet G; int rc = gguf_open_set(&G, P("does_not_exist.gguf"), NULL); CHECK(rc < 0); CHECK_ERR(&G, "does_not_exist"); }
    { const char *ed = P("emptydir"); mkd(ed); GgufSet G; int rc = gguf_open_set(&G, ed, NULL); CHECK(rc < 0); CHECK_ERR(&G, "no .gguf"); rmdir(ed); }
    bfree(&K); bfree(&F);
#undef FRESH
}

/* every single-bit flip of every metadata byte must be accepted or refused — never crash */
static void test_mutations(void) {
    const char *path = P("mut.gguf"); write_demo(path, 32);
    Buf B = slurp(path);
    GgufSet G; int rc = gguf_open_set(&G, path, NULL); CHECK(rc == 0);
    int64_t meta = G.files[0].data_off; gguf_close(&G);
    int accepted = 0, refused = 0;
    for (int64_t i = 0; i < meta; i++) for (int bit = 0; bit < 8; bit += 7) {   /* bit 0 and bit 7 */
        B.b[i] ^= (uint8_t)(1u << bit);
        write_file(path, &B);
        rc = gguf_open_set(&G, path, NULL);
        if (rc == 0) { accepted++; CHECK(G.nt >= 0); gguf_close(&G); }
        else { refused++; if (!G.err[0]) { printf("FAIL: silent refusal at byte %lld bit %d\n", (long long)i, bit); fails++; } }
        B.b[i] ^= (uint8_t)(1u << bit);
    }
    printf("  mutation sweep: %lld metadata bytes, %d accepted, %d refused\n", (long long)meta, accepted, refused);
    CHECK(refused > 0 && accepted > 0);
    bfree(&B); unlink(path);
}

static void test_mirror(void) {
    const char *path = P("mir.gguf"); write_demo(path, 32);
    const char *mdir = P("mirror"); mkd(mdir);
    char mp[1100]; snprintf(mp, sizeof mp, "%s/mir.gguf", mdir);
    GgufSet G; CHECK(gguf_open_set(&G, path, NULL) == 0);
    CHECK(gguf_mirror_init(&G, mdir) == 0 && G.files[0].mfd == -1);          /* absent: stays on the primary */
    copy_file(path, mp);
    CHECK(gguf_mirror_init(&G, mdir) == 1 && G.files[0].mfd >= 0);
    CHECK(gguf_fd_rep(&G, 0, 1) == G.files[0].mfd && gguf_fd_rep(&G, 0, 0) == G.files[0].fd);
    /* a payload byte may differ (we only compare the metadata region) — but a header byte may not */
    { Buf B = slurp(mp); B.b[G.files[0].data_off - 1] ^= 1; write_file(mp, &B); bfree(&B); }
    CHECK(gguf_mirror_init(&G, mdir) == 0);
    { Buf B = slurp(path); bu8(&B, 0); write_file(mp, &B); bfree(&B); }      /* size differs */
    CHECK(gguf_mirror_init(&G, mdir) == 0);
    gguf_close(&G); unlink(mp); rmdir(mdir); unlink(path);
}

int main(int argc, char **argv) {
    if (argc > 1) {                                   /* CLI: dump for the Python cross-check */
        GgufSet G;
        if (gguf_open_set(&G, argv[1], getenv("COLI_MODEL_DIRS"))) { fprintf(stderr, "gguf: %s\n", G.err); return 1; }
        gguf_dump(&G, stdout, 1); gguf_close(&G); return 0;
    }
    mkd(TMP);
    test_valid(32);
    test_valid(64);
    test_split();
    test_malformed();
    test_mutations();
    test_mirror();
    unlink(P("demo32.gguf")); unlink(P("demo64.gguf")); rmdir(TMP);
    if (fails) { printf("gguf tests: %d FAILED\n", fails); return 1; }
    puts("gguf reader tests: ok");
    return 0;
}
