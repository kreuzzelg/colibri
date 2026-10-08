/* qwen38_cuda_chain.h -- Qwen3.8 Flash Next's layers as a dense chain on the CUDA device
 * (cuda_chain.h). Included once by qwen38_core.h in a COLI_CUDA build, after the CPU
 * forward it stands in for; COLI_CUDA_CHAIN=1 asks for it (q38cc_setup decides at
 * startup, after the tier placed the trunk). The CUDA twin of qwen38_chain.h, with the
 * same frames and the same state contract, on qwen36_cuda_chain.h's table:
 *   - four hyper-connection streams (hyper, S x 4H) stay on the device; every block
 *     reads them through its gated residual (per-stream norm, the low-rank down/up
 *     pair, the stream mix, the inject weights) and writes back hyper += inject x block;
 *   - the PLE layer: the n-gram table rows stay on the host (disk reads, in token
 *     order, with the n-gram history); only their rows go up, then key and value
 *     projections, the gate and the dilated convolution with its ring on the device;
 *   - QSA attention: the K/V/index-key rows on the device (the host's copies kept
 *     canonical), each block's pooled key computed once on the device when a step
 *     completes the block, and the indexer's top-k selection per query row on the
 *     device, feeding the attention's selection list;
 *   - the MTP head stays on the CPU: the chain hands it every row's four streams when
 *     the model has one; a verify forward (S = 2 to Q38_SPEC_ROWS) runs row-wise (the
 *     warp GEMV computes each row on its own: a decode step's bits) and snapshots the
 *     DeltaNet state, the conv rings and the PLE ring after each of its rows but the
 *     last on the device, one slot per row, so a draft rejected after row r rolls back
 *     by swapping slot r's device buffers in (q38cc_rollback).
 * Per layer the host gets the MoE input rows and the router logits, and the K/V and
 * index-key rows of an attention layer; it sends the routed sum back. The shared
 * expert runs on the device (frame A2) while the host computes the routed experts
 * through the fp8 tier.
 *
 * The matrices: the tier's resident int8 copies where the placer put them (w->gpu);
 * every other matrix of a layer, the final mixer and lm_head the chain uploads itself
 * (w->cc): the trunk's int8 rows when the CPU holds them (the same bits), else the
 * BF16/F32 rows as f32. So the chain runs whatever the placement did; what it costs in
 * VRAM beyond the tier's copies are the small matrices the placer never sees (the
 * gated residuals' low-rank pairs, the DeltaNet b/a rows, the PLE projections).
 *
 * State ownership as in qwen36's: the host's KV/index caches canonical, device mirrors
 * behind a watermark (kv_valid), pooled block keys behind pk_valid; the DeltaNet
 * state, conv rings and PLE ring on the device while the chain runs (dn_where),
 * brought back before the host reads them (prefix cache, pins) and pushed up after
 * the host writes them (reset, restore). Q38_DN_GPU's per-layer step is not set up
 * beside the chain. A device lost while it holds that state: the engine rebuilds it
 * on the CPU from the prefix record and continues there (q38cc_recover).
 * COLI_CUDA_CHAIN_FAULT=n fakes the loss at the n-th frame. */
#include "cuda_chain.h"

#define Q38CC_HOST 0
#define Q38CC_DEV  1
#define Q38CC_BOTH 2

typedef struct {
    int ok, failed, rows, cap, dev;
    int nl, head;                     /* every layer, and the final mixer and lm_head */
    size_t prm_floats;
    CcBuf *prm;
    size_t *o_an, *o_mn, *o_qn, *o_kn, *o_iqn, *o_ikn, *o_conv, *o_dn;
    size_t o_fn, o_ple, o_pconv;
    ColiCudaTensor **t_sg;            /* the shared expert's gate row, f32 */
    CcBuf **rec, **ring, **kc, **vc, **ik, **pk;
    CcBuf **rec_snap[Q38_SPEC_SNAPS], **ring_snap[Q38_SPEC_SNAPS];   /* a verify's copies, slot r after row r */
    int snap_slots;
    CcBuf *ple_ring;                  /* [ple_nslots][W][SL]: current at ple_slot[0], the copy after row r at ple_slot[1+r] */
    int ple_slot[Q38_SPEC_ROWS], ple_nslots;
    int *kv_valid, *pk_valid, *attn_ord, n_attn;
    int dn_where, host_zero, snap_valid;
    CcBuf *hyper, *hn, *low, *mix, *mixed, *inj_a, *inj_m, *blk, *q, *k, *v, *ip, *ctx, *qsc, *sel;
    CcBuf *qkv, *z, *ab, *cv, *dny, *lg, *gs, *us, *hs, *ds, *sgd;
    CcBuf *emb, *keys, *val, *gated, *normv;
    CcBuf *mixd, *lgd, *kvd, *outd, *hypd, *find, *routed, *cs, *pcs;
    CcBuf *gseg, *nseg;               /* a deep verify's PLE rows past its first copy (q38cc_ple) */
    float *host_routed, *host_emb;
    unsigned long long forwards, frames, fault_at, dec_n;
    double host_ms, wait_ms, dec_wait_ms, dec_host_ms;
} Q38CChain;
static Q38CChain *q38cc_of(Model *m) { return (Q38CChain *)m->cchain; }
static int g_cuda_chain = 0;     /* COLI_CUDA_CHAIN asked for it and the chain is up */

static int q38cc_env_on(void) { const char *e = getenv("COLI_CUDA_CHAIN"); return e && *e == '1'; }
static int q38cc_rows_env(void) {
    const char *e = getenv("COLI_CUDA_CHAIN_ROWS");
    int r = e && *e ? atoi(e) : 256;
    return r < 1 ? 1 : r > 8192 ? 8192 : r;
}
static void q38cc_fatal(const char *what) {
    fprintf(stderr, "[chain] qwen38: %s -- stopping (COLI_CUDA_CHAIN=0 keeps the state on the CPU)\n", what);
    exit(1);
}
static int q38cc_geometry_ok(const Cfg *c) {
    int attn = 0, dn = 0, kd = c->dn_kdim;
    int kd_ok = kd == 4 || kd == 8 || kd == 16 || kd == 32 || kd == 64 || kd == 128 || kd == 256;
    for (int i = 0; i < c->layers; i++) { if (c->is_attn[i]) attn = 1; else dn = 1; }
    if (attn && (c->head_dim > 256 || c->q_heads % c->kv_heads || (c->rotary_dim & 1) || c->rotary_dim > c->head_dim ||
                 c->idx_kheads != 1 || c->idx_dim > 256 || c->rotary_dim > c->idx_dim || c->idx_ratio < 1)) return 0;
    if (dn && (c->dn_vdim > 128 || !kd_ok || c->dn_convk < 2 || c->dn_convk > 9 || c->dn_vheads % c->dn_kheads)) return 0;
    if (c->ple_layer >= 0 && c->ple_layer < c->layers && (c->ple_convk - 1) * c->ngram_size > 32) return 0;
    return c->hc_count > 0 && c->hc_width == c->hc_count * c->hidden;
}
static size_t q38cc_ple_cells(const Cfg *c) { return (size_t)c->hc_width * (c->ple_convk - 1) * c->ngram_size; }
static int q38cc_selrow(const Cfg *c) { return 1 + c->idx_budget + c->idx_ratio - 1; }
static size_t q38cc_kv_stride(Q38CChain *ch, const Cfg *c) {   /* per attention layer in kvd: K, V, IK rows */
    return (size_t)ch->rows * (2 * (size_t)c->kv_heads * c->head_dim + c->idx_dim);
}

/* ---- the matrices ------------------------------------------------------------------ */
/* A weight's device copy: the tier's (w->gpu) or the chain's own (w->cc, made here from
 * the trunk's int8 rows, else the BF16/F32 rows as f32). NULL: nothing to upload, or
 * the device refused. */
static ColiCudaTensor *q38cc_t(Q38CChain *ch, Q38Weight *w) {
    if (!w) return NULL;
    if (w->gpu) {
        int d = -1;
        ColiCudaTensor *t = qt_dense_tensor(w->gpu - 1, &d);
        if (t && d == ch->dev) return t;
        if (t) return NULL;   /* on another card: the chain does not span cards */
    }
    if (w->cc) return (ColiCudaTensor *)w->cc;
    ColiCudaTensor *t = NULL;
    if (w->q8 && w->q8sc) {
        if (!coli_cuda_tensor_upload(&t, w->q8, w->q8sc, 1, w->cols, w->rows, ch->dev)) return NULL;
    } else if (w->data && (w->kind == Q38_WEIGHT_F32 || w->kind == Q38_WEIGHT_BF16)) {
        size_t n = (size_t)w->rows * w->cols;
        float *f = NULL, *ones = falloc(w->rows);
        for (int o = 0; o < w->rows; o++) ones[o] = 1.f;
        if (w->kind == Q38_WEIGHT_BF16) {
            f = falloc((int64_t)n);
            const uint16_t *src = (const uint16_t *)w->data;
            for (size_t i = 0; i < n; i++) f[i] = bf16_to_f32(src[i]);
        }
        int ok = coli_cuda_tensor_upload(&t, f ? f : (const float *)w->data, ones, 0, w->cols, w->rows, ch->dev);
        free(f); free(ones);
        if (!ok) return NULL;
    } else return NULL;
    w->cc = t;
    return t;
}
/* f32 rows [O x I] as a resident fmt 0 tensor (unit scales: the kernel reads none) */
static ColiCudaTensor *q38cc_f32_rows(int dev, const float *w, int I, int O) {
    ColiCudaTensor *t = NULL;
    float *ones = falloc(O);
    for (int o = 0; o < O; o++) ones[o] = 1.f;
    int ok = coli_cuda_tensor_upload(&t, w, ones, 0, I, O, dev);
    free(ones);
    return ok ? t : NULL;
}
#define Q38CC_MATS 24
static int q38cc_mats(Model *m, int i, Q38Weight **w, const char **nm) {
    Cfg *c = &m->c; Layer *l = &m->L[i]; int n = 0;
    Q38Weight *all[] = {&l->attn_gr.down, &l->attn_gr.up, &l->attn_gr.inject, &l->mlp_gr.down, &l->mlp_gr.up,
                        &l->mlp_gr.inject, &l->router, &l->sh_g, &l->sh_u, &l->sh_d};
    static const char *names[] = {"hcad", "hcau", "hcai", "hcmd", "hcmu", "hcmi", "router", "shg", "shu", "shd"};
    for (size_t k = 0; k < sizeof all / sizeof *all; k++) { nm[n] = names[k]; w[n++] = all[k]; }
    if (c->is_attn[i]) { nm[n] = "attnq"; w[n++] = &l->q; nm[n] = "attnk"; w[n++] = &l->k; nm[n] = "attnv"; w[n++] = &l->v;
                         nm[n] = "attno"; w[n++] = &l->o; nm[n] = "qsaidx"; w[n++] = &l->idx_qk; }
    else { nm[n] = "dnqkv"; w[n++] = &l->dn_qkv; nm[n] = "dnz"; w[n++] = &l->dn_z; nm[n] = "dnb"; w[n++] = &l->dn_b;
           nm[n] = "dna"; w[n++] = &l->dn_a; nm[n] = "dnout"; w[n++] = &l->dn_out; }
    if (i == c->ple_layer) { nm[n] = "plekey"; w[n++] = &l->ple_key; nm[n] = "plevalue"; w[n++] = &l->ple_value; }
    return n;
}
static void q38cc_free_own(Q38CChain *ch, Model *m) {
    Cfg *c = &m->c; int L = c->layers;
    for (int i = 0; i < L; i++) {
        Q38Weight *w[Q38CC_MATS]; const char *nm[Q38CC_MATS]; int n = q38cc_mats(m, i, w, nm);
        for (int k = 0; k < n; k++) if (w[k]->cc) { coli_cuda_tensor_free((ColiCudaTensor *)w[k]->cc); w[k]->cc = NULL; }
        if (ch->t_sg && ch->t_sg[i]) { coli_cuda_tensor_free(ch->t_sg[i]); ch->t_sg[i] = NULL; }
    }
    Q38Weight *tail[] = {&m->final_gr.down, &m->final_gr.up, &m->lm_head};
    for (size_t k = 0; k < 3; k++) if (tail[k]->cc) { coli_cuda_tensor_free((ColiCudaTensor *)tail[k]->cc); tail[k]->cc = NULL; }
    CcBuf ***arrs[] = {&ch->rec, &ch->ring, &ch->kc, &ch->vc, &ch->ik, &ch->pk};
    for (size_t a = 0; a < sizeof arrs / sizeof *arrs; a++) if (*arrs[a]) for (int i = 0; i < L; i++) { cc_free((*arrs[a])[i]); (*arrs[a])[i] = NULL; }
    for (int sl = 0; sl < Q38_SPEC_SNAPS; sl++) {
        if (ch->rec_snap[sl]) for (int i = 0; i < L; i++) { cc_free(ch->rec_snap[sl][i]); ch->rec_snap[sl][i] = NULL; }
        if (ch->ring_snap[sl]) for (int i = 0; i < L; i++) { cc_free(ch->ring_snap[sl][i]); ch->ring_snap[sl][i] = NULL; }
    }
    cc_free(ch->ple_ring); ch->ple_ring = NULL;
    cc_free(ch->prm); ch->prm = NULL;
}

/* The parameters the kernels read (norm weights, the DeltaNet constants, the PLE's), in
 * one buffer; the attention layers' order (n_attn). */
static int q38cc_arena(Q38CChain *ch, Model *m) {
    Cfg *c = &m->c; int L = c->layers, W = c->hc_width;
    int VH = c->dn_vheads, CD = c->dn_conv_dim, CK = c->dn_convk;
    int ple = c->ple_layer >= 0 && c->ple_layer < L;
    size_t n = 0;
    ch->n_attn = 0;
    for (int i = 0; i < L; i++) {
        ch->o_an[i] = n; n += W; ch->o_mn[i] = n; n += W;
        if (c->is_attn[i]) {
            ch->attn_ord[i] = ch->n_attn++;
            ch->o_qn[i] = n; n += c->head_dim; ch->o_kn[i] = n; n += c->head_dim;
            ch->o_iqn[i] = n; n += c->idx_dim; ch->o_ikn[i] = n; n += c->idx_dim;
        } else {
            ch->o_conv[i] = n; n += (size_t)CD * CK;
            ch->o_dn[i] = n; n += 2 * (size_t)VH + c->dn_vdim;
        }
    }
    ch->o_fn = n; n += W;
    if (ple) { ch->o_ple = n; n += 3 * (size_t)W; ch->o_pconv = n; n += (size_t)W * c->ple_convk; }
    float *a = calloc(n, sizeof(float));
    if (!a) return 0;
    for (int i = 0; i < L; i++) {
        Layer *l = &m->L[i];
        memcpy(a + ch->o_an[i], l->attn_gr.norm, W * sizeof(float));
        memcpy(a + ch->o_mn[i], l->mlp_gr.norm, W * sizeof(float));
        if (c->is_attn[i]) {
            memcpy(a + ch->o_qn[i], l->qn, c->head_dim * sizeof(float)); memcpy(a + ch->o_kn[i], l->kn, c->head_dim * sizeof(float));
            memcpy(a + ch->o_iqn[i], l->idx_qn, c->idx_dim * sizeof(float)); memcpy(a + ch->o_ikn[i], l->idx_kn, c->idx_dim * sizeof(float));
        } else {
            memcpy(a + ch->o_conv[i], l->dn_conv, (size_t)CD * CK * sizeof(float));
            memcpy(a + ch->o_dn[i], l->dn_alog, VH * sizeof(float));
            memcpy(a + ch->o_dn[i] + VH, l->dn_dtbias, VH * sizeof(float));
            memcpy(a + ch->o_dn[i] + 2 * VH, l->dn_norm, c->dn_vdim * sizeof(float));
        }
    }
    memcpy(a + ch->o_fn, m->final_gr.norm, W * sizeof(float));
    if (ple) {
        Layer *pl = &m->L[c->ple_layer];
        memcpy(a + ch->o_ple, pl->ple_norm_key, W * sizeof(float));
        memcpy(a + ch->o_ple + W, pl->ple_norm_query, W * sizeof(float));
        memcpy(a + ch->o_ple + 2 * (size_t)W, pl->ple_norm_conv, W * sizeof(float));
        memcpy(a + ch->o_pconv, pl->ple_conv, (size_t)W * c->ple_convk * sizeof(float));
    }
    ch->prm = cc_buf(n * sizeof(float), CC_DEV);
    int ok = ch->prm && cc_begin() && cc_write(ch->prm, 0, a, n * sizeof(float)) && cc_submit(1);
    free(a);
    if (!ok) { cc_free(ch->prm); ch->prm = NULL; return 0; }
    ch->prm_floats = n;
    return 1;
}

/* The chain on the device, at startup after the tier placed the trunk and the CPU
 * quantized its int8 rows: every layer's matrices on one device (the tier's or the
 * chain's own), the gate rows, the state, the parameters. NULL = the chain cannot run
 * (the message says why). */
static Q38CChain *q38cc_setup(Model *m, int mux_slots) {
    Cfg *c = &m->c; int L = c->layers;
    if (m->cchain) return ((Q38CChain *)m->cchain)->ok ? (Q38CChain *)m->cchain : NULL;
    if (!cc_available()) { fprintf(stderr, "[chain] qwen38: the CUDA backend has no chain (an older DLL); the per-matrix path\n"); return NULL; }
    if (!qt_ready()) { fprintf(stderr, "[chain] qwen38: no CUDA expert tier; the chain stays off\n"); return NULL; }
    if (mux_slots > 1) { fprintf(stderr, "[chain] qwen38: KV_SLOTS > 1 keeps one state per conversation on the host; the chain stays off\n"); return NULL; }
    if (m->range_begin != 0 || m->range_end != L || !m->lm_head.rows || !q38cc_geometry_ok(c)) {
        fprintf(stderr, "[chain] qwen38: a model or geometry its kernels do not take; the per-matrix path\n");
        return NULL;
    }
    Q38CChain *ch = (Q38CChain *)calloc(1, sizeof *ch);
    if (!ch) return NULL;
    m->cchain = ch;
    ch->dev = -1;
#define Q38CC_ARR(f, t) if (!(ch->f = (t *)calloc((size_t)L, sizeof(t)))) return NULL;
    Q38CC_ARR(o_an, size_t); Q38CC_ARR(o_mn, size_t); Q38CC_ARR(o_qn, size_t); Q38CC_ARR(o_kn, size_t);
    Q38CC_ARR(o_iqn, size_t); Q38CC_ARR(o_ikn, size_t); Q38CC_ARR(o_conv, size_t); Q38CC_ARR(o_dn, size_t);
    Q38CC_ARR(t_sg, ColiCudaTensor *); Q38CC_ARR(rec, CcBuf *); Q38CC_ARR(ring, CcBuf *); Q38CC_ARR(rec_snap[0], CcBuf *);
    Q38CC_ARR(ring_snap[0], CcBuf *); Q38CC_ARR(kc, CcBuf *); Q38CC_ARR(vc, CcBuf *); Q38CC_ARR(ik, CcBuf *); Q38CC_ARR(pk, CcBuf *);
    Q38CC_ARR(kv_valid, int); Q38CC_ARR(pk_valid, int); Q38CC_ARR(attn_ord, int);
#undef Q38CC_ARR
    ch->snap_slots = 1;
    /* the device: the card the tier placed the trunk on (the first placed matrix names it) */
    for (int i = 0; i < L && ch->dev < 0; i++) {
        Q38Weight *w[Q38CC_MATS]; const char *nm[Q38CC_MATS]; int n = q38cc_mats(m, i, w, nm);
        for (int k = 0; k < n && ch->dev < 0; k++) if (w[k]->gpu) { int d = -1; if (qt_dense_tensor(w[k]->gpu - 1, &d)) ch->dev = d; }
    }
    if (ch->dev < 0 && m->lm_head.gpu) { int d = -1; if (qt_dense_tensor(m->lm_head.gpu - 1, &d)) ch->dev = d; }
    if (ch->dev < 0) { fprintf(stderr, "[chain] qwen38: the tier placed no trunk matrix (Q38_TRUNK_GPU); the chain stays off\n"); return NULL; }
    if (!cc_init(ch->dev)) { fprintf(stderr, "[chain] qwen38: the chain's context on device %d did not come up\n", ch->dev); return NULL; }
    size_t own = 0, tier = 0;
    for (int i = 0; i < L; i++) {
        Q38Weight *w[Q38CC_MATS]; const char *nm[Q38CC_MATS]; int n = q38cc_mats(m, i, w, nm);
        for (int k = 0; k < n; k++) {
            if (!q38cc_t(ch, w[k])) {
                fprintf(stderr, "[chain] qwen38: %s of layer %d did not reach device %d; the chain stays off\n", nm[k], i, ch->dev);
                q38cc_free_own(ch, m);
                return NULL;
            }
            if (!w[k]->cc) tier += (size_t)w[k]->rows * w[k]->cols;
            else own += (size_t)w[k]->rows * w[k]->cols * (w[k]->q8 ? 1 : 4);
        }
        if (!(ch->t_sg[i] = q38cc_f32_rows(ch->dev, m->L[i].sh_gate, c->hidden, 1))) {
            fprintf(stderr, "[chain] qwen38: the shared expert's gate row of layer %d did not reach the device; the chain stays off\n", i);
            q38cc_free_own(ch, m);
            return NULL;
        }
        own += (size_t)c->hidden * 4;
        if (!c->is_attn[i]) {
            size_t nr = (size_t)c->dn_vheads * c->dn_kdim * c->dn_vdim, nc = (size_t)c->dn_conv_dim * (c->dn_convk - 1);
            if (!(ch->rec[i] = cc_buf(nr * 4, CC_DEV)) || !(ch->ring[i] = cc_buf(nc * 4, CC_DEV)) ||
                !(ch->rec_snap[0][i] = cc_buf(nr * 4, CC_DEV)) || !(ch->ring_snap[0][i] = cc_buf(nc * 4, CC_DEV))) {
                fprintf(stderr, "[chain] qwen38: device memory for layer %d's DeltaNet state refused; the chain stays off\n", i);
                q38cc_free_own(ch, m);
                return NULL;
            }
        }
        if (i == c->ple_layer) {
            if (!(ch->ple_ring = cc_buf(2 * q38cc_ple_cells(c) * 4, CC_DEV))) { q38cc_free_own(ch, m); return NULL; }
            ch->ple_nslots = 2; ch->ple_slot[0] = 0; ch->ple_slot[1] = 1;
        }
    }
    Q38Weight *tail[] = {&m->final_gr.down, &m->final_gr.up, &m->lm_head};
    static const char *tnm[] = {"the final mixer's down", "the final mixer's up", "lm_head"};
    for (int k = 0; k < 3; k++) if (!q38cc_t(ch, tail[k])) {
        fprintf(stderr, "[chain] qwen38: %s did not reach device %d; the chain stays off\n", tnm[k], ch->dev);
        q38cc_free_own(ch, m);
        return NULL;
    }
    if (!q38cc_arena(ch, m)) {
        fprintf(stderr, "[chain] qwen38: device memory for the chain's parameters refused; the chain stays off\n");
        q38cc_free_own(ch, m);
        return NULL;
    }
    ch->nl = L; ch->head = 1;
    ch->dn_where = Q38CC_HOST;
    { const char *e = getenv("COLI_CUDA_CHAIN_FAULT"); ch->fault_at = e && *e ? strtoull(e, NULL, 10) : 0; }
    ch->ok = 1;
    fprintf(stderr, "[chain] qwen38: %d layers on CUDA device %d (%d QSA), the final mixer and lm_head too; %.2f GiB of the tier's "
                    "trunk, %.1f MiB uploaded by the chain, %.1f MiB of parameters\n",
            L, ch->dev, ch->n_attn, tier / 1073741824.0, own / 1048576.0, ch->prm_floats * 4 / 1048576.0);
    return ch;
}

/* the scratch buffers for `rows` rows (kvd's stride is ch->rows') */
static int q38cc_bufs(Q38CChain *ch, Model *m, int rows) {
    Cfg *c = &m->c; int H = c->hidden, W = c->hc_width, C = c->hc_count, R = c->hc_rank;
    int QH = c->q_heads, D = c->head_dim, kvo = c->kv_heads * D, ipw = (c->idx_qheads + 1) * c->idx_dim;
    int V = c->dn_vheads * c->dn_vdim, E = c->experts, SI = c->shared_inter;
    int Ep = c->ngram_heads * c->ngram_head_dim; if (Ep < 1) Ep = 1;
    size_t r = (size_t)rows, F = 4;
    int nbmax = m->kv_cap / (c->idx_ratio > 0 ? c->idx_ratio : 1) + 1;
    int ok = cc_reserve(&ch->hyper, r * W * F, CC_DEV) && cc_reserve(&ch->hn, r * W * F, CC_DEV) && cc_reserve(&ch->low, r * R * F, CC_DEV) &&
             cc_reserve(&ch->mix, r * W * F, CC_DEV) && cc_reserve(&ch->mixed, r * H * F, CC_DEV) && cc_reserve(&ch->inj_a, r * C * F, CC_DEV) &&
             cc_reserve(&ch->inj_m, r * C * F, CC_DEV) && cc_reserve(&ch->blk, r * H * F, CC_DEV) && cc_reserve(&ch->q, r * QH * 2 * D * F, CC_DEV) &&
             cc_reserve(&ch->k, r * kvo * F, CC_DEV) && cc_reserve(&ch->v, r * kvo * F, CC_DEV) && cc_reserve(&ch->ip, r * ipw * F, CC_DEV) &&
             cc_reserve(&ch->ctx, r * QH * D * F, CC_DEV) && cc_reserve(&ch->qsc, r * 2 * (size_t)nbmax * F, CC_DEV) &&
             cc_reserve(&ch->sel, r * q38cc_selrow(c) * F, CC_DEV) && cc_reserve(&ch->qkv, r * c->dn_conv_dim * F, CC_DEV) &&
             cc_reserve(&ch->z, r * V * F, CC_DEV) && cc_reserve(&ch->ab, r * 2 * c->dn_vheads * F, CC_DEV) &&
             cc_reserve(&ch->cv, r * c->dn_conv_dim * F, CC_DEV) && cc_reserve(&ch->dny, r * V * F, CC_DEV) &&
             cc_reserve(&ch->lg, r * E * F, CC_DEV) && cc_reserve(&ch->gs, r * SI * F, CC_DEV) && cc_reserve(&ch->us, r * SI * F, CC_DEV) &&
             cc_reserve(&ch->hs, r * SI * F, CC_DEV) && cc_reserve(&ch->ds, r * H * F, CC_DEV) && cc_reserve(&ch->sgd, r * F, CC_DEV) &&
             cc_reserve(&ch->keys, r * W * F, CC_DEV) && cc_reserve(&ch->val, r * H * F, CC_DEV) &&
             cc_reserve(&ch->gated, r * W * F, CC_DEV) && cc_reserve(&ch->normv, r * W * F, CC_DEV) &&
             cc_reserve(&ch->mixd, r * H * F, CC_DOWN) && cc_reserve(&ch->lgd, r * E * F, CC_DOWN) &&
             cc_reserve(&ch->kvd, (size_t)(ch->n_attn ? ch->n_attn : 1) * q38cc_kv_stride(ch, c) * F, CC_DOWN) &&
             cc_reserve(&ch->outd, 2 * (size_t)c->vocab * F, CC_DOWN) && cc_reserve(&ch->routed, r * H * F, CC_UP) &&
             cc_reserve(&ch->emb, r * Ep * F, CC_UP) && cc_reserve(&ch->cs, r * (c->rotary_dim > 0 ? c->rotary_dim : 2) * F, CC_UP) &&
             cc_reserve(&ch->pcs, (size_t)nbmax * (c->rotary_dim > 0 ? c->rotary_dim : 2) * F, CC_UP);
    return ok;
}
/* the K/V, index-key and pooled-key mirrors at the host's capacity */
static int q38cc_mirror(Q38CChain *ch, Model *m) {
    Cfg *c = &m->c; int kvo = c->kv_heads * c->head_dim;
    int nbmax = m->kv_cap / (c->idx_ratio > 0 ? c->idx_ratio : 1) + 1;
    if (ch->cap != m->kv_cap) {
        for (int i = 0; i < c->layers; i++) {
            if (!c->is_attn[i]) continue;
            cc_free(ch->kc[i]); cc_free(ch->vc[i]); cc_free(ch->ik[i]); cc_free(ch->pk[i]);
            ch->kv_valid[i] = ch->pk_valid[i] = 0;
            ch->kc[i] = cc_buf((size_t)kvo * m->kv_cap * 4, CC_DEV);
            ch->vc[i] = cc_buf((size_t)kvo * m->kv_cap * 4, CC_DEV);
            ch->ik[i] = cc_buf((size_t)c->idx_dim * m->kv_cap * 4, CC_DEV);
            ch->pk[i] = cc_buf((size_t)c->idx_dim * nbmax * 4, CC_DEV);
            if (!ch->kc[i] || !ch->vc[i] || !ch->ik[i] || !ch->pk[i]) { ch->cap = 0; return 0; }
        }
        ch->cap = m->kv_cap;
    }
    return 1;
}
static int q38cc_scratch(Q38CChain *ch, Model *m, int rows) {
    Cfg *c = &m->c; int H = c->hidden;
    int Ep = c->ngram_heads * c->ngram_head_dim; if (Ep < 1) Ep = 1;
    size_t r = (size_t)rows;
    int need_rows = ch->rows < rows;
    if (need_rows) ch->rows = rows;   /* kvd's stride follows */
    if (!q38cc_bufs(ch, m, rows)) return 0;
    if (need_rows) {
        float *hr = realloc(ch->host_routed, r * H * sizeof(float)), *he;
        if (!hr) return 0;
        ch->host_routed = hr;
        he = realloc(ch->host_emb, r * Ep * sizeof(float));
        if (!he) return 0;
        ch->host_emb = he;
    }
    return 1;
}

/* ---- state between the host and the device ------------------------------------ */
static void q38_layer_forward(Model *m,int i,float *hyper,const int *ids,int S,int pos_base,
                              float *mixed,float *inject,float *block);
static void q38_embed_row(Model *m,int id,int abs_pos,float *out);
/* The device was lost with the newest DeltaNet state, conv rings and PLE ring on it.
 * The host's K/V/index rows are canonical, those are not: rebuild them on the CPU by
 * running the `upto` positions the prefix record names (the PLE n-gram history replayed
 * with them), then leave the chain off. */
static void q38cc_recover(Model *m, int upto) {
    Q38CChain *ch = q38cc_of(m);
    Cfg *c = &m->c; int H = c->hidden, W = c->hc_width, C = c->hc_count, nl = c->layers;
    int ple = c->ple_layer >= 0 && c->ple_layer < nl;
    g_cuda_chain = 0;
    if (ch) { ch->failed = 1; ch->dn_where = Q38CC_HOST; ch->host_zero = 0; ch->snap_valid = 0; }
    for (int i = 0; i < nl; i++) {
        if (c->is_attn[i]) continue;
        memset(m->DN_rec[i], 0, (size_t)c->dn_vheads * c->dn_kdim * c->dn_vdim * sizeof(float));
        memset(m->DN_conv[i], 0, (size_t)c->dn_conv_dim * (c->dn_convk - 1) * sizeof(float));
    }
    if (ple && m->PLE_conv_state) memset(m->PLE_conv_state, 0, q38cc_ple_cells(c) * sizeof(float));
    if (ple && m->ple_history) m->ple_history_len = 0;
    if (upto <= 0) return;
    if (m->kvp.tainted || m->kvp.len < upto || !m->kvp.fed)
        q38cc_fatal("the device was lost with a recurrent state its token ids do not describe (an image)");
    fprintf(stderr, "[chain] qwen38: the device was lost; rebuilding the state of %d positions on the CPU, "
                    "which runs from here on\n", upto);
    int *ids = (int *)malloc((size_t)upto * sizeof(int));
    float *hyper = falloc((int64_t)upto * W), *mixed = falloc((int64_t)upto * H);
    float *inject = falloc((int64_t)upto * C), *block = falloc((int64_t)upto * H);
    if (!ids) { fprintf(stderr, "OOM rebuilding the state\n"); exit(1); }
    memcpy(ids, m->kvp.fed, (size_t)upto * sizeof(int));
    for (int s = 0; s < upto; s++) {
        float *e = hyper + (int64_t)s * W;
        q38_embed_row(m, ids[s], s, e);
        for (int b = 1; b < C; b++) memcpy(e + (int64_t)b * H, e, (size_t)H * sizeof(float));
    }
    float *pref = m->ple_pref; int pref_rows = m->ple_pref_rows, snap = m->snap_rows, rowwise = g_q38_rowwise;
    m->ple_pref = NULL; m->ple_pref_rows = 0; m->snap_rows = 0; g_q38_rowwise = 0;
    for (int i = 0; i < nl; i++) q38_layer_forward(m, i, hyper, ids, upto, 0, mixed, inject, block);
    free(m->ple_pref);
    m->ple_pref = pref; m->ple_pref_rows = pref_rows; m->snap_rows = snap; g_q38_rowwise = rowwise;
    free(ids); free(hyper); free(mixed); free(inject); free(block);
}
static int q38cc_sync_one(Model *m, Q38CChain *ch) {
    if (!ch || !ch->ok || ch->dn_where != Q38CC_DEV) return 1;
    Cfg *c = &m->c;
    size_t nr = (size_t)c->dn_vheads * c->dn_kdim * c->dn_vdim, nc = (size_t)c->dn_conv_dim * (c->dn_convk - 1);
    int ok = 1;
    for (int i = 0; i < c->layers && ok; i++) {
        if (c->is_attn[i]) continue;
        ok = cc_read(ch->rec[i], 0, m->DN_rec[i], nr * 4) && cc_read(ch->ring[i], 0, m->DN_conv[i], nc * 4);
    }
    if (ok && ch->ple_ring && m->PLE_conv_state)
        ok = cc_read(ch->ple_ring, (size_t)ch->ple_slot[0] * q38cc_ple_cells(c), m->PLE_conv_state, q38cc_ple_cells(c) * 4);
    if (ok) ch->dn_where = Q38CC_BOTH;
    return ok;
}
static void q38cc_sync_host(Model *m) {
    if (!q38cc_sync_one(m, q38cc_of(m))) q38cc_recover(m, m->kv_len);
}
static void q38cc_host_wrote(Model *m, int zero) {
    Q38CChain *ch = q38cc_of(m);
    if (!ch || !ch->ok) return;
    ch->dn_where = Q38CC_HOST; ch->host_zero = zero; ch->snap_valid = 0;
}
static void q38cc_cpu_step(Model *m, int pos_base) {
    Q38CChain *ch = q38cc_of(m);
    if (!ch || !ch->ok) return;
    if (!q38cc_sync_one(m, ch)) { q38cc_recover(m, m->kv_len); return; }
    ch->dn_where = Q38CC_HOST; ch->host_zero = 0; ch->snap_valid = 0;
    int R = m->c.idx_ratio > 0 ? m->c.idx_ratio : 1;
    for (int i = 0; i < m->c.layers; i++) {
        if (ch->kv_valid[i] > pos_base) ch->kv_valid[i] = pos_base;
        if (ch->pk_valid[i] > pos_base / R) ch->pk_valid[i] = pos_base / R;
    }
}
static void q38cc_rollback(Model *m, int slot, int len) {
    Q38CChain *ch = q38cc_of(m);
    if (!ch || !ch->ok) return;
    int R = m->c.idx_ratio > 0 ? m->c.idx_ratio : 1;
    for (int i = 0; i < m->c.layers; i++) {
        if (ch->kv_valid[i] > len) ch->kv_valid[i] = len;
        if (ch->pk_valid[i] > len / R) ch->pk_valid[i] = len / R;
    }
    if (slot < 0 || slot >= ch->snap_valid || ch->dn_where != Q38CC_DEV) { ch->snap_valid = 0; return; }
    for (int i = 0; i < m->c.layers; i++) {
        if (m->c.is_attn[i]) continue;
        CcBuf *t = ch->rec[i]; ch->rec[i] = ch->rec_snap[slot][i]; ch->rec_snap[slot][i] = t;
        t = ch->ring[i]; ch->ring[i] = ch->ring_snap[slot][i]; ch->ring_snap[slot][i] = t;
    }
    if (ch->ple_ring) { int t = ch->ple_slot[0]; ch->ple_slot[0] = ch->ple_slot[1 + slot]; ch->ple_slot[1 + slot] = t; }
    ch->snap_valid = 0;
}
static int q38cc_spec_slots(Q38CChain *ch, Model *m, int rows) {
    Cfg *c = &m->c; int L = c->layers;
    if (rows > Q38_SPEC_SNAPS) rows = Q38_SPEC_SNAPS;
    size_t nr = (size_t)c->dn_vheads * c->dn_kdim * c->dn_vdim, nc = (size_t)c->dn_conv_dim * (c->dn_convk - 1);
    for (int sl = ch->snap_slots; sl < rows; sl++) {
        if (!ch->rec_snap[sl] && !(ch->rec_snap[sl] = (CcBuf **)calloc((size_t)L, sizeof(CcBuf *)))) return 0;
        if (!ch->ring_snap[sl] && !(ch->ring_snap[sl] = (CcBuf **)calloc((size_t)L, sizeof(CcBuf *)))) return 0;
        for (int i = 0; i < L; i++) {
            if (c->is_attn[i]) continue;
            if (!ch->rec_snap[sl][i] && !(ch->rec_snap[sl][i] = cc_buf(nr * 4, CC_DEV))) return 0;
            if (!ch->ring_snap[sl][i] && !(ch->ring_snap[sl][i] = cc_buf(nc * 4, CC_DEV))) return 0;
        }
        ch->snap_slots = sl + 1;
    }
    if (ch->ple_ring && ch->ple_nslots < rows + 1) {
        size_t pc = q38cc_ple_cells(c);
        CcBuf *nb = cc_buf((size_t)Q38_SPEC_ROWS * pc * 4, CC_DEV);
        if (!nb) return 0;
        if (!cc_begin() || !cc_copy(nb, 0, ch->ple_ring, (size_t)ch->ple_slot[0] * pc, pc) || !cc_submit(1)) { cc_free(nb); return 0; }
        cc_free(ch->ple_ring);
        ch->ple_ring = nb; ch->ple_nslots = Q38_SPEC_ROWS;
        for (int r = 0; r < Q38_SPEC_ROWS; r++) ch->ple_slot[r] = r;
        ch->snap_valid = 0;
    }
    return 1;
}
static int q38cc_push_state(Q38CChain *ch, Model *m, int pos_base) {
    Cfg *c = &m->c; int ok = 1;
    if (ch->dn_where == Q38CC_HOST) {
        size_t nr = (size_t)c->dn_vheads * c->dn_kdim * c->dn_vdim, nc = (size_t)c->dn_conv_dim * (c->dn_convk - 1);
        for (int i = 0; i < c->layers && ok; i++) {
            if (c->is_attn[i]) continue;
            if (ch->host_zero) ok = cc_zero(ch->rec[i], 0, nr) && cc_zero(ch->ring[i], 0, nc);
            else ok = cc_write(ch->rec[i], 0, m->DN_rec[i], nr * 4) && cc_write(ch->ring[i], 0, m->DN_conv[i], nc * 4);
        }
        if (ok && ch->ple_ring && m->PLE_conv_state) {
            size_t pc = q38cc_ple_cells(c), off = (size_t)ch->ple_slot[0] * pc;
            ok = ch->host_zero ? cc_zero(ch->ple_ring, off, pc) : cc_write(ch->ple_ring, off, m->PLE_conv_state, pc * 4);
        }
        ch->dn_where = Q38CC_BOTH;
    }
    int D = c->head_dim, KVH = c->kv_heads, ID = c->idx_dim, R = c->idx_ratio;
    for (int i = 0; i < c->layers && ok; i++) {
        if (!c->is_attn[i]) continue;
        if (ch->pk_valid[i] > pos_base / R) ch->pk_valid[i] = pos_base / R;   /* blocks this step rewrites */
        if (ch->kv_valid[i] >= pos_base) continue;
        int t0 = ch->kv_valid[i], n = pos_base - t0;
        for (int h = 0; h < KVH && ok; h++) {
            size_t src = ((size_t)h * m->kv_cap + t0) * D, dst = ((size_t)h * ch->cap + t0) * D;
            ok = cc_write(ch->kc[i], dst, m->K[i] + src, (size_t)n * D * 4) && cc_write(ch->vc[i], dst, m->V[i] + src, (size_t)n * D * 4);
        }
        ok = ok && cc_write(ch->ik[i], (size_t)t0 * ID, m->IK[i] + (size_t)t0 * ID, (size_t)n * ID * 4);
        ch->kv_valid[i] = pos_base;
        if (ch->pk_valid[i] > t0 / R) ch->pk_valid[i] = t0 / R;
    }
    return ok;
}

/* ---- pieces ---------------------------------------------------------------------- */
static int q38cc_gr_read(Q38CChain *ch, Model *m, GatedResidual *g, size_t norm_off, int n, CcBuf *inj) {
    Cfg *c = &m->c; int H = c->hidden, W = c->hc_width, C = c->hc_count, R = c->hc_rank;
    CcNorm np = {n * C, H, C, 0, W, H, 0, W, H, (int)norm_off, C, CC_NORM_ADD1, c->eps, 1.f};
    CcEw lo = {CC_EW_HC_LOW, n * R, R, C, 0, 1, 0, 0, 0, 0, 0, (float)C};
    CcEw mx = {CC_EW_HC_MIX, n * H, H, C, 0, 1, 0, 0, 0, 0, 0, (float)C};
    int ok = cc_norm(ch->hyper, ch->prm, ch->hn, &np) &&
             cc_matmul(q38cc_t(ch, &g->down), ch->hn, 0, ch->low, 0, n) && cc_ew(ch->low, ch->low, NULL, NULL, NULL, &lo) &&
             cc_matmul(q38cc_t(ch, &g->up), ch->low, 0, ch->mix, 0, n) && cc_ew(ch->mixed, ch->mix, ch->hn, NULL, NULL, &mx);
    if (ok && inj) {
        CcEw ij = {CC_EW_HC_INJ, n * C, C, C, 0, 1, 0, 0, 0, 0, 0, (float)C};
        ok = cc_matmul(q38cc_t(ch, &g->inject), ch->hn, 0, inj, 0, n) && cc_ew(inj, inj, NULL, NULL, NULL, &ij);
    }
    return ok;
}
static int q38cc_gr_apply(Q38CChain *ch, Model *m, CcBuf *inj, int n) {
    Cfg *c = &m->c;
    CcEw ap = {CC_EW_HC_APPLY, n * c->hc_width, c->hidden, c->hc_count, 0, 1, 0, 0, 0, 0, 0, 1.f};
    return cc_ew(ch->hyper, inj, ch->blk, NULL, NULL, &ap);
}
static int q38cc_attention(Q38CChain *ch, Model *m, Layer *l, int i, int n, int pb) {
    Cfg *c = &m->c;
    int QH = c->q_heads, KVH = c->kv_heads, D = c->head_dim, kvo = KVH * D, qo = QH * 2 * D, half = c->rotary_dim / 2;
    int IQ = c->idx_qheads, ID = c->idx_dim, R = c->idx_ratio, ipw = (IQ + 1) * ID;
    int ok = cc_matmul(q38cc_t(ch, &l->q), ch->mixed, 0, ch->q, 0, n) && cc_matmul(q38cc_t(ch, &l->k), ch->mixed, 0, ch->k, 0, n) &&
             cc_matmul(q38cc_t(ch, &l->v), ch->mixed, 0, ch->v, 0, n) && cc_matmul(q38cc_t(ch, &l->idx_qk), ch->mixed, 0, ch->ip, 0, n);
    CcNorm kn = {n * KVH, D, KVH, 0, kvo, D, 0, kvo, D, (int)ch->o_kn[i], 0, CC_NORM_ADD1, c->eps, 1.f};
    CcNorm qn = {n * QH, D, QH, 0, qo, 2 * D, 0, qo, 2 * D, (int)ch->o_qn[i], 0, CC_NORM_ADD1, c->eps, 1.f};
    CcNorm in = {n * IQ, ID, IQ, 0, ipw, ID, 0, ipw, ID, (int)ch->o_iqn[i], 0, CC_NORM_ADD1, c->eps, 1.f};
    ok = ok && cc_norm(ch->k, ch->prm, ch->k, &kn) && cc_norm(ch->q, ch->prm, ch->q, &qn) && cc_norm(ch->ip, ch->prm, ch->ip, &in);
    if (ok && half) {
        CcRope rk = {n * KVH, KVH, 0, kvo, D, half, 0, 2 * half}, rq = {n * QH, QH, 0, qo, 2 * D, half, 0, 2 * half};
        CcRope ri = {n * IQ, IQ, 0, ipw, ID, half, 0, 2 * half};
        ok = cc_rope(ch->k, ch->cs, &rk) && cc_rope(ch->q, ch->cs, &rq) && cc_rope(ch->ip, ch->cs, &ri);
    }
    if (!ok) return 0;
    /* the rows into the device caches and the host's */
    CcRegion *rg = malloc(sizeof *rg * (size_t)n * (KVH > 1 ? KVH : 1));
    CcRegion *ri = malloc(sizeof *ri * (size_t)n);
    if (!rg || !ri) { free(rg); free(ri); return 0; }
    for (int s = 0; s < n; s++) {
        for (int h = 0; h < KVH; h++)
            rg[s * KVH + h] = (CcRegion){((size_t)h * ch->cap + pb + s) * D, (size_t)s * kvo + (size_t)h * D, (size_t)D};
        ri[s] = (CcRegion){(size_t)(pb + s) * ID, (size_t)s * ipw + (size_t)IQ * ID, (size_t)ID};
    }
    size_t ko = (size_t)ch->attn_ord[i] * q38cc_kv_stride(ch, c);
    ok = cc_copy_regions(ch->kc[i], ch->k, rg, n * KVH) && cc_copy_regions(ch->vc[i], ch->v, rg, n * KVH) &&
         cc_copy_regions(ch->ik[i], ch->ip, ri, n) &&
         cc_copy(ch->kvd, ko, ch->k, 0, (size_t)n * kvo) &&
         cc_copy(ch->kvd, ko + (size_t)ch->rows * kvo, ch->v, 0, (size_t)n * kvo);
    for (int s = 0; s < n && ok; s++)   /* index keys back, compact */
        ok = cc_copy(ch->kvd, ko + 2 * (size_t)ch->rows * kvo + (size_t)s * ID, ch->ip, (size_t)s * ipw + (size_t)IQ * ID, ID);
    free(rg); free(ri);
    /* the pooled keys of the blocks this step completes (and any behind the watermark) */
    int nb = (pb + n) / R, b0 = ch->pk_valid[i];
    if (ok && nb > b0) {
        float *pc = (float *)cc_ptr(ch->pcs);
        if (half) for (int b = b0; b < nb; b++) for (int j = 0; j < half; j++) {
            float ang = (float)(b * R) / powf(c->theta, (float)(2 * j) / c->rotary_dim);
            pc[((b - b0) * half + j) * 2] = cosf(ang); pc[((b - b0) * half + j) * 2 + 1] = sinf(ang);
        }
        CcQsa pp = {0, ID, R, b0, half, 0, 0, 0, 0, 0, 0, 0, 0, c->eps, 0, (int)ch->o_ikn[i], 0, nb - b0};
        ok = cc_qsa(ch->ik[i], ch->prm, ch->pk[i], ch->pcs, NULL, NULL, &pp);
        ch->pk_valid[i] = nb;
    }
    int nbmax = (int)(cc_bytes(ch->qsc) / 4 / (2 * (size_t)ch->rows));
    CcQsa ps = {1, ID, R, 0, 0, n, pb, c->idx_budget, IQ, 0, ipw, nbmax, q38cc_selrow(c), c->eps, 0, 0, 0, 0};
    CcAttn at = {n, QH, KVH, D, pb, ch->cap, 0, qo, 2 * D, D, qo, 2 * D, 1, 0, QH * D, 0, q38cc_selrow(c),
                 1.f / sqrtf((float)D), 0, 0};
    return ok && cc_qsa(ch->ip, NULL, ch->pk[i], NULL, ch->qsc, ch->sel, &ps) &&
           cc_attn(ch->q, ch->kc[i], ch->vc[i], ch->ctx, ch->q, ch->sel, &at) &&
           cc_matmul(q38cc_t(ch, &l->o), ch->ctx, 0, ch->blk, 0, n);
}
/* ns: the chunk's rows whose state a verify copies (rows 0..ns-1, into slots c0..): the
 * convolution and the recurrence split into one dispatch per copy, each ending on its
 * row (the same bits). */
static int q38cc_deltanet(Q38CChain *ch, Model *m, Layer *l, int i, int n, int c0, int ns) {
    Cfg *c = &m->c;
    int VH = c->dn_vheads, CD = c->dn_conv_dim, V = VH * c->dn_vdim;
    int ok = cc_matmul(q38cc_t(ch, &l->dn_qkv), ch->mixed, 0, ch->qkv, 0, n) && cc_matmul(q38cc_t(ch, &l->dn_z), ch->mixed, 0, ch->z, 0, n) &&
             cc_matmul(q38cc_t(ch, &l->dn_b), ch->mixed, 0, ch->ab, 0, n) &&
             cc_matmul(q38cc_t(ch, &l->dn_a), ch->mixed, 0, ch->ab, (size_t)ch->rows * VH, n);
    int segs = ns > 1 ? ns : 1;
    for (int r = 0; r < segs && ok; r++) {
        int s0 = ns > 1 ? r : 0, len = ns > 1 && r < ns - 1 ? 1 : n - s0, snap_row = ns ? 0 : -1, slot = ns ? c0 + r : 0;
        CcDnConv cp = {len, CD, c->dn_convk, s0 * CD, CD, s0 * CD, CD, snap_row, 1, (int)ch->o_conv[i], 0, 0};
        CcDnRec rp = {len, VH, c->dn_kheads, c->dn_vdim, c->dn_kheads * c->dn_kdim, s0 * CD, CD, s0 * VH, VH,
                      ch->rows * VH + s0 * VH, VH, s0 * V, V, s0 * V, V, snap_row, 1, c->eps,
                      1.f / sqrtf((float)c->dn_kdim), 0, 0, (int)ch->o_dn[i]};
        ok = cc_dnconv(ch->qkv, ch->prm, ch->ring[i], ch->cv, ch->ring_snap[slot][i], &cp) &&
             cc_dnrec(c->dn_kdim, ch->cv, ch->ab, ch->z, ch->rec[i], ch->prm, ch->dny, ch->rec_snap[slot][i], &rp);
    }
    return ok && cc_matmul(q38cc_t(ch, &l->dn_out), ch->dny, 0, ch->blk, 0, n);
}
/* the n-gram rows of n tokens, in order, as q38_ple fills them (history and the verify's snapshot) */
static void q38cc_ple_rows(Model *m, const int *ids, int c0, int n, float *emb) {
    Cfg *c = &m->c; int E = c->ngram_heads * c->ngram_head_dim;
    for (int r = 0; r < n; r++) {
        int s = c0 + r; float *e = emb + (size_t)r * E;
        int64_t p1 = m->ple_history_len >= 1 ? m->ple_history[m->ple_history_len - 1] : c->eos_id;
        int64_t p2 = m->ple_history_len >= 2 ? m->ple_history[m->ple_history_len - 2] : c->eos_id;
        if (m->ple_pref && s < m->ple_pref_rows)
            memcpy(e, m->ple_pref + (int64_t)s * c->ngram_heads * c->ngram_head_dim, (size_t)E * sizeof(float));
        else for (int h = 0; h < c->ngram_heads; h++) {
            int ng = h < c->heads_per_ngram ? 2 : 3;
            q38_ple_row(m, q38_hash_row(m, h, ng, ids[s], p1, p2), e + (int64_t)h * c->ngram_head_dim);
        }
        if (ids[s] == c->eos_id) m->ple_history_len = 0;
        else if (m->ple_history_len == 0) { m->ple_history[0] = ids[s]; m->ple_history_len = 1; }
        else if (m->ple_history_len == 1) { m->ple_history[1] = ids[s]; m->ple_history_len = 2; }
        else { m->ple_history[0] = m->ple_history[1]; m->ple_history[1] = ids[s]; }
        if (s < m->snap_rows) {
            memcpy(m->snap_ple_history[s], m->ple_history, sizeof(m->snap_ple_history[s]));
            m->snap_ple_history_len[s] = m->ple_history_len;
        }
    }
}
static int q38cc_ple(Q38CChain *ch, Model *m, int n, int c0, int ns) {
    Cfg *c = &m->c; Layer *l = &m->L[c->ple_layer];
    int W = c->hc_width, H = c->hidden, E = c->ngram_heads * c->ngram_head_dim;
    size_t pc = q38cc_ple_cells(c);
    int ok = cc_write(ch->emb, 0, ch->host_emb, (size_t)n * E * 4) &&
             cc_matmul(q38cc_t(ch, &l->ple_key), ch->emb, 0, ch->keys, 0, n) &&
             cc_matmul(q38cc_t(ch, &l->ple_value), ch->emb, 0, ch->val, 0, n);
    CcPle g = {0, n, c->hc_count, H, c->ple_convk, c->ngram_size, 0, 0, 0, -1, 0, c->eps, (int)ch->o_ple, 0, 0};
    ok = ok && cc_ple(ch->keys, ch->hyper, ch->val, ch->prm, ch->gated, ch->normv, NULL, NULL, &g);
    if (ns <= 1) {
        int snap = ns ? ch->ple_slot[1 + c0] : ch->ple_slot[1];
        CcPle cv = {1, n, c->hc_count, H, c->ple_convk, c->ngram_size, 0, 0, 0, ns ? 0 : -1,
                    (int)((size_t)snap * pc), c->eps, 0, (int)ch->o_pconv, (int)((size_t)ch->ple_slot[0] * pc)};
        return ok && cc_ple(NULL, ch->hyper, NULL, NULL, ch->gated, ch->normv, ch->prm, ch->ple_ring, &cv);
    }
    ok = ok && cc_reserve(&ch->gseg, (size_t)n * W * 4, CC_DEV) && cc_reserve(&ch->nseg, (size_t)n * W * 4, CC_DEV);
    for (int r = 0; r < ns && ok; r++) {
        int len = r < ns - 1 ? 1 : n - r;
        CcBuf *gb = ch->gated, *nb = ch->normv;
        if (r) {
            ok = cc_copy(ch->gseg, 0, ch->gated, (size_t)r * W, (size_t)len * W) &&
                 cc_copy(ch->nseg, 0, ch->normv, (size_t)r * W, (size_t)len * W);
            gb = ch->gseg; nb = ch->nseg;
        }
        CcPle cv = {1, len, c->hc_count, H, c->ple_convk, c->ngram_size, 0, r * W, 0, 0,
                    (int)((size_t)ch->ple_slot[1 + c0 + r] * pc), c->eps, 0, (int)ch->o_pconv,
                    (int)((size_t)ch->ple_slot[0] * pc)};
        ok = ok && cc_ple(NULL, ch->hyper, NULL, NULL, gb, nb, ch->prm, ch->ple_ring, &cv);
    }
    return ok;
}
static int q38cc_shared(Q38CChain *ch, Model *m, Layer *l, int i, int n) {
    Cfg *c = &m->c; int SI = c->shared_inter;
    CcEw sw = {CC_EW_SWIGLU, n * SI, SI, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
    return cc_matmul(q38cc_t(ch, &l->sh_g), ch->mixed, 0, ch->gs, 0, n) && cc_matmul(q38cc_t(ch, &l->sh_u), ch->mixed, 0, ch->us, 0, n) &&
           cc_ew(ch->hs, ch->gs, ch->us, NULL, NULL, &sw) && cc_matmul(q38cc_t(ch, &l->sh_d), ch->hs, 0, ch->ds, 0, n) &&
           cc_matmul(ch->t_sg[i], ch->mixed, 0, ch->sgd, 0, n);
}
/* block = routed + sigmoid(gate) * shared (q38_moe's out), then hyper += inject x block */
static int q38cc_moe_apply(Q38CChain *ch, Model *m, int n) {
    Cfg *c = &m->c;
    CcEw cb = {CC_EW_COMBINE, n * c->hidden, c->hidden, 1, 1 | 2 | 4 | 8, 1, 0, 0, 0, 0, 0, 1.f};
    return cc_ew(ch->blk, NULL, ch->routed, ch->ds, ch->sgd, &cb) && q38cc_gr_apply(ch, m, ch->inj_m, n);
}
static int q38cc_frame_begin(Q38CChain *ch) {
    ch->frames++;
    if (ch->fault_at && ch->frames == ch->fault_at) { fprintf(stderr, "[chain] qwen38: fault injected at frame %llu\n", ch->frames); return 0; }
    return cc_begin();
}

/* Every layer for S rows. hyper_h: the rows' streams in, the final ones out when
 * want_streams; mixed_h (or NULL): every row's final mixed row, for the prefill
 * read-out; logit: the last nlogits rows' logits. 0 = not taken (nothing changed, or
 * the device was lost and the state rebuilt: the CPU runs the step), 1 = done. */
static int q38cc_forward(Model *m, const int *ids, int S, int pos_base, int nlogits, float *hyper_h,
                         int want_streams, float *mixed_h, float *logit) {
    if (!g_cuda_chain || m->mux_rows) return 0;
    Q38CChain *ch = q38cc_of(m);
    if (!ch || !ch->ok || ch->failed) return 0;
    Cfg *c = &m->c; int H = c->hidden, W = c->hc_width, L = c->layers, E = c->experts, V = c->vocab;
    int CH = q38cc_rows_env(), rows = S < CH ? S : CH;
    if (m->snap_rows > 1 && !q38cc_spec_slots(ch, m, m->snap_rows)) {
        fprintf(stderr, "[chain] qwen38: device memory for a verify's %d copies refused; this verify on the CPU\n", m->snap_rows);
        return 0;
    }
    if (!q38cc_mirror(ch, m) || !q38cc_scratch(ch, m, rows) || (want_streams && !cc_reserve(&ch->hypd, (size_t)rows * W * 4, CC_DOWN)) ||
        (nlogits > 2 && !cc_reserve(&ch->outd, (size_t)nlogits * V * 4, CC_DOWN)) ||
        (mixed_h && !cc_reserve(&ch->find, (size_t)rows * H * 4, CC_DOWN))) {
        fprintf(stderr, "[chain] qwen38: device memory for %d rows refused; the per-matrix path\n", rows);
        ch->failed = 1;
        return 0;
    }
    int kvo = c->kv_heads * c->head_dim, ID = c->idx_dim, half = c->rotary_dim / 2;
    int snapped = 0;
    double w0 = ch->wait_ms, h0 = ch->host_ms;
    for (int c0 = 0; c0 < S; c0 += rows) {
        int n = S - c0 < rows ? S - c0 : rows, pb = pos_base + c0;
        int ns = m->snap_rows - c0 < n ? m->snap_rows - c0 : n;   /* the rows of this chunk a verify copies */
        if (ns < 0) ns = 0;
        if (ns) snapped = c0 + ns;
        if (half) {
            float *cs = (float *)cc_ptr(ch->cs);
            for (int s = 0; s < n; s++) for (int j = 0; j < half; j++) {
                float ang = (float)(pb + s) / powf(c->theta, (float)(2 * j) / c->rotary_dim);
                cs[(s * half + j) * 2] = cosf(ang); cs[(s * half + j) * 2 + 1] = sinf(ang);
            }
        }
        if (c->ple_layer >= 0 && c->ple_layer < L) q38cc_ple_rows(m, ids, c0, n, ch->host_emb);
        if (!q38cc_frame_begin(ch) || !cc_write(ch->hyper, 0, hyper_h + (size_t)c0 * W, (size_t)n * W * 4) ||
            !q38cc_push_state(ch, m, pb)) goto lost;
        int ok = 1, pending = 0;
        for (int i = 0; i < L && ok; i++) {
            Layer *l = &m->L[i];
            if (pending) ok = q38cc_moe_apply(ch, m, n);
            if (ok && i == c->ple_layer) ok = q38cc_ple(ch, m, n, c0, ns);
            ok = ok && q38cc_gr_read(ch, m, &l->attn_gr, ch->o_an[i], n, ch->inj_a);
            ok = ok && (c->is_attn[i] ? q38cc_attention(ch, m, l, i, n, pb) : q38cc_deltanet(ch, m, l, i, n, c0, ns));
            ok = ok && q38cc_gr_apply(ch, m, ch->inj_a, n);
            ok = ok && q38cc_gr_read(ch, m, &l->mlp_gr, ch->o_mn[i], n, ch->inj_m);
            ok = ok && cc_matmul(q38cc_t(ch, &l->router), ch->mixed, 0, ch->lg, 0, n) &&
                 cc_copy(ch->lgd, 0, ch->lg, 0, (size_t)n * E) && cc_copy(ch->mixd, 0, ch->mixed, 0, (size_t)n * H);
            double t0 = now_s();
            ok = ok && cc_submit(1);   /* A1 */
            m->timers.seconds[Q38_TM_DENSE_MATMUL] += now_s() - t0; ch->wait_ms += (now_s() - t0) * 1e3;
            if (!ok) break;
            if (c->is_attn[i]) {
                const float *kv = (const float *)cc_ptr(ch->kvd) + (size_t)ch->attn_ord[i] * q38cc_kv_stride(ch, c);
                for (int s = 0; s < n; s++) {
                    for (int h = 0; h < c->kv_heads; h++) {
                        size_t dst = ((size_t)h * m->kv_cap + pb + s) * c->head_dim, src = (size_t)s * kvo + (size_t)h * c->head_dim;
                        memcpy(m->K[i] + dst, kv + src, c->head_dim * sizeof(float));
                        memcpy(m->V[i] + dst, kv + (size_t)ch->rows * kvo + src, c->head_dim * sizeof(float));
                    }
                    memcpy(m->IK[i] + (size_t)(pb + s) * ID, kv + 2 * (size_t)ch->rows * kvo + (size_t)s * ID, ID * sizeof(float));
                }
            }
            ok = q38cc_frame_begin(ch) && q38cc_shared(ch, m, l, i, n) && cc_submit(0);   /* A2 */
            double t1 = now_s();
            q38_moe_ex(m, l, i, (const float *)cc_ptr(ch->mixd), n, ch->host_routed, (const float *)cc_ptr(ch->lgd), 1);
            memcpy(cc_ptr(ch->routed), ch->host_routed, (size_t)n * H * sizeof(float));
            ch->host_ms += (now_s() - t1) * 1e3;
            ok = ok && q38cc_frame_begin(ch);
            pending = 1;
        }
        if (ok) ok = q38cc_moe_apply(ch, m, n);
        if (ok && want_streams) ok = cc_copy(ch->hypd, 0, ch->hyper, 0, (size_t)n * W);
        int lo0 = S - nlogits > c0 ? S - nlogits - c0 : 0, lo_n = n - lo0;   /* this chunk's logit rows */
        int need_final = mixed_h || (lo0 < n && c0 + n > S - nlogits);
        if (ok && need_final) ok = q38cc_gr_read(ch, m, &m->final_gr, ch->o_fn, n, NULL);
        if (ok && mixed_h) ok = cc_copy(ch->find, 0, ch->mixed, 0, (size_t)n * H);
        if (ok && need_final && lo_n > 0 && c0 + n > S - nlogits) {
            int dst = c0 + lo0 - (S - nlogits);
            ok = cc_matmul(q38cc_t(ch, &m->lm_head), ch->mixed, (size_t)lo0 * H, ch->outd, (size_t)dst * V, lo_n);
        }
        double t0 = now_s();
        ok = ok && cc_submit(1);
        m->timers.seconds[Q38_TM_DENSE_MATMUL] += now_s() - t0; ch->wait_ms += (now_s() - t0) * 1e3;
        if (!ok) goto lost;
        if (want_streams) memcpy(hyper_h + (size_t)c0 * W, cc_ptr(ch->hypd), (size_t)n * W * 4);
        if (mixed_h) memcpy(mixed_h + (size_t)c0 * H, cc_ptr(ch->find), (size_t)n * H * 4);
        for (int i = 0; i < L; i++) if (c->is_attn[i]) ch->kv_valid[i] = pb + n;
        ch->dn_where = Q38CC_DEV; ch->host_zero = 0;
    }
    memcpy(logit, cc_ptr(ch->outd), (size_t)nlogits * V * sizeof(float));
    ch->snap_valid = snapped;
    ch->forwards++;
    if (S <= Q38_SPEC_ROWS) { ch->dec_n++; ch->dec_wait_ms += ch->wait_ms - w0; ch->dec_host_ms += ch->host_ms - h0; }   /* a decode step, or a verify's */
    return 1;
lost:   /* a frame failed: the device is gone (or would not take a command); the CPU takes over */
    if (!cc_lost()) cc_finish();
    q38cc_recover(m, pos_base);
    return 0;
}

static void q38cc_report(Model *m) {
    Q38CChain *ch = m ? q38cc_of(m) : NULL;
    if (!ch || !ch->ok || !ch->forwards) return;
    CcStats st; cc_stats(&st);
    fprintf(stderr, "[chain] qwen38: %llu forwards, %llu frames (%llu ops, %llu matmuls), %.1f ms waiting for the device, "
                    "%.1f ms of routed experts on the host, %.1f MiB on the device; per decode step %.2f ms waiting, %.2f ms of experts (%llu steps)\n",
            ch->forwards, st.frames, st.ops, st.matmuls, ch->wait_ms, ch->host_ms, st.dev_bytes / 1048576.0,
            ch->dec_n ? ch->dec_wait_ms / ch->dec_n : 0.0, ch->dec_n ? ch->dec_host_ms / ch->dec_n : 0.0, ch->dec_n);
}
