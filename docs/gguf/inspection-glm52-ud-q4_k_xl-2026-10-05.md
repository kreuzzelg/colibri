# Inspection report: `unsloth/GLM-5.2-GGUF` · `UD-Q4_K_XL` with the phase-1 reader

Date: 2026-10-05 · Branch: `gguf/p1-reader` · Tools: `c/gguf.h` (via `tests/test_gguf <dir>`),
`c/ggufinfo.py` (via `coli gguf inspect`), `coli doctor --deep`.

## Method

Only metadata was fetched (phase 1 reads headers, never weights):

- part 1 (`…-00001-of-00011.gguf`, 9.4 MB) downloaded completely — unsloth writes the
  whole model metadata into a tensor-less first part;
- parts 2–11: the first 2 MB of each (their metadata is 7–15 KB), then each file was
  extended **sparsely** to its real size (`os.truncate`), so the reader's bounds checks
  see the true geometry while the payload reads as zeros. Total download ≈ 30 MB.

Phase-1 exit criterion "inspects a real GGUF" is **met**: both readers index the
11-part set in well under a second (C reader: 0.38 s wall for 1809 tensors), agree on
every file, key and tensor, and `coli doctor --deep` passes every GGUF check (the
`memory.ram` failure below is the container's small RAM, correctly reported).

```
[  ok] model.gguf.header  GGUF v3 · 11 parts · 1809 tensors · glm-dsa · Glm-5.2
[  ok] model.gguf.splits  11 parts, split.no/split.count/split.tensors.count consistent
[  ok] model.gguf.arch    architecture glm-dsa → glm engine
[  ok] model.gguf.types   all tensor types are in the v1 set: Q4_K 150 · Q5_K 74 · Q8_0 872 · Q6_K 4 · F32 709
[  ok] model.gguf.mtp_precision MTP head eh_proj is Q8_0 (8.50 bpw)
[  ok] model.tokenizer    embedded tokenizer: gpt2 · pre glm4 · 154880 tokens
[fail] memory.ram         the resident dense set (21.0 GB) exceeds available RAM
[  ok] model.gguf.payload every sized tensor is readable to its last byte (1809 tensors)
```

## What the file says about GLM-5.2 (facts the design documents assumed)

| Key | Value | Design impact |
|---|---|---|
| `general.architecture` | `glm-dsa` | as assumed (REQ §4) |
| `glm-dsa.block_count` | **79** | llama.cpp counts the NextN block **inside** `block_count`; trunk = 78, MTP = `blk.78`. Fixed in `ggufinfo.summarize` (was looking at `blk.79`). |
| `glm-dsa.nextn_predict_layers` | 1 | |
| `glm-dsa.leading_dense_block_count` | 3 | `first_dense = 3` |
| `glm-dsa.embedding_length` | 6144 | |
| `glm-dsa.feed_forward_length` | 12288 | dense MLP of blk 0–2 |
| `glm-dsa.expert_feed_forward_length` | 2048 | `moe_inter = 2048` (the "≈19 MB/expert" figure assumed this) |
| `glm-dsa.expert_count` / `expert_used_count` | 256 / 8 | |
| `glm-dsa.expert_shared_count` | 1 | |
| `glm-dsa.expert_weights_scale` / `expert_weights_norm` / `expert_gating_func` | 2.5 / true / 2 (sigmoid) | `routed_scale`, `norm_topk`, gating as the engine expects |
| `glm-dsa.expert_group_count` / `expert_group_used_count` | 1 / 1 | `n_group = 1` as the engine requires |
| `glm-dsa.attention.head_count` | 64 | |
| `glm-dsa.attention.q_lora_rank` | **2048** | ARCH §6.1 fixture used 1536 (DeepSeek value); GLM-5.2 is 2048 |
| `glm-dsa.attention.kv_lora_rank` | 512 | |
| `glm-dsa.attention.key_length_mla` / `rope.dimension_count` | 256 / 64 | `qk_nope = 192`, `qk_rope = 64` |
| `glm-dsa.attention.value_length_mla` | 256 | `v_head = 256` |
| `glm-dsa.attention.key_length` / `value_length` | 576 / 512 | the compressed-KV widths (kv_lora + rope, kv_lora) |
| `glm-dsa.rope.freq_base` | 8 000 000 | |
| `glm-dsa.attention.layer_norm_rms_epsilon` | 1e-5 | |
| `glm-dsa.context_length` | 1 048 576 | |
| `glm-dsa.attention.indexer.head_count` / `key_length` / `top_k` | 32 / 128 / 2048 | no `indexer.types` key: **every** block 0–78 carries indexer tensors |
| `tokenizer.ggml.model` / `pre` | `gpt2` / `glm4` | cl100k family, as planned |
| `tokenizer.ggml.{bos,eos,eot,eom}_token_id` | 154822 / 154820 / 154827 / 154829 | three stop ids available by key, `<|user|>`/`<|observation|>` still resolved by name |
| `general.sampling.temp` / `top_p` | 1.0 / 0.95 | |

## Tensor inventory (1809 tensors, 467.28 GB)

Per block (blk.3 as the representative MoE layer; types in parentheses):

```
attn_norm F32 [6144]            ffn_norm F32 [6144]
attn_q_a Q8_0 [6144→2048]       attn_q_a_norm F32 [2048]       attn_q_b Q8_0 [2048→16384]
attn_kv_a_mqa Q8_0 [6144→576]   attn_kv_a_norm F32 [512]
attn_k_b Q8_0 {192,512,64}      attn_v_b Q8_0 {512,256,64}     (NO attn_kv_b)
attn_output Q8_0 [16384→6144]
ffn_gate_inp F32 [6144→256]     exp_probs_b F32 [256]
ffn_gate_exps Q4_K {6144,2048,256}   ffn_up_exps Q4_K {6144,2048,256}   ffn_down_exps Q5_K {2048,6144,256}
ffn_gate_shexp / ffn_up_shexp Q8_0 [6144→2048]   ffn_down_shexp Q8_0 [2048→6144]
indexer.attn_q_b Q8_0 [2048→4096]   indexer.attn_k Q8_0 [6144→128]   indexer.proj F32 [6144→32]
indexer.k_norm.weight / .bias F32 [128]
```

Blocks 0–2: dense `ffn_gate/ffn_up [6144→12288]`, `ffn_down [12288→6144]`, all `Q8_0`.
Block 78 (MTP): the full attention + MoE set above **plus** `nextn.eh_proj Q8_0 [12288→6144]`,
`nextn.enorm`, `nextn.hnorm`, `nextn.shared_head_norm` (F32). `nextn.embed_tokens` and
`nextn.shared_head_head` are not present (they would duplicate `token_embd`/`output`).
Top level: `token_embd Q8_0 [6144×154880]`, `output Q8_0`, `output_norm F32`.

Type mix across the routed experts (UD = unsloth "dynamic", per-layer choices):

| tensor family | types (layers) |
|---|---|
| `ffn_gate_exps`, `ffn_up_exps` | Q4_K (75), Q5_K (1) |
| `ffn_down_exps` | Q5_K (72), Q6_K (4) |
| every attention / shared-expert / indexer matrix | Q8_0 |
| routers, norms, `exp_probs_b`, `indexer.proj` | F32 |

## Consequences for the architecture

1. **MLA split confirmed.** No `attn_kv_b`; only the absorbed `attn_k_b` (per head
   transposed, `ne = {192, 512, 64}`) and `attn_v_b` (`{512, 256, 64}`), both `Q8_0`.
   ARCHITECTURE §7.3 policy 2 is the v1 path. Widening `attn_k_b` to f16:
   64 × 192 × 512 × 2 B = 12.6 MB per layer, **≈1.0 GB** for 79 layers.
2. **The resident dense set is 21.0 GB**, not ~10 GB: this quant keeps every attention,
   shared-expert, indexer and embedding matrix at `Q8_0`. The colibrì int4-g64
   container holds the same set in 9.9 GB. On a 25 GB host this leaves ~4 GB for the
   expert cache and KV — tight but feasible; on 16 GB it does not fit. The precision
   invariant forbids re-quantizing at load, so small hosts need a GGUF whose
   attention is Q4_K/Q5_K (e.g. `UD-Q4_K_M` / `UD-Q4_K_S`, to be inspected), or
   phase-5 `Q8_0`-on-GPU residency. Record this in REQ §10 (risk) and in `coli doctor`
   (it already fails `memory.ram` on the dense set alone).
3. **Expert bytes: 22.81 MB per expert** (gate/up Q4_K 7.08 MB each + down Q5_K 8.65 MB)
   vs 21.2 MB for int4-g64 (+7.5%). 446 GB of experts in total, 76 MoE layers × 256.
4. **MTP head is usable**: `eh_proj` is `Q8_0` (8.5 bpw) → above the FR-23 threshold.
   The MTP block's own attention/MoE tensors are plain `blk.78.*` names, as the name
   table assumes; `nextn.embed_tokens` / `shared_head_head` are absent → reuse
   `token_embd` / `output`, as today.
5. **Indexer on every layer** (no `indexer.types`): `idx_type[i] = 1` for all i when the
   key is missing and the tensors exist — the "derive from which layers carry
   `indexer.attn_k`" rule in ARCH §7.1 handles it. `indexer.proj` is F32, `attn_q_b`
   and `attn_k` are `Q8_0`.
6. **`q_lora_rank = 2048`**, `qk_nope = 192`, `v_head = 256`: the tiny fixture and the
   §7.3 size estimate used DeepSeek-like numbers; corrected here (fixture dimensions
   are arbitrary anyway, only the key names matter).
7. **Split layout**: part 1 is metadata-only (0 tensors); parts 2–11 carry ≈8 blocks
   each with their own small metadata (split keys + alignment). Tensor data offsets are
   32-byte aligned; `data_off` of the weight parts is 7–15 KB. The reader's
   "retain KV from part 0 only" rule is exactly right for this layout.
8. **`block_count` convention**: `n_layers = block_count − nextn_predict_layers`.
   Applied to `ggufinfo.summarize` (field `trunk_layers`); `ts_cfg` in phase 3 must do
   the same.

## Reproduction

```sh
B=https://huggingface.co/unsloth/GLM-5.2-GGUF/resolve/main/UD-Q4_K_XL
curl -L -o GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf $B/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf
for i in $(seq -w 2 11); do f=GLM-5.2-UD-Q4_K_XL-000$i-of-00011.gguf; curl -L -r 0-2097151 -o $f $B/$f; done
# extend parts 2..11 sparsely to the sizes the HF tree API reports, then:
./coli gguf inspect .            # or: python3 ggufinfo.py . --tensors
./coli doctor --model . --deep
./tests/test_gguf .              # raw F/K/T dump from the C reader
```
