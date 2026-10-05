# GGUF support for colibrì — requirements

Status: **draft for review** (experiment branch `claude/epic-edison-u3ncsq`).
Companion document: [ARCHITECTURE.md](ARCHITECTURE.md). Decisions already taken
by the maintainer of this branch are marked **[decided]**; everything else is a
proposal and open to change.

## 1. Purpose

Let the GLM-5.2 engine (`c/colibri.c`) run **directly from GGUF files produced
by the llama.cpp ecosystem**, streaming routed experts from the GGUF exactly the
way it streams them from the colibrì safetensors container today: per-layer
LRU, learned pin set, one-layer-ahead prefetch, VRAM/RAM/NVMe placement,
dual-SSD mirror and N-drive split.

Why this matters:

- **No 756 GB conversion.** Today a user needs either the 372 GB pre-converted
  colibrì container or a one-time FP8→int4 conversion that touches the entire
  FP8 checkpoint. GGUF builds of GLM-5.2 (e.g. `unsloth/GLM-5.2-GGUF`,
  `sunshaohui/GLM-5.2-GGUF`) are already downloaded by thousands of people.
- **One model file for several engines.** A GGUF that already serves llama.cpp,
  Ollama or LM Studio can serve colibrì too; the research question "placement
  vs. speed on the same bytes" becomes a controlled A/B between engines.
- **Quantization research surface.** K-quants (`Q4_K`, `Q5_K`, `Q6_K`) are a
  well-studied family with public quality numbers. Running them unmodified in
  colibrì's tiering lets the project measure "bytes moved per useful token" on
  formats the community already trusts, next to the in-house fmt 1–6 family.

## 2. Decisions already taken [decided]

| # | Decision | Consequence |
|---|---|---|
| D1 | **Direction: load GGUF directly in the engine.** Not an offline GGUF→safetensors converter, not a GGUF exporter. | New reader + new kernels in C. The offline-converter and export directions are explicitly out of scope (see §5). |
| D2 | **Quant coverage for v1: `F32`, `F16`, `BF16`, `Q4_0`, `Q8_0` *and* the K-quants `Q4_K`, `Q5_K`, `Q6_K`.** | Covers the common Hugging Face uploads (`Q4_K_M`, `Q5_K_M`, `Q6_K`, `Q8_0`, `UD-Q4_K_XL`). I-quants, `Q2_K`/`Q3_K`, ternary and `MXFP4` are later phases. |
| D3 | **Pure C, zero dependencies stays.** No linking or vendoring of `ggml`/`llama.cpp`. | Own GGUF parser (`gguf.h`), own block-format kernels (`gq.h`). Reimplementing a documented block layout is fine (the IQ3 grid in `quant.h` already does this, MIT-attributed); importing library code is not. |
| D4 | **Deliverable of this step: documents only, in English, under `docs/gguf/`.** | No engine code on this branch yet. The phase plan in ARCHITECTURE.md §12 is the proposal for the implementation branches. |
| D5 | **Experiment branch.** Everything lands on `claude/epic-edison-u3ncsq` (and children), never directly on `main`/`dev`. | `main` keeps its oracle gate untouched; promotion to upstream `dev` is a separate decision once Phase 4 measurements exist. |

## 3. Background: what exists today

- The engine indexes **`*.safetensors` shards** through `c/st.h` (`st_init_multi`,
  `st_find`, `st_read_raw`, mirror/split replicas, `O_DIRECT` twins) and reads
  `config.json`, `generation_config.json` and `tokenizer.json` from the same
  directory. There is no other model source.
- Quantized weights live in a **colibrì-specific layout**: a `U8` tensor with the
  original HF name plus a `<name>.qs` f32 scale sidecar; the format (`fmt`
  1–6) is *inferred* from the byte counts (`qt_resolve_fmt`). Scales are stored
  **outside** the weight bytes (fmt 1–5) or in-block (fmt 6).
- Each routed expert's three matrices are **adjacent in the file**, so one
  expert is **one `pread`** into a slab; the `QT` structs are views into that
  slab. The O_DIRECT path reads a 4 KiB-aligned window into the slab.
- The GPU backends (`backend_cuda.cu`, `backend_metal.mm`) accept `fmt` 0–4 (+6
  on CUDA); unknown formats fall back to CPU.
- The Python side (`c/coli`, `c/doctor.py`, `c/resource_plan.py`,
  `c/openai_server.py`) assumes a **directory** with `config.json` +
  `*.safetensors`, and picks the engine from `model_type`.
- Project invariants (README, CONTRIBUTING): dependency-free default CPU build;
  **placement never changes model precision or router semantics**; every change
  keeps the token-exact oracle (`32/32 TF + 20/20 greedy`) green; `make check`
  must run without a model download.

## 4. Glossary

| Term | Meaning |
|---|---|
| GGUF | Single-file (optionally split) container of ggml: header, key/value metadata, tensor table, aligned tensor data. Version 3. |
| ggml type | Per-tensor element type (`GGML_TYPE_*`), e.g. `Q4_K = 12`, `Q6_K = 14`, `Q8_0 = 8`. One GGUF mixes several types. |
| block / super-block | Smallest quantization unit: 32 elements (`Q4_0`, `Q8_0`) or 256 elements (K-quants). Scales live **inside** the block. |
| `glm-dsa` | llama.cpp architecture name for GLM-5 / GLM-5.2 (HF class `GlmMoeDsaForCausalLM`). GLM-4.5/4.6 are `glm4moe` and are **not** this model family. |
| `blk.N.*` | GGUF tensor naming (`blk.N.ffn_gate_exps.weight`, …). |
| NextN | llama.cpp's name for the MTP head; stored as block `n_layer` with `nextn.*` extras. |
| Indexer | GLM-5.2's DSA lightning indexer (`blk.N.indexer.*`), present only on "full" layers. |
| Colibrì container | The project's own safetensors layout (`U8` + `.qs`), fmt 1–6. |

## 5. Scope

### In scope (v1, phases 1–4)

- Load GLM-5.2 (`general.architecture == "glm-dsa"`) from a GGUF file or a split
  GGUF set, with the quant types of D2, on the dependency-free CPU path.
- Streaming of routed experts from the GGUF with all existing placement
  features (LRU cache, `PIN`/`AUTOPIN`, `PILOT`, `PIPE`, `URING`, `DIRECT`,
  `COLI_MMAP`, `COLI_MODEL_MIRROR`, `COLI_MODEL_DIRS`).
- Config, tokenizer and stop tokens taken **from the GGUF metadata**; HF sidecar
  files optional, never required.
- `coli chat/serve/web/run/info/plan/doctor` working unchanged for a GGUF model
  (`COLI_MODEL` may point at a `.gguf` file or at a directory containing one).
- MTP (NextN) and the DSA indexer loaded from the GGUF **when present and when
  their precision allows** (see FR-23/24).
- GPU backends: CPU fallback for all GGML formats in v1; native CUDA kernels for
  `Q4_K`/`Q6_K`/`Q8_0` in Phase 5.

### Out of scope (for now)

- Converting GGUF to the colibrì container, or exporting colibrì containers to
  GGUF (rejected directions, D1).
- I-quants (`IQ2_*`, `IQ3_*`, `IQ4_*`), `Q2_K`, `Q3_K`, `Q4_1`, `Q5_0`, `Q5_1`,
  `TQ1_0`/`TQ2_0`, `MXFP4` — Phase 6 candidates; v1 must **refuse** such tensors
  with a precise message, never guess.
- Other model families (`glm4moe`, `deepseek2`, `olmoe`, `kimi`, Inkling) in
  the engine. The reader and kernels are family-agnostic by construction, and
  OLMoE-GGUF is proposed as a *test vehicle* (ARCHITECTURE.md §11.4), but no
  product commitment.
- Vision towers, multimodal projector tensors (`mmproj`).
- Writing GGUF (no `gguf_write`).
- llama.cpp feature parity (sampling, grammar dialects, server API).

## 6. Functional requirements

Priority: **MUST** (v1 acceptance), **SHOULD** (v1 if cheap, else Phase 5),
**MAY** (later). Phase numbers refer to ARCHITECTURE.md §12.

### 6.1 Container reading

| ID | Requirement | Prio | Phase |
|---|---|---|---|
| FR-1 | Parse GGUF **version 3** headers: magic `GGUF`, tensor count, KV count, all 13 metadata value types including nested arrays; honour `general.alignment` (default 32). Reject version ≠ 3 with a message naming the version. | MUST | 1 |
| FR-2 | Treat the file as **untrusted input**: bounded string lengths, bounded array lengths, overflow-checked dimension products, `offset + nbytes <= data_size` for every tensor, `ne[0] % block_size == 0` for block types, alignment power-of-two. Any violation exits with the tensor/key name (same posture as `qt_resolve_fmt` and `cfg_root`). | MUST | 1 |
| FR-3 | Support **split GGUF** sets (`<name>-00001-of-0000N.gguf`): validate `split.no`/`split.count`/`split.tensors.count`, load every part, resolve each tensor to `(file, absolute offset)`. Parts may live on different drives (`COLI_MODEL_DIRS`). | MUST | 1 |
| FR-4 | Build the same name→tensor hash index the safetensors path has (`st_find` cost ≈ O(1); GLM-5.2 has ~100k logical experts but only ~1.5k GGUF tensors, so this is cheap). | MUST | 1 |
| FR-5 | Expose per-tensor: name, ggml type, `n_dims`, `ne[0..3]`, byte size (`ne[1..] × row_size(type, ne[0])`), file descriptor(s), absolute offset. | MUST | 1 |
| FR-6 | Index both model sources behind one **tensor-source interface** so engine code never asks "is this safetensors or GGUF?" outside the loader. | MUST | 3 |
| FR-7 | Dual-SSD mirror (`COLI_MODEL_MIRROR`): a mirror part is accepted only if its size and its header+KV+tensor-table bytes are identical to the primary (GGUF analogue of today's header comparison). | MUST | 4 |

### 6.2 Quantized formats

| ID | Requirement | Prio | Phase |
|---|---|---|---|
| FR-8 | Compute **natively on GGML block layouts** for `F32`, `F16`, `BF16`, `Q4_0`, `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K`: dequantize-on-the-fly GEMV/GEMM with f32 activations. **No re-quantization at load** (would be double quantization; forbidden by the precision invariant). Widening (e.g. `F16`→f32 for a tiny norm vector) is allowed. | MUST | 2 |
| FR-9 | Per-tensor type dispatch: a single GGUF mixes types (e.g. `Q4_K_M` uses `Q6_K` for some `ffn_down_exps` and `Q8_0`/`F32` for small tensors). The engine must never assume one type per file or per layer. | MUST | 2 |
| FR-10 | Unsupported types are **refused at load** with the tensor name and type name (`blk.12.ffn_down_exps.weight: IQ3_XXS is not supported yet`), never silently approximated. | MUST | 2 |
| FR-11 | Scalar reference kernels for every supported type, plus SIMD paths for AVX2 and NEON (AVX-512/VNNI/i8mm optional). Scalar and SIMD results must agree within the tolerance used by the existing `i4_acc512_selftest`. | MUST | 2 |
| FR-12 | A fused gate+up path for same-type expert pairs (today `matmul_i4_pair` / `matmul_i4_grouped_pair`), so GGUF experts do not lose the `S==1` fusion. | SHOULD | 2 |
| FR-13 | An optional **int8-activation dot path** for K-quants (`Q8_K`-style activation blocks, the analogue of today's `IDOT`), behind the existing `IDOT` knob, *measured* for quality like the original (+0.117 nat/token note in `matmul_qt_ex`). | MAY | 5 |
| FR-14 | Each GGUF-backed `QT` reports its resident bytes (`qt_bytes`) and the planner reports bytes per expert from the true row sizes, so `coli plan` / `RAM_GB` budgets stay correct. | MUST | 2–3 |

### 6.3 Model assembly (GLM-5.2 / `glm-dsa`)

| ID | Requirement | Prio | Phase |
|---|---|---|---|
| FR-15 | Fill `Cfg` from GGUF metadata (`glm-dsa.embedding_length`, `block_count`, `expert_count`, `expert_used_count`, `expert_feed_forward_length`, `feed_forward_length`, `leading_dense_block_count`, `expert_shared_count`, `expert_weights_scale`, `expert_weights_norm`, `expert_gating_func`, `attention.head_count`, `attention.q_lora_rank`, `attention.kv_lora_rank`, `attention.key_length_mla`, `attention.value_length_mla`, `rope.dimension_count`, `rope.freq_base`, `attention.layer_norm_rms_epsilon`, indexer keys, `nextn_predict_layers`). The same `CKR` range validation as `load_cfg` applies. | MUST | 3 |
| FR-16 | Map every HF tensor name the engine uses (`model.layers.%d.self_attn.q_a_proj.weight`, …) to its GGUF name (`blk.%d.attn_q_a.weight`, …) in **one table**, so `model_init` keeps its HF vocabulary and the mapping is reviewable in one place. | MUST | 3 |
| FR-17 | Locate expert `e` of layer `L` as **three slices of three 3-D tensors** (`blk.L.ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`; shape `{ne0, ne1, n_expert}`), each slice contiguous: `offset + e × ne1 × row_size(type, ne0)`. | MUST | 3–4 |
| FR-18 | Reconcile MLA weights: GGUF `glm-dsa` files carry the **absorbed split** `attn_k_b` (`{qk_nope, kv_lora, n_head}`, per-head transposed) and `attn_v_b` (`{kv_lora, v_head, n_head}`) and may or may not carry the fused `attn_kv_b`. The engine must work with either, without narrowing precision (see ARCHITECTURE.md §7.3). | MUST | 3 |
| FR-19 | Build the tokenizer from `tokenizer.ggml.tokens`, `merges`, `token_type`, `bos/eos_token_id`, `tokenizer.ggml.pre` (→ pre-tokenizer family `cl100k` for GLM) **in memory**, reusing `tok.h`'s structures. A `tokenizer.json` next to the GGUF, if present, may be preferred but is never required. | MUST | 3 |
| FR-20 | Stop tokens: union of `tokenizer.ggml.eos_token_id`, `eot_token_id` if present, and the GLM control tokens resolved by name (`<|user|>`, `<|observation|>`, `<|endoftext|>`), mirroring today's `config.json` ∪ `generation_config.json` union. | MUST | 3 |
| FR-21 | `coli` chooses the engine from `general.architecture`: `glm-dsa` → `colibri`; `glm4moe`, `deepseek2`, anything else → explicit "not supported by this engine" (no silent GLM fallback as today's `model_arch()` does for unknown `model_type`). | MUST | 3 |
| FR-22 | Lossless end-to-end gate: a `glm-dsa` GGUF written at `F32`/`F16` from the tiny oracle snapshot (`c/glm_tiny`) reproduces `ref_glm.json` **32/32 TF and 20/20 greedy**, exactly like the safetensors path. | MUST | 3 |
| FR-23 | **MTP/NextN**: load `blk.<n_layer>.*` + `blk.<n_layer>.nextn.*` as the MTP layer when the complete set is present. If `eh_proj` or the NextN attention tensors are stored below 8 bits (`Q4_*`, `Q5_*`, `Q6_K`), disable MTP with a one-line warning citing issue #8 (int4 heads → ~0% acceptance) unless `MTP=1` is forced. | MUST | 4 |
| FR-24 | **DSA indexer**: load `blk.N.indexer.{attn_q_b, attn_k, proj, k_norm}` for the layers that have them; activate `has_dsa` under the same rules as today (`index_topk>0`, all full layers present, `DSA=0` disables). Indexer-less GGUFs run dense MLA as today's non-`--indexer` containers do. | SHOULD | 4 |
| FR-25 | Routed-expert count may be smaller than 256 (REAP-pruned GGUFs); nothing may hard-code 256 (the engine already probes `n_experts-1`). | MUST | 3 |

### 6.4 Streaming and placement

| ID | Requirement | Prio | Phase |
|---|---|---|---|
| FR-26 | Expert load = up to **three reads** (gate/up/down slices, possibly in different split files) into the existing slab; `QT` views point into the slab; `fslab` unused for in-block-scale types. Order reads by file offset as today. | MUST | 4 |
| FR-27 | `DIRECT=1`: per-slice 4 KiB-aligned window reads (GGUF offsets are only 32-byte aligned). Keep the slab slack (`+8192`) rule. | MUST | 4 |
| FR-28 | `COLI_MMAP=1`: views into mapped parts at the slice offsets, same pre-touch and Metal registration behaviour. | SHOULD | 4 |
| FR-29 | `URING=1`: one SQE per slice; the "requires quantized expert tensors" check becomes "requires a locatable expert", source-agnostic. | SHOULD | 4 |
| FR-30 | `PILOT`/readahead/`PREFETCH` issue `WILLNEED` per slice on the replica that will serve the read. | SHOULD | 4 |
| FR-31 | `PIN`, `AUTOPIN`, `REPIN`, `.coli_usage`, `.coli_kv`, `STATS`, the web **Brain/Atlas** pages work unchanged: expert identity is `(layer, eid)`, independent of the source. Sidecar files get a GGUF-specific location (ARCHITECTURE.md §9.3) so two GGUFs in one directory never share usage history or KV state. | MUST | 4 |
| FR-32 | VRAM expert tier (`CUDA_EXPERT_GB`): experts in GGML formats stay on the CPU tier in v1 (`coli_cuda_tensor_upload` refuses unknown `fmt` → CPU, as it does for fmt 5). Dense tensors likewise. Native CUDA kernels for `Q4_K`/`Q6_K`/`Q8_0` are Phase 5. | MUST (fallback) / SHOULD (kernels) | 4 / 5 |
| FR-33 | Metal: same CPU fallback in v1; batched Metal `moe_gemv` for K-quants is Phase 5+. | MAY | 5+ |

### 6.5 Tooling and UX

| ID | Requirement | Prio | Phase |
|---|---|---|---|
| FR-34 | A **stdlib-only Python GGUF reader** (`c/ggufinfo.py`) shared by `coli`, `doctor.py`, `resource_plan.py` and the gateway: header, KV, tensor table, per-type sizes, split handling. No `gguf-py`, no numpy (the `make check` Python tests are stdlib-only). | MUST | 1 |
| FR-35 | `coli doctor` gains checks `model.gguf.header`, `model.gguf.splits`, `model.gguf.types` (lists unsupported types), `model.gguf.arch`, `model.gguf.mtp_precision`; `--deep` verifies every tensor's `offset+nbytes` against the file size. | MUST | 1–3 |
| FR-36 | `coli info` / `coli plan` show dense vs expert bytes, per-expert bytes, the type mix (`Q4_K 71% · Q6_K 27% · F32 2%`) and the `general.name`/`general.file_type`. | SHOULD | 3 |
| FR-37 | `coli convert` is untouched. A new `coli gguf inspect <file>` (or `coli info --gguf`) prints the metadata/tensor table for debugging. | MAY | 1 |
| FR-38 | The web dashboard's model line shows the GGUF `general.name` and the type mix instead of "int4". | MAY | 4 |

## 7. Non-functional requirements

| ID | Requirement |
|---|---|
| NFR-1 | **Zero behaviour change for safetensors models.** The oracle gate (`SNAP=./glm_tiny TF=1 ./colibri 64 16 16` → 32/32, 20/20) and every existing C/Python test stay green at every phase. The safetensors loader keeps its code path; the tensor-source interface wraps it rather than rewriting it. |
| NFR-2 | **Dependency-free.** `make portable` still needs only a C compiler with OpenMP; `make check` still needs only the Python standard library. Test fixtures are generated by in-repo stdlib Python or checked in as small binaries with provenance (as `tests/fixtures/e8_case.bin`). |
| NFR-3 | **Precision invariant.** A weight byte read from a GGUF is used as the author quantized it. Permitted transformations: widening to f32/f16 for layout reasons (FR-18), activation quantization behind an explicit knob (FR-13). Forbidden: silent re-quantization, silent type substitution, dropping tensors the HF model uses. |
| NFR-4 | **Untrusted input.** Hostile GGUF files must terminate with a diagnostic, never read out of bounds, over-allocate or hang (fuzz the reader as `tests/fuzz_rans.c` fuzzes rANS). |
| NFR-5 | **Performance target (hypothesis, to measure).** On a disk-bound host, decode speed from a `Q4_K_M` GGUF is within **±10%** of the gs64 int4 container at equal bytes per expert (both ≈4.5 bpw). On a compute-bound host (full residency), CPU `Q4_K` GEMV may cost up to ~1.5× the fmt=4 kernel until the int8-activation path (FR-13) lands; this is recorded as a measured number, not hidden. |
| NFR-6 | **Memory.** Resident RSS for a GGUF model ≤ resident RSS of the equivalent container + 2% (index tables, tokenizer). Expert slabs are sized from true slice bytes (`Q6_K` down-projections are 1.46× `Q4_K`). |
| NFR-7 | **Startup.** Indexing a 400 GB split GGUF (reading only headers and tensor tables) ≤ 2 s on NVMe; tokenizer construction from ~155k tokens ≤ 1 s. |
| NFR-8 | **Portability.** Linux, macOS, Windows (MinGW) via `compat.h`; little-endian only (GGUF is little-endian; refuse to compile on big-endian as `rans.h` does). |
| NFR-9 | **Code shape.** New code lives in new headers (`gguf.h`, `gq.h`, `src.h`) plus small, reviewable edits at the existing seams in `colibri.c` (`model_init`, `qt_from_disk`, `expert_load_impl`, `uring_load_add`, `matmul_qt_ex`, `qt_bytes`, `expert_gate_up`). No second engine file. |
| NFR-10 | **Documentation.** `docs/ENVIRONMENT.md` and `docs/SETTINGS.md` updated for every new knob (per `docs/MAINTAINING-DOCS.md`); a user page `docs/gguf.md` once Phase 4 runs end to end; CHANGELOG entry under *Unreleased*. |
| NFR-11 | **Licensing.** Block layouts are re-implemented from the public specification/`ggml-common.h` definitions; any table copied verbatim (e.g. a non-linear codebook) carries the same MIT attribution `quant.h` uses for the IQ3 grid. |

## 8. Compatibility matrix (target state after Phase 4; Phase 5 items in italics)

| ggml type | bpw | CPU exact | CPU IDOT | fused gate+up | CUDA | Metal | mmap | uring | DIRECT |
|---|---|---|---|---|---|---|---|---|---|
| `F32` | 32 | ✓ (existing `matmul`) | – | – | *✓ (fmt 0)* | – | ✓ | ✓ | ✓ |
| `F16` / `BF16` | 16 | ✓ | – | – | *P5* | – | ✓ | ✓ | ✓ |
| `Q8_0` | 8.5 | ✓ | *P5* | – | *P5* | – | ✓ | ✓ | ✓ |
| `Q4_0` | 4.5 | ✓ | *P5* | ✓ | *P5* | – | ✓ | ✓ | ✓ |
| `Q4_K` | 4.5 | ✓ | *P5* | ✓ | *P5* | *P6* | ✓ | ✓ | ✓ |
| `Q5_K` | 5.5 | ✓ | *P5* | ✓ | *P5* | – | ✓ | ✓ | ✓ |
| `Q6_K` | 6.56 | ✓ | *P5* | ✓ | *P5* | – | ✓ | ✓ | ✓ |
| `Q2_K`, `Q3_K`, `IQ*`, `TQ*`, `MXFP4` | – | refused (FR-10) | | | | | | | |

"–" = CPU fallback for that tensor; the run continues and the startup log says which tensors stayed on CPU and why (today's `[CUDA] … disabled` convention).

## 9. Acceptance criteria

v1 is accepted when all of the following hold on the experiment branch:

1. **`make check` green** on Linux, macOS and Windows CI, including the new unit
   tests (`test_gguf`, `test_gq_kernels`, `test_gguf_load`, Python
   `test_ggufinfo`, `test_doctor` GGUF cases).
2. **Lossless oracle**: FR-22 passes (32/32 TF, 20/20 greedy) from an `F16`
   GGUF of `glm_tiny`; the safetensors oracle still passes.
3. **Kernel exactness**: for every supported type, the C dequantization of a
   random tensor equals the in-repo Python reference dequantizer bit-for-bit;
   SIMD and scalar GEMV agree within tolerance; `Q4_K`/`Q5_K`/`Q6_K` fixtures
   produced by `llama-quantize` from the `F16` tiny GGUF dequantize to the same
   values llama.cpp's own `dequantize_row_*` produces (fixtures checked in with
   their generating command).
4. **Real model**: `coli chat`, `coli serve`, `coli web`, `coli doctor --deep`
   run GLM-5.2 from a public `Q4_K_M`-class GGUF on at least one Linux host and
   one macOS host; the per-turn stats line shows hit rate, GB read and
   `MIRROR:`/`SPLIT` lines as for the container.
5. **A/B published** (per `docs/benchmarks.md` protocol): same host, same
   prompt set, cold and warm cache, GGUF `Q4_K_M` vs gs64 int4 container:
   tok/s, TTFT, expert hit rate, bytes read/token, RSS, plus a quality probe
   (the `coli bench` MMLU/HellaSwag subset or the `SCORE` log-likelihood path).
   Negative results are published too.
6. **No regression** in the safetensors path on that A/B host (tok/s within
   noise of the pre-branch commit).

## 10. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| `glm-dsa` GGUFs may lack `attn_kv_b` and ship only the absorbed `attn_k_b`/`attn_v_b` (per-head transposed). | Attention path mismatch. | FR-18; ARCHITECTURE §7.3 reconciles by widening `attn_k_b` to f16/f32 at load (≈0.6–1.3 GB resident for 78 layers) in v1, transposed kernel later. Verify on the first real file (`coli gguf inspect`). |
| MTP head quantized too coarsely in public GGUFs. | Draft acceptance collapses (#8). | FR-23: auto-disable below 8 bits with a warning; measure acceptance in the A/B. |
| llama.cpp's `glm-dsa` indexer/tensor set is still moving (indexer runtime landed after the loader; metadata keys changed once). | Files in the wild differ by converter version. | Reader tolerates missing optional keys with defaults; `doctor` prints `general.*` provenance; the name table is one place to patch. |
| Public quants keep the dense set at `Q8_0` (`UD-Q4_K_XL`: 21.0 GB resident vs 9.9 GB for the int4 container). | Small hosts cannot hold the dense set; re-quantizing is forbidden by NFR-3. | `coli doctor` fails `memory.ram` on the dense set alone (done, phase 1); inspect `UD-Q4_K_M`/`UD-Q4_K_S` for a 4–5-bit attention set; phase 5 can keep `Q8_0` dense tensors on the GPU. See the [inspection report](inspection-glm52-ud-q4_k_xl-2026-10-05.md). |
| K-quant dequant cost on CPU-bound hosts. | Slower than fmt=4 at full residency. | NFR-5 measured honestly; FR-13 int8-activation path in Phase 5; fused pair (FR-12). |
| Three reads per expert instead of one, possibly across split files. | More IOPS, worse O_DIRECT efficiency on some drives. | Offset-ordered reads, `PIPE`/`URING` batching per slice; measure on NVMe; an optional on-disk "expert index" cache is a Phase 6 idea, not v1. |
| Tokenizer drift (byte-level BPE from GGUF arrays vs HF `tokenizer.json`). | Wrong tokens → wrong outputs. | Test: encode/decode of `tests/tok_o200k_cases.txt`-style corpus must match the `tokenizer.json` path on the same model; prefer the HF file when present. |
| Scope creep toward other architectures. | Never finishing GLM-5.2. | Family-agnostic reader, GLM-only assembly; OLMoE-GGUF only as a test vehicle. |

## 11. Open questions for the maintainer

1. **Which public GGUF is the reference artefact** for the A/B? Proposal:
   `unsloth/GLM-5.2-GGUF` `UD-Q4_K_XL` — inspected on 2026-10-05: 11 parts,
   1809 tensors, type mix `Q4_K`/`Q5_K`/`Q6_K`/`Q8_0`/`F32` only (all in the v1
   set), MTP head `Q8_0`, indexer on every layer, dense set 21.0 GB
   ([report](inspection-glm52-ud-q4_k_xl-2026-10-05.md)). Open: whether a
   lighter dense set (`UD-Q4_K_M`) should be the reference for ≤25 GB hosts.
2. **MTP policy**: auto-disable below 8 bits (FR-23) or hard-refuse? Proposal:
   warn + disable, `MTP=1` forces.
3. **Indexer in v1?** Proposal: load if present (SHOULD), otherwise dense MLA;
   not an acceptance blocker.
4. **Sidecar location** for `.coli_usage`/`.coli_kv` next to a GGUF
   (ARCHITECTURE §9.3 proposes `<dir>/.coli-<basename>/`).
5. **Upstream intent**: should the phases be shaped as PRs against
   `JustVugg/colibri` `dev` from the start (small, each with `make check`), or
   as one experiment branch merged later? The architecture assumes PR-sized
   phases either way.

## 12. Assumptions

- GLM-5.2 is the only engine target for v1; `olmoe.c`, `inkling.c`, `kimi_k3.c`
  are untouched.
- GGUF files are produced by llama.cpp's converter/quantizer (`glm-dsa`
  architecture, GGUF v3, `general.quantization_version = 2`); hand-made or
  third-party writers are supported only insofar as they follow the spec.
- Tensor orientation follows ggml: `ne[0]` is the input (fastest) dimension, so
  a 2-D GGUF tensor is row-major `[O][I]` exactly like colibrì's `QT`.
- Expert tensors are 3-D `{ne0, ne1, n_expert}` with expert-major contiguous
  slices (confirmed from `llama-model` shapes
  `ffn_gate_exps {n_embd, n_ff_exp, n_expert}`,
  `ffn_down_exps {n_ff_exp, n_embd, n_expert}`).
- GLM-5.2 dimensions divide by 256 where K-quants are used; where they do not
  (e.g. `attn_k_b` with `ne0 = qk_nope`), `llama-quantize` already stored a
  compatible type, and FR-9/FR-10 handle whatever is there.

## Sources consulted

- GGUF specification: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
- Block layouts: `ggml/src/ggml-common.h` in https://github.com/ggml-org/llama.cpp
- `glm-dsa` tensor shapes and keys: `src/models/glm-dsa.cpp`, `src/llama-arch.cpp` (llama.cpp master, 2026-10)
- Split handling: `src/llama-model-loader.cpp`
- GLM-5.2 GGUF availability: https://huggingface.co/unsloth/GLM-5.2-GGUF, https://huggingface.co/sunshaohui/GLM-5.2-GGUF
- In-repo: `c/st.h`, `c/colibri.c` (`load_cfg`, `qt_resolve_fmt`, `qt_from_disk`, `model_init`, `expert_load_impl`, `uring_load_add`, `matmul_qt_ex`), `c/quant.h`, `c/tok.h`, `c/coli`, `c/doctor.py`, `c/resource_plan.py`, `CONTRIBUTING.md`, `docs/benchmarks.md`
