# GGUF support — design documents

Experiment branch `claude/epic-edison-u3ncsq`. These documents define what
"GGUF support" means for colibrì and how it fits the existing engine without
changing the safetensors path or the project's precision invariant.

## Status

| Phase | Branch | State |
|---|---|---|
| 0 — documents | `claude/epic-edison-u3ncsq` | done |
| 1 — reader | `gguf/p1-reader` | implemented: `c/gguf.h`, `c/ggufinfo.py`, `c/tools/make_gguf_fixture.py`, `tests/test_gguf.c`, `tests/test_ggufinfo.py`, `coli gguf inspect`, `coli doctor` GGUF checks. No engine wiring. Verified on `unsloth/GLM-5.2-GGUF` `UD-Q4_K_XL` (11 parts, 1809 tensors): [inspection report](inspection-glm52-ud-q4_k_xl-2026-10-05.md). |
| 2 — kernels | — | not started |
| 3 — assembly | — | not started |
| 4 — streaming | — | not started |

Try phase 1 on any GGUF (no model weights are read, only headers):

```sh
./coli gguf inspect /path/to/model-00001-of-00009.gguf      # or the directory, or a single file
./coli doctor --model /path/to/model.gguf --deep
make -C c tests/test_gguf && ./c/tests/test_gguf /path/to/model.gguf   # raw dump from the C reader
```

| Document | Content |
|---|---|
| [REQUIREMENTS.md](REQUIREMENTS.md) | Decisions taken, scope, functional and non-functional requirements (FR/NFR), compatibility matrix, acceptance criteria, risks, open questions. |
| [ARCHITECTURE.md](ARCHITECTURE.md) | As-is seams in `c/colibri.c`, the new components (`gguf.h`, `gq.h`, `src.h`, `glm_names.h`, `ggufinfo.py`), `glm-dsa` name/metadata mapping, expert streaming from 3-D tensors, Python tooling, testing strategy, phased plan. |

Summary of the decisions:

- **Load GGUF directly** in the GLM-5.2 engine (no converter, no exporter).
- **v1 quant types:** `F32`, `F16`, `BF16`, `Q4_0`, `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K`,
  computed natively on the ggml block layouts — never re-quantized.
- **Pure C, zero dependencies:** own reader and kernels; no `ggml`/`llama.cpp` code.
- **Phased delivery** (reader → kernels → assembly → streaming → GPU → breadth),
  each phase `make check`-green and oracle-green, measured per `docs/benchmarks.md`.
