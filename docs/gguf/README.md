# GGUF support — design documents

Experiment branch `claude/epic-edison-u3ncsq`. Nothing here is implemented yet;
these documents define what "GGUF support" means for colibrì and how it fits
the existing engine without changing the safetensors path or the project's
precision invariant.

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
