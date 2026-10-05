#!/usr/bin/env python3
"""Dependency-free GGUF v3 reader for colibri's Python tools.

Used by `coli gguf inspect`, `coli doctor` and (from phase 3 on) `coli info/plan`
and the API gateway. Reads ONLY the metadata region of each part (header,
key/value pairs, tensor table) — never the tensor payload — and applies the same
bounds as the C reader in `gguf.h`, so a file the engine would refuse is
reported here with the same message shape.

    python3 ggufinfo.py <model.gguf | dir> [--tensors] [--json]

Standard library only: `make check` must run without numpy or gguf-py.
"""
import json
import os
import re
import statistics
import struct
import sys
from pathlib import Path

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
DEFAULT_ALIGNMENT = 32
MAX_STR = 64 << 10
MAX_TENSORS = 1 << 20
MAX_KV = 1 << 16
MAX_ARR = 1 << 24
MAX_META = 1 << 30
MAX_ALIGN = 1 << 20
MAX_DIMS = 4
MAX_SPLITS = 512

# metadata value types: id -> (name, struct format or None)
VALUE_TYPES = {
    0: ("u8", "<B"), 1: ("i8", "<b"), 2: ("u16", "<H"), 3: ("i16", "<h"),
    4: ("u32", "<I"), 5: ("i32", "<i"), 6: ("f32", "<f"), 7: ("bool", "<?"),
    8: ("str", None), 9: ("arr", None), 10: ("u64", "<Q"), 11: ("i64", "<q"), 12: ("f64", "<d"),
}
T_STR, T_ARR = 8, 9

# ggml tensor types: id -> (name, elements per block, bytes per block)
GGML_TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292), 16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98), 19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136), 24: ("I8", 1, 1), 25: ("I16", 1, 2),
    26: ("I32", 1, 4), 27: ("I64", 1, 8), 28: ("F64", 1, 8), 29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2), 34: ("TQ1_0", 256, 54), 35: ("TQ2_0", 256, 66), 39: ("MXFP4", 32, 17),
}
TYPE_IDS = {name: tid for tid, (name, _, _) in GGML_TYPES.items()}
# what the v1 engine will compute on (docs/gguf/REQUIREMENTS.md D2)
V1_SUPPORTED = {TYPE_IDS[n] for n in ("F32", "F16", "BF16", "Q4_0", "Q8_0", "Q4_K", "Q5_K", "Q6_K")}
# architectures the GLM-5.2 engine assembles (phase 3); everything else is refused loudly
ENGINE_ARCHS = {"glm-dsa": "glm"}

SPLIT_RE = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$")
EXPERT_RE = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")
INDEXER_RE = re.compile(r"^blk\.(\d+)\.indexer\.attn_k\.weight$")


class GgufError(ValueError):
    """A file the engine would refuse; the message names the key/tensor and the rule."""


def type_name(tid):
    return GGML_TYPES[tid][0] if tid in GGML_TYPES else f"unknown({tid})"


def row_size(tid, ne0):
    """Bytes of one row of ne0 elements, or None if the type is unknown / ne0 is not whole blocks."""
    if tid not in GGML_TYPES:
        return None
    _, block, tsize = GGML_TYPES[tid]
    if ne0 < 0 or ne0 % block:
        return None
    return ne0 // block * tsize


def bits_per_weight(tid):
    if tid not in GGML_TYPES:
        return None
    _, block, tsize = GGML_TYPES[tid]
    return 8.0 * tsize / block


class TensorInfo:
    __slots__ = ("name", "type", "ne", "nbytes", "rel_off", "off", "file")

    def __init__(self, name, tid, ne, nbytes, rel_off, off, file):
        self.name, self.type, self.ne = name, tid, ne
        self.nbytes, self.rel_off, self.off, self.file = nbytes, rel_off, off, file

    @property
    def type_name(self):
        return type_name(self.type)

    def as_dict(self):
        return {"name": self.name, "type": self.type, "type_name": self.type_name, "ne": list(self.ne),
                "nbytes": self.nbytes, "offset": self.off, "file": self.file}


class GgufPart:
    def __init__(self, path):
        self.path = str(path)
        self.size = 0
        self.version = 0
        self.alignment = DEFAULT_ALIGNMENT
        self.data_off = 0
        self.kv = {}            # key -> python value (arrays -> list; nested arrays -> None)
        self.kv_types = {}      # key -> (type, array element type or None, length)
        self.tensors = []
        self.split_no = None
        self.split_count = None
        self.split_tensors = None


class _Reader:
    """Sequential bounded reader over the metadata region of one part."""

    def __init__(self, fh, size, path):
        self.fh, self.size, self.path, self.pos = fh, size, path, 0

    def read(self, n, what="metadata"):
        if n < 0 or n > self.size - self.pos:
            raise GgufError(f"{self.path}: {what} runs past the end of the file "
                            f"(need {n} bytes at {self.pos}, file is {self.size})")
        data = self.fh.read(n)
        if len(data) != n:
            raise GgufError(f"{self.path}: short read in {what} at {self.pos} — truncated file?")
        self.pos += n
        return data

    def u32(self):
        return struct.unpack("<I", self.read(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.read(8))[0]

    def string(self, what):
        n = self.u64()
        if n > MAX_STR:
            raise GgufError(f"{self.path}: {what} is {n} bytes long (limit {MAX_STR})")
        raw = self.read(n, what)
        if b"\0" in raw:
            raise GgufError(f"{self.path}: {what} contains a NUL byte")
        return raw.decode("utf-8", errors="replace")

    def value(self, vtype, key, depth=0):
        if vtype not in VALUE_TYPES:
            raise GgufError(f"{self.path}: key {key} has unknown value type {vtype}")
        if vtype == T_STR:
            return self.string(key), None, 1
        if vtype != T_ARR:
            fmt = VALUE_TYPES[vtype][1]
            return struct.unpack(fmt, self.read(struct.calcsize(fmt), key))[0], None, 1
        atype, n = self.u32(), self.u64()
        if atype not in VALUE_TYPES:
            raise GgufError(f"{self.path}: array {key} has unknown element type {atype}")
        if n > MAX_ARR:
            raise GgufError(f"{self.path}: array {key} has {n} elements (limit {MAX_ARR})")
        if atype == T_ARR:
            if depth >= 1:
                raise GgufError(f"{self.path}: array {key} nests deeper than 2 levels")
            for _ in range(n):                    # each element is itself an array value
                self.value(T_ARR, key, depth + 1)
            return None, atype, n
        if atype == T_STR:
            return [self.string(key) for _ in range(n)], atype, n
        fmt = VALUE_TYPES[atype][1]
        esz = struct.calcsize(fmt)
        if esz * n > MAX_META:
            raise GgufError(f"{self.path}: metadata exceeds {MAX_META} bytes")
        raw = self.read(esz * n, key)
        return [v[0] for v in struct.iter_unpack(fmt, raw)], atype, n


def read_part(path, keep_kv=True):
    """Parse one GGUF part's metadata. keep_kv=False retains only split.* / general.alignment
    (parts > 0 of a split set repeat the whole model metadata)."""
    path = Path(path)
    part = GgufPart(path)
    part.size = path.stat().st_size
    with path.open("rb") as fh:
        rd = _Reader(fh, part.size, str(path))
        if rd.read(4) != GGUF_MAGIC:
            raise GgufError(f"{path}: not a GGUF file")
        part.version = rd.u32()
        if part.version != GGUF_VERSION:
            raise GgufError(f"{path}: GGUF version {part.version} is not supported (need {GGUF_VERSION})")
        n_tensors, n_kv = rd.u64(), rd.u64()
        if n_tensors > MAX_TENSORS:
            raise GgufError(f"{path}: {n_tensors} tensors declared (limit {MAX_TENSORS})")
        if n_kv > MAX_KV:
            raise GgufError(f"{path}: {n_kv} metadata keys declared (limit {MAX_KV})")
        for _ in range(n_kv):
            key = rd.string("metadata key")
            vtype = rd.u32()
            value, atype, n = rd.value(vtype, key)
            if keep_kv or key.startswith("split.") or key == "general.alignment":
                if key in part.kv:
                    raise GgufError(f"{path}: duplicate metadata key {key}")
                part.kv[key] = value
                part.kv_types[key] = (vtype, atype, n)
        align = part.kv.get("general.alignment", DEFAULT_ALIGNMENT)
        if not isinstance(align, int) or isinstance(align, bool) or align <= 0 or align > MAX_ALIGN or align & (align - 1):
            raise GgufError(f"{path}: general.alignment={align!r} is not a power of two in [1,{MAX_ALIGN}]")
        part.alignment = align
        for key, attr in (("split.no", "split_no"), ("split.count", "split_count"),
                          ("split.tensors.count", "split_tensors")):
            v = part.kv.get(key)
            if isinstance(v, int) and not isinstance(v, bool):
                setattr(part, attr, v)
        infos = []
        for _ in range(n_tensors):
            name = rd.string("tensor name")
            nd = rd.u32()
            if nd < 1 or nd > MAX_DIMS:
                raise GgufError(f"{path}: tensor {name} has {nd} dimensions (1..{MAX_DIMS})")
            ne = [rd.u64() for _ in range(nd)]
            for d, e in enumerate(ne):
                if e < 1 or e > (1 << 63) - 1:
                    raise GgufError(f"{path}: tensor {name} has dimension {d} = {e}")
            ne += [1] * (MAX_DIMS - nd)
            tid, rel_off = rd.u32(), rd.u64()
            if tid in GGML_TYPES:
                rs = row_size(tid, ne[0])
                if rs is None:
                    raise GgufError(f"{path}: tensor {name}: ne[0]={ne[0]} is not a multiple of the "
                                    f"{type_name(tid)} block ({GGML_TYPES[tid][1]})")
                nbytes = ne[1] * ne[2] * ne[3] * rs
                if nbytes >= 1 << 63:
                    raise GgufError(f"{path}: tensor {name}: shape overflows 64 bits")
            else:
                nbytes = None
            infos.append((name, tid, ne, nbytes, rel_off))
        part.data_off = (rd.pos + align - 1) // align * align
        if part.data_off > part.size:
            raise GgufError(f"{path}: tensor data section starts past the end of the file")
        dsz = part.size - part.data_off
        seen = set()
        for name, tid, ne, nbytes, rel_off in infos:
            if name in seen:
                raise GgufError(f"{path}: duplicate tensor name {name}")
            seen.add(name)
            if rel_off % align:
                raise GgufError(f"{path}: tensor {name} offset {rel_off} is not {align}-byte aligned")
            if rel_off > dsz or (nbytes is not None and nbytes > dsz - rel_off):
                raise GgufError(f"{path}: tensor {name} ({nbytes} bytes at data+{rel_off}) runs past the "
                                f"end of the file ({dsz} data bytes)")
            part.tensors.append(TensorInfo(name, tid, ne, nbytes, rel_off, part.data_off + rel_off, None))
    return part


def is_gguf_source(path):
    """True for a .gguf file or a directory that holds .gguf files (and no config.json)."""
    p = Path(path)
    if p.is_file():
        return p.suffix.lower() == ".gguf"
    if p.is_dir():
        return not (p / "config.json").exists() and any(p.glob("*.gguf"))
    return False


def _locate(dirs, base):
    for d in dirs:
        cand = Path(d) / base
        if cand.is_file():
            return cand
    return None


def resolve_parts(path, extra_dirs=()):
    """Paths of every part of the model at `path` (file, split part, or directory), in split order."""
    p = Path(path)
    if p.is_dir():
        stems = {}
        for f in sorted(p.glob("*.gguf")):
            m = SPLIT_RE.match(f.name)
            stems.setdefault(m.group(1) if m else f.name, []).append(f)
        if not stems:
            raise GgufError(f"{p}: no .gguf file in this directory")
        if len(stems) > 1:
            raise GgufError(f"{p}: several GGUF models in one directory — pass the model file itself")
        first = min(next(iter(stems.values())))
        search = [p]
    else:
        if not p.is_file():
            raise GgufError(f"{p}: no such file")
        first = p
        search = [p.parent]
    search += [Path(d) for d in extra_dirs if d]
    m = SPLIT_RE.match(first.name)
    if not m:
        return [first]
    stem, count = m.group(1), int(m.group(3))
    if count < 1 or count > MAX_SPLITS:
        raise GgufError(f"{first}: {count} parts (limit {MAX_SPLITS})")
    parts = []
    for k in range(1, count + 1):
        want = f"{stem}-{k:05d}-of-{count:05d}.gguf"
        found = _locate(search, want)
        if found is None:
            where = " or ".join(str(d) for d in search)
            raise GgufError(f"{first}: part {k} of {count} ({want}) not found in {where}")
        parts.append(found)
    return parts


def open_set(path, extra_dirs=()):
    """Parse every part of the model at `path`; returns [GgufPart] with split consistency checked."""
    if isinstance(extra_dirs, str):
        extra_dirs = [d for d in re.split(r"[;,]", extra_dirs) if d]
    paths = resolve_parts(path, extra_dirs)
    parts = [read_part(pp, keep_kv=(i == 0)) for i, pp in enumerate(paths)]
    for i, part in enumerate(parts):
        for t in part.tensors:
            t.file = i
    if len(parts) > 1 or (parts[0].split_count or 0) > 1:
        for i, part in enumerate(parts):
            if part.split_count != len(parts):
                raise GgufError(f"{part.path}: split.count={part.split_count} but {len(parts)} part file(s) found")
            if part.split_no != i:
                raise GgufError(f"{part.path}: split.no={part.split_no}, expected {i} from its file name")
        total = sum(len(p.tensors) for p in parts)
        if parts[0].split_tensors is not None and parts[0].split_tensors != total:
            raise GgufError(f"{parts[0].path}: split.tensors.count={parts[0].split_tensors} but the parts hold {total} tensors")
    names = set()
    for part in parts:
        for t in part.tensors:
            if t.name in names:
                raise GgufError(f"{part.path}: duplicate tensor name {t.name}")
            names.add(t.name)
    return parts


def all_tensors(parts):
    return [t for p in parts for t in p.tensors]


def summarize(parts):
    """Model-level facts the tools print and check (dense/expert split, type mix, MTP precision...)."""
    kv = parts[0].kv
    arch = kv.get("general.architecture")
    tensors = all_tensors(parts)
    mix = {}
    unknown, unsupported = [], []
    total = 0
    for t in tensors:
        if t.type not in GGML_TYPES:
            unknown.append({"name": t.name, "type": t.type})
            continue
        entry = mix.setdefault(t.type_name, {"bytes": 0, "tensors": 0})
        entry["bytes"] += t.nbytes
        entry["tensors"] += 1
        total += t.nbytes
        if t.type not in V1_SUPPORTED:
            unsupported.append({"name": t.name, "type": t.type_name})
    per_layer = {}
    experts_per_layer = None
    expert_bytes = 0
    for t in tensors:
        m = EXPERT_RE.match(t.name)
        if not m or t.nbytes is None:
            continue
        n_exp = t.ne[2]
        experts_per_layer = n_exp if experts_per_layer is None else experts_per_layer
        per_layer[int(m.group(1))] = per_layer.get(int(m.group(1)), 0) + t.nbytes // n_exp
        expert_bytes += t.nbytes
    block_count = kv.get(f"{arch}.block_count") if arch else None
    nextn = kv.get(f"{arch}.nextn_predict_layers") if arch else None
    # llama.cpp counts the NextN (MTP) blocks INSIDE block_count: GLM-5.2 has 78 trunk layers
    # + 1 MTP layer = block_count 79, nextn_predict_layers 1, MTP tensors under blk.78.*
    # (verified on unsloth/GLM-5.2-GGUF). Trunk layers = block_count - nextn.
    mtp = None
    trunk_layers = block_count
    if isinstance(block_count, int):
        n_next = nextn if isinstance(nextn, int) and not isinstance(nextn, bool) else 0
        trunk_layers = block_count - n_next
        for L in range(block_count - 1, max(trunk_layers - 1, -1), -1) if n_next else range(block_count, block_count + 1):
            eh = next((t for t in tensors if t.name == f"blk.{L}.nextn.eh_proj.weight"), None)
            if eh is not None:
                mtp = {"layer": L, "eh_proj_type": eh.type_name, "eh_proj_bits": bits_per_weight(eh.type)}
                break
    indexer_layers = sorted(int(INDEXER_RE.match(t.name).group(1)) for t in tensors if INDEXER_RE.match(t.name))
    tokens = kv.get("tokenizer.ggml.tokens")
    return {
        "path": parts[0].path,
        "parts": len(parts),
        "architecture": arch,
        "engine": ENGINE_ARCHS.get(arch),
        "name": kv.get("general.name"),
        "file_type": kv.get("general.file_type"),
        "quantization_version": kv.get("general.quantization_version"),
        "tensors": len(tensors),
        "total_bytes": total,
        "file_bytes": sum(p.size for p in parts),
        "type_mix": dict(sorted(mix.items(), key=lambda kv_: -kv_[1]["bytes"])),
        "unknown_types": unknown,
        "unsupported_v1": unsupported,
        "block_count": block_count,
        "nextn_predict_layers": nextn,
        "trunk_layers": trunk_layers,
        "expert_count": kv.get(f"{arch}.expert_count") if arch else None,
        "expert_layers": len(per_layer),
        "experts_per_layer": experts_per_layer,
        "expert_bytes": expert_bytes,
        "dense_bytes": total - expert_bytes,
        "typical_expert_bytes": int(statistics.median(per_layer.values())) if per_layer else 0,
        "mtp": mtp,
        "indexer_layers": indexer_layers,
        "tokenizer": {
            "model": kv.get("tokenizer.ggml.model"),
            "pre": kv.get("tokenizer.ggml.pre"),
            "tokens": len(tokens) if isinstance(tokens, list) else 0,
            "eos_token_id": kv.get("tokenizer.ggml.eos_token_id"),
            "has_chat_template": isinstance(kv.get("tokenizer.chat_template"), str),
        },
    }


def _fmt_bytes(n):
    if n is None:
        return "?"
    for unit, div in (("TB", 1e12), ("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if n >= div:
            return f"{n / div:.2f} {unit}"
    return f"{n} B"


def _fmt_value(v, meta):
    vtype, atype, n = meta
    if vtype == T_ARR:
        if v is None:
            return f"[{n} × nested array]"
        head = ", ".join(repr(x) if isinstance(x, str) else f"{x}" for x in v[:4])
        return f"[{n} × {VALUE_TYPES[atype][0]}] {head}{', …' if n > 4 else ''}"
    if isinstance(v, str):
        return repr(v if len(v) <= 120 else v[:117] + "...")
    return str(v)


def format_inspect(parts, summary, tensors=False):
    lines = [f"gguf · {summary['path']}"]
    kv = parts[0].kv
    for i, p in enumerate(parts):
        lines.append(f"  part {i}: {p.path} · {_fmt_bytes(p.size)} · {len(p.tensors)} tensors · data at {p.data_off} · align {p.alignment}")
    lines.append("")
    lines.append(f"architecture  {summary['architecture']}  ({'engine: ' + summary['engine'] if summary['engine'] else 'not supported by this engine'})")
    lines.append(f"name          {summary['name']}")
    lines.append(f"tensors       {summary['tensors']} · {_fmt_bytes(summary['total_bytes'])} known"
                 + (f" · {len(summary['unknown_types'])} of unknown type" if summary['unknown_types'] else ""))
    mix = " · ".join(f"{k} {100.0 * v['bytes'] / summary['total_bytes']:.0f}% ({v['tensors']})"
                     for k, v in summary["type_mix"].items()) if summary["total_bytes"] else "-"
    lines.append(f"type mix      {mix}")
    if summary["unsupported_v1"]:
        names = ", ".join(f"{u['name']} ({u['type']})" for u in summary["unsupported_v1"][:5])
        lines.append(f"unsupported   {len(summary['unsupported_v1'])} tensor(s) outside the v1 type set: {names}"
                     + (" …" if len(summary["unsupported_v1"]) > 5 else ""))
    if summary["expert_layers"]:
        lines.append(f"experts       {summary['expert_layers']} MoE layers × {summary['experts_per_layer']} experts · "
                     f"{_fmt_bytes(summary['typical_expert_bytes'])} per expert · {_fmt_bytes(summary['expert_bytes'])} total")
        lines.append(f"dense         {_fmt_bytes(summary['dense_bytes'])} resident")
    mtp = summary["mtp"]
    lines.append(f"layers        {summary['trunk_layers']} trunk + {summary['nextn_predict_layers'] or 0} nextn (block_count {summary['block_count']})")
    lines.append(f"mtp           {'blk.%d nextn · eh_proj %s (%.2f bpw)' % (mtp['layer'], mtp['eh_proj_type'], mtp['eh_proj_bits']) if mtp else 'absent'}")
    lines.append(f"indexer       {str(len(summary['indexer_layers'])) + ' layers' if summary['indexer_layers'] else 'absent'}")
    tok = summary["tokenizer"]
    lines.append(f"tokenizer     {tok['model']} · pre {tok['pre']} · {tok['tokens']} tokens · eos {tok['eos_token_id']}"
                 + (" · chat template" if tok["has_chat_template"] else ""))
    lines.append("")
    lines.append("metadata")
    for key in sorted(kv):
        if key in ("tokenizer.ggml.tokens", "tokenizer.ggml.merges", "tokenizer.ggml.scores", "tokenizer.ggml.token_type"):
            lines.append(f"  {key:<44} {_fmt_value(kv[key], parts[0].kv_types[key]).split(']')[0]}]")
        else:
            lines.append(f"  {key:<44} {_fmt_value(kv[key], parts[0].kv_types[key])}")
    if tensors:
        lines.append("")
        lines.append(f"{'tensor':<48} {'type':<8} {'shape':<28} {'bytes':>14} {'offset':>14} part")
        for t in all_tensors(parts):
            last = max([i for i, e in enumerate(t.ne) if e != 1] or [0])
            shape = "×".join(str(e) for e in t.ne[:last + 1])
            lines.append(f"{t.name:<48} {t.type_name:<8} {shape:<28} {t.nbytes if t.nbytes is not None else '?':>14} {t.off:>14} {t.file}")
    return "\n".join(lines)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="ggufinfo", description="inspect a GGUF model (metadata only)")
    ap.add_argument("path")
    ap.add_argument("--tensors", action="store_true", help="also list every tensor")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--dirs", default=os.environ.get("COLI_MODEL_DIRS", ""),
                    help="extra directories to search for split parts (';' or ',' separated)")
    a = ap.parse_args(argv)
    try:
        parts = open_set(a.path, a.dirs)
    except (GgufError, OSError) as error:
        print(f"gguf: {error}", file=sys.stderr)
        return 1
    summary = summarize(parts)
    if a.json:
        out = dict(summary)
        if a.tensors:
            out["tensor_table"] = [t.as_dict() for t in all_tensors(parts)]
        out["metadata"] = {k: (v if not isinstance(v, list) or len(v) <= 16 else f"[{len(v)} values]")
                           for k, v in parts[0].kv.items()}
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        print(format_inspect(parts, summary, tensors=a.tensors))
    return 0


if __name__ == "__main__":
    sys.exit(main())
