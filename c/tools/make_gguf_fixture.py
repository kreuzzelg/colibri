#!/usr/bin/env python3
"""Standard-library GGUF v3 writer for tests and fixtures.

Not a model converter: it writes exactly the bytes the reader is asked about
(`gguf.h`, `ggufinfo.py`), including deliberately malformed files, so the test
suites never need llama.cpp's gguf-py. Also used by `test_doctor.py` to lay out
a tiny synthetic `glm-dsa` model.

    python3 tools/make_gguf_fixture.py out.gguf          # demo single file
    python3 tools/make_gguf_fixture.py out_dir --split 3 # demo split set

Payload bytes are deterministic (a per-tensor xorshift stream), so a reader can
check that offsets land on the intended bytes.
"""
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ggufinfo import GGML_TYPES, TYPE_IDS, VALUE_TYPES, T_ARR, T_STR, row_size  # noqa: E402

U8, I8, U16, I16, U32, I32, F32, BOOL, STR, ARR, U64, I64, F64 = range(13)


def _pack_str(s):
    raw = s.encode("utf-8") if isinstance(s, str) else bytes(s)
    return struct.pack("<Q", len(raw)) + raw


def _pack_value(vtype, value):
    if vtype == STR:
        return _pack_str(value)
    if vtype == ARR:
        atype, items = value
        out = struct.pack("<IQ", atype, len(items))
        for item in items:
            if atype == ARR:                      # nested: each element is a full array value
                out += _pack_value(ARR, item)
            else:
                out += _pack_value(atype, item)
        return out
    fmt = VALUE_TYPES[vtype][1]
    return struct.pack(fmt, value)


def tensor_bytes(name, nbytes):
    """Deterministic payload: xorshift64 seeded from the name."""
    x = 0x9E3779B97F4A7C15
    for ch in name.encode():
        x = ((x ^ ch) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    x = x or 1
    out = bytearray()
    while len(out) < nbytes:
        x ^= (x << 13) & 0xFFFFFFFFFFFFFFFF
        x ^= x >> 7
        x ^= (x << 17) & 0xFFFFFFFFFFFFFFFF
        out += x.to_bytes(8, "little")
    return bytes(out[:nbytes])


class GgufWriter:
    def __init__(self, alignment=32, version=3, magic=b"GGUF"):
        self.alignment = alignment
        self.version = version
        self.magic = magic
        self.kv = []        # (key, vtype, value)
        self.tensors = []   # (name, tid, ne, payload bytes or None, explicit_offset or None)
        self.declared_tensor_count = None
        self.declared_kv_count = None

    # -- metadata -------------------------------------------------------------
    def add(self, key, vtype, value):
        self.kv.append((key, vtype, value))
        return self

    def add_str(self, key, value):
        return self.add(key, STR, value)

    def add_u32(self, key, value):
        return self.add(key, U32, value)

    def add_arr(self, key, atype, items):
        return self.add(key, ARR, (atype, list(items)))

    # -- tensors ----------------------------------------------------------------
    def add_tensor(self, name, tid, ne, payload=None, offset=None):
        """ne in ggml order (ne0 fastest). payload None -> deterministic bytes of the right size;
        a type unknown to the table needs an explicit payload."""
        if isinstance(tid, str):
            tid = TYPE_IDS[tid]
        ne = list(ne)
        if payload is None:
            rs = row_size(tid, ne[0])
            if rs is None:
                raise ValueError(f"{name}: cannot size type {tid} with ne0={ne[0]}; pass payload")
            n = rs
            for e in ne[1:]:
                n *= e
            payload = tensor_bytes(name, n)
        self.tensors.append((name, tid, ne, payload, offset))
        return self

    # -- serialisation ------------------------------------------------------------
    def build(self):
        al = self.alignment
        if "general.alignment" not in [k for k, _, _ in self.kv] and al != 32:
            self.kv.insert(0, ("general.alignment", U32, al))
        head = bytearray(self.magic)
        head += struct.pack("<I", self.version)
        head += struct.pack("<Q", len(self.tensors) if self.declared_tensor_count is None else self.declared_tensor_count)
        head += struct.pack("<Q", len(self.kv) if self.declared_kv_count is None else self.declared_kv_count)
        for key, vtype, value in self.kv:
            head += _pack_str(key) + struct.pack("<I", vtype) + _pack_value(vtype, value)
        # lay out payloads
        offsets, data = [], bytearray()
        for name, tid, ne, payload, offset in self.tensors:
            if offset is None:
                pad = (-len(data)) % al
                data += b"\0" * pad
                offset = len(data)
            offsets.append(offset)
            if payload is not None and offset == len(data):
                data += payload
            elif payload is not None:
                end = offset + len(payload)
                if end > len(data):
                    data += b"\0" * (end - len(data))
                data[offset:end] = payload
        for (name, tid, ne, payload, _), offset in zip(self.tensors, offsets):
            head += _pack_str(name) + struct.pack("<I", len(ne))
            for e in ne:
                head += struct.pack("<Q", e)
            head += struct.pack("<IQ", tid, offset)
        head += b"\0" * ((-len(head)) % al)
        return bytes(head) + bytes(data)

    def write(self, path):
        blob = self.build()
        with open(path, "wb") as fh:
            fh.write(blob)
        return len(blob)


def demo_writer(arch="demo", alignment=32):
    """One file exercising every metadata value type and a few tensor types."""
    w = GgufWriter(alignment=alignment)
    w.add_str("general.architecture", arch)
    w.add_str("general.name", "gguf fixture")
    w.add_u32("general.quantization_version", 2)
    w.add_u32("general.file_type", 15)
    w.add("test.u8", U8, 200).add("test.i8", I8, -5).add("test.u16", U16, 60000).add("test.i16", I16, -300)
    w.add("test.u32", U32, 4000000000).add("test.i32", I32, -70000).add("test.f32", F32, 1.5)
    w.add("test.bool", BOOL, True).add("test.u64", U64, 1 << 40).add("test.i64", I64, -(1 << 40)).add("test.f64", F64, 2.25)
    w.add_arr("test.arr_u8", U8, [1, 2, 3]).add_arr("test.arr_i32", I32, [-1, 0, 1, 2])
    w.add_arr("test.arr_f32", F32, [0.5, -0.5]).add_arr("test.arr_str", STR, ["a", "bb", "ccc"])
    w.add_arr("test.arr_nested", ARR, [(U8, [1, 2]), (STR, ["x"])])
    w.add("test.empty_str", STR, "").add_arr("test.empty_arr", I32, [])
    w.add_u32(f"{arch}.block_count", 2)
    w.add_arr("tokenizer.ggml.tokens", STR, ["<s>", "a", "b", "c"])
    w.add_tensor("token_embd.weight", "F32", [4, 4])
    w.add_tensor("blk.0.ffn_gate_exps.weight", "Q4_K", [256, 3, 2])
    w.add_tensor("blk.0.ffn_down_exps.weight", "Q6_K", [256, 2, 2])
    w.add_tensor("blk.0.attn_norm.weight", "F16", [4])
    w.add_tensor("blk.1.attn_q_a.weight", "Q8_0", [64, 2])
    w.add_tensor("blk.1.odd.weight", 99, [32], payload=b"\x55" * 40)     # unknown type: indexed, not sized
    return w


def write_split_set(out_dir, stem, parts, alignment=32):
    """parts: list of (kv list, tensor list) in split order. Each part gets split.* keys
    (part 0 keeps the model metadata, others only what llama.cpp's gguf-split writes)."""
    os.makedirs(out_dir, exist_ok=True)
    total = sum(len(t) for _, t in parts)
    paths = []
    for i, (kvs, tensors) in enumerate(parts):
        w = GgufWriter(alignment=alignment)
        for key, vtype, value in kvs:
            w.add(key, vtype, value)
        w.add("split.no", U16, i).add("split.count", U16, len(parts)).add("split.tensors.count", I32, total)
        for t in tensors:
            w.add_tensor(*t)
        path = os.path.join(out_dir, f"{stem}-{i + 1:05d}-of-{len(parts):05d}.gguf")
        w.write(path)
        paths.append(path)
    return paths


def tiny_glm_dsa(hidden=256, n_ff_exp=256, n_expert=4, block_count=2, dense_lead=1, vocab=64,
                 expert_type="Q4_K", down_type="Q6_K", mtp_type="Q8_0", indexer_layers=(1,),
                 with_mtp=True):
    """A synthetic glm-dsa layout with the tensor names the engine will map (ARCHITECTURE.md §6.1).
    Shapes are tiny but block-aligned; the payload is noise. Used by doctor/ggufinfo tests."""
    arch = "glm-dsa"
    w = GgufWriter()
    w.add_str("general.architecture", arch).add_str("general.name", "tiny glm-dsa fixture")
    w.add_u32("general.quantization_version", 2).add_u32("general.file_type", 15)
    # llama.cpp convention: block_count includes the NextN block(s); `block_count` here is the
    # TRUNK depth, so the key written is block_count + 1 when the MTP layer is present.
    w.add_u32(f"{arch}.block_count", block_count + (1 if with_mtp else 0)).add_u32(f"{arch}.embedding_length", hidden)
    w.add_u32(f"{arch}.expert_count", n_expert).add_u32(f"{arch}.expert_used_count", 2)
    w.add_u32(f"{arch}.expert_feed_forward_length", n_ff_exp).add_u32(f"{arch}.feed_forward_length", 2 * n_ff_exp)
    w.add_u32(f"{arch}.leading_dense_block_count", dense_lead).add_u32(f"{arch}.expert_shared_count", 1)
    w.add_u32(f"{arch}.attention.head_count", 2).add_u32(f"{arch}.attention.q_lora_rank", 256)
    w.add_u32(f"{arch}.attention.kv_lora_rank", 256).add_u32(f"{arch}.attention.key_length_mla", 64)
    w.add_u32(f"{arch}.attention.value_length_mla", 32).add_u32(f"{arch}.rope.dimension_count", 32)
    w.add(f"{arch}.attention.layer_norm_rms_epsilon", F32, 1e-5).add(f"{arch}.rope.freq_base", F32, 10000.0)
    w.add(f"{arch}.expert_weights_scale", F32, 2.5).add(f"{arch}.expert_weights_norm", BOOL, True)
    w.add_u32(f"{arch}.expert_gating_func", 2)
    if with_mtp:
        w.add_u32(f"{arch}.nextn_predict_layers", 1)
    if indexer_layers:
        w.add_u32(f"{arch}.attention.indexer.head_count", 2).add_u32(f"{arch}.attention.indexer.key_length", 32)
        w.add_u32(f"{arch}.attention.indexer.top_k", 64)
    w.add_str("tokenizer.ggml.model", "gpt2").add_str("tokenizer.ggml.pre", "glm4")
    w.add_arr("tokenizer.ggml.tokens", STR, [f"t{i}" for i in range(vocab)])
    w.add_arr("tokenizer.ggml.token_type", I32, [1] * vocab)
    w.add_arr("tokenizer.ggml.merges", STR, ["t 1", "t 2"])
    w.add_u32("tokenizer.ggml.eos_token_id", vocab - 1)
    w.add_str("tokenizer.chat_template", "{{ messages }}")
    w.add_tensor("token_embd.weight", "F16", [hidden, vocab])
    w.add_tensor("output_norm.weight", "F32", [hidden])
    w.add_tensor("output.weight", "Q8_0", [hidden, vocab])
    layers = block_count + (1 if with_mtp else 0)
    for i in range(layers):
        w.add_tensor(f"blk.{i}.attn_norm.weight", "F32", [hidden])
        w.add_tensor(f"blk.{i}.ffn_norm.weight", "F32", [hidden])
        w.add_tensor(f"blk.{i}.attn_q_a.weight", "Q8_0", [hidden, 256])
        w.add_tensor(f"blk.{i}.attn_q_a_norm.weight", "F32", [256])
        w.add_tensor(f"blk.{i}.attn_q_b.weight", "Q8_0", [256, 2 * 64])
        w.add_tensor(f"blk.{i}.attn_kv_a_mqa.weight", "Q8_0", [hidden, 256 + 32])
        w.add_tensor(f"blk.{i}.attn_kv_a_norm.weight", "F32", [256])
        w.add_tensor(f"blk.{i}.attn_k_b.weight", "F16", [32, 256, 2])
        w.add_tensor(f"blk.{i}.attn_v_b.weight", "F16", [256, 32, 2])
        w.add_tensor(f"blk.{i}.attn_output.weight", "Q8_0", [2 * 32, hidden])
        if i < dense_lead and i < block_count:
            w.add_tensor(f"blk.{i}.ffn_gate.weight", expert_type, [hidden, 2 * n_ff_exp])
            w.add_tensor(f"blk.{i}.ffn_up.weight", expert_type, [hidden, 2 * n_ff_exp])
            w.add_tensor(f"blk.{i}.ffn_down.weight", down_type, [2 * n_ff_exp, hidden])
        else:
            w.add_tensor(f"blk.{i}.ffn_gate_inp.weight", "F32", [hidden, n_expert])
            w.add_tensor(f"blk.{i}.exp_probs_b.bias", "F32", [n_expert])
            w.add_tensor(f"blk.{i}.ffn_gate_exps.weight", expert_type, [hidden, n_ff_exp, n_expert])
            w.add_tensor(f"blk.{i}.ffn_up_exps.weight", expert_type, [hidden, n_ff_exp, n_expert])
            w.add_tensor(f"blk.{i}.ffn_down_exps.weight", down_type, [n_ff_exp, hidden, n_expert])
            w.add_tensor(f"blk.{i}.ffn_gate_shexp.weight", expert_type, [hidden, n_ff_exp])
            w.add_tensor(f"blk.{i}.ffn_up_shexp.weight", expert_type, [hidden, n_ff_exp])
            w.add_tensor(f"blk.{i}.ffn_down_shexp.weight", down_type, [n_ff_exp, hidden])
        if i in indexer_layers and i < block_count:
            w.add_tensor(f"blk.{i}.indexer.attn_q_b.weight", "F16", [256, 2 * 32])
            w.add_tensor(f"blk.{i}.indexer.attn_k.weight", "F16", [hidden, 32])
            w.add_tensor(f"blk.{i}.indexer.proj.weight", "F16", [hidden, 2])
            w.add_tensor(f"blk.{i}.indexer.k_norm.weight", "F32", [32])
            w.add_tensor(f"blk.{i}.indexer.k_norm.bias", "F32", [32])
    if with_mtp:
        L = block_count
        w.add_tensor(f"blk.{L}.nextn.eh_proj.weight", mtp_type, [2 * hidden, hidden])
        w.add_tensor(f"blk.{L}.nextn.enorm.weight", "F32", [hidden])
        w.add_tensor(f"blk.{L}.nextn.hnorm.weight", "F32", [hidden])
        w.add_tensor(f"blk.{L}.nextn.shared_head_norm.weight", "F32", [hidden])
    return w


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="write GGUF test fixtures (no model weights involved)")
    ap.add_argument("out", help="output file, or directory with --split")
    ap.add_argument("--split", type=int, default=0, help="write the demo as N parts into OUT/")
    ap.add_argument("--glm", action="store_true", help="write the tiny synthetic glm-dsa layout instead of the demo")
    ap.add_argument("--alignment", type=int, default=32)
    a = ap.parse_args(argv)
    if a.split:
        demo = demo_writer(alignment=a.alignment)
        chunks = [[] for _ in range(a.split)]
        for i, t in enumerate(demo.tensors):
            chunks[i % a.split].append(t[:4])
        parts = [(demo.kv if i == 0 else [], chunk) for i, chunk in enumerate(chunks)]
        for p in write_split_set(a.out, "demo", parts, alignment=a.alignment):
            print(p)
        return 0
    w = tiny_glm_dsa() if a.glm else demo_writer(alignment=a.alignment)
    n = w.write(a.out)
    print(f"{a.out}: {n} bytes, {len(w.tensors)} tensors, {len(w.kv)} keys")
    return 0


if __name__ == "__main__":
    sys.exit(main())
