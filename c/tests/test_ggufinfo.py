"""ggufinfo.py (the stdlib GGUF reader behind coli/doctor) against fixtures written by
tools/make_gguf_fixture.py, plus a cross-check of the C reader (tests/test_gguf, if
built) on the same files: both must see the same files, keys and tensor table."""
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import ggufinfo
from ggufinfo import GgufError, open_set, read_part, summarize, format_inspect, V1_SUPPORTED, TYPE_IDS
from tools.make_gguf_fixture import (GgufWriter, demo_writer, tiny_glm_dsa, write_split_set,
                                     tensor_bytes, U8, U16, U32, I32, F32, STR, ARR)

HERE = Path(__file__).resolve().parent
C_READER = HERE / ("test_gguf.exe" if sys.platform == "win32" else "test_gguf")


class GgufInfoTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, writer, name="m.gguf"):
        path = self.root / name
        writer.write(path)
        return path

    # -- happy paths ------------------------------------------------------------
    def test_every_value_type_roundtrips(self):
        part = read_part(self.write(demo_writer()))
        kv = part.kv
        self.assertEqual(part.version, 3)
        self.assertEqual(kv["general.architecture"], "demo")
        self.assertEqual(kv["test.u8"], 200)
        self.assertEqual(kv["test.i8"], -5)
        self.assertEqual(kv["test.u16"], 60000)
        self.assertEqual(kv["test.i16"], -300)
        self.assertEqual(kv["test.u32"], 4000000000)
        self.assertEqual(kv["test.i32"], -70000)
        self.assertEqual(kv["test.f32"], 1.5)
        self.assertIs(kv["test.bool"], True)
        self.assertEqual(kv["test.u64"], 1 << 40)
        self.assertEqual(kv["test.i64"], -(1 << 40))
        self.assertEqual(kv["test.f64"], 2.25)
        self.assertEqual(kv["test.arr_u8"], [1, 2, 3])
        self.assertEqual(kv["test.arr_i32"], [-1, 0, 1, 2])
        self.assertEqual(kv["test.arr_f32"], [0.5, -0.5])
        self.assertEqual(kv["test.arr_str"], ["a", "bb", "ccc"])
        self.assertIsNone(kv["test.arr_nested"])          # parsed and bounds-checked, not retained
        self.assertEqual(part.kv_types["test.arr_nested"], (ARR, ARR, 2))
        self.assertEqual(kv["test.empty_str"], "")
        self.assertEqual(kv["test.empty_arr"], [])
        self.assertEqual(kv["tokenizer.ggml.tokens"], ["<s>", "a", "b", "c"])

    def test_tensor_table_and_offsets(self):
        path = self.write(demo_writer())
        part = read_part(path)
        self.assertEqual(part.alignment, 32)
        self.assertEqual(part.data_off % 32, 0)
        by = {t.name: t for t in part.tensors}
        gate = by["blk.0.ffn_gate_exps.weight"]
        self.assertEqual((gate.type_name, gate.ne), ("Q4_K", [256, 3, 2, 1]))
        self.assertEqual(gate.nbytes, 3 * 2 * 144)
        self.assertEqual(gate.off, part.data_off + gate.rel_off)
        self.assertEqual(by["blk.0.ffn_down_exps.weight"].nbytes, 2 * 2 * 210)
        self.assertEqual(by["blk.1.attn_q_a.weight"].nbytes, 2 * (64 // 32) * 34)
        self.assertEqual(by["blk.0.attn_norm.weight"].nbytes, 8)
        odd = by["blk.1.odd.weight"]
        self.assertEqual(odd.type, 99)
        self.assertIsNone(odd.nbytes)
        self.assertTrue(odd.type_name.startswith("unknown"))
        # the offsets land on the deterministic payload the writer produced
        with path.open("rb") as fh:
            fh.seek(gate.off)
            self.assertEqual(fh.read(gate.nbytes), tensor_bytes(gate.name, gate.nbytes))

    def test_alignment_64(self):
        part = read_part(self.write(demo_writer(alignment=64)))
        self.assertEqual(part.alignment, 64)
        self.assertEqual(part.data_off % 64, 0)
        for t in part.tensors:
            self.assertEqual(t.rel_off % 64, 0)

    def test_split_set(self):
        demo = demo_writer()
        chunks = [[], [], []]
        for i, t in enumerate(demo.tensors):
            chunks[i % 3].append(t[:4])
        sdir = self.root / "split"
        parts_spec = [(demo.kv if i == 0 else [("general.architecture", STR, "demo")], c) for i, c in enumerate(chunks)]
        paths = write_split_set(sdir, "demo", parts_spec)
        for opener in (sdir, paths[0], paths[2]):
            parts = open_set(opener)
            self.assertEqual(len(parts), 3)
            self.assertEqual([p.split_no for p in parts], [0, 1, 2])
            self.assertEqual(sum(len(p.tensors) for p in parts), 6)
            self.assertEqual(parts[0].kv["general.architecture"], "demo")
            self.assertNotIn("general.architecture", parts[1].kv)      # repeated metadata dropped
            files = {t.name: t.file for p in parts for t in p.tensors}
            self.assertEqual(files["blk.0.ffn_gate_exps.weight"], 1)
        # a part on another drive is found through extra dirs
        drive2 = self.root / "drive2"
        drive2.mkdir()
        moved = drive2 / Path(paths[2]).name
        shutil.move(paths[2], moved)
        with self.assertRaisesRegex(GgufError, "part 3 of 3"):
            open_set(paths[0])
        parts = open_set(paths[0], str(drive2))
        self.assertEqual(Path(parts[2].path), moved)
        shutil.move(moved, paths[2])
        # two models in one directory is ambiguous
        demo_writer().write(sdir / "other.gguf")
        with self.assertRaisesRegex(GgufError, "several GGUF models"):
            open_set(sdir)
        self.assertTrue(ggufinfo.is_gguf_source(sdir))
        self.assertTrue(ggufinfo.is_gguf_source(paths[0]))
        self.assertFalse(ggufinfo.is_gguf_source(self.root / "nope"))

    def test_split_consistency(self):
        sdir = self.root / "bad"
        paths = write_split_set(sdir, "m", [([("general.architecture", STR, "demo")], [("a", "F32", [4])]),
                                           ([], [("b", "F32", [4])])])
        with open(paths[1], "r+b") as fh:
            blob = fh.read()
            i = blob.index(b"split.no") + len("split.no") + 4
            fh.seek(i)
            fh.write(struct.pack("<H", 7))
        with self.assertRaisesRegex(GgufError, r"split\.no=7"):
            open_set(sdir)
        with open(paths[1], "r+b") as fh:
            fh.seek(i)
            fh.write(struct.pack("<H", 1))
        with open(paths[0], "r+b") as fh:
            blob = fh.read()
            i = blob.index(b"split.tensors.count") + len("split.tensors.count") + 4
            fh.seek(i)
            fh.write(struct.pack("<i", 9))
        with self.assertRaisesRegex(GgufError, r"split\.tensors\.count=9"):
            open_set(sdir)

    def test_summary_of_tiny_glm_dsa(self):
        parts = open_set(self.write(tiny_glm_dsa()))
        s = summarize(parts)
        self.assertEqual(s["architecture"], "glm-dsa")
        self.assertEqual(s["engine"], "glm")
        self.assertEqual(s["block_count"], 3)          # 2 trunk + 1 nextn, llama.cpp convention
        self.assertEqual(s["trunk_layers"], 2)
        self.assertEqual(s["nextn_predict_layers"], 1)
        self.assertEqual(s["expert_count"], 4)
        self.assertEqual(s["expert_layers"], 2)          # blk.1 (routed) + blk.2 (MTP layer)
        self.assertEqual(s["experts_per_layer"], 4)
        # one expert = gate + up (Q4_K, 256x256) + down (Q6_K, 256x256)
        self.assertEqual(s["typical_expert_bytes"], 2 * 256 * 144 + 256 * 210)
        self.assertEqual(s["expert_bytes"] + s["dense_bytes"], s["total_bytes"])
        self.assertEqual(s["unsupported_v1"], [])
        self.assertEqual(s["unknown_types"], [])
        self.assertEqual(s["mtp"], {"layer": 2, "eh_proj_type": "Q8_0", "eh_proj_bits": 8.5})
        self.assertEqual(s["indexer_layers"], [1])
        self.assertEqual(s["tokenizer"]["tokens"], 64)
        self.assertEqual(s["tokenizer"]["pre"], "glm4")
        self.assertTrue(s["tokenizer"]["has_chat_template"])
        self.assertEqual(list(s["type_mix"])[0], "Q4_K")
        text = format_inspect(parts, s, tensors=True)
        self.assertIn("engine: glm", text)
        self.assertIn("eh_proj Q8_0", text)
        self.assertIn("blk.1.ffn_gate_exps.weight", text)

    def test_summary_flags_unsupported_and_low_precision_mtp(self):
        w = tiny_glm_dsa(expert_type="IQ3_XXS", mtp_type="Q4_K", with_mtp=True, indexer_layers=())
        s = summarize(open_set(self.write(w)))
        names = {u["type"] for u in s["unsupported_v1"]}
        self.assertEqual(names, {"IQ3_XXS"})
        self.assertEqual(s["mtp"]["eh_proj_bits"], 4.5)
        self.assertEqual(s["indexer_layers"], [])
        self.assertEqual(s["engine"], "glm")
        s2 = summarize(open_set(self.write(demo_writer(arch="glm4moe"), "o.gguf")))
        self.assertIsNone(s2["engine"])
        self.assertEqual(len(s2["unknown_types"]), 1)

    def test_v1_type_set(self):
        self.assertEqual({ggufinfo.type_name(t) for t in V1_SUPPORTED},
                         {"F32", "F16", "BF16", "Q4_0", "Q8_0", "Q4_K", "Q5_K", "Q6_K"})
        self.assertEqual(ggufinfo.row_size(TYPE_IDS["Q4_K"], 512), 288)
        self.assertIsNone(ggufinfo.row_size(TYPE_IDS["Q4_K"], 300))
        self.assertEqual(ggufinfo.bits_per_weight(TYPE_IDS["Q4_K"]), 4.5)

    # -- refusals -----------------------------------------------------------------
    def bad(self, writer, regex, name="bad.gguf"):
        path = self.write(writer, name)
        with self.assertRaisesRegex(GgufError, regex):
            read_part(path)

    def test_refusals(self):
        self.bad(GgufWriter(magic=b"GGUX").add_str("a", "b"), "not a GGUF")
        self.bad(GgufWriter(version=2).add_str("a", "b"), "version 2")
        w = GgufWriter().add_str("a", "b"); w.declared_tensor_count = 1 << 40
        self.bad(w, "limit")
        w = GgufWriter().add_str("a", "b"); w.declared_kv_count = 5
        self.bad(w, "end of the file")
        self.bad(GgufWriter().add_str("k" * (ggufinfo.MAX_STR + 1), "b"), "limit")
        self.bad(GgufWriter().add("general.alignment", U32, 48), "power of two")
        self.bad(GgufWriter().add("general.alignment", STR, "32"), "power of two")
        self.bad(GgufWriter().add_arr("x", ARR, [(ARR, [(U8, [1])])]), "nests deeper")
        self.bad(GgufWriter().add_tensor("t", "Q4_K", [100, 2], payload=b"\0" * 64), "not a multiple")
        self.bad(GgufWriter().add_tensor("t", "F32", [4, 4], payload=b"\0" * 64, offset=16), "aligned")
        self.bad(GgufWriter().add_tensor("t", "F32", [4, 4], payload=b"", offset=1 << 20), "runs past")   # 64 bytes declared at EOF
        self.bad(GgufWriter().add_tensor("t", "F32", [4]).add_tensor("t", "F32", [4]), "duplicate")
        self.bad(GgufWriter().add_str("k", "v").add_str("k", "w"), "duplicate metadata key")
        path = self.write(demo_writer(), "trunc.gguf")
        data = path.read_bytes()[:60]
        path.write_bytes(data)
        with self.assertRaisesRegex(GgufError, "end of the file|short read"):
            read_part(path)
        with self.assertRaisesRegex(GgufError, "no such file"):
            open_set(self.root / "missing.gguf")
        empty = self.root / "empty"
        empty.mkdir()
        with self.assertRaisesRegex(GgufError, r"no \.gguf"):
            open_set(empty)

    def test_cli(self):
        path = self.write(tiny_glm_dsa())
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = ggufinfo.main([str(path), "--json", "--tensors"])
        self.assertEqual(rc, 0)
        import json
        out = json.loads(buf.getvalue())
        self.assertEqual(out["architecture"], "glm-dsa")
        self.assertEqual(len(out["tensor_table"]), 61)
        self.assertEqual(out["metadata"]["tokenizer.ggml.tokens"], "[64 values]")
        self.assertEqual(ggufinfo.main([str(self.root / "missing.gguf")]), 1)

    # -- C/Python cross-check ---------------------------------------------------------
    @unittest.skipUnless(C_READER.exists(), "tests/test_gguf not built (run make test-c first)")
    def test_c_reader_agrees(self):
        files = [self.write(demo_writer(), "demo.gguf"), self.write(demo_writer(alignment=64), "demo64.gguf"),
                 self.write(tiny_glm_dsa(), "glm.gguf")]
        demo = demo_writer()
        chunks = [[], [], []]
        for i, t in enumerate(demo.tensors):
            chunks[i % 3].append(t[:4])
        files.append(self.root / "split")
        write_split_set(files[-1], "demo", [(demo.kv if i == 0 else [], c) for i, c in enumerate(chunks)])
        for f in files:
            parts = open_set(f)
            out = subprocess.run([str(C_READER), str(f)], capture_output=True, text=True, check=False)
            self.assertEqual(out.returncode, 0, out.stderr)
            lines = out.stdout.splitlines()
            c_files = [l.split() for l in lines if l.startswith("F ")]
            c_kv = {l.split()[1]: l.split()[2:5] for l in lines if l.startswith("K ")}
            c_t = {l.split()[1]: l.split()[2:] for l in lines if l.startswith("T ")}
            self.assertEqual(len(c_files), len(parts))
            for rec, p in zip(c_files, parts):
                self.assertEqual((int(rec[3]), int(rec[4]), int(rec[5])), (p.size, p.data_off, p.alignment))
            for key, (vtype, atype, n) in parts[0].kv_types.items():
                self.assertIn(key, c_kv, key)
                self.assertEqual(c_kv[key][0], ggufinfo.VALUE_TYPES[vtype][0])
                self.assertEqual(c_kv[key][2], str(n))
            py_t = {t.name: t for p in parts for t in p.tensors}
            self.assertEqual(set(py_t), set(c_t))
            for name, t in py_t.items():
                typ, nd, ne0, ne1, ne2, ne3, file, off, nbytes = c_t[name]
                self.assertEqual(int(typ), t.type)
                self.assertEqual([int(ne0), int(ne1), int(ne2), int(ne3)], t.ne)
                self.assertEqual(int(file), t.file)
                self.assertEqual(int(off), t.off)
                self.assertEqual(int(nbytes), -1 if t.nbytes is None else t.nbytes)


if __name__ == "__main__":
    unittest.main()
