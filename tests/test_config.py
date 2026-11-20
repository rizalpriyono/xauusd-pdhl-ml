"""Tes konfigurasi: kedua berkas `configs/*.json` dan validasinya."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pdhl_data.config import ConfigError, load_config, parse_config  # noqa: E402


class TestConfigFiles(unittest.TestCase):
    def test_tesis_kunci_1(self):
        cfg = load_config(ROOT / "configs" / "tesis.json")
        lab = cfg["label"]
        self.assertEqual((lab["a"], lab["b"], lab["T"], lab["k"]), (2.0, 1.0, 8, 0))
        self.assertEqual(cfg["expected_valid"], {"dev": 1841, "calib": 227, "test": 341})

    def test_every_config_has_reference(self):
        for path in (ROOT / "configs").glob("*.json"):
            with self.subTest(config=path.name):
                name = load_config(path)["name"]
                self.assertEqual(path.stem, name)
                self.assertTrue((ROOT / "references" / f"{name}.json").exists())


class TestValidation(unittest.TestCase):
    BASE = json.loads((ROOT / "configs" / "tesis.json").read_text())

    def _with(self, section, **kw):
        raw = copy.deepcopy(self.BASE)
        if section:
            raw[section].update(kw)
        else:
            raw.update(kw)
        return raw

    def test_rejects_invalid_label_values(self):
        for bad in ({"a": 0}, {"b": -1}, {"T": 0}, {"k": -1}, {"T": 2.5}, {"a": True}, {"a": "2"}):
            with self.subTest(bad=bad), self.assertRaises(ConfigError):
                parse_config(self._with("label", **bad))

    def test_normalizes_label_types(self):
        cfg = parse_config(self._with("label", a=3, T=12.0))
        self.assertEqual((cfg["label"]["a"], cfg["label"]["T"]), (3.0, 12))
        self.assertIsInstance(cfg["label"]["T"], int)

    def test_rejects_non_finite_and_non_object_sections(self):
        with self.assertRaises(ConfigError):
            parse_config(self._with("label", T=float("inf")))
        for key in ("lock", "rujukan", "report", "label"):
            with self.subTest(section=key), self.assertRaises(ConfigError):
                parse_config(self._with(None, **{key: "bukan objek"}))
        with self.assertRaises(ConfigError):
            parse_config(["bukan", "objek"])

    def test_rejects_bad_name(self):
        for bad in ("Tesis", "a b", "", "../x", "tesis\n"):
            with self.subTest(bad=bad), self.assertRaises(ConfigError):
                parse_config(self._with(None, name=bad))

    def test_rejects_bad_expected_valid(self):
        for bad in ({"dev": 1}, {"dev": 1, "calib": 2, "test": -1}, {"dev": 1.5, "calib": 2, "test": 3}):
            with self.subTest(bad=bad), self.assertRaises(ConfigError):
                parse_config(self._with(None, expected_valid=bad))

    def test_expected_valid_optional(self):
        raw = self._with(None)
        del raw["expected_valid"]
        self.assertIsNone(parse_config(raw)["expected_valid"])

    def test_rejects_incomplete_lock_report_rujukan(self):
        for section, kw in (("lock", {"manifest": "x/M.json"}), ("lock", {"aturan": ""}),
                            ("report", {"sample_seed": "1"}), ("rujukan", {"label": None})):
            with self.subTest(section=section, kw=kw), self.assertRaises(ConfigError):
                parse_config(self._with(section, **kw))

    def test_parse_does_not_mutate_input(self):
        raw = self._with(None)
        before = copy.deepcopy(raw)
        parse_config(raw)
        self.assertEqual(raw, before)

    def test_load_config_from_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.json"
            p.write_text(json.dumps(self._with(None, name="x")))
            self.assertEqual(load_config(p)["name"], "x")


if __name__ == "__main__":
    unittest.main()
