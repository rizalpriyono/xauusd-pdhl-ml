"""Tes reproduksi end-to-end: data mentah asli -> berlabel, dibandingkan dengan `references/*.json`.

Dilewati bila `data/raw/XAUUSD_H1.csv` tidak ada (misalnya di CI, karena data mentah tidak
disimpan di repo). Setiap konfigurasi butuh sekitar 3 detik.
"""
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pdhl_data import common  # noqa: E402
from pdhl_data.__main__ import main  # noqa: E402
from pdhl_data.config import load_config  # noqa: E402
from pdhl_data.pipeline import load_reference, run, run_paths, verify  # noqa: E402

RAW = ROOT / "data" / "raw" / "XAUUSD_H1.csv"


def _quiet(fn, *args, **kw):
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return fn(*args, **kw)


@unittest.skipUnless(RAW.exists(), "data/raw/XAUUSD_H1.csv tidak ada")
class TestReproduksi(unittest.TestCase):
    def test_all_configs_match_reference(self):
        for cfg_path in sorted((ROOT / "configs").glob("*.json")):
            with self.subTest(config=cfg_path.stem), tempfile.TemporaryDirectory() as tmp:
                rc = _quiet(main, ["run", "--config", str(cfg_path), "--raw", str(RAW),
                                   "--runs-dir", tmp])
                self.assertEqual(rc, 0, "verifikasi terhadap references/ gagal")
                cfg = load_config(cfg_path)
                paths = run_paths(cfg, tmp)
                self.assertTrue((paths.output / "laporan_label.md").exists())
                self.assertTrue((paths.de_output / "laporan_data_engineering.md").exists())
                # data/ hanya berisi baris s.d. 2024-12-31 (guard lolos untuk setiap berkas)
                for p in paths.data.glob("*.csv"):
                    common.load_model_csv(p)
                with self.assertRaises(common.TestDataLeak):
                    common.load_model_csv(paths.lock / "events_test.csv")

    def test_partial_steps_and_verify_command(self):
        cfg_path = ROOT / "configs" / "tesis.json"
        cfg = load_config(cfg_path)
        with tempfile.TemporaryDirectory() as tmp:
            _quiet(run, cfg, RAW, tmp, steps=("events",), report=False)
            results = verify(run_paths(cfg, tmp), load_reference(ROOT / "references" / "tesis.json"))
            status = {name: s for name, s, _ in results}
            self.assertEqual(status["de_output/gold_pdhl_events.csv"], "OK")
            self.assertEqual(status["data/labels_dev.csv"], "HILANG")
            rc = _quiet(main, ["verify", "--config", str(cfg_path), "--runs-dir", tmp])
            self.assertEqual(rc, 1)
            # lengkapi split + labels, lalu tahap labels saja tidak butuh data mentah
            _quiet(run, cfg, RAW, tmp, steps=("split", "labels"), report=False)
            rc = _quiet(main, ["run", "--config", str(cfg_path), "--runs-dir", tmp, "--steps",
                               "labels", "--raw", str(Path(tmp) / "tidak_ada.csv"), "--no-report"])
            self.assertEqual(rc, 0)
            rc = _quiet(main, ["verify", "--config", str(cfg_path), "--runs-dir", tmp])
            self.assertEqual(rc, 0)

    def test_tampered_reference_is_detected(self):
        cfg = load_config(ROOT / "configs" / "tesis.json")
        ref = load_reference(ROOT / "references" / "tesis.json")
        with tempfile.TemporaryDirectory() as tmp:
            _quiet(run, cfg, RAW, tmp, steps=("events",), report=False)
            bad = json.loads(json.dumps(ref))
            bad["files"]["de_output/silver_sessions.csv"] = "0" * 64
            bad["rekonsiliasi"]["events"]["events_valid"] += 1
            status = {name: s for name, s, _ in verify(run_paths(cfg, tmp), bad)}
            self.assertEqual(status["de_output/silver_sessions.csv"], "BEDA")
            self.assertEqual(status["rekonsiliasi.events"], "BEDA")


class TestCli(unittest.TestCase):
    def test_verify_without_reference_fails_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc = _quiet(main, ["verify", "--config", str(ROOT / "configs" / "tesis.json"),
                               "--runs-dir", tmp, "--reference", str(Path(tmp) / "tidak_ada.json")])
            self.assertEqual(rc, 2)

    def test_unknown_step_rejected(self):
        cfg = load_config(ROOT / "configs" / "tesis.json")
        with self.assertRaises(ValueError):
            run(cfg, RAW, "runs", steps=("fitur",))

    def test_check_raw(self):
        from pdhl_data.pipeline import check_raw
        with self.assertRaises(FileNotFoundError):
            check_raw(ROOT / "tidak_ada.csv", None)
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw.csv"
            raw.write_text("x\n")
            self.assertIsNone(check_raw(raw, None))
            self.assertIsNone(check_raw(raw, {"raw": {"sha256": common.sha256_file(raw)}}))
            self.assertIn("acuan", check_raw(raw, {"raw": {"sha256": "0" * 64}}))


if __name__ == "__main__":
    unittest.main()
