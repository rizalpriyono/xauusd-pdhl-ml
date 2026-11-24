"""Tes parameter Kunci-1 dan cek data masukan pemodelan Gradient Boosting."""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model.gradient_boosting import jalur, kunci  # noqa: E402
from pdhl_data import common  # noqa: E402
from pdhl_data.config import load_config  # noqa: E402

KUNCI_DIR = ROOT / "configs" / "kunci_1"
CFG = load_config(ROOT / "configs" / "tesis.json")


def salin_kunci(tmp):
    dst = Path(tmp) / "kunci_1"
    shutil.copytree(KUNCI_DIR, dst)
    return dst


def ubah_json(path, mutate):
    raw = json.loads(path.read_text())
    mutate(raw)
    path.write_text(json.dumps(raw))


class TestKunciRepo(unittest.TestCase):
    def test_salinan_repo_cocok_dengan_manifest(self):
        sha = kunci.cek_manifest(KUNCI_DIR)
        self.assertEqual(set(sha), {"kunci_1.json", "kunci_1_fitur.json", "kunci_1_rq2.json",
                                    "KUNCI-1_parameter_label.md", "RENCANA_ANALISIS_RQ2.md"})

    def test_muat_kunci_nilai_terkunci(self):
        k = kunci.muat_kunci(CFG, KUNCI_DIR)
        self.assertAlmostEqual(k.theta, 2 / 3, places=9)
        self.assertEqual((k.laplace, k.clip, k.toleransi_seri), (1, 1e-15, 1e-6))
        self.assertEqual(k.k, 0)
        self.assertEqual(k.teknis["seed_model"], 20261119)
        self.assertEqual(k.teknis["lightgbm"]["max_depth"], [2, 3])

    def test_kunci_dir_bawaan_di_samping_config(self):
        self.assertEqual(kunci.kunci_dir_bawaan(ROOT / "configs" / "tesis.json"), KUNCI_DIR)


class TestKunciRusak(unittest.TestCase):
    def test_berkas_diubah_ditolak_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = salin_kunci(tmp)
            ubah_json(d / "kunci_1_fitur.json", lambda r: r["B1"].update(theta=0.5))
            with self.assertRaises(kunci.KunciError):
                kunci.cek_manifest(d)

    def test_berkas_hilang_ditolak(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = salin_kunci(tmp)
            (d / "RENCANA_ANALISIS_RQ2.md").unlink()
            with self.assertRaises(kunci.KunciError):
                kunci.cek_manifest(d)

    def test_manifest_tanpa_berkas_yang_dimuat_ditolak(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = salin_kunci(tmp)
            ubah_json(d / kunci.MANIFEST, lambda r: r["sha256"].pop("kunci_1_rq2.json"))
            with self.assertRaises(kunci.KunciError):
                kunci.cek_manifest(d)

    def test_label_config_berbeda_ditolak(self):
        cfg = {**CFG, "label": {**CFG["label"], "a": 3.0}}
        with self.assertRaises(kunci.KunciError):
            kunci.muat_kunci(cfg, KUNCI_DIR)

    def test_theta_tidak_valid_ditolak(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "k.json"
            p.write_text(json.dumps({"B1": {"theta": 1.5}, "jumlah_fitur_akhir": 15,
                                     "fitur_tambahan": [{}, {}, {}]}))
            with self.assertRaises(kunci.KunciError):
                kunci.load_fitur(p)

    def test_jumlah_fitur_harus_12_tambah_tambahan(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "k.json"
            p.write_text(json.dumps({"B1": {"theta": 0.6}, "jumlah_fitur_akhir": 14,
                                     "fitur_tambahan": [{}, {}, {}]}))
            with self.assertRaises(kunci.KunciError):
                kunci.load_fitur(p)

    def test_rq2_menyimpang_ditolak(self):
        rusak = {
            "kelas_salah": lambda r: r.update(kelas=["Netral", "Reversal", "Continuation"]),
            "bootstrap_kecil": lambda r: r["rq2_1"]["bootstrap"].update(n=100),
            "clip_salah": lambda r: r["rq2_1"].update(clip_probabilitas=0.0),
            "laplace_salah": lambda r: r["rq2_1"].update(B1_laplace_bila_sel_kosong=-1),
            "toleransi_salah": lambda r: r["teknis"].update(toleransi_seri_logloss=-1),
            "delta_bukan_grid": lambda r: r["pendukung"].update(delta_utama=0.15),
        }
        for name, mutate in rusak.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp) / "rq2.json"
                p.write_text((KUNCI_DIR / "kunci_1_rq2.json").read_text())
                ubah_json(p, mutate)
                with self.assertRaises(kunci.KunciError):
                    kunci.load_rq2(p)


class TestDataInput(unittest.TestCase):
    def _run(self, tmp):
        paths = jalur.ModelPaths(Path(tmp) / "tesis")
        paths.input.mkdir(parents=True)
        for i, n in enumerate(jalur.INPUT_FILES):
            (paths.input / n).write_text(f"session_date\n2016-01-0{i + 1}\n")
        return paths

    def _acuan(self, paths, **ganti):
        files = {f"data/{n}": common.sha256_file(paths.input / n) for n in jalur.INPUT_FILES}
        files.update(ganti)
        return {"files": files}

    def test_folder_keluaran_terpisah_dari_data_pipeline(self):
        paths = jalur.model_paths(CFG, "/tmp/runs")
        self.assertEqual(paths.input, Path("/tmp/runs/tesis/data").resolve())
        self.assertEqual(paths.data, Path("/tmp/runs/tesis/model/gradient_boosting/data").resolve())
        self.assertEqual(paths.output.name, "output")
        self.assertNotEqual(paths.data, paths.input)

    def test_hash_cocok(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._run(tmp)
            sha = jalur.cek_data_input(paths, self._acuan(paths))
            self.assertEqual(set(sha), set(jalur.INPUT_FILES))

    def test_hash_beda_ditolak(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._run(tmp)
            acuan = self._acuan(paths, **{"data/bars_devcalib.csv": "0" * 64})
            with self.assertRaises(jalur.DataInputBerbeda) as ctx:
                jalur.cek_data_input(paths, acuan)
            self.assertIn("bars_devcalib.csv", str(ctx.exception))
            self.assertIn("pdhl_data run", str(ctx.exception))

    def test_tanpa_acuan_hanya_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._run(tmp)
            self.assertEqual(len(jalur.cek_data_input(paths, None)), len(jalur.INPUT_FILES))

    def test_berkas_hilang(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._run(tmp)
            (paths.input / "labels_dev.csv").unlink()
            with self.assertRaises(FileNotFoundError):
                jalur.cek_data_input(paths, None)

    def test_info_sumber(self):
        info = jalur.info_sumber({"a.csv": "x"})
        self.assertEqual(info["sumber_sha256"], {"a.csv": "x"})
        self.assertIn("versi_pdhl_data", info)
        self.assertIn("commit", info)


if __name__ == "__main__":
    unittest.main()
