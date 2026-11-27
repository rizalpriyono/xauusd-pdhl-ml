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
        self.assertEqual((k.a, k.b, k.ece_bin, k.delta), (2.0, 1.0, 10, 0.1))
        self.assertEqual(k.delta_grid, (0.0, 0.1, 0.2, 0.3))
        self.assertEqual(k.teknis["temperature_rentang"], [0.05, 20.0])
        self.assertEqual(k.teknis["seed_model"], 20261119)
        self.assertEqual(k.teknis["lightgbm"]["max_depth"], [2, 3])

    def test_kunci_dir_bawaan_di_samping_config(self):
        self.assertEqual(kunci.kunci_dir_bawaan(ROOT / "configs" / "tesis.json"), KUNCI_DIR)

    def test_manifest_adendum_mencakup_seluruh_berkas(self):
        sha = kunci.cek_manifest_adendum(KUNCI_DIR)
        self.assertEqual(set(sha), {"kunci_1.json", "kunci_1_fitur.json", "kunci_1_rq2.json",
                                    "KUNCI-1_parameter_label.md", "RENCANA_ANALISIS_RQ2.md",
                                    "ADENDUM_3.md", "kunci_1_adendum_3.json",
                                    "ADENDUM_4.md", "kunci_1_adendum_4.json"})

    def test_muat_kunci_fitur_tambahan_adendum_4(self):
        k = kunci.muat_kunci(CFG, KUNCI_DIR)
        f13, f14, f15 = k.fitur_tambahan
        self.assertEqual([f["nama"] for f in k.fitur_tambahan],
                         ["f13_wick_penolakan", "f14_cmf21_d", "f15_pe48"])
        self.assertEqual((f14["jendela_bar"], f14["bertanda_d"]), (21, True))
        self.assertEqual((f15["jendela_return"], f15["orde_m"], f15["tunda"], f15["bertanda_d"]),
                         (48, 3, 1, False))
        self.assertEqual(len(k.manifest_sha256), 9)
        self.assertEqual(k.manifest_sha256[kunci.ADENDUM_FITUR_JSON],
                         common.sha256_file(KUNCI_DIR / kunci.ADENDUM_FITUR_JSON))
        self.assertAlmostEqual(k.theta, 2 / 3, places=9)              # B1 tidak diubah Adendum 4


class TestAdendumRusak(unittest.TestCase):
    """Adendum 4 (fitur 14-15) hanya dipakai bila rantai manifestnya utuh."""

    def _ditolak(self, rusak, fungsi=None):
        with tempfile.TemporaryDirectory() as tmp:
            d = salin_kunci(tmp)
            rusak(d)
            with self.assertRaises(kunci.KunciError):
                (fungsi or (lambda p: kunci.muat_kunci(CFG, p)))(d)

    def test_adendum_diubah_ditolak(self):
        self._ditolak(lambda d: ubah_json(d / kunci.ADENDUM_FITUR_JSON,
                                          lambda r: r["fitur_tambahan"][1].update(jendela_bar=14)))

    def test_manifest_adendum_hilang_ditolak(self):
        self._ditolak(lambda d: (d / kunci.MANIFEST_ADENDUM).unlink())

    def test_manifest_sebelumnya_berubah_ditolak(self):
        def rusak(d):
            p = d / "MANIFEST_KUNCI_1_ADENDUM_3.json"
            p.write_text(p.read_text() + "\n")
        self._ditolak(rusak, kunci.cek_manifest_adendum)

    def test_manifest_adendum_tanpa_berkas_fitur_ditolak(self):
        self._ditolak(lambda d: ubah_json(d / kunci.MANIFEST_ADENDUM,
                                          lambda r: r["sha256"].pop(kunci.ADENDUM_FITUR_JSON)),
                      kunci.cek_manifest_adendum)

    def test_manifest_adendum_tanpa_rantai_ke_manifest_dasar_ditolak(self):
        self._ditolak(lambda d: ubah_json(d / kunci.MANIFEST_ADENDUM,
                                          lambda r: r["manifest_sebelumnya"].pop(kunci.MANIFEST)),
                      kunci.cek_manifest_adendum)

    def test_isi_adendum_tidak_valid_ditolak(self):
        rusak = {
            "nomor_adendum": lambda r: r.update(adendum=5),
            "nomor_fitur": lambda r: r["fitur_tambahan"][0].update(no=12),
            "nama_kosong": lambda r: r["fitur_tambahan"][2].update(nama=""),
            "jumlah_fitur": lambda r: r.update(jumlah_fitur_akhir=14),
            "kurang_satu": lambda r: r["fitur_tambahan"].pop(),
            "bukan_daftar": lambda r: r.update(fitur_tambahan=None),
            "theta_beda": lambda r: r["B1"].update(theta=0.5),
        }
        for nama, mutate in rusak.items():
            with self.subTest(nama), tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp) / "adendum.json"
                p.write_text((KUNCI_DIR / kunci.ADENDUM_FITUR_JSON).read_text())
                ubah_json(p, mutate)
                with self.assertRaises(kunci.KunciError):
                    kunci.load_adendum_fitur(p, 0.6666666667)

    def test_isi_adendum_repo_valid(self):
        raw = kunci.load_adendum_fitur(KUNCI_DIR / kunci.ADENDUM_FITUR_JSON, 0.6666666667)
        self.assertEqual(raw["jumlah_fitur_akhir"], 15)


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
            "ece_bin_nol": lambda r: r["rq2_1"].update(ece_bin=0),
            "ece_bin_pecahan": lambda r: r["rq2_1"].update(ece_bin=10.5),
            "delta_grid_bukan_angka": lambda r: r["pendukung"].update(delta_grid=[0.1, "x"]),
            "delta_grid_null": lambda r: r["pendukung"].update(delta_grid=None),
            "rentang_tau_bool": lambda r: r["teknis"].update(temperature_rentang=[True, 20.0]),
            "rentang_tau_terbalik": lambda r: r["teknis"].update(temperature_rentang=[20.0, 0.05]),
            "rentang_tau_nol": lambda r: r["teknis"].update(temperature_rentang=[0.0, 20.0]),
            "rentang_tau_hilang": lambda r: r["teknis"].pop("temperature_rentang"),
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
