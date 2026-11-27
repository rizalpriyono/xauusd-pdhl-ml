"""Tes perbandingan rekonsiliasi baru dengan hasil lama dan CLI `python -m model.gradient_boosting`."""
import contextlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gb_sintetis import make_run  # noqa: E402
from model.gradient_boosting import bandingkan as bd  # noqa: E402
from model.gradient_boosting import jalur  # noqa: E402
from model.gradient_boosting.__main__ import main  # noqa: E402
from pdhl_data import common  # noqa: E402

LAMA = ROOT / "references" / "gradient_boosting_lama"
CONFIG = str(ROOT / "configs" / "tesis.json")


class TestRatakanDanStatus(unittest.TestCase):
    def test_ratakan_bersarang_dan_abaikan_jejak_audit(self):
        obj = {"a": 1, "b": {"c": [10, {"d": 2.5}]}, "sumber_sha256": {"x": "h"},
               "lingkungan": {"python": "3.10"}, "commit": None}
        self.assertEqual(bd.ratakan(obj), {"a": 1, "b.c[0]": 10, "b.c[1].d": 2.5})

    def test_status(self):
        self.assertEqual(bd.status(1, 1), "IDENTIK")
        self.assertEqual(bd.status(1, 1.0), "IDENTIK")
        self.assertEqual(bd.status("x", "x"), "IDENTIK")
        self.assertEqual(bd.status(1000.0, 1000.0 + 5e-7), "BEDA_KECIL")
        self.assertEqual(bd.status(0.5, 0.5 + 5e-10), "BEDA_KECIL")
        self.assertEqual(bd.status(0.9233, 0.9234), "BEDA")
        self.assertEqual(bd.status("d2_n100_m50", "d2_n100_m20"), "BEDA")
        self.assertEqual(bd.status(True, 1), "BEDA")
        self.assertEqual(bd.status(float("nan"), float("nan")), "IDENTIK")
        self.assertEqual(bd.status(float("nan"), 1.0), "BEDA")

    def test_bandingkan_kunci_satu_sisi(self):
        rows = bd.bandingkan({"a": 1, "b": 2}, {"a": 1, "c": 3})
        self.assertEqual([(r["kunci"], r["status"]) for r in rows],
                         [("a", "IDENTIK"), ("b", "HANYA_LAMA"), ("c", "HANYA_BARU")])

    def test_acuan_lama_identik_dengan_dirinya(self):
        for nama in bd.BERKAS:
            lama = json.loads((LAMA / nama).read_text())
            rows = bd.bandingkan(lama, lama)
            self.assertTrue(rows)
            self.assertTrue(all(r["status"] == "IDENTIK" for r in rows), nama)

    def test_sha256_acuan_lama_sesuai_asal(self):
        asal = json.loads((LAMA / bd.ASAL).read_text())
        self.assertEqual(asal["config"], "tesis")
        for nama in bd.BERKAS:
            self.assertEqual(common.sha256_file(LAMA / nama), asal["sha256"][nama], nama)

    def test_run_bandingkan_menulis_laporan(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = jalur.ModelPaths(Path(tmp) / "tesis")
            paths.output.mkdir(parents=True)
            lama = json.loads((LAMA / "fitur_rekonsiliasi.json").read_text())
            baru = {**lama, "dev_sum_f02_rentang_hari_sebelumnya":
                    lama["dev_sum_f02_rentang_hari_sebelumnya"] + 1e-7,
                    "dev_b1_terpenuhi": lama["dev_b1_terpenuhi"] + 1,
                    "versi_pdhl_data": "1.1.0"}
            (paths.output / "fitur_rekonsiliasi.json").write_text(json.dumps(baru))
            ringkasan = bd.run_bandingkan(paths, LAMA, "tesis")
            c = ringkasan["fitur_rekonsiliasi.json"]
            self.assertEqual((c["BEDA_KECIL"], c["BEDA"], c["HANYA_BARU"]), (1, 1, 0))
            self.assertIsNone(ringkasan["walk_forward_rekonsiliasi.json"])
            teks = (paths.output / "bandingkan_lama.md").read_text()
            self.assertIn("dev_b1_terpenuhi", teks)
            self.assertIn("belum dijalankan", teks)
            self.assertIn("BEDA_KECIL 1", bd.ringkas(ringkasan))

    def test_acuan_lama_diubah_ditolak(self):
        with tempfile.TemporaryDirectory() as tmp:
            acuan = Path(tmp) / "lama"
            shutil.copytree(LAMA, acuan)
            p = acuan / "walk_forward_rekonsiliasi.json"
            p.write_text(p.read_text().replace("d2_n100_m50", "d3_n300_m20"))
            paths = jalur.ModelPaths(Path(tmp) / "tesis")
            with self.assertRaises(bd.AcuanLamaRusak):
                bd.run_bandingkan(paths, acuan, "tesis")

    def test_konfigurasi_lain_dilewati(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = jalur.ModelPaths(Path(tmp) / "sensitivitas")
            self.assertIsNone(bd.run_bandingkan(paths, LAMA, "sensitivitas"))
            self.assertIsNone(bd.run_bandingkan(paths, Path(tmp) / "tidak_ada", "tesis"))
            self.assertIn("dilewati", bd.ringkas(None))


def jalankan_cli(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = main(list(argv))
    return rc, out.getvalue(), err.getvalue()


class TestCli(unittest.TestCase):
    def _run(self, tmp):
        paths = jalur.ModelPaths(Path(tmp) / "tesis")
        make_run(paths, n_sessions=60)
        return paths

    def test_fitur_pembanding_bandingkan(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._run(tmp)
            umum = ("--config", CONFIG, "--runs-dir", tmp, "--reference", str(Path(tmp) / "x.json"),
                    "--tanpa-acuan")
            for tahap in ("fitur", "pembanding", "bandingkan"):
                rc, out, err = jalankan_cli(tahap, *umum)
                self.assertEqual(rc, 0, err)
                self.assertIn(f"[{tahap}]", out)
                if tahap != "bandingkan":         # tanpa acuan: hanya peringatan, tetap jalan
                    self.assertIn("[peringatan]", err)
            for nama in ("features_dev.csv", "pembanding_oof_dev.csv"):
                self.assertTrue((paths.data / nama).exists(), nama)
            for nama in ("laporan_fitur.md", "laporan_pembanding.md", "bandingkan_lama.md"):
                self.assertTrue((paths.output / nama).exists(), nama)

    def test_data_berbeda_dari_acuan_gagal(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._run(tmp)
            acuan = Path(tmp) / "acuan.json"
            acuan.write_text(json.dumps({"files": {}}))
            rc, _, err = jalankan_cli("fitur", "--config", CONFIG, "--runs-dir", tmp,
                                      "--reference", str(acuan))
            self.assertEqual(rc, 1)
            self.assertIn("DataInputBerbeda", err)

    def test_acuan_bawaan_menolak_data_sintetis(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._run(tmp)
            rc, _, err = jalankan_cli("fitur", "--config", CONFIG, "--runs-dir", tmp)
            self.assertEqual(rc, 1)
            self.assertIn("pdhl_data run", err)

    def test_atr_kosong_gagal_dengan_pesan(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._run(tmp)
            rows = common.read_csv(paths.input / "bars_devcalib.csv")
            rows[0]["atr14_h1"] = ""
            common.write_csv(paths.input / "bars_devcalib.csv", rows, list(rows[0]))
            rc, _, err = jalankan_cli("fitur", "--config", CONFIG, "--runs-dir", tmp,
                                      "--reference", str(Path(tmp) / "x.json"), "--tanpa-acuan")
            self.assertEqual(rc, 1)
            self.assertIn("AtrKosong", err)

    def test_acuan_tidak_ada_gagal_tanpa_opsi(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._run(tmp)
            rc, _, err = jalankan_cli("fitur", "--config", CONFIG, "--runs-dir", tmp,
                                      "--reference", str(Path(tmp) / "x.json"))
            self.assertEqual(rc, 1)
            self.assertIn("--tanpa-acuan", err)
            self.assertFalse(paths.root.exists())

    def test_semua_berurutan(self):
        dipanggil = []

        def palsu(tahap, *args):
            dipanggil.append(tahap)
            return tahap

        with tempfile.TemporaryDirectory() as tmp:
            self._run(tmp)
            with mock.patch("model.gradient_boosting.__main__._jalankan", side_effect=palsu):
                rc, _, err = jalankan_cli("semua", "--config", CONFIG, "--runs-dir", tmp,
                                          "--tanpa-acuan", "--reference", str(Path(tmp) / "x"))
        self.assertEqual(rc, 0, err)
        self.assertEqual(dipanggil, ["fitur", "pembanding", "walk-forward", "model-akhir",
                                     "bandingkan"])

    def test_model_akhir_tanpa_tahap_sebelumnya_gagal(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._run(tmp)
            rc, _, err = jalankan_cli("model-akhir", "--config", CONFIG, "--runs-dir", tmp,
                                      "--reference", str(Path(tmp) / "x.json"), "--tanpa-acuan")
            self.assertEqual(rc, 1)
            self.assertIn("[galat] FileNotFoundError", err)
            self.assertIn("fitur", err)
            self.assertFalse(paths.model.exists())

    def test_tahap_tidak_dikenal(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["latih", "--config", CONFIG])


if __name__ == "__main__":
    unittest.main()
