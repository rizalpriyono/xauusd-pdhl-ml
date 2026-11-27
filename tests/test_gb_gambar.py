"""Tes tahap `gambar`: format angka, ringkasan data, kurva belajar, dan 22 gambar dari data asli.

Fungsi format dan ringkasan murni jalan tanpa matplotlib (CI). Tes data asli dilewati bila
matplotlib, LightGBM, atau `runs/tesis` (sesuai references/tesis.json) tidak ada.
"""
import importlib.util
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model.gradient_boosting import gambar, jalur, kunci  # noqa: E402
from model.gradient_boosting import model_akhir as ma  # noqa: E402
from model.gradient_boosting.gambar import data, gaya, kurva_belajar, proses_3_4  # noqa: E402
from pdhl_data.config import load_config  # noqa: E402
from test_gb_model_akhir import BERKAS_ASLI, RUN_ASLI, run_asli_sesuai_acuan  # noqa: E402

KUNCI = kunci.muat_kunci(load_config(ROOT / "configs" / "tesis.json"), ROOT / "configs" / "kunci_1")
ADA_VIZ = all(importlib.util.find_spec(m) is not None for m in ("matplotlib", "lightgbm"))
R, C, N = "Reversal", "Continuation", "Netral"
M = {"m_rev": 0.5, "m_cont": -0.25}


def baris(i, tanggal, label, r_rev, r_cont, b1=False):
    return {"event_id": f"e{i}", "session_date": tanggal, "label": label, "R_rev": r_rev,
            "R_cont": r_cont, "b1": b1}


class TestFormat(unittest.TestCase):
    def test_angka_koma_dan_minus(self):
        self.assertEqual(gaya.angka(0.92327, 4), "0,9233")
        self.assertEqual(gaya.angka(-0.0465, 4), "−0,0465")
        self.assertEqual(gaya.angka(-0.00001, 2), "0,00")          # tanpa "−0,00"

    def test_bulat_dan_persen(self):
        self.assertEqual(gaya.bulat(1841), "1.841")
        self.assertEqual(gaya.bulat(59093), "59.093")
        self.assertEqual(gaya.persen(0.094), "9,4%")


class TestRingkasan(unittest.TestCase):
    ROWS = [baris(0, "2019-01-07", R, 2.0, -1.0), baris(1, "2019-03-05", C, -1.0, 0.5, True),
            baris(2, "2020-02-03", N, 0.4, -0.2), baris(3, "2020-06-01", R, 2.0, -1.0, True)]
    P = [[0.6, 0.2, 0.2], [0.05, 0.9, 0.05], [0.2, 0.6, 0.2], [0.25, 0.55, 0.2]]

    def test_label_per_tahun(self):
        self.assertEqual(data.label_per_tahun(self.ROWS),
                         {2019: {R: 1, C: 1, N: 0}, 2020: {R: 1, C: 0, N: 1}})

    def test_rekomendasi_dan_r_kumulatif(self):
        reks = data.rekomendasi_strategi(self.ROWS, self.P, 2.0, 1.0, M, 0.1)
        self.assertEqual(reks["S0"], [R] * 4)
        self.assertEqual(reks["S1"], [ma.ABSTAIN, R, ma.ABSTAIN, R])
        # E_rev = 1,1 / −0,775 / −0,1 / 0,05; E_cont = −0,55 / 0,3875 / 0,05 / −0,025 (δ = 0,1)
        self.assertEqual(reks["SM"], [R, C, ma.ABSTAIN, ma.ABSTAIN])
        self.assertEqual(data.r_kumulatif(self.ROWS, reks["S0"]), [2.0, 1.0, 1.4, 3.4])

    def test_sm_per_tahun_dihitung_tangan(self):
        h = data.sm_per_tahun(self.ROWS, self.P, 2.0, 1.0, M, 0.1)
        self.assertEqual([x["tahun"] for x in h], [2019, 2020])
        # 2019: Reversal (+2) dan Continuation (+0,5) -> cakupan 100%, rata-rata 1,25R
        self.assertEqual((h[0]["cakupan"], h[0]["r_per_rekomendasi"]), (1.0, 1.25))
        # 2020: keduanya abstain -> tanpa rekomendasi
        self.assertEqual((h[1]["cakupan"], h[1]["r_per_rekomendasi"]), (0.0, None))

    def test_e_rev_cocok_dengan_nilai_harapan(self):
        for got, exp in zip(data.e_rev(self.P, 2.0, 1.0, M), (1.1, -0.775, -0.1, 0.05)):
            self.assertAlmostEqual(got, exp, places=12)

    def test_ringkas_pohon_termasuk_pohon_satu_daun(self):
        dump = {"feature_names": ["f08_waktu_sejak_open_sesi", "f04_sesi=Asia"],
                "tree_info": [
                    {"tree_structure": {"split_feature": 0, "split_gain": 3.0, "threshold": 16.5,
                                        "internal_count": 10,
                                        "left_child": {"leaf_index": 0, "leaf_value": -1.0,
                                                       "leaf_count": 6},
                                        "right_child": {"split_feature": 1, "split_gain": 1.0,
                                                        "threshold": 0.5, "internal_count": 4,
                                                        "left_child": {"leaf_index": 1,
                                                                       "leaf_value": 0.1,
                                                                       "leaf_count": 1},
                                                        "right_child": {"leaf_index": 2,
                                                                        "leaf_value": 0.2,
                                                                        "leaf_count": 3}}}},
                    {"tree_structure": {"leaf_value": -0.5, "leaf_count": 10}}]}   # satu daun
        s = data.ringkas_pohon(dump)
        self.assertEqual(s["jumlah_pohon"], 2)
        self.assertEqual(sorted(s["isi_daun"]), [1, 3, 6, 10])
        self.assertEqual(s["split"], {"f08_waktu_sejak_open_sesi": 1, "f04_sesi": 1})
        self.assertEqual(s["gain"], {"f08_waktu_sejak_open_sesi": 3.0, "f04_sesi": 1.0})

    def test_label_simpul(self):
        nama = ["f08_waktu_sejak_open_sesi", "f04_sesi=Asia"]
        biasa = {"split_feature": 0, "threshold": 16.5, "internal_count": 1841}
        onehot = {"split_feature": 1, "threshold": 0.5, "internal_count": 233}
        self.assertEqual(proses_3_4._label_node(biasa, nama),
                         "f08 waktu sejak open ≤ 16,5?\nn = 1.841")
        self.assertEqual(proses_3_4._label_node(onehot, nama), "f04 sesi bukan Asia?\nn = 233")

    def test_daftar_22_gambar_unik(self):
        nama = [n for n, _ in gambar.DAFTAR]
        self.assertEqual(len(nama), 22)
        self.assertEqual(len(set(nama)), 22)
        self.assertTrue(all(n.endswith(".png") for n in nama))

    def test_kurva_gabung_dibobot_jumlah_sentuhan(self):
        lipatan = [{"n_val": 1, "ll_val": [1.0, 0.5]}, {"n_val": 3, "ll_val": [0.6, 0.2]}]
        for got, exp in zip(kurva_belajar.gabung(lipatan, "ll_val", "n_val"), (0.7, 0.275)):
            self.assertAlmostEqual(got, exp, places=12)

    def test_kurva_harus_sama_dengan_skor_walk_forward(self):
        skor = {"d2_n2_m50": 0.5, "d2_n3_m50": 0.4}
        kurva_belajar.cek_cocok([0.9, 0.5, 0.4], skor, 2, 50, [2, 3])          # cocok: lolos
        with self.assertRaises(ValueError):
            kurva_belajar.cek_cocok([0.9, 0.5, 0.41], skor, 2, 50, [2, 3])

    def test_akurasi_pembanding_kelas_terbanyak(self):
        latih = [baris(i, "2019-01-07", lab, 0, 0, b1)
                 for i, (lab, b1) in enumerate([(C, False), (C, False), (R, True), (R, True),
                                                (N, False)])]
        val = [baris(9, "2020-01-06", C, 0, 0, False), baris(8, "2020-01-07", R, 0, 0, True),
               baris(7, "2020-01-08", N, 0, 0, True)]
        # B0: kelas terbanyak C dan R seri (2-2) -> argmax ambil urutan CLASSES pertama (Reversal)
        # B1 wick: kelompok b1 -> R (2/2); kelompok non-b1 -> C (2/3)
        acc = kurva_belajar.akurasi_pembanding(latih, val, laplace=1)
        self.assertAlmostEqual(acc["B0"], 1 / 3)
        self.assertAlmostEqual(acc["B1"], 2 / 3)

    def test_masukan_hilang_ditolak(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError) as ctx:
                data.muat(jalur.ModelPaths(Path(tmp) / "uji"), KUNCI)
            self.assertIn("jalankan tahap", str(ctx.exception))


@unittest.skipUnless(ADA_VIZ and run_asli_sesuai_acuan()
                     and (RUN_ASLI.output / "model_akhir_rekonsiliasi.json").exists(),
                     "butuh matplotlib, lightgbm, dan runs/tesis sampai tahap model-akhir")
class TestGambarDataAsli(unittest.TestCase):
    """Semua gambar dibuat dari salinan keluaran asli di folder sementara."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.paths = jalur.ModelPaths(Path(cls.tmp.name) / "tesis")
        tambahan = (RUN_ASLI.output / "pembanding_rekonsiliasi.json",
                    RUN_ASLI.output / "model_akhir_rekonsiliasi.json",
                    RUN_ASLI.data / "model_akhir_calib.csv", RUN_ASLI.model / ma.MODEL_BERKAS)
        for p in BERKAS_ASLI + tambahan:
            tujuan = cls.paths.run_root / p.relative_to(RUN_ASLI.run_root)
            tujuan.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, tujuan)
        cls.hasil = gambar.run_gambar(cls.paths, KUNCI)
        cls.D = data.muat(cls.paths, KUNCI)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_22_png_tertulis(self):
        self.assertEqual(len(self.hasil["berkas"]), 22)
        for nama in self.hasil["berkas"]:
            p = self.paths.gambar / nama
            self.assertGreater(p.stat().st_size, 10_000, nama)
            self.assertEqual(p.read_bytes()[:8], b"\x89PNG\r\n\x1a\n", nama)

    def test_kurva_cakupan_cocok_dengan_rekonsiliasi(self):
        a, b, m, _ = data.parameter_keputusan(self.D)
        rek = self.D["recon"]["model_akhir"]
        seri = (("cakupan_oof_dev", self.D["oof_rows"], self.D["P_oof"]),
                ("pratinjau_calib", self.D["calib"], data.peluang_calib(self.D)))
        for kunci_rek, rows, P in seri:
            h = {x["delta"]: x for x in ma.pratinjau_keputusan(rows, P, a, b, m, KUNCI.delta_grid)}
            for ref in rek[kunci_rek]["SM"]:
                for k in ("cakupan", "n_rev", "n_cont", "r_per_event"):
                    self.assertAlmostEqual(h[ref["delta"]][k], ref[k], places=9,
                                           msg=f"{kunci_rek} δ={ref['delta']} {k}")

    def test_sm_per_tahun_konsisten_dengan_gabungan(self):
        a, b, m, delta = data.parameter_keputusan(self.D)
        h = data.sm_per_tahun(self.D["oof_rows"], self.D["P_oof"], a, b, m, delta)
        self.assertEqual(sum(x["n"] for x in h), 1152)
        cak = sum(x["cakupan"] * x["n"] for x in h) / 1152
        ref = next(x for x in self.D["recon"]["model_akhir"]["cakupan_oof_dev"]["SM"]
                   if x["delta"] == delta)
        self.assertAlmostEqual(cak, ref["cakupan"], places=12)

    def test_tidak_ada_figur_tertinggal(self):
        import matplotlib.pyplot as plt
        self.assertEqual(plt.get_fignums(), [])
        self.assertFalse(self.paths.gambar.with_name("gambar.sementara").exists())

    def test_model_berbeda_dari_rekonsiliasi_ditolak(self):
        with tempfile.TemporaryDirectory() as tmp:
            salinan = jalur.ModelPaths(Path(tmp) / "tesis")
            shutil.copytree(self.paths.run_root, salinan.run_root)
            p = salinan.model / ma.MODEL_BERKAS
            p.write_text(p.read_text() + "\n")
            with self.assertRaises(ValueError):
                data.muat(salinan, KUNCI)

    def test_kurva_belajar_data_asli(self):
        k = kurva_belajar.kurva_terpilih(self.D)                # termasuk cek skor 100 dan 300
        self.assertEqual(len(k["ll_val"]), 300)
        self.assertLess(k["ll_latih"][299], k["ll_latih"][99])  # latih terus turun
        self.assertGreater(k["ll_val"][299], k["ll_val"][99])   # validasi naik: overfitting
        cont = sum(r["label"] == C for r in self.D["oof_rows"]) / len(self.D["oof_rows"])
        self.assertAlmostEqual(k["akurasi_pembanding"]["B0"], cont, places=12)

    def test_struktur_model(self):
        s = data.struktur_model(self.D["model_path"])
        self.assertEqual(s["jumlah_pohon"], 300)
        self.assertEqual(len(s["isi_daun"]), 1161)          # 261 pohon × 4 daun + 39 × 3
        self.assertEqual(sum(1 for c in s["isi_daun"] if c < 50), 11)   # run Adendum 4
        self.assertEqual(set(s["gain"]) - {"f04_sesi", "f05_hari"},
                         set(data.NUMERIK))           # semua 15 fitur terpakai


if __name__ == "__main__":
    unittest.main()
