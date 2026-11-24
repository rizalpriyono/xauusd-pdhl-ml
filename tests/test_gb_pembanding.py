"""Tes tahap 2 baseline non-model B0 dan B1 (RM-23; Kunci-1 Adendum 1-2, keputusan 22 Nov 2026).

Port dari `Pemodelan PDH-PDL/tests/test_pembanding.py`, ditambah tes `run_pembanding`.
"""
import datetime as dt
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gb_sintetis import make_run  # noqa: E402
from model.gradient_boosting import jalur, kunci  # noqa: E402
from model.gradient_boosting import pembanding as pb  # noqa: E402
from model.gradient_boosting.fitur import run_fitur  # noqa: E402
from pdhl_data import common  # noqa: E402
from pdhl_data.config import load_config  # noqa: E402

R, C, N = common.CLASSES
THETA = 2 / 3
KUNCI = kunci.muat_kunci(load_config(ROOT / "configs" / "tesis.json"), ROOT / "configs" / "kunci_1")


def baris(i, tanggal, label, b1=False, akhir=None):
    """Satu event gabungan sintetis (bentuk keluaran `pb.gabung`)."""
    return {"event_id": f"e{i}", "session_date": tanggal, "label": label,
            "label_end_session_date": akhir or tanggal, "f13": 0.9 if b1 else 0.1, "b1": b1}


class TestPeluang(unittest.TestCase):
    def test_b0_adalah_frekuensi_kelas(self):
        rows = [baris(i, "2017-01-02", lab) for i, lab in enumerate([R, C, C, N])]
        b0 = pb.latih_b0(rows)
        self.assertEqual(b0["n"], 4)
        self.assertEqual(b0["jumlah"], {R: 1, C: 2, N: 1})
        self.assertEqual(b0["peluang"], {R: 0.25, C: 0.5, N: 0.25})

    def test_b1_dipisah_menurut_aturan_wick(self):
        labels_ya, labels_tidak = [R, R, C, N], [C, C, C, N, R]
        rows = [baris(i, "2017-01-02", lab, b1=True) for i, lab in enumerate(labels_ya)]
        rows += [baris(10 + i, "2017-01-02", lab) for i, lab in enumerate(labels_tidak)]
        b1 = pb.latih_b1(rows, laplace=1)
        ya, tidak = b1["terpenuhi"], b1["tidak_terpenuhi"]
        self.assertEqual((ya["n"], tidak["n"]), (4, 5))
        self.assertFalse(ya["laplace"] or tidak["laplace"])
        self.assertEqual(ya["peluang"], {R: 0.5, C: 0.25, N: 0.25})
        self.assertAlmostEqual(tidak["peluang"][C], 0.6)

    def test_laplace_hanya_pada_kelompok_yang_kehilangan_kelas(self):
        rows = [baris(i, "2017-01-02", lab, b1=True) for i, lab in enumerate([R, R, C])]
        rows += [baris(10 + i, "2017-01-02", lab) for i, lab in enumerate([R, C, N, N])]
        b1 = pb.latih_b1(rows, laplace=1)
        ya, tidak = b1["terpenuhi"], b1["tidak_terpenuhi"]
        self.assertTrue(ya["laplace"])
        self.assertEqual(ya["jumlah"], {R: 2, C: 1, N: 0})          # jumlah asli tetap dilaporkan
        self.assertEqual(ya["peluang"], {R: 3 / 6, C: 2 / 6, N: 1 / 6})
        self.assertFalse(tidak["laplace"])
        self.assertEqual(tidak["peluang"], {R: 0.25, C: 0.25, N: 0.5})

    def test_kelompok_tanpa_event_menjadi_seragam(self):
        rows = [baris(i, "2017-01-02", lab) for i, lab in enumerate([R, C, N])]
        ya = pb.latih_b1(rows, laplace=1)["terpenuhi"]
        self.assertEqual(ya["n"], 0)
        self.assertTrue(ya["laplace"])
        for k in common.CLASSES:
            self.assertAlmostEqual(ya["peluang"][k], 1 / 3)

    def test_peluang_selalu_berjumlah_satu(self):
        rows = [baris(i, "2017-01-02", lab, b1=i % 2 == 0)
                for i, lab in enumerate([R, C, C, N, C, R, N])]
        model = pb.latih_pembanding(rows, laplace=1)
        for r in rows:
            for p in pb.prediksi(model, r).values():
                self.assertAlmostEqual(sum(p), 1.0)

    def test_prediksi_memakai_kelompok_event(self):
        rows = [baris(0, "2017-01-02", R, b1=True), baris(1, "2017-01-02", C, b1=True),
                baris(2, "2017-01-02", N, b1=True), baris(3, "2017-01-02", C)]
        model = pb.latih_pembanding(rows, laplace=1)
        self.assertEqual(pb.prediksi(model, rows[0])["B1"], [1 / 3, 1 / 3, 1 / 3])
        self.assertEqual(pb.prediksi(model, rows[3])["B1"], [0.25, 0.5, 0.25])   # (0,1,0)+1 → /4
        self.assertEqual(pb.prediksi(model, rows[3])["B0"], [0.25, 0.5, 0.25])


class TestLogLoss(unittest.TestCase):
    def test_log_loss_natural(self):
        self.assertAlmostEqual(pb.log_loss([0.2, 0.5, 0.3], C, clip=1e-15), -math.log(0.5))

    def test_peluang_nol_dipotong_lalu_dinormalkan(self):
        got = pb.log_loss([0.0, 1.0, 0.0], R, clip=1e-15)
        self.assertAlmostEqual(got, -math.log(1e-15 / (1 + 2e-15)))

    def test_menolak_label_asing(self):
        with self.assertRaises(ValueError):
            pb.log_loss([0.2, 0.5, 0.3], "Breakout", clip=1e-15)

    def test_format_selisih_tanpa_nol_negatif(self):
        self.assertEqual(pb._selisih(-0.00001), "0.0000")
        self.assertEqual(pb._selisih(-0.0052), "-0.0052")
        self.assertEqual(pb._selisih(0.0043), "+0.0043")


class TestPembagianData(unittest.TestCase):
    def test_purging_membuang_jendela_label_yang_melewati_batas(self):
        rows = [baris(0, "2017-06-01", R), baris(1, "2018-12-31", C, akhir="2019-01-02"),
                baris(2, "2018-12-28", N), baris(3, "2019-03-01", C)]
        latih, purged = pb.data_latih(rows, dt.date(2019, 1, 1))
        self.assertEqual([r["event_id"] for r in latih], ["e0", "e2"])
        self.assertEqual([r["event_id"] for r in purged], ["e1"])

    def test_batas_purging_inklusif(self):
        batas = dt.date(2019, 1, 1)
        rows = [baris(0, "2018-12-28", C, akhir="2019-01-01"),   # label_end == batas → purge
                baris(1, "2019-01-01", R),                       # session_date == batas → bukan latih
                baris(2, "2018-12-28", N, akhir="2018-12-31")]   # masih di dalam → latih
        latih, purged = pb.data_latih(rows, batas)
        self.assertEqual([r["event_id"] for r in latih], ["e2"])
        self.assertEqual([r["event_id"] for r in purged], ["e0"])

    def test_model_akhir_purging_di_batas_dev_kalibrasi(self):
        rows = [baris(0, "2023-06-01", R), baris(1, "2023-12-29", C, akhir="2024-01-02"),
                baris(2, "2023-12-28", N)]
        akhir = pb.model_akhir(rows, laplace=1)
        self.assertEqual((akhir["n_latih"], akhir["n_purged"]), (2, 1))
        self.assertEqual(akhir["B0"]["jumlah"], {R: 1, C: 0, N: 1})

    def test_log_loss_walk_forward_dihitung_tangan(self):
        rows = [baris(i, "2018-03-01", lab) for i, lab in enumerate([R, C, C, N])]
        rows += [baris(10, "2019-03-01", C), baris(11, "2019-04-01", R, b1=True)]
        oof, lipatan = pb.walk_forward(rows, (2019,), laplace=1, clip=1e-15)
        # B0 = (1/4, 2/4, 1/4); B1 wick < 2/3 sama dengan B0; B1 wick >= 2/3 kosong → seragam.
        self.assertAlmostEqual(oof[0]["ll_b0"], -math.log(0.5))
        self.assertAlmostEqual(oof[0]["ll_b1"], -math.log(0.5))
        self.assertAlmostEqual(oof[1]["ll_b0"], -math.log(0.25))
        self.assertAlmostEqual(oof[1]["ll_b1"], -math.log(1 / 3))
        self.assertAlmostEqual(lipatan[0]["log_loss_b0"], (math.log(2) + math.log(4)) / 2)

    def test_laporan_tahan_tahun_validasi_kosong(self):
        rows = [baris(i, "2018-03-01", lab) for i, lab in enumerate([R, C, N])]
        rows.append(baris(9, "2019-03-01", C))
        oof, lipatan = pb.walk_forward(rows, (2019, 2020), laplace=1, clip=1e-15)
        self.assertIsNone(lipatan[1]["log_loss_b0"])
        recon = {"theta_b1": THETA, "laplace": 1, "catatan": "-", "akhir": pb.model_akhir(rows, 1),
                 "walk_forward": lipatan,
                 "oof": {"n": len(oof), "log_loss_b0": oof[0]["ll_b0"], "log_loss_b1": oof[0]["ll_b1"]}}
        teks = pb.laporan(recon)
        self.assertIn("| 2020 | 0 | – | – | – |", teks)

    def test_walk_forward_hanya_belajar_dari_masa_lalu(self):
        rows, i = [], 0
        for tahun in range(2016, 2024):
            lab = R if tahun < 2019 else N          # sebelum 2019 semua Reversal
            for bulan in (2, 5, 8, 11):
                rows.append(baris(i, f"{tahun}-{bulan:02d}-03", lab, b1=bulan == 5))
                i += 1
        oof, lipatan = pb.walk_forward(rows, (2019, 2020, 2021, 2022, 2023), laplace=1, clip=1e-15)
        self.assertEqual(len(oof), 20)
        self.assertEqual([f["tahun_validasi"] for f in lipatan], [2019, 2020, 2021, 2022, 2023])
        f2019 = lipatan[0]
        self.assertEqual(f2019["n_latih"], 12)
        self.assertEqual(f2019["B0"]["peluang"][R], 1.0)     # tidak ada label 2019 yang bocor
        self.assertEqual(lipatan[1]["n_latih"], 16)
        for r in oof:
            self.assertEqual(int(r["session_date"][:4]), r["tahun_validasi"])
        self.assertTrue(all(f["n_purged"] == 0 for f in lipatan))

    def test_walk_forward_menghitung_purging_per_lipatan(self):
        rows = [baris(0, "2017-03-01", R), baris(1, "2018-12-31", C, akhir="2019-01-02"),
                baris(2, "2019-04-01", N)]
        _, lipatan = pb.walk_forward(rows, (2019,), laplace=1, clip=1e-15)
        self.assertEqual((lipatan[0]["n_latih"], lipatan[0]["n_purged"]), (1, 1))

    def test_lipatan_tanpa_data_latih_ditolak(self):
        rows = [baris(0, "2019-04-01", N)]
        with self.assertRaises(ValueError):
            pb.walk_forward(rows, (2019,), laplace=1, clip=1e-15)


class TestGabungDanValidasi(unittest.TestCase):
    def _fitur(self, ids, b1=("false",) * 3, f13=("0.1",) * 3):
        return [{"event_id": e, "session_date": "2017-01-02", "f13_wick_penolakan": f,
                 "b1_terpenuhi": b} for e, b, f in zip(ids, b1, f13)]

    def _label(self, ids, labels=(R, C, N)):
        return [{"event_id": e, "session_date": "2017-01-02", "label": lab,
                 "label_end_session_date": "2017-01-03"} for e, lab in zip(ids, labels)]

    def test_gabung_menyatukan_fitur_dan_label(self):
        rows = pb.gabung(self._fitur(["a", "b", "c"]), self._label(["a", "b", "c"]), THETA)
        self.assertEqual([r["label"] for r in rows], [R, C, N])
        self.assertFalse(any(r["b1"] for r in rows))

    def test_gabung_menolak_event_id_tidak_sejajar(self):
        with self.assertRaises(ValueError):
            pb.gabung(self._fitur(["a", "b", "c"]), self._label(["a", "c", "b"]), THETA)

    def test_gabung_menolak_jumlah_baris_berbeda(self):
        with self.assertRaises(ValueError):
            pb.gabung(self._fitur(["a", "b", "c"]), self._label(["a", "b"]), THETA)

    def test_gabung_menolak_label_asing(self):
        with self.assertRaises(ValueError):
            pb.gabung(self._fitur(["a", "b", "c"]), self._label(["a", "b", "c"], (R, "X", N)),
                      THETA)

    def test_gabung_menolak_flag_b1_yang_tidak_sesuai_aturan(self):
        fitur = self._fitur(["a", "b", "c"], b1=("false", "false", "false"),
                            f13=("0.1", "0.7", "0.2"))          # 0.7 >= 2/3 tetapi flag false
        with self.assertRaises(ValueError):
            pb.gabung(fitur, self._label(["a", "b", "c"]), THETA)

    def test_gabung_menolak_event_id_ganda(self):
        with self.assertRaises(ValueError):
            pb.gabung(self._fitur(["a", "a", "c"]), self._label(["a", "a", "c"]), THETA)

    def test_gabung_menolak_f13_tidak_berhingga(self):
        fitur = self._fitur(["a", "b", "c"], f13=("0.1", "nan", "0.2"))
        with self.assertRaises(ValueError):
            pb.gabung(fitur, self._label(["a", "b", "c"]), THETA)

    def test_gabung_menolak_akhir_label_sebelum_sesi(self):
        label = self._label(["a", "b", "c"])
        label[1]["label_end_session_date"] = "2016-12-30"
        with self.assertRaises(ValueError):
            pb.gabung(self._fitur(["a", "b", "c"]), label, THETA)

    def test_muat_data_menolak_baris_data_uji(self):
        with tempfile.TemporaryDirectory() as tmp:
            fp, lp = Path(tmp) / "f.csv", Path(tmp) / "l.csv"
            fp.write_text("event_id,session_date,f13_wick_penolakan,b1_terpenuhi\n"
                          "a,2025-01-03,0.1,false\n")
            lp.write_text("event_id,session_date,label,label_end_session_date\n"
                          "a,2025-01-03,Netral,2025-01-03\n")
            with self.assertRaises(common.TestDataLeak):
                pb.muat_data(fp, lp, THETA)


class TestRunPembanding(unittest.TestCase):
    def test_menulis_keluaran(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = jalur.ModelPaths(Path(tmp) / "uji")
            _, events = make_run(paths, n_sessions=400)            # 2016-01 s.d. 2017-07
            sumber = jalur.info_sumber(jalur.cek_data_input(paths, None))
            run_fitur(paths, KUNCI, sumber)
            recon = pb.run_pembanding(paths, KUNCI, sumber, tahun_validasi=(2017,))
            n2017 = sum(e["session_date"].startswith("2017") for e in events)
            self.assertEqual(recon["oof"]["n"], n2017)
            self.assertEqual(recon["akhir"]["n_latih"], len(events))
            self.assertEqual(recon["laplace"], 1)
            self.assertEqual(recon["sumber_pipeline_sha256"], sumber["sumber_sha256"])
            oof = common.load_model_csv(paths.data / "pembanding_oof_dev.csv")
            self.assertEqual(len(oof), n2017)
            saved = json.loads((paths.output / "pembanding_rekonsiliasi.json").read_text())
            self.assertEqual(saved["oof"]["n"], n2017)
            self.assertIn("B1", (paths.output / "laporan_pembanding.md").read_text())
            self.assertIn("B0", pb.ringkas(recon))

    def test_tanpa_fitur_ditolak(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = jalur.ModelPaths(Path(tmp) / "uji")
            make_run(paths, n_sessions=30)
            with self.assertRaises(FileNotFoundError):
                pb.run_pembanding(paths, KUNCI, {})


if __name__ == "__main__":
    unittest.main()
