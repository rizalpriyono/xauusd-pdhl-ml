"""Tes tahap 1 (13 fitur sekuensial v2) & tahap 2 (pembanding B0/B1) jalur LSTM (stdlib-only).

Seluruh tes di berkas ini berjalan murni dengan Python standar tanpa PyTorch maupun NumPy.
"""
import datetime as dt
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gb_sintetis import make_bars, make_event, make_run, mirror_bars, mirror_event  # noqa: E402
from model.lstm import fitur, jalur, kunci, laporan_fitur, pembanding  # noqa: E402
from pdhl_data import common  # noqa: E402
from pdhl_data.config import load_config  # noqa: E402

CFG = load_config(ROOT / "configs" / "tesis.json")
KUNCI = kunci.muat_kunci(CFG, ROOT / "configs" / "kunci_1")


class TestKunciDanJalur(unittest.TestCase):
    def test_muat_kunci_resmi_dan_turunan_lima_seed(self):
        self.assertEqual(KUNCI.seed_model, 20261119)
        self.assertEqual(KUNCI.seeds, (20261119, 20261120, 20261121, 20261122, 20261123))
        self.assertEqual(KUNCI.lookbacks, (23, 40, 46, 60, 96))
        self.assertEqual(KUNCI.hidden_sizes, (16, 32))
        self.assertAlmostEqual(KUNCI.clip, 1e-15)
        self.assertAlmostEqual(KUNCI.toleransi_seri, 1e-6)

    def test_kunci_berubah_ditolak(self):
        with tempfile.TemporaryDirectory() as tmp:
            kdir = Path(tmp) / "kunci_1"
            shutil.copytree(ROOT / "configs" / "kunci_1", kdir)
            p = kdir / "kunci_1.json"
            p.write_text(p.read_text() + " ")
            with self.assertRaises(kunci.KunciError):
                kunci.muat_kunci(CFG, kdir)

    def test_cek_data_input_menolak_hash_berbeda(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = jalur.model_paths(CFG, tmp)
            make_run(paths, n_sessions=25, seed=3)
            ref = {"files": {f"data/{n}": "0" * 64 for n in jalur.INPUT_FILES}}
            with self.assertRaises(jalur.DataInputBerbeda):
                jalur.cek_data_input(paths, ref)


class TestFiturSekuensial(unittest.TestCase):
    def test_point_in_time_menolak_akses_masa_depan(self):
        v = fitur.PointInTime([10, 20, 30], limit=1)
        self.assertEqual(v[0], 10)
        self.assertEqual(v[1], 20)
        with self.assertRaises(fitur.LookAheadError):
            _ = v[2]
        with self.assertRaises(fitur.LookAheadError):
            _ = v[-1]
        with self.assertRaises(TypeError):
            _ = v[0:2]

    def test_simetri_arah_pdh_pdl_identik_untuk_13_fitur(self):
        # 25 sesi = 575 bar -> cukup untuk EMA50, ATR120, dan 20 sesi volume pada t=500
        bars = make_bars(n_sessions=25, seed=5)
        t = 500
        ev_pdh = make_event(bars, t, level_type="PDH")
        ev_pdl = mirror_event(ev_pdh)
        m_bars = mirror_bars(bars)

        seq_pdh, f13_pdh = fitur.hitung_sekuens_event(ev_pdh, bars, lookback=23)
        seq_pdl, f13_pdl = fitur.hitung_sekuens_event(ev_pdl, m_bars, lookback=23)

        self.assertAlmostEqual(f13_pdh, f13_pdl, places=9)
        self.assertEqual(len(seq_pdh), 23)
        self.assertEqual(len(seq_pdh[0]), 13)
        for row_a, row_b in zip(seq_pdh, seq_pdl):
            for a, b in zip(row_a, row_b):
                self.assertAlmostEqual(a, b, places=9)

    def test_tanpa_kebocoran_masa_depan_ketika_bar_sesudah_diubah(self):
        bars = make_bars(n_sessions=25, seed=9)
        t = 500
        ev = make_event(bars, t, level_type="PDH")
        seq_asli, _ = fitur.hitung_sekuens_event(ev, bars, lookback=40)

        bars_rusak = [dict(b) for b in bars]
        for i in range(t + 1, len(bars_rusak)):
            bars_rusak[i]["open"] = 9999.0
            bars_rusak[i]["high"] = 10000.0
            bars_rusak[i]["low"] = 1.0
            bars_rusak[i]["close"] = 5000.0
            bars_rusak[i]["tick_volume"] = 1e9
            bars_rusak[i]["atr14_h1"] = 500.0
        seq_rusak, _ = fitur.hitung_sekuens_event(ev, bars_rusak, lookback=40)
        self.assertEqual(seq_asli, seq_rusak)

    def test_atr_kosong_ditolak(self):
        bars = make_bars(n_sessions=3, seed=2)
        rows = [{**b, "atr14_h1": "" if i == 0 else str(b["atr14_h1"])} for i, b in enumerate(bars)]
        with self.assertRaises(fitur.AtrKosong):
            fitur.prepare_bars(rows)

    def test_guard_data_uji_ditolak(self):
        bars = make_bars(n_sessions=25, seed=4)
        ev = make_event(bars, 500, level_type="PDH", split="test")
        ev["session_date"] = "2025-01-05"
        with self.assertRaises(common.TestDataLeak):
            fitur.hitung_semua_sekuens([ev], bars, lookback=23, theta=KUNCI.theta)

    def test_run_fitur_end_to_end_sintetis(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = jalur.model_paths(CFG, tmp)
            # make_run mulai event di sesi ke-20 (bar >= 460) -> aman untuk lookback=23 + 20 sesi vol
            make_run(paths, n_sessions=30, seed=13)
            sumber = jalur.info_sumber(jalur.cek_data_input(paths, None))
            recon_f = fitur.run_fitur(paths, KUNCI, sumber, lookback_audit=23)
            lap_path = laporan_fitur.tulis_laporan_fitur(paths)

            self.assertTrue(lap_path.exists())
            self.assertEqual(recon_f["n_fitur_sekuensial"], 13)
            self.assertEqual(recon_f["versi_fitur"], "v2_13fitur_theory_driven")
            self.assertTrue(all(r["mismatch"] == 0 for r in recon_f["uji_pemotongan"]))
            self.assertTrue((paths.data / "seq_meta_dev.csv").exists())
            self.assertTrue((paths.output / "fitur_rekonsiliasi.json").exists())


class TestPembandingDanPurging(unittest.TestCase):
    def test_purging_buang_event_melintasi_batas(self):
        rows = [
            {"event_id": "e1", "session_date": "2018-12-28", "label": "Reversal",
             "label_end_session_date": "2018-12-31", "f13": 0.2, "b1": False},
            {"event_id": "e2", "session_date": "2018-12-31", "label": "Continuation",
             "label_end_session_date": "2019-01-02", "f13": 0.8, "b1": True},
            {"event_id": "e3", "session_date": "2019-01-03", "label": "Netral",
             "label_end_session_date": "2019-01-03", "f13": 0.1, "b1": False},
        ]
        latih, purged = pembanding.data_latih(rows, dt.date(2019, 1, 1))
        self.assertEqual([r["event_id"] for r in latih], ["e1"])
        self.assertEqual([r["event_id"] for r in purged], ["e2"])

    def test_laplace_b1_dan_log_loss_identik_kunci1(self):
        rows = [
            {"event_id": "e1", "session_date": "2018-01-02", "label": "Reversal",
             "label_end_session_date": "2018-01-02", "f13": 0.8, "b1": True},
            {"event_id": "e2", "session_date": "2018-01-03", "label": "Continuation",
             "label_end_session_date": "2018-01-03", "f13": 0.1, "b1": False},
        ]
        m = pembanding.latih_pembanding(rows, laplace=1)
        self.assertTrue(m["B1"]["terpenuhi"]["laplace"])
        self.assertAlmostEqual(m["B1"]["terpenuhi"]["peluang"]["Reversal"], 2 / 4)
        self.assertAlmostEqual(m["B1"]["terpenuhi"]["peluang"]["Netral"], 1 / 4)
        ll = pembanding.log_loss([0.5, 0.3, 0.2], "Reversal", clip=1e-15)
        self.assertAlmostEqual(ll, -math.log(0.5), places=12)


if __name__ == "__main__":
    unittest.main()
