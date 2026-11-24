"""Tes tahap 1 rekayasa fitur (Proposal v2 IV.II.5, Tabel 4.5, dan adendum Kunci-1 17 Nov 2026).

Port dari `Pemodelan PDH-PDL/tests/test_fitur.py`, ditambah tes guard ATR (adendum 2026-11-24).
"""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gb_sintetis import make_bars, make_event, make_run, mirror_bars, mirror_event  # noqa: E402
from model.gradient_boosting import jalur, kunci  # noqa: E402
from model.gradient_boosting.fitur import (FEATURE_COLS, AtrKosong, LookAheadError,  # noqa: E402
                                           PointInTime, b1_terpenuhi, build_context,
                                           cek_tanpa_kosong, compute_ema, engulfing_d, hari,
                                           hitung_fitur, hitung_semua, pilih_titik_potong,
                                           prepare_bars, reconciliation, run_fitur, sesi,
                                           uji_pemotongan, volume_relatif, wick_penolakan)
from pdhl_data import common  # noqa: E402
from pdhl_data.config import load_config  # noqa: E402

THETA = 2 / 3
KUNCI = kunci.muat_kunci(load_config(ROOT / "configs" / "tesis.json"), ROOT / "configs" / "kunci_1")


class TestHelpers(unittest.TestCase):
    def test_sesi_batas_jam(self):
        for h, s in ((18, "Asia"), (23, "Asia"), (0, "Asia"), (2, "Asia"), (3, "London"),
                     (7, "London"), (8, "NewYork"), (16, "NewYork")):
            self.assertEqual(sesi(h), s, h)
        with self.assertRaises(ValueError):
            sesi(17)

    def test_hari_dari_session_date(self):
        self.assertEqual(hari("2016-01-04"), "Senin")
        self.assertEqual(hari("2016-01-08"), "Jumat")
        with self.assertRaises(ValueError):
            hari("2016-01-09")

    def test_wick_pdh_dan_pdl(self):
        self.assertAlmostEqual(wick_penolakan(100.0, 101.0, 99.9, 100.2, +1), 0.8 / 1.1)
        self.assertAlmostEqual(wick_penolakan(100.0, 101.0, 99.0, 100.5, -1), 1.0 / 2.0)
        self.assertEqual(wick_penolakan(100.0, 100.0, 100.0, 100.0, +1), 0.0)

    def test_engulfing_bertanda_d(self):
        prev_bear, prev_bull = (100.5, 100.0), (100.0, 100.5)
        bull_engulf, bear_engulf = (99.9, 100.6), (100.6, 99.9)
        self.assertEqual(engulfing_d(*prev_bear, *bull_engulf, +1), +1)   # bullish di PDH
        self.assertEqual(engulfing_d(*prev_bull, *bear_engulf, +1), -1)   # bearish di PDH
        self.assertEqual(engulfing_d(*prev_bull, *bear_engulf, -1), +1)   # bearish di PDL
        self.assertEqual(engulfing_d(*prev_bear, *bull_engulf, -1), -1)   # bullish di PDL

    def test_engulfing_tidak_terpenuhi(self):
        self.assertEqual(engulfing_d(100.0, 100.5, 99.9, 100.6, +1), 0)   # warna sama
        self.assertEqual(engulfing_d(100.5, 100.0, 100.1, 100.4, +1), 0)  # badan tidak menutupi
        self.assertEqual(engulfing_d(100.2, 100.2, 99.9, 100.6, +1), 0)   # sebelumnya doji
        self.assertEqual(engulfing_d(100.5, 100.0, 100.0, 100.5, +1), +1)  # batas sama dihitung

    def test_ema_seed_dan_rekursi(self):
        ema = compute_ema([1.0, 2.0, 3.0, 4.0, 5.0], n=3)
        self.assertEqual(ema[:2], [None, None])
        self.assertAlmostEqual(ema[2], 2.0)
        self.assertAlmostEqual(ema[3], 0.5 * 4.0 + 0.5 * 2.0)
        self.assertAlmostEqual(ema[4], 0.5 * 5.0 + 0.5 * 3.0)
        for v in compute_ema([7.0] * 60, n=50)[49:]:
            self.assertAlmostEqual(v, 7.0, places=12)

    def test_b1_batas_dua_pertiga_terpenuhi(self):
        theta_json = 0.6666666667                      # nilai di kunci_1_fitur.json
        self.assertTrue(b1_terpenuhi(2 / 3, theta_json))
        self.assertTrue(b1_terpenuhi(wick_penolakan(100.0, 103.0, 100.0, 101.0, +1), theta_json))
        self.assertFalse(b1_terpenuhi(0.66, theta_json))


class TestPointInTime(unittest.TestCase):
    def test_indeks_masa_depan_ditolak(self):
        view = PointInTime([10, 11, 12, 13], limit=2)
        self.assertEqual(view[2], 12)
        with self.assertRaises(LookAheadError):
            view[3]
        with self.assertRaises(LookAheadError):
            view[-1]
        with self.assertRaises(TypeError):
            view[0:2]


class TestVolumeRelatif(unittest.TestCase):
    def setUp(self):
        self.bars = make_bars(25)
        self.ctx = build_context(self.bars)

    def test_rasio_jam_sama_20_sesi(self):
        t = 23 * 22 + 10                       # sesi ke-23, jam HOURS[10]
        view = PointInTime(self.bars, limit=t)
        h = self.bars[t]["ny_hour"]
        hist = [b["tick_volume"] for b in self.bars[23 * 2: 23 * 22] if b["ny_hour"] == h]
        self.assertEqual(len(hist), 20)
        expect = self.bars[t]["tick_volume"] / (sum(hist) / 20)
        self.assertAlmostEqual(volume_relatif(view, self.ctx, t), expect)

    def test_sesi_tanpa_jam_itu_dilewati(self):
        t = 23 * 22 + 10
        h = self.bars[t]["ny_hour"]
        bars = [b for b in self.bars if not (b["ny_hour"] == h and b["session_date"] in
                                              {self.bars[23 * k]["session_date"] for k in (5, 6)})]
        bars = [{**b, "bar_idx": i} for i, b in enumerate(bars)]
        t2 = t - 2
        self.assertEqual(bars[t2]["ny_hour"], h)
        window = set(sorted({b["session_date"] for b in self.bars})[2:22])   # 20 sesi sebelumnya
        hist = [b["tick_volume"] for b in bars[:t2]
                if b["ny_hour"] == h and b["session_date"] in window]
        self.assertEqual(len(hist), 18)       # 2 dari 20 sesi tidak punya bar jam h
        val = volume_relatif(PointInTime(bars, limit=t2), build_context(bars), t2)
        self.assertAlmostEqual(val, bars[t2]["tick_volume"] / (sum(hist) / len(hist)))

    def test_riwayat_kurang_dari_10_kosong(self):
        t = 23 * 9 + 5
        self.assertIsNone(volume_relatif(PointInTime(self.bars, limit=t), self.ctx, t))
        t = 23 * 10 + 5
        self.assertIsNotNone(volume_relatif(PointInTime(self.bars, limit=t), self.ctx, t))


class TestHitungFitur(unittest.TestCase):
    def setUp(self):
        self.bars = make_bars(30)
        self.ctx = build_context(self.bars)
        self.t = 23 * 27 + 12                  # jam 6 (London), sesi ke-28
        self.ev = make_event(self.bars, self.t, "PDH")

    def test_nilai_sesuai_rumus(self):
        b, t, e = self.bars, self.t, self.ev
        f = hitung_fitur(e, b, self.ctx, THETA)
        atr, lvl, d = b[t]["atr14_h1"], float(e["level_price"]), 1
        self.assertEqual(f["f01_jenis_level"], 1)
        self.assertAlmostEqual(f["f02_rentang_hari_sebelumnya"], 3.0 / 5.0)
        self.assertAlmostEqual(f["f03_jarak_level_open_sesi"],
                               d * (lvl - float(e["session_open"])) / atr)
        self.assertEqual(f["f04_sesi"], sesi(b[t]["ny_hour"]))
        self.assertEqual(f["f05_hari"], hari(b[t]["session_date"]))
        mean_atr = sum(x["atr14_h1"] for x in b[t - 119: t + 1]) / 120
        self.assertAlmostEqual(f["f06_rezim_volatilitas"], atr / mean_atr)
        self.assertAlmostEqual(f["f07_momentum_pendekatan"], (b[t]["close"] - b[t - 6]["close"]) / atr)
        self.assertEqual(f["f08_waktu_sejak_open_sesi"], b[t]["bar_in_session"])
        self.assertAlmostEqual(f["f09_posisi_close"], (b[t]["close"] - lvl) / atr)
        self.assertAlmostEqual(f["f10_penetrasi"], (b[t]["high"] - lvl) / atr)
        ema = compute_ema([x["close"] for x in b], 50)[t]
        self.assertAlmostEqual(f["f11_jarak_ema50"], (b[t]["close"] - ema) / atr)
        week_open = next(x["open"] for x in b if x["week_start"] == b[t]["week_start"])
        self.assertAlmostEqual(f["f12_return_minggu"], (b[t]["close"] - week_open) / 5.0)
        self.assertAlmostEqual(f["f13_wick_penolakan"],
                               wick_penolakan(b[t]["open"], b[t]["high"], b[t]["low"],
                                              b[t]["close"], 1))
        self.assertEqual(f["f14_engulfing_d"],
                         engulfing_d(b[t - 1]["open"], b[t - 1]["close"], b[t]["open"],
                                     b[t]["close"], 1))
        self.assertAlmostEqual(f["f15_volume_relatif"], volume_relatif(
            PointInTime(b, limit=t), self.ctx, t))
        self.assertEqual(f["b1_terpenuhi"], b1_terpenuhi(f["f13_wick_penolakan"], THETA))
        self.assertEqual(set(FEATURE_COLS) - set(f), set())

    def test_cermin_pdh_pdl_sama(self):
        f_pdh = hitung_fitur(self.ev, self.bars, self.ctx, THETA)
        mb = mirror_bars(self.bars)
        f_pdl = hitung_fitur(mirror_event(self.ev), mb, build_context(mb), THETA)
        self.assertEqual(f_pdl["f01_jenis_level"], 0)
        for col in FEATURE_COLS:
            if col == "f01_jenis_level":
                continue
            if isinstance(f_pdh[col], str):
                self.assertEqual(f_pdh[col], f_pdl[col], col)
            else:
                self.assertAlmostEqual(f_pdh[col], f_pdl[col], places=9, msg=col)

    def test_bar_masa_depan_tidak_berpengaruh(self):
        before = hitung_fitur(self.ev, self.bars, self.ctx, THETA)
        rusak = [b if i <= self.t else {**b, "open": 1e6, "high": 2e6, "low": 0.0, "close": 1e6,
                                        "tick_volume": 1e9, "atr14_h1": 99.0}
                 for i, b in enumerate(self.bars)]
        self.assertEqual(hitung_fitur(self.ev, rusak, build_context(rusak), THETA), before)

    def test_event_tidak_cocok_dengan_bar_ditolak(self):
        bad = {**self.ev, "touch_close": repr(self.bars[self.t]["close"] + 0.5)}
        with self.assertRaises(ValueError):
            hitung_fitur(bad, self.bars, self.ctx, THETA)
        bad = {**self.ev, "direction_d": "-1"}
        with self.assertRaises(ValueError):
            hitung_fitur(bad, self.bars, self.ctx, THETA)

    def test_hasil_dict_baru_event_tidak_berubah(self):
        snap = dict(self.ev)
        hitung_fitur(self.ev, self.bars, self.ctx, THETA)
        self.assertEqual(self.ev, snap)


class TestHitungSemuaDanPemotongan(unittest.TestCase):
    def setUp(self):
        self.bars = make_bars(30)
        self.events = [make_event(self.bars, 23 * s + h, lt)
                       for s, h, lt in ((20, 3, "PDH"), (22, 15, "PDL"), (25, 8, "PDH"),
                                        (28, 20, "PDL"))]

    def test_hitung_semua(self):
        rows = hitung_semua(self.events, self.bars, THETA)
        self.assertEqual([r["event_id"] for r in rows], [e["event_id"] for e in self.events])

    def test_data_uji_ditolak(self):
        ev = [{**self.events[0], "session_date": "2025-01-06"}]
        with self.assertRaises(common.TestDataLeak):
            hitung_semua(ev, self.bars, THETA)

    def test_bar_idx_harus_kontinu(self):
        bars = self.bars[:5] + self.bars[6:]
        with self.assertRaises(ValueError):
            hitung_semua(self.events, bars, THETA)

    def test_uji_pemotongan_identik(self):
        cuts = [int(e["decision_bar_idx"]) for e in self.events] + [len(self.bars) - 1]
        res = uji_pemotongan(self.events, self.bars, cuts, THETA)
        self.assertEqual(sum(r["mismatch"] for r in res), 0)
        self.assertEqual([r["n_event"] for r in res], [1, 2, 3, 4, 4])


class TestValidasiDanTepi(unittest.TestCase):
    def setUp(self):
        self.bars = make_bars(30)
        self.ctx = build_context(self.bars)
        self.t = 23 * 27 + 12
        self.ev = make_event(self.bars, self.t, "PDH")

    def test_event_tidak_konsisten_ditolak(self):
        b = self.bars[self.t]
        rusak = {
            "entry": {"entry_price": repr(b["close"] + 1)},
            "atr": {"atr14_h1": repr(b["atr14_h1"] + 1)},
            "level_bukan_pdh": {"level_price": repr(b["high"] - 0.2)},
            "tidak_menyentuh": {"level_price": repr(b["high"] + 1), "pdh": repr(b["high"] + 1)},
            "urutan_bar": {"touch_bar_in_session": str(b["bar_in_session"] + 1)},
            "open_sesi": {"session_open": repr(float(self.ev["session_open"]) + 1)},
            "touch_sesudah_keputusan": {"touch_bar_idx": str(self.t + 1)},
            "tidak_finite": {"atr14_d_prev": "nan"},
            "atr_harian_nol": {"atr14_d_prev": "0"},
        }
        for name, patch in rusak.items():
            with self.subTest(name), self.assertRaises(ValueError):
                hitung_fitur({**self.ev, **patch}, self.bars, self.ctx, THETA)

    def test_riwayat_pendek_menjadi_kosong(self):
        f = hitung_fitur(make_event(self.bars, 3, "PDL"), self.bars, self.ctx, THETA)
        for col in ("f06_rezim_volatilitas", "f07_momentum_pendekatan", "f11_jarak_ema50",
                    "f15_volume_relatif"):
            self.assertIsNone(f[col], col)
        self.assertIsNotNone(f["f14_engulfing_d"])
        f0 = hitung_fitur(make_event(self.bars, 0, "PDH"), self.bars, self.ctx, THETA)
        self.assertIsNone(f0["f14_engulfing_d"])

    def test_split_harus_sesuai_tanggal(self):
        with self.assertRaises(ValueError):
            hitung_semua([{**self.ev, "split": "calib"}], self.bars, THETA)
        with self.assertRaises(common.TestDataLeak):
            hitung_semua([{**self.ev, "split": "test"}], self.bars, THETA)

    def test_konteks_menolak_jam_ganda_dan_urutan_mundur(self):
        dup = self.bars[:3] + [{**self.bars[2], "bar_idx": 3}]
        with self.assertRaises(ValueError):
            build_context(dup)
        mundur = self.bars[23:25] + [{**self.bars[0], "bar_idx": 2}]
        with self.assertRaises(ValueError):
            build_context(mundur)

    def test_input_tidak_valid(self):
        with self.assertRaises(ValueError):
            sesi(24)
        with self.assertRaises(ValueError):
            wick_penolakan(100.0, 99.0, 98.0, 100.5, +1)
        self.assertEqual(PointInTime([1, 2], limit=1).limit, 1)

    def test_rekonsiliasi_dan_titik_potong(self):
        events = [make_event(self.bars, 23 * s + 4, lt)
                  for s, lt in ((20, "PDH"), (24, "PDL"), (27, "PDH"))]
        rows = hitung_semua(events, self.bars, THETA)
        rec = reconciliation(rows)
        self.assertEqual((rec["dev_n"], rec["calib_n"]), (3, 0))
        self.assertEqual(sum(rec[f"dev_f14_engulfing_d_{v}"] for v in (-1, 0, 1)), 3)
        self.assertEqual(sum(rec[k] for k in rec if k.startswith("dev_f04_sesi_")), 3)
        self.assertEqual(rec["dev_null_f15_volume_relatif"], 0)
        self.assertEqual(rec["dev_b1_terpenuhi"], sum(r["b1_terpenuhi"] for r in rows))
        cuts = pilih_titik_potong(events, self.bars, dates=(events[1]["session_date"],))
        self.assertEqual(cuts, [int(events[1]["decision_bar_idx"]), len(self.bars) - 1])


def raw_bar(**ubah):
    raw = {"bar_idx": "0", "ts_ny": "2016-01-03 18:00:00", "ny_hour": "18",
           "session_date": "2016-01-04", "week_start": "2016-01-04", "bar_in_session": "1",
           "open": "1.5", "high": "2", "low": "1", "close": "1.8", "tick_volume": "300",
           "atr14_h1": "1.2"}
    raw.update(ubah)
    return raw


class TestPrepareDanGuardAtr(unittest.TestCase):
    def test_prepare_bars_menolak_nan(self):
        with self.assertRaises(ValueError):
            prepare_bars([raw_bar(open="nan")])

    def test_prepare_bars_mengubah_tipe(self):
        raw = [raw_bar()]
        b = prepare_bars(raw)[0]
        self.assertEqual((b["ny_hour"], b["bar_in_session"], b["tick_volume"]), (18, 1, 300.0))
        self.assertEqual(b["atr14_h1"], 1.2)
        self.assertEqual(raw[0]["open"], "1.5")

    def test_atr_bar_kosong_ditolak(self):
        """Data pipeline < 1.1.0: atr14_h1 13 bar pertama 2015 kosong -> harus ditolak."""
        with self.assertRaises(AtrKosong) as ctx:
            prepare_bars([raw_bar(atr14_h1="")])
        self.assertIn("1.1.0", str(ctx.exception))

    def test_atr_bar_nol_atau_tidak_finite_ditolak(self):
        for v in ("0", "-1", "nan", "inf"):
            with self.subTest(v), self.assertRaises(ValueError):
                prepare_bars([raw_bar(atr14_h1=v)])

    def test_atr_harian_event_kosong_ditolak(self):
        bars = make_bars(30)
        ev = {**make_event(bars, 23 * 27 + 12, "PDH"), "atr14_d_prev": ""}
        with self.assertRaises(AtrKosong):
            hitung_fitur(ev, bars, build_context(bars), THETA)

    def test_fitur_kosong_ditolak(self):
        bars = make_bars(30)
        ctx = build_context(bars)
        lengkap = hitung_fitur(make_event(bars, 23 * 27 + 12, "PDH"), bars, ctx, THETA)
        cek_tanpa_kosong([lengkap])
        pendek = hitung_fitur(make_event(bars, 3, "PDL"), bars, ctx, THETA)
        with self.assertRaises(ValueError):
            cek_tanpa_kosong([lengkap, pendek])

    def test_jumlah_fitur_sama_dengan_kunci(self):
        k = kunci.load_fitur(ROOT / "configs" / "kunci_1" / "kunci_1_fitur.json")
        self.assertEqual(k["jumlah_fitur_akhir"], len(FEATURE_COLS))
        self.assertAlmostEqual(KUNCI.theta, THETA, places=9)


class TestRunFitur(unittest.TestCase):
    def test_menulis_keluaran_dan_rekonsiliasi(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = jalur.ModelPaths(Path(tmp) / "uji")
            _, events = make_run(paths, n_sessions=40)
            sha = jalur.cek_data_input(paths, None)
            recon = run_fitur(paths, KUNCI, jalur.info_sumber(sha))
            self.assertEqual(recon["dev_n"], len(events))
            self.assertEqual(recon["calib_n"], 0)
            self.assertEqual(sum(r["mismatch"] for r in recon["uji_pemotongan"]), 0)
            self.assertEqual(recon["sumber_sha256"], sha)
            rows = common.load_model_csv(paths.data / "features_dev.csv")
            self.assertEqual([r["event_id"] for r in rows], [e["event_id"] for e in events])
            self.assertTrue((paths.output / "fitur_rekonsiliasi.json").exists())
            self.assertFalse((paths.input / "features_dev.csv").exists())

    def test_atr_kosong_tidak_menulis_apa_pun(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = jalur.ModelPaths(Path(tmp) / "uji")
            bars, _ = make_run(paths, n_sessions=40)
            rows = common.read_csv(paths.input / "bars_devcalib.csv")
            rows[0]["atr14_h1"] = ""
            common.write_csv(paths.input / "bars_devcalib.csv", rows, list(rows[0]))
            with self.assertRaises(AtrKosong):
                run_fitur(paths, KUNCI, {})
            self.assertFalse(paths.root.exists())


if __name__ == "__main__":
    unittest.main()
