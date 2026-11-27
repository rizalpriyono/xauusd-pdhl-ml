"""Tes tahap 1 rekayasa fitur (Proposal v2 IV.II.5, Tabel 4.5, dan Kunci-1 Adendum 1 dan 4).

Port dari `Pemodelan PDH-PDL/tests/test_fitur.py`, ditambah tes guard ATR (adendum 2026-11-24) dan
tes fitur 14-15 Adendum 4 (25 November 2026): CMF-21 bertanda d dan permutation entropy 48 return.
"""
import dataclasses
import math
import random
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gb_sintetis import make_bars, make_event, make_run, mirror_bars, mirror_event  # noqa: E402
from model.gradient_boosting import jalur, kunci  # noqa: E402
from model.gradient_boosting.fitur import (CMF_N, FEATURE_COLS, PE_M, PE_N, PE_TUNDA,  # noqa: E402
                                           AtrKosong, LookAheadError, PointInTime, b1_terpenuhi,
                                           build_context, cek_kunci_fitur, cek_tanpa_kosong,
                                           chaikin_money_flow, cmf_d, compute_ema, hari,
                                           hitung_fitur, hitung_semua, money_flow_multiplier,
                                           pe_log_return, permutation_entropy,
                                           pilih_titik_potong, pola_ordinal, prepare_bars,
                                           reconciliation, run_fitur, sesi, uji_pemotongan,
                                           wick_penolakan)
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


def _mfm(b):
    """Money flow multiplier satu bar, ditulis ulang di tes agar tidak bergantung pada kode."""
    return ((b["close"] - b["low"]) - (b["high"] - b["close"])) / (b["high"] - b["low"])


def _ubah_bar(bars, i, **ubah):
    return [{**b, **ubah} if j == i else b for j, b in enumerate(bars)]


class _Rekam:
    """Tampilan bar yang mencatat setiap indeks yang dibaca."""

    def __init__(self, bars):
        self.bars, self.dibaca = bars, []

    def __getitem__(self, i):
        self.dibaca.append(i)
        return self.bars[i]


class TestChaikinMoneyFlow(unittest.TestCase):
    """Fitur 14 (Adendum 4): d x CMF atas 21 bar terakhir sampai bar keputusan."""

    def setUp(self):
        self.bars = make_bars(4)               # 92 bar
        self.t = 60

    def test_mfm_posisi_close_dalam_rentang(self):
        self.assertEqual(money_flow_multiplier(10.0, 0.0, 10.0), 1.0)    # close di high
        self.assertEqual(money_flow_multiplier(10.0, 0.0, 0.0), -1.0)    # close di low
        self.assertEqual(money_flow_multiplier(10.0, 0.0, 5.0), 0.0)     # close di tengah
        self.assertAlmostEqual(money_flow_multiplier(4.0, 2.0, 3.5), 0.5)
        self.assertEqual(money_flow_multiplier(7.0, 7.0, 7.0), 0.0)      # H = L
        with self.assertRaises(ValueError):
            money_flow_multiplier(10.0, 0.0, 11.0)                       # close di luar rentang

    def test_cmf_hitung_tangan(self):
        self.assertAlmostEqual(
            chaikin_money_flow([(10.0, 0.0, 10.0, 100.0), (10.0, 0.0, 0.0, 300.0)]), -0.5)
        self.assertAlmostEqual(
            chaikin_money_flow([(4.0, 2.0, 3.5, 50.0), (7.0, 7.0, 7.0, 150.0)]), 0.125)

    def test_cmf_volume_nol_negatif_dan_kosong(self):
        self.assertEqual(chaikin_money_flow([(10.0, 0.0, 10.0, 0.0), (10.0, 0.0, 0.0, 0.0)]), 0.0)
        with self.assertRaises(ValueError):
            chaikin_money_flow([(10.0, 0.0, 10.0, -1.0)])
        with self.assertRaises(ValueError):
            chaikin_money_flow([])

    def test_jendela_21_bar_termasuk_bar_keputusan(self):
        b, t = self.bars, self.t
        jendela = b[t - 20: t + 1]
        expect = (sum(_mfm(x) * x["tick_volume"] for x in jendela)
                  / sum(x["tick_volume"] for x in jendela))
        view = PointInTime(b, limit=t)
        self.assertEqual((CMF_N, len(jendela)), (21, 21))
        self.assertAlmostEqual(cmf_d(view, t, +1), expect)
        self.assertAlmostEqual(cmf_d(view, t, -1), -expect)             # bertanda d
        self.assertTrue(-1.0 <= cmf_d(view, t, +1) <= 1.0)

    def test_bar_di_luar_jendela_tidak_berpengaruh(self):
        t = self.t
        asal = cmf_d(PointInTime(self.bars, limit=t), t, +1)

        def ubah(i):
            return _ubah_bar(self.bars, i, close=self.bars[i]["high"], tick_volume=1e6)
        self.assertEqual(cmf_d(PointInTime(ubah(t - 21), limit=t), t, +1), asal)
        self.assertNotEqual(cmf_d(PointInTime(ubah(t - 20), limit=t), t, +1), asal)

    def test_hanya_membaca_21_bar_terakhir(self):
        view = _Rekam(self.bars)
        cmf_d(view, self.t, +1)
        self.assertEqual(sorted(view.dibaca), list(range(self.t - 20, self.t + 1)))

    def test_riwayat_kurang_dari_21_bar_kosong(self):
        self.assertIsNone(cmf_d(PointInTime(self.bars, limit=19), 19, +1))
        self.assertIsNotNone(cmf_d(PointInTime(self.bars, limit=20), 20, +1))

    def test_bar_sesudah_batas_tidak_bisa_dibaca(self):
        with self.assertRaises(LookAheadError):
            cmf_d(PointInTime(self.bars, limit=self.t - 1), self.t, +1)


class TestPermutationEntropy(unittest.TestCase):
    """Fitur 15 (Adendum 4): permutation entropy ternormalisasi, orde 3, atas 48 log return."""

    def setUp(self):
        self.bars = make_bars(4)
        self.t = 70

    def test_pola_ordinal_dan_nilai_seri(self):
        self.assertEqual(pola_ordinal((1.0, 2.0, 3.0)), (0, 1, 2))
        self.assertEqual(pola_ordinal((3.0, 1.0, 2.0)), (1, 2, 0))
        self.assertEqual(pola_ordinal((1.0, 1.0, 2.0)), (0, 1, 2))     # seri: urutan kemunculan
        self.assertEqual(pola_ordinal((2.0, 1.0, 1.0)), (1, 2, 0))
        self.assertEqual(pola_ordinal((5.0, 5.0, 5.0)), (0, 1, 2))

    def test_deret_teratur_bernilai_nol(self):
        for xs in ([float(i) for i in range(48)], [float(-i) for i in range(48)], [7.0] * 48):
            pe = permutation_entropy(xs)
            self.assertEqual(pe, 0.0)
            self.assertEqual(math.copysign(1.0, pe), 1.0)              # bukan -0.0

    def test_hitung_tangan(self):
        # jendela (1,2,3) -> (0,1,2); (2,3,2) -> (0,2,1); (3,2,1) -> (2,1,0): tiga pola, p = 1/3
        self.assertAlmostEqual(permutation_entropy([1.0, 2.0, 3.0, 2.0, 1.0]),
                               math.log(3) / math.log(6))
        # jendela (1,3,2) -> (0,2,1); (3,2,4) -> (1,0,2); (2,4,3) -> (0,2,1): p = 2/3 dan 1/3
        h = -(2 / 3) * math.log(2 / 3) - (1 / 3) * math.log(1 / 3)
        self.assertAlmostEqual(permutation_entropy([1.0, 3.0, 2.0, 4.0, 3.0]), h / math.log(6))

    def test_hitung_tangan_dengan_nilai_seri(self):
        # seri menurut kemunculan: (1,1,2) -> (0,1,2); (1,2,2) -> (0,1,2); (2,2,1) -> (2,0,1),
        # jadi p = 2/3 dan 1/3. Aturan seri lain (mis. kemunculan terbalik) memberi tiga pola.
        h = -(2 / 3) * math.log(2 / 3) - (1 / 3) * math.log(1 / 3)
        self.assertAlmostEqual(permutation_entropy([1.0, 1.0, 2.0, 2.0, 1.0]), h / math.log(6))

    def test_enam_pola_merata_bernilai_satu(self):
        xs = [0.0, 1.0, 5.0, 4.0, 3.0, 7.0, 2.0, 6.0]                  # 6 jendela, 6 pola berbeda
        self.assertEqual(len({pola_ordinal(xs[i:i + 3]) for i in range(6)}), 6)
        self.assertAlmostEqual(permutation_entropy(xs), 1.0)

    def test_negasi_tidak_mengubah_nilai(self):
        rnd = random.Random(3)
        xs = [rnd.uniform(-1, 1) for _ in range(48)]
        pe = permutation_entropy(xs)
        self.assertAlmostEqual(pe, permutation_entropy([-x for x in xs]), places=12)
        self.assertTrue(0.0 < pe <= 1.0)

    def test_deret_terlalu_pendek_ditolak(self):
        with self.assertRaises(ValueError):
            permutation_entropy([1.0, 2.0])

    def test_48_log_return_dari_49_close(self):
        b, t = self.bars, self.t
        ret = [math.log(b[i]["close"] / b[i - 1]["close"]) for i in range(t - 47, t + 1)]
        self.assertEqual((PE_N, PE_M, PE_TUNDA, len(ret)), (48, 3, 1, 48))
        self.assertEqual(PE_N - (PE_M - 1) * PE_TUNDA, 46)              # jumlah pola ordinal
        pe = pe_log_return(PointInTime(b, limit=t), t)
        self.assertAlmostEqual(pe, permutation_entropy(ret), places=12)
        self.assertTrue(0.0 <= pe <= 1.0)

    def test_close_di_luar_49_bar_tidak_berpengaruh(self):
        t = self.t
        asal = pe_log_return(PointInTime(self.bars, limit=t), t)
        rusak = _ubah_bar(self.bars, t - 49, close=self.bars[t - 49]["close"] * 3)
        self.assertEqual(pe_log_return(PointInTime(rusak, limit=t), t), asal)

    def test_hanya_membaca_49_close_terakhir(self):
        view = _Rekam(self.bars)
        pe_log_return(view, self.t)
        self.assertEqual(sorted(view.dibaca), list(range(self.t - 48, self.t + 1)))

    def test_riwayat_kurang_dari_49_close_kosong(self):
        self.assertIsNone(pe_log_return(PointInTime(self.bars, limit=47), 47))
        self.assertIsNotNone(pe_log_return(PointInTime(self.bars, limit=48), 48))

    def test_close_tidak_positif_ditolak(self):
        t = self.t
        rusak = _ubah_bar(self.bars, t - 5, close=0.0)
        with self.assertRaises(ValueError):
            pe_log_return(PointInTime(rusak, limit=t), t)

    def test_bar_sesudah_batas_tidak_bisa_dibaca(self):
        with self.assertRaises(LookAheadError):
            pe_log_return(PointInTime(self.bars, limit=self.t - 1), self.t)


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
        jendela = b[t - 20: t + 1]
        self.assertAlmostEqual(f["f14_cmf21_d"],
                               d * sum(_mfm(x) * x["tick_volume"] for x in jendela)
                               / sum(x["tick_volume"] for x in jendela))
        ret = [math.log(b[i]["close"] / b[i - 1]["close"]) for i in range(t - 47, t + 1)]
        self.assertAlmostEqual(f["f15_pe48"], permutation_entropy(ret))
        self.assertEqual(f["b1_terpenuhi"], b1_terpenuhi(f["f13_wick_penolakan"], THETA))
        self.assertEqual(set(FEATURE_COLS) - set(f), set())
        self.assertFalse({"f14_engulfing_d", "f15_volume_relatif"} & set(f))

    def test_keputusan_sesudah_sentuhan_memakai_bar_keputusan(self):
        """k > 0: f14 dan f15 dihitung sampai bar keputusan t*, bukan bar sentuhan t."""
        b, t = self.bars, self.t
        dec = t + 3
        ev = {**self.ev, "decision_bar_idx": str(dec), "decision_ts_ny": b[dec]["ts_ny"],
              "entry_price": repr(b[dec]["close"]), "atr14_h1": repr(b[dec]["atr14_h1"])}
        f = hitung_fitur(ev, b, self.ctx, THETA)
        jendela = b[dec - 20: dec + 1]
        self.assertAlmostEqual(f["f14_cmf21_d"],
                               sum(_mfm(x) * x["tick_volume"] for x in jendela)
                               / sum(x["tick_volume"] for x in jendela))
        ret = [math.log(b[i]["close"] / b[i - 1]["close"]) for i in range(dec - 47, dec + 1)]
        self.assertAlmostEqual(f["f15_pe48"], permutation_entropy(ret))
        di_sentuhan = hitung_fitur(self.ev, b, self.ctx, THETA)
        self.assertNotEqual(f["f14_cmf21_d"], di_sentuhan["f14_cmf21_d"])

    def test_cermin_pdh_pdl_sama(self):
        f_pdh = hitung_fitur(self.ev, self.bars, self.ctx, THETA)
        mb = mirror_bars(self.bars)
        f_pdl = hitung_fitur(mirror_event(self.ev), mb, build_context(mb), THETA)
        self.assertEqual(f_pdl["f01_jenis_level"], 0)
        # f15 tidak bertanda d dan dihitung dari log return, yang tidak antisimetris terhadap cermin
        # aritmetika harga (M - harga). Karena itu f15 hanya dicek rentangnya.
        for col in FEATURE_COLS:
            if col in ("f01_jenis_level", "f15_pe48"):
                continue
            if isinstance(f_pdh[col], str):
                self.assertEqual(f_pdh[col], f_pdl[col], col)
            else:
                self.assertAlmostEqual(f_pdh[col], f_pdl[col], places=9, msg=col)
        for f in (f_pdh, f_pdl):
            self.assertTrue(0.0 <= f["f15_pe48"] <= 1.0)

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
                    "f14_cmf21_d", "f15_pe48"):
            self.assertIsNone(f[col], col)
        f20 = hitung_fitur(make_event(self.bars, 20, "PDH"), self.bars, self.ctx, THETA)
        self.assertIsNotNone(f20["f14_cmf21_d"])          # 21 bar tersedia
        self.assertIsNone(f20["f15_pe48"])
        f48 = hitung_fitur(make_event(self.bars, 48, "PDH"), self.bars, self.ctx, THETA)
        self.assertIsNotNone(f48["f15_pe48"])             # 49 close tersedia

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
        self.assertEqual(sum(rec[k] for k in rec if k.startswith("dev_f04_sesi_")), 3)
        self.assertEqual((rec["dev_null_f14_cmf21_d"], rec["dev_null_f15_pe48"]), (0, 0))
        self.assertAlmostEqual(rec["dev_sum_f15_pe48"], sum(r["f15_pe48"] for r in rows), places=6)
        self.assertFalse([k for k in rec if "engulfing" in k or "volume_relatif" in k])
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


class TestFiturSesuaiKunci(unittest.TestCase):
    """Nama dan parameter fitur 13-15 di kode harus sama dengan Adendum 4 Kunci-1."""

    def test_kunci_repo_cocok(self):
        cek_kunci_fitur(KUNCI)
        self.assertEqual([f["nama"] for f in KUNCI.fitur_tambahan], FEATURE_COLS[12:])

    def test_kunci_menyimpang_ditolak(self):
        f13, f14, f15 = KUNCI.fitur_tambahan
        rusak = {
            "nama": (f13, {**f14, "nama": "f14_engulfing_d"}, f15),
            "jendela_cmf": (f13, {**f14, "jendela_bar": 20}, f15),
            "tanda_cmf": (f13, {**f14, "bertanda_d": False}, f15),
            "jendela_pe": (f13, f14, {**f15, "jendela_return": 24}),
            "orde_pe": (f13, f14, {**f15, "orde_m": 4}),
            "tunda_pe": (f13, f14, {**f15, "tunda": 2}),
            "tanda_pe": (f13, f14, {**f15, "bertanda_d": True}),
            "kurang_satu": (f13, f14),
        }
        for nama, tambahan in rusak.items():
            with self.subTest(nama), self.assertRaises(kunci.KunciError):
                cek_kunci_fitur(dataclasses.replace(KUNCI, fitur_tambahan=tambahan))


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
            self.assertTrue({"f14_cmf21_d", "f15_pe48"} <= set(rows[0]))
            self.assertEqual(recon["kunci1_sha256_sesuai_manifest"],
                             {n: KUNCI.manifest_sha256[n]
                              for n in (kunci.FITUR_JSON, kunci.ADENDUM_FITUR_JSON)})
            self.assertFalse({"f14_engulfing_d", "f15_volume_relatif"} & set(rows[0]))
            self.assertTrue((paths.output / "fitur_rekonsiliasi.json").exists())
            self.assertFalse((paths.input / "features_dev.csv").exists())

    def test_kunci_fitur_menyimpang_tidak_menulis_apa_pun(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = jalur.ModelPaths(Path(tmp) / "uji")
            make_run(paths, n_sessions=40)
            f13, f14, f15 = KUNCI.fitur_tambahan
            k = dataclasses.replace(KUNCI, fitur_tambahan=(f13, {**f14, "jendela_bar": 20}, f15))
            with self.assertRaises(kunci.KunciError):
                run_fitur(paths, k, {})
            self.assertFalse(paths.root.exists())

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
