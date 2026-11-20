"""Tes tahap 1 (bronze -> silver -> gold) pada data sintetis: sesi, pembersihan, event, data_gap, DST."""
import datetime as dt
import sys
import tempfile
import unittest
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pdhl_data import events as ev  # noqa: E402

NY, UTC = ZoneInfo("America/New_York"), ZoneInfo("UTC")


def session_times(session_date, n=23, start_hour=18):
    """Waktu buka bar NY sebuah sesi normal: (tanggal sesi - 1) 18.00 s.d. tanggal sesi 16.00."""
    start = dt.datetime.combine(session_date - dt.timedelta(days=1), dt.time(start_hour), NY)
    return [start + dt.timedelta(hours=i) for i in range(n)]


def raw_row(ts_ny, o, h, low, c, minutes=60, volume=100):
    return {"ts_utc": ts_ny.astimezone(UTC), "open": o, "high": h, "low": low, "close": c,
            "bar_minutes": minutes, "tick_volume": volume}


def session_rows(session_date, high, low, price, overrides=None, n=23, start_hour=18):
    """Bar datar (open = close = price) dengan high/low tetap; overrides: {indeks: (h, l)}."""
    rows = []
    for i, ts in enumerate(session_times(session_date, n, start_hour)):
        h, lo = (overrides or {}).get(i, (high, low))
        rows.append(raw_row(ts, price, h, lo, price))
    return rows


def week_2016_03():
    """Minggu 7-11 Mar 2016 (sebelum DST). Skenario event (lihat docstring tiap tes)."""
    d = dt.date
    return [
        *session_rows(d(2016, 3, 7), 100.5, 99.5, 100.0),                          # Sen: PDH/PDL hari Selasa
        *session_rows(d(2016, 3, 8), 100.4, 99.6, 100.0, {4: (100.7, 99.6)}),       # Sel: PDH tersentuh bar ke-5
        *session_rows(d(2016, 3, 9), 101.2, 100.8, 101.0),                          # Rab: dibuka di atas PDH
        *session_rows(d(2016, 3, 10), 101.1, 100.9, 101.0, {22: (101.1, 100.7)}),   # Kam: PDL di bar terakhir
        *session_rows(d(2016, 3, 11), 101.05, 100.95, 101.0, {20: (101.15, 100.95)}),  # Jum: PDH dekat tutup
    ]


def pipeline(rows, k=0, horizon=8):
    bars, dq, _ = ev.build_silver_bars(rows)
    sessions = ev.build_sessions(bars)
    return bars, dq, sessions, {e["event_id"]: e for e in ev.detect_events(bars, sessions, k, horizon)}


class TestPrimitives(unittest.TestCase):
    def test_wilder(self):
        out = ev.wilder([1.0, 2.0, 3.0, 4.0], n=3)
        self.assertEqual(out[:2], [None, None])
        self.assertAlmostEqual(out[2], 2.0)
        self.assertAlmostEqual(out[3], (2.0 * 2 + 4.0) / 3)
        self.assertEqual(ev.wilder([1.0, 2.0], n=3), [None, None])

    def test_true_range(self):
        self.assertEqual(ev.true_range(10.0, 8.0, None), 2.0)
        self.assertEqual(ev.true_range(10.0, 8.0, 12.0), 4.0)     # gap turun
        self.assertEqual(ev.true_range(10.0, 8.0, 5.0), 5.0)      # gap naik

    def test_split_of_date(self):
        self.assertIsNone(ev.split_of_date(dt.date(2015, 12, 31)))
        self.assertEqual(ev.split_of_date(dt.date(2023, 12, 31)), "dev")
        self.assertEqual(ev.split_of_date(dt.date(2024, 1, 2)), "calib")
        self.assertEqual(ev.split_of_date(dt.date(2026, 7, 31)), "test")
        self.assertIsNone(ev.split_of_date(dt.date(2026, 8, 3)))


class TestBronze(unittest.TestCase):
    def _write(self, tmp, lines):
        p = Path(tmp) / "raw.csv"
        p.write_text("Time\tOpen\tHigh\tLow\tClose\tVolume\n" + "\n".join(lines))
        return p

    def test_reads_seven_columns_utc(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = ev.read_bronze(self._write(tmp, ["2016-03-07 14:00:00\t1\t2\t0.5\t1.5\t60\t321"]))
        self.assertEqual(rows[0]["ts_utc"], dt.datetime(2016, 3, 7, 14, tzinfo=UTC))
        self.assertEqual((rows[0]["bar_minutes"], rows[0]["tick_volume"]), (60, 321))

    def test_rejects_wrong_column_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "7"):
                ev.read_bronze(self._write(tmp, ["2016-03-07 14:00:00\t1\t2\t0.5\t1.5\t60"]))

    def test_rejects_empty_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                ev.read_bronze(self._write(tmp, []))


class TestSilver(unittest.TestCase):
    def test_sunday_evening_bar_belongs_to_monday_session(self):
        bars, _, sessions, _ = pipeline(week_2016_03())
        self.assertEqual(bars[0]["ts_ny"].strftime("%a %H:%M"), "Sun 18:00")
        self.assertEqual(bars[0]["session_date"], dt.date(2016, 3, 7))
        self.assertEqual(bars[0]["week_start"], dt.date(2016, 3, 7))
        self.assertEqual([s["n_bars"] for s in sessions], [23] * 5)

    def test_cleaning_drops_duplicate_invalid_and_break_bars(self):
        rows = week_2016_03()
        ts_break = dt.datetime(2016, 3, 7, 17, tzinfo=NY)               # jeda harian 17.00 NY
        rows += [dict(rows[5]),                                          # duplikat timestamp
                 raw_row(ts_break, 100.0, 100.1, 99.9, 100.0, minutes=1)]
        rows[7] = {**rows[7], "low": 100.2}                               # low > open: tidak valid
        bars, dq, _, _ = pipeline(rows)
        self.assertEqual((dq["duplikat"], dq["ohlc_tidak_valid"], dq["bar_jeda_17_ny"]), (1, 1, 1))
        self.assertEqual(len(bars), 5 * 23 - 1)

    def test_missing_bars_are_flagged_not_imputed(self):
        rows = week_2016_03()
        del rows[23 + 2]                                   # bar ke-3 sesi Selasa hilang
        rows = rows[1:]                                    # sesi Senin dibuka 19.00, bukan 18.00
        bars, _, sessions, _ = pipeline(rows)
        flagged = [b["bar_idx"] for b in bars if b["missing_before"]]
        self.assertEqual(flagged, [0, 22 + 2])
        self.assertEqual([s["has_data_gap"] for s in sessions], [True, True, False, False, False])

    def test_atr_starts_after_14_bars(self):
        bars, _, _, _ = pipeline(week_2016_03())
        self.assertIsNone(bars[12]["atr14_h1"])
        self.assertAlmostEqual(bars[13]["atr14_h1"], 1.0)          # rentang 100.5 - 99.5


class TestEvents(unittest.TestCase):
    def test_levels_status_and_first_touch(self):
        _, _, sessions, events = pipeline(week_2016_03())
        mon, tue, wed = sessions[0], sessions[1], sessions[2]
        self.assertEqual((mon["pdh_status"], mon["pdl_status"]), ("no_prev_session", "no_prev_session"))
        self.assertEqual((tue["pdh"], tue["pdl"]), (100.5, 99.5))
        self.assertEqual((tue["pdh_status"], tue["pdl_status"]), ("event", "not_touched"))
        self.assertEqual(wed["pdh_status"], "opened_beyond")
        self.assertEqual(set(events), {"2016-03-08_PDH", "2016-03-10_PDL", "2016-03-11_PDH"})
        e = events["2016-03-08_PDH"]
        self.assertEqual((e["touch_bar_in_session"], e["direction_d"], e["level_price"]), (5, 1, 100.5))
        self.assertEqual(events["2016-03-10_PDL"]["direction_d"], -1)

    def test_validity_and_week_close_exclusion(self):
        _, _, _, events = pipeline(week_2016_03())
        self.assertTrue(events["2016-03-08_PDH"]["is_valid"])
        thu = events["2016-03-10_PDL"]
        self.assertTrue(thu["is_valid"])
        self.assertEqual(thu["window_max_gap_h"], 2.0)          # jeda harian 16.00 -> 18.00
        fri = events["2016-03-11_PDH"]
        self.assertEqual((fri["is_valid"], fri["exclusion_reason"]),
                         (False, "window_crosses_week_close"))
        self.assertEqual(fri["bars_to_week_close"], 2)

    def test_decision_bar_k(self):
        bars, _, _, events = pipeline(week_2016_03(), k=1)
        e = events["2016-03-08_PDH"]
        self.assertEqual(e["decision_bar_idx"], e["touch_bar_idx"] + 1)
        self.assertEqual(e["entry_price"], bars[e["decision_bar_idx"]]["close"])
        self.assertTrue(events["2016-03-10_PDL"]["decision_in_next_session"])

    def test_data_gap_before_decision(self):
        rows = week_2016_03()
        del rows[23 + 2]                                   # Selasa: bar hilang sebelum sentuhan
        _, _, _, events = pipeline(rows)
        e = events["2016-03-08_PDH"]
        self.assertEqual((e["is_valid"], e["gap_before_decision"]), (False, True))
        self.assertEqual(e["exclusion_reason"], "data_gap")

    def test_data_gap_in_label_window(self):
        rows = week_2016_03()
        del rows[4 * 23 + 3]                               # Jumat bar ke-4: di jendela label Kamis
        _, _, _, events = pipeline(rows)
        e = events["2016-03-10_PDL"]
        self.assertEqual((e["is_valid"], e["gap_in_label_window"]), (False, True))

    def test_data_gap_in_previous_session(self):
        rows = week_2016_03()
        del rows[3 * 23 + 5]                               # Kamis: sumber PDH/PDL hari Jumat
        _, _, _, events = pipeline(rows)
        self.assertTrue(events["2016-03-11_PDH"]["gap_in_prev_session"])
        self.assertIn("data_gap", events["2016-03-11_PDH"]["exclusion_reason"])

    def test_end_of_data_excludes_event(self):
        # Data berhenti 2 bar setelah sentuhan Kamis. Bar terakhir minggu = bar terakhir data, jadi
        # alasan yang tercatat adalah window_crosses_week_close (dicek lebih dulu).
        rows = week_2016_03()[:-21]
        _, _, _, events = pipeline(rows)
        e = events["2016-03-10_PDL"]
        self.assertEqual((e["is_valid"], e["exclusion_reason"]), (False, "window_crosses_week_close"))
        self.assertIsNone(e["window_end_session_date"])


class TestDst(unittest.TestCase):
    def _two_weeks(self, extra=()):
        d = dt.date
        rows = [*session_rows(d(2016, 3, 11), 100.5, 99.5, 100.0),      # EST (UTC-5)
                *session_rows(d(2016, 3, 14), 100.5, 99.5, 100.0)]      # EDT (UTC-4)
        return rows + list(extra)

    def test_dst_change_passes_with_clean_break(self):
        rows = self._two_weeks()
        bars, _, _ = ev.build_silver_bars(rows)
        out = ev.dst_verification(rows, ev.build_sessions(bars))
        self.assertEqual([(r["role"], r["utc_offset"], r["status"]) for r in out],
                         [("sebelum", -5, "PASS"), ("sesudah", -4, "PASS")])

    def test_full_bar_at_17_ny_fails(self):
        bad = raw_row(dt.datetime(2016, 3, 14, 17, tzinfo=NY), 100.0, 100.1, 99.9, 100.0, minutes=60)
        rows = self._two_weeks([bad])
        bars, _, _ = ev.build_silver_bars(rows)
        out = ev.dst_verification(rows, ev.build_sessions(bars))
        self.assertEqual(out[1]["status"], "GAGAL")
        self.assertEqual(out[1]["raw_bars_17_full"], 1)


class TestRunEvents(unittest.TestCase):
    def test_writes_outputs_and_reconciliation(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw.csv"
            lines = [f"{r['ts_utc']:%Y-%m-%d %H:%M:%S}\t{r['open']}\t{r['high']}\t{r['low']}\t"
                     f"{r['close']}\t60\t100" for r in week_2016_03()]
            raw.write_text("Time\tOpen\tHigh\tLow\tClose\tVolume\n" + "\n".join(lines))
            out = Path(tmp) / "de"
            recon = ev.run_events(raw, out, 0, 8, write_report=False)
            self.assertEqual((recon["events_all"], recon["events_valid"]), (3, 2))
            self.assertEqual(recon["valid_dev_PDH"] + recon["valid_dev_PDL"], 2)
            names = sorted(p.name for p in out.iterdir())
            self.assertEqual(names, ["de_rekonsiliasi.json", "dq_dst_verification.csv",
                                     "gold_pdhl_events.csv", "silver_sessions.csv",
                                     "silver_xauusd_h1.csv"])
            first = (out / "gold_pdhl_events.csv").read_bytes().split(b"\n")[0]
            self.assertTrue(first.endswith(b"\r"))         # format CRLF dipertahankan


if __name__ == "__main__":
    unittest.main()
