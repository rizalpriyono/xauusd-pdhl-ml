"""Tes pelabelan triple barrier dan satuan R (Proposal v2, IV.II.3 dan Tabel 4.3)."""
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pdhl_data.labels import barriers, label_event, label_events  # noqa: E402

A, B, T = 2.0, 1.0, 8


def bar(high, low, close=None, open_=None):
    close = (high + low) / 2 if close is None else close
    open_ = close if open_ is None else open_
    return {"open": open_, "high": high, "low": low, "close": close}


def flat(n, high=100.4, low=99.6, close=100.0):
    return [bar(high, low, close) for _ in range(n)]


class TestBarriers(unittest.TestCase):
    def test_pdh_target_below_stop_above(self):
        self.assertEqual(barriers(100.0, 1.0, +1, A, B), (98.0, 101.0))

    def test_pdl_mirrored(self):
        self.assertEqual(barriers(100.0, 1.0, -1, A, B), (102.0, 99.0))


class TestLabelEventPDH(unittest.TestCase):
    def test_reversal_when_target_first(self):
        w = [bar(100.5, 99.0)] + [bar(99.5, 97.9)] + flat(T - 2)
        r = label_event(w, 100.0, 1.0, +1, A, B, T)
        self.assertEqual(r["label"], "Reversal")
        self.assertEqual(r["barrier_bar_offset"], 2)
        self.assertFalse(r["both_same_bar"])
        self.assertEqual((r["R_rev"], r["R_cont"]), (2.0, -1.0))

    def test_target_touched_exactly_counts(self):
        w = [bar(100.2, 98.0)] + flat(T - 1)
        self.assertEqual(label_event(w, 100.0, 1.0, +1, A, B, T)["label"], "Reversal")

    def test_continuation_when_stop_first(self):
        w = [bar(101.0, 99.8)] + [bar(99.0, 97.0)] + flat(T - 2)
        r = label_event(w, 100.0, 1.0, +1, A, B, T)
        self.assertEqual(r["label"], "Continuation")
        self.assertEqual(r["barrier_bar_offset"], 1)
        self.assertEqual((r["R_rev"], r["R_cont"]), (-1.0, 0.5))

    def test_both_barriers_same_bar_is_continuation(self):
        w = [bar(101.2, 97.5)] + flat(T - 1)
        r = label_event(w, 100.0, 1.0, +1, A, B, T)
        self.assertEqual(r["label"], "Continuation")
        self.assertTrue(r["both_same_bar"])
        self.assertEqual(r["barrier_bar_offset"], 1)

    def test_gap_through_stop_is_continuation(self):
        w = [bar(102.0, 101.3, close=101.8, open_=101.5)] + flat(T - 1)
        r = label_event(w, 100.0, 1.0, +1, A, B, T)
        self.assertEqual(r["label"], "Continuation")
        self.assertFalse(r["both_same_bar"])

    def test_netral_r_from_close_of_bar_T(self):
        w = flat(T - 1) + [bar(100.0, 99.2, close=99.5)]
        r = label_event(w, 100.0, 1.0, +1, A, B, T)
        self.assertEqual(r["label"], "Netral")
        self.assertIsNone(r["barrier_bar_offset"])
        self.assertEqual(r["close_T"], 99.5)
        self.assertAlmostEqual(r["R_rev"], 0.5)     # sell: (100 - 99.5) / (1 x ATR)
        self.assertAlmostEqual(r["R_cont"], -0.25)  # buy: (99.5 - 100) / (2 x ATR)

    def test_bar_after_T_is_ignored(self):
        with self.assertRaises(ValueError):
            label_event(flat(T + 1), 100.0, 1.0, +1, A, B, T)

    def test_atr_must_be_positive(self):
        with self.assertRaises(ValueError):
            label_event(flat(T), 100.0, 0.0, +1, A, B, T)

    def test_direction_must_be_unit(self):
        with self.assertRaises(ValueError):
            label_event(flat(T), 100.0, 1.0, 0, A, B, T)


class TestLabelEventPDL(unittest.TestCase):
    def test_reversal_when_target_above_first(self):
        w = [bar(102.0, 99.5)] + flat(T - 1)
        r = label_event(w, 100.0, 1.0, -1, A, B, T)
        self.assertEqual(r["label"], "Reversal")
        self.assertEqual((r["R_rev"], r["R_cont"]), (2.0, -1.0))

    def test_continuation_when_stop_below_first(self):
        w = [bar(100.5, 99.0)] + [bar(102.5, 100.0)] + flat(T - 2)
        r = label_event(w, 100.0, 1.0, -1, A, B, T)
        self.assertEqual(r["label"], "Continuation")
        self.assertEqual(r["barrier_bar_offset"], 1)

    def test_both_barriers_same_bar_is_continuation(self):
        w = [bar(102.3, 98.7)] + flat(T - 1)
        r = label_event(w, 100.0, 1.0, -1, A, B, T)
        self.assertEqual(r["label"], "Continuation")
        self.assertTrue(r["both_same_bar"])

    def test_netral_r_mirrored(self):
        w = flat(T - 1) + [bar(100.8, 100.0, close=100.5)]
        r = label_event(w, 100.0, 1.0, -1, A, B, T)
        self.assertEqual(r["label"], "Netral")
        self.assertAlmostEqual(r["R_rev"], 0.5)     # buy: (100.5 - 100) / 1
        self.assertAlmostEqual(r["R_cont"], -0.25)  # sell: (100 - 100.5) / 2


class TestInvariants(unittest.TestCase):
    def test_random_walk_invariants(self):
        rng = random.Random(7)
        seen = set()
        for _ in range(3000):
            d = rng.choice((1, -1))
            price, w = 100.0, []
            for _ in range(T):
                o = price
                price += rng.gauss(0, 0.6)
                hi = max(o, price) + abs(rng.gauss(0, 0.3))
                lo = min(o, price) - abs(rng.gauss(0, 0.3))
                w.append(bar(hi, lo, close=price, open_=o))
            r = label_event(w, 100.0, 1.0, d, A, B, T)
            seen.add(r["label"])
            if r["label"] == "Netral":
                self.assertTrue(-1 < r["R_rev"] < A / B)
                self.assertTrue(-1 < r["R_cont"] < B / A)
            else:
                self.assertIn(r["barrier_bar_offset"], range(1, T + 1))
        self.assertEqual(seen, {"Reversal", "Continuation", "Netral"})


def make_bars(session_dates, closes=None, atr=1.0):
    out = []
    for i, sd in enumerate(session_dates):
        c = 100.0 if closes is None else closes[i]
        out.append({"bar_idx": str(i), "session_date": sd, "ts_ny": f"{sd} {i % 24:02d}:00:00",
                    "open": str(c), "high": str(c + 0.4), "low": str(c - 0.4), "close": str(c),
                    "atr14_h1": str(atr), "missing_before": "false"})
    return out


def make_event(eid, split, sd, dec_idx, entry=100.0, atr=1.0, level="PDH"):
    return {"event_id": eid, "split": split, "session_date": sd, "level_type": level,
            "direction_d": "1" if level == "PDH" else "-1", "decision_bar_idx": str(dec_idx),
            "decision_ts_ny": f"{sd} 10:00:00", "entry_price": str(entry), "atr14_h1": str(atr),
            "decision_k": "0", "horizon_T": str(T)}


class TestLabelEvents(unittest.TestCase):
    KUNCI = {"a": A, "b": B, "T": T, "k": 0}
    ENDS = {"dev": "2023-12-31", "calib": "2024-12-31"}

    def test_labels_and_purges_window_crossing_split_end(self):
        bars = make_bars(["2023-12-28"] * 10 + ["2024-01-02"] * 10)
        events = [make_event("ok", "dev", "2023-12-28", 0),
                  make_event("crosses", "dev", "2023-12-28", 5)]   # bar 13 milik 2024
        rows, purged = label_events(events, bars, self.KUNCI, self.ENDS)
        self.assertEqual([r["event_id"] for r in rows], ["ok"])
        self.assertEqual([p["event_id"] for p in purged], ["crosses"])
        self.assertEqual(rows[0]["label"], "Netral")
        self.assertEqual(rows[0]["label_end_bar_idx"], 8)

    def test_purges_window_beyond_available_bars(self):
        bars = make_bars(["2024-12-30"] * 6)
        rows, purged = label_events([make_event("late", "calib", "2024-12-30", 2)],
                                    bars, self.KUNCI, self.ENDS)
        self.assertEqual(rows, [])
        self.assertEqual(purged[0]["purge_reason"], "jendela_label_melewati_data_tersedia")

    def test_entry_must_match_decision_bar_close(self):
        bars = make_bars(["2023-06-01"] * 12)
        with self.assertRaises(ValueError):
            label_events([make_event("x", "dev", "2023-06-01", 0, entry=101.0)],
                         bars, self.KUNCI, self.ENDS)

    def test_window_ending_exactly_on_split_end_is_kept(self):
        bars = make_bars(["2023-12-29"] * 4 + ["2023-12-31"] * 5 + ["2024-01-02"] * 3)
        rows, purged = label_events([make_event("edge", "dev", "2023-12-29", 0)],
                                    bars, self.KUNCI, self.ENDS)
        self.assertEqual(([r["event_id"] for r in rows], purged), (["edge"], []))
        self.assertEqual(rows[0]["label_end_session_date"], "2023-12-31")

    def test_decision_bar_beyond_data_is_purged(self):
        bars = make_bars(["2024-12-30"] * 4)
        rows, purged = label_events([make_event("far", "calib", "2024-12-30", 7)],
                                    bars, self.KUNCI, self.ENDS)
        self.assertEqual((rows, len(purged)), ([], 1))

    def test_split_inconsistent_with_date_raises(self):
        bars = make_bars(["2024-03-01"] * 12)
        with self.assertRaises(ValueError):
            label_events([make_event("x", "dev", "2024-03-01", 0)], bars, self.KUNCI, self.ENDS)

    def test_nan_price_raises(self):
        bars = make_bars(["2023-06-01"] * 12)
        bars[3]["low"] = "nan"
        with self.assertRaises(ValueError):
            label_events([make_event("n", "dev", "2023-06-01", 0)], bars, self.KUNCI, self.ENDS)

    def test_refuses_test_events(self):
        bars = make_bars(["2024-06-03"] * 12)
        with self.assertRaises(RuntimeError):
            label_events([make_event("t", "test", "2025-01-02", 0)], bars, self.KUNCI, self.ENDS)

    def test_flags_label_crossing_calendar_year(self):
        bars = make_bars(["2022-12-30"] * 4 + ["2023-01-02"] * 8)
        rows, _ = label_events([make_event("y", "dev", "2022-12-30", 1)],
                               bars, self.KUNCI, self.ENDS)
        self.assertTrue(rows[0]["label_crosses_year"])



# ---------------------------------------------------------------- geometri lain (a=4, b=2, T=24)
# Parameter di luar Kunci-1, hanya untuk memastikan fungsi label berlaku umum (bukan konfigurasi).
A4, B2, T24 = 4.0, 2.0, 24


def bar_art(high, low, close=None):
    return {"high": high, "low": low, "close": (high + low) / 2 if close is None else close}


def flat_art(n, price=100.0):
    return [bar_art(price + 0.5, price - 0.5, price) for _ in range(n)]


class TestLabelGeometriLain(unittest.TestCase):
    def test_barriers_pdh(self):
        self.assertEqual(barriers(100.0, 1.0, +1, A4, B2), (96.0, 102.0))

    def test_barriers_pdl(self):
        self.assertEqual(barriers(100.0, 1.0, -1, A4, B2), (104.0, 98.0))

    def test_reversal_r(self):
        w = flat_art(5) + [bar_art(99.0, 95.9)] + flat_art(T24 - 6, 97.0)
        r = label_event(w, 100.0, 1.0, +1, A4, B2, T24)
        self.assertEqual((r["label"], r["barrier_bar_offset"]), ("Reversal", 6))
        self.assertEqual((r["R_rev"], r["R_cont"]), (2.0, -1.0))

    def test_continuation_r(self):
        w = flat_art(3) + [bar_art(102.1, 100.0)] + flat_art(T24 - 4, 101.0)
        r = label_event(w, 100.0, 1.0, +1, A4, B2, T24)
        self.assertEqual((r["label"], r["R_rev"], r["R_cont"]), ("Continuation", -1.0, 0.5))

    def test_same_bar_is_continuation(self):
        w = [bar_art(102.5, 95.5)] + flat_art(T24 - 1)
        r = label_event(w, 100.0, 1.0, +1, A4, B2, T24)
        self.assertEqual((r["label"], r["both_same_bar"]), ("Continuation", True))

    def test_hit_after_T_is_ignored(self):
        w = flat_art(T24 - 1) + [bar_art(100.5, 99.5, 99.0)]
        r = label_event(w, 100.0, 1.0, +1, A4, B2, T24)
        self.assertEqual(r["label"], "Netral")
        # sell di 100, close ke-T = 99: +1 / (2 x ATR) = +0,5R; continuation = -1 / (4 x ATR)
        self.assertAlmostEqual(r["R_rev"], 0.5)
        self.assertAlmostEqual(r["R_cont"], -0.25)

    def test_r_cont_identity(self):
        for w in ([bar_art(99.0, 95.9)] + flat_art(T24 - 1), [bar_art(102.1, 100.0)] + flat_art(T24 - 1),
                  flat_art(T24 - 1) + [bar_art(101.0, 100.0, 100.7)]):
            r = label_event(w, 100.0, 1.0, +1, A4, B2, T24)
            self.assertAlmostEqual(r["R_cont"], -(B2 / A4) * r["R_rev"])


if __name__ == "__main__":
    unittest.main()
