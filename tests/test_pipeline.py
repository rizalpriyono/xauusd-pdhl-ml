"""Tes alur lengkap split (build, manifest, rekonsiliasi) dan rekonsiliasi label pada data sintetis."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pdhl_data import common  # noqa: E402
from pdhl_data import labels as ltb  # noqa: E402
from pdhl_data import split as sk  # noqa: E402
from pdhl_data.pipeline import RunPaths  # noqa: E402

KUNCI = {"a": 2.0, "b": 1.0, "T": 4, "k": 0, "tanggal_kunci": "2026-11-07"}
CFG = {"name": "uji", "label": KUNCI,
       "lock": {"aturan": "aturan uji", "kunci_key": "kunci_1", "manifest": "M.json",
                "dibuka_pada": "Kunci-2"}}


def write_de_output(root, bar_idx=None, horizon="4"):
    sessions = ["2015-06-01", "2016-02-01", "2024-03-01", "2024-12-31", "2025-01-02", "2026-07-31"]
    bars = [{"bar_idx": i, "session_date": sd, "close": 100.0 + i}
            for i, sd in enumerate(s for s in sessions for _ in range(3))]
    if bar_idx:
        for i, v in bar_idx.items():
            bars[i]["bar_idx"] = v
    events = [
        {"event_id": "2016-02-01_PDH", "session_date": "2016-02-01", "split": "dev",
         "level_type": "PDH", "is_valid": "true"},
        {"event_id": "2016-02-01_PDL", "session_date": "2016-02-01", "split": "dev",
         "level_type": "PDL", "is_valid": "false"},
        {"event_id": "2024-03-01_PDL", "session_date": "2024-03-01", "split": "calib",
         "level_type": "PDL", "is_valid": "true"},
        {"event_id": "2025-01-02_PDH", "session_date": "2025-01-02", "split": "test",
         "level_type": "PDH", "is_valid": "true"},
        {"event_id": "2026-07-31_PDL", "session_date": "2026-07-31", "split": "test",
         "level_type": "PDL", "is_valid": "false"},
    ]
    for e in events:
        e.update(decision_k="0", horizon_T=horizon)
    common.write_csv(root / "gold_pdhl_events.csv", events,
                     ["event_id", "session_date", "split", "level_type", "decision_k", "horizon_T",
                      "is_valid"])
    common.write_csv(root / "silver_xauusd_h1.csv", bars, ["bar_idx", "session_date", "close"])
    common.write_csv(root / "silver_sessions.csv",
                     [{"session_date": s, "split": common.split_of(s) or ""} for s in sessions],
                     ["session_date", "split"])


class TestBuild(unittest.TestCase):
    def test_build_separates_test_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_de_output(root)
            data_files, lock_files, stats = sk.build(root, KUNCI)
            self.assertEqual(stats["n_valid"], {"dev": 1, "calib": 1, "test": 1})
            self.assertEqual(stats["n_test_all"], 2)
            self.assertEqual((stats["bars_devcalib"], stats["bars_test"]), (12, 6))
            self.assertEqual(stats["bar_idx_test"], [12, 17])
            self.assertEqual(stats["test_valid_ids_sha256"], common.sha256_ids(["2025-01-02_PDH"]))
            self.assertNotIn("2025", "".join(data_files.values()))
            self.assertIn("2025-01-02_PDH", lock_files["events_test.csv"])
            self.assertIn("2026-07-31_PDL", lock_files["events_test.csv"])

            raw = root / "raw.csv"
            raw.write_text("x\n")
            info = sk.manifest_info(CFG, stats, data_files, raw)
            self.assertEqual(info["event_uji"]["valid"], 1)
            self.assertEqual(info["kunci_1"]["T"], 4)
            self.assertEqual(info["konfigurasi"], "uji")
            self.assertIn("raw.csv", info["sumber"])
            self.assertEqual(set(info["skrip"]), {f"pdhl_data/{n}" for n in sk.SCRIPTS})
            recon = sk.reconciliation(stats)
            self.assertEqual(recon["locked_test_bars"], 6)
            self.assertEqual(recon["sum_close_test"], round(sum(100.0 + i for i in range(12, 18)), 4))

    def test_build_rejects_gap_in_bar_idx(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_de_output(root, bar_idx={5: 99})
            with self.assertRaises(ValueError):
                sk.build(root, KUNCI)

    def test_build_rejects_kunci_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_de_output(root)
            with self.assertRaisesRegex(ValueError, "Jalankan ulang"):
                sk.build(root, {**KUNCI, "T": 8})   # event dideteksi dengan T=4

    def test_build_rejects_counts_different_from_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_de_output(root, horizon="8")
            with self.assertRaisesRegex(ValueError, "acuan"):
                sk.build(root, {**KUNCI, "T": 8}, {"dev": 1841, "calib": 227, "test": 341})

    def test_build_accepts_matching_reference_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_de_output(root)
            _, _, stats = sk.build(root, KUNCI, {"dev": 1, "calib": 1, "test": 1})
            self.assertEqual(stats["n_valid"]["test"], 1)

    def test_run_split_writes_data_and_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = RunPaths(root / "run")
            paths.de_output.mkdir(parents=True)
            write_de_output(paths.de_output)
            raw = root / "raw.csv"
            raw.write_text("x\n")
            cfg = {**CFG, "expected_valid": None}
            recon = sk.run_split(cfg, paths, raw)
            self.assertEqual(recon["locked_test_events_valid"], 1)
            self.assertTrue((paths.lock / "M.json").exists())
            self.assertIn("Kunci-2", (paths.lock / "README.md").read_text())
            self.assertEqual(len(common.load_model_csv(paths.data / "events_dev.csv")), 1)
            # run ulang dengan isi sama: kunci hanya diverifikasi
            again = sk.run_split(cfg, paths, raw)
            self.assertEqual(recon, again)

    def _locked_run(self, root):
        paths = RunPaths(root / "run")
        paths.de_output.mkdir(parents=True)
        write_de_output(paths.de_output)
        raw = root / "raw.csv"
        raw.write_text("x\n")
        cfg = {**CFG, "expected_valid": None}
        sk.run_split(cfg, paths, raw)
        return cfg, paths, raw

    def test_changed_label_parameters_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, paths, raw = self._locked_run(Path(tmp))
            changed = {**cfg, "label": {**KUNCI, "a": 3.0}}
            with self.assertRaisesRegex(sk.LockViolation, "adendum"):
                sk.run_split(changed, paths, raw)
            with self.assertRaisesRegex(sk.LockViolation, "adendum"):
                ltb.run_labels({**cfg, "label": {**KUNCI, "b": 0.5}}, paths)

    def test_labels_refuse_tampered_data_or_missing_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, paths, _ = self._locked_run(Path(tmp))
            p = paths.data / "events_dev.csv"
            p.write_text(p.read_text() + "2016-02-02_PDH,2016-02-02,dev,PDH,0,4,true\n")
            with self.assertRaisesRegex(sk.LockViolation, "berbeda"):
                ltb.run_labels(cfg, paths)
            with self.assertRaisesRegex(sk.LockViolation, "split dulu"):
                ltb.run_labels(cfg, RunPaths(Path(tmp) / "kosong"))


def lbl_row(split, label, offset, both=False, r_rev=0.0, r_cont=0.0, crosses=False):
    return {"split": split, "label": label, "barrier_bar_offset": offset, "both_same_bar": both,
            "R_rev": r_rev, "R_cont": r_cont, "label_crosses_year": crosses}


class TestLabelReconciliation(unittest.TestCase):
    def test_counts_and_sums(self):
        rows = [lbl_row("dev", "Reversal", 2, r_rev=2.0, r_cont=-1.0),
                lbl_row("dev", "Continuation", 1, both=True, r_rev=-1.0, r_cont=0.5),
                lbl_row("dev", "Netral", None, r_rev=0.4, r_cont=-0.2, crosses=True),
                lbl_row("calib", "Netral", None, r_rev=-0.5, r_cont=0.25)]
        recon = ltb.reconciliation(rows, [{"split": "calib"}])
        self.assertEqual((recon["dev_Reversal"], recon["dev_Continuation"], recon["dev_Netral"]),
                         (1, 1, 1))
        self.assertEqual(recon["dev_both_same_bar"], 1)
        self.assertEqual(recon["dev_sum_barrier_offset"], 3)
        self.assertAlmostEqual(recon["dev_sum_R_rev"], 1.4)
        self.assertEqual((recon["dev_purged"], recon["calib_purged"]), (0, 1))
        self.assertEqual(recon["label_crosses_year"], 1)


class TestLabelEventsErrors(unittest.TestCase):
    KUNCI = {"a": 2.0, "b": 1.0, "T": 4, "k": 0}

    def _bars(self, n=10):
        return [{"bar_idx": str(i), "session_date": "2020-03-02", "ts_ny": f"2020-03-02 {i:02d}:00:00",
                 "high": "100.4", "low": "99.6", "close": "100.0", "atr14_h1": "1.0",
                 "missing_before": "false"} for i in range(n)]

    def _event(self, **kw):
        e = {"event_id": "e", "split": "dev", "session_date": "2020-03-02", "level_type": "PDH",
             "direction_d": "1", "decision_bar_idx": "1", "decision_ts_ny": "2020-03-02 01:00:00",
             "entry_price": "100.0", "atr14_h1": "1.0"}
        return {**e, **kw}

    def test_missing_bar_in_window_raises(self):
        bars = self._bars()
        bars[3]["missing_before"] = "true"
        with self.assertRaises(ValueError):
            ltb.label_events([self._event()], bars, self.KUNCI)

    def test_atr_mismatch_raises(self):
        with self.assertRaises(ValueError):
            ltb.label_events([self._event(atr14_h1="1.5")], self._bars(), self.KUNCI)

    def test_non_contiguous_bars_raise(self):
        bars = self._bars()
        bars[4]["bar_idx"] = "40"
        with self.assertRaises(ValueError):
            ltb.label_events([self._event()], bars, self.KUNCI)

    def test_barrier_timestamp_recorded(self):
        bars = self._bars()
        bars[3].update(high="101.5")
        rows, _ = ltb.label_events([self._event()], bars, self.KUNCI)
        self.assertEqual((rows[0]["label"], rows[0]["barrier_ts_ny"]),
                         ("Continuation", "2020-03-02 03:00:00"))


if __name__ == "__main__":
    unittest.main()
