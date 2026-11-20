"""Tes pembagian data, penguncian data uji, dan guard kebocoran (Proposal v2, IV.II.4)."""
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pdhl_data import common  # noqa: E402
from pdhl_data.split import LockViolation, split_events, split_rows_by_date, write_lock  # noqa: E402


def ev(eid, split, sd, valid=True, k="0", T="8"):
    return {"event_id": eid, "split": split, "session_date": sd, "decision_k": k,
            "horizon_T": T, "is_valid": "true" if valid else "false"}


class TestGuard(unittest.TestCase):
    def test_rejects_rows_after_calib_end(self):
        with self.assertRaises(common.TestDataLeak):
            common.assert_no_test_data([{"session_date": "2024-12-31"},
                                        {"session_date": "2025-01-02"}], source="x")

    def test_accepts_rows_up_to_calib_end(self):
        common.assert_no_test_data([{"session_date": "2015-01-02"},
                                    {"session_date": "2024-12-31"}], source="x")

    def test_rejects_path_inside_lock_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / "kunci_uji"
            lock.mkdir()
            with self.assertRaises(common.TestDataLeak):
                common.assert_not_locked_path(lock / "events_test.csv", lock_dir=lock)
            with self.assertRaises(common.TestDataLeak):
                common.assert_not_locked_path(lock, lock_dir=lock)
            common.assert_not_locked_path(Path(tmp) / "data" / "events_dev.csv", lock_dir=lock)

    def test_default_guard_rejects_every_kunci_uji_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("kunci_uji", "kunci_uji_lama"):
                with self.subTest(lock=name), self.assertRaises(common.TestDataLeak):
                    common.assert_not_locked_path(Path(tmp) / "runs" / name / "events_test.csv")
            common.assert_not_locked_path(Path(tmp) / "runs" / "tesis" / "data" / "events_dev.csv")

    def test_default_guard_is_case_insensitive(self):
        for name in ("KUNCI_UJI", "Kunci_Uji_Lama"):
            with self.subTest(lock=name), self.assertRaises(common.TestDataLeak):
                common.assert_not_locked_path(Path("/x") / name / "a.csv")

    def test_load_model_csv_refuses_lock_path_even_with_dev_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "kunci_uji" / "x.csv"
            common.write_csv(p, [{"session_date": "2020-03-03"}], ["session_date"])
            with self.assertRaises(common.TestDataLeak):
                common.load_model_csv(p)

    def test_load_model_csv_refuses_test_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.csv"
            common.write_csv(p, [{"session_date": "2025-03-03", "v": "1"}], ["session_date", "v"])
            with self.assertRaises(common.TestDataLeak):
                common.load_model_csv(p)

    def test_load_model_csv_refuses_file_without_date_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "fitur.csv"
            common.write_csv(p, [{"decision_ts_ny": "2025-02-03 10:00:00"}], ["decision_ts_ny"])
            with self.assertRaises(common.TestDataLeak):
                common.load_model_csv(p)

    def test_rejects_rows_without_date_or_empty_date(self):
        with self.assertRaises(common.TestDataLeak):
            common.assert_no_test_data([{"ts": "2025-01-02"}], source="x")
        with self.assertRaises(common.TestDataLeak):
            common.assert_no_test_data([{"session_date": ""}], source="x")

    def test_parse_bool_is_strict(self):
        self.assertTrue(common.parse_bool("true"))
        self.assertFalse(common.parse_bool("false"))
        for bad in ("True", "1", "", "ya"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                common.parse_bool(bad)

    def test_sha256_ids_is_order_independent(self):
        self.assertEqual(common.sha256_ids(["b", "a", "c"]), common.sha256_ids(["c", "b", "a"]))
        self.assertNotEqual(common.sha256_ids(["a"]), common.sha256_ids(["a", "b"]))


class TestSplitEvents(unittest.TestCase):
    EVENTS = [ev("d1", "dev", "2016-01-05"), ev("d2", "dev", "2023-12-29", valid=False),
              ev("c1", "calib", "2024-06-03"), ev("t1", "test", "2025-01-02"),
              ev("t2", "test", "2026-07-30", valid=False)]

    def test_dev_calib_valid_only_test_keeps_all(self):
        parts = split_events(self.EVENTS, k=0, T=8)
        self.assertEqual([e["event_id"] for e in parts["dev"]], ["d1"])
        self.assertEqual([e["event_id"] for e in parts["calib"]], ["c1"])
        self.assertEqual([e["event_id"] for e in parts["test"]], ["t1", "t2"])

    def test_rejects_events_detected_with_other_parameters(self):
        with self.assertRaises(ValueError):
            split_events(self.EVENTS + [ev("x", "dev", "2017-01-03", k="1")], k=0, T=8)

    def test_rejects_split_label_inconsistent_with_date(self):
        with self.assertRaises(ValueError):
            split_events([ev("bad", "calib", "2025-02-03")], k=0, T=8)

    def test_rows_by_date(self):
        rows = [{"session_date": s} for s in ("2015-01-02", "2024-12-31", "2025-01-02")]
        before, after = split_rows_by_date(rows)
        self.assertEqual(len(before), 2)
        self.assertEqual(after, [{"session_date": "2025-01-02"}])


class TestWriteLock(unittest.TestCase):
    FILES = {"events_test.csv": "a,b\n1,2\n"}

    def test_files_read_only_and_manifest_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / "kunci_uji"
            m = write_lock(lock, self.FILES, {"info": 1})
            mode = os.stat(lock / "events_test.csv").st_mode
            self.assertFalse(mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
            self.assertIn("events_test.csv", m["berkas"])
            self.assertTrue((lock / "MANIFEST_KUNCI_UJI.json").exists())

    def test_rerun_with_same_content_is_noop(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / "kunci_uji"
            first = write_lock(lock, self.FILES, {"info": 1})
            again = write_lock(lock, self.FILES, {"info": 2})
            self.assertEqual(first["dikunci_pada"], again["dikunci_pada"])

    @unittest.skipIf(hasattr(os, "geteuid") and os.geteuid() == 0,
                     "root mengabaikan izin berkas (mis. Google Colab); kunci tetap dijaga SHA-256")
    def test_lock_dir_is_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / "kunci_uji"
            write_lock(lock, self.FILES, {})
            self.assertFalse(os.stat(lock).st_mode & stat.S_IWUSR)
            with self.assertRaises(PermissionError):
                (lock / "baru.csv").write_text("x")

    def test_stale_tmp_from_failed_run_is_discarded(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / "kunci_uji"
            stale = Path(tmp) / "kunci_uji.tmp"
            stale.mkdir()
            (stale / "events_test.csv").write_text("setengah jadi")
            os.chmod(stale / "events_test.csv", 0o444)
            write_lock(lock, self.FILES, {})
            self.assertFalse(stale.exists())
            self.assertEqual((lock / "events_test.csv").read_text(), self.FILES["events_test.csv"])

    def test_tampered_locked_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / "kunci_uji"
            write_lock(lock, self.FILES, {})
            os.chmod(lock, 0o755)
            os.chmod(lock / "events_test.csv", 0o644)
            (lock / "events_test.csv").write_text("a,b\n9,9\n")
            with self.assertRaises(LockViolation):
                write_lock(lock, self.FILES, {})

    def test_deleted_locked_file_raises_lock_violation(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / "kunci_uji"
            write_lock(lock, self.FILES, {})
            os.chmod(lock, 0o755)
            (lock / "events_test.csv").unlink()
            with self.assertRaises(LockViolation):
                write_lock(lock, self.FILES, {})

    def test_foreign_file_in_lock_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / "kunci_uji"
            write_lock(lock, self.FILES, {})
            os.chmod(lock, 0o755)
            (lock / "fitur_test.csv").write_text("x\n")
            with self.assertRaisesRegex(LockViolation, "asing"):
                write_lock(lock, self.FILES, {})

    def test_rerun_with_changed_content_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / "kunci_uji"
            write_lock(lock, self.FILES, {})
            with self.assertRaises(LockViolation):
                write_lock(lock, {"events_test.csv": "a,b\n1,3\n"}, {})


if __name__ == "__main__":
    unittest.main()
