"""Utilitas bersama: periode penelitian, IO CSV, hash, dan guard data uji.

Semua kode pemodelan WAJIB memuat data lewat `load_model_csv`. Loader ini menolak berkas di dalam
folder kunci uji (nama folder diawali `kunci_uji`) dan menolak baris dengan session_date sesudah
akhir kalibrasi (2024-12-31), sehingga data uji tidak terbaca sebelum eksekusi tunggal.
"""
import csv
import datetime as dt
import hashlib
import io
from pathlib import Path

CLASSES = ("Reversal", "Continuation", "Netral")

WARMUP_START = dt.date(2015, 1, 1)      # pemanasan ATR; event baru dihitung mulai 2016
RESEARCH_START = dt.date(2016, 1, 1)
DEV_END = dt.date(2023, 12, 31)
CALIB_END = dt.date(2024, 12, 31)
RESEARCH_END = dt.date(2026, 7, 31)
SPLITS = (("dev", DEV_END), ("calib", CALIB_END), ("test", RESEARCH_END))
SPLIT_END = {"dev": DEV_END.isoformat(), "calib": CALIB_END.isoformat()}   # bagian yang dilabeli
LOCK_DIR_PREFIX = "kunci_uji"


class TestDataLeak(RuntimeError):
    """Kode pemodelan mencoba membaca data uji sebelum eksekusi tunggal."""

    __test__ = False   # bukan kelas tes unittest/pytest


def parse_date(s):
    return dt.date.fromisoformat(s)


def parse_bool(s):
    """Boolean dari CSV pipeline ("true"/"false"); nilai lain dianggap error, bukan False."""
    if s == "true":
        return True
    if s == "false":
        return False
    raise ValueError(f"nilai boolean tidak dikenal: {s!r}")


def split_of(session_date):
    """Bagian data (dev/calib/test) dari session_date berformat ISO; None di luar periode."""
    d = parse_date(session_date)
    if d < RESEARCH_START or d > RESEARCH_END:
        return None
    if d <= DEV_END:
        return "dev"
    return "calib" if d <= CALIB_END else "test"


# ---------------------------------------------------------------- IO
def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def csv_header(path):
    with open(path, newline="") as f:
        return next(csv.reader(f))


def fmt(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return repr(round(v, 10))
    return "" if v is None else str(v)


def csv_text(rows, cols):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(cols)
    for r in rows:
        w.writerow([fmt(r.get(c)) for c in cols])
    return buf.getvalue()


def write_csv(path, rows, cols):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(csv_text(rows, cols))


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    return sha256_bytes(Path(path).read_bytes())


def sha256_ids(ids):
    """Checksum daftar event_id yang tidak bergantung urutan; sama dengan
    sha2(concat_ws(';', array_sort(collect_list(event_id))), 256) di Spark."""
    return sha256_bytes(";".join(sorted(ids)).encode())


# ---------------------------------------------------------------- guard data uji
def _is_lock_dir(path):
    return path.name.lower().startswith(LOCK_DIR_PREFIX)     # sistem berkas macOS tidak peka huruf


def assert_not_locked_path(path, lock_dir=None):
    """Tolak path di dalam folder kunci uji.

    Tanpa `lock_dir`, setiap folder yang namanya diawali `kunci_uji` (tidak peka huruf) dianggap
    kunci: kunci di `runs/<config>/kunci_uji/` maupun kunci resmi lama `Pemodelan PDH-PDL/kunci_uji/`."""
    p = Path(path).resolve()
    if lock_dir is not None:
        lock = Path(lock_dir).resolve()
        hit = p == lock or lock in p.parents
    else:
        hit = any(_is_lock_dir(q) for q in (p, *p.parents))
    if hit:
        raise TestDataLeak(f"{path} berada di folder kunci uji: data uji terkunci sampai "
                           "eksekusi tunggal")


def assert_no_test_data(rows, date_col="session_date", source=""):
    if rows and date_col not in rows[0]:
        raise TestDataLeak(f"{source}: kolom {date_col} tidak ada, guard data uji tidak bisa dicek")
    for r in rows:
        v = r[date_col]
        if not v:
            raise TestDataLeak(f"{source}: ada baris dengan {date_col} kosong; tidak bisa dicek")
        if parse_date(v[:10]) > CALIB_END:
            raise TestDataLeak(f"{source}: baris {date_col}={v} sesudah {CALIB_END}; "
                               "data uji terkunci sampai eksekusi tunggal")


def load_model_csv(path, date_col="session_date"):
    """Satu-satunya pintu masuk data untuk kode pemodelan (GB, LSTM, baseline, evaluasi)."""
    assert_not_locked_path(path)
    if date_col not in csv_header(path):
        raise TestDataLeak(f"{path}: kolom {date_col} tidak ada, guard data uji tidak bisa dicek")
    rows = read_csv(path)
    assert_no_test_data(rows, date_col, source=str(path))
    return rows
