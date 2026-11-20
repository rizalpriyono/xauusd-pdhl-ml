"""Tahap 2: pembagian data dan penguncian data uji.

Proposal Final v2, IV.II.4 (Tabel 4.4) dan IV.II.11. Masukannya keluaran tahap 1 (`de_output/`).
Keluarannya:

- `data/`      : event valid dev (2016-2023) dan calib (2024), bar dan sesi s.d. 2024-12-31
                 (termasuk pemanasan 2015). Hanya folder ini yang dibaca kode pemodelan.
- `kunci_uji/` : event, bar, dan sesi Jan 2025 - Jul 2026 beserta manifest SHA-256 bertanggal.
                 Berkas dibuat hanya-baca dan baru dibuka pada eksekusi tunggal.

Menjalankan ulang aman: bila manifest sudah ada, isi baru harus identik dengan yang terkunci,
kalau tidak proses berhenti (LockViolation).
"""
import datetime as dt
import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

from . import common
from .common import CALIB_END

READ_ONLY = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
READ_ONLY_DIR = READ_ONLY | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
SOURCES = ("gold_pdhl_events.csv", "silver_xauusd_h1.csv", "silver_sessions.csv")
PACKAGE_DIR = Path(__file__).resolve().parent
SCRIPTS = ("events.py", "split.py", "labels.py", "common.py", "config.py")
LOCK_PARAMS = ("a", "b", "T", "k", "tanggal_kunci")

LOCK_README = """# Data uji terkunci (Januari 2025 - Juli 2026), konfigurasi `{name}`

Folder ini berisi event, bar, dan sesi data uji tertutup. Berkas dibuat hanya-baca oleh
`python -m pdhl_data run --config configs/{name}.json`.

- Jangan dibuka, diubah, atau dipakai untuk label, fitur, model, atau analisis sebelum
  **{dibuka_pada}**.
- Pada eksekusi tunggal, cocokkan dulu SHA-256 setiap berkas dengan `{manifest}`.
  Bila ada yang berbeda, kunci dianggap rusak dan harus dilaporkan.
- Kode pemodelan memuat data lewat `pdhl_data.common.load_model_csv`, yang menolak path di folder ini.
"""


class LockViolation(RuntimeError):
    """Isi data uji berbeda dari yang tercatat di manifest kunci."""


def split_events(events, k, T):
    """dev/calib: hanya event valid. test: semua event (beserta is_valid) untuk dikunci."""
    parts = {"dev": [], "calib": [], "test": []}
    for e in events:
        if int(e["decision_k"]) != k or int(e["horizon_T"]) != T:
            raise ValueError(
                f"{e['event_id']}: event dideteksi dengan k={e['decision_k']}, T={e['horizon_T']}, "
                f"sedangkan konfigurasi k={k}, T={T}. Jalankan ulang tahap events.")
        expected = common.split_of(e["session_date"])
        if e["split"] != expected:
            raise ValueError(f"{e['event_id']}: split={e['split']} tidak sesuai tanggal ({expected})")
        if e["split"] == "test":
            parts["test"].append(e)
        elif common.parse_bool(e["is_valid"]):
            parts[e["split"]].append(e)
    ids = [e["event_id"] for p in parts.values() for e in p]
    if len(ids) != len(set(ids)):
        raise ValueError("Ada event_id yang muncul lebih dari sekali")
    return parts


def split_rows_by_date(rows, date_col="session_date"):
    before = [r for r in rows if common.parse_date(r[date_col]) <= CALIB_END]
    after = [r for r in rows if common.parse_date(r[date_col]) > CALIB_END]
    return before, after


def file_entry(text):
    return {"sha256": common.sha256_bytes(text.encode()), "baris_data": text.count("\n") - 1}


def _rmtree_force(path):
    for p in [path, *path.rglob("*")]:
        os.chmod(p, stat.S_IRWXU)
    shutil.rmtree(path)


def write_lock(lock_dir, files, info, manifest_name="MANIFEST_KUNCI_UJI.json", readme=""):
    """Tulis berkas uji sekali (hanya-baca) + manifest. Bila sudah terkunci, hanya verifikasi.

    Penulisan pertama dilakukan di folder sementara lalu di-rename, sehingga kegagalan di tengah
    jalan tidak meninggalkan kunci setengah jadi. Berkas dibuat 444 dan foldernya 555."""
    lock_dir = Path(lock_dir)
    manifest_path = lock_dir / manifest_name
    entries = {name: file_entry(text) for name, text in files.items()}

    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        locked = {n: v["sha256"] for n, v in old["berkas"].items()}
        if locked != {n: v["sha256"] for n, v in entries.items()}:
            raise LockViolation("Isi data uji berbeda dari manifest kunci. Jangan menimpa kunci; "
                                "periksa perubahan pada keluaran data engineering.")
        for name, sha in locked.items():
            try:
                actual = common.sha256_file(lock_dir / name)
            except OSError as exc:
                raise LockViolation(f"{name} di {lock_dir} tidak bisa dibaca: {exc}") from exc
            if actual != sha:
                raise LockViolation(f"{name} di {lock_dir} berubah sejak dikunci")
        allowed = set(locked) | {manifest_name, "README.md"}
        extra = sorted(p.name for p in lock_dir.iterdir() if p.name not in allowed)
        if extra:
            raise LockViolation(f"Berkas asing di {lock_dir}: {', '.join(extra)}")
        os.chmod(lock_dir, READ_ONLY_DIR)
        return old

    if lock_dir.exists() and any(lock_dir.iterdir()):
        raise LockViolation(f"{lock_dir} tidak kosong tetapi tidak punya {manifest_name}")
    tmp = lock_dir.with_name(lock_dir.name + ".tmp")
    if tmp.exists():
        _rmtree_force(tmp)                       # sisa run yang gagal; belum pernah menjadi kunci
    tmp.mkdir(parents=True)
    for name, text in files.items():
        (tmp / name).write_text(text)
    (tmp / "README.md").write_text(readme)
    manifest = {
        "dikunci_pada": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "berkas": entries,
        **info,
    }
    (tmp / manifest_name).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    for p in tmp.iterdir():
        os.chmod(p, READ_ONLY)
    if lock_dir.exists():
        lock_dir.rmdir()                         # folder kosong
    tmp.rename(lock_dir)
    os.chmod(lock_dir, READ_ONLY_DIR)
    return manifest


def _check_bar_prefix(bars_dc, bars_test):
    idx = [int(b["bar_idx"]) for b in bars_dc]
    if idx != list(range(len(idx))):
        raise ValueError("bar dev/calib bukan prefiks bar_idx yang kontinu")
    if bars_test and int(bars_test[0]["bar_idx"]) != len(idx):
        raise ValueError("bar uji tidak tepat melanjutkan bar dev/calib")


def _sum(rows, col):
    return round(sum(float(r[col]) for r in rows), 4)


def build(de_output, label, expected_valid=None):
    """Pisahkan keluaran tahap 1 menjadi berkas pemodelan dan berkas uji (belum ditulis)."""
    k, T = label["k"], label["T"]
    src = {name: Path(de_output) / name for name in SOURCES}
    ev_cols, bar_cols, ses_cols = (common.csv_header(src[n]) for n in SOURCES)
    events = common.read_csv(src["gold_pdhl_events.csv"])
    parts = split_events(events, k, T)
    bars_dc, bars_test = split_rows_by_date(common.read_csv(src["silver_xauusd_h1.csv"]))
    _check_bar_prefix(bars_dc, bars_test)
    ses_dc, ses_test = split_rows_by_date(common.read_csv(src["silver_sessions.csv"]))

    if not parts["test"]:
        raise ValueError("Tidak ada event uji; periksa keluaran data engineering")
    test_valid = [e for e in parts["test"] if common.parse_bool(e["is_valid"])]
    n_valid = {"dev": len(parts["dev"]), "calib": len(parts["calib"]), "test": len(test_valid)}
    if sum(n_valid.values()) != sum(common.parse_bool(e["is_valid"]) for e in events):
        raise ValueError("Gabungan event valid ketiga bagian tidak sama dengan total event valid")
    if expected_valid is not None and n_valid != expected_valid:
        raise ValueError(f"Jumlah event valid {n_valid} != acuan {expected_valid}")

    data_files = {"events_dev.csv": common.csv_text(parts["dev"], ev_cols),
                  "events_calib.csv": common.csv_text(parts["calib"], ev_cols),
                  "bars_devcalib.csv": common.csv_text(bars_dc, bar_cols),
                  "sessions_devcalib.csv": common.csv_text(ses_dc, ses_cols)}
    lock_files = {"events_test.csv": common.csv_text(parts["test"], ev_cols),
                  "bars_test.csv": common.csv_text(bars_test, bar_cols),
                  "sessions_test.csv": common.csv_text(ses_test, ses_cols)}
    stats = {
        "n_valid": n_valid, "n_test_all": len(parts["test"]),
        "test_valid_by_level": {lv: sum(e["level_type"] == lv for e in test_valid)
                                for lv in ("PDH", "PDL")},
        "test_valid_ids_sha256": common.sha256_ids(e["event_id"] for e in test_valid),
        "test_dates": [parts["test"][0]["session_date"], parts["test"][-1]["session_date"]],
        "bars_devcalib": len(bars_dc), "bars_test": len(bars_test),
        "sum_close_devcalib": _sum(bars_dc, "close"), "sum_close_test": _sum(bars_test, "close"),
        "bar_idx_test": [int(bars_test[0]["bar_idx"]), int(bars_test[-1]["bar_idx"])],
        "sessions_devcalib": len(ses_dc), "sessions_test": len(ses_test),
        "sessions_test_research": sum(s["split"] == "test" for s in ses_test),
        "src": src,
    }
    return data_files, lock_files, stats


def git_commit():
    """Commit HEAD repo ini dan apakah ada perubahan yang belum di-commit (None bila bukan repo)."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=PACKAGE_DIR, capture_output=True, text=True,
                              check=True, timeout=10).stdout.strip()

    try:
        head = git("rev-parse", "HEAD")
        dirty = git("status", "--porcelain", "--", ".")
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return {"commit": head, "ada_perubahan_belum_commit": bool(dirty)}


def lock_params(cfg):
    return {key: cfg["label"][key] for key in LOCK_PARAMS}


def read_manifest(cfg, paths):
    p = paths.lock / cfg["lock"]["manifest"]
    return json.loads(p.read_text()) if p.exists() else None


def check_lock_params(cfg, manifest):
    """Parameter label konfigurasi harus sama dengan yang tercatat saat data uji dikunci."""
    locked = manifest.get(cfg["lock"]["kunci_key"])
    if locked != lock_params(cfg):
        raise LockViolation(
            f"Parameter label konfigurasi {lock_params(cfg)} berbeda dari yang tercatat saat data "
            f"uji dikunci ({locked}). Perubahan parameter dicatat sebagai adendum dengan nama "
            "konfigurasi baru; run lama tidak ditimpa.")


def check_data_files(manifest, data_dir):
    """Berkas pemodelan di data/ harus sama dengan yang tercatat di manifest kunci."""
    for name, entry in manifest.get("berkas_pemodelan", {}).items():
        p = Path(data_dir) / name
        if not p.exists() or common.sha256_file(p) != entry["sha256"]:
            raise LockViolation(f"{p} berbeda dari saat data uji dikunci; jalankan ulang tahap split")


def manifest_info(cfg, stats, data_files, raw_path):
    return {
        "aturan": cfg["lock"]["aturan"],
        "konfigurasi": cfg["name"],
        cfg["lock"]["kunci_key"]: lock_params(cfg),
        "event_uji": {"semua": stats["n_test_all"], "valid": stats["n_valid"]["test"],
                      "valid_per_level": stats["test_valid_by_level"],
                      "sha256_event_id_valid": stats["test_valid_ids_sha256"],
                      "rentang_session_date": stats["test_dates"]},
        "bar_uji": {"jumlah": stats["bars_test"], "sum_close": stats["sum_close_test"],
                    "bar_idx": stats["bar_idx_test"]},
        "sesi_uji": {"jumlah": stats["sessions_test"],
                     "sesi_periode_penelitian": stats["sessions_test_research"]},
        "berkas_pemodelan": {n: file_entry(t) for n, t in data_files.items()},
        "sumber": {str(Path(raw_path).name): common.sha256_file(raw_path),
                   **{n: common.sha256_file(p) for n, p in stats["src"].items()}},
        "skrip": {f"pdhl_data/{n}": common.sha256_file(PACKAGE_DIR / n) for n in SCRIPTS},
        "git": git_commit(),
    }


def reconciliation(stats):
    return {
        "model_events_dev": stats["n_valid"]["dev"],
        "model_events_calib": stats["n_valid"]["calib"],
        "model_bars_devcalib": stats["bars_devcalib"],
        "sum_close_devcalib": stats["sum_close_devcalib"],
        "locked_test_events_all": stats["n_test_all"],
        "locked_test_events_valid": stats["n_valid"]["test"],
        "locked_test_bars": stats["bars_test"],
        "sum_close_test": stats["sum_close_test"],
        "test_valid_ids_sha256": stats["test_valid_ids_sha256"],
    }


def run_split(cfg, paths, raw_path):
    """Tulis/verifikasi kunci uji dulu; `data/` baru ditulis bila isinya sama dengan saat dikunci."""
    existing = read_manifest(cfg, paths)
    if existing is not None:
        check_lock_params(cfg, existing)
    data_files, lock_files, stats = build(paths.de_output, cfg["label"], cfg["expected_valid"])
    lock = cfg["lock"]
    readme = LOCK_README.format(name=cfg["name"], dibuka_pada=lock["dibuka_pada"],
                                manifest=lock["manifest"])
    manifest = write_lock(paths.lock, lock_files, manifest_info(cfg, stats, data_files, raw_path),
                          manifest_name=lock["manifest"], readme=readme)
    locked_data = {n: v["sha256"] for n, v in manifest.get("berkas_pemodelan", {}).items()}
    if locked_data != {n: file_entry(t)["sha256"] for n, t in data_files.items()}:
        raise LockViolation("Berkas pemodelan berbeda dari saat data uji dikunci; data/ tidak "
                            "ditimpa. Periksa perubahan pada keluaran data engineering.")
    paths.data.mkdir(parents=True, exist_ok=True)
    for name, text in data_files.items():
        (paths.data / name).write_text(text)
        common.load_model_csv(paths.data / name)          # guard: tidak ada baris uji di data/

    recon = reconciliation(stats)
    paths.output.mkdir(parents=True, exist_ok=True)
    (paths.output / "split_rekonsiliasi.json").write_text(json.dumps(recon, indent=2) + "\n")
    print(f"[split] dev={stats['n_valid']['dev']} calib={stats['n_valid']['calib']} "
          f"test={stats['n_valid']['test']} (dikunci {manifest['dikunci_pada']} di {paths.lock})")
    return recon
