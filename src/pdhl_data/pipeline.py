"""Orkestrasi pipeline: data mentah -> event -> split + kunci uji -> label -> laporan -> verifikasi.

Struktur keluaran satu run (`runs/<config>/`):

    de_output/   silver_xauusd_h1.csv, silver_sessions.csv, gold_pdhl_events.csv,
                 dq_dst_verification.csv, de_rekonsiliasi.json, laporan_data_engineering.md
    data/        events_{dev,calib}.csv, bars_devcalib.csv, sessions_devcalib.csv,
                 labels_{dev,calib}.csv            <- satu-satunya folder untuk pemodelan
    kunci_uji/   events_test.csv, bars_test.csv, sessions_test.csv, MANIFEST (hanya-baca)
    output/      split_rekonsiliasi.json, label_rekonsiliasi.json, label_purged.csv,
                 laporan_label.md
"""
import json
from dataclasses import dataclass
from pathlib import Path

from . import common
from .events import run_events
from .labels import run_labels
from .labels_report import write_report as write_label_report
from .split import run_split

STEPS = ("events", "split", "labels")
RECON_FILES = {"events": "de_output/de_rekonsiliasi.json",
               "split": "output/split_rekonsiliasi.json",
               "labels": "output/label_rekonsiliasi.json"}


@dataclass(frozen=True)
class RunPaths:
    root: Path

    @property
    def de_output(self):
        return self.root / "de_output"

    @property
    def data(self):
        return self.root / "data"

    @property
    def lock(self):
        return self.root / "kunci_uji"

    @property
    def output(self):
        return self.root / "output"


def run_paths(cfg, runs_dir):
    return RunPaths(Path(runs_dir).resolve() / cfg["name"])


def load_reference(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def check_raw(raw_path, reference):
    """Kembalikan pesan peringatan bila data mentah berbeda dari acuan (None bila cocok)."""
    if not Path(raw_path).exists():
        raise FileNotFoundError(f"Data mentah tidak ditemukan: {raw_path} (lihat data/raw/README.md)")
    if reference is None:
        return None
    expected = reference["raw"]["sha256"]
    actual = common.sha256_file(raw_path)
    if actual != expected:
        return (f"SHA-256 {Path(raw_path).name} = {actual[:12]}…, acuan {expected[:12]}…. "
                "Hasil tidak akan sama dengan acuan.")
    return None


def run(cfg, raw_path, runs_dir, steps=STEPS, report=True):
    """Jalankan tahap yang diminta secara berurutan; mengembalikan RunPaths."""
    unknown = set(steps) - set(STEPS)
    if unknown:
        raise ValueError(f"tahap tidak dikenal: {sorted(unknown)}; pilihan: {', '.join(STEPS)}")
    paths = run_paths(cfg, runs_dir)
    paths.output.mkdir(parents=True, exist_ok=True)
    label = cfg["label"]
    if "events" in steps:
        run_events(Path(raw_path), paths.de_output, label["k"], label["T"], write_report=report)
    if "split" in steps:
        run_split(cfg, paths, Path(raw_path))
    if "labels" in steps:
        run_labels(cfg, paths)
        if report:
            write_label_report(cfg, paths)
    return paths


def verify(paths, reference):
    """Bandingkan SHA-256 berkas keluaran dan angka rekonsiliasi dengan acuan.

    Mengembalikan daftar (nama, status, keterangan); status OK, BEDA, atau HILANG."""
    results = []
    for rel, expected in reference["files"].items():
        p = paths.root / rel
        if not p.exists():
            results.append((rel, "HILANG", "berkas belum dibuat"))
            continue
        actual = common.sha256_file(p)
        status = "OK" if actual == expected else "BEDA"
        results.append((rel, status, actual[:12]))
    for stage, expected in reference.get("rekonsiliasi", {}).items():
        p = paths.root / RECON_FILES[stage]
        if not p.exists():
            results.append((f"rekonsiliasi.{stage}", "HILANG", str(p)))
            continue
        actual = json.loads(p.read_text())
        diff = sorted(key for key in expected.keys() | actual.keys()
                      if expected.get(key) != actual.get(key))
        results.append((f"rekonsiliasi.{stage}", "BEDA" if diff else "OK",
                        ", ".join(diff[:5]) if diff else f"{len(expected)} angka"))
    return results
