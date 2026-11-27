"""Folder masukan/keluaran pemodelan Gradient Boosting dan cek data masukan.

Masukan dibaca dari `runs/<config>/data/` (keluaran pipeline `pdhl_data`). Keluaran ditulis ke
`runs/<config>/model/gradient_boosting/{data,output,model}/`, sehingga folder `data/` pipeline, yang
dicatat di manifest kunci uji, tidak pernah ditulisi kode pemodelan.

Sebelum tahap apa pun dijalankan, SHA-256 data masukan dicocokkan dengan `references/<config>.json`.
Pemeriksaan ini mencegah pemodelan memakai data dari pipeline lama, yaitu sebelum adendum ATR
2026-11-24 (v1.1.0), saat ATR awal 2015 masih kosong.
"""
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pdhl_data
from pdhl_data import common

INPUT_FILES = ("bars_devcalib.csv", "events_dev.csv", "events_calib.csv",
               "labels_dev.csv", "labels_calib.csv")
REPO_ROOT = Path(__file__).resolve().parents[3]


class DataInputBerbeda(RuntimeError):
    """Data masukan tidak sama dengan acuan reproduksi pipeline."""


@dataclass(frozen=True)
class ModelPaths:
    run_root: Path             # runs/<config>

    @property
    def input(self):
        return self.run_root / "data"

    @property
    def root(self):
        return self.run_root / "model" / "gradient_boosting"

    @property
    def data(self):
        return self.root / "data"

    @property
    def output(self):
        return self.root / "output"

    @property
    def model(self):
        return self.root / "model"

    @property
    def gambar(self):
        return self.root / "gambar"


def model_paths(cfg, runs_dir):
    return ModelPaths(Path(runs_dir).resolve() / cfg["name"])


def cek_data_input(paths, reference):
    """SHA-256 data masukan; harus sama dengan `reference["files"]` bila acuan ada.

    Mengembalikan {nama berkas: sha256}. `reference` None berarti tidak ada acuan (hash tetap
    dicatat di rekonsiliasi, tetapi tidak dicocokkan)."""
    hilang = [n for n in INPUT_FILES if not (paths.input / n).exists()]
    if hilang:
        raise FileNotFoundError(f"{paths.input}: belum ada {', '.join(hilang)}. Jalankan pipeline "
                                "data dulu: `python -m pdhl_data run --config configs/<nama>.json`")
    sha = {n: common.sha256_file(paths.input / n) for n in INPUT_FILES}
    if reference is None:
        return sha
    files = reference.get("files", {})
    beda = [n for n in INPUT_FILES if files.get(f"data/{n}") != sha[n]]
    if beda:
        raise DataInputBerbeda(
            f"data masukan berbeda dari acuan reproduksi: {', '.join(beda)}. Jalankan ulang "
            "`python -m pdhl_data run --config configs/<nama>.json` dengan kode pipeline versi "
            f">= 1.1.0 (adendum ATR 2026-11-24). Versi terpasang: {pdhl_data.__version__}.")
    return sha


def git_commit():
    """Commit HEAD dan status perubahan kode pemodelan (None bila bukan repo git)."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True,
                              check=True, timeout=10).stdout.strip()

    try:
        head = git("rev-parse", "HEAD")
        dirty = git("status", "--porcelain", "--", "src", "configs")
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return {"commit": head, "ada_perubahan_belum_commit": bool(dirty)}


def info_sumber(sha):
    """Jejak audit untuk rekonsiliasi: hash masukan, versi pipeline data, dan commit kode."""
    return {"sumber_sha256": dict(sha), "versi_pdhl_data": pdhl_data.__version__,
            "commit": git_commit()}
