"""Parameter terkunci Kunci-1 untuk pemodelan Gradient Boosting.

`configs/kunci_1/` berisi salinan byte-identik berkas Kunci-1 (7 dan 17 November 2026) beserta
`MANIFEST_KUNCI_1.json`. Sebelum tahap apa pun dijalankan, SHA-256 setiap berkas dicocokkan dengan
manifest, dan a/b/T/k di `configs/<nama>.json` harus sama dengan `kunci_1.json`. Dengan begitu
parameter fitur, baseline, dan walk-forward tidak bisa berubah diam-diam.

Berkas yang dipakai:
    kunci_1_fitur.json  fitur tambahan 13-15 dan aturan B1 (theta wick)
    kunci_1_rq2.json    kelas, clip log loss, Laplace B1, grid/teknis LightGBM, seed, toleransi seri
"""
import json
import math
from dataclasses import dataclass
from pathlib import Path

from pdhl_data import common
from pdhl_data.common import CLASSES

MANIFEST = "MANIFEST_KUNCI_1.json"
LABEL_JSON = "kunci_1.json"
FITUR_JSON = "kunci_1_fitur.json"
RQ2_JSON = "kunci_1_rq2.json"
LABEL_KEYS = ("a", "b", "T", "k")
FITUR_DASAR = 12          # fitur 1-12 Proposal Final v2 IV.II.5 (Tabel 4.5)
MAKS_TAMBAHAN = 3


class KunciError(ValueError):
    """Berkas Kunci-1 tidak ada, berubah, atau tidak konsisten."""


@dataclass(frozen=True)
class Kunci:
    """Nilai Kunci-1 yang dipakai tahap pemodelan."""
    theta: float               # aturan B1: f13_wick_penolakan >= theta
    laplace: int               # koreksi Laplace B1 bila satu kelompok kehilangan kelas
    clip: float                # batas bawah peluang untuk log loss
    toleransi_seri: float      # selisih log loss yang dianggap seri saat memilih konfigurasi
    k: int                     # jeda candle keputusan (label)
    teknis: dict               # bagian `teknis` kunci_1_rq2.json (grid LightGBM, seed)
    manifest_sha256: dict      # nama berkas Kunci-1 -> SHA-256 (sudah dicek)


def _baca_json(path):
    path = Path(path)
    if not path.exists():
        raise KunciError(f"{path}: berkas Kunci-1 tidak ada")
    with open(path) as f:
        return json.load(f)


def _angka(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def kunci_dir_bawaan(config_path):
    """Folder Kunci-1 bawaan: `kunci_1/` di samping berkas konfigurasi."""
    return Path(config_path).resolve().parent / "kunci_1"


def cek_manifest(kunci_dir):
    """SHA-256 setiap berkas Kunci-1 harus sama dengan manifest; mengembalikan salinan hash."""
    kunci_dir = Path(kunci_dir)
    expected = _baca_json(kunci_dir / MANIFEST).get("sha256")
    if not isinstance(expected, dict) or not expected:
        raise KunciError(f"{kunci_dir / MANIFEST}: bagian sha256 kosong")
    tanpa_hash = [n for n in (LABEL_JSON, FITUR_JSON, RQ2_JSON) if n not in expected]
    if tanpa_hash:
        raise KunciError(f"{kunci_dir / MANIFEST}: berkas yang dimuat tidak tercatat di manifest: "
                         f"{', '.join(tanpa_hash)}")
    beda = []
    for name, sha in expected.items():
        p = kunci_dir / name
        if not p.exists():
            beda.append(f"{name} (tidak ada)")
        elif common.sha256_file(p) != sha:
            beda.append(name)
    if beda:
        raise KunciError(f"berkas Kunci-1 berbeda dari {MANIFEST}: {', '.join(beda)}. "
                         "Perubahan Kunci-1 hanya lewat adendum bertanggal dan manifest baru.")
    return dict(expected)


def load_fitur(path):
    """Fitur tambahan dan aturan B1 (Adendum 1 Kunci-1, 17 November 2026)."""
    raw = _baca_json(path)
    theta = raw.get("B1", {}).get("theta")
    if not _angka(theta) or not 0 < theta < 1:
        raise KunciError(f"{path}: B1.theta harus angka di antara 0 dan 1")
    extra = raw.get("fitur_tambahan", [])
    if len(extra) > MAKS_TAMBAHAN or raw.get("jumlah_fitur_akhir") != FITUR_DASAR + len(extra):
        raise KunciError(f"{path}: jumlah_fitur_akhir harus {FITUR_DASAR} + jumlah fitur tambahan "
                         f"(maks. {MAKS_TAMBAHAN})")
    return raw


def load_rq2(path):
    """Rencana analisis RQ2 (Adendum 2 Kunci-1). Nilai inti harus sama dengan Proposal Final v2."""
    raw = _baca_json(path)
    rq21, rq22, sup = raw.get("rq2_1", {}), raw.get("rq2_2", {}), raw.get("pendukung", {})
    teknis = raw.get("teknis", {})
    boot = rq21.get("bootstrap", {})
    clip, laplace = rq21.get("clip_probabilitas"), rq21.get("B1_laplace_bila_sel_kosong")
    tol = teknis.get("toleransi_seri_logloss")
    checks = {
        "kelas harus [Reversal, Continuation, Netral]": tuple(raw.get("kelas", ())) == CLASSES,
        "bootstrap harus 2.000 ulangan klaster week_start":
            boot.get("n") == 2000 and boot.get("klaster") == "week_start",
        "K plasebo harus 1.000": rq22.get("K") == 1000,
        "SS plasebo harus 20 set": rq22.get("ss_jumlah_set") == 20,
        "delta_utama harus 0,1 dan ada di delta_grid":
            sup.get("delta_utama") == 0.1 and 0.1 in sup.get("delta_grid", []),
        "clip_probabilitas harus di antara 0 dan 1": _angka(clip) and 0 < clip < 1,
        "B1_laplace_bila_sel_kosong harus bilangan bulat >= 0":
            isinstance(laplace, int) and not isinstance(laplace, bool) and laplace >= 0,
        "toleransi_seri_logloss harus >= 0": _angka(tol) and tol >= 0,
        "teknis.lightgbm harus ada": isinstance(teknis.get("lightgbm"), dict),
    }
    bad = [msg for msg, ok in checks.items() if not ok]
    if bad:
        raise KunciError(f"{path}: " + "; ".join(bad))
    return raw


def cek_label(cfg_label, kunci_label):
    """a/b/T/k konfigurasi pipeline harus sama dengan kunci_1.json."""
    beda = [key for key in LABEL_KEYS
            if not (_angka(kunci_label.get(key)) and float(cfg_label[key]) == float(kunci_label[key]))]
    if beda:
        raise KunciError(f"parameter label konfigurasi berbeda dari {LABEL_JSON}: {', '.join(beda)}")


def muat_kunci(cfg, kunci_dir):
    """Cek manifest, cek label, lalu kembalikan nilai Kunci-1 untuk tahap pemodelan."""
    kunci_dir = Path(kunci_dir)
    sha = cek_manifest(kunci_dir)
    label = _baca_json(kunci_dir / LABEL_JSON)
    cek_label(cfg["label"], label)
    fitur = load_fitur(kunci_dir / FITUR_JSON)
    rq2 = load_rq2(kunci_dir / RQ2_JSON)
    rq21 = rq2["rq2_1"]
    return Kunci(theta=float(fitur["B1"]["theta"]),
                 laplace=int(rq21["B1_laplace_bila_sel_kosong"]),
                 clip=float(rq21["clip_probabilitas"]),
                 toleransi_seri=float(rq2["teknis"]["toleransi_seri_logloss"]),
                 k=int(label["k"]), teknis=rq2["teknis"], manifest_sha256=sha)
