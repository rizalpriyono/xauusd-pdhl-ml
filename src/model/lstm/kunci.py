"""Parameter terkunci Kunci-1 dan spesifikasi eksperimen LSTM.

Memverifikasi `configs/kunci_1/` terhadap `MANIFEST_KUNCI_1.json` dan mencocokkan a/b/T/k dengan
`configs/<nama>.json` — identik dengan `model.gradient_boosting.kunci`. Selain nilai Kunci-1
(theta B1, Laplace, clip log loss, toleransi seri, dan SEED_MODEL), modul ini juga menetapkan
konfigurasi tetap grid dan pelatihan LSTM agar eksperimen dapat direproduksi 100%.
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
FITUR_DASAR = 12
MAKS_TAMBAHAN = 3

# Spesifikasi tetap grid dan pelatihan LSTM (turunan SEED_MODEL Kunci-1)
SEED_MODEL = 20261119
N_SEEDS = 5
LOOKBACKS_TERKUNCI = (23, 40, 46, 60, 96)
HIDDEN_TERKUNCI = (16, 32)
LSTM_TEKNIS_TERKUNCI = {
    "num_layers": 1,
    "dropout": 0.3,
    "lr": 1e-3,
    "weight_decay": 1e-2,
    "grad_clip": 1.0,
    "batch_size": 64,
    "max_epochs": 80,
    "patience": 12,
    "early_stop_frac": 0.15,
    "clip_features": 10.0,
    "n_seeds": N_SEEDS,
}


class KunciError(ValueError):
    """Berkas Kunci-1 tidak ada, berubah, atau tidak konsisten."""


@dataclass(frozen=True)
class Kunci:
    """Nilai Kunci-1 dan spesifikasi teknis LSTM yang dipakai tahap pemodelan."""
    theta: float               # aturan B1: f13_wick_penolakan >= theta
    laplace: int               # koreksi Laplace B1 bila satu kelompok kehilangan kelas
    clip: float                # batas bawah peluang untuk log loss
    toleransi_seri: float      # selisih log loss yang dianggap seri saat memilih konfigurasi
    k: int                     # jeda candle keputusan (label)
    seed_model: int            # seed dasar Kunci-1 (20261119)
    seeds: tuple               # 5 seed berurutan: SEED_MODEL .. SEED_MODEL + 4
    lookbacks: tuple           # kandidat panjang jendela L
    hidden_sizes: tuple        # kandidat jumlah unit tersembunyi LSTM
    lstm_teknis: dict          # hiperparameter pelatihan tetap LSTM
    manifest_sha256: dict      # nama berkas Kunci-1 -> SHA-256 (sudah dicek)
    a: float = 2.0             # target posisi reversal (kelipatan ATR, Kunci-1)
    b: float = 1.0             # stop posisi reversal (kelipatan ATR, Kunci-1)
    T: int = 8                 # horizon triple barrier (candle H1, Kunci-1)
    temperature_rentang: tuple = (0.05, 20.0)   # rentang golden-section τ (RM-26)
    ece_bin: int = 10          # jumlah bin ECE & diagram reliabilitas (Kunci-1)
    delta_utama: float = 0.1   # ambang nilai harapan utama RM-27 (satuan R)
    delta_grid: tuple = (0.0, 0.1, 0.2, 0.3)    # grid kurva cakupan-kinerja RM-27


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
    ece_bin = rq21.get("ece_bin")
    tol = teknis.get("toleransi_seri_logloss")
    seed_model = teknis.get("seed_model")
    temp_range = teknis.get("temperature_rentang")
    delta_grid = sup.get("delta_grid")
    checks = {
        "kelas harus [Reversal, Continuation, Netral]": tuple(raw.get("kelas", ())) == CLASSES,
        "bootstrap harus 2.000 ulangan klaster week_start":
            boot.get("n") == 2000 and boot.get("klaster") == "week_start",
        "K plasebo harus 1.000": rq22.get("K") == 1000,
        "SS plasebo harus 20 set": rq22.get("ss_jumlah_set") == 20,
        "delta_utama harus 0,1 dan ada di delta_grid":
            sup.get("delta_utama") == 0.1 and isinstance(delta_grid, list)
            and all(_angka(d) and d >= 0 for d in delta_grid) and 0.1 in delta_grid,
        "clip_probabilitas harus di antara 0 dan 1": _angka(clip) and 0 < clip < 1,
        "B1_laplace_bila_sel_kosong harus bilangan bulat >= 0":
            isinstance(laplace, int) and not isinstance(laplace, bool) and laplace >= 0,
        "ece_bin harus bilangan bulat >= 2":
            isinstance(ece_bin, int) and not isinstance(ece_bin, bool) and ece_bin >= 2,
        "toleransi_seri_logloss harus >= 0": _angka(tol) and tol >= 0,
        "temperature_rentang harus [lo, hi] dengan 0 < lo < hi":
            isinstance(temp_range, list) and len(temp_range) == 2
            and all(_angka(v) for v in temp_range) and 0 < temp_range[0] < temp_range[1],
        f"seed_model harus {SEED_MODEL}":
            isinstance(seed_model, int) and not isinstance(seed_model, bool) and seed_model == SEED_MODEL,
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
    """Cek manifest, cek label, lalu kembalikan nilai Kunci-1 + spesifikasi LSTM."""
    kunci_dir = Path(kunci_dir)
    sha = cek_manifest(kunci_dir)
    label = _baca_json(kunci_dir / LABEL_JSON)
    cek_label(cfg["label"], label)
    fitur = load_fitur(kunci_dir / FITUR_JSON)
    rq2 = load_rq2(kunci_dir / RQ2_JSON)
    rq21 = rq2["rq2_1"]
    sup = rq2["pendukung"]
    seed_dasar = int(rq2["teknis"]["seed_model"])
    t_lo, t_hi = rq2["teknis"]["temperature_rentang"]
    return Kunci(
        theta=float(fitur["B1"]["theta"]),
        laplace=int(rq21["B1_laplace_bila_sel_kosong"]),
        clip=float(rq21["clip_probabilitas"]),
        toleransi_seri=float(rq2["teknis"]["toleransi_seri_logloss"]),
        k=int(label["k"]),
        seed_model=seed_dasar,
        seeds=tuple(seed_dasar + i for i in range(N_SEEDS)),
        lookbacks=LOOKBACKS_TERKUNCI,
        hidden_sizes=HIDDEN_TERKUNCI,
        lstm_teknis=dict(LSTM_TEKNIS_TERKUNCI),
        manifest_sha256=sha,
        a=float(label["a"]),
        b=float(label["b"]),
        T=int(label["T"]),
        temperature_rentang=(float(t_lo), float(t_hi)),
        ece_bin=int(rq21["ece_bin"]),
        delta_utama=float(sup["delta_utama"]),
        delta_grid=tuple(float(d) for d in sup["delta_grid"]),
    )
