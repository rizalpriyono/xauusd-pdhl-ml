"""Parameter terkunci Kunci-1 untuk pemodelan Gradient Boosting.

`configs/kunci_1/` berisi salinan byte-identik berkas Kunci-1 (7 dan 17 November 2026) beserta
`MANIFEST_KUNCI_1.json`, ditambah adendum bertanggal yang masing-masing punya manifest sendiri.
Sebelum tahap apa pun dijalankan, SHA-256 setiap berkas dicocokkan dengan manifest dasar dan dengan
manifest adendum terakhir (`MANIFEST_KUNCI_1_ADENDUM_4.json`, yang juga mencatat hash manifest
sebelumnya), dan a/b/T/k di `configs/<nama>.json` harus sama dengan `kunci_1.json`. Dengan begitu
parameter fitur, baseline, dan walk-forward tidak bisa berubah diam-diam.

Berkas yang dipakai:
    kunci_1.json            parameter label a/b/T/k (a/b juga dipakai nilai harapan R)
    kunci_1_fitur.json      aturan B1 (theta wick). Daftar fitur tambahannya (Adendum 1) sudah
                            digantikan Adendum 4
    kunci_1_adendum_4.json  fitur tambahan 13-15 yang berlaku (Adendum 4, 25 November 2026):
                            f13 wick penolakan, f14 CMF-21 bertanda d, f15 permutation entropy 48
    kunci_1_rq2.json        kelas, clip log loss, Laplace B1, grid/teknis LightGBM, seed,
                            toleransi seri, bin ECE, rentang τ, δ dan grid δ
"""
import json
import math
from dataclasses import dataclass
from pathlib import Path

from pdhl_data import common
from pdhl_data.common import CLASSES

MANIFEST = "MANIFEST_KUNCI_1.json"
MANIFEST_ADENDUM = "MANIFEST_KUNCI_1_ADENDUM_4.json"   # manifest terakhir, s.d. Adendum 4
LABEL_JSON = "kunci_1.json"
FITUR_JSON = "kunci_1_fitur.json"
ADENDUM_FITUR_JSON = "kunci_1_adendum_4.json"
ADENDUM_FITUR_NO = 4
RQ2_JSON = "kunci_1_rq2.json"
LABEL_KEYS = ("a", "b", "T", "k")
FITUR_DASAR = 12          # fitur 1-12 Proposal Final v2 IV.II.5 (Tabel 4.5)
MAKS_TAMBAHAN = 3
NO_FITUR_TAMBAHAN = (13, 14, 15)


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
    a: float                   # barrier target label (kelipatan ATR), dipakai nilai harapan R
    b: float                   # barrier stop label (kelipatan ATR)
    ece_bin: int               # jumlah bin ECE dan diagram reliabilitas
    delta: float               # ambang nilai harapan aturan keputusan (R)
    delta_grid: tuple          # δ untuk kurva cakupan (deskriptif)
    teknis: dict               # bagian `teknis` kunci_1_rq2.json (grid LightGBM, seed, rentang τ)
    fitur_tambahan: tuple      # fitur 13-15 yang berlaku (Adendum 4): nama dan parameter jendela
    manifest_sha256: dict      # nama berkas Kunci-1 dan adendum -> SHA-256 (sudah dicek)


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
    beda = _beda_hash(kunci_dir, expected)
    if beda:
        raise KunciError(f"berkas Kunci-1 berbeda dari {MANIFEST}: {', '.join(beda)}. "
                         "Perubahan Kunci-1 hanya lewat adendum bertanggal dan manifest baru.")
    return dict(expected)


def _beda_hash(kunci_dir, expected):
    """Nama berkas yang tidak ada atau SHA-256-nya berbeda dari `expected`."""
    beda = []
    for name, sha in expected.items():
        p = kunci_dir / name
        if not p.exists():
            beda.append(f"{name} (tidak ada)")
        elif common.sha256_file(p) != sha:
            beda.append(name)
    return beda


def cek_manifest_adendum(kunci_dir):
    """Manifest adendum terakhir harus cocok dengan seluruh berkas dan manifest sebelumnya.

    Manifest ini mencatat ulang kelima berkas dasar, berkas setiap adendum, dan SHA-256 manifest
    sebelumnya. Rantai Kunci-1 -> adendum karena itu tidak bisa diganti sebagian. Mengembalikan
    salinan hash berkasnya."""
    kunci_dir = Path(kunci_dir)
    path = kunci_dir / MANIFEST_ADENDUM
    raw = _baca_json(path)
    expected, sebelumnya = raw.get("sha256"), raw.get("manifest_sebelumnya")
    if not isinstance(expected, dict) or ADENDUM_FITUR_JSON not in expected:
        raise KunciError(f"{path}: {ADENDUM_FITUR_JSON} tidak tercatat di bagian sha256")
    if not isinstance(sebelumnya, dict) or MANIFEST not in sebelumnya:
        raise KunciError(f"{path}: manifest_sebelumnya harus memuat {MANIFEST}")
    beda = _beda_hash(kunci_dir, expected) + _beda_hash(kunci_dir, sebelumnya)
    if beda:
        raise KunciError(f"berkas Kunci-1 berbeda dari {MANIFEST_ADENDUM}: {', '.join(beda)}. "
                         "Perubahan Kunci-1 hanya lewat adendum bertanggal dan manifest baru.")
    return dict(expected)


def load_adendum_fitur(path, theta_dasar):
    """Fitur tambahan 13-15 yang berlaku (Adendum 4 Kunci-1, 25 November 2026).

    Adendum 4 hanya mengganti fitur tambahan; aturan B1 tetap, sehingga theta di adendum harus
    sama dengan `theta_dasar` dari `kunci_1_fitur.json`."""
    raw = _baca_json(path)
    extra, b1 = raw.get("fitur_tambahan"), raw.get("B1")
    theta = b1.get("theta") if isinstance(b1, dict) else None
    checks = {
        f"adendum harus {ADENDUM_FITUR_NO}": raw.get("adendum") == ADENDUM_FITUR_NO,
        "fitur_tambahan harus daftar fitur no. 13, 14, 15 yang bernama":
            isinstance(extra, list) and all(isinstance(e, dict) for e in extra)
            and tuple(e.get("no") for e in extra) == NO_FITUR_TAMBAHAN
            and all(isinstance(e.get("nama"), str) and e["nama"] for e in extra),
        f"jumlah_fitur_akhir harus {FITUR_DASAR + len(NO_FITUR_TAMBAHAN)}":
            raw.get("jumlah_fitur_akhir") == FITUR_DASAR + len(NO_FITUR_TAMBAHAN),
        f"B1.theta harus sama dengan {FITUR_JSON} (aturan B1 tidak diubah Adendum 4)":
            _angka(theta) and theta == theta_dasar,
    }
    bad = [msg for msg, ok in checks.items() if not ok]
    if bad:
        raise KunciError(f"{path}: " + "; ".join(bad))
    return raw


def load_fitur(path):
    """Aturan B1 dan fitur tambahan Adendum 1 (17 November 2026).

    Yang masih berlaku dari berkas ini adalah aturan B1. Daftar fitur tambahannya digantikan
    Adendum 4 (`load_adendum_fitur`)."""
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
    ece_bin, rentang_tau = rq21.get("ece_bin"), teknis.get("temperature_rentang")
    grid = sup.get("delta_grid")
    checks = {
        "kelas harus [Reversal, Continuation, Netral]": tuple(raw.get("kelas", ())) == CLASSES,
        "bootstrap harus 2.000 ulangan klaster week_start":
            boot.get("n") == 2000 and boot.get("klaster") == "week_start",
        "K plasebo harus 1.000": rq22.get("K") == 1000,
        "SS plasebo harus 20 set": rq22.get("ss_jumlah_set") == 20,
        "delta_utama harus 0,1 dan ada di delta_grid":
            sup.get("delta_utama") == 0.1 and isinstance(grid, list) and 0.1 in grid,
        "clip_probabilitas harus di antara 0 dan 1": _angka(clip) and 0 < clip < 1,
        "B1_laplace_bila_sel_kosong harus bilangan bulat >= 0":
            isinstance(laplace, int) and not isinstance(laplace, bool) and laplace >= 0,
        "toleransi_seri_logloss harus >= 0": _angka(tol) and tol >= 0,
        "teknis.lightgbm harus ada": isinstance(teknis.get("lightgbm"), dict),
        "ece_bin harus bilangan bulat >= 1":
            isinstance(ece_bin, int) and not isinstance(ece_bin, bool) and ece_bin >= 1,
        "delta_grid harus daftar angka": isinstance(grid, list) and all(_angka(d) for d in grid),
        "temperature_rentang harus [bawah, atas] dengan 0 < bawah < atas":
            isinstance(rentang_tau, list) and len(rentang_tau) == 2
            and all(_angka(t) for t in rentang_tau) and 0 < rentang_tau[0] < rentang_tau[1],
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
    """Cek kedua manifest, cek label, lalu kembalikan nilai Kunci-1 untuk tahap pemodelan."""
    kunci_dir = Path(kunci_dir)
    sha = {**cek_manifest(kunci_dir), **cek_manifest_adendum(kunci_dir)}
    label = _baca_json(kunci_dir / LABEL_JSON)
    cek_label(cfg["label"], label)
    fitur = load_fitur(kunci_dir / FITUR_JSON)
    adendum = load_adendum_fitur(kunci_dir / ADENDUM_FITUR_JSON, fitur["B1"]["theta"])
    rq2 = load_rq2(kunci_dir / RQ2_JSON)
    rq21, sup = rq2["rq2_1"], rq2["pendukung"]
    return Kunci(theta=float(fitur["B1"]["theta"]),
                 laplace=int(rq21["B1_laplace_bila_sel_kosong"]),
                 clip=float(rq21["clip_probabilitas"]),
                 toleransi_seri=float(rq2["teknis"]["toleransi_seri_logloss"]),
                 k=int(label["k"]), a=float(label["a"]), b=float(label["b"]),
                 ece_bin=int(rq21["ece_bin"]), delta=float(sup["delta_utama"]),
                 delta_grid=tuple(float(d) for d in sup["delta_grid"]),
                 teknis=rq2["teknis"],
                 fitur_tambahan=tuple(dict(e) for e in adendum["fitur_tambahan"]),
                 manifest_sha256=sha)
