"""Bandingkan rekonsiliasi baru dengan hasil lama (dihitung sebelum adendum ATR 2026-11-24).

Hasil lama disimpan di `references/gradient_boosting_lama/` (salinan `Pemodelan PDH-PDL/output/*.json`,
22 November 2026). Setiap angka di rekonsiliasi diratakan menjadi `jalur.kunci` -> nilai, lalu diberi
status:

    IDENTIK     sama persis
    BEDA_KECIL  |selisih| <= 1e-9 x max(1, |nilai|); yang diharapkan dari adendum ATR (atr14_d_prev
                event dev 2016 berubah <= 4,6e-10 relatif, sehingga f02/f12 ikut bergeser sangat kecil)
    BEDA        perlu ditinjau
    HANYA_LAMA / HANYA_BARU   kunci hanya ada di salah satu sisi

Kunci jejak audit (hash sumber, lingkungan, versi, commit) memang pasti berbeda, jadi diabaikan.
Modul ini hanya menulis laporan; ia tidak menghentikan proses.

Sesudah Kunci-1 Adendum 4 (25 November 2026) fitur 14-15 diganti, sedangkan acuan lama dihitung
dengan fitur 14-15 lama. Karena itu yang masih diharapkan IDENTIK atau BEDA_KECIL hanyalah fitur
f01-f13 dan seluruh baseline B0/B1. Angka f14/f15 lama muncul sebagai HANYA_LAMA dan yang baru
sebagai HANYA_BARU, sedangkan walk-forward dan model akhir memang berbeda (BEDA).
"""
import json
import math
from collections import Counter
from pathlib import Path

from pdhl_data import common

BERKAS = ("fitur_rekonsiliasi.json", "pembanding_rekonsiliasi.json",
          "walk_forward_rekonsiliasi.json", "model_akhir_rekonsiliasi.json")
ASAL = "asal.json"
DIABAIKAN = frozenset({"sumber_sha256", "sumber_pipeline_sha256", "lingkungan",
                       "versi_pdhl_data", "commit"})
TOL = 1e-9
STATUS = ("IDENTIK", "BEDA_KECIL", "BEDA", "HANYA_LAMA", "HANYA_BARU")
MAKS_BARIS = 100
_KOSONG = object()


class AcuanLamaRusak(ValueError):
    """Berkas acuan lama tidak sama dengan SHA-256 yang tercatat di `asal.json`."""


def ratakan(obj, awalan=""):
    """Dict/list bersarang -> {jalur: nilai daun}; kunci di DIABAIKAN dilewati."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k not in DIABAIKAN:
                out.update(ratakan(v, f"{awalan}.{k}" if awalan else str(k)))
        return out
    if isinstance(obj, list):
        out = {}
        for i, v in enumerate(obj):
            out.update(ratakan(v, f"{awalan}[{i}]"))
        return out
    return {awalan: obj}


def _angka(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def status(lama, baru, tol=TOL):
    if lama is _KOSONG:
        return "HANYA_BARU"
    if baru is _KOSONG:
        return "HANYA_LAMA"
    if type(lama) is type(baru) and lama == baru:
        return "IDENTIK"
    if _angka(lama) and _angka(baru):
        if lama == baru or (math.isnan(lama) and math.isnan(baru)):
            return "IDENTIK"
        if abs(lama - baru) <= tol * max(1.0, abs(lama), abs(baru)):
            return "BEDA_KECIL"
    return "BEDA"


def bandingkan(lama, baru, tol=TOL):
    """Baris perbandingan untuk satu berkas rekonsiliasi (urutan kunci lama, lalu kunci baru)."""
    fl, fb = ratakan(lama), ratakan(baru)
    kunci = list(fl) + [k for k in fb if k not in fl]
    rows = []
    for k in kunci:
        a, b = fl.get(k, _KOSONG), fb.get(k, _KOSONG)
        s = status(a, b, tol)
        selisih = b - a if s in ("BEDA_KECIL", "BEDA") and _angka(a) and _angka(b) else None
        rows.append({"kunci": k, "lama": None if a is _KOSONG else a,
                     "baru": None if b is _KOSONG else b, "selisih": selisih, "status": s})
    return rows


def _sel(v):
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:.10g}"
    return str(v).replace("|", "\\|")


def laporan(hasil, asal):
    lines = ["# Perbandingan dengan Hasil Lama (Gradient Boosting)", "",
             f"Acuan lama: {asal.get('keterangan', '–')}", "",
             f"Status: IDENTIK = sama persis; BEDA_KECIL = |selisih| ≤ {TOL:g} × max(1, |nilai|), "
             "efek yang diharapkan dari adendum ATR; BEDA = perlu ditinjau. Hash sumber, "
             "lingkungan, versi, dan commit diabaikan.", "",
             "Catatan Adendum 4: acuan lama dihitung dengan fitur 14-15 lama. Yang masih diharapkan "
             "IDENTIK atau BEDA_KECIL hanyalah fitur f01-f13 dan seluruh baseline B0/B1. Angka "
             "f14/f15, walk-forward, dan model akhir memang berbeda dari acuan lama.", "",
             "## Ringkasan", "",
             "| Berkas | " + " | ".join(STATUS) + " |", "|---|" + "---|" * len(STATUS)]
    for nama, rows in hasil.items():
        if rows is None:
            lines.append(f"| {nama} | belum dijalankan |" + " |" * (len(STATUS) - 1))
            continue
        c = Counter(r["status"] for r in rows)
        lines.append(f"| {nama} | " + " | ".join(str(c[s]) for s in STATUS) + " |")
    for nama, rows in hasil.items():
        beda = [r for r in rows or [] if r["status"] != "IDENTIK"]
        if not beda:
            continue
        lines += ["", f"## {nama}: {len(beda)} angka tidak identik", "",
                  "| Kunci | Lama | Baru | Selisih | Status |", "|---|---|---|---|---|"]
        lines += [f"| `{r['kunci']}` | {_sel(r['lama'])} | {_sel(r['baru'])} | "
                  f"{_sel(r['selisih'])} | {r['status']} |" for r in beda[:MAKS_BARIS]]
        if len(beda) > MAKS_BARIS:
            lines.append(f"\n… {len(beda) - MAKS_BARIS} baris lain tidak ditampilkan.")
    return "\n".join(lines) + "\n"


def cek_acuan(acuan_dir, asal):
    """SHA-256 setiap berkas acuan lama harus sama dengan `asal.json`."""
    sha = asal.get("sha256", {})
    beda = [n for n in BERKAS
            if not (acuan_dir / n).exists() or common.sha256_file(acuan_dir / n) != sha.get(n)]
    if beda:
        raise AcuanLamaRusak(f"{acuan_dir}: berkas acuan lama berbeda dari {ASAL}: "
                             f"{', '.join(beda)}")


def run_bandingkan(paths, acuan_dir, cfg_name):
    """Tulis `bandingkan_lama.md`; mengembalikan {berkas: Counter status} atau None bila dilewati.

    Dilewati bila acuan tidak ada atau dibuat untuk konfigurasi lain."""
    acuan_dir = Path(acuan_dir)
    if not (acuan_dir / ASAL).exists():
        return None
    asal = json.loads((acuan_dir / ASAL).read_text())
    if asal.get("config") != cfg_name:
        return None
    cek_acuan(acuan_dir, asal)
    hasil = {}
    for nama in BERKAS:
        baru = paths.output / nama
        hasil[nama] = (bandingkan(json.loads((acuan_dir / nama).read_text()),
                                  json.loads(baru.read_text())) if baru.exists() else None)
    paths.output.mkdir(parents=True, exist_ok=True)
    (paths.output / "bandingkan_lama.md").write_text(laporan(hasil, asal))
    return {n: (Counter(r["status"] for r in rows) if rows is not None else None)
            for n, rows in hasil.items()}


def ringkas(ringkasan):
    if ringkasan is None:
        return "Bandingkan: acuan lama tidak ada atau untuk konfigurasi lain, dilewati"
    lines = ["Bandingkan dengan hasil lama:"]
    for nama, c in ringkasan.items():
        lines.append(f"  {nama}: " + ("belum dijalankan" if c is None else
                                      ", ".join(f"{s} {c[s]}" for s in STATUS if c[s])))
    return "\n".join(lines)
