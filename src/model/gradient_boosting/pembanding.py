"""Tahap 2 (RM-23): baseline non-model B0 dan B1.

Port setia dari `Pemodelan PDH-PDL/pembanding.py`. B0 = frekuensi kelas pada data latih. B1 = frekuensi
kelas pada data latih, dipisah menurut aturan heuristik wick (f13 >= 2/3, `kunci_1_fitur.json`);
kelompok yang tidak memiliki salah satu kelas diberi koreksi Laplace (+1 per kelas,
`kunci_1_rq2.json`). Keduanya dihitung:

1. sekali pada seluruh data pengembangan 2016-2023 (pembanding akhir untuk Kunci-2), dan
2. ulang pada data latih setiap lipatan walk-forward (validasi 2019-2023), dengan purging event
   yang jendela labelnya melewati batas lipatan, sehingga tersedia prediksi out-of-fold (OOF)
   B0/B1 yang sebanding dengan prediksi OOF model.

B2 (regresi logistik) tidak dihitung: dihapus atas keputusan 22 Nov 2026, menunggu Adendum 3.
Hanya data dev yang dimuat (lewat `load_model_csv`); data kalibrasi dan uji tidak dipakai.
"""
import datetime as dt
import json
import math
from collections import Counter

from pdhl_data import common
from pdhl_data.common import CLASSES, DEV_END

from .fitur import b1_terpenuhi

TAHUN_VALIDASI = (2019, 2020, 2021, 2022, 2023)
SINGKAT = {"Reversal": "rev", "Continuation": "cont", "Netral": "net"}
OOF_COLS = (["event_id", "session_date", "tahun_validasi", "label", "b1_terpenuhi"]
            + [f"{m}_p_{SINGKAT[k]}" for m in ("b0", "b1") for k in CLASSES]
            + ["ll_b0", "ll_b1"])


# ---------------------------------------------------------------- data
def gabung(fitur_rows, label_rows, theta):
    """Satukan fitur dan label per event; tolak ketidaksejajaran dan flag B1 yang tidak konsisten."""
    if len(fitur_rows) != len(label_rows):
        raise ValueError(f"jumlah baris fitur ({len(fitur_rows)}) != label ({len(label_rows)})")
    out, seen = [], set()
    for f, lab in zip(fitur_rows, label_rows):
        eid = f["event_id"]
        if eid != lab["event_id"] or f["session_date"] != lab["session_date"]:
            raise ValueError(f"event tidak sejajar: fitur {eid} vs label {lab['event_id']}")
        if eid in seen:
            raise ValueError(f"{eid}: event_id ganda")
        seen.add(eid)
        if lab["label"] not in CLASSES:
            raise ValueError(f"{eid}: label tidak dikenal {lab['label']!r}")
        if lab["label_end_session_date"] < lab["session_date"]:
            raise ValueError(f"{eid}: label_end_session_date sebelum session_date")
        f13 = float(f["f13_wick_penolakan"])
        if not math.isfinite(f13):
            raise ValueError(f"{eid}: f13 tidak berhingga ({f13})")
        flag = common.parse_bool(f["b1_terpenuhi"])
        if flag != b1_terpenuhi(f13, theta):
            raise ValueError(f"{f['event_id']}: b1_terpenuhi={flag} tidak sesuai f13={f13} "
                             f"dan theta={theta}")
        out.append({"event_id": f["event_id"], "session_date": f["session_date"],
                    "label": lab["label"], "label_end_session_date": lab["label_end_session_date"],
                    "f13": f13, "b1": flag})
    return out


def muat_data(fitur_path, label_path, theta):
    """Muat fitur dan label dev lewat loader ber-guard, lalu gabungkan."""
    return gabung(common.load_model_csv(fitur_path), common.load_model_csv(label_path), theta)


def data_latih(rows, batas):
    """Event sebelum `batas`; event yang jendela labelnya mencapai `batas` di-purge."""
    latih, purged = [], []
    for r in rows:
        if common.parse_date(r["session_date"]) >= batas:
            continue
        if common.parse_date(r["label_end_session_date"]) >= batas:
            purged.append(r)
        else:
            latih.append(r)
    return latih, purged


# ---------------------------------------------------------------- pembanding
def hitung_kelas(rows):
    c = Counter(r["label"] for r in rows)
    return {k: c.get(k, 0) for k in CLASSES}


def _ringkas(rows, laplace):
    """Jumlah asli, flag Laplace, dan peluang satu sel."""
    jumlah = hitung_kelas(rows)
    pakai = laplace if min(jumlah.values()) == 0 else 0
    total = sum(jumlah.values()) + pakai * len(CLASSES)
    return {"n": len(rows), "jumlah": jumlah, "laplace": bool(pakai),
            "peluang": {k: (jumlah[k] + pakai) / total for k in CLASSES}}


def latih_b0(rows):
    if not rows:
        raise ValueError("B0 butuh sedikitnya satu event latih")
    s = _ringkas(rows, laplace=0)
    return {"n": s["n"], "jumlah": s["jumlah"], "peluang": s["peluang"]}


def latih_b1(rows, laplace):
    return {"terpenuhi": _ringkas([r for r in rows if r["b1"]], laplace),
            "tidak_terpenuhi": _ringkas([r for r in rows if not r["b1"]], laplace)}


def latih_pembanding(rows, laplace):
    return {"B0": latih_b0(rows), "B1": latih_b1(rows, laplace)}


def prediksi(model, row):
    """Peluang [Reversal, Continuation, Netral] dari B0 dan B1 untuk satu event."""
    sel = model["B1"]["terpenuhi" if row["b1"] else "tidak_terpenuhi"]
    return {"B0": [model["B0"]["peluang"][k] for k in CLASSES],
            "B1": [sel["peluang"][k] for k in CLASSES]}


def log_loss(peluang, label, clip):
    """−ln p(label); peluang dipotong ke [clip, 1] lalu dinormalkan ulang (Kunci-1 Adendum 2)."""
    if label not in CLASSES:
        raise ValueError(f"label tidak dikenal {label!r}")
    p = [min(max(x, clip), 1.0) for x in peluang]
    return -math.log(p[CLASSES.index(label)] / sum(p))


def _rerata(xs):
    return sum(xs) / len(xs) if xs else None


# ---------------------------------------------------------------- skema
def model_akhir(rows, laplace):
    """Pembanding akhir dari seluruh dev (purging di batas dev-kalibrasi)."""
    latih, purged = data_latih(rows, DEV_END + dt.timedelta(days=1))
    return {"n_latih": len(latih), "n_purged": len(purged), **latih_pembanding(latih, laplace)}


def walk_forward(rows, tahun_validasi, laplace, clip):
    """Prediksi OOF B0/B1: latih pada 2016..y−1 (dengan purging), validasi pada tahun y."""
    oof, lipatan = [], []
    for y in tahun_validasi:
        latih, purged = data_latih(rows, dt.date(y, 1, 1))
        if not latih:
            raise ValueError(f"lipatan {y}: tidak ada data latih")
        model = latih_pembanding(latih, laplace)
        val = [r for r in rows if int(r["session_date"][:4]) == y]
        ll = {"B0": [], "B1": []}
        for r in val:
            p = prediksi(model, r)
            out = {"event_id": r["event_id"], "session_date": r["session_date"],
                   "tahun_validasi": y, "label": r["label"], "b1_terpenuhi": r["b1"]}
            for m in ("B0", "B1"):
                for k, v in zip(CLASSES, p[m]):
                    out[f"{m.lower()}_p_{SINGKAT[k]}"] = v
                out[f"ll_{m.lower()}"] = log_loss(p[m], r["label"], clip)
                ll[m].append(out[f"ll_{m.lower()}"])
            oof.append(out)
        lipatan.append({"tahun_validasi": y, "n_latih": len(latih), "n_purged": len(purged),
                        "n_validasi": len(val), **model,
                        "log_loss_b0": _rerata(ll["B0"]), "log_loss_b1": _rerata(ll["B1"])})
    return oof, lipatan


# ---------------------------------------------------------------- laporan
def _p(sel):
    return " / ".join(f"{sel['peluang'][k]:.4f}" for k in CLASSES)


def _selisih(x):
    """Selisih bertanda 4 desimal tanpa '-0.0000'."""
    x = round(x, 4)
    return f"{x + 0.0:+.4f}" if x else "0.0000"


def _baris_ll(nama, n, ll0, ll1):
    """Satu baris tabel log loss; tahun tanpa event validasi ditulis '–'."""
    if ll0 is None or ll1 is None:
        return f"| {nama} | {n} | – | – | – |"
    return f"| {nama} | {n} | {ll0:.4f} | {ll1:.4f} | {_selisih(ll1 - ll0)} |"


def _tanda_laplace(sel):
    return " ᴸ" if sel["laplace"] else ""


def _baris_akhir(nama, sel):
    j = sel["jumlah"]
    return (f"| {nama} | {sel['n']} | " + " | ".join(f"{sel['peluang'][k]:.4f} ({j[k]})"
                                                   for k in CLASSES)
            + f" | {'ya' if sel.get('laplace') else 'tidak'} |")


def laporan(recon):
    a, wf, o = recon["akhir"], recon["walk_forward"], recon["oof"]
    b1 = a["B1"]
    lines = [
        "# Laporan Pembanding Non-Model B0 dan B1 (RM-23)", "",
        "Sumber: `model.gradient_boosting.pembanding`, data pengembangan saja (`features_dev.csv`, "
        "`labels_dev.csv`). "
        f"Aturan B1: f13 ≥ {recon['theta_b1']}. Laplace +{recon['laplace']} per kelas hanya bila "
        "satu kelompok tidak memiliki salah satu kelas. Data kalibrasi dan uji tidak dimuat.",
        "", f"Catatan: {recon['catatan']}", "",
        "## 1. Pembanding akhir (seluruh dev 2016–2023)", "",
        f"Event latih {a['n_latih']}, di-purge {a['n_purged']}. Isi sel: peluang (jumlah event).", "",
        "| Pembanding | n | Reversal | Continuation | Netral | Laplace |", "|---|---|---|---|---|---|",
        _baris_akhir("B0 · semua event", {**a["B0"], "laplace": False}),
        _baris_akhir("B1 · wick ≥ 2/3", b1["terpenuhi"]),
        _baris_akhir("B1 · wick < 2/3", b1["tidak_terpenuhi"]), "",
        "## 2. Pembanding per lipatan walk-forward", "",
        "Peluang ditulis Reversal / Continuation / Netral, diestimasi dari data latih lipatan.", "",
        "| Validasi | n latih | purge | n validasi | B0 | B1 wick ≥ 2/3 (n) | B1 wick < 2/3 (n) |",
        "|---|---|---|---|---|---|---|",
    ]
    for f in wf:
        ya, tidak = f["B1"]["terpenuhi"], f["B1"]["tidak_terpenuhi"]
        lines.append(f"| {f['tahun_validasi']} | {f['n_latih']} | {f['n_purged']} | "
                     f"{f['n_validasi']} | {_p(f['B0'])} | "
                     f"{_p(ya)} ({ya['n']}){_tanda_laplace(ya)} | "
                     f"{_p(tidak)} ({tidak['n']}){_tanda_laplace(tidak)} |")
    lines += ["", "ᴸ = koreksi Laplace dipakai.", "",
              "## 3. Log loss OOF pembanding (deskriptif)", "",
              "Rata-rata log loss per event pada tahun validasi. Ini acuan untuk prediksi OOF "
              "model nanti, bukan uji hipotesis; kecil = lebih baik.", "",
              "| Validasi | n | B0 | B1 | B1 − B0 |", "|---|---|---|---|---|"]
    for f in wf:
        lines.append(_baris_ll(f["tahun_validasi"], f["n_validasi"],
                               f["log_loss_b0"], f["log_loss_b1"]))
    lines += [_baris_ll("**Gabungan**", o["n"], o["log_loss_b0"], o["log_loss_b1"]), "",
              f"Purging: {a['n_purged']} event pada batas dev–kalibrasi dan "
              f"{sum(f['n_purged'] for f in wf)} event pada batas lipatan. Angka 0 adalah sifat "
              "data: tidak ada event dev yang jendela labelnya melintasi pergantian tahun "
              "(`label_crosses_year` = 0 pada rekonsiliasi label). Aturan purging tetap dijalankan "
              "dan jalurnya diuji dengan data sintetis.",
              ""]
    return "\n".join(lines)


# ---------------------------------------------------------------- tahap
CATATAN = ("B2 (regresi logistik) tidak dihitung: dihapus atas keputusan 22 Nov 2026, "
           "menunggu Adendum 3 Kunci-1.")


def run_pembanding(paths, kunci, sumber, tahun_validasi=TAHUN_VALIDASI):
    """Pembanding akhir + OOF per lipatan; tulis CSV OOF, rekonsiliasi, dan laporan.

    `sumber` = hasil `jalur.info_sumber` (data masukan pipeline). Mengembalikan rekonsiliasi."""
    fp, lp = paths.data / "features_dev.csv", paths.input / "labels_dev.csv"
    if not fp.exists():
        raise FileNotFoundError(f"{fp} belum ada; jalankan tahap `fitur` dulu")
    rows = muat_data(fp, lp, kunci.theta)
    akhir = model_akhir(rows, kunci.laplace)
    oof, lipatan = walk_forward(rows, tahun_validasi, kunci.laplace, kunci.clip)
    recon = {
        "rm": "RM-23", "kelas": list(CLASSES), "theta_b1": kunci.theta, "laplace": kunci.laplace,
        "clip": kunci.clip,
        "sumber_sha256": {fp.name: common.sha256_file(fp), lp.name: common.sha256_file(lp)},
        "catatan": CATATAN, "akhir": akhir, "walk_forward": lipatan,
        "oof": {"n": len(oof), "log_loss_b0": _rerata([r["ll_b0"] for r in oof]),
                "log_loss_b1": _rerata([r["ll_b1"] for r in oof])},
        "versi_pdhl_data": sumber.get("versi_pdhl_data"), "commit": sumber.get("commit"),
        "sumber_pipeline_sha256": sumber.get("sumber_sha256"),
    }
    teks = laporan(recon)                     # semua isi siap sebelum ada berkas yang ditulis
    common.write_csv(paths.data / "pembanding_oof_dev.csv", oof, OOF_COLS)
    paths.output.mkdir(parents=True, exist_ok=True)
    (paths.output / "pembanding_rekonsiliasi.json").write_text(
        json.dumps(recon, indent=2, ensure_ascii=False) + "\n")
    (paths.output / "laporan_pembanding.md").write_text(teks)
    return recon


def ringkas(recon):
    """Baris ringkasan untuk konsol."""
    akhir, o = recon["akhir"], recon["oof"]
    b1 = akhir["B1"]
    return "\n".join([
        f"Pembanding akhir: {akhir['n_latih']} event dev, purge {akhir['n_purged']}",
        f"  B0              : {_p(akhir['B0'])}",
        f"  B1 wick >= 2/3  : {_p(b1['terpenuhi'])} (n {b1['terpenuhi']['n']})",
        f"  B1 wick <  2/3  : {_p(b1['tidak_terpenuhi'])} (n {b1['tidak_terpenuhi']['n']})",
        f"OOF {recon['walk_forward'][0]['tahun_validasi']}-"
        f"{recon['walk_forward'][-1]['tahun_validasi']}: {o['n']} event, log loss B0 "
        f"{_f(o['log_loss_b0'])}, B1 {_f(o['log_loss_b1'])}",
    ])


def _f(x):
    return "–" if x is None else f"{x:.4f}"
