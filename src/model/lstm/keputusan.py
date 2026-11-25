"""Tahap 6 (RM-27): aturan keputusan berbasis nilai harapan R (Reversal / Continuation / Abstain).

Menghitung parameter rata-rata R event Netral (`m_rev`, `m_cont`) murni dari `labels_dev.csv`
(2016–2023), lalu mengubah probabilitas terkalibrasi RM-26 menjadi nilai harapan:
    E_rev  = (a/b)·p_rev + (−1)·p_cont + m_rev·p_net
    E_cont = (−1)·p_rev + (b/a)·p_cont + m_cont·p_net = −(b/a)·E_rev

Untuk Kunci-1 (`a = 2, b = 1, δ = 0,1R`):
    E_rev >= +0,1R          -> Reversal     (realized_R = R_rev)
    E_rev <= −0,2R          -> Continuation (realized_R = R_cont, setara E_cont >= +0,1R)
    −0,2R < E_rev < +0,1R   -> Abstain      (realized_R = 0,0)

Modul ini murni memakai pustaka standar Python (stdlib). Data uji tidak pernah dimuat.
"""
import datetime as dt
import json
import math

from pdhl_data import common
from pdhl_data.common import CLASSES, DEV_END

from . import pembanding

KEPUTUSAN_COLS = [
    "event_id", "session_date", "level_type", "label",
    "p_rev_cal", "p_cont_cal", "p_net_cal",
    "E_rev", "E_cont", "keputusan",
    "R_rev", "R_cont", "R_none", "realized_R",
]
CATATAN = ("m_rev dan m_cont diestimasi murni dari event Netral dev (2016–2023); penerapan pada "
           "calib 2024 (227 event) bersifat verifikasi pra-Kunci-2 (bukan uji hipotesis resmi "
           "karena τ diestimasi pada 2024); data uji kunci_uji tetap tersegel.")


# ---------------------------------------------------------------- rumus inti RM-27
def estimasi_m_netral(labels_dev_rows, a, b):
    """Hitung m_rev dan m_cont dari event Netral dev setelah purging batas dev–kalibrasi."""
    batas_calib = DEV_END + dt.timedelta(days=1)
    latih, purged = pembanding.data_latih(labels_dev_rows, batas_calib)
    netral = [r for r in latih if r["label"] == "Netral"]
    if not netral:
        raise ValueError("tidak ada event Netral pada data dev untuk mengestimasi m_rev/m_cont")

    r_revs = [float(r["R_rev"]) for r in netral]
    r_conts = [float(r["R_cont"]) for r in netral]
    if not all(math.isfinite(v) for v in (*r_revs, *r_conts)):
        raise ValueError("nilai R_rev/R_cont Netral dev tidak finite")

    m_rev = sum(r_revs) / len(r_revs)
    m_cont = sum(r_conts) / len(r_conts)
    rasio_teori = -float(b) / float(a)
    if abs(m_cont - rasio_teori * m_rev) > 1e-8:
        raise ValueError(
            f"identitas m_cont = -(b/a)*m_rev tidak terpenuhi: m_cont={m_cont}, "
            f"-(b/a)*m_rev={rasio_teori * m_rev}"
        )

    return {
        "n_dev_latih": len(latih),
        "n_dev_purged": len(purged),
        "n_netral_dev": len(netral),
        "m_rev": m_rev,
        "m_cont": m_cont,
        "rasio_b_per_a": float(b) / float(a),
    }


def nilai_harapan(p_rev, p_cont, p_net, m_rev, m_cont, a, b):
    """Hitung pasangan nilai harapan (E_rev, E_cont) dalam satuan R."""
    ab = float(a) / float(b)
    ba = float(b) / float(a)
    e_rev = ab * p_rev - 1.0 * p_cont + m_rev * p_net
    e_cont = -1.0 * p_rev + ba * p_cont + m_cont * p_net
    return e_rev, e_cont


def putuskan(e_rev, e_cont, delta):
    """Tentukan Reversal / Continuation / Abstain dari nilai harapan dan ambang `delta >= 0`."""
    d = float(delta)
    if d < 0 or not math.isfinite(d):
        raise ValueError(f"delta harus finite dan >= 0, didapat {delta}")
    if e_rev >= d and e_rev > e_cont:
        return "Reversal"
    if e_cont >= d and e_cont > e_rev:
        return "Continuation"
    return "Abstain"


def realized_r(keputusan, r_rev, r_cont, r_none=0.0):
    """Imbal hasil nyata (satuan R) sesuai keputusan yang dipilih."""
    if keputusan == "Reversal":
        return float(r_rev)
    if keputusan == "Continuation":
        return float(r_cont)
    if keputusan == "Abstain":
        return float(r_none)
    raise ValueError(f"keputusan tidak dikenal: {keputusan!r}")


def _rerata(xs):
    return sum(xs) / len(xs) if xs else None


def ringkas_keputusan_rows(rows):
    """Ringkas cakupan, distribusi aksi, total R, rata-rata R/event, & rata-rata R/rekomendasi."""
    n = len(rows)
    if n == 0:
        raise ValueError("rows keputusan kosong")
    n_rev = sum(r["keputusan"] == "Reversal" for r in rows)
    n_cont = sum(r["keputusan"] == "Continuation" for r in rows)
    n_abs = sum(r["keputusan"] == "Abstain" for r in rows)
    aktif = [r for r in rows if r["keputusan"] != "Abstain"]
    total_r = sum(r["realized_R"] for r in rows)

    per_aksi = {}
    for aksi in ("Reversal", "Continuation", "Abstain"):
        sub = [r for r in rows if r["keputusan"] == aksi]
        tot = sum(r["realized_R"] for r in sub)
        per_aksi[aksi] = {
            "n": len(sub),
            "proporsi": len(sub) / n,
            "total_R": tot,
            "mean_R": _rerata([r["realized_R"] for r in sub]),
        }

    per_level = {}
    for lv in ("PDH", "PDL"):
        sub = [r for r in rows if r["level_type"] == lv]
        sub_aktif = [r for r in sub if r["keputusan"] != "Abstain"]
        per_level[lv] = {
            "n": len(sub),
            "n_rekomendasi": len(sub_aktif),
            "cakupan": len(sub_aktif) / len(sub) if sub else None,
            "total_R": sum(r["realized_R"] for r in sub),
            "mean_R_semua": _rerata([r["realized_R"] for r in sub]),
            "mean_R_rekomendasi": _rerata([r["realized_R"] for r in sub_aktif]),
        }

    return {
        "n_event": n,
        "n_reversal": n_rev,
        "n_continuation": n_cont,
        "n_abstain": n_abs,
        "n_rekomendasi": len(aktif),
        "cakupan": len(aktif) / n,
        "total_R": total_r,
        "mean_R_semua": total_r / n,
        "mean_R_rekomendasi": _rerata([r["realized_R"] for r in aktif]),
        "per_keputusan": per_aksi,
        "per_level": per_level,
    }


def terapkan_aturan(prob_rows, label_rows, *, m_rev, m_cont, a, b, delta):
    """Terapkan aturan RM-27 atas daftar probabilitas dan label ber-R; kembalikan (rows, ringkasan)."""
    if len(prob_rows) != len(label_rows):
        raise ValueError(f"jumlah prediksi ({len(prob_rows)}) != label ({len(label_rows)})")
    out = []
    for pr, lab in zip(prob_rows, label_rows):
        if pr["event_id"] != lab["event_id"] or pr["session_date"] != lab["session_date"]:
            raise ValueError(f"event tidak sejajar: {pr['event_id']} vs {lab['event_id']}")
        p_rev, p_cont, p_net = float(pr["p_rev"]), float(pr["p_cont"]), float(pr["p_net"])
        e_rev, e_cont = nilai_harapan(p_rev, p_cont, p_net, m_rev, m_cont, a, b)
        aksi = putuskan(e_rev, e_cont, delta)
        r_rev, r_cont = float(lab["R_rev"]), float(lab["R_cont"])
        r_none = float(lab["R_none"])
        rr = realized_r(aksi, r_rev, r_cont, r_none)
        out.append({
            "event_id": pr["event_id"],
            "session_date": pr["session_date"],
            "level_type": lab["level_type"],
            "label": lab["label"],
            "p_rev_cal": p_rev,
            "p_cont_cal": p_cont,
            "p_net_cal": p_net,
            "E_rev": e_rev,
            "E_cont": e_cont,
            "keputusan": aksi,
            "R_rev": r_rev,
            "R_cont": r_cont,
            "R_none": r_none,
            "realized_R": rr,
        })
    return out, ringkas_keputusan_rows(out)


# ---------------------------------------------------------------- validasi prasyarat
def cek_kalibrasi_segar(paths):
    """Pastikan hasil RM-26 ada dan `calib_pred_lstm.csv` cocok dengan rekonsiliasi RM-26."""
    kp = paths.output / "kalibrasi_rekonsiliasi.json"
    cp = paths.data / "calib_pred_lstm.csv"
    if not kp.exists() or not cp.exists():
        raise FileNotFoundError(f"{kp} / {cp} belum ada; jalankan tahap `kalibrasi` dulu")
    kal_recon = json.loads(kp.read_text())
    dicatat = kal_recon.get("sumber_sha256", {})
    if dicatat.get(cp.name) != common.sha256_file(cp):
        raise ValueError(f"{cp.name} berbeda dari rekonsiliasi RM-26; jalankan ulang `kalibrasi`")
    if dicatat.get("labels_calib.csv") != common.sha256_file(paths.input / "labels_calib.csv"):
        raise ValueError("labels_calib.csv berubah sejak RM-26; jalankan ulang `kalibrasi`")
    return kal_recon, common.sha256_file(kp)


# ---------------------------------------------------------------- laporan
def _f(x, nd=4):
    return "–" if x is None else f"{x:.{nd}f}"


def _r(x, nd=4):
    if x is None:
        return "–"
    v = round(x, nd)
    return f"{v + 0.0:+.{nd}f}" if v else f"0.{'0' * nd}"


def laporan(recon):
    par = recon["parameter_keputusan"]
    u = recon["calib_utama"]
    lines = [
        "# Laporan Aturan Keputusan RM-27 — Jalur LSTM", "",
        f"Sumber: `model.lstm.keputusan`. Parameter `m_rev` & `m_cont` diestimasi dari event "
        f"Netral `dev` (`n = {par['n_netral_dev']}` dari `{par['n_dev_latih']}` event dev, purge "
        f"`{par['n_dev_purged']}`), lalu diterapkan pada prediksi terkalibrasi `calib` 2024 "
        f"(`n = {u['n_event']}`, `τ = {recon['tau']:.6f}`).", "",
        f"Catatan: {recon['catatan']}", "",
        "## 1. Parameter Aturan Keputusan Terkunci (dari `dev` + Kunci-1)", "",
        f"- **Parameter label (Kunci-1):** `a = {par['a']:g}`, `b = {par['b']:g}` "
        f"(menang Reversal `+{par['a'] / par['b']:g}R`, menang Continuation `+{par['b'] / par['a']:g}R`, kalah `−1R`).",
        f"- **Rata-rata R Netral `dev`:** `m_rev = {par['m_rev']:+.6f}R`, "
        f"`m_cont = {par['m_cont']:+.6f}R` (`= −{par['rasio_b_per_a']:g} × m_rev`).",
        f"- **Ambang utama:** `δ = {par['delta_utama']:g}R` → "
        f"**Reversal** bila `E_rev ≥ +{par['ambang_e_rev_reversal']:g}R`, "
        f"**Continuation** bila `E_rev ≤ {par['ambang_e_rev_continuation']:g}R` "
        f"(`E_cont ≥ +{par['delta_utama']:g}R`), selain itu **Abstain (`0R`)**.", "",
        "## 2. Ringkasan Keputusan pada `calib` 2024 (`δ = 0,1R`, Diagnostik Pra-Uji)", "",
        "| Model | Rev | Cont | Abstain | Cakupan | Total `realized_R` | Rata-rata R / Event | Rata-rata R / Rekomendasi |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for nama, m in (
        (f"**LSTM Calibrated (`τ = {recon['tau']:.4f}`)**", u),
        ("LSTM Raw (`τ = 1,0000`)", recon["calib_raw_utama"]),
        ("B0 akhir (`dev`)", recon["pembanding_calib_utama"]["B0"]),
        ("B1 akhir (`dev`)", recon["pembanding_calib_utama"]["B1"]),
    ):
        lines.append(
            f"| {nama} | {m['n_reversal']} | {m['n_continuation']} | {m['n_abstain']} | "
            f"{100 * m['cakupan']:.1f}% | {_r(m['total_R'])} | {_r(m['mean_R_semua'])} | "
            f"{_r(m['mean_R_rekomendasi'])} |"
        )

    lines += [
        "", "## 3. Kurva Cakupan–Kinerja pada Grid `δ` (`calib` 2024, LSTM Terkalibrasi)", "",
        "| `δ` (R) | Batas `E_rev` Rev / Cont | Rev | Cont | Abstain | Cakupan | Total R | Mean R (Semua) | Mean R (Rekomendasi) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ab = par["a"] / par["b"]
    for item in recon["kurva_delta_cal"]:
        d = item["delta"]
        tanda = "**" if math.isclose(d, par["delta_utama"]) else ""
        lines.append(
            f"| {tanda}{d:.1f}{tanda} | `≥ +{d:.1f}` / `≤ -{ab * d:.1f}` | {item['n_reversal']} | "
            f"{item['n_continuation']} | {item['n_abstain']} | {100 * item['cakupan']:.1f}% | "
            f"{_r(item['total_R'])} | {_r(item['mean_R_semua'])} | {_r(item['mean_R_rekomendasi'])} |"
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------- tahap
def run_keputusan(paths, kunci, sumber):
    """Estimasi m_rev/m_cont dari dev Netral, terapkan RM-27 pada calib 2024, & tulis keluaran."""
    kal_recon, kal_sha = cek_kalibrasi_segar(paths)
    pb_recon = json.loads((paths.output / "pembanding_rekonsiliasi.json").read_text())

    labels_dev = common.load_model_csv(paths.input / "labels_dev.csv")
    labels_calib = common.load_model_csv(paths.input / "labels_calib.csv")
    calib_preds = common.load_model_csv(paths.data / "calib_pred_lstm.csv")

    est = estimasi_m_netral(labels_dev, kunci.a, kunci.b)
    m_rev, m_cont = est["m_rev"], est["m_cont"]
    ab = kunci.a / kunci.b

    cal_probs = [
        {"event_id": r["event_id"], "session_date": r["session_date"],
         "p_rev": float(r["p_rev_cal"]), "p_cont": float(r["p_cont_cal"]), "p_net": float(r["p_net_cal"])}
        for r in calib_preds
    ]
    raw_probs = [
        {"event_id": r["event_id"], "session_date": r["session_date"],
         "p_rev": float(r["p_rev_raw"]), "p_cont": float(r["p_cont_raw"]), "p_net": float(r["p_net_raw"])}
        for r in calib_preds
    ]

    rows_utama, ring_utama = terapkan_aturan(
        cal_probs, labels_calib, m_rev=m_rev, m_cont=m_cont, a=kunci.a, b=kunci.b, delta=kunci.delta_utama
    )
    _, ring_raw_utama = terapkan_aturan(
        raw_probs, labels_calib, m_rev=m_rev, m_cont=m_cont, a=kunci.a, b=kunci.b, delta=kunci.delta_utama
    )

    kurva_delta_cal = []
    for d in kunci.delta_grid:
        _, rg = terapkan_aturan(
            cal_probs, labels_calib, m_rev=m_rev, m_cont=m_cont, a=kunci.a, b=kunci.b, delta=d
        )
        kurva_delta_cal.append({"delta": float(d), **rg})

    # Evaluasi pembanding B0 & B1 akhir pada calib 2024 dengan aturan RM-27 yang sama
    model_pb = pb_recon["akhir"]
    b0_probs, b1_probs = [], []
    for r in calib_preds:
        flag = common.parse_bool(r["b1_terpenuhi"])
        p = pembanding.prediksi(model_pb, {"b1": flag})
        b0_probs.append({"event_id": r["event_id"], "session_date": r["session_date"],
                         "p_rev": p["B0"][0], "p_cont": p["B0"][1], "p_net": p["B0"][2]})
        b1_probs.append({"event_id": r["event_id"], "session_date": r["session_date"],
                         "p_rev": p["B1"][0], "p_cont": p["B1"][1], "p_net": p["B1"][2]})
    _, ring_b0 = terapkan_aturan(
        b0_probs, labels_calib, m_rev=m_rev, m_cont=m_cont, a=kunci.a, b=kunci.b, delta=kunci.delta_utama
    )
    _, ring_b1 = terapkan_aturan(
        b1_probs, labels_calib, m_rev=m_rev, m_cont=m_cont, a=kunci.a, b=kunci.b, delta=kunci.delta_utama
    )

    out_csv = paths.data / "keputusan_calib_lstm.csv"
    common.write_csv(out_csv, rows_utama, KEPUTUSAN_COLS)

    recon = {
        "rm": "RM-27",
        "kelas": list(CLASSES),
        "terpilih": dict(kal_recon["terpilih"]),
        "tau": float(kal_recon["tau"]),
        "parameter_keputusan": {
            "a": float(kunci.a),
            "b": float(kunci.b),
            "delta_utama": float(kunci.delta_utama),
            "delta_grid": [float(d) for d in kunci.delta_grid],
            "ambang_e_rev_reversal": float(kunci.delta_utama),
            "ambang_e_rev_continuation": -ab * float(kunci.delta_utama),
            **est,
        },
        "calib_utama": ring_utama,
        "calib_raw_utama": ring_raw_utama,
        "kurva_delta_cal": kurva_delta_cal,
        "pembanding_calib_utama": {"B0": ring_b0, "B1": ring_b1},
        "sumber_sha256": {
            "kalibrasi_rekonsiliasi.json": kal_sha,
            "calib_pred_lstm.csv": common.sha256_file(paths.data / "calib_pred_lstm.csv"),
            "labels_dev.csv": common.sha256_file(paths.input / "labels_dev.csv"),
            "labels_calib.csv": common.sha256_file(paths.input / "labels_calib.csv"),
            "keputusan_calib_lstm.csv": common.sha256_file(out_csv),
        },
        "catatan": CATATAN,
        "versi_pdhl_data": sumber.get("versi_pdhl_data"),
        "commit": sumber.get("commit"),
        "sumber_pipeline_sha256": sumber.get("sumber_sha256"),
    }

    teks = laporan(recon)
    paths.output.mkdir(parents=True, exist_ok=True)
    (paths.output / "keputusan_rekonsiliasi.json").write_text(
        json.dumps(recon, indent=2, ensure_ascii=False) + "\n"
    )
    (paths.output / "laporan_keputusan.md").write_text(teks)
    return recon


def ringkas(recon):
    p = recon["parameter_keputusan"]
    u = recon["calib_utama"]
    return "\n".join([
        f"Aturan keputusan RM-27: m_rev = {p['m_rev']:+.4f}R, m_cont = {p['m_cont']:+.4f}R "
        f"(n Netral dev = {p['n_netral_dev']}), δ = {p['delta_utama']:g}R",
        f"  calib 2024 (n={u['n_event']}): Rev {u['n_reversal']} / Cont {u['n_continuation']} / "
        f"Abstain {u['n_abstain']} (cakupan {100 * u['cakupan']:.1f}%)",
        f"  realized_R (pra-uji): total {u['total_R']:+.4f}R, "
        f"mean/event {u['mean_R_semua']:+.4f}R, mean/rekom {_r(u['mean_R_rekomendasi'])}R",
    ])
