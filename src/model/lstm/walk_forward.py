"""Tahap 3 (RM-24): validasi walk-forward LSTM dan pemilihan konfigurasi.

Sepuluh konfigurasi LSTM (5 lookback L ∈ {23, 40, 46, 60, 96} × 2 hidden_size ∈ {16, 32})
dilatih pada lima lipatan walk-forward berjangkar: latih 2016..y-1 (dengan purging identik jalur
Gradient Boosting), validasi tahun y, untuk y = 2019..2023 (Proposal v2 IV.II.4).

Konfigurasi dengan rata-rata log loss out-of-fold (OOF) terkecil dipilih; selisih ≤ toleransi seri
Kunci-1 (1e-6) dianggap seri dan dimenangkan konfigurasi yang lebih sederhana (`hidden_size` lebih
kecil, lalu `lookback_L` lebih pendek). Probabilitas walk-forward tidak dikalibrasi (kalibrasi
hanya memakai data 2024 pada tahap model akhir).

Prediksi OOF model terpilih dibandingkan secara deskriptif dengan OOF pembanding B0/B1 (tahap 2).
Hanya data dev yang dimuat. Butuh PyTorch dan NumPy (`pip install -e '.[lstm]'`), kecuali saat
diuji memakai pelatih palsu (stdlib).
"""
import datetime as dt
import itertools
import json
import platform

from pdhl_data import common
from pdhl_data.common import CLASSES

from . import arsitektur, fitur, pembanding
from .kunci import HIDDEN_TERKUNCI, LOOKBACKS_TERKUNCI, SEED_MODEL

if CLASSES != ("Reversal", "Continuation", "Netral"):
    raise RuntimeError("kolom p_rev/p_cont/p_net mengandaikan urutan kelas Kunci-1")

TAHUN_VALIDASI = pembanding.TAHUN_VALIDASI
OOF_COLS = ["event_id", "session_date", "tahun_validasi", "label", "konfigurasi",
            "p_rev", "p_cont", "p_net", "ll"]


# ---------------------------------------------------------------- konfigurasi
def konfigurasi_grid(kunci):
    """Sepuluh konfigurasi (L × hidden); menolak kunci yang menyimpang dari grid terkunci."""
    beda = []
    if tuple(kunci.lookbacks) != LOOKBACKS_TERKUNCI:
        beda.append("lookbacks")
    if tuple(kunci.hidden_sizes) != HIDDEN_TERKUNCI:
        beda.append("hidden_sizes")
    if kunci.seed_model != SEED_MODEL:
        beda.append("seed_model")
    if beda:
        raise ValueError(f"teknis LSTM menyimpang dari spesifikasi terkunci: {', '.join(beda)}")
    return [
        {"id": f"L{L}_h{h}", "lookback_L": int(L), "hidden_size": int(h)}
        for L, h in itertools.product(kunci.lookbacks, kunci.hidden_sizes)
    ]


def kesederhanaan(cfg):
    """Kunci urut: makin kecil makin sederhana (hidden lebih kecil, lalu jendela L lebih pendek)."""
    return (cfg["hidden_size"], cfg["lookback_L"])


def himpunan_seri(skor, cfgs, tol):
    """Konfigurasi yang selisihnya dengan skor terbaik ≤ tol (termasuk yang terbaik)."""
    terbaik = min(skor[c["id"]] for c in cfgs)
    return [c for c in cfgs if skor[c["id"]] - terbaik <= tol]


def pilih_konfigurasi(skor, cfgs, tol):
    return min(himpunan_seri(skor, cfgs, tol), key=kesederhanaan)


def ringkasan_pemilihan(skor, cfgs, tol):
    """Peringkat, pilihan, himpunan seri, dan konfigurasi terdekat untuk laporan."""
    terpilih = pilih_konfigurasi(skor, cfgs, tol)
    urut = sorted(cfgs, key=lambda c: (skor[c["id"]], kesederhanaan(c)))
    lain = [c for c in urut if c["id"] != terpilih["id"]]
    return {
        "terpilih": terpilih,
        "peringkat": {c["id"]: i for i, c in enumerate(urut, 1)},
        "seri": [c["id"] for c in himpunan_seri(skor, cfgs, tol)],
        "terdekat": lain[0]["id"] if lain else None,
        "selisih_ke_terdekat": skor[lain[0]["id"]] - skor[terpilih["id"]] if lain else None,
    }


# ---------------------------------------------------------------- metrik pendukung (stdlib)
def _rerata(xs):
    return sum(xs) / len(xs) if xs else None


def brier_score_rows(oof_rows):
    """Multi-class Brier score deskriptif: mean(sum_k (p_ik - y_ik)^2)."""
    if not oof_rows:
        return None
    total = 0.0
    for r in oof_rows:
        p = (r["p_rev"], r["p_cont"], r["p_net"])
        yi = CLASSES.index(r["label"])
        total += sum((p[k] - (1.0 if k == yi else 0.0)) ** 2 for k in range(len(CLASSES)))
    return total / len(oof_rows)


def ece_score_rows(oof_rows, n_bins=10):
    """Expected Calibration Error (ECE) berbasis confidence maksimum (deskriptif)."""
    if not oof_rows:
        return None
    n = len(oof_rows)
    bins = [[] for _ in range(n_bins)]
    for r in oof_rows:
        p = (r["p_rev"], r["p_cont"], r["p_net"])
        conf = max(p)
        pred = p.index(conf)
        acc = 1.0 if pred == CLASSES.index(r["label"]) else 0.0
        b = min(int(conf * n_bins), n_bins - 1)
        bins[b].append((conf, acc))
    return sum(
        (len(items) / n) * abs(_rerata([a for _, a in items]) - _rerata([c for c, _ in items]))
        for items in bins if items
    )


# ---------------------------------------------------------------- walk-forward
def _latih_default(X_tr, y_tr, cfg, teknis, seeds, clip, device):
    return arsitektur.latih_lstm(X_tr, y_tr, cfg, teknis, seeds=seeds, clip=clip, device=device)


def walk_forward_model(
    rows_by_L,
    cfgs,
    teknis,
    seeds,
    tahun,
    clip,
    *,
    latih=None,
    prediksi=None,
    device="cpu",
    verbose=True,
):
    """Prediksi OOF setiap konfigurasi pada setiap lipatan; data latih memakai purging.

    `rows_by_L` memetakan `lookback_L -> list[dict]` dengan kunci:
    `event_id, session_date, label, label_end_session_date, x` (sekuens L × F).
    """
    latih_fn = latih or (lambda X, y, c, t: _latih_default(X, y, c, t, seeds, clip, device))
    prediksi_fn = prediksi or arsitektur.prediksi_lstm

    oof, lipatan = [], []
    # Proses per konfigurasi agar progres per (L, hidden) langsung terlihat jelas di konsol
    for cfg in cfgs:
        L = cfg["lookback_L"]
        rows = rows_by_L[L]
        cfg_lls, cfg_epochs = [], []
        for y in tahun:
            data, purged = pembanding.data_latih(rows, dt.date(y, 1, 1))
            if not data:
                raise ValueError(f"lipatan {y}: tidak ada data latih")
            val = [r for r in rows if int(r["session_date"][:4]) == y]
            if not val:
                raise ValueError(f"lipatan {y}: tidak ada event validasi")
            X_tr = [r["x"] for r in data]
            y_tr = [CLASSES.index(r["label"]) for r in data]
            X_val = [r["x"] for r in val]

            model = latih_fn(X_tr, y_tr, cfg, teknis)
            preds = prediksi_fn(model, X_val)
            if len(preds) != len(val):
                raise ValueError(f"lipatan {y}: {len(preds)} prediksi untuk {len(val)} event")

            lls = []
            for r, p in zip(val, preds):
                if len(p) != len(CLASSES) or abs(sum(p) - 1.0) > 1e-6 or min(p) < 0:
                    raise ValueError(f"{r['event_id']}: peluang tidak valid {p}")
                ll = pembanding.log_loss(p, r["label"], clip)
                oof.append({
                    "event_id": r["event_id"],
                    "session_date": r["session_date"],
                    "tahun_validasi": y,
                    "label": r["label"],
                    "konfigurasi": cfg["id"],
                    "p_rev": p[0],
                    "p_cont": p[1],
                    "p_net": p[2],
                    "ll": ll,
                })
                lls.append(ll)
            cfg_lls.extend(lls)
            fold_epochs = model.get("best_epochs") if isinstance(model, dict) else None
            if fold_epochs:
                cfg_epochs.extend(fold_epochs)
            lipatan.append({
                "tahun_validasi": y,
                "konfigurasi": cfg["id"],
                "n_latih": len(data),
                "n_purged": len(purged),
                "n_validasi": len(val),
                "log_loss": _rerata(lls),
                "avg_best_epoch": _rerata(fold_epochs) if fold_epochs else None,
            })
        if verbose:
            ep_txt = f" | avg_epoch={_rerata(cfg_epochs):.1f}" if cfg_epochs else ""
            print(f"  [lstm-wf] {cfg['id']:<8} log_loss OOF = {_rerata(cfg_lls):.4f}{ep_txt}",
                  flush=True)
    return oof, lipatan


def skor_konfigurasi(oof, cfgs):
    """Rata-rata log loss OOF per event (semua tahun validasi digabung) untuk setiap konfigurasi."""
    return {c["id"]: _rerata([r["ll"] for r in oof if r["konfigurasi"] == c["id"]]) for c in cfgs}


def banding_pembanding(oof_terpilih, pembanding_rows):
    """Selisih log loss OOF model − B0/B1 per tahun dan gabungan (deskriptif)."""
    pb = {r["event_id"]: r for r in pembanding_rows}
    if {r["event_id"] for r in oof_terpilih} != set(pb) or len(pb) != len(oof_terpilih):
        raise ValueError("event OOF model dan pembanding tidak sama")
    for r in oof_terpilih:
        b = pb[r["event_id"]]
        if ("label" in b and b["label"] != r["label"]) or \
                ("tahun_validasi" in b and int(b["tahun_validasi"]) != r["tahun_validasi"]):
            raise ValueError(f"{r['event_id']}: label/tahun berbeda dengan berkas pembanding")

    def ringkas_bagian(rs):
        m = [r["ll"] for r in rs]
        b0 = [float(pb[r["event_id"]]["ll_b0"]) for r in rs]
        b1 = [float(pb[r["event_id"]]["ll_b1"]) for r in rs]
        return {
            "n": len(rs),
            "ll_model": _rerata(m),
            "ll_b0": _rerata(b0),
            "ll_b1": _rerata(b1),
            "D_b0": _rerata([a - b for a, b in zip(m, b0)]),
            "D_b1": _rerata([a - b for a, b in zip(m, b1)]),
        }

    tahun = sorted({r["tahun_validasi"] for r in oof_terpilih})
    return {
        "per_tahun": [
            {"tahun": t, **ringkas_bagian([r for r in oof_terpilih if r["tahun_validasi"] == t])}
            for t in tahun
        ],
        "gabungan": ringkas_bagian(oof_terpilih),
    }


# ---------------------------------------------------------------- laporan
def _f(x):
    return "–" if x is None else f"{x:.4f}"


def _d(x):
    if x is None:
        return "–"
    x = round(x, 4)
    return f"{x + 0.0:+.4f}" if x else "0.0000"


def laporan(recon):
    cfgs, skor, per = recon["konfigurasi"], recon["skor"], recon["per_tahun"]
    pilih, peringkat = recon["terpilih"]["id"], recon["peringkat"]
    tahun = recon["tahun_validasi"]
    env = recon["lingkungan"]
    lines = [
        "# Laporan Validasi Walk-Forward LSTM (RM-24)", "",
        "Sumber: `model.lstm.walk_forward`, data pengembangan saja. Lingkungan: Python "
        f"{env['python']}, PyTorch {env['torch']}, NumPy {env['numpy']} ({env['platform']}).",
        "",
        "Probabilitas walk-forward **tidak dikalibrasi** (kalibrasi hanya memakai data 2024 pada "
        f"tahap model akhir). Seeds: `{recon['seeds']}` (rata-rata logits lintas "
        f"{len(recon['seeds'])} seed). Catatan: {recon['catatan']}", "",
        "## 1. Log loss OOF per konfigurasi", "",
        "Rata-rata log loss per event (makin kecil makin baik). Gabungan = rata-rata seluruh "
        f"{recon['oof_n_per_konfigurasi']} event validasi {tahun[0]}–{tahun[-1]}, dasar "
        "pemilihan.", "",
        "| # | Konfigurasi | lookback_L | hidden_size | "
        + " | ".join(str(t) for t in tahun) + " | Gabungan | Brier | ECE | Peringkat |",
        "|" + "---|" * (8 + len(tahun)),
    ]
    brier_map, ece_map = recon["brier"], recon["ece"]
    for i, c in enumerate(cfgs, 1):
        tanda = "**" if c["id"] == pilih else ""
        lines.append(
            f"| {i} | {tanda}{c['id']}{tanda} | {c['lookback_L']} | {c['hidden_size']} | "
            + " | ".join(_f(per[c["id"]].get(str(t))) for t in tahun)
            + f" | {tanda}{_f(skor[c['id']])}{tanda} | {_f(brier_map[c['id']])} "
            + f"| {_f(ece_map[c['id']])} | {peringkat[c['id']]} |"
        )
    p = recon["pemilihan"]
    lines += [
        "", "## 2. Konfigurasi terpilih", "",
        f"**{pilih}** (lookback_L {recon['terpilih']['lookback_L']}, hidden_size "
        f"{recon['terpilih']['hidden_size']}), log loss OOF {_f(skor[pilih])} "
        f"(Brier {_f(brier_map[pilih])}, ECE {_f(ece_map[pilih])}).", "",
        f"- Konfigurasi terdekat ({p['terdekat'] or '–'}): selisih log loss "
        f"{_d(p['selisih_ke_terdekat'])} terhadap konfigurasi terpilih.",
        f"- Himpunan seri dengan skor terbaik (selisih ≤ {recon['toleransi_seri']}): "
        f"{', '.join(p['seri'])}"
        + (" (hanya konfigurasi terpilih, tidak ada seri)." if len(p["seri"]) == 1
           else "; aturan kesederhanaan dipakai."), "",
        "## 3. Kestabilan terhadap pembanding (deskriptif, bukan bukti keunggulan)", "",
        "Selisih log loss OOF model terpilih − pembanding (B0/B1 diestimasi ulang per "
        "lipatan, RM-23). Negatif = model lebih baik.", "",
        "| Tahun | n | LSTM | B0 | B1 | Model − B0 | Model − B1 |",
        "|---|---|---|---|---|---|---|",
    ]
    b = recon["banding_pembanding"]
    for t in b["per_tahun"]:
        lines.append(f"| {t['tahun']} | {t['n']} | {_f(t['ll_model'])} | {_f(t['ll_b0'])} | "
                     f"{_f(t['ll_b1'])} | {_d(t['D_b0'])} | {_d(t['D_b1'])} |")
    g = b["gabungan"]
    lines += [f"| **Gabungan** | {g['n']} | {_f(g['ll_model'])} | {_f(g['ll_b0'])} | "
              f"{_f(g['ll_b1'])} | {_d(g['D_b0'])} | {_d(g['D_b1'])} |", ""]
    return "\n".join(lines)


def _lingkungan():
    def versi(nama):
        try:
            return __import__(nama).__version__
        except ModuleNotFoundError:
            return None
    return {
        "python": platform.python_version(),
        "torch": versi("torch"),
        "numpy": versi("numpy"),
        "platform": platform.platform(),
    }


# ---------------------------------------------------------------- tahap
CATATAN = ("perbandingan apple-to-apple dengan jalur Gradient Boosting pada lipatan, purging, "
           "label, dan pembanding B0/B1 yang identik (Kunci-1); model akhir & kalibrasi 2024 "
           "belum dijalankan di tahap ini.")


def cek_pembanding_segar(paths, mp, lp):
    """OOF pembanding harus dihitung dari seq_meta_dev.csv dan labels_dev.csv yang sama."""
    rp = paths.output / "pembanding_rekonsiliasi.json"
    dicatat = json.loads(rp.read_text()).get("sumber_sha256", {}) if rp.exists() else {}
    if any(dicatat.get(p.name) != common.sha256_file(p) for p in (mp, lp)):
        raise ValueError(f"{paths.data / 'pembanding_oof_dev.csv'} tidak dihitung dari fitur/label "
                         "yang sekarang; jalankan ulang tahap `pembanding`")


def _muat_rows_per_lookback(paths, lookbacks, theta):
    """Bangun kamus `L -> list[dict]` siap walk-forward untuk setiap kandidat lookback L."""
    bars = fitur.prepare_bars(common.load_model_csv(paths.input / "bars_devcalib.csv"))
    events_dev = common.load_model_csv(paths.input / "events_dev.csv")
    labels_dev = common.load_model_csv(paths.input / "labels_dev.csv")
    meta_dev = common.load_model_csv(paths.data / "seq_meta_dev.csv")
    gab = pembanding.gabung(meta_dev, labels_dev, theta)
    if [e["event_id"] for e in events_dev] != [r["event_id"] for r in gab]:
        raise ValueError("urutan event_id events_dev.csv tidak cocok dengan seq_meta_dev.csv")

    ctx = fitur.build_context(bars)
    rows_by_L = {}
    for L in sorted(set(lookbacks)):
        seqs, _ = fitur.hitung_semua_sekuens(events_dev, bars, int(L), theta, ctx=ctx)
        rows_by_L[int(L)] = [{**r, "x": s} for r, s in zip(gab, seqs)]
    return rows_by_L


def run_walk_forward(
    paths,
    kunci,
    sumber,
    *,
    cfgs=None,
    tahun_validasi=TAHUN_VALIDASI,
    latih=None,
    prediksi=None,
    device="cpu",
    verbose=True,
):
    """Walk-forward semua konfigurasi LSTM, pilih terbaik, bandingkan B0/B1, tulis keluaran."""
    mp = paths.data / "seq_meta_dev.csv"
    lp = paths.input / "labels_dev.csv"
    pp = paths.data / "pembanding_oof_dev.csv"
    for p, tahap in ((mp, "fitur"), (pp, "pembanding")):
        if not p.exists():
            raise FileNotFoundError(f"{p} belum ada; jalankan tahap `{tahap}` dulu")
    cek_pembanding_segar(paths, mp, lp)

    cfgs = cfgs or konfigurasi_grid(kunci)
    rows_by_L = _muat_rows_per_lookback(paths, [c["lookback_L"] for c in cfgs], kunci.theta)

    oof, lipatan = walk_forward_model(
        rows_by_L, cfgs, kunci.lstm_teknis, kunci.seeds, tahun_validasi, kunci.clip,
        latih=latih, prediksi=prediksi, device=device, verbose=verbose,
    )
    skor = skor_konfigurasi(oof, cfgs)
    brier_map = {c["id"]: brier_score_rows([r for r in oof if r["konfigurasi"] == c["id"]]) for c in cfgs}
    ece_map = {c["id"]: ece_score_rows([r for r in oof if r["konfigurasi"] == c["id"]]) for c in cfgs}
    pilihan = ringkasan_pemilihan(skor, cfgs, kunci.toleransi_seri)
    terpilih, peringkat = pilihan["terpilih"], pilihan["peringkat"]
    banding = banding_pembanding(
        [r for r in oof if r["konfigurasi"] == terpilih["id"]],
        common.load_model_csv(pp),
    )

    avg_epochs = {}
    for c in cfgs:
        eps = [f["avg_best_epoch"] for f in lipatan
               if f["konfigurasi"] == c["id"] and f["avg_best_epoch"] is not None]
        avg_epochs[c["id"]] = _rerata(eps) if eps else None

    recon = {
        "rm": "RM-24",
        "kelas": list(CLASSES),
        "fitur_sekuensial": list(fitur.FEATURE_NAMES),
        "lingkungan": _lingkungan(),
        "params_tetap": dict(kunci.lstm_teknis),
        "seeds": list(kunci.seeds),
        "device": device,
        "clip": kunci.clip,
        "toleransi_seri": kunci.toleransi_seri,
        "tahun_validasi": list(tahun_validasi),
        "sumber_sha256": {p.name: common.sha256_file(p) for p in (mp, lp, pp)},
        "catatan": CATATAN,
        "konfigurasi": cfgs,
        "skor": skor,
        "brier": brier_map,
        "ece": ece_map,
        "avg_best_epoch": avg_epochs,
        "peringkat": peringkat,
        "per_tahun": {
            c["id"]: {str(f["tahun_validasi"]): f["log_loss"] for f in lipatan
                      if f["konfigurasi"] == c["id"]}
            for c in cfgs
        },
        "lipatan": [
            {k: f[k] for k in ("tahun_validasi", "n_latih", "n_purged", "n_validasi")}
            for f in lipatan if f["konfigurasi"] == cfgs[0]["id"]
        ],
        "terpilih": terpilih,
        "pemilihan": {k: pilihan[k] for k in ("seri", "terdekat", "selisih_ke_terdekat")},
        "oof_n_per_konfigurasi": sum(r["konfigurasi"] == cfgs[0]["id"] for r in oof),
        "banding_pembanding": banding,
        "versi_pdhl_data": sumber.get("versi_pdhl_data"),
        "commit": sumber.get("commit"),
        "sumber_pipeline_sha256": sumber.get("sumber_sha256"),
    }
    teks = laporan(recon)
    common.write_csv(paths.data / "walk_forward_oof_dev.csv", oof, OOF_COLS)
    paths.output.mkdir(parents=True, exist_ok=True)
    (paths.output / "walk_forward_rekonsiliasi.json").write_text(
        json.dumps(recon, indent=2, ensure_ascii=False) + "\n"
    )
    (paths.output / "laporan_walk_forward.md").write_text(teks)
    return recon


def ringkas(recon):
    skor, peringkat = recon["skor"], recon["peringkat"]
    lines = [
        f"Walk-forward LSTM: {len(recon['konfigurasi'])} konfigurasi × "
        f"{len(recon['tahun_validasi'])} lipatan, {recon['oof_n_per_konfigurasi']} event OOF "
        "per konfigurasi"
    ]
    for c in sorted(recon["konfigurasi"], key=lambda c: peringkat[c["id"]]):
        lines.append(f"  {peringkat[c['id']]:2d}. {c['id']:<8} log loss OOF {skor[c['id']]:.4f}")
    g = recon["banding_pembanding"]["gabungan"]
    lines.append(
        f"Terpilih: {recon['terpilih']['id']} | model − B0 {g['D_b0']:+.4f}, "
        f"model − B1 {g['D_b1']:+.4f}"
    )
    return "\n".join(lines)
