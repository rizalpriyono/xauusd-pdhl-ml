"""Tahap 5 (RM-26): kalibrasi probabilitas Temperature Scaling τ pada `calib` 2024 (227 event).

Memuat 5 checkpoint `.pt` + `scaler` beku dari RM-25 dalam mode `eval()` (tanpa melatih ulang
bobot model), menghitung rata-rata *logits* 5 seed pada data kalibrasi 2024, lalu mencari satu
skalar suhu `τ ∈ [0,05; 20,0]` memakai *golden-section search* (Kunci-1 `RENCANA_ANALISIS_RQ2.md`
§5) yang meminimalkan *log loss* kalibrasi: `Softmax(logits / τ)`.

Seluruh fungsi inti kalibrasi dan evaluasi reliabilitas memakai pustaka standar Python (stdlib)
sehingga deterministik dan dapat diuji penuh tanpa PyTorch.
"""
import json
import math

from pdhl_data import common
from pdhl_data.common import CLASSES

from . import arsitektur, fitur, pembanding

if CLASSES != ("Reversal", "Continuation", "Netral"):
    raise RuntimeError("kolom logit/p mengandaikan urutan kelas Kunci-1")

CALIB_PRED_COLS = [
    "event_id", "session_date", "level_type", "label", "b1_terpenuhi",
    "logit_rev", "logit_cont", "logit_net",
    "p_rev_raw", "p_cont_raw", "p_net_raw", "ll_raw",
    "p_rev_cal", "p_cont_cal", "p_net_cal", "ll_cal",
]
CATATAN = ("temperature scaling τ dicari dengan golden-section search pada rentang Kunci-1 "
           "[0,05; 20,0] di data calib 2024 tanpa melatih ulang bobot LSTM; B0 dan B1 akhir "
           "tidak dikalibrasi (RENCANA_ANALISIS_RQ2.md §2.1); data uji tidak dimuat.")


# ---------------------------------------------------------------- matematika stdlib
def softmax_row(logits, tau=1.0):
    """Softmax stabil secara numerik atas satu vektor logits dibagi suhu tau > 0."""
    t = float(tau)
    if not (math.isfinite(t) and t > 0):
        raise ValueError(f"tau harus finite dan > 0, didapat {tau}")
    scaled = [float(z) / t for z in logits]
    m = max(scaled)
    exps = [math.exp(v - m) for v in scaled]
    s = sum(exps)
    return [v / s for v in exps]


def log_loss_logits(logits_list, labels, tau, clip):
    """Rata-rata log loss Kunci-1 atas daftar logits pada suhu `tau`."""
    if not logits_list or len(logits_list) != len(labels):
        raise ValueError("logits_list dan labels harus tidak kosong dan sama panjang")
    total = 0.0
    for z, lab in zip(logits_list, labels):
        total += pembanding.log_loss(softmax_row(z, tau=tau), lab, clip)
    return total / len(labels)


def cari_tau_golden_section(
    logits_list,
    labels,
    rentang=(0.05, 20.0),
    clip=1e-15,
    tol=1e-10,
    max_iter=120,
):
    """Cari τ ∈ [lo, hi] yang meminimalkan log loss dengan metode golden-section (stdlib)."""
    lo, hi = float(rentang[0]), float(rentang[1])
    if not (math.isfinite(lo) and math.isfinite(hi) and 0 < lo < hi):
        raise ValueError(f"rentang tau tidak valid: {rentang}")

    def f(t):
        return log_loss_logits(logits_list, labels, t, clip)

    gr = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc, fd = f(c), f(d)
    iters = 0
    for iters in range(1, max_iter + 1):
        if b - a <= tol:
            break
        if fc <= fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = f(d)

    mid = (a + b) / 2.0
    kandidat = [(f(x), x) for x in (lo, a, mid, b, hi)]
    best_ll, best_tau = min(kandidat, key=lambda item: item[0])
    return {
        "tau": float(best_tau),
        "log_loss": float(best_ll),
        "iterasi": int(iters),
        "rentang": [lo, hi],
        "toleransi": float(tol),
        "metode": "golden_section",
    }


def _rerata(xs):
    return sum(xs) / len(xs) if xs else None


def brier_score(probs_list, labels):
    """Multi-class Brier score (rentang 0..2, sesuai RENCANA_ANALISIS_RQ2.md §2.3)."""
    if not probs_list:
        return None
    total = 0.0
    for p, lab in zip(probs_list, labels):
        yi = CLASSES.index(lab)
        total += sum((p[k] - (1.0 if k == yi else 0.0)) ** 2 for k in range(len(CLASSES)))
    return total / len(probs_list)


def _buat_bin_reliabilitas(pairs, n_bins):
    """Bangun 10 bin selebar 0,1 pada [0, 1] beserta ECE berbobot jumlah event."""
    n = len(pairs)
    bins = [[] for _ in range(n_bins)]
    for pred_p, y_bin in pairs:
        idx = min(max(int(pred_p * n_bins), 0), n_bins - 1)
        bins[idx].append((pred_p, y_bin))
    ece = 0.0
    detail = []
    for b in range(n_bins):
        items = bins[b]
        lo = round(b / n_bins, 4)
        hi = round((b + 1) / n_bins, 4)
        if items:
            mp = _rerata([x for x, _ in items])
            of = _rerata([y for _, y in items])
            ece += (len(items) / n) * abs(of - mp)
        else:
            mp, of = None, None
        detail.append({
            "bin_idx": b,
            "lo": lo,
            "hi": hi,
            "n": len(items),
            "mean_pred": mp,
            "obs_freq": of,
        })
    return ece, detail


def evaluasi_probabilitas(probs_list, labels, level_types, clip, n_bins=10):
    """Hitung log loss (total & PDH/PDL), Brier, ECE top-conf, ECE per-kelas, & tabel reliabilitas."""
    n = len(labels)
    if n == 0 or len(probs_list) != n or len(level_types) != n:
        raise ValueError("panjang probs_list, labels, dan level_types harus sama dan > 0")

    lls = [pembanding.log_loss(p, lab, clip) for p, lab in zip(probs_list, labels)]
    ll_per_level = {}
    for lv in ("PDH", "PDL"):
        sub = [ll for ll, t in zip(lls, level_types) if t == lv]
        ll_per_level[lv] = {"n": len(sub), "log_loss": _rerata(sub)}

    top_pairs = []
    for p, lab in zip(probs_list, labels):
        conf = max(p)
        pred_idx = p.index(conf)
        top_pairs.append((conf, 1.0 if pred_idx == CLASSES.index(lab) else 0.0))
    ece_top, rel_top = _buat_bin_reliabilitas(top_pairs, n_bins)

    ece_kelas, rel_kelas = {}, {}
    for k_idx, cls in enumerate(CLASSES):
        pairs = [(p[k_idx], 1.0 if lab == cls else 0.0) for p, lab in zip(probs_list, labels)]
        e_k, d_k = _buat_bin_reliabilitas(pairs, n_bins)
        ece_kelas[cls] = e_k
        rel_kelas[cls] = d_k

    return {
        "n": n,
        "log_loss": _rerata(lls),
        "brier": brier_score(probs_list, labels),
        "ece_top": ece_top,
        "ece_per_kelas": ece_kelas,
        "ece_per_kelas_rerata": _rerata(list(ece_kelas.values())),
        "log_loss_per_level": ll_per_level,
        "reliabilitas_top": rel_top,
        "reliabilitas_per_kelas": rel_kelas,
    }


# ---------------------------------------------------------------- validasi prasyarat
def cek_model_akhir_segar(paths):
    """Pastikan artefak RM-25 (5 checkpoint .pt + scaler + rekonsiliasi) utuh & cocok hash."""
    mp = paths.output / "model_akhir_rekonsiliasi.json"
    if not mp.exists():
        raise FileNotFoundError(f"{mp} belum ada; jalankan tahap `model-akhir` dulu")
    ma_recon = json.loads(mp.read_text())

    wf_p = paths.output / "walk_forward_rekonsiliasi.json"
    if not wf_p.exists() or ma_recon.get("sumber_sha256", {}).get(wf_p.name) != common.sha256_file(wf_p):
        raise ValueError(f"{mp.name} tidak sinkron dengan {wf_p.name}; jalankan ulang `model-akhir`")

    art = ma_recon["artefak"]
    scaler_rel = art["scaler"]["berkas"]
    scaler_path = paths.root / scaler_rel
    if not scaler_path.exists() or common.sha256_file(scaler_path) != art["scaler"]["sha256"]:
        raise ValueError(f"scaler {scaler_path} tidak ada atau SHA-256 berbeda dari RM-25")

    ckpt_paths = []
    for item in art["checkpoints"]:
        p = paths.root / item["berkas"]
        if not p.exists() or common.sha256_file(p) != item["sha256"]:
            raise ValueError(f"checkpoint {p} tidak ada atau SHA-256 berbeda dari RM-25")
        ckpt_paths.append(p)

    return ma_recon, common.sha256_file(mp), ckpt_paths, scaler_path


def _muat_pembanding_akhir(paths):
    pp = paths.output / "pembanding_rekonsiliasi.json"
    if not pp.exists():
        raise FileNotFoundError(f"{pp} belum ada; jalankan tahap `pembanding` dulu")
    return json.loads(pp.read_text()), common.sha256_file(pp)


def muat_data_calib(paths, lookback, theta):
    """Muat data kalibrasi 2024 lewat guard dan bangun sekuens 3D."""
    bars = fitur.prepare_bars(common.load_model_csv(paths.input / "bars_devcalib.csv"))
    events_calib = common.load_model_csv(paths.input / "events_calib.csv")
    labels_calib = common.load_model_csv(paths.input / "labels_calib.csv")
    meta_calib = common.load_model_csv(paths.data / "seq_meta_calib.csv")

    gab = pembanding.gabung(meta_calib, labels_calib, theta)
    if [e["event_id"] for e in events_calib] != [r["event_id"] for r in gab]:
        raise ValueError("urutan event_id events_calib.csv tidak cocok dengan seq_meta_calib.csv")

    seqs, _ = fitur.hitung_semua_sekuens(events_calib, bars, int(lookback), theta)
    rows = [{**r, "level_type": e["level_type"]} for r, e in zip(gab, events_calib)]
    return seqs, rows


# ---------------------------------------------------------------- laporan
def _f(x):
    return "–" if x is None else f"{x:.4f}"


def _d(x):
    if x is None:
        return "–"
    x = round(x, 4)
    return f"{x + 0.0:+.4f}" if x else "0.0000"


def laporan(recon):
    raw, cal = recon["metrik_raw"], recon["metrik_cal"]
    b0, b1 = recon["pembanding_calib"]["B0"], recon["pembanding_calib"]["B1"]
    opt = recon["optimasi_tau"]
    lines = [
        "# Laporan Kalibrasi Temperature Scaling LSTM (RM-26)", "",
        f"Sumber: `model.lstm.kalibrasi`, data kalibrasi 2024 (`n = {recon['n_calib']}` event). "
        f"Model akhir `{recon['terpilih']['id']}` dimuat dari 5 checkpoint `.pt` + scaler RM-25 "
        "dalam mode `eval()` tanpa melatih ulang bobot.", "",
        f"Catatan: {recon['catatan']}", "",
        "## 1. Hasil Pencarian Suhu `τ` (Golden-Section Search)", "",
        f"- **Rentang pencarian (Kunci-1):** `[{opt['rentang'][0]}, {opt['rentang'][1]}]` "
        f"(toleransi `{opt['toleransi']}`, `{opt['iterasi']}` iterasi).",
        f"- **Suhu optimal `τ*`:** **`{recon['tau']:.6f}`**",
        f"- **Efek pada Log-Loss 2024:** `{_f(raw['log_loss'])}` → **`{_f(cal['log_loss'])}`** "
        f"(selisih `{_d(cal['log_loss'] - raw['log_loss'])}`).", "",
        "## 2. Perbandingan Metrik Kalibrasi pada `calib` 2024 (Deskriptif)", "",
        "| Model / Tahap | Log-Loss | Brier | ECE (Top) | ECE Rev | ECE Cont | ECE Net | LL PDH | LL PDL |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for nama, m in (
        ("LSTM Raw (`τ = 1,0000`)", raw),
        (f"**LSTM Calibrated (`τ = {recon['tau']:.4f}`)**", cal),
        ("B0 akhir (`dev`, tanpa τ)", b0),
        ("B1 akhir (`dev`, tanpa τ)", b1),
    ):
        ek = m["ece_per_kelas"]
        lv = m["log_loss_per_level"]
        lines.append(
            f"| {nama} | {_f(m['log_loss'])} | {_f(m['brier'])} | {_f(m['ece_top'])} | "
            f"{_f(ek['Reversal'])} | {_f(ek['Continuation'])} | {_f(ek['Netral'])} | "
            f"{_f(lv['PDH']['log_loss'])} | {_f(lv['PDL']['log_loss'])} |"
        )

    lines += [
        "", "## 3. Tabel Reliabilitas 10-Bin (Top-Confidence, Sebelum vs Sesudah `τ`)", "",
        "| Bin | n (Raw) | Mean Pred (Raw) | Obs Acc (Raw) | n (Cal) | Mean Pred (Cal) | Obs Acc (Cal) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for br, bc in zip(raw["reliabilitas_top"], cal["reliabilitas_top"]):
        if br["n"] == 0 and bc["n"] == 0:
            continue
        lines.append(
            f"| [{br['lo']:.1f}, {br['hi']:.1f}] | {br['n']} | {_f(br['mean_pred'])} | "
            f"{_f(br['obs_freq'])} | {bc['n']} | {_f(bc['mean_pred'])} | {_f(bc['obs_freq'])} |"
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------- tahap
def run_kalibrasi(paths, kunci, sumber, *, prediksi_logits=None, device="cpu"):
    """Hitung logits model akhir pada calib 2024, cari τ optimal, evaluasi & tulis keluaran."""
    ma_recon, ma_sha, ckpt_paths, scaler_path = cek_model_akhir_segar(paths)
    pb_recon, pb_sha = _muat_pembanding_akhir(paths)
    terpilih = dict(ma_recon["terpilih"])

    seqs_calib, rows_calib = muat_data_calib(paths, terpilih["lookback_L"], kunci.theta)
    labels = [r["label"] for r in rows_calib]
    levels = [r["level_type"] for r in rows_calib]

    if prediksi_logits is None:
        bundle = arsitektur.muat_bundle(
            ckpt_paths, scaler_path, terpilih, kunci.lstm_teknis,
            seeds=kunci.seeds, device=device,
        )
        raw_logits_arr = arsitektur.prediksi_logits_lstm(bundle, seqs_calib)
        logits_list = [[float(v) for v in row] for row in raw_logits_arr]
    else:
        logits_list = [[float(v) for v in row] for row in prediksi_logits(seqs_calib)]

    if len(logits_list) != len(rows_calib):
        raise ValueError(f"jumlah baris logits ({len(logits_list)}) != calib ({len(rows_calib)})")

    opt = cari_tau_golden_section(
        logits_list, labels, rentang=kunci.temperature_rentang, clip=kunci.clip
    )
    tau = opt["tau"]

    probs_raw = [softmax_row(z, tau=1.0) for z in logits_list]
    probs_cal = [softmax_row(z, tau=tau) for z in logits_list]

    metrik_raw = evaluasi_probabilitas(probs_raw, labels, levels, kunci.clip, n_bins=kunci.ece_bin)
    metrik_cal = evaluasi_probabilitas(probs_cal, labels, levels, kunci.clip, n_bins=kunci.ece_bin)
    if metrik_cal["log_loss"] > metrik_raw["log_loss"] + 1e-12:
        raise RuntimeError(
            f"kalibrasi gagal: log loss sesudah tau ({metrik_cal['log_loss']:.6f}) "
            f"> sebelum tau ({metrik_raw['log_loss']:.6f})"
        )

    # Pembanding akhir B0 & B1 (dari dev 2016–2023, tanpa kalibrasi) pada calib 2024
    model_pb = pb_recon["akhir"]
    probs_b0 = [pembanding.prediksi(model_pb, r)["B0"] for r in rows_calib]
    probs_b1 = [pembanding.prediksi(model_pb, r)["B1"] for r in rows_calib]
    metrik_b0 = evaluasi_probabilitas(probs_b0, labels, levels, kunci.clip, n_bins=kunci.ece_bin)
    metrik_b1 = evaluasi_probabilitas(probs_b1, labels, levels, kunci.clip, n_bins=kunci.ece_bin)

    csv_rows = []
    for r, z, pr, pc in zip(rows_calib, logits_list, probs_raw, probs_cal):
        csv_rows.append({
            "event_id": r["event_id"],
            "session_date": r["session_date"],
            "level_type": r["level_type"],
            "label": r["label"],
            "b1_terpenuhi": r["b1"],
            "logit_rev": z[0],
            "logit_cont": z[1],
            "logit_net": z[2],
            "p_rev_raw": pr[0],
            "p_cont_raw": pr[1],
            "p_net_raw": pr[2],
            "ll_raw": pembanding.log_loss(pr, r["label"], kunci.clip),
            "p_rev_cal": pc[0],
            "p_cont_cal": pc[1],
            "p_net_cal": pc[2],
            "ll_cal": pembanding.log_loss(pc, r["label"], kunci.clip),
        })

    pred_path = paths.data / "calib_pred_lstm.csv"
    common.write_csv(pred_path, csv_rows, CALIB_PRED_COLS)

    recon = {
        "rm": "RM-26",
        "kelas": list(CLASSES),
        "terpilih": terpilih,
        "n_calib": len(rows_calib),
        "calib_event_id_sha256": common.sha256_ids([r["event_id"] for r in rows_calib]),
        "calib_rentang_tanggal": [rows_calib[0]["session_date"], rows_calib[-1]["session_date"]],
        "clip": kunci.clip,
        "ece_bin": kunci.ece_bin,
        "tau": tau,
        "optimasi_tau": opt,
        "metrik_raw": metrik_raw,
        "metrik_cal": metrik_cal,
        "perbaikan_log_loss": metrik_cal["log_loss"] - metrik_raw["log_loss"],
        "perbaikan_brier": metrik_cal["brier"] - metrik_raw["brier"],
        "perbaikan_ece_top": metrik_cal["ece_top"] - metrik_raw["ece_top"],
        "pembanding_calib": {"B0": metrik_b0, "B1": metrik_b1},
        "sumber_sha256": {
            "model_akhir_rekonsiliasi.json": ma_sha,
            "pembanding_rekonsiliasi.json": pb_sha,
            "seq_meta_calib.csv": common.sha256_file(paths.data / "seq_meta_calib.csv"),
            "labels_calib.csv": common.sha256_file(paths.input / "labels_calib.csv"),
            "calib_pred_lstm.csv": common.sha256_file(pred_path),
        },
        "catatan": CATATAN,
        "versi_pdhl_data": sumber.get("versi_pdhl_data"),
        "commit": sumber.get("commit"),
        "sumber_pipeline_sha256": sumber.get("sumber_sha256"),
    }

    teks = laporan(recon)
    paths.output.mkdir(parents=True, exist_ok=True)
    (paths.output / "kalibrasi_rekonsiliasi.json").write_text(
        json.dumps(recon, indent=2, ensure_ascii=False) + "\n"
    )
    (paths.output / "laporan_kalibrasi.md").write_text(teks)
    return recon


def ringkas(recon):
    raw, cal = recon["metrik_raw"], recon["metrik_cal"]
    b0, b1 = recon["pembanding_calib"]["B0"], recon["pembanding_calib"]["B1"]
    return "\n".join([
        f"Kalibrasi τ LSTM (calib 2024, n={recon['n_calib']}): "
        f"τ = {recon['tau']:.6f} ({recon['optimasi_tau']['iterasi']} iter golden-section)",
        f"  Log-Loss : raw {raw['log_loss']:.4f} -> cal {cal['log_loss']:.4f} "
        f"({recon['perbaikan_log_loss']:+.4f}) | B0 {b0['log_loss']:.4f}, B1 {b1['log_loss']:.4f}",
        f"  Brier/ECE: Brier {raw['brier']:.4f}->{cal['brier']:.4f} | "
        f"ECE(top) {raw['ece_top']:.4f}->{cal['ece_top']:.4f}",
    ])
