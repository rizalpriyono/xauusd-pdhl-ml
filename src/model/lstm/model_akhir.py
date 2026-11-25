"""Tahap 4 (RM-25): pelatihan model akhir LSTM pada seluruh data pengembangan (2016–2023).

Tidak menggunakan walk-forward lagi: hanya konfigurasi pemenang dari RM-24 (`L46_h32`) yang
dilatih penuh satu kali pada seluruh event `dev` (setelah purging batas dev–kalibrasi) lintas 5
seed Kunci-1 (`20261119..20261123`), lalu disimpan sebagai 5 checkpoint `.pt` + 1 `scaler` JSON.

Data kalibrasi (2024) dan data uji (`kunci_uji/`) tidak dimuat sama sekali pada tahap ini.
"""
import datetime as dt
import json
import platform

from pdhl_data import common
from pdhl_data.common import CLASSES, DEV_END

from . import arsitektur, fitur, pembanding

CATATAN = ("model akhir dilatih sekali pada seluruh dev 2016–2023 (tanpa walk-forward) memakai "
           "konfigurasi terpilih RM-24; split internal 85% latih + 15% early stopping kronologis "
           "identik dengan RM-24; data calib 2024 dan kunci_uji tidak dimuat.")


# ---------------------------------------------------------------- validasi prasyarat
def cek_walk_forward_segar(paths):
    """Pastikan hasil RM-24 ada dan dihitung dari berkas dev/pembanding yang sekarang."""
    wp = paths.output / "walk_forward_rekonsiliasi.json"
    if not wp.exists():
        raise FileNotFoundError(f"{wp} belum ada; jalankan tahap `walk-forward` dulu")
    wf_recon = json.loads(wp.read_text())
    dicatat = wf_recon.get("sumber_sha256", {})
    cek_berkas = (
        paths.data / "seq_meta_dev.csv",
        paths.input / "labels_dev.csv",
        paths.data / "pembanding_oof_dev.csv",
    )
    for p in cek_berkas:
        if not p.exists():
            raise FileNotFoundError(f"{p} belum ada; jalankan tahap prasyarat lebih dulu")
        if dicatat.get(p.name) != common.sha256_file(p):
            raise ValueError(
                f"{wp.name} tidak dihitung dari {p.name} yang sekarang; jalankan ulang `walk-forward`"
            )
    terpilih = wf_recon.get("terpilih")
    if not isinstance(terpilih, dict) or not {"id", "lookback_L", "hidden_size"} <= set(terpilih):
        raise ValueError(f"{wp.name}: bagian `terpilih` tidak lengkap")
    return wf_recon, common.sha256_file(wp)


def muat_data_dev(paths, lookback, theta):
    """Muat seluruh event dev via guard, terapkan purging batas dev–calib, dan bangun sekuens 3D."""
    bars = fitur.prepare_bars(common.load_model_csv(paths.input / "bars_devcalib.csv"))
    events_dev = common.load_model_csv(paths.input / "events_dev.csv")
    labels_dev = common.load_model_csv(paths.input / "labels_dev.csv")
    meta_dev = common.load_model_csv(paths.data / "seq_meta_dev.csv")

    gab = pembanding.gabung(meta_dev, labels_dev, theta)
    if [e["event_id"] for e in events_dev] != [r["event_id"] for r in gab]:
        raise ValueError("urutan event_id events_dev.csv tidak cocok dengan seq_meta_dev.csv")

    gab_dengan_event = [{**r, "_event": e} for r, e in zip(gab, events_dev)]
    batas_calib = DEV_END + dt.timedelta(days=1)
    latih_rows, purged_rows = pembanding.data_latih(gab_dengan_event, batas_calib)
    if not latih_rows:
        raise ValueError("tidak ada event dev yang lolos purging untuk melatih model akhir")

    seqs, _ = fitur.hitung_semua_sekuens(
        [r["_event"] for r in latih_rows], bars, int(lookback), theta
    )
    X_tr = seqs
    y_tr = [CLASSES.index(r["label"]) for r in latih_rows]
    return {
        "X_tr": X_tr,
        "y_tr": y_tr,
        "latih_rows": latih_rows,
        "purged_rows": purged_rows,
    }


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


def _max_abs_diff(a, b):
    return max(abs(float(x) - float(y)) for ra, rb in zip(a, b) for x, y in zip(ra, rb))


# ---------------------------------------------------------------- laporan
def laporan(recon):
    cfg = recon["terpilih"]
    env = recon["lingkungan"]
    sp = recon["split_internal"]
    lines = [
        "# Laporan Model Akhir LSTM (RM-25)", "",
        "Sumber: `model.lstm.model_akhir`, seluruh data pengembangan `dev` (2016–2023) saja. "
        f"Lingkungan: Python {env['python']}, PyTorch {env['torch']}, NumPy {env['numpy']} "
        f"({env['platform']}).", "",
        f"Catatan: {recon['catatan']}", "",
        "## 1. Konfigurasi & Pembagian Internal", "",
        f"- **Konfigurasi pemenang RM-24:** `{cfg['id']}` "
        f"(`lookback_L = {cfg['lookback_L']}`, `hidden_size = {cfg['hidden_size']}`, "
        f"13 fitur sekuensial v2).",
        f"- **Event dev lolos purging batas 2024-01-01:** `{recon['n_latih']}` "
        f"(di-purge: `{recon['n_purged']}`).",
        f"- **Split kronologis 85% / 15%:** `{sp['n_in_train']}` event awal untuk pembaruan bobot "
        f"& `fit` `SequenceScaler3D` ({sp['in_train_rentang'][0]} s.d. {sp['in_train_rentang'][1]}), "
        f"`{sp['n_early_stop']}` event ujung untuk *early stopping* "
        f"({sp['early_stop_rentang'][0]} s.d. {sp['early_stop_rentang'][1]}).",
        f"- **Uji round-trip reload disk:** selisih maksimum logits = "
        f"`{recon['uji_roundtrip']['max_abs_diff_logits']:.2e}` "
        f"(diuji pada {recon['uji_roundtrip']['n_event_uji']} event — **LULUS**).", "",
        "## 2. Ringkasan 5 Seed & Checkpoint Tersimpan", "",
        "| Seed | Best Epoch | Best Early-Stop Log Loss | Berkas Checkpoint | Ukuran (byte) | SHA-256 (16 hex) |",
        "|---:|---:|---:|---|---:|---|",
    ]
    for s in recon["ringkasan_seed"]:
        lines.append(
            f"| `{s['seed']}` | {s['best_epoch']} | {s['best_es_loss']:.4f} | "
            f"`{s['berkas']}` | {s['bytes']} | `{s['sha256'][:16]}…` |"
        )
    sc = recon["artefak"]["scaler"]
    lines += [
        "",
        f"- **Rata-rata best epoch:** `{recon['avg_best_epoch']:.1f}` · "
        f"**Rata-rata early-stop log loss:** `{recon['avg_best_es_loss']:.4f}`.",
        f"- **Berkas scaler bersama:** `{sc['berkas']}` ({sc['bytes']} byte, "
        f"SHA-256 `{sc['sha256'][:16]}…`), dilewati pada indeks siklik `{recon['scaler_ringkasan']['skip_indices']}`.",
        "",
        "## 3. Statistik `SequenceScaler3D` (di-fit hanya pada 85% awal `dev`)", "",
        "| Indeks | Fitur | Mode | Mean | Std |",
        "|---:|---|---|---:|---:|",
    ]
    skip = set(recon["scaler_ringkasan"]["skip_indices"])
    for i, (nama, mu, sd) in enumerate(zip(
        recon["scaler_ringkasan"]["feature_names"],
        recon["scaler_ringkasan"]["mean"],
        recon["scaler_ringkasan"]["std"],
    )):
        mode = "Skip (siklik)" if i in skip else "z-score + clip ±10"
        lines.append(f"| {i} | `{nama}` | {mode} | {mu:.4f} | {sd:.4f} |")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------- tahap
def run_model_akhir(
    paths,
    kunci,
    sumber,
    *,
    latih=None,
    simpan=None,
    muat=None,
    prediksi_logits=None,
    device="cpu",
    verbose=True,
):
    """Latih pemenang RM-24 (5 seed) pada seluruh dev, simpan 5 .pt + scaler, dan audit round-trip."""
    wf_recon, wf_sha = cek_walk_forward_segar(paths)
    terpilih = dict(wf_recon["terpilih"])
    data = muat_data_dev(paths, terpilih["lookback_L"], kunci.theta)
    X_tr, y_tr, latih_rows = data["X_tr"], data["y_tr"], data["latih_rows"]

    latih_fn = latih or (
        lambda X, y, c, t: arsitektur.latih_lstm(
            X, y, c, t, seeds=kunci.seeds, clip=kunci.clip, device=device
        )
    )
    simpan_fn = simpan or (
        lambda b, d, n_fit: arsitektur.simpan_bundle(
            b, d, n_fit_events=n_fit, feature_names=fitur.FEATURE_NAMES
        )
    )
    muat_fn = muat or (
        lambda cpaths, spath, c, t: arsitektur.muat_bundle(
            cpaths, spath, c, t, seeds=kunci.seeds, device=device
        )
    )
    logits_fn = prediksi_logits or arsitektur.prediksi_logits_lstm

    if verbose:
        print(f"  [lstm-akhir] melatih {terpilih['id']} pada {len(X_tr)} event dev "
              f"× {len(kunci.seeds)} seed ({kunci.seeds[0]}..{kunci.seeds[-1]})...", flush=True)

    bundle = latih_fn(X_tr, y_tr, terpilih, kunci.lstm_teknis)
    members = bundle["members"]
    if len(members) != len(kunci.seeds):
        raise ValueError(f"jumlah seed terlatih ({len(members)}) != Kunci-1 ({len(kunci.seeds)})")

    n_in_train = int(members[0]["n_in_train"])
    n_early_stop = int(members[0]["n_early_stop"])
    if n_in_train + n_early_stop != len(X_tr):
        raise ValueError("split internal 85/15 tidak menjumlah ke n_latih")

    saved = simpan_fn(bundle, paths.checkpoints, n_in_train)
    scaler_path = saved["scaler_path"]
    ckpt_paths = saved["checkpoint_paths"]
    if len(ckpt_paths) != len(members):
        raise ValueError("jumlah berkas checkpoint tidak sama dengan jumlah seed")

    # Uji round-trip reload dari disk agar jaminan reproduksi terverifikasi langsung
    n_uji = min(32, len(X_tr))
    X_uji = X_tr[:n_uji]
    logits_mem = logits_fn(bundle, X_uji)
    reloaded = muat_fn(ckpt_paths, scaler_path, terpilih, kunci.lstm_teknis)
    logits_disk = logits_fn(reloaded, X_uji)
    max_diff = _max_abs_diff(logits_mem, logits_disk)
    if max_diff > 1e-12:
        raise RuntimeError(
            f"uji round-trip reload checkpoint gagal: selisih logits {max_diff:.3e} > 1e-12"
        )

    scaler_json = json.loads(scaler_path.read_text())
    ringkasan_seed = []
    ckpt_artefak = []
    for m, p in zip(members, ckpt_paths):
        sha = common.sha256_file(p)
        sz = p.stat().st_size
        rel = f"checkpoints/{p.name}"
        ringkasan_seed.append({
            "seed": int(m["seed"]),
            "best_epoch": int(m["best_epoch"]),
            "best_es_loss": float(m["best_es_loss"]),
            "berkas": rel,
            "bytes": sz,
            "sha256": sha,
        })
        ckpt_artefak.append({"seed": int(m["seed"]), "berkas": rel, "bytes": sz, "sha256": sha})

    scaler_artefak = {
        "berkas": f"checkpoints/{scaler_path.name}",
        "bytes": scaler_path.stat().st_size,
        "sha256": common.sha256_file(scaler_path),
    }

    recon = {
        "rm": "RM-25",
        "kelas": list(CLASSES),
        "fitur_sekuensial": list(fitur.FEATURE_NAMES),
        "lingkungan": _lingkungan(),
        "terpilih": terpilih,
        "params_tetap": dict(kunci.lstm_teknis),
        "seeds": list(kunci.seeds),
        "device": device,
        "clip": kunci.clip,
        "n_latih": len(latih_rows),
        "n_purged": len(data["purged_rows"]),
        "dev_event_id_sha256": common.sha256_ids([r["event_id"] for r in latih_rows]),
        "dev_rentang_tanggal": [latih_rows[0]["session_date"], latih_rows[-1]["session_date"]],
        "split_internal": {
            "early_stop_frac": float(kunci.lstm_teknis["early_stop_frac"]),
            "n_in_train": n_in_train,
            "n_early_stop": n_early_stop,
            "in_train_rentang": [
                latih_rows[0]["session_date"],
                latih_rows[n_in_train - 1]["session_date"],
            ],
            "early_stop_rentang": [
                latih_rows[n_in_train]["session_date"],
                latih_rows[-1]["session_date"],
            ],
        },
        "ringkasan_seed": ringkasan_seed,
        "avg_best_epoch": sum(s["best_epoch"] for s in ringkasan_seed) / len(ringkasan_seed),
        "avg_best_es_loss": sum(s["best_es_loss"] for s in ringkasan_seed) / len(ringkasan_seed),
        "scaler_ringkasan": {
            "feature_names": scaler_json["feature_names"],
            "skip_indices": scaler_json["skip_indices"],
            "clip_val": scaler_json["clip_val"],
            "mean": scaler_json["mean"],
            "std": scaler_json["std"],
        },
        "artefak": {
            "checkpoints": ckpt_artefak,
            "scaler": scaler_artefak,
        },
        "uji_roundtrip": {
            "n_event_uji": n_uji,
            "max_abs_diff_logits": max_diff,
            "lulus": True,
        },
        "sumber_sha256": {
            "walk_forward_rekonsiliasi.json": wf_sha,
            "seq_meta_dev.csv": common.sha256_file(paths.data / "seq_meta_dev.csv"),
            "labels_dev.csv": common.sha256_file(paths.input / "labels_dev.csv"),
        },
        "catatan": CATATAN,
        "versi_pdhl_data": sumber.get("versi_pdhl_data"),
        "commit": sumber.get("commit"),
        "sumber_pipeline_sha256": sumber.get("sumber_sha256"),
    }

    teks = laporan(recon)
    paths.output.mkdir(parents=True, exist_ok=True)
    (paths.output / "model_akhir_rekonsiliasi.json").write_text(
        json.dumps(recon, indent=2, ensure_ascii=False) + "\n"
    )
    (paths.output / "laporan_model_akhir.md").write_text(teks)
    return recon


def ringkas(recon):
    cfg = recon["terpilih"]
    sp = recon["split_internal"]
    eps = ", ".join(str(s["best_epoch"]) for s in recon["ringkasan_seed"])
    return "\n".join([
        f"Model akhir LSTM: {cfg['id']} dilatih penuh pada {recon['n_latih']} event dev "
        f"(purge {recon['n_purged']}; split 85/15 = {sp['n_in_train']}/{sp['n_early_stop']})",
        f"  5 seed best_epoch : [{eps}] (rata-rata {recon['avg_best_epoch']:.1f}, "
        f"ES log loss {recon['avg_best_es_loss']:.4f})",
        f"  Tersimpan         : {len(recon['artefak']['checkpoints'])} checkpoint .pt + "
        f"{recon['artefak']['scaler']['berkas']} | round-trip diff "
        f"{recon['uji_roundtrip']['max_abs_diff_logits']:.2e}",
    ])
