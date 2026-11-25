"""Tahap 7 (🔒 Kunci-2): pembekuan penuh komponen LSTM & segel manifest SHA-256 pra-uji.

Membekukan seluruh komponen yang dibutuhkan untuk eksekusi tunggal pada data uji:
- 5 checkpoint bobot PyTorch `.pt` (`L46_h32`, seed `20261119..20261123`)
- 1 `scaler_L46_h32.json` (`SequenceScaler3D` dari 85% awal `dev`)
- 1 skalar suhu kalibrasi `τ` (RM-26, `calib` 2024)
- parameter aturan keputusan `m_rev`, `m_cont`, `δ = 0,1R` & `delta_grid` (RM-27, `dev`)
- baseline B0 & B1 akhir (`dev` 2016–2023)
- hash SHA-256 skrip `src/model/lstm/*.py`, manifest Kunci-1, dan masukan `runs/<nama>/data/`

Seluruh berkas beku ditulis secara atomik ke `runs/<nama>/model/lstm/kunci_2/` dengan izin
hanya-baca (`0o444` untuk berkas, `0o555` untuk folder), identik dengan pola `pdhl_data/split.py`.
Menjalankan ulang `kunci-2` aman: bila manifest sudah ada, seluruh isi diverifikasi identik dan
tidak pernah ditimpa diam-diam (`Kunci2Error`). Data uji `kunci_uji/` tidak pernah dibuka.
"""
import datetime as dt
import json
import os
import shutil
import stat
from pathlib import Path

from pdhl_data import common

from . import fitur, kalibrasi, keputusan, model_akhir

READ_ONLY = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
READ_ONLY_DIR = READ_ONLY | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH

PACKAGE_DIR = Path(__file__).resolve().parent
MANIFEST_NAME = "MANIFEST_KUNCI_2_LSTM.json"
PARAM_NAME = "kunci_2_lstm.json"
CERT_NAME = "KUNCI-2_lstm_beku.md"
SCRIPTS = (
    "__init__.py", "__main__.py", "arsitektur.py", "fitur.py", "jalur.py",
    "kalibrasi.py", "keputusan.py", "kunci.py", "kunci_2.py", "laporan_fitur.py",
    "model_akhir.py", "pembanding.py", "walk_forward.py",
)
ATURAN = ("Pembekuan penuh komponen LSTM sebelum menyentuh data uji (Proposal Final v2, IV.II.11). "
          "Perubahan komponen apa pun setelah Kunci-2 dianggap pelanggaran segel.")


class Kunci2Error(RuntimeError):
    """Segel Kunci-2 rusak, berubah, tidak lengkap, atau gagal verifikasi rantai integritas."""


def _rmtree_force(path):
    for p in [path, *path.rglob("*")]:
        os.chmod(p, stat.S_IRWXU)
    shutil.rmtree(path)


def _baca_json_wajib(path, tahap):
    if not path.exists():
        raise FileNotFoundError(f"{path} belum ada; jalankan tahap `{tahap}` dulu")
    return json.loads(path.read_text())


def _audit_rantai_tahap(paths):
    """Verifikasi rantai integritas RM-22 -> RM-23 -> RM-24 -> RM-25 -> RM-26 -> RM-27."""
    f_recon = _baca_json_wajib(paths.output / "fitur_rekonsiliasi.json", "fitur")
    pb_recon = _baca_json_wajib(paths.output / "pembanding_rekonsiliasi.json", "pembanding")
    wf_recon, _ = model_akhir.cek_walk_forward_segar(paths)
    ma_recon, ma_sha, ckpt_paths, scaler_path = kalibrasi.cek_model_akhir_segar(paths)
    kal_recon, kal_sha = keputusan.cek_kalibrasi_segar(paths)
    kep_p = paths.output / "keputusan_rekonsiliasi.json"
    kep_recon = _baca_json_wajib(kep_p, "keputusan")

    if kep_recon.get("sumber_sha256", {}).get("kalibrasi_rekonsiliasi.json") != kal_sha:
        raise Kunci2Error("keputusan_rekonsiliasi.json tidak sinkron dengan RM-26; jalankan ulang `keputusan`")
    if kal_recon.get("sumber_sha256", {}).get("model_akhir_rekonsiliasi.json") != ma_sha:
        raise Kunci2Error("kalibrasi_rekonsiliasi.json tidak sinkron dengan RM-25; jalankan ulang `kalibrasi`")
    kep_csv = paths.data / "keputusan_calib_lstm.csv"
    if not kep_csv.exists() or kep_recon["sumber_sha256"].get(kep_csv.name) != common.sha256_file(kep_csv):
        raise Kunci2Error("keputusan_calib_lstm.csv berubah sejak RM-27; jalankan ulang `keputusan`")

    return {
        "fitur": f_recon,
        "pembanding": pb_recon,
        "walk_forward": wf_recon,
        "model_akhir": ma_recon,
        "kalibrasi": kal_recon,
        "keputusan": kep_recon,
        "ckpt_paths": ckpt_paths,
        "scaler_path": scaler_path,
    }


def _skrip_sha256():
    out = {}
    for name in SCRIPTS:
        p = PACKAGE_DIR / name
        if not p.exists():
            raise Kunci2Error(f"skrip {p} tidak ditemukan saat membuat segel Kunci-2")
        out[f"model/lstm/{name}"] = common.sha256_file(p)
    return out


def _artefak_tahap_sha256(paths):
    rel_files = (
        "data/seq_meta_dev.csv",
        "data/seq_meta_calib.csv",
        "data/pembanding_oof_dev.csv",
        "data/walk_forward_oof_dev.csv",
        "data/calib_pred_lstm.csv",
        "data/keputusan_calib_lstm.csv",
        "output/fitur_rekonsiliasi.json",
        "output/pembanding_rekonsiliasi.json",
        "output/walk_forward_rekonsiliasi.json",
        "output/model_akhir_rekonsiliasi.json",
        "output/kalibrasi_rekonsiliasi.json",
        "output/keputusan_rekonsiliasi.json",
    )
    out = {}
    for rel in rel_files:
        p = paths.root / rel
        if not p.exists():
            raise Kunci2Error(f"artefak tahap hilang: {p}")
        out[rel] = {"sha256": common.sha256_file(p), "bytes": p.stat().st_size}
    return out


def _bangun_param_beku(cfg_name, kunci, rantai, sumber):
    ma = rantai["model_akhir"]
    kal = rantai["kalibrasi"]
    kep = rantai["keputusan"]
    pb = rantai["pembanding"]
    par = kep["parameter_keputusan"]
    return {
        "konfigurasi": cfg_name,
        "jalur": "lstm",
        "aturan": ATURAN,
        "kunci_1_label": {"a": kunci.a, "b": kunci.b, "T": kunci.T, "k": kunci.k},
        "fitur": {
            "versi": rantai["fitur"]["versi_fitur"],
            "n_fitur": len(fitur.FEATURE_NAMES),
            "nama_fitur": list(fitur.FEATURE_NAMES),
            "skip_scaler_indices": list(ma["scaler_ringkasan"]["skip_indices"]),
            "clip_features": float(kunci.lstm_teknis["clip_features"]),
        },
        "model": {
            "terpilih": dict(ma["terpilih"]),
            "params_tetap": dict(ma["params_tetap"]),
            "seeds": list(ma["seeds"]),
            "n_dev_latih": int(ma["n_latih"]),
            "n_dev_purged": int(ma["n_purged"]),
            "split_internal": dict(ma["split_internal"]),
            "berkas_checkpoint": [p.name for p in rantai["ckpt_paths"]],
            "berkas_scaler": rantai["scaler_path"].name,
        },
        "kalibrasi": {
            "tau": float(kal["tau"]),
            "metode": kal["optimasi_tau"]["metode"],
            "rentang": list(kal["optimasi_tau"]["rentang"]),
            "n_calib": int(kal["n_calib"]),
            "log_loss_raw_2024": float(kal["metrik_raw"]["log_loss"]),
            "log_loss_cal_2024": float(kal["metrik_cal"]["log_loss"]),
            "ece_top_raw_2024": float(kal["metrik_raw"]["ece_top"]),
            "ece_top_cal_2024": float(kal["metrik_cal"]["ece_top"]),
        },
        "aturan_keputusan": {
            "m_rev": float(par["m_rev"]),
            "m_cont": float(par["m_cont"]),
            "n_netral_dev": int(par["n_netral_dev"]),
            "delta_utama": float(par["delta_utama"]),
            "delta_grid": list(par["delta_grid"]),
            "ambang_e_rev_reversal": float(par["ambang_e_rev_reversal"]),
            "ambang_e_rev_continuation": float(par["ambang_e_rev_continuation"]),
        },
        "pembanding_akhir_dev": pb["akhir"],
        "lingkungan": dict(ma["lingkungan"]),
        "versi_pdhl_data": sumber.get("versi_pdhl_data"),
        "commit": sumber.get("commit"),
    }


def _bangun_sertifikat_md(param, berkas_entries):
    m = param["model"]
    kal = param["kalibrasi"]
    ak = param["aturan_keputusan"]
    env = param["lingkungan"]
    lines = [
        f"# 🔒 Kunci-2: Pembekuan Penuh Model LSTM (`{param['konfigurasi']}`)", "",
        "Dokumen ini adalah sertifikat pembekuan komponen model LSTM sebelum menyentuh data uji "
        "`kunci_uji/` (2025–Jul 2026). Seluruh berkas di folder `kunci_2/` berstatus hanya-baca "
        "(`0o444`) dan disegel dengan SHA-256 pada `MANIFEST_KUNCI_2_LSTM.json`.", "",
        "## 1. Ringkasan Komponen Beku", "",
        f"- **Arsitektur & Konfigurasi (RM-24/25):** `{m['terpilih']['id']}` "
        f"(`lookback_L = {m['terpilih']['lookback_L']}`, `hidden_size = {m['terpilih']['hidden_size']}`, "
        f"`{param['fitur']['n_fitur']}` fitur v2, 5 seed `{m['seeds'][0]}..{m['seeds'][-1]}`).",
        f"- **Data latih penuh (RM-25):** `{m['n_dev_latih']}` event `dev` 2016–2023 "
        f"(purge `{m['n_dev_purged']}`, split 85/15 = `{m['split_internal']['n_in_train']}` / "
        f"`{m['split_internal']['n_early_stop']}`).",
        f"- **Kalibrasi Temperature Scaling (RM-26):** **`τ = {kal['tau']:.6f}`** "
        f"(`calib` 2024, `n = {kal['n_calib']}`; Log-Loss `{kal['log_loss_raw_2024']:.4f}` → "
        f"`{kal['log_loss_cal_2024']:.4f}`, ECE `{kal['ece_top_raw_2024']:.4f}` → "
        f"`{kal['ece_top_cal_2024']:.4f}`).",
        f"- **Aturan Keputusan (RM-27):** `m_rev = {ak['m_rev']:+.6f}R`, "
        f"`m_cont = {ak['m_cont']:+.6f}R` (`n = {ak['n_netral_dev']}` Netral `dev`), "
        f"`δ = {ak['delta_utama']:g}R` (`E_rev ≥ +{ak['ambang_e_rev_reversal']:g}R` → Reversal, "
        f"`E_rev ≤ {ak['ambang_e_rev_continuation']:g}R` → Continuation, di antaranya → Abstain `0R`).",
        f"- **Lingkungan:** Python `{env['python']}`, PyTorch `{env['torch']}`, "
        f"NumPy `{env['numpy']}` (`{env['platform']}`).",
        "- **Status data uji (`kunci_uji/`):** **BELUM DISENTUH / TETAP TERSEGEL.**", "",
        "## 2. Daftar Berkas Beku & Sidik Jari SHA-256", "",
        "| Berkas (`kunci_2/`) | Ukuran (byte) | SHA-256 |",
        "|---|---:|---|",
    ]
    for name in sorted(berkas_entries):
        e = berkas_entries[name]
        lines.append(f"| `{name}` | {e['bytes']} | `{e['sha256']}` |")
    lines += [
        "", "## 3. Cara Verifikasi Sebelum Eksekusi Tunggal Uji", "",
        "```bash",
        f"PYTHONPATH=src python3 -m model.lstm verifikasi-kunci-2 --config configs/{param['konfigurasi']}.json",
        "```", "",
    ]
    return "\n".join(lines)


def _tulis_atau_cek_segel(kunci2_dir, files_bytes, manifest_info):
    """Tulis folder `kunci_2/` sekali (atomik & hanya-baca). Bila sudah ada, verifikasi identik."""
    kunci2_dir = Path(kunci2_dir)
    manifest_path = kunci2_dir / MANIFEST_NAME
    entries = {
        name: {"sha256": common.sha256_bytes(raw), "bytes": len(raw)}
        for name, raw in files_bytes.items()
    }

    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        locked = old.get("berkas_beku", {})
        if {n: v["sha256"] for n, v in locked.items()} != {n: v["sha256"] for n, v in entries.items()}:
            raise Kunci2Error(
                f"Isi komponen LSTM berbeda dari segel {MANIFEST_NAME}. Kunci-2 tidak boleh "
                "ditimpa diam-diam; periksa perubahan pada RM-25..RM-27."
            )
        for name, meta in locked.items():
            p = kunci2_dir / name
            if not p.exists():
                raise Kunci2Error(f"berkas beku hilang dari {kunci2_dir}: {name}")
            if common.sha256_file(p) != meta["sha256"]:
                raise Kunci2Error(f"berkas beku {name} di {kunci2_dir} berubah sejak disegel")
        allowed = set(locked) | {MANIFEST_NAME}
        extra = sorted(p.name for p in kunci2_dir.iterdir() if p.name not in allowed)
        if extra:
            raise Kunci2Error(f"berkas asing di {kunci2_dir}: {', '.join(extra)}")
        os.chmod(kunci2_dir, READ_ONLY_DIR)
        return old, False

    if kunci2_dir.exists() and any(kunci2_dir.iterdir()):
        raise Kunci2Error(f"{kunci2_dir} tidak kosong tetapi tidak punya {MANIFEST_NAME}")

    tmp = kunci2_dir.with_name(kunci2_dir.name + ".tmp")
    if tmp.exists():
        _rmtree_force(tmp)
    tmp.mkdir(parents=True)
    for name, raw in files_bytes.items():
        (tmp / name).write_bytes(raw)
    manifest = {
        "dikunci_pada": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "berkas_beku": entries,
        **manifest_info,
    }
    (tmp / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    for p in tmp.iterdir():
        os.chmod(p, READ_ONLY)
    if kunci2_dir.exists():
        kunci2_dir.rmdir()
    tmp.rename(kunci2_dir)
    os.chmod(kunci2_dir, READ_ONLY_DIR)
    return manifest, True


# ---------------------------------------------------------------- tahap & verifikasi
def run_kunci_2(paths, kunci, sumber, cfg_name):
    """Audit rantai RM-22..RM-27, bekukan semua komponen LSTM, dan segel MANIFEST_KUNCI_2_LSTM."""
    rantai = _audit_rantai_tahap(paths)
    param = _bangun_param_beku(cfg_name, kunci, rantai, sumber)

    files_bytes = {}
    for p in rantai["ckpt_paths"]:
        files_bytes[p.name] = p.read_bytes()
    files_bytes[rantai["scaler_path"].name] = rantai["scaler_path"].read_bytes()
    files_bytes[PARAM_NAME] = (json.dumps(param, indent=2, ensure_ascii=False) + "\n").encode()

    prelim_entries = {
        name: {"sha256": common.sha256_bytes(raw), "bytes": len(raw)}
        for name, raw in files_bytes.items()
    }
    files_bytes[CERT_NAME] = _bangun_sertifikat_md(param, prelim_entries).encode()

    manifest_info = {
        "rm": "Kunci-2",
        "konfigurasi": cfg_name,
        "jalur": "lstm",
        "aturan": ATURAN,
        "terpilih": dict(rantai["model_akhir"]["terpilih"]),
        "tau": float(rantai["kalibrasi"]["tau"]),
        "m_rev": float(rantai["keputusan"]["parameter_keputusan"]["m_rev"]),
        "m_cont": float(rantai["keputusan"]["parameter_keputusan"]["m_cont"]),
        "delta_utama": float(kunci.delta_utama),
        "artefak_tahap": _artefak_tahap_sha256(paths),
        "kunci_1_sha256": dict(kunci.manifest_sha256),
        "sumber_pipeline_sha256": dict(sumber.get("sumber_sha256", {})),
        "skrip_lstm_sha256": _skrip_sha256(),
        "lingkungan": dict(rantai["model_akhir"]["lingkungan"]),
        "versi_pdhl_data": sumber.get("versi_pdhl_data"),
        "git": sumber.get("commit"),
    }

    manifest, baru = _tulis_atau_cek_segel(paths.kunci_2, files_bytes, manifest_info)
    return {"manifest": manifest, "baru_disegel": baru, "dir": str(paths.kunci_2)}


def verifikasi_kunci_2(paths, kunci, sumber=None):
    """Verifikasi 100% integritas segel Kunci-2 pada disk terhadap MANIFEST_KUNCI_2_LSTM.json."""
    kunci2_dir = paths.kunci_2
    mp = kunci2_dir / MANIFEST_NAME
    if not mp.exists():
        raise Kunci2Error(f"{mp} tidak ditemukan; jalankan tahap `kunci-2` dulu")
    manifest = json.loads(mp.read_text())

    locked = manifest.get("berkas_beku", {})
    if not locked:
        raise Kunci2Error(f"{mp}: bagian `berkas_beku` kosong")

    beda_beku = []
    for name, meta in locked.items():
        p = kunci2_dir / name
        if not p.exists():
            beda_beku.append(f"{name} (hilang)")
        elif common.sha256_file(p) != meta["sha256"] or p.stat().st_size != meta["bytes"]:
            beda_beku.append(name)
    if beda_beku:
        raise Kunci2Error(f"berkas beku Kunci-2 tidak cocok manifest: {', '.join(beda_beku)}")

    allowed = set(locked) | {MANIFEST_NAME}
    extra = sorted(p.name for p in kunci2_dir.iterdir() if p.name not in allowed)
    if extra:
        raise Kunci2Error(f"ditemukan berkas asing di {kunci2_dir}: {', '.join(extra)}")

    if manifest.get("kunci_1_sha256") != kunci.manifest_sha256:
        raise Kunci2Error("hash Kunci-1 di manifest Kunci-2 berbeda dari configs/kunci_1/ saat ini")

    if sumber is not None:
        if manifest.get("sumber_pipeline_sha256") != sumber.get("sumber_sha256"):
            raise Kunci2Error("hash data pipeline di manifest Kunci-2 berbeda dari runs/<cfg>/data/")

    skrip_now = _skrip_sha256()
    if manifest.get("skrip_lstm_sha256") != skrip_now:
        beda_skrip = [k for k, v in skrip_now.items() if manifest.get("skrip_lstm_sha256", {}).get(k) != v]
        raise Kunci2Error(f"skrip LSTM berubah sejak Kunci-2 disegel: {', '.join(beda_skrip)}")

    return {
        "ok": True,
        "dikunci_pada": manifest["dikunci_pada"],
        "n_berkas_beku": len(locked),
        "n_skrip": len(skrip_now),
        "terpilih": manifest["terpilih"]["id"],
        "tau": manifest["tau"],
        "m_rev": manifest["m_rev"],
        "delta_utama": manifest["delta_utama"],
        "manifest_sha256": common.sha256_file(mp),
    }


def ringkas(res):
    m = res["manifest"]
    status = "BARU DISEGEL (read-only)" if res["baru_disegel"] else "TERVERIFIKASI UTUH (sudah disegel)"
    return "\n".join([
        f"Kunci-2 LSTM [{status}] di {res['dir']}",
        f"  Dikunci pada : {m['dikunci_pada']} | {len(m['berkas_beku'])} berkas beku + {MANIFEST_NAME}",
        f"  Komponen     : {m['terpilih']['id']} (5 .pt + scaler), τ = {m['tau']:.6f}, "
        f"m_rev = {m['m_rev']:+.4f}R, δ = {m['delta_utama']:g}R",
    ])


def ringkas_verifikasi(res):
    return (
        f"Verifikasi Kunci-2 LSTM: OK ({res['n_berkas_beku']}/{res['n_berkas_beku']} berkas beku, "
        f"{res['n_skrip']}/{res['n_skrip']} skrip, Kunci-1 & data masukan cocok) | "
        f"{res['terpilih']}, τ={res['tau']:.6f}, m_rev={res['m_rev']:+.4f}R, δ={res['delta_utama']:g}R | "
        f"manifest SHA-256 {res['manifest_sha256'][:16]}…"
    )
