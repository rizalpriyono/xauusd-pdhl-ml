"""CLI: `python -m model.lstm <tahap> --config configs/<nama>.json`.

Tahap:
    fitur              tahap 1 (RM-22): 13 fitur sekuensial point-in-time (v2) + uji pemotongan
    pembanding         tahap 2 (RM-23): baseline non-model B0/B1 (akhir + OOF walk-forward)
    walk-forward       tahap 3 (RM-24): validasi walk-forward 10 konfigurasi LSTM
    model-akhir        tahap 4 (RM-25): latih pemenang RM-24 penuh pada seluruh dev (5 seed)
    kalibrasi          tahap 5 (RM-26): temperature scaling τ pada calib 2024 (golden-section)
    keputusan          tahap 6 (RM-27): aturan keputusan E_rev/E_cont (m_rev dev, δ=0,1R)
    kunci-2            tahap 7 (Kunci-2): bekukan seluruh komponen & segel manifest SHA-256
    verifikasi-kunci-2 audit integritas segel Kunci-2 terhadap MANIFEST_KUNCI_2_LSTM.json
    lanjutan           model-akhir -> kalibrasi -> keputusan -> kunci-2
    semua              fitur -> pembanding -> walk-forward -> model-akhir -> kalibrasi -> keputusan -> kunci-2

Sebelum tahap apa pun: berkas Kunci-1 dicek terhadap manifest, a/b/T/k konfigurasi dicek terhadap
Kunci-1, dan SHA-256 data masukan `runs/<nama>/data/` dicek terhadap `references/<nama>.json`.

Contoh:
    PYTHONPATH=src python3 -m model.lstm lanjutan --config configs/tesis.json --device cpu
    PYTHONPATH=src python3 -m model.lstm verifikasi-kunci-2 --config configs/tesis.json
Kode keluar: 0 = selesai, 1 = galat (pesan di stderr).
"""
import argparse
import sys
from pathlib import Path

from pdhl_data.common import TestDataLeak
from pdhl_data.config import ConfigError, load_config
from pdhl_data.pipeline import load_reference

from . import (
    fitur,
    jalur,
    kalibrasi,
    keputusan,
    kunci,
    kunci_2,
    laporan_fitur,
    model_akhir,
    pembanding,
    walk_forward,
)

URUTAN_AWAL = ("fitur", "pembanding", "walk-forward")
URUTAN_LANJUTAN = ("model-akhir", "kalibrasi", "keputusan", "kunci-2")
URUTAN = URUTAN_AWAL + URUTAN_LANJUTAN
TAHAP = URUTAN + ("verifikasi-kunci-2", "lanjutan", "semua")
GALAT = (kunci.KunciError, kunci_2.Kunci2Error, jalur.DataInputBerbeda, fitur.AtrKosong,
         fitur.LookAheadError, TestDataLeak, ConfigError, FileNotFoundError,
         ModuleNotFoundError, ValueError, RuntimeError)


def _repo_dir(config_path):
    return Path(config_path).resolve().parent.parent


def _jalankan(tahap, paths, k, sumber, device, cfg_name):
    if tahap == "fitur":
        recon = fitur.run_fitur(paths, k, sumber)
        laporan_fitur.tulis_laporan_fitur(paths)
        return fitur.ringkas(recon)
    if tahap == "pembanding":
        return pembanding.ringkas(pembanding.run_pembanding(paths, k, sumber))
    if tahap == "walk-forward":
        return walk_forward.ringkas(walk_forward.run_walk_forward(paths, k, sumber, device=device))
    if tahap == "model-akhir":
        return model_akhir.ringkas(model_akhir.run_model_akhir(paths, k, sumber, device=device))
    if tahap == "kalibrasi":
        return kalibrasi.ringkas(kalibrasi.run_kalibrasi(paths, k, sumber, device=device))
    if tahap == "keputusan":
        return keputusan.ringkas(keputusan.run_keputusan(paths, k, sumber))
    if tahap == "kunci-2":
        return kunci_2.ringkas(kunci_2.run_kunci_2(paths, k, sumber, cfg_name))
    return kunci_2.ringkas_verifikasi(kunci_2.verifikasi_kunci_2(paths, k, sumber))


def _sumber(args, cfg, paths):
    """Hash data masukan, dicocokkan dengan acuan. Acuan wajib ada kecuali `--tanpa-acuan`."""
    ref_path = args.reference or _repo_dir(args.config) / "references" / f"{cfg['name']}.json"
    reference = load_reference(ref_path)
    if reference is None:
        if not args.tanpa_acuan:
            raise FileNotFoundError(
                f"acuan {ref_path} tidak ada; data masukan tidak bisa dicek. "
                "Pakai --tanpa-acuan untuk konfigurasi tanpa acuan reproduksi"
            )
        print("[peringatan] --tanpa-acuan: hash data masukan hanya dicatat, tidak dicocokkan",
              file=sys.stderr)
    return jalur.info_sumber(jalur.cek_data_input(paths, reference))


def jalankan(args):
    cfg = load_config(args.config)
    k = kunci.muat_kunci(cfg, args.kunci_dir or kunci.kunci_dir_bawaan(args.config))
    paths = jalur.model_paths(cfg, args.runs_dir)
    if args.tahap == "semua":
        daftar = URUTAN
    elif args.tahap == "lanjutan":
        daftar = URUTAN_LANJUTAN
    else:
        daftar = (args.tahap,)
    print(f"[config] {cfg['name']}: Kunci-1 cocok dengan manifest ({len(k.manifest_sha256)} berkas), "
          f"theta B1 = {k.theta}, seeds = {k.seeds[0]}..{k.seeds[-1]}")
    sumber = _sumber(args, cfg, paths)
    for tahap in daftar:
        print(f"[{tahap}]", flush=True)
        print(_jalankan(tahap, paths, k, sumber, args.device, cfg["name"]), flush=True)
    print(f"[selesai] keluaran di {paths.root}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m model.lstm",
        description="Pemodelan LSTM PDH/PDL XAUUSD H1: RM-22 fitur s.d. Kunci-2 pembekuan pra-uji",
    )
    ap.add_argument("tahap", choices=TAHAP)
    ap.add_argument("--config", required=True, help="configs/<nama>.json")
    ap.add_argument("--runs-dir", type=Path, default=Path("runs"))
    ap.add_argument("--kunci-dir", type=Path, help="folder Kunci-1 (bawaan: <folder config>/kunci_1)")
    ap.add_argument("--reference", type=Path, help="acuan hash data (bawaan references/<nama>.json)")
    ap.add_argument("--tanpa-acuan", action="store_true",
                    help="izinkan berjalan bila acuan hash tidak ada (hash hanya dicatat)")
    ap.add_argument("--device", default="cpu", help="perangkat PyTorch: cpu, mps, atau cuda")
    args = ap.parse_args(argv)
    try:
        return jalankan(args)
    except GALAT as exc:
        print(f"[galat] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
