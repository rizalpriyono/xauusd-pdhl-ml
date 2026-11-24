"""CLI: `python -m model.gradient_boosting <tahap> --config configs/<nama>.json`.

Tahap:
    fitur          tahap 1: 15 fitur point-in-time + laporan fitur
    pembanding     tahap 2: baseline non-model B0/B1 (akhir + OOF walk-forward)
    walk-forward   tahap 3: validasi walk-forward 8 konfigurasi LightGBM dan pemilihan konfigurasi
    bandingkan     selisih rekonsiliasi baru terhadap hasil lama (references/gradient_boosting_lama/)
    semua          fitur -> pembanding -> walk-forward -> bandingkan

Sebelum tahap apa pun: berkas Kunci-1 dicek terhadap manifest, a/b/T/k konfigurasi dicek terhadap
Kunci-1, dan SHA-256 data masukan `runs/<nama>/data/` dicek terhadap `references/<nama>.json`.

Contoh:
    PYTHONPATH=src python3 -m model.gradient_boosting semua --config configs/tesis.json
    PYTHONPATH=src python3 -m model.gradient_boosting fitur --config configs/tesis.json
Kode keluar: 0 = selesai, 1 = galat (pesan di stderr).
"""
import argparse
import sys
from pathlib import Path

from pdhl_data.common import TestDataLeak
from pdhl_data.config import ConfigError, load_config
from pdhl_data.pipeline import load_reference

from . import bandingkan, fitur, jalur, kunci, laporan_fitur, pembanding, walk_forward

URUTAN = ("fitur", "pembanding", "walk-forward", "bandingkan")
TAHAP = URUTAN + ("semua",)
GALAT = (kunci.KunciError, jalur.DataInputBerbeda, fitur.AtrKosong, fitur.LookAheadError,
         bandingkan.AcuanLamaRusak, TestDataLeak, ConfigError, FileNotFoundError,
         ModuleNotFoundError, ValueError)


def _repo_dir(config_path):
    return Path(config_path).resolve().parent.parent


def _jalankan(tahap, paths, k, sumber, acuan_lama, cfg_name):
    if tahap == "fitur":
        recon = fitur.run_fitur(paths, k, sumber)
        laporan_fitur.tulis_laporan_fitur(paths)
        return fitur.ringkas(recon)
    if tahap == "pembanding":
        return pembanding.ringkas(pembanding.run_pembanding(paths, k, sumber))
    if tahap == "walk-forward":
        return walk_forward.ringkas(walk_forward.run_walk_forward(paths, k, sumber))
    return bandingkan.ringkas(bandingkan.run_bandingkan(paths, acuan_lama, cfg_name))


def _sumber(args, cfg, paths):
    """Hash data masukan, dicocokkan dengan acuan. Acuan wajib ada kecuali `--tanpa-acuan`."""
    ref_path = args.reference or _repo_dir(args.config) / "references" / f"{cfg['name']}.json"
    reference = load_reference(ref_path)
    if reference is None:
        if not args.tanpa_acuan:
            raise FileNotFoundError(f"acuan {ref_path} tidak ada; data masukan tidak bisa dicek. "
                                    "Pakai --tanpa-acuan untuk konfigurasi tanpa acuan reproduksi")
        print("[peringatan] --tanpa-acuan: hash data masukan hanya dicatat, tidak dicocokkan",
              file=sys.stderr)
    return jalur.info_sumber(jalur.cek_data_input(paths, reference))


def jalankan(args):
    cfg = load_config(args.config)
    k = kunci.muat_kunci(cfg, args.kunci_dir or kunci.kunci_dir_bawaan(args.config))
    paths = jalur.model_paths(cfg, args.runs_dir)
    daftar = URUTAN if args.tahap == "semua" else (args.tahap,)
    print(f"[config] {cfg['name']}: Kunci-1 cocok dengan manifest ({len(k.manifest_sha256)} berkas), "
          f"theta B1 = {k.theta}")
    sumber = _sumber(args, cfg, paths) if set(daftar) - {"bandingkan"} else None
    acuan_lama = args.acuan_lama or _repo_dir(args.config) / "references" / "gradient_boosting_lama"
    for tahap in daftar:
        print(f"[{tahap}]")
        print(_jalankan(tahap, paths, k, sumber, acuan_lama, cfg["name"]))
    print(f"[selesai] keluaran di {paths.root}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m model.gradient_boosting",
                                 description="Pemodelan Gradient Boosting PDH/PDL: fitur, baseline, "
                                             "walk-forward")
    ap.add_argument("tahap", choices=TAHAP)
    ap.add_argument("--config", required=True, help="configs/<nama>.json")
    ap.add_argument("--runs-dir", type=Path, default=Path("runs"))
    ap.add_argument("--kunci-dir", type=Path, help="folder Kunci-1 (bawaan: <folder config>/kunci_1)")
    ap.add_argument("--reference", type=Path, help="acuan hash data (bawaan references/<nama>.json)")
    ap.add_argument("--tanpa-acuan", action="store_true",
                    help="izinkan berjalan bila acuan hash tidak ada (hash hanya dicatat)")
    ap.add_argument("--acuan-lama", type=Path,
                    help="hasil lama untuk `bandingkan` (bawaan references/gradient_boosting_lama)")
    args = ap.parse_args(argv)
    try:
        return jalankan(args)
    except GALAT as exc:
        print(f"[galat] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
