"""CLI: `python -m pdhl_data {run,verify} --config configs/<nama>.json`.

Contoh:
    python -m pdhl_data run --config configs/tesis.json
    python -m pdhl_data run --config configs/tesis.json --steps labels
    python -m pdhl_data verify --config configs/tesis.json
"""
import argparse
import sys
from pathlib import Path

from .config import load_config
from .pipeline import STEPS, check_raw, load_reference, run, run_paths, verify

DEFAULT_RAW = Path("data/raw/XAUUSD_H1.csv")


def _reference_path(args, cfg):
    if args.reference:
        return Path(args.reference)
    return Path(args.config).resolve().parent.parent / "references" / f"{cfg['name']}.json"


def _print_verify(results):
    width = max(len(name) for name, _, _ in results)
    for name, status, note in results:
        print(f"  {status:6} {name:<{width}}  {note}")
    bad = [r for r in results if r[1] != "OK"]
    print(f"[verify] {len(results) - len(bad)}/{len(results)} OK")
    return not bad


def _cmd_verify(args, cfg):
    """Kode keluar: 0 = semua OK, 1 = ada BEDA/HILANG, 2 = berkas acuan tidak ada."""
    ref_path = _reference_path(args, cfg)
    reference = load_reference(ref_path)
    if reference is None:
        print(f"[verify] acuan {ref_path} tidak ada; tidak ada yang diverifikasi", file=sys.stderr)
        return 2
    return 0 if _print_verify(verify(run_paths(cfg, args.runs_dir), reference)) else 1


def _cmd_run(args, cfg):
    steps = STEPS if args.steps == "all" else tuple(s.strip() for s in args.steps.split(","))
    reference = load_reference(_reference_path(args, cfg))
    if {"events", "split"} & set(steps):          # tahap labels saja tidak membaca data mentah
        warning = check_raw(args.raw, reference)
        if warning:
            print(f"[peringatan] {warning}", file=sys.stderr)
    lab = cfg["label"]
    print(f"[config] {cfg['name']}: a={lab['a']:g} b={lab['b']:g} T={lab['T']} k={lab['k']}")
    paths = run(cfg, args.raw, args.runs_dir, steps=steps, report=not args.no_report)
    print(f"[selesai] keluaran di {paths.root}")
    if args.no_verify or steps != STEPS:
        return 0
    if reference is None:
        print(f"[peringatan] acuan {_reference_path(args, cfg)} tidak ada; verifikasi dilewati",
              file=sys.stderr)
        return 0
    return _cmd_verify(args, cfg)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m pdhl_data",
                                 description="Pipeline data PDH/PDL XAUUSD H1: mentah -> berlabel")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "verify"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True, help="configs/<nama>.json")
        p.add_argument("--runs-dir", type=Path, default=Path("runs"))
        p.add_argument("--reference", type=Path, help="acuan hash (bawaan references/<nama>.json)")
        if name == "run":
            p.add_argument("--raw", type=Path, default=DEFAULT_RAW)
            p.add_argument("--steps", default="all", help=f"all atau gabungan dari {','.join(STEPS)}")
            p.add_argument("--no-report", action="store_true", help="lewati laporan Markdown")
            p.add_argument("--no-verify", action="store_true", help="lewati verifikasi acuan")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    return _cmd_run(args, cfg) if args.cmd == "run" else _cmd_verify(args, cfg)


if __name__ == "__main__":
    sys.exit(main())
