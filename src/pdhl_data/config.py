"""Konfigurasi pipeline: satu berkas JSON per judul penelitian (`configs/*.json`).

Konfigurasi memuat parameter label terkunci (a, b, T, k), jumlah event valid acuan, nama manifest
kunci uji, seed laporan, dan rujukan pasal untuk teks laporan. Hanya parameter label yang
memengaruhi data; sisanya untuk pemeriksaan dan laporan.
"""
import json
import math
import re
from pathlib import Path

NAME_RE = re.compile(r"[a-z0-9][a-z0-9_-]*")       # dipakai dengan fullmatch
REQUIRED_RUJUKAN = ("label", "R", "pembagian", "kalibrasi", "keputusan", "deskriptif")


class ConfigError(ValueError):
    """Berkas konfigurasi tidak valid."""


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _section(raw, key, path):
    value = raw.get(key, {})
    if not isinstance(value, dict):
        raise ConfigError(f"{path}: {key} harus objek JSON")
    return value


def _validate_label(label, path):
    a, b, horizon, k = (label.get(key) for key in ("a", "b", "T", "k"))
    if not all(_is_num(v) for v in (a, b, horizon, k)):
        raise ConfigError(f"{path}: label.a, b, T, k harus angka")
    if not (a > 0 and b > 0):
        raise ConfigError(f"{path}: label.a dan label.b harus > 0")
    if horizon != int(horizon) or horizon < 1 or k != int(k) or k < 0:
        raise ConfigError(f"{path}: label.T harus bilangan bulat >= 1 dan label.k bilangan bulat >= 0")
    if not isinstance(label.get("tanggal_kunci"), str):
        raise ConfigError(f"{path}: label.tanggal_kunci wajib diisi (YYYY-MM-DD)")
    return {**label, "a": float(a), "b": float(b), "T": int(horizon), "k": int(k)}


def _validate_expected(expected, path):
    if expected is None:
        return None
    if set(expected) != {"dev", "calib", "test"} or not all(
            isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in expected.values()):
        raise ConfigError(f"{path}: expected_valid harus {{dev, calib, test}} berisi bilangan bulat")
    return dict(expected)


def parse_config(raw, path="<config>"):
    """Validasi isi konfigurasi dan kembalikan salinan baru yang sudah dinormalisasi."""
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: isi konfigurasi harus objek JSON")
    name = raw.get("name")
    if not isinstance(name, str) or not NAME_RE.fullmatch(name):
        raise ConfigError(f"{path}: name harus huruf kecil/angka/-/_ (dipakai sebagai folder run)")
    lock = _section(raw, "lock", path)
    manifest = lock.get("manifest", "")
    if not (isinstance(manifest, str) and manifest.endswith(".json") and "/" not in manifest):
        raise ConfigError(f"{path}: lock.manifest harus nama berkas .json")
    for key in ("aturan", "dibuka_pada", "kunci_key"):
        if not isinstance(lock.get(key), str) or not lock[key]:
            raise ConfigError(f"{path}: lock.{key} wajib diisi")
    seed = _section(raw, "report", path).get("sample_seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ConfigError(f"{path}: report.sample_seed harus bilangan bulat")
    rujukan = _section(raw, "rujukan", path)
    missing = [key for key in REQUIRED_RUJUKAN if not isinstance(rujukan.get(key), str)]
    if missing:
        raise ConfigError(f"{path}: rujukan belum lengkap: {', '.join(missing)}")
    return {
        **raw,
        "label": _validate_label(_section(raw, "label", path), path),
        "expected_valid": _validate_expected(raw.get("expected_valid"), path),
        "lock": dict(lock),
        "rujukan": dict(rujukan),
    }


def load_config(path):
    path = Path(path)
    with open(path) as f:
        return parse_config(json.load(f), path)
