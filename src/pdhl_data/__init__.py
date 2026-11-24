"""Pipeline data PDH/PDL XAUUSD H1: data mentah -> event -> split + kunci uji -> label.

Modul:
    common         periode penelitian, IO CSV, hash, guard data uji (`load_model_csv`)
    config         pemuatan dan validasi `configs/*.json`
    events         tahap 1: bronze -> silver -> gold (deteksi event PDH/PDL)
    split          tahap 2: pembagian dev/calib/test dan penguncian data uji
    labels         tahap 3: label triple barrier dan satuan R
    pipeline       orkestrasi dan verifikasi terhadap `references/*.json`
"""
from .common import TestDataLeak, load_model_csv

__all__ = ["TestDataLeak", "load_model_csv"]
__version__ = "1.1.0"
