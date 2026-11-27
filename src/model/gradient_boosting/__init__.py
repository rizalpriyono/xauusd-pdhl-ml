"""Pemodelan Gradient Boosting (LightGBM) untuk sentuhan pertama PDH/PDL XAUUSD H1.

Tahap (urutan menjalankan):
    fitur          tahap 1 (RM-22): 15 fitur point-in-time untuk event dev dan calib
    laporan_fitur  laporan deskriptif fitur (tidak dikondisikan pada label)
    pembanding     tahap 2 (RM-23): baseline non-model B0 (frekuensi kelas) dan B1 (heuristik wick)
    walk_forward   tahap 3 (RM-24): validasi walk-forward 8 konfigurasi LightGBM dan pemilihannya
    model_akhir    tahap 4 (RM-25 s.d. RM-27): model akhir, kalibrasi τ, parameter keputusan
                   (laporan dan diagram reliabilitas di `laporan_model_akhir`)
    bandingkan     selisih angka rekonsiliasi baru terhadap hasil lama (sebelum adendum ATR)

Pendukung:
    kunci          memuat dan memvalidasi parameter Kunci-1 (`configs/kunci_1/`, cek manifest SHA-256)
    jalur          folder masukan/keluaran dan cek SHA-256 data masukan terhadap `references/`

Semua data dimuat lewat `pdhl_data.load_model_csv`, sehingga data uji tidak bisa terbaca.
CLI: `python -m model.gradient_boosting <tahap>`, tahap = fitur, pembanding, walk-forward,
model-akhir, bandingkan, atau semua.
"""
