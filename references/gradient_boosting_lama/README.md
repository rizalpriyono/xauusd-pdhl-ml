# Hasil lama Gradient Boosting (acuan pembanding)

Berkas JSON di folder ini adalah salinan byte-identik `Pemodelan PDH-PDL/output/` (folder lama di
luar repo) untuk tahap fitur (RM-22), baseline B0/B1 (RM-23), dan walk-forward (RM-24), dihitung
pada 22 November 2026. Semuanya dihitung dari data pipeline **sebelum** adendum ATR 2026-11-24, yaitu
saat `atr14_h1`, `atr14_d`/`atr14_d_prev`, dan PDH/PDL awal 2015 masih kosong. SHA-256 setiap berkas
dicatat di `asal.json`.

Berkas ini tidak dipakai untuk menghitung apa pun. `python -m model.gradient_boosting bandingkan`
hanya memakainya untuk menunjukkan apakah perhitungan ulang dengan data v1.1.0 mengubah angka. Hasil
perbandingan ditulis ke `runs/tesis/model/gradient_boosting/output/bandingkan_lama.md`. Lihat
[docs/PEMODELAN_GRADIENT_BOOSTING.md](../../docs/PEMODELAN_GRADIENT_BOOSTING.md).
