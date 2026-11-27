# Hasil lama Gradient Boosting (acuan pembanding)

Berkas JSON di folder ini adalah salinan byte-identik `Pemodelan PDH-PDL/output/` (folder lama di
luar repo) untuk tahap fitur (RM-22), baseline B0/B1 (RM-23), walk-forward (RM-24), serta model
akhir, kalibrasi τ, dan parameter keputusan (RM-25 s.d. RM-27), dihitung pada 22 November 2026. Semuanya dihitung dari data pipeline **sebelum** adendum ATR 2026-11-24, yaitu
saat `atr14_h1`, `atr14_d`/`atr14_d_prev`, dan PDH/PDL awal 2015 masih kosong. SHA-256 setiap berkas
dicatat di `asal.json`.

Semua berkas ini dihitung dengan fitur 14–15 lama (`f14_engulfing_d`, `f15_volume_relatif`).
Sesudah Kunci-1 Adendum 4 (25 November 2026) fitur 14–15 diganti, sehingga yang masih diharapkan
sama dengan acuan ini hanyalah fitur f01–f13 dan seluruh baseline B0/B1. Angka f14/f15,
*walk-forward*, dan model akhir dari run baru memang berbeda.

Berkas ini tidak dipakai untuk menghitung apa pun. `python -m model.gradient_boosting bandingkan`
hanya memakainya untuk menunjukkan apakah perhitungan ulang dengan data v1.1.0 mengubah angka. Hasil
perbandingan ditulis ke `runs/tesis/model/gradient_boosting/output/bandingkan_lama.md`. Lihat
[docs/PEMODELAN_GRADIENT_BOOSTING.md](../../docs/PEMODELAN_GRADIENT_BOOSTING.md).
