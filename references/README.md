# Acuan reproduksi

`tesis.json` berisi SHA-256 setiap berkas keluaran dan angka rekonsiliasi dari pipeline **lama**
yang sudah diverifikasi sebelum repo ini dibuat: `Data Engineering PDH-PDL/output/` dan
`Pemodelan PDH-PDL/` (kunci uji 2026-11-07). Pipeline lama itu sudah direkonsiliasi dengan notebook
Databricks (29/29 dan 26/26 OK) dan dengan implementasi independen ketiga (19 Nov 2026, 0 beda).

Acuan dibuat dari berkas lama, bukan dari kode di repo ini, sehingga `verify` benar-benar memeriksa
bahwa kode baru menghasilkan byte yang sama. Sebelum disimpan, hash `sumber` dan `berkas_pemodelan`
di manifest kunci lama dicocokkan dengan berkas lama (semuanya cocok). Hash data uji (`kunci_uji/*`)
diambil dari manifest kunci; berkas uji tidak dibuka.

Laporan Markdown (`laporan_*.md`) dan manifest kunci tidak dimasukkan karena memuat tanggal pembuatan.
