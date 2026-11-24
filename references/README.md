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

## Adendum 2026-11-24: pemanasan dari riwayat sebelum 2015

Kode versi 1.1.0 memakai bar sebelum 2015-01-01 (sejak 2009-11-16) sebagai riwayat TR/ATR(14) H1,
ATR(14) harian, dan PDH/PDL sesi pertama. Bar tersebut tetap tidak masuk silver. Sebelumnya bar itu dibuang
sebelum indikator dihitung, sehingga awal 2015 punya `atr14_h1`, `atr14_d`, dan PDH/PDL yang kosong.

Akibatnya, 6 hash berkas dan 2 angka rekonsiliasi di `tesis.json` kini berasal dari **kode yang sudah
diperbaiki**, bukan dari pipeline lama. Nilai lama dan barunya dicatat di kunci `adendum`
(diabaikan oleh `verify`):

| Berkas | Yang berubah |
|---|---|
| `de_output/silver_xauusd_h1.csv`, `data/bars_devcalib.csv` | `atr14_h1` 474 bar Januari 2015, `hours_since_prev` bar 0 |
| `de_output/silver_sessions.csv`, `data/sessions_devcalib.csv` | PDH/PDL sesi 2015-01-02, `atr14_d`/`atr14_d_prev` s.d. 2016-09-23 |
| `de_output/gold_pdhl_events.csv`, `data/events_dev.csv` | `atr14_d_prev` event dev 2016 (maks. 4,6e-10 relatif) |

Hash lain tetap hash pipeline lama, termasuk `kunci_uji/*` (= kunci resmi 2026-11-07),
`data/labels_*`, `data/events_calib.csv`, dan `output/*`. Dengan begitu, `verify` tetap membuktikan bahwa data
uji, event, dan label tidak berubah. Kunci `runs/tesis/kunci_uji/` dibuat ulang pada tanggal adendum karena
`berkas_pemodelan` di manifestnya ikut berubah. Isi data ujinya byte-identik dengan kunci lama.
