# xauusd-pdhl-ml

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rizalpriyono/xauusd-pdhl-ml/blob/main/notebooks/00_setup_colab.ipynb)

Pipeline data untuk penelitian tesis tentang *first touch* Previous-Day High/Low (PDH/PDL) pada XAUUSD H1.
Pipeline ini mengolah data mentah sampai menjadi data berlabel *triple barrier*, siap dipakai eksperimen model
(**Gradient Boosting vs LSTM**).

```
XAUUSD_H1.csv ──► events ──────────────► split + kunci uji ─────────► labels
 (bronze, UTC)    silver bar & sesi,      dev 2016–2023 / calib 2024   Reversal / Continuation / Netral
                  gold event PDH/PDL      uji 2025–Jul 2026 DIKUNCI    + R_rev, R_cont
```

Hanya memakai pustaka standar Python ≥ 3.10. Data mentah `data/raw/XAUUSD_H1.csv` sudah ada di repo.

## Konfigurasi

| `configs/` | a | b | T | k | Event valid dev / calib / uji | Label dev (Rev / Cont / Net) | Label calib |
|---|---|---|---|---|---|---|---|
| `tesis.json` (Kunci-1, nilai cadangan Tabel 4.2) | 2 | 1 | 8 | 0 | 1.841 / 227 / 341 | 414 / 1.042 / 385 | 62 / 125 / 40 |

- a dan b dalam kelipatan ATR(14) H1, T dalam *candle* H1, dan k adalah jeda *candle* sesudah sentuhan.
- Parameter label hanya dibaca dari konfigurasi; jangan menulis angka langsung di kode.
- Perubahan parameter (misalnya analisis sensitivitas label) dibuat sebagai konfigurasi baru dengan nama lain.

## Menjalankan di Google Colab

Klik badge **Open in Colab** di atas, lalu pilih *Runtime → Run all*. Notebook
[`notebooks/00_setup_colab.ipynb`](notebooks/00_setup_colab.ipynb) akan:

1. clone repo,
2. cek SHA-256 data mentah,
3. menjalankan pipeline dan verifikasi (`19/19 OK`),
4. menjalankan tes,
5. memuat data berlabel ke pandas lewat `load_model_csv`.

Runtime CPU sudah cukup.

- Isi `/content` hilang saat runtime berhenti. Pipeline cukup dijalankan ulang (sekitar 3 detik, deterministik), sedangkan
  hasil model disimpan ke Google Drive.
- Colab berjalan sebagai *root*, sehingga izin hanya-baca pada `kunci_uji/` tidak mengikat. Kunci uji tetap
  dijaga oleh manifest SHA-256 (`LockViolation`) dan guard `load_model_csv`.
- Logika baru (fitur, model, evaluasi) ditulis di `src/` beserta tesnya. Notebook hanya memanggil.

## Menjalankan secara lokal

```bash
git clone https://github.com/rizalpriyono/xauusd-pdhl-ml.git
```

```bash
cd xauusd-pdhl-ml
```

```bash
PYTHONPATH=src python3 -m pdhl_data run --config configs/tesis.json
```

Setiap run menulis ke `runs/tesis/`, lalu otomatis memverifikasi hasilnya terhadap `references/tesis.json`
(harus `19/19 OK`). Pilihan lain:

- `--steps events,split,labels` menjalankan sebagian tahap saja (verifikasi dilewati).
- `--no-report` melewati laporan Markdown. `--raw <path>` memakai data mentah dari lokasi lain.
- `python3 -m pdhl_data verify --config configs/tesis.json` memverifikasi ulang tanpa menjalankan pipeline.
  Kode keluarnya: 0 = OK, 1 = ada BEDA/HILANG, 2 = berkas acuan tidak ada.
- `pip install -e .` menyediakan perintah `pdhl-data run --config ...` tanpa `PYTHONPATH`.

Untuk menjalankan tes (cakupan baris ±98%, termasuk reproduksi end-to-end dari data mentah):

```bash
python3 -m unittest discover -s tests -v
```

## Keluaran `runs/tesis/`

| Folder | Isi | Dipakai pemodelan? |
|---|---|---|
| `de_output/` | `silver_xauusd_h1.csv`, `silver_sessions.csv`, `gold_pdhl_events.csv`, `dq_dst_verification.csv`, `de_rekonsiliasi.json`, `laporan_data_engineering.md` | Tidak (berisi periode uji) |
| `data/` | `events_{dev,calib}.csv`, `bars_devcalib.csv`, `sessions_devcalib.csv`, **`labels_{dev,calib}.csv`** | **Ya, hanya folder ini** |
| `kunci_uji/` | `events_test.csv`, `bars_test.csv`, `sessions_test.csv`, manifest SHA-256 (hanya-baca) | Tidak sebelum eksekusi tunggal |
| `output/` | `split_rekonsiliasi.json`, `label_rekonsiliasi.json`, `label_purged.csv`, `laporan_label.md` | Laporan |

Kolom dan aturan lengkap ada di [docs/SPESIFIKASI_DATA.md](docs/SPESIFIKASI_DATA.md).

## Aturan data uji

- Data uji (2025-01-02 – 2026-07-31) dikunci di `runs/tesis/kunci_uji/`. Data ini tidak dipakai untuk
  label, fitur, model, atau analisis sebelum Kunci-2.
- Kode pemodelan **wajib** memuat data lewat `load_model_csv`:

  ```python
  from pdhl_data import load_model_csv
  labels = load_model_csv("runs/tesis/data/labels_dev.csv")   # TestDataLeak bila ada baris > 2024
  ```

  Loader menolak path di folder `kunci_uji*` dan baris dengan `session_date` > 2024-12-31.
- Run ulang tidak menimpa kunci. Proses berhenti dengan `LockViolation` bila:
  - isi data uji berubah,
  - a/b/T/k di konfigurasi berbeda dari yang tercatat di manifest, atau
  - berkas di `data/` berbeda dari saat kunci dibuat.
- Kunci fisik (hanya-baca + SHA-256) hanya berlaku untuk `kunci_uji/`. `data/raw/XAUUSD_H1.csv` dan
  `de_output/` tetap memuat seluruh periode termasuk 2025–2026, dan **tidak boleh** dibaca kode pemodelan.
  Bila dibaca lewat `load_model_csv`, baris 2025–2026 ditolak.
- Kunci resmi tetap yang lama, yaitu 2026-11-07 (`Pemodelan PDH-PDL/kunci_uji/`, di luar repo). Kunci di
  `runs/` adalah reproduksi dengan SHA-256 yang sama.

## Reproduksi

`references/tesis.json` dibuat dari keluaran pipeline lama yang sudah direkonsiliasi dengan implementasi
independen, bukan dari kode ini. Kode di repo ini menghasilkan 16 berkas dan 55 angka rekonsiliasi yang
identik byte demi byte dengan pipeline lama, **kecuali** 6 berkas dan 2 angka yang tercatat di `adendum`.
Pengecualian ini berasal dari adendum 2026-11-24: pemanasan ATR/PDH-PDL kini memakai riwayat sebelum
2015, sehingga `atr14_h1`, `atr14_d`, dan PDH/PDL tidak lagi kosong di awal 2015. Data uji, event, dan
label tidak berubah. Lihat [references/README.md](references/README.md).

| Modul | Asal (folder lama di luar repo) |
|---|---|
| `src/pdhl_data/events.py`, `events_report.py` | `Data Engineering PDH-PDL/pdhl_events_local.py`, `pdhl_events_report.py` |
| `src/pdhl_data/split.py` | `Pemodelan PDH-PDL/split_dan_kunci_uji.py` |
| `src/pdhl_data/labels.py`, `labels_report.py` | `Pemodelan PDH-PDL/label_triple_barrier.py`, `laporan_label.py` |
| `src/pdhl_data/common.py`, `config.py` | `Pemodelan PDH-PDL/common.py`; `kunci_1.json` → `configs/tesis.json` |

## Struktur repo

```
configs/        parameter label (tesis.json)
references/     SHA-256 dan angka rekonsiliasi acuan
src/pdhl_data/  paket pipeline (common, config, events, split, labels, pipeline, CLI)
tests/          unittest (event, split/guard, label, konfigurasi, reproduksi end-to-end)
notebooks/      notebook Google Colab (00_setup_colab.ipynb)
docs/           spesifikasi data
data/raw/       XAUUSD_H1.csv (data mentah)
runs/           keluaran run (tidak di-commit)
```
