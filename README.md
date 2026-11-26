# xauusd-pdhl-ml

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rizalpriyono/xauusd-pdhl-ml/blob/main/notebooks/00_setup_colab.ipynb)

Pipeline data untuk penelitian tesis tentang *first touch* Previous-Day High/Low (PDH/PDL) pada XAUUSD H1.
Pipeline ini mengolah data mentah sampai menjadi data berlabel *triple barrier*, siap dipakai eksperimen model
(**Gradient Boosting vs LSTM**). Tahap pemodelan tersedia pada dua subpaket sejajar:
`src/model/gradient_boosting/` (lihat [Pemodelan Gradient Boosting](#pemodelan-gradient-boosting)) dan
`src/model/lstm/` (lihat [Pemodelan LSTM](#pemodelan-lstm)).

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
| `model/gradient_boosting/` | `data/` (fitur, OOF baseline dan *walk-forward*), `output/` (rekonsiliasi, laporan) | Hasil pemodelan GB (ditulis `python -m model.gradient_boosting`) |
| `model/lstm/` | `data/`, `output/`, `checkpoints/` (5 `.pt` + scaler), `kunci_2/` (segel SHA-256 hanya-baca) | Hasil pemodelan LSTM (ditulis `python -m model.lstm`) |

Kolom dan aturan lengkap ada di [docs/SPESIFIKASI_DATA.md](docs/SPESIFIKASI_DATA.md).

## Pemodelan Gradient Boosting

Kode di `src/model/gradient_boosting/` menjalankan tiga tahap pertama pemodelan pada `runs/tesis/data/`
dan menulis hasilnya ke `runs/tesis/model/gradient_boosting/{data,output}/`:

| Tahap | Perintah | Isi |
|---|---|---|
| 1 | `fitur` | 15 fitur *point-in-time* (dev + calib), uji pemotongan, laporan fitur |
| 2 | `pembanding` | Baseline B0 (frekuensi kelas) dan B1 (*wick* ≥ 2/3): akhir + OOF 2019–2023 |
| 3 | `walk-forward` | 8 konfigurasi LightGBM × 5 lipatan, pilih *log loss* OOF terkecil |
| – | `bandingkan` | Selisih angka terhadap hasil lama (`references/gradient_boosting_lama/`) |

```bash
pip install -e '.[model]'
```

```bash
PYTHONPATH=src python3 -m model.gradient_boosting semua --config configs/tesis.json
```

- Parameter dibaca dari salinan Kunci-1 di `configs/kunci_1/`, yang dicek terhadap manifest SHA-256.
- Data masukan dicek terhadap `references/tesis.json`.
- Bar tanpa `atr14_h1` atau event tanpa `atr14_d_prev` langsung ditolak (data pipeline < 1.1.0).
- Tahap fitur dan baseline hanya memakai stdlib; LightGBM hanya dibutuhkan tahap *walk-forward*.

Proses, rumus, dan hasil lama sebagai acuan ada di
[docs/PEMODELAN_GRADIENT_BOOSTING.md](docs/PEMODELAN_GRADIENT_BOOSTING.md).

## Pemodelan LSTM

Kode di `src/model/lstm/` menjalankan seluruh tahap pemodelan **LSTM v2 (13 fitur sekuensial)** dari RM-22
sampai pembekuan pra-uji 🔒 Kunci-2 pada `runs/tesis/data/` (tanpa menyentuh `kunci_uji/`) dan menulis
hasilnya ke `runs/tesis/model/lstm/{data,output,checkpoints,kunci_2}/`:

| Tahap | Perintah | Isi |
|---|---|---|
| 1 (RM-22) | `fitur` | 13 fitur sekuensial *point-in-time* (v2), uji pemotongan (`L=96`), laporan fitur |
| 2 (RM-23) | `pembanding` | Baseline B0 & B1 (*wick* ≥ 2/3): identik SHA-256 dengan Gradient Boosting |
| 3 (RM-24) | `walk-forward` | 10 konfigurasi LSTM (`L × hidden`) × 5 lipatan × 5 seed Kunci-1 → `L46_h32` |
| 4 (RM-25) | `model-akhir` | Latih penuh pemenang `L46_h32` di seluruh `dev` (1.841 event × 5 seed) → 5 `.pt` + scaler |
| 5 (RM-26) | `kalibrasi` | *Temperature scaling* `τ` pada `calib` 2024 (227 event) via *golden-section search* |
| 6 (RM-27) | `keputusan` | Aturan keputusan `E_rev`/`E_cont` (`m_rev` Netral `dev`, `δ = 0,1R`) & `realized_R` |
| 7 (Kunci-2) | `kunci-2` | Bekukan seluruh komponen (`read-only`) & segel `MANIFEST_KUNCI_2_LSTM.json` (SHA-256) |
| – | `verifikasi-kunci-2` | Audit integritas segel Kunci-2 terhadap manifest SHA-256 (murni stdlib) |

**Instalasi khusus jalur LSTM saja (tanpa LightGBM):**

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e '.[lstm]'           # atau: pip install -r requirements-lstm.txt
```

**Menjalankan:**

```bash
# Pastikan data pipeline v1.1.0 sudah dibuat terlebih dulu:
PYTHONPATH=src python3 -m pdhl_data run --config configs/tesis.json

# Jalankan seluruh alur RM-22 s.d. Kunci-2:
PYTHONPATH=src python3 -m model.lstm semua --config configs/tesis.json --device cpu

# Atau jalankan RM-25 -> RM-26 -> RM-27 -> Kunci-2 saja di atas hasil RM-22..24:
PYTHONPATH=src python3 -m model.lstm lanjutan --config configs/tesis.json --device cpu

# Verifikasi segel Kunci-2:
PYTHONPATH=src python3 -m model.lstm verifikasi-kunci-2 --config configs/tesis.json
```

- Setelah `pip install -e '.[lstm]'`, perintah `pdhl-lstm ...` setara dengan `PYTHONPATH=src python3 -m model.lstm ...`.
- Tahap `fitur`, `pembanding`, `keputusan`, `kunci-2`, dan `verifikasi-kunci-2` berjalan murni dengan stdlib; `torch` & `numpy` hanya dibutuhkan untuk `walk-forward`, `model-akhir`, dan `kalibrasi`.
- Dokumentasi lengkap ada di [docs/lstm/PEMODELAN_LSTM.md](docs/lstm/PEMODELAN_LSTM.md) dan panduan interaktif pemula di [docs/lstm/panduan_lstm_pemula.html](docs/lstm/panduan_lstm_pemula.html).

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
configs/        parameter label (tesis.json) dan salinan Kunci-1 (kunci_1/, dengan manifest SHA-256)
references/     SHA-256 dan angka rekonsiliasi acuan; gradient_boosting_lama/ = hasil lama pemodelan
src/pdhl_data/  paket pipeline (common, config, events, split, labels, pipeline, CLI)
src/model/gradient_boosting/
                pemodelan GB: kunci, jalur, fitur, laporan_fitur, pembanding, walk_forward, bandingkan, CLI
src/model/lstm/ pemodelan LSTM: kunci, jalur, fitur, arsitektur, pembanding, walk_forward, model_akhir,
                kalibrasi, keputusan, kunci_2, CLI
tests/          unittest (pipeline data, test_gb_*, test_lstm_*)
notebooks/      notebook Google Colab (00_setup_colab.ipynb)
docs/           spesifikasi data, pemodelan GB, dan subfolder docs/lstm/ (panduan HTML & dokumentasi LSTM)
requirements-model.txt / requirements-lstm.txt
                daftar dependensi terpisah untuk jalur GB dan jalur LSTM
data/raw/       XAUUSD_H1.csv (data mentah)
runs/           keluaran run (tidak di-commit)
```
