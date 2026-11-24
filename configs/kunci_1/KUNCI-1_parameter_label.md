# Kunci-1: Parameter Label

**Tanggal kunci:** 7 November 2026 (minggu 6)
**Dasar:** Proposal Final v2, IV.I.4 dan Tabel 4.2
**Berkas parameter:** `kunci_1.json` (SHA-256 `79312eb3c948c51f97350c539e140d888e2f71498ec03f863241e5687449b810`)

Dokumen ini tidak diubah setelah tanggal di atas. Setiap perubahan dicatat sebagai adendum bertanggal di bagian bawah.

## 1. Keputusan

Hingga tenggat Tahap 1 (akhir minggu 5, 31 Oktober 2026), jumlah responden yang lolos penyaringan belum mencapai 30. Sesuai IV.I.4, parameter label ditetapkan dengan **nilai cadangan**.

| Parameter | Makna | Nilai terkunci |
| --- | --- | --- |
| *a* | Jarak target posisi *reversal* (× ATR(14) H1 pada waktu keputusan) | **2,0** |
| *b* | Jarak *stop* posisi *reversal* (× ATR) | **1,0** |
| *T* | Horizon maksimum (jumlah *candle* H1 setelah waktu keputusan) | **8** |
| *k* | Waktu keputusan: penutupan *candle* ke-*k* setelah *candle* sentuhan | **0** |

Dasar rujukan setiap nilai ada di `../Rujukan Nilai Cadangan.md`. Nilai ini merupakan asumsi peneliti, bukan hasil optimasi pada data.

## 2. Pembanding deskriptif dari responden (diisi peneliti)

Sesuai IV.I.4 (v2), median dan modus jawaban responden yang tersedia tetap dilaporkan sebagai pembanding, tetapi tidak menggantikan nilai cadangan.

| Ukuran | Nilai |
| --- | --- |
| Jumlah pengisi total | … |
| Jumlah lolos penyaringan | … (< 30) |
| Median (IQR) *a* | … |
| Median (IQR) *b* | … |
| Median (IQR) *T* | … |
| Modus *k* (beserta proporsi jawaban "saat menyentuh level" dan "konfirmasi *timeframe* kecil" yang dipetakan ke *k* = 0) | … |

## 3. Bagian Kunci-1 yang belum dikunci

Fitur tambahan (IV.II.5), aturan B1 (IV.II.6), dan rencana analisis RQ2 masih harus ditetapkan sebelum pemodelan memakainya (baris RM-13 dan RM-14 di Notion). Bila jawaban terbuka tidak memenuhi syarat "disebut ≥ 2 responden", model hanya memakai 12 fitur dasar, dan B1 memakai aturan bawaan (fitur 9 < 0).

## 4. Konsekuensi

- Deteksi event yang sudah ada (k = 0, T = 8; 2.409 event valid) tetap berlaku dan tidak perlu dijalankan ulang.
- Label *triple barrier* dan nilai R untuk data pengembangan dan kalibrasi dihitung dengan parameter ini (`label_triple_barrier.py`, laporan di `output/laporan_label.md`).
- Analisis sensitivitas label (IV.II.3: *a* ∈ {1,5; 2; 3}, *T* ∈ {4; 8; 12}) bersifat deskriptif dan tidak mengubah nilai terkunci.

## Adendum

### Adendum 1 (17 November 2026): fitur tambahan dan aturan B1

Berkas: `kunci_1_fitur.json`. Bagian 3 di atas kini terisi untuk fitur tambahan dan B1. Rencana analisis RQ2 (RM-14) belum dikunci.

**Responden.** Total pengisi 11. Yang lolos penyaringan 9; 2 gugur karena 1.1 = "Kurang dari 3 bulan".

**Frekuensi konfirmasi 2.3 (responden lolos).**

| Konfirmasi | n | Bisa dihitung dari H1 (daftar K) | Keputusan |
| --- | --- | --- | --- |
| K2 Wick penolakan / pin bar | 5 | Ya | Fitur 13 dan aturan B1 |
| K5 Retest setelah breakout | 4 | Sebagian (hanya k ≥ 1) | Tidak dipakai: k = 0 terkunci |
| K3 Engulfing | 3 | Ya | Fitur 14 |
| K7 Lonjakan volume | 3 | Ya (*tick volume*) | Fitur 15 |
| K1 Ditutup kembali di dalam range kemarin | 3 | Ya | Sudah ada (fitur 9) |
| K4 Ditutup menembus level | 3 | Ya | Sudah ada (fitur 9) |
| K9 Sesi London/New York | 3 | Ya | Sudah ada (fitur 4) |
| K6 BOS/CHoCH di *timeframe* kecil | 3 | Tidak | Keterbatasan |
| K10 Berita/fundamental | 2 | Tidak | Kebutuhan tak terpenuhi (keterbatasan 2) |
| Lainnya: MSS (*double top/bottom*) | 1 | – | Disebut < 2 responden |
| K8 Indikator | 0 | – | Hanya dicentang responden yang gugur |

Jawaban terbuka 2.2 ("momentum", "sweep" lalu *close* kembali) sudah tertangkap fitur 7, 9, dan 10. Isian 3.10 kosong. Angka bulat dan *pivot point* tidak disebut, jadi tidak dimasukkan.

**Fitur tambahan (k = 0; hanya candle sentuhan dan candle sebelumnya).**

| No. | Fitur | Definisi |
| --- | --- | --- |
| 13 | Rasio wick penolakan | Wick searah *reversal* / (high − low) candle sentuhan. PDH: high − max(open, close); PDL: min(open, close) − low. Bernilai 0 bila high = low. |
| 14 | Engulfing (bertanda *d*) | Badan candle sentuhan menutupi badan candle sebelumnya dan warnanya berlawanan. +1 bila searah *continuation* (bullish di PDH, bearish di PDL), −1 bila searah *reversal*, 0 bila tidak ada. |
| 15 | Volume relatif | *Tick volume* candle sentuhan / rata-rata *tick volume* bar dengan jam New York yang sama pada 20 sesi sebelumnya (minimal 10 bar). |

Jumlah fitur akhir: 15.

**Aturan B1.** Konfirmasi berstatus "Ya" yang paling sering dicentang adalah K2, sehingga aturan cadangan (fitur 9 < 0) tidak dipakai. B1 terpenuhi bila fitur 13 ≥ 2/3, yaitu wick searah *reversal* minimal dua pertiga rentang candle sentuhan. Probabilitas B1 adalah frekuensi kelas pada data pengembangan, dipisah menurut terpenuhi atau tidaknya aturan.

**Catatan tafsir.** Pada aturan 4.3 kuesioner, "konfirmasi berikutnya" tidak dibaca sebagai "kecuali konfirmasi B1". IV.II.5 memilih informasi yang paling sering disebut, dan model perlu memuat informasi B1 agar uji model vs B1 setara, seperti B1 cadangan yang juga memakai fitur dasar 9.

### Adendum 2 (17 November 2026): pembanding responden dan rencana analisis RQ2

Dengan adendum ini seluruh isi Kunci-1 sudah dikunci: parameter label, fitur tambahan, aturan B1, dan rencana analisis RQ2. Integritas berkas dicatat di `MANIFEST_KUNCI_1.json` (SHA-256).

**Pembanding deskriptif dari responden (IV.I.4 v2).** Angka di tabel ini mengisi bagian 2 di atas. Nilainya hanya pembanding dan tidak menggantikan nilai cadangan.

| Ukuran | Nilai |
| --- | --- |
| Jumlah pengisi total | 11 |
| Jumlah lolos penyaringan | 9 (< 30) |
| Median (IQR) *a* | 3,0 (2,75–3,0) ATR; n = 8, 1 jawaban "tidak entry" tidak dapat dikonversi |
| Median (IQR) *b* | 1,0 (0,94–1,63) ATR; n = 8, 1 jawaban tidak dapat dikonversi |
| Median rasio *a*/*b* per responden | 2,0 |
| Median (Q1–Q3) *T*, ordinal | ±2 jam (±2 jam – lebih dari 1 hari); n = 9 |
| Modus *k* | 0 (7 dari 9); *k* = 1: 1, *k* = 2: 1 |
| Proporsi "konfirmasi *timeframe* kecil" dan "saat menyentuh" yang dipetakan ke *k* = 0 | 5/9 (55,6%) dan 0/9 |

Median responden untuk *b* dan *k* sama dengan nilai cadangan, dan median rasio *a*/*b* juga 2. Median *a* (3) lebih besar dan median *T* (2 jam) jauh lebih pendek daripada nilai cadangan (*a* = 2, *T* = 8). Perbedaan ini dibahas di laporan sebagai keterbatasan. *T* = 2 tidak ditambahkan ke grid sensitivitas, agar grid tetap seperti di proposal.

**Rencana analisis RQ2.** Dokumennya `RENCANA_ANALISIS_RQ2.md`, dan parameternya ada di `kunci_1_rq2.json` (dimuat lewat `common.load_kunci1_rq2`). Isinya:

- RQ2.1: *log loss* (ln, *clip* 10⁻¹⁵), *bootstrap* klaster `week_start` 2.000× dengan `SEED_BOOT` = 20261117 dan indeks bersama untuk B0/B1/B2, interval persentil 95%, dan kriteria batas atas < 0 untuk ketiga *baseline*. B1 memakai Laplace +1 hanya bila sebuah sel kosong. B2 adalah regresi logistik L2 dengan C = 1 (tidak disetel), dikalibrasi τ sendiri. ECE memakai 10 bin selebar 0,1.
- RQ2.2: K = 1.000 set plasebo, *seed* 20261118 + *j*, (*s*, *u*) diundi ulang bersama sampai valid (maksimum 1.000), dan f02 tetap memakai rentang asli. P*rev* dihitung atas Januari 2016 – Juli 2026 dengan *p* dua sisi (*p* < 0,05) dan *effect size* (pp, persentil, *z*). SS memakai set *j* = 1–20, dan nilai seri dihitung tidak melampaui.
- Pendukung: δ utama 0,1R, grid δ {0; 0,1; 0,2; 0,3}, *spread* {0; 0,15; 0,25; 0,35; 0,50; 1,00} USD (biaya *S*/(*b*·ATR) untuk *reversal* dan *S*/(*a*·ATR) untuk *continuation*, sama dengan Artikel 1), serta sensitivitas label *a* ∈ {1,5; 2; 3} × *T* ∈ {4; 8; 12}.
- Teknis: pengodean, nilai kosong, LightGBM deterministik (`SEED_MODEL` = 20261119), aturan seri 10⁻⁶ ke konfigurasi yang lebih sederhana, dan τ dicari dengan *golden-section* pada [0,05; 20].
