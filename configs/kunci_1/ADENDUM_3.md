# Kunci-1, Adendum 3: Penyesuaian Rancangan Analisis setelah Perumusan Ulang Pertanyaan Penelitian

**Tanggal:** 25 November 2026, sebelum Kunci-2 (direncanakan 5 Desember 2026) dan sebelum data uji dibuka
**Dasar:**

- Perumusan ulang masalah dan pertanyaan penelitian (*Penguatan Inti Permasalahan*, 21 November 2026).
- Keputusan tahap pemodelan 22 November 2026 (`Metodologi Pemodelan ML PDH-PDL.md`, Bagian 12).
- Keputusan peneliti 25 November 2026.

**Parameter terbaca mesin:** `kunci_1_adendum_3.json`
**Integritas:** `MANIFEST_KUNCI_1_ADENDUM_3.json`. Manifest lama (`MANIFEST_KUNCI_1.json`) dan kelima berkas Kunci-1 tidak diubah.

Adendum ini **tidak** mengubah:

- parameter label;
- 15 fitur dan aturan B1;
- grid dan parameter teknis LightGBM;
- *seed*;
- rentang τ;
- δ dan grid δ;
- grid *spread*;
- parameter *bootstrap*.

Yang berubah adalah pembanding, pertanyaan yang diuji, dan cara menarik kesimpulan. Bila isi adendum ini bertentangan dengan `RENCANA_ANALISIS_RQ2.md` atau `kunci_1_rq2.json`, adendum ini yang berlaku.

## 1. Penomoran pertanyaan penelitian

Penomoran RQ di adendum ini mengikuti rumusan baru (21 November 2026). Dokumen Kunci-1 sebelumnya memakai penomoran *Proposal Final v2*.

| RQ baru | Rumusan singkat | Padanan di Kunci-1 |
| --- | --- | --- |
| RQ1 | Model mengestimasi peluang *reversal*/*continuation* lebih baik daripada peluang dasar (B0) dan aturan yang mewakili heuristik *trader* (B1) | RQ2.1 (`RENCANA_ANALISIS_RQ2.md` §2) |
| RQ2 | Seleksi sentuhan berbasis peluang model menghasilkan R bersih yang lebih baik setelah biaya daripada tanpa seleksi (S0) dan seleksi heuristik (S1) | Analisis pendukung §4 butir 3–5, yang sebelumnya deskriptif |
| – | Kekhususan PDH/PDL terhadap level plasebo | RQ2.2 (§3), **dihapus** |
| – | Evaluasi pengguna | RQ3 proposal, **dihapus** |

## 2. Perubahan

### 2.1 Pembanding B2 dihapus (RQ1)

**Alasan.** Masalah penelitian ini adalah heuristik *trader* yang belum diuji secara sistematis. Karena itu, pembanding yang relevan adalah peluang dasar dan heuristik itu sendiri, bukan model *machine learning* lain.

**Kriteria RQ1.** RQ1 didukung bila batas atas interval 95% untuk *D*k* < 0 bagi **B0 dan B1**.

- *D*k* adalah rata-rata (ℓ model − ℓ B*k*) pada *event* uji yang sama.
- Kerangka *intersection–union* atas dua pembanding ini setara dengan uji satu sisi α = 0,025 per perbandingan, sehingga koreksi uji jamak tidak diperlukan.
- Metrik, *clip*, *bootstrap* klaster minggu (2.000 ulangan, `SEED_BOOT` = 20261117), dan indeks ulangan bersama tetap seperti `RENCANA_ANALISIS_RQ2.md` §2.2.

**Isian Kunci-1 yang tidak berlaku lagi:** `kunci_1_rq2.json`: `rq2_1.baseline` (unsur "B2"), `rq2_1.B2`, dan unsur B2 pada Tabel R2-1 dan R2-2.

### 2.2 Kandidat regresi logistik dihapus

Langkah pemilihan *gradient boosting* vs regresi logistik (IV.II.6) dan isian `teknis.logreg` tidak berlaku. Dengan rancangan ini, LightGBM menjadi model akhir. Validasi *walk-forward* 22 November 2026 sudah dijalankan dengan ketentuan ini.

### 2.3 Metrik klasifikasi sebagai metrik pendukung (RQ1)

Kelas prediksi adalah kelas dengan peluang tertinggi (*argmax*). Metrik yang dilaporkan:

- *confusion matrix* 3 × 3;
- *precision*, *recall*, dan F1 per kelas, ditambah *macro*-F1;
- ROC-AUC dan PR-AUC satu-lawan-semua, dengan garis acuan PR-AUC sama dengan prevalensi kelas;
- *precision* rekomendasi *reversal*, dibandingkan dengan titik impas *b*/(*a* + *b*) = 1/3.

*Log loss* tetap menjadi metrik utama. Tidak ada keputusan yang bergantung pada metrik klasifikasi. Akurasi tidak dijadikan ukuran utama karena kelasnya tidak seimbang.

### 2.4 RQ2 menjadi pertanyaan utama

**Strategi:**

- **SM**: aturan keputusan model dengan δ = 0,1R.
- **S0**: semua sentuhan diambil sebagai *reversal*.
- **S1**: *reversal* bila f13 ≥ 2/3, selain itu *abstain*.

**Ukuran:**

- Ukuran utama adalah rata-rata **R bersih per *event***, dengan *abstain* dihitung 0.
- Ukuran deskriptif: R bersih per transaksi, cakupan, jumlah transaksi, R kumulatif, dan *maximum drawdown* (dalam R).

**Biaya.** Untuk setiap *event* *i*:

R bersih*i*(*S*) = R*i* − *S* · *c*i

Koefisien biayanya:

- *c*i = 1/(*b* · ATR*i*) untuk *reversal*;
- *c*i = 1/(*a* · ATR*i*) untuk *continuation*;
- *c*i = 0 untuk *abstain*.

ATR*i* adalah ATR *event*. *S* diambil dari grid `pendukung.spread_usd`, yaitu {0; 0,15; 0,25; 0,35; 0,50; 1,00} USD per oz.

***Spread* impas** (deskriptif):

- Untuk setiap strategi: *S** = Σ R*i* / Σ *c*i. Bila Σ R*i* ≤ 0, strategi dilaporkan tidak impas.
- Untuk selisih SM − S*k*: *S**Δ = ΔR̄(0) / Δc̄.

**Inferensi.** Pada setiap *S*, selisih ΔR̄(*S*) untuk SM − S0 dan SM − S1 diberi interval persentil 95%. Intervalnya diperoleh dari *bootstrap* klaster minggu yang sama dengan RQ1 (2.000 ulangan, `SEED_BOOT`), dengan indeks ulangan bersama untuk kedua selisih dan semua *S*.

**Tafsir per selisih dan per *S*:**

| Interval 95% ΔR̄(*S*) | Tafsir |
| --- | --- |
| Batas bawah > 0 | SM lebih baik daripada S*k* pada *S* |
| Batas atas < 0 | SM lebih buruk daripada S*k* pada *S* |
| Mencakup 0 | **Tidak dapat disimpulkan**, bukan "tidak lebih baik" |

**Keputusan RQ2.**

- RQ2 didukung pada *spread* *S* hanya bila SM lebih baik daripada S0 **dan** S1 pada *S* tersebut.
- Kesimpulan dirumuskan sebagai rentang *S* tempat RQ2 didukung, lalu dibaca bersama *S**Δ.
- Grid *spread* adalah analisis sensitivitas atas satu hipotesis, bukan enam hipotesis terpisah, sehingga tidak ada koreksi uji jamak.
- Tidak ada klaim profitabilitas di luar periode uji.

### 2.5 Uji level plasebo (RQ2.2) dihapus

**Alasan:**

- Kekhususan PDH/PDL terhadap level acak tidak termasuk dua RQ yang baru.
- Uji ini belum diimplementasikan.
- Keputusan ini diambil sebelum data uji dibuka.

Kekhususan level dilaporkan sebagai keterbatasan dan penelitian lanjutan.

**Isian Kunci-1 yang tidak berlaku lagi:** `kunci_1_rq2.json`: `rq2_2` dan `SEED_PLASEBO`; `RENCANA_ANALISIS_RQ2.md` §3; serta Tabel R2-4, R2-5, dan Gambar R2-2.

### 2.6 RQ3 (evaluasi pengguna) dihapus

RQ3 tidak menjadi bagian tesis.

### 2.7 Perbandingan *gradient boosting* vs LSTM tidak dipakai

Rencana dari bimbingan 19 November 2026 ini tidak dipakai.

## 3. Penegasan yang tidak mengubah ketetapan

**δ = 0,1R** adalah margin untuk biaya dan sisa ketidakpastian kalibrasi. Nilainya ditetapkan lebih dahulu, tanpa disetel pada data. Grid δ {0; 0,1; 0,2; 0,3}R hanya untuk kurva cakupan–kinerja.

**Analisis pendukung lain** tetap deskriptif seperti `RENCANA_ANALISIS_RQ2.md` §4:

- proporsi kelas per tahun;
- waktu hingga *barrier*;
- kurva cakupan–kinerja;
- sensitivitas label;
- pembanding responden.

## 4. Informasi yang sudah dilihat sebelum adendum ini

Agar keputusan di atas dapat dinilai, berikut hal yang sudah diketahui peneliti saat adendum ditulis. **Data uji belum dibuka.** Semua kode pemodelan hanya memuat data lewat `load_model_csv`, yang menolak data uji.

**Hasil pengembangan:**

- validasi *walk-forward* 2019–2023 (log loss OOF dan selisih terhadap B0/B1);
- kalibrasi τ pada 2024;
- pratinjau aturan keputusan pada 2024 dan OOF (`laporan_model_akhir.md` §4).

**Perkiraan kuat uji (25 November 2026)** dari prediksi OOF 2019–2023:

- Datanya 1.152 *event* dalam 261 minggu. Peluang OOF belum dikalibrasi τ.
- *m*rev dan *m*cont dihitung dari seluruh Netral dev.
- Interval dihitung dengan *bootstrap* klaster minggu 2.000×, lalu setengah-lebarnya diskalakan ke data uji (sekitar 82 minggu) dengan faktor √(261/82) ≈ 1,78.

| Ukuran (OOF, δ = 0,1R) | Nilai | Perkiraan setengah-lebar interval di data uji |
| --- | --- | --- |
| *D*0 (model − B0) | −0,0465 (IK95 −0,0620 s.d. −0,0305) | ±0,028 |
| *D*1 (model − B1) | −0,0445 (IK95 −0,0599 s.d. −0,0284) | ±0,028 |
| ΔR̄ SM − S0 pada *S* = 0 / 0,25 / 1,00 | +0,040 / +0,078 / +0,190 | ±0,12 |
| ΔR̄ SM − S1 pada *S* = 0 / 0,25 / 1,00 | +0,001 / −0,022 / −0,089 | ±0,07 |
| *S** SM / S1 / S0 | 0,118 / 0,508 USD / tidak impas | – |

Angka-angka ini **tidak** dipakai untuk mengubah δ, pembanding S1, grid *spread*, atau kriteria "lebih baik". Satu-satunya tambahan setelah angka ini terlihat adalah aturan tafsir "tidak dapat disimpulkan" (§2.4). Aturan itu tidak mengubah kapan RQ2 dinyatakan didukung.

## 5. Keluaran yang dilaporkan (menggantikan `RENCANA_ANALISIS_RQ2.md` §6)

- **Tabel R1-1:** *log loss*, Brier, dan ECE per kelas untuk model, B0, dan B1 pada data uji, beserta rincian PDH/PDL.
- **Tabel R1-2:** *D*0, *D*1, interval 95%, proporsi ulangan dengan *D*k* < 0, dan keputusan RQ1.
- **Tabel R1-3:** metrik klasifikasi pendukung (§2.3).
- **Gambar R1-1:** diagram reliabilitas 2024 (sebelum/sesudah τ) dan data uji.
- **Tabel R1-4:** kestabilan *walk-forward* per tahun (pengembangan; bukan bukti keunggulan).
- **Tabel R2-1:** SM, S0, dan S1 pada setiap *S*: cakupan, jumlah transaksi, R bersih per *event* dan per transaksi, ΔR̄(*S*) beserta interval 95%, tafsir per selisih, dan keputusan RQ2.
- **Tabel R2-2:** *S** setiap strategi dan *S**Δ untuk SM − S0 dan SM − S1.
- **Gambar R2-1:** R kumulatif SM, S0, dan S1 pada data uji.
- **Gambar R2-2:** kurva cakupan–kinerja δ.
- **Tabel R2-3 dan seterusnya:** analisis pendukung §3.
