# Rencana Analisis RQ2 (Kunci-1, Adendum 2)

**Tanggal kunci:** 17 November 2026 (minggu 8), sebelum validasi *walk-forward* (RM-24) dan sebelum data uji dibuka
**Dasar:** Proposal Final v2, IV.II.6–IV.II.8 dan IV.IV
**Parameter terbaca mesin:** `kunci_1_rq2.json` (dimuat lewat `common.load_kunci1_rq2`)

Dokumen ini tidak menambah hipotesis. Isinya mengoperasionalkan IV.IV agar setiap pilihan analisis sudah tetap sebelum hasil terlihat: definisi metrik, *bootstrap*, *seed*, konstruksi plasebo, dan analisis pendukung. Perubahan sesudah tanggal kunci ditulis sebagai amandemen bertanggal beserta alasannya. Analisis yang tidak tercantum di sini dilabeli **eksploratif**.

## 1. Data dan unit analisis

- Unit analisis adalah *event* valid (*first touch*, *k* = 0) dengan label Kunci-1: *a* = 2, *b* = 1, *T* = 8.
- Fitur yang dipakai adalah 15 fitur pada Adendum 1 (`kunci_1_fitur.json`).
- Data uji tertutup terdiri atas 341 *event* valid (Januari 2025 – Juli 2026). *Event* yang jendela labelnya melewati akhir data dikeluarkan, dan jumlahnya dilaporkan.
- Urutan kelas di semua berkas adalah [Reversal, Continuation, Netral].

## 2. RQ2.1: Keterprediksian terhadap *baseline* (IV.IV.1)

### 2.1 Prediktor

| Prediktor | Estimasi | Kalibrasi |
| --- | --- | --- |
| Model akhir | Konfigurasi terpilih (IV.II.6), dilatih pada dev 2016–2023 | *Temperature* τ pada 2024 |
| B0 | Frekuensi kelas pada dev | Tidak |
| B1 | Frekuensi kelas pada dev, dipisah menurut f13 ≥ 2/3. Bila satu sel tidak memiliki salah satu kelas, sel itu diberi koreksi Laplace (+1 per kelas) | Tidak |
| B2 | Regresi logistik multinomial atas f01, f06, f09; L2 dengan C = 1 (tidak disetel); f06 dan f09 distandardisasi dengan rata-rata dan simpangan baku dev | τ sendiri pada 2024 |

### 2.2 Metrik utama dan uji

- **Log loss.** ℓ*i* = −ln *p*(*i*, *y*i), memakai logaritma natural. Probabilitas dipotong ke [10⁻¹⁵, 1] lalu dinormalkan ulang.
- **Selisih berpasangan.** Untuk setiap *baseline* B*k*: *D*k* = rata-rata (ℓ*i*(model) − ℓ*i*(B*k*)) atas semua *event* uji.
- ***Bootstrap* klaster.** Klaster adalah minggu kalender (`week_start`, Senin, waktu New York, dari `session_date`). Minggu diambil ulang dengan pengembalian sebanyak 2.000 kali dengan *seed* `SEED_BOOT` = 20261117. Statistik setiap ulangan adalah rata-rata per *event*. Indeks ulangan yang sama dipakai untuk B0, B1, dan B2.
- **Interval.** Interval persentil 95% (2,5–97,5).
- **Keputusan.** RQ2.1 didukung hanya bila batas atas interval < 0 untuk **ketiga** *baseline*. Ini sama dengan uji satu sisi α = 0,025 per perbandingan dalam kerangka *intersection–union*, sehingga koreksi uji jamak tidak diperlukan.
- **Pelaporan.** *D*k*, interval 95%, dan proporsi ulangan dengan *D*k* < 0 dilaporkan untuk setiap B*k*.

### 2.3 Metrik pendukung (deskriptif)

- ***Brier score* multikelas:** (1/*N*) Σ*i* Σ*c* (*p*ic − *y*ic)², dengan rentang 0–2.
- **ECE per kelas:** satu-lawan-semua, 10 bin selebar 0,1 pada [0, 1], dan setiap bin dibobot jumlah *event*-nya.
- **Diagram reliabilitas:** memakai bin yang sama, dengan jumlah *event* per bin ditampilkan.
- **Rincian per jenis level:** *log loss* untuk PDH dan PDL ditampilkan terpisah.

### 2.4 Kestabilan *walk-forward* 2019–2023 (bukan bukti keunggulan)

Prediksi *out-of-fold* model terpilih dibandingkan dengan B0, B1, dan B2 yang diestimasi ulang pada data latih setiap lipatan. Probabilitas *walk-forward* tidak dikalibrasi, karena kalibrasi hanya memakai 2024. *D*k* dilaporkan per tahun dan gabungan.

## 3. RQ2.2: Kekhususan PDH/PDL melalui uji level plasebo (IV.IV.2)

### 3.1 Konstruksi level plasebo

- **Cakupan.** Setiap sesi dengan PDH, PDL, dan ATR harian (`atr14_d_prev`) yang tersedia mendapat level plasebo. Set ke-*j* (*j* = 1, …, 1.000) memakai pembangkit acak dengan *seed* `SEED_PLASEBO` + *j* (`SEED_PLASEBO` = 20261118). Sesi diproses berurutan menurut waktu, dengan U diundi lebih dahulu lalu L.
- **Level atas.** U = PDH + *s* · *u* · ATR*d*, dengan *s* diundi dari {−1, +1} dan *u* ~ Seragam[0,25; 1,0]. U diterima bila U > (PDH + PDL)/2, |U − PDH| ≥ 0,25 ATR*d*, dan |U − PDL| ≥ 0,25 ATR*d*. Bila tidak memenuhi, (*s*, *u*) diundi ulang bersama, paling banyak 1.000 kali. Pengundian hampir pasti berhasil karena *s* = +1 selalu memenuhi syarat.
- **Level bawah.** L = PDL + *s*′ · *u*′ · ATR*d* dicerminkan dari U: L diterima bila L < (PDH + PDL)/2 dan jaraknya ≥ 0,25 ATR*d* dari PDL maupun PDH.
- **Event, label, dan fitur.** Aturannya identik dengan IV.II.2–IV.II.5, termasuk aturan `data_gap`, *first touch*, pembukaan di luar level, dan jendela label yang melintasi akhir pekan. U diperlakukan seperti PDH (*d* = +1, f01 = 1) dan L seperti PDL. Fitur yang bergantung pada level (f03, f09, f10) memakai level plasebo. Fitur lain dihitung sama persis, termasuk f02 yang tetap memakai rentang PDH − PDL asli.

### 3.2 Statistik utama

- **Proporsi *reversal*.** P*rev* = jumlah label *Reversal* / jumlah *event* valid berlabel lengkap, dengan PDH dan PDL digabung. Periodenya Januari 2016 – Juli 2026, dihitung dalam eksekusi tunggal. Purging antarbagian tidak berlaku di sini karena statistik ini tidak melibatkan pelatihan.
- **Nilai *p*.** P̄ adalah rata-rata P*rev* dari 1.000 set plasebo, dan *p* = (1 + #{*j* : |P*rev*(*j*) − P̄| ≥ |P*rev*(PDH/PDL) − P̄|}) / 1.001. H0 ditolak bila *p* < 0,05.
- ***Effect size*.** Dilaporkan selisih P*rev*(PDH/PDL) − P̄ dalam poin persentase, persentilnya dalam sebaran plasebo, dan *z* = selisih / simpangan baku plasebo.
- **Deskriptif tambahan.** Proporsi ketiga kelas, serta rincian PDH vs U dan PDL vs L.

### 3.3 Statistik pendukung: *skill score*

- **Set yang dipakai.** Set plasebo *j* = 1, …, 20 (dua puluh set pertama menurut indeks, ditetapkan sebelum hasil terlihat).
- **Prosedur per set.** Untuk setiap set, konfigurasi model akhir yang terpilih pada data asli dilatih ulang tanpa seleksi ulang (hiperparameter dan *seed* sama) pada dev plasebo 2016–2023. Model lalu dikalibrasi τ pada plasebo 2024 dan dievaluasi pada plasebo uji. B0 diambil dari frekuensi dev plasebo.
- **Skor.** SS = 1 − *log loss*(model) / *log loss*(B0), dan SS PDH/PDL dihitung dengan cara yang sama pada data uji asli.
- **Kriteria.** SS PDH/PDL harus > SS seluruh 20 set, setara dengan *p* empiris satu sisi 1/21 ≈ 0,048. Nilai seri dihitung sebagai tidak melampaui.

### 3.4 Penafsiran

Penafsiran mengikuti IV.IV.2 tanpa perubahan: bukti kekhususan, tidak didukung, atau hasil bercampur yang dilaporkan apa adanya.

## 4. Analisis pendukung (IV.IV.3; deskriptif, tanpa uji hipotesis)

1. Proporsi setiap kelas per tahun pada data pengembangan.
2. Sebaran waktu hingga *barrier* pertama (*offset* 1–8) dan proporsi Netral pada data pengembangan.
3. **Aturan keputusan pada data uji** (δ = 0,1R; *m*rev dan *m*cont dari Netral dev):
   - cakupan (proporsi rekomendasi arah);
   - rata-rata R per *event*, dengan abstain dihitung 0;
   - rata-rata R per rekomendasi arah.

   Interval *bootstrap* klaster mingguan (`SEED_BOOT`) disertakan sebagai keterangan, bukan uji.
4. **Kurva cakupan–kinerja** untuk δ ∈ {0; 0,1; 0,2; 0,3}R pada data uji.
5. **Sensitivitas biaya *spread*** pada data uji (δ = 0,1R):
   - *S* ∈ {0; 0,15; 0,25; 0,35; 0,50; 1,00} USD per oz.
   - Biaya per keputusan dalam R: *reversal* = *S* / (*b* · ATR), *continuation* = *S* / (*a* · ATR), abstain = 0. ATR yang dipakai adalah ATR *event*.
   - Dilaporkan rata-rata R bersih per *event* dan per rekomendasi arah. Grid ini sama dengan protokol Artikel 1. Tidak ada klaim profitabilitas.
6. **Sensitivitas label** pada data pengembangan, dengan *a* ∈ {1,5; 2; 3}, *b* = 1, dan *T* ∈ {4; 8; 12} (9 kombinasi):
   - Untuk setiap kombinasi dilaporkan jumlah *event*, proporsi kelas, dan *log loss* OOF konfigurasi model terpilih (lipatan *walk-forward* sama).
   - Himpunan *event* berubah menurut *T* karena aturan akhir pekan, sehingga *n* dilaporkan per kombinasi.
   - Analisis ini tidak dipakai untuk memilih parameter.
7. **Pembanding responden.** Median responden (*a* = 3, *T* = 2; lihat Adendum 2) dicatat sebagai pembanding. *T* = 2 berada di luar grid sensitivitas dan tidak ditambahkan agar grid tetap seperti di proposal.

## 5. Konfigurasi teknis yang dikunci

Bagian ini membatasi pilihan saat implementasi RM-24 sampai RM-27. Model akhir dan τ tetap dikunci di Kunci-2.

| Hal | Ketetapan |
| --- | --- |
| Pengodean | f04 dan f05 *one-hot* untuk kedua kandidat; f01 biner; f08 dan f14 numerik |
| Nilai kosong | Dev dan calib tidak punya nilai kosong. Bila data uji punya, regresi logistik memakai median dev, sedangkan LightGBM menangani NaN secara bawaan |
| LightGBM | `objective = multiclass`, *learning rate* 0,05, grid IV.II.6, parameter lain bawaan, `seed` = `SEED_MODEL`, `deterministic = true`, `num_threads = 1` |
| Regresi logistik lengkap | Multinomial, *solver* lbfgs, `max_iter` 5.000, C ∈ {0,1; 1; 10}, standardisasi dengan statistik data latih lipatan |
| Seri pada pemilihan | Bila selisih rata-rata *log loss* ≤ 10⁻⁶, dipilih konfigurasi yang lebih sederhana: pohon lebih sedikit, kedalaman lebih kecil, *min_child_samples* lebih besar, dan untuk regresi logistik C lebih kecil |
| GB vs regresi logistik | Seperti IV.II.6: *bootstrap* klaster mingguan 2.000×, `SEED_BOOT`, batas atas < 0 |
| *Temperature scaling* | τ meminimalkan *log loss* 2024, dicari dengan *golden-section* pada [0,05; 20] atas logit (*raw score* LightGBM atau *decision function* regresi logistik) |
| *Seed* | `SEED_BOOT` = 20261117, `SEED_PLASEBO` = 20261118 (+ *j*), `SEED_MODEL` = 20261119 |
| Versi pustaka | Python, NumPy, scikit-learn, LightGBM, dan SHAP dicatat di Kunci-2 |

## 6. Keluaran RQ2 yang dilaporkan

- **Tabel R2-1:** *log loss*, Brier, dan ECE untuk model, B0, B1, dan B2 pada data uji.
- **Tabel R2-2:** *D*k* dan interval 95% untuk setiap *baseline*, beserta keputusan RQ2.1.
- **Gambar R2-1:** diagram reliabilitas 2024 (sebelum dan sesudah τ) dan data uji.
- **Tabel R2-3:** kestabilan *walk-forward* per tahun.
- **Tabel R2-4 dan Gambar R2-2:** P*rev* PDH/PDL terhadap sebaran 1.000 plasebo, dengan *p*, selisih (pp), persentil, dan *z*.
- **Tabel R2-5:** SS PDH/PDL terhadap 20 SS plasebo.
- **Tabel R2-6 sampai R2-9:** analisis pendukung bagian 4.
