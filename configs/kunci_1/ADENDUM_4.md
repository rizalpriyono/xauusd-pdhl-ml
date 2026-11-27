# Kunci-1, Adendum 4: Fitur Tambahan Berbasis Literatur (Opsi B)

**Tanggal:** 25 November 2026, sebelum Kunci-2 (direncanakan 5 Desember 2026) dan sebelum data uji dibuka
**Dasar:**

- Perumusan ulang masalah dan pertanyaan penelitian (*Penguatan Inti Permasalahan*, 21 November 2026).
- Kunci-1 Adendum 3 (25 November 2026).
- Keputusan peneliti 25 November 2026: memakai **Opsi B**.

**Parameter terbaca mesin:** `kunci_1_adendum_4.json` (strukturnya sama dengan `kunci_1_fitur.json`)
**Integritas:** `MANIFEST_KUNCI_1_ADENDUM_4.json`. Manifest dan berkas Kunci-1 sebelumnya tidak diubah.
**Status implementasi:** **belum dijalankan.** Atas instruksi peneliti, rekayasa fitur dan pemodelan belum diulang. Fitur baru belum pernah dihitung pada data mana pun.

Adendum ini **tidak** mengubah:

- parameter label (`kunci_1.json`);
- 12 fitur dasar (Tabel 4.5);
- B0, aturan B1, dan strategi S1 (rasio *wick* ≥ θ = 2/3);
- grid dan parameter teknis LightGBM, *seed*, rentang τ, δ dan grid δ, grid *spread*, *bootstrap*;
- rumusan dan kriteria RQ1 dan RQ2 (Adendum 3).

Yang berubah hanya **cara memilih fitur tambahan** dan **dua dari tiga fitur tambahan**. Bila isi adendum ini bertentangan dengan `kunci_1_fitur.json`, adendum ini yang berlaku.

## 1. Alasan

1. **Input model dan pembanding tidak lagi berasal dari sumber yang sama.** Kuesioner Tahap 1 menetapkan aturan yang mewakili heuristik *trader* (B1/S1). Bila fitur model juga diturunkan dari responden yang sama, model hanya mengulang isyarat responden lalu dibandingkan dengan responden itu sendiri.
2. **Dasar kuesioner untuk fitur lemah.** Hanya 9 responden yang lolos penyaringan. *Engulfing* dan volume masing-masing disebut 3 responden.
3. **Literatur terbaru menyediakan alternatif** yang dapat dihitung dari OHLC dan *tick volume* H1 pada *k* = 0.
4. **Bukan karena hasil.** Fitur lama sudah mengungguli B0 pada validasi *walk-forward* (*D*0 = −0,0465). Penggantian ini tidak didorong oleh kinerja yang buruk.

**Mengapa Opsi B (wick tetap).** Rasio *wick* tetap ada di model agar perbandingan model dengan B1 tetap **bersarang**: model memuat informasi yang dipakai B1, ditambah informasi lain. Bila model kalah dari B1, sebabnya jelas ada pada model, bukan pada informasi yang hilang. Opsi A (ketiga fitur diganti, termasuk rentang candle sentuhan) tidak dipakai. Dukungan literatur untuk rentang candle paling lemah, dan perbandingannya tidak lagi bersarang.

## 2. Kriteria pemilihan fitur tambahan (menggantikan IV.II.5 paragraf terakhir)

Paling banyak tiga fitur tambahan. Setiap fitur harus memenuhi semua syarat berikut:

1. dipakai sebagai fitur atau ukuran pasar dalam sekurang-kurangnya dua studi *peer-reviewed* terbit 2021–2026, atau memiliki mekanisme teoretis yang jelas terhadap *reversal*/*continuation* di level harga;
2. dapat dihitung dari OHLC atau *tick volume* H1 hanya dengan bar sampai *t*\* (*k* = 0);
3. tidak tumpang tindih dengan 12 fitur dasar;
4. ditetapkan sebelum nilainya dihitung pada data.

## 3. Fitur tambahan yang berlaku

Notasi mengikuti `docs/PEMODELAN_GRADIENT_BOOSTING.md` §3: *t* = candle sentuhan, *t*\* = candle keputusan (*k* = 0 → *t*\* = *t*), *d* = +1 (PDH) / −1 (PDL). "Bar" adalah bar H1 yang tersedia di data, berurutan menurut waktu. Bar yang hilang tidak ditambal, dan jeda akhir pekan tidak dianggap bar.

| No. | Kolom | Status | Rumus |
| --- | --- | --- | --- |
| 13 | `f13_wick_penolakan` | **Tetap** (definisi sama; dasar diganti ke literatur) | *Wick* searah *reversal* / (H*t* − L*t*). PDH: H − max(O, C); PDL: min(O, C) − L. Bernilai 0 bila H = L. |
| 14 | `f14_cmf21_d` | **Baru**, menggantikan `f14_engulfing_d` | *d* · Σ MFM*i*·V*i* / Σ V*i* atas 21 bar terakhir sampai *t*\* (termasuk *t*\*). MFM*i* = ((C*i* − L*i*) − (H*i* − C*i*)) / (H*i* − L*i*), dan 0 bila H*i* = L*i*. V = *tick volume*. Bernilai 0 bila Σ V = 0. Rentang [−1, 1]. Positif berarti tekanan menuju atau melewati level (beli di PDH, jual di PDL). |
| 15 | `f15_pe48` | **Baru**, menggantikan `f15_volume_relatif` | *Permutation entropy* ternormalisasi (Bandt & Pompe, 2002) atas 48 *log return* terakhir sampai *t*\*, *r*i = ln(C*i* / C*i*−1), dengan 49 *close*. Orde *m* = 3, tunda τ = 1, sehingga ada 46 pola ordinal. PE = −Σ *p*π ln *p*π / ln 3!, rentang [0, 1]. Nilai yang sama diurutkan menurut urutan kemunculan. Tidak bertanda *d*. |

**Pilihan jendela:**

- **CMF 21 bar** mengikuti Grądzki dkk. (2025), yang memakai CMF 21 periode sebagai fitur (diperiksa pada teks lengkap, 25 November 2026).
- **PE 48 return, *m* = 3.** Aturan praktis PE mensyaratkan jumlah pola jauh lebih besar daripada *m*!, dengan varian 5·*m*! ≤ *N* (Cuesta-Frau dkk., 2019).
  - Untuk *m* = 3, syarat itu adalah 30 pola.
  - Jendela 24 return hanya menghasilkan 22 pola, jadi tidak memenuhi syarat.
  - 48 return (46 pola) adalah kelipatan siklus harian terpendek yang memenuhi syarat. Jendela ini juga mencakup hari sebelumnya, saat PDH/PDL terbentuk.
  - *m* = 3 adalah orde terkecil yang lazim (rentang 3–7), sesuai panjang jendela yang pendek.

Jumlah fitur akhir tetap **15**. Kolom `b1_terpenuhi` tetap dihitung dari f13 untuk B1 dan S1.

## 4. Dasar literatur

| Fitur | Mekanisme | Rujukan |
| --- | --- | --- |
| f13 rasio *wick* | Penolakan harga di level. Rasio ≥ 2/3 adalah bentuk ternormalisasi rentang dari definisi *hammer*/*shooting star* (bayangan ≥ 2 × badan, bayangan lawan kecil) | Nison (1991); Morris (2006); Li dkk. (2024): panjang *upper*/*lower shadow* ternormalisasi sebagai fitur; Moser & Brauneis (2026, *IREF*): pola berbasis *shadow* pada data H1; Layegh dkk. (2026): *Hammer*/*Hanging Man* pada BTC H1; Vahidpour dkk. (2024): pola *Hammer*/*Shooting Star* sebagai fitur |
| f14 CMF-21 | Arah aliran order menjelang level: *take-profit* menumpuk di level (pembalikan), *stop-loss* tepat di luar level (kelanjutan). Aliran beli/jual diperkirakan dari posisi *close* dalam rentang bar dan volume | Grądzki dkk. (2025, *Financial Innovation*): CMF 21 dan MFI 14 sebagai fitur pada data jam; Nokar dkk. (2026, *Applied Artificial Intelligence*): *order flow imbalance* pada BTC H1; dasar teori: Easley, López de Prado & O'Hara (2012); Osler (2003, 2005) |
| f15 PE-48 | Keteraturan jalur harga menjelang sentuhan. PE rendah berarti pola berulang (lebih dapat diprediksi), PE tinggi berarti mendekati acak. Ini fitur rezim, bukan fitur arah | Nokar dkk. (2026): PE sebagai fitur pada H1; Mensi, Sensoy, Vo & Kang (2022, *NAJEF* 62): PE pada data intraday termasuk emas; Fernandes dkk. (2025, *Physica A* 664): PE dan informasi Fisher pada emas dan kripto; definisi: Bandt & Pompe (2002) |

Rujukan Li dkk. (2024), Moser & Brauneis (2026), Layegh dkk. (2026), Vahidpour dkk. (2024), Nokar dkk. (2026), dan Grądzki dkk. (2025) sudah ada di SLR v2 (`SLR_v2_IntiMasalah_PDHPDL.xlsx`, lembar Review_FullText). Mensi dkk. (2022), Fernandes dkk. (2025), Bandt & Pompe (2002), Cuesta-Frau dkk. (2019), dan Easley dkk. (2012) adalah rujukan pendukung di luar SLR.

## 5. Konsekuensi

**Tafsir RQ1.** Perbandingan model dengan B1 tetap bersarang, karena model memuat f13. RQ1 menjawab: apakah kondisi pasar dari literatur, ditambah isyarat *wick*, mengestimasi peluang lebih baik daripada heuristik *wick* saja dan peluang dasar?

**Peran kuesioner** kini hanya sebagai:

- sumber aturan yang mewakili heuristik *trader* (B1/S1);
- bukti latar belakang (rata-rata lebih dari tiga isyarat, tidak ada dua responden dengan kombinasi yang sama);
- pembanding deskriptif parameter label.

Kuesioner tidak lagi menjadi sumber fitur model.

**Isian Kunci-1 yang tidak berlaku lagi:**

- `kunci_1_fitur.json`: entri `fitur_tambahan` no. 14 dan 15, kolom `sumber`/`n` pada entri no. 13 (dasar f13 kini literatur), dan kalimat `sumber` di tingkat atas.
- Bagian `B1` di berkas itu tetap berlaku: B1 tetap bersumber dari K2 kuesioner.
- `KUNCI-1_parameter_label.md`, tabel fitur tambahan Adendum 1, menjadi catatan sejarah.

**Dokumen tesis yang perlu disesuaikan:**

- *Proposal Final v2* IV.I: diagram alur Tahap 1 → Kunci-1 dan paragraf matriks persyaratan. "Fitur tambahan" tidak lagi menjadi keluaran kuesioner.
- IV.II.5 paragraf terakhir diganti dengan kriteria §2. Kalimat bahwa jarak ke angka bulat (Quiroga-Garcia dkk., 2022) dan *pivot point* "hanya dimasukkan bila disebut responden" ditulis ulang. Keduanya tidak memenuhi kriteria §2 saat ini dan tidak dipakai.
- Bagian data: kalimat "*tick volume* hanya digunakan apabila terpilih sebagai fitur tambahan dari RQ1" diganti. *Tick volume* dipakai f14. Sifatnya yang spesifik penyedia dicatat sebagai keterbatasan.
- Bab II/III: tambahkan literatur aliran order (CMF/*order flow imbalance*) dan *permutation entropy*.

**Kode (belum dikerjakan):**

- `src/model/gradient_boosting/fitur.py`: rumus f14/f15 baru lewat `PointInTime`, termasuk uji pemotongan.
- `kunci.py`: memuat `kunci_1_adendum_4.json` dan memeriksa `MANIFEST_KUNCI_1_ADENDUM_4.json`.
- Diperbarui: `laporan_fitur.py`, gambar fitur, tes `test_gb_*`, dan `docs/PEMODELAN_GRADIENT_BOOSTING.md`.
- Lalu tahap fitur → pembanding → *walk-forward* → model akhir dijalankan ulang. Hasil B0 dan B1 tidak berubah karena tidak bergantung pada fitur model.

**Hasil lama** pemodelan dengan fitur 14–15 lama (22–24 November 2026, `runs/tesis/model/gradient_boosting/`) disimpan sebagai catatan dan digantikan oleh hasil run ulang.

## 6. Informasi yang sudah dilihat sebelum adendum ini

**Data uji belum dibuka.** Yang sudah diketahui peneliti saat adendum ini ditulis:

- hasil pengembangan dengan 15 fitur lama: *walk-forward* 2019–2023, kalibrasi τ 2024, pratinjau aturan keputusan (Adendum 3 §4);
- run eksplorasi label 4/2/24 dengan 15 fitur lama pada dev/kalibrasi (`runs/label_4_2_24/`, 25 November 2026);
- **belum diketahui:** nilai f14/f15 baru pada data mana pun, hubungannya dengan label, dan kinerja model dengan fitur baru. Rumus dan jendela pada §3 ditetapkan sebelum perhitungan apa pun.

## 7. Rujukan baru

Bandt, C., & Pompe, B. (2002). Permutation entropy: A natural complexity measure for time series. *Physical Review Letters*, *88*(17), 174102. https://doi.org/10.1103/PhysRevLett.88.174102

Cuesta-Frau, D., Murillo-Escobar, J. P., Orrego, D. A., & Delgado-Trejos, E. (2019). Embedded dimension and time series length. Practical influence on permutation entropy and its applications. *Entropy*, *21*(4), 385. https://doi.org/10.3390/e21040385

Easley, D., López de Prado, M. M., & O'Hara, M. (2012). Flow toxicity and liquidity in a high-frequency world. *The Review of Financial Studies*, *25*(5), 1457–1493. https://doi.org/10.1093/rfs/hhs053

Fernandes, L. H. S., Figueirôa, J. R. A., Martins, C. M. F., & Martins, A. M. F. (2025). The battle of informational efficiency: Cryptocurrencies vs. classical assets. *Physica A: Statistical Mechanics and its Applications*, *664*, 130427. https://doi.org/10.1016/j.physa.2025.130427

Mensi, W., Sensoy, A., Vo, X. V., & Kang, S. H. (2022). Pricing efficiency and asymmetric multifractality of major asset classes before and during COVID-19 crisis. *The North American Journal of Economics and Finance*, *62*, 101773. https://doi.org/10.1016/j.najef.2022.101773

Rujukan lain ada di SLR v2 dan di `Rujukan Nilai Cadangan.md`. Kelima DOI di atas sudah dicek di Crossref (25 November 2026). **Perlu dicek sebelum dikutip:** Nison (1991) dan Morris (2006) belum dibaca langsung.
