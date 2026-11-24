# Spesifikasi Data PDH/PDL XAUUSD H1

Rujukan pasal mengikuti *Proposal Final v2* (tesis). Parameter label (a = 2, b = 1, T = 8, k = 0) dibaca dari `configs/tesis.json`.

## 1. Keputusan desain (tahap events)

| Pasal | Implementasi |
|---|---|
| IV.II.1 Sesi 17.00–17.00 NY | `session_date` adalah tanggal NY dari (waktu buka bar + 7 jam). Dengan begitu, bar Minggu pukul 18.00 masuk ke sesi Senin dan tidak ada sesi hari Minggu. |
| IV.II.1 Konversi zona waktu | Sumber data **UTC** → America/New_York (DST ditangani otomatis). Waktu NY disimpan sebagai string *wall-clock*. |
| IV.II.1 Verifikasi jeda 17.00–18.00 | Seluruh 42 minggu di sekitar 21 pergantian DST (2016–2026) berstatus PASS: setiap sesi ditutup paling lambat pada bar 16.00 dan tidak ada bar penuh (> 1 menit) di jam 17.00 pada data mentah. Dua minggu diberi catatan *data gap* (lihat "Temuan data"). |
| IV.II.2 (1) *First touch* | Bar pertama dalam sesi dengan `high ≥ PDH` / `low ≤ PDL`. Satu sesi menghasilkan paling banyak dua event. |
| IV.II.2 (2) Dibuka di luar level | `open` bar pertama sesi `> PDH` / `< PDL` menghasilkan status `opened_beyond` dan tidak ada event. |
| IV.II.2 (3) Waktu keputusan | `decision_bar_idx = touch_bar_idx + k`, dan `entry_price` adalah *close* bar tersebut. |
| IV.II.2 (4) Akhir pekan | Event dikeluarkan bila `decision_bar_idx + T > bar terakhir minggu itu`. Minggu dihitung mulai Senin dari `session_date`. |
| IV.II.3 ATR | `atr14_h1`: ATR(14) Wilder H1 sampai dengan bar keputusan. `atr14_d_prev`: ATR(14) Wilder harian dari sesi-sesi yang sudah selesai. Kedua ATR dihitung atas seluruh riwayat bronze sejak 2009-11-16 (setelah pembersihan yang sama), lalu dipotong ke silver mulai 2015-01-01, sehingga tidak ada nilai kosong sejak bar/sesi silver pertama. |
| Tambahan: `data_gap` | Event dikeluarkan bila ada bar hilang (a) di sesi event sebelum atau pada bar keputusan, (b) di sesi sebelumnya yang menjadi sumber PDH/PDL, atau (c) di dalam jendela label. Bar hilang = celah > 1 jam di tengah sesi, atau sesi tidak dibuka pukul 18.00. Aturan ini tidak ada di proposal, ditetapkan sebelum label dihitung, dan perlu ditambahkan ke IV.II.2. |
| IV.II.4 Pembagian data | Berdasarkan `session_date`: dev 2016–2023, calib 2024, test Jan 2025 – Jul 2026. |

## 2. Temuan data mentah

- **Skema.** Header CSV hanya memuat 6 nama, padahal setiap baris berisi 7 kolom. Kolom ke-6 adalah `BarMinutes` (jumlah menit bertick, umumnya 60) dan kolom ke-7 adalah `TickVolume`.
- **Zona waktu.** Data dicatat dalam UTC. Sejak 2018 tidak ada satu pun bar Senin–Kamis pada pukul 17.00 NY, baik sebelum maupun sesudah DST.
- **Bar di jam jeda.** Ada 126 bar pada pukul 17.00 NY di periode 2015–2017. Hampir semuanya bar 1 menit (tick datar O = H = L = C pada 2016–2017), jadi semuanya dibuang. Bila dibiarkan, bar-bar ini akan menjadi *candle* pertama sesi dan merusak aturan "dibuka di luar level".
- **Libur.** Tidak ada sesi pada Good Friday, Natal, dan Tahun Baru. Hari libur AS ditutup lebih awal (sesi 19–22 bar). Kedua kondisi ini hanya diberi penanda dan tidak dikeluarkan, karena proposal final tidak memiliki aturan sesi tidak lengkap. Draf `bab-iv-research-methodology-revisi.md` masih memuat aturan "< 20 bar dikeluarkan". Jika aturan itu ingin dipakai, saring dengan `prev_session_partial` dan `session_partial`.
- **Bar hilang (`data_gap`).** Sesi 2025-03-10 baru dibuka Senin pukul 00.00 (6 bar Minggu malam hilang), dan sesi 2017-03-13 baru dibuka pukul 19.00 (1 bar hilang). Jeda hariannya tetap pada 17.00, jadi ini bukan pergeseran zona waktu. Selain itu ada 4 bar yang hilang di tengah sesi (2016-09-02, 2019-07-04, 2020-08-12, 2023-11-29). Bar hilang tidak diimputasi dan tidak diisi dari penyedia lain. Sebagai gantinya, 7 event yang sentuhan pertama atau PDH/PDL-nya tidak bisa dipastikan dikeluarkan: 5 di dev dan 2 di test (2025-03-10_PDL dan 2025-03-11_PDH). Satu event lain (2025-03-07_PDH) juga kena `data_gap`, tetapi event itu memang sudah keluar karena melintasi akhir pekan.
- **Jumlah event lebih sedikit daripada perkiraan Tabel 4.4.** Rata-ratanya 0,88 event valid per sesi, bukan 1,1. Rinciannya: dev 1.841 (perkiraan ± 2.200), calib 227 (± 280), dan test 341 (± 450), total 2.409. Angka di Tabel 4.4 perlu diperbarui.

- **Pemanasan dari riwayat sebelum 2015 (adendum 2026-11-24).** Sebelumnya bar sebelum 2015-01-01 dibuang sebelum indikator dihitung. Akibatnya `atr14_h1` kosong pada 13 bar pertama 2015, `atr14_d`/`atr14_d_prev` kosong pada 13/14 sesi pertama, dan PDH/PDL sesi 2015-01-02 kosong. Kini riwayat 2009–2014 dipakai untuk TR/ATR H1, ATR harian, `hours_since_prev` bar pertama, dan PDH/PDL sesi pertama, tetapi tidak masuk silver. ATR H1 sudah identik dengan nilai lama sejak 2015-01-30. ATR harian identik sejak 2016-09-23, dengan selisih `atr14_d_prev` di event dev 2016 maksimum 4,6e-10 relatif. Jumlah event, label, dan seluruh data uji tidak berubah. Rinciannya ada di `references/tesis.json` → `adendum`.

## 3. Kolom penting `gold_pdhl_events` / `events_*.csv`

| Kolom | Arti |
|---|---|
| `event_id` | `<session_date>_<PDH/PDL>` |
| `direction_d` | +1 untuk PDH dan −1 untuk PDL (faktor arah fitur, IV.II.5) |
| `touch_*` | Bar sentuhan pertama. `touch_ts_ny` adalah waktu **buka** bar. |
| `decision_ts_ny`, `entry_price` | Bar keputusan (sentuhan + k). Keputusan diambil saat bar ini ditutup. |
| `atr14_h1`, `atr14_d_prev` | Satuan normalisasi untuk *barrier*, fitur, dan level plasebo |
| `bars_to_week_close` | Jumlah bar dari keputusan sampai penutupan minggu. Jendela label tidak melintasi akhir pekan bila nilainya ≥ T. |
| `window_max_gap_h` | Celah waktu terbesar di dalam jendela label (2 jam = jeda harian normal) |
| `label_window_crosses_split` | Jendela label masuk ke bagian data berikutnya (saat ini 0 kasus) |
| `gap_before_decision`, `gap_in_prev_session`, `gap_in_label_window` | Kondisi (a), (b), dan (c) dari aturan `data_gap` |
| `exclusion_reason`, `is_valid` | `window_crosses_week_close`, `insufficient_future_bars`, dan/atau `data_gap` (dipisah `;`). `is_valid` = tidak ada alasan. |

## 4. Pembagian data dan kunci uji (tahap split, IV.II.4)

- Pembagian berdasarkan `session_date`: dev 2016–2023, calib 2024, test 2025-01-02 – 2026-07-31.
- `data/events_dev.csv` dan `data/events_calib.csv` hanya berisi event **valid**. Data uji berisi
  **semua** event (beserta `is_valid`) agar bisa diaudit pada eksekusi tunggal.
- `data/bars_devcalib.csv` dan `data/sessions_devcalib.csv` memuat semua bar/sesi s.d. 2024-12-31,
  termasuk pemanasan 2015, sehingga `bar_idx` = indeks baris (prefiks kontinu). Bar uji melanjutkan
  `bar_idx` tepat sesudahnya.
- Kunci uji ditulis sekali lewat folder `.tmp` lalu di-rename (atomik), berkas 444 dan folder 555,
  dengan manifest SHA-256 bertanggal. Run berikutnya hanya memverifikasi; isi berbeda →
  `LockViolation`, dan `data/` tidak ditimpa.
- Guard `load_model_csv` menolak path di folder mana pun yang namanya diawali `kunci_uji`, menolak
  berkas tanpa kolom `session_date`, dan menolak baris dengan `session_date` > 2024-12-31.

## 5. Definisi label (tahap labels, IV.II.3)

Setiap event diperlakukan sebagai kandidat posisi *reversal*: *sell* pada PDH (d = +1) dan *buy* pada PDL (d = −1), dibuka pada close *candle* keputusan.

- target = entry − d·a·ATR, stop = entry + d·b·ATR.
- Barrier diperiksa pada *candle* ke-1 sampai ke-T sesudah keputusan.
- Stop tersentuh, termasuk bila bersamaan dengan target dalam satu *candle* → *Continuation*. Hanya target tersentuh → *Reversal*. Tidak ada yang tersentuh → Netral.
- R mengikuti Tabel 4.3: Reversal (+a/b, −1), Continuation (−1, +b/a), Netral (−d·Δ/(b·ATR), d·Δ/(a·ATR)) dengan Δ = close ke-T − entry. Tidak bertransaksi = 0.

### Kolom `labels_{dev,calib}.csv`

| Kolom | Arti |
|---|---|
| `event_id`, `split`, `session_date`, `level_type`, `direction_d` | Identitas event (urutan baris sama dengan `events_*.csv`) |
| `decision_bar_idx`, `decision_ts_ny`, `entry_price`, `atr14_h1` | Bar keputusan, harga masuk, satuan ATR |
| `a`, `b`, `T`, `k` | Parameter label dari konfigurasi |
| `target_price`, `stop_price` | *Barrier* posisi *reversal* |
| `label` | `Reversal`, `Continuation`, atau `Netral` |
| `barrier_bar_offset`, `barrier_ts_ny` | *Candle* ke-berapa (1..T) *barrier* pertama tersentuh; kosong bila Netral |
| `both_same_bar` | Target dan stop tersentuh di *candle* yang sama (dilabeli Continuation) |
| `close_T` | Close *candle* ke-T (dasar R Netral) |
| `label_end_bar_idx`, `label_end_session_date`, `label_crosses_year` | Akhir jendela label; dipakai untuk *purging*/embargo walk-forward |
| `R_rev`, `R_cont`, `R_none` | Hasil dalam satuan R bila memilih *reversal*, *continuation*, atau tidak bertransaksi |
