# Data mentah

`XAUUSD_H1.csv` disimpan di repo apa adanya, sehingga clone (termasuk di Google Colab) langsung bisa
menjalankan pipeline. `.gitattributes` menandainya `binary` agar git tidak mengubah akhir baris.

| | |
|---|---|
| Berkas | `XAUUSD_H1.csv` (6,1 MB) |
| Sumber | Ekspor data H1 XAUUSD dari Blue Capital Trading (MetaTrader), disertakan untuk reproduksi penelitian |
| SHA-256 | `e4ae9238f177fc7d09555f8e23762456641636fa510b321105745f58e840bfdb` |
| Rentang | 16 Nov 2009 – 31 Jul 2026, 100.000 bar. Pipeline membuang bar sebelum 2015; 2015 hanya untuk pemanasan ATR |
| Format | UTC, dipisah tab, header 6 nama tetapi 7 kolom per baris: `Time, Open, High, Low, Close, BarMinutes, TickVolume` |

Berkas ini juga memuat periode uji (2025 – Jul 2026). Kerahasiaan data uji dijaga di tingkat label dan
evaluasi: pipeline mengunci event/bar uji di `runs/tesis/kunci_uji/`, dan kode pemodelan hanya memuat
data lewat `load_model_csv`, yang menolak baris sesudah 2024-12-31.

Cek berkas:

```bash
shasum -a 256 data/raw/XAUUSD_H1.csv
```

Bila hash berbeda, pipeline tetap berjalan tetapi mencetak peringatan, dan `verify` akan menandai
berkas keluaran sebagai BEDA.
