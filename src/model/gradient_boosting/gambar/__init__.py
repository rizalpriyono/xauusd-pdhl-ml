"""Tahap `gambar`: 22 visualisasi PNG untuk dokumentasi proses pemodelan (Proses 0–6).

Semua gambar dibuat dari keluaran tahap sebelumnya (`runs/<config>/data/`,
`runs/<config>/model/gradient_boosting/{data,output,model}/`) dan hanya memakai data dev dan calib
lewat `load_model_csv`. Tidak ada model yang dilatih ulang dan tidak ada parameter yang dipilih di
sini. Nilai harapan, rekomendasi, dan ringkasan strategi memakai fungsi `model_akhir` yang sama.
Satu-satunya pelatihan ulang ada di `kurva_belajar`: model lipatan walk-forward dengan parameter
yang sama, dicek harus menghasilkan skor walk-forward yang identik.

Modul:
    gaya         palet tervalidasi (skill dataviz), format angka Indonesia, penyimpanan PNG
    data         memuat masukan dan ringkasan murni yang bisa dites tanpa matplotlib
    proses_0_2   data berlabel, rekayasa fitur, pembanding (6 gambar)
    proses_3_4   walk-forward dan model akhir (6 gambar)
    kurva_belajar   log loss dan akurasi per iterasi boosting pada lipatan walk-forward
    proses_3_kurva  kurva belajar latih vs validasi (3 gambar)
    proses_5_6   kalibrasi τ dan aturan keputusan (7 gambar)

Butuh matplotlib (`pip install -e '.[viz]'`) dan LightGBM untuk gambar struktur model.
Keluaran: `runs/<config>/model/gradient_boosting/gambar/*.png`.
"""
import shutil

from . import data, gaya, kurva_belajar, proses_0_2, proses_3_4, proses_3_kurva, proses_5_6

DAFTAR = proses_0_2.DAFTAR + proses_3_4.DAFTAR + proses_3_kurva.DAFTAR + proses_5_6.DAFTAR


def _impor_matplotlib():
    try:
        import matplotlib
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(f"{exc.name} belum terpasang. Pasang dependensi visualisasi: "
                                  "`pip install -e '.[viz]'`") from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def run_gambar(paths, kunci):
    """Buat semua gambar; mengembalikan folder dan daftar nama berkas yang ditulis.

    Gambar dibuat di folder sementara dan baru dipindah ke `paths.gambar` bila semuanya berhasil,
    sehingga kegagalan di tengah jalan tidak meninggalkan campuran PNG lama dan baru."""
    plt = _impor_matplotlib()
    dasar = data.muat(paths, kunci)
    D = {**dasar, "struktur": data.struktur_model(dasar["model_path"]),
         "kurva": kurva_belajar.kurva_terpilih(dasar)}
    sementara = paths.gambar.with_name(paths.gambar.name + ".sementara")
    shutil.rmtree(sementara, ignore_errors=True)
    try:
        with plt.rc_context(gaya.RC):
            berkas = [fungsi(D, plt, sementara / nama).name for nama, fungsi in DAFTAR]
    except BaseException:
        shutil.rmtree(sementara, ignore_errors=True)
        raise
    finally:
        plt.close("all")
    paths.gambar.mkdir(parents=True, exist_ok=True)
    for nama in berkas:
        (sementara / nama).replace(paths.gambar / nama)
    sementara.rmdir()
    return {"folder": str(paths.gambar), "berkas": berkas}


def ringkas(hasil):
    """Baris ringkasan untuk konsol."""
    return f"Gambar: {len(hasil['berkas'])} PNG di {hasil['folder']}"
