"""Gambar kurva belajar Proses 3: log loss dan akurasi per iterasi boosting, latih vs validasi.

Data kurva dihitung oleh `kurva_belajar.kurva_terpilih` (sudah dicek sama dengan skor walk-forward).
Garis tegak = jumlah pohon terpilih; kurva tidak dipakai untuk memilih jumlah pohon.
"""
from .gaya import (KATEGORI, PREDIKTOR_WARNA, SURFACE, TEKS, TEKS_2, angka, bingkai,
                   format_sumbu, persen, simpan)

LATIH, VALIDASI = KATEGORI[1], KATEGORI[0]
CATATAN = ("Deskriptif; lipatan walk-forward dev 2016–2023 saja. Kurva tidak dipakai untuk memilih "
           "jumlah pohon (konfigurasi terkunci).")


def _penanda_terpilih(ax, n, y, teks):
    ax.axvline(n, color=TEKS, lw=1)
    ax.scatter([n], [y], s=64, color=VALIDASI, edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.annotate(teks, (n, y), xytext=(10, 18), textcoords="offset points", fontsize=8, color=TEKS)


def kurva_logloss(D, plt, path):
    """P3: log loss latih vs validasi gabungan per iterasi, dengan acuan B0 OOF."""
    k = D["kurva"]
    n = k["terpilih"]["n_estimators"]
    x = list(range(1, len(k["ll_val"]) + 1))
    i_min = min(range(len(x)), key=k["ll_val"].__getitem__)
    b0 = D["recon"]["pembanding"]["oof"]["log_loss_b0"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(x, k["ll_latih"], color=LATIH, label="Latih (data yang dipakai melatih)")
    ax.plot(x, k["ll_val"], color=VALIDASI, label="Validasi (tahun berikutnya, OOF)")
    ax.fill_between(x, k["ll_latih"], k["ll_val"], color=VALIDASI, alpha=0.08, lw=0)
    ax.axhline(b0, color=PREDIKTOR_WARNA["B0"], lw=1.2, ls=(0, (1, 2)),
               label=f"B0 OOF ({angka(b0, 4)})")
    _penanda_terpilih(ax, n, k["ll_val"][n - 1],
                      f"terpilih: {n} pohon → {angka(k['ll_val'][n - 1], 4)}")
    ax.scatter([x[i_min]], [k["ll_val"][i_min]], s=48, color=SURFACE, edgecolor=VALIDASI,
               linewidth=2, zorder=3)
    ax.annotate(f"terendah: iterasi {x[i_min]} → {angka(k['ll_val'][i_min], 4)}",
                (x[i_min], k["ll_val"][i_min]), xytext=(-40, 22), textcoords="offset points",
                ha="center", fontsize=8, color=TEKS,
                arrowprops={"arrowstyle": "-", "color": TEKS_2, "lw": 0.8})
    ujung = len(x) - 1
    ax.annotate(f"jarak latih–validasi\n{angka(k['ll_val'][n - 1] - k['ll_latih'][n - 1], 3)} di "
                f"{n} pohon, {angka(k['ll_val'][ujung] - k['ll_latih'][ujung], 3)} di {x[ujung]}",
                (x[ujung], (k["ll_val"][ujung] + k["ll_latih"][ujung]) / 2), xytext=(-8, 0),
                textcoords="offset points", ha="right", va="center", fontsize=8, color=TEKS_2)
    format_sumbu(ax, 2)
    ax.set_xlim(0, len(x) + 5)
    ax.set_xlabel("Iterasi boosting (jumlah pohon per kelas)")
    ax.set_ylabel("Log loss (makin kecil makin baik)")
    ax.legend(loc="lower left")
    t = k["terpilih"]
    bingkai(fig, f"Proses 3 · Kurva belajar log loss: kedalaman {t['max_depth']}, minimal "
                 f"{t['min_child_samples']} per daun",
            "Latih terus turun, validasi berhenti membaik lalu naik: tanda overfitting. "
            f"Rata-rata {len(k['lipatan'])} lipatan walk-forward, dibobot jumlah sentuhan.",
            CATATAN)
    return simpan(fig, path, plt)


def kurva_per_tahun(D, plt, path):
    """P3: log loss latih dan validasi per iterasi untuk setiap tahun validasi."""
    k = D["kurva"]
    n = k["terpilih"]["n_estimators"]
    fig, axs = plt.subplots(1, len(k["lipatan"]), figsize=(12, 3.9), sharey=True)
    for ax, f in zip(axs, k["lipatan"]):
        x = list(range(1, len(f["ll_val"]) + 1))
        i_min = min(range(len(x)), key=f["ll_val"].__getitem__)
        ax.plot(x, f["ll_latih"], color=LATIH, lw=1.6, label="Latih")
        ax.plot(x, f["ll_val"], color=VALIDASI, lw=1.6, label="Validasi")
        ax.axvline(n, color=TEKS, lw=1)
        ax.scatter([x[i_min]], [f["ll_val"][i_min]], s=36, color=SURFACE, edgecolor=VALIDASI,
                   linewidth=2, zorder=3)
        ax.set_title(f"Validasi {f['tahun']} · terendah it. {x[i_min]}", fontsize=9)
        ax.set_xlabel("Iterasi")
        format_sumbu(ax, 2)
    axs[0].set_ylabel("Log loss")
    axs[0].legend(loc="lower left", fontsize=7.5)
    bingkai(fig, "Proses 3 · Kurva belajar per tahun validasi",
            f"Garis tegak = {n} pohon (terpilih); titik = iterasi dengan log loss validasi "
            "terendah "
            "di tahun itu.", CATATAN)
    return simpan(fig, path, plt)


def kurva_akurasi(D, plt, path):
    """P3: akurasi latih vs validasi gabungan per iterasi, dengan acuan B0/B1."""
    k = D["kurva"]
    n = k["terpilih"]["n_estimators"]
    x = list(range(1, len(k["akurasi_val"]) + 1))
    acuan = k["akurasi_pembanding"]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(x, k["akurasi_latih"], color=LATIH, label="Latih")
    ax.plot(x, k["akurasi_val"], color=VALIDASI, label="Validasi (OOF)")
    sama = abs(acuan["B0"] - acuan["B1"]) < 1e-12
    label_acuan = (f"B0 dan B1 ({persen(acuan['B0'])}; keduanya selalu menebak kelas terbanyak)"
                   if sama else f"B0 ({persen(acuan['B0'])})")
    ax.axhline(acuan["B0"], color=PREDIKTOR_WARNA["B0"], lw=1.2, ls=(0, (1, 2)),
               label=label_acuan)
    if not sama:
        ax.axhline(acuan["B1"], color=PREDIKTOR_WARNA["B1"], lw=1.2, ls=(0, (1, 2)),
                   label=f"B1 ({persen(acuan['B1'])})")
    _penanda_terpilih(ax, n, k["akurasi_val"][n - 1],
                      f"terpilih: {n} pohon → {persen(k['akurasi_val'][n - 1])}")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: persen(v, 0)))
    ax.set_xlim(0, len(x) + 5)
    ax.set_xlabel("Iterasi boosting (jumlah pohon per kelas)")
    ax.set_ylabel("Akurasi (kelas dengan peluang tertinggi)")
    ax.legend(loc="upper left")
    bingkai(fig, "Proses 3 · Kurva belajar akurasi: latih vs validasi",
            f"Pada {n} pohon, akurasi validasi {persen(k['akurasi_val'][n - 1])} vs B0 "
            f"{persen(acuan['B0'])}. Akurasi hanya pelengkap; ukuran utama tetap log loss.",
            CATATAN)
    return simpan(fig, path, plt)


DAFTAR = (("p3_4_kurva_logloss.png", kurva_logloss),
          ("p3_5_kurva_logloss_per_tahun.png", kurva_per_tahun),
          ("p3_6_kurva_akurasi.png", kurva_akurasi))
