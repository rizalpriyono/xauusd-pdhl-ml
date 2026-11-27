"""Gambar Proses 0 (data berlabel), Proses 1 (rekayasa fitur), dan Proses 2 (pembanding B0/B1)."""
from pdhl_data.common import CLASSES

from . import data
from .gaya import (CATATAN_DESKRIPTIF, DIVERGEN, KELAS_WARNA, PREDIKTOR_WARNA, SPLIT_WARNA,
                   SURFACE, TEKS, TEKS_2, angka, bingkai, bulat, format_sumbu, persen, simpan)

NAMA_SPLIT = {"dev": "Dev 2016–2023", "calib": "Calib 2024"}
LABEL_FITUR = {
    "f01_jenis_level": "f01 jenis level (1 = PDH)",
    "f02_rentang_hari_sebelumnya": "f02 rentang hari lalu",
    "f03_jarak_level_open_sesi": "f03 jarak level–open",
    "f06_rezim_volatilitas": "f06 rezim volatilitas",
    "f07_momentum_pendekatan": "f07 momentum",
    "f08_waktu_sejak_open_sesi": "f08 waktu sejak open",
    "f09_posisi_close": "f09 posisi close",
    "f10_penetrasi": "f10 penetrasi",
    "f11_jarak_ema50": "f11 jarak EMA50",
    "f12_return_minggu": "f12 return minggu",
    "f13_wick_penolakan": "f13 wick penolakan",
    "f14_cmf21_d": "f14 CMF-21",
    "f15_pe48": "f15 permutation entropy",
}


def _kolom(rows, c):
    return [float(r[c]) for r in rows]


def label_per_tahun(D, plt, path):
    """P0: proporsi label per tahun (dev + calib), batang bertumpuk 100%."""
    cnt = data.label_per_tahun(D["dev"] + D["calib"])
    tahun = list(cnt)
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    bawah = [0.0] * len(tahun)
    for k in CLASSES:
        v = [cnt[t][k] / sum(cnt[t].values()) for t in tahun]
        ax.bar(tahun, v, bottom=bawah, width=0.62, color=KELAS_WARNA[k], edgecolor=SURFACE,
               linewidth=2, label=k)
        for x, b0, h in zip(tahun, bawah, v):
            ax.text(x, b0 + h / 2, persen(h, 0), ha="center", va="center", fontsize=7.5, color=TEKS)
        bawah = [b0 + h for b0, h in zip(bawah, v)]
    for t in tahun:
        ax.text(t, 1.015, bulat(sum(cnt[t].values())), ha="center", va="bottom", fontsize=7.5,
                color=TEKS_2)
    ax.axvline(2023.5, color=TEKS_2, lw=1)
    ax.text(2023.45, 1.075, "dev ←", ha="right", fontsize=8, color=TEKS_2)
    ax.text(2023.55, 1.075, "→ calib", ha="left", fontsize=8, color=TEKS_2)
    ax.set_xticks(tahun)
    ax.set_ylim(0, 1.13)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: persen(v, 0) if v <= 1 else ""))
    ax.set_ylabel("Proporsi sentuhan")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, -0.08))
    terbanyak = {max(cnt[t], key=cnt[t].get) for t in tahun}
    pola = (f"{terbanyak.pop()} terbanyak di setiap tahun" if len(terbanyak) == 1
            else "kelas terbanyak berganti antartahun")
    bingkai(fig, "Proses 0 · Komposisi label per tahun",
            f"Angka di atas batang = jumlah sentuhan. {pola}; data uji 2025–2026 dikunci dan "
            "tidak ditampilkan.")
    return simpan(fig, path, plt)


def f13_wick(D, plt, path):
    """P1: sebaran f13 wick penolakan, dev vs calib, dengan ambang B1."""
    theta = D["kunci"].theta
    fig, ax = plt.subplots(figsize=(8, 4.4))
    bins = [i / 25 for i in range(26)]
    teks, porsi = [], {}
    for s in ("dev", "calib"):
        v = _kolom(D["fitur"][s], "f13_wick_penolakan")
        ax.hist(v, bins=bins, density=True, histtype="step", lw=2, color=SPLIT_WARNA[s],
                label=f"{NAMA_SPLIT[s]} (n = {bulat(len(v))})")
        porsi[s] = sum(x >= theta - 1e-9 for x in v) / len(v)
        teks.append(f"{s} {persen(porsi[s])}")
    ax.axvspan(theta, 1, color="#f0efec", zorder=0)
    ax.axvline(theta, color=TEKS, lw=1)
    ax.text(theta + 0.012, ax.get_ylim()[1] * 0.93,
            f"B1 terpenuhi (f13 ≥ θ = {angka(theta, 3)})\n" + " · ".join(teks),
            fontsize=8, color=TEKS, va="top")
    ax.set_xlim(0, 1)
    format_sumbu(ax, 1, "x")
    format_sumbu(ax, 1, "y")
    ax.set_xlabel("f13 wick penolakan (panjang wick searah reversal / rentang candle)")
    ax.set_ylabel("Kepadatan")
    ax.legend(loc="upper center")
    bingkai(fig, "Proses 1 · Sebaran wick penolakan (f13) dan ambang B1",
            f"Syarat B1 terpenuhi pada {persen(porsi['dev'])} sentuhan dev dan "
            f"{persen(porsi['calib'])} sentuhan calib.")
    return simpan(fig, path, plt)


def _bins(v, lo, hi):
    unik = sorted(set(v))
    if len(unik) <= 25 and all(float(x).is_integer() for x in unik):
        return [u - 0.5 for u in range(int(min(unik)), int(max(unik)) + 2)]
    return [lo + (hi - lo) * i / 24 for i in range(25)]


def sebaran_fitur(D, plt, path):
    """P1: small multiples 13 fitur numerik, dev vs calib (persentil 1–99)."""
    fig, axs = plt.subplots(4, 4, figsize=(11, 9.4))
    for ax, c in zip(axs.flat, data.NUMERIK):
        semua = sorted(_kolom(D["fitur"]["dev"], c) + _kolom(D["fitur"]["calib"], c))
        lo, hi = semua[int(0.01 * len(semua))], semua[int(0.99 * len(semua)) - 1]
        bins = _bins(semua, lo, hi)
        for s in ("dev", "calib"):
            v = [x for x in _kolom(D["fitur"][s], c) if bins[0] <= x <= bins[-1]]
            ax.hist(v, bins=bins, density=True, histtype="step", lw=1.6, color=SPLIT_WARNA[s],
                    label=NAMA_SPLIT[s])
        ax.set_title(LABEL_FITUR[c], fontsize=8.5)
        unik = sorted(set(semua))
        if len(unik) <= 5:                       # fitur diskret: tanda sumbu di nilai aslinya
            ax.set_xticks(unik, [bulat(u) if u >= 0 else "−" + bulat(-u) for u in unik])
        ax.tick_params(labelsize=7)
        ax.set_yticks([])
        if len(unik) > 5:
            format_sumbu(ax, 1, "x")
    for ax in list(axs.flat)[len(data.NUMERIK):]:
        ax.axis("off")
    h, l = axs.flat[0].get_legend_handles_labels()
    axs.flat[-1].legend(h, l, loc="center", fontsize=9)
    bingkai(fig, "Proses 1 · Sebaran 13 fitur numerik: dev vs calib",
            "Kepadatan per fitur; sumbu x dipotong pada persentil 1–99. Fitur sengaja tidak "
            "dipisah menurut label.")
    return simpan(fig, path, plt)


def korelasi_fitur(D, plt, path):
    """P1: peta panas korelasi Pearson antarfitur numerik (dev), segitiga bawah."""
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    cols = list(data.NUMERIK)
    R = data.korelasi(D["fitur"]["dev"], cols)
    M = np.ma.masked_where(np.triu(np.ones_like(R, dtype=bool), k=0), R)
    fig, ax = plt.subplots(figsize=(8.6, 7.6))
    cmap = LinearSegmentedColormap.from_list("divergen", DIVERGEN)
    im = ax.imshow(M, cmap=cmap, vmin=-1, vmax=1)
    for i in range(len(cols)):
        for j in range(i):
            if abs(R[i, j]) >= 0.2:
                ax.text(j, i, angka(R[i, j], 2), ha="center", va="center", fontsize=7, color=TEKS)
    nama = [c[:3] for c in cols]
    ax.set_xticks(range(len(cols)), nama)
    ax.set_yticks(range(len(cols)), [LABEL_FITUR[c] for c in cols])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.75, label="Korelasi Pearson")
    cb.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: angka(v, 1)))
    bingkai(fig, f"Proses 1 · Korelasi antarfitur (dev, n = {bulat(len(D['fitur']['dev']))})",
            "Angka ditampilkan bila |r| ≥ 0,2. Merah = berlawanan arah, biru = searah.")
    return simpan(fig, path, plt)


def peluang_pembanding(D, plt, path):
    """P2: peluang tiga kelas B0 dan kedua kelompok B1, batang horizontal bertumpuk."""
    ak = D["recon"]["pembanding"]["akhir"]
    baris = [(f"B1 · wick < 2/3 (n = {bulat(ak['B1']['tidak_terpenuhi']['n'])})",
              ak["B1"]["tidak_terpenuhi"]["peluang"]),
             (f"B1 · wick ≥ 2/3 (n = {bulat(ak['B1']['terpenuhi']['n'])})",
              ak["B1"]["terpenuhi"]["peluang"]),
             (f"B0 · semua sentuhan (n = {bulat(ak['B0']['n'])})", ak["B0"]["peluang"])]
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    kiri = [0.0] * len(baris)
    for k in CLASSES:
        v = [p[k] for _, p in baris]
        ax.barh(range(len(baris)), v, left=kiri, height=0.5, color=KELAS_WARNA[k],
                edgecolor=SURFACE, linewidth=2, label=k)
        for y, x0, w in zip(range(len(baris)), kiri, v):
            ax.text(x0 + w / 2, y, persen(w), ha="center", va="center", fontsize=8, color=TEKS)
        kiri = [x0 + w for x0, w in zip(kiri, v)]
    ax.set_yticks(range(len(baris)), [n for n, _ in baris])
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: persen(v, 0)))
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, -0.14))
    bingkai(fig, "Proses 2 · Peluang pembanding akhir (dev 2016–2023)",
            "Wick panjang terutama menggeser peluang dari Continuation ke Netral; peluang "
            "Reversal hanya naik sekitar 2 poin.")
    return simpan(fig, path, plt)


def logloss_pembanding(D, plt, path):
    """P2: log loss OOF B0 vs B1 per tahun validasi."""
    p = D["recon"]["pembanding"]
    tahun = [f["tahun_validasi"] for f in p["walk_forward"]]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for nama, kolom in (("B0", "log_loss_b0"), ("B1", "log_loss_b1")):
        v = [f[kolom] for f in p["walk_forward"]]
        ax.plot(tahun, v, marker="o", ms=8, mec=SURFACE, mew=2, color=PREDIKTOR_WARNA[nama],
                label=f"{nama} (gabungan {angka(p['oof']['log_loss_' + nama.lower()], 4)})")
        ax.annotate(nama, (tahun[-1], v[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=8, color=TEKS)
    ax.set_xticks(tahun)
    format_sumbu(ax, 3)
    ax.set_ylabel("Log loss OOF (makin kecil makin baik)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    selisih = p["oof"]["log_loss_b1"] - p["oof"]["log_loss_b0"]
    buruk = [str(f["tahun_validasi"]) for f in p["walk_forward"]
             if f["log_loss_b1"] - f["log_loss_b0"] > 5e-5]
    bingkai(fig, "Proses 2 · Log loss OOF pembanding per tahun validasi",
            f"Selisih gabungan B1 − B0 = {angka(selisih, 4)} per sentuhan"
            + (f"; B1 lebih buruk pada {', '.join(buruk)}." if buruk else "."),
            CATATAN_DESKRIPTIF)
    return simpan(fig, path, plt)


DAFTAR = (("p0_1_label_per_tahun.png", label_per_tahun),
          ("p1_1_f13_wick_b1.png", f13_wick),
          ("p1_2_sebaran_fitur.png", sebaran_fitur),
          ("p1_3_korelasi_fitur.png", korelasi_fitur),
          ("p2_1_peluang_pembanding.png", peluang_pembanding),
          ("p2_2_logloss_oof_pembanding.png", logloss_pembanding))
