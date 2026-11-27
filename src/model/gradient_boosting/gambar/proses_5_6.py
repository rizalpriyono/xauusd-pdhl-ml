"""Gambar Proses 5 (kalibrasi τ) dan Proses 6 (aturan keputusan)."""
import datetime as dt

from pdhl_data.common import CLASSES

from .. import model_akhir as ma
from . import data
from .gaya import (KATEGORI, KELAS_WARNA, PREDIKTOR_WARNA, SPLIT_WARNA, STRATEGI_WARNA, SURFACE,
                   TEKS, TEKS_2, angka, bingkai, bulat, format_sumbu, persen, simpan)

ZONA = (("Continuation", KELAS_WARNA["Continuation"]), ("Abstain", "#f0efec"),
        ("Reversal", KELAS_WARNA["Reversal"]))


def _catatan_calib(D):
    return (f"Deskriptif, data kalibrasi 2024 (n = {bulat(len(D['calib']))}); data uji tidak "
            "dipakai.")


def _batas_zona(D):
    """E_rev ≥ δ → Reversal; E_cont ≥ δ ⇔ E_rev ≤ −δ·a/b → Continuation (E_cont = −(b/a)·E_rev)."""
    a, b, _, delta = data.parameter_keputusan(D)
    return -delta * a / b, delta


def logloss_tau(D, plt, path):
    """P5: log loss data 2024 sebagai fungsi τ, dengan τ terpilih dan acuan B0."""
    Z = [[float(r[f"z_{s}"]) for s in ("rev", "cont", "net")] for r in D["pred_calib"]]
    Y = [r["label"] for r in D["calib"]]
    k = D["recon"]["model_akhir"]["kalibrasi"]
    clip = D["kunci"].clip
    cari = D["kunci"].teknis["temperature_rentang"]
    t0, t1 = max(0.5, cari[0]), min(2.0, cari[1])
    ts = [t0 + (t1 - t0) * i / 300 for i in range(301)]
    ll = [ma.log_loss_tau(Z, Y, t, clip) for t in ts]
    b0 = D["recon"]["model_akhir"]["pembanding_calib"]["B0"]["log_loss"]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(ts, ll, color=KATEGORI[0], label="Model akhir, log loss 2024")
    ax.axhline(b0, color=PREDIKTOR_WARNA["B0"], lw=1.5,
               label=f"B0 tanpa kalibrasi ({angka(b0, 4)})")
    for t, teks, dx, dy in ((1.0, "τ = 1 (tanpa kalibrasi)", -150, 45),
                            (k["tau"], "τ terpilih", 40, 70)):
        y = ma.log_loss_tau(Z, Y, t, clip)
        ax.scatter([t], [y], s=64, color=KATEGORI[0] if t != 1.0 else TEKS_2, edgecolor=SURFACE,
                   linewidth=2, zorder=3)
        ax.annotate(f"{teks}: {angka(t, 4)} → {angka(y, 6)}", (t, y), xytext=(dx, dy),
                    textcoords="offset points", fontsize=8, color=TEKS,
                    arrowprops={"arrowstyle": "-", "color": TEKS_2, "lw": 0.8})
    format_sumbu(ax, 2, "x")
    format_sumbu(ax, 3)
    ax.set_xlabel("τ (pembagi skor mentah)")
    ax.set_ylabel("Log loss 2024 (makin kecil makin baik)")
    ax.legend(loc="upper right")
    bingkai(fig, "Proses 5 · Mencari τ: log loss data 2024 untuk setiap nilai τ",
            f"Ditampilkan τ {angka(t0, 1)}–{angka(t1, 1)} dari rentang pencarian Kunci-1 "
            f"{angka(cari[0], 2)}–{angka(cari[1], 0)}. Dasar kurva datar di sekitar 1: peluang "
            "mentah sudah hampir terkalibrasi.",
            _catatan_calib(D))
    return simpan(fig, path, plt)


def reliabilitas(D, plt, path):
    """P5: diagram reliabilitas tiga kelas, sebelum vs sesudah τ."""
    k = D["recon"]["model_akhir"]["kalibrasi"]
    fig, axs = plt.subplots(1, 3, figsize=(11, 4.4))
    for ax, kelas in zip(axs, CLASSES):
        ax.plot([0, 1], [0, 1], color="#c3c2b7", lw=1, zorder=1)
        for nama, warna, kunci in (("Sebelum τ", KATEGORI[1], "sebelum"),
                                   ("Sesudah τ", KATEGORI[0], "sesudah")):
            bins = [b for b in k[kunci]["reliabilitas"][kelas] if b["n"]]
            ax.plot([b["p_rerata"] for b in bins], [b["frek"] for b in bins], color=warna,
                    lw=1.6, zorder=2, label=nama)
            ax.scatter([b["p_rerata"] for b in bins], [b["frek"] for b in bins],
                       s=[14 + 3 * b["n"] ** 0.5 * 4 for b in bins], color=warna,
                       edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ece = k["sesudah"]["ece"][kelas]
        ax.set_title(f"{kelas} · ECE sesudah τ {angka(ece, 3)}", fontsize=9)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_aspect("equal")
        format_sumbu(ax, 1, "x")
        format_sumbu(ax, 1)
        ax.set_xlabel("Peluang rata-rata per bin")
    axs[0].set_ylabel("Frekuensi teramati")
    axs[0].legend(loc="upper left")
    bingkai(fig, "Proses 5 · Diagram reliabilitas data 2024, sebelum dan sesudah τ",
            "Garis abu-abu = kalibrasi sempurna. Titik lebih besar = lebih banyak sentuhan di bin; "
            "bin di ujung berisi sedikit sentuhan, jadi bukan bukti salah kalibrasi.",
            _catatan_calib(D))
    fig.subplots_adjust(bottom=0.2)                # panel persegi: beri ruang untuk catatan kaki
    return simpan(fig, path, plt)


def sebaran_peluang(D, plt, path):
    """P5: sebaran peluang per kelas sebelum vs sesudah τ (2024)."""
    P0, P1 = data.peluang_calib(D, False), data.peluang_calib(D, True)
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.9))
    bins = [i / 40 for i in range(41)]
    for j, (ax, kelas) in enumerate(zip(axs, CLASSES)):
        for nama, warna, P in (("Sebelum τ", KATEGORI[1], P0), ("Sesudah τ", KATEGORI[0], P1)):
            ax.hist([p[j] for p in P], bins=bins, histtype="step", lw=1.6, color=warna, label=nama)
        ax.set_title(f"Peluang {kelas}", fontsize=9)
        ax.set_xlim(0, 0.9)
        format_sumbu(ax, 1, "x")
        ax.set_xlabel("Peluang")
    axs[0].set_ylabel("Jumlah sentuhan")
    axs[0].legend(loc="upper right")
    tau = D["recon"]["model_akhir"]["kalibrasi"]["tau"]
    bingkai(fig, f"Proses 5 · Sebaran peluang 2024 sebelum dan sesudah τ = {angka(tau, 4)}",
            "Kedua garis hampir berimpit: τ hanya sedikit melunakkan peluang.",
            _catatan_calib(D))
    return simpan(fig, path, plt)


def _histogram_e(ax, E, reks, judul_panel, batas):
    """Histogram E_rev; jumlah per zona dihitung dari rekomendasi aturan keputusan."""
    lo, hi = -0.9, 0.9
    b_cont, b_rev = batas
    hitung = {"Continuation": reks.count("Continuation"), "Abstain": reks.count(ma.ABSTAIN),
              "Reversal": reks.count("Reversal")}
    for (nama, warna), (a, b) in zip(ZONA, ((lo, b_cont), (b_cont, b_rev), (b_rev, hi))):
        ax.axvspan(a, b, color=warna, alpha=0.12 if nama != "Abstain" else 0.6, zorder=0)
        n = hitung[nama]
        ax.text((a + b) / 2, 1.0, f"{nama}\n{bulat(n)} ({persen(n / len(E), 0)})", ha="center",
                va="top", fontsize=8, color=TEKS, transform=ax.get_xaxis_transform())
    ax.hist([min(max(e, lo), hi) for e in E], bins=[lo + i * 0.05 for i in range(37)],
            color=TEKS_2, edgecolor=SURFACE, linewidth=1)
    for x in batas:
        ax.axvline(x, color=TEKS, lw=1)
    ax.set_title(judul_panel, fontsize=9)
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
    format_sumbu(ax, 1, "x")
    ax.set_xlabel("E_rev (R)")


def sebaran_erev(D, plt, path):
    """P6: sebaran nilai harapan reversal dengan tiga zona keputusan (δ = 0,1R)."""
    a, b, m, delta = data.parameter_keputusan(D)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
    batas = _batas_zona(D)
    for ax, P, judul_panel in ((axs[0], data.peluang_calib(D),
                                f"Calib 2024, peluang terkalibrasi (n = {bulat(len(D['calib']))})"),
                               (axs[1], D["P_oof"],
                                f"OOF walk-forward 2019–2023 (n = {bulat(len(D['P_oof']))})")):
        _histogram_e(ax, data.e_rev(P, a, b, m), data.rekomendasi_sm(P, a, b, m, delta),
                     judul_panel, batas)
    axs[0].set_ylabel("Jumlah sentuhan")
    bingkai(fig, "Proses 6 · Nilai harapan reversal (E_rev) dan zona keputusan",
            f"Reversal bila E_rev ≥ {angka(batas[1], 1)}R; Continuation bila E_rev ≤ "
            f"{angka(batas[0], 1)}R (setara E_cont ≥ {angka(delta, 1)}R); selain itu abstain. "
            "Nilai di luar ±0,9 ditumpuk di tepi.")
    return simpan(fig, path, plt)


def cakupan_kinerja(D, plt, path):
    """P6: cakupan dan R per rekomendasi pada grid δ Kunci-1 saja (dua panel, satu sumbu).

    Sengaja tanpa grid δ yang lebih halus, agar gambar tidak menjadi permukaan penyetelan δ."""
    a, b, m, delta = data.parameter_keputusan(D)
    seri = (("OOF 2019–2023 (tanpa τ)", SPLIT_WARNA["dev"], D["oof_rows"], D["P_oof"]),
            ("Calib 2024 (τ terpilih)", SPLIT_WARNA["calib"], D["calib"], data.peluang_calib(D)))
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
    for nama, warna, rows, P in seri:
        h = ma.pratinjau_keputusan(rows, P, a, b, m, D["kunci"].delta_grid)
        x = [r["delta"] for r in h]
        rr = [(r["delta"], r["r_per_rekomendasi"]) for r in h if r["r_per_rekomendasi"] is not None]
        for ax, xs, ys in ((axs[0], x, [r["cakupan"] for r in h]),
                           (axs[1], [d for d, _ in rr], [v for _, v in rr])):
            ax.plot(xs, ys, color=warna, label=nama, marker="o", ms=8, mec=SURFACE, mew=2)
    for ax in axs:
        ax.axvline(delta, color=TEKS, lw=1)
        ax.text(delta + 0.004, 0.98, f"δ terkunci = {angka(delta, 1)}R", fontsize=7.5, color=TEKS,
                va="top",
                transform=ax.get_xaxis_transform())
        format_sumbu(ax, 2, "x")
        ax.set_xlabel("δ (ambang nilai harapan, R)")
    axs[1].axhline(0, color=TEKS_2, lw=1)
    axs[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: persen(v, 0)))
    format_sumbu(axs[1], 2)
    axs[0].set_title("Cakupan (porsi sentuhan yang diberi saran transaksi)", fontsize=9)
    axs[1].set_title("R kotor per rekomendasi", fontsize=9)
    axs[0].legend(loc="upper right")
    bingkai(fig, "Proses 6 · Kurva cakupan–kinerja untuk berbagai δ",
            "Hanya titik grid δ Kunci-1 "
            + "{" + "; ".join(angka(d, 1) for d in D["kunci"].delta_grid) + "}"
            + ". Makin tinggi δ, makin sedikit transaksi.",
            "Deskriptif; R kotor tanpa biaya spread; tidak dipakai untuk memilih δ. Data uji tidak "
            "dipakai.")
    return simpan(fig, path, plt)


def r_kumulatif(D, plt, path):
    """P6: R kotor kumulatif OOF 2019–2023 untuk SM (δ = 0,1R), S0, dan S1."""
    a, b, m, delta = data.parameter_keputusan(D)
    rows = D["oof_rows"]
    reks = data.rekomendasi_strategi(rows, D["P_oof"], a, b, m, delta)
    tgl = [dt.date.fromisoformat(r["session_date"]) for r in rows]
    nama = {"SM": f"SM · seleksi model (δ = {angka(delta, 1)}R)", "S0": "S0 · selalu reversal",
            "S1": "S1 · reversal bila wick ≥ 2/3"}
    fig, ax = plt.subplots(figsize=(9, 4.6))
    akhir = {}
    for s in ("SM", "S0", "S1"):
        c = data.r_kumulatif(rows, reks[s])
        ax.plot(tgl, c, color=STRATEGI_WARNA[s], lw=1.8, label=f"{nama[s]}: {angka(c[-1], 1)}R")
        akhir[s] = c[-1]
    y_label, batas = {}, None                      # label ujung tidak boleh bertumpuk
    for s in sorted(akhir, key=akhir.get, reverse=True):
        y_label[s] = akhir[s] if batas is None else min(akhir[s], batas - 3.5)
        batas = y_label[s]
    for s, y in y_label.items():
        ax.annotate(s, (tgl[-1], akhir[s]), xytext=(tgl[-1] + dt.timedelta(days=20), y),
                    textcoords="data", va="center",
                    fontsize=8, color=TEKS)
    ax.axhline(0, color=TEKS_2, lw=1)
    format_sumbu(ax, 0)
    ax.set_ylabel("R kotor kumulatif")
    ax.legend(loc="upper left")
    bingkai(fig, "Proses 6 · R kotor kumulatif pada prediksi OOF 2019–2023",
            f"n = {bulat(len(rows))} sentuhan, urut waktu; biaya spread belum dikurangi.",
            "Deskriptif; m_rev/m_cont dari seluruh dev ikut memuat tahun validasi. Data uji tidak "
            "dipakai.")
    return simpan(fig, path, plt)


def sm_per_tahun(D, plt, path):
    """P6: cakupan dan R per rekomendasi SM (δ = 0,1R) per tahun OOF."""
    a, b, m, delta = data.parameter_keputusan(D)
    h = data.sm_per_tahun(D["oof_rows"], D["P_oof"], a, b, m, delta)
    tahun = [r["tahun"] for r in h]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, kolom, fmt in ((axs[0], "cakupan", lambda v: persen(v)),
                           (axs[1], "r_per_rekomendasi", lambda v: angka(v, 3))):
        v = [r[kolom] if r[kolom] is not None else 0.0 for r in h]
        ax.bar(tahun, v, width=0.55, color=KATEGORI[0], edgecolor=SURFACE, linewidth=2)
        for x, y in zip(tahun, v):
            ax.annotate(fmt(y), (x, y), xytext=(0, 4 if y >= 0 else -12),
                        textcoords="offset points", ha="center", fontsize=8, color=TEKS)
        ax.set_xticks(tahun)
        ax.grid(axis="x", visible=False)
    axs[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: persen(v, 0)))
    axs[1].axhline(0, color=TEKS_2, lw=1)
    format_sumbu(axs[1], 2)
    rr = [r["r_per_rekomendasi"] or 0.0 for r in h]
    axs[1].set_ylim(min(min(rr) * 1.4, -0.1), max(max(rr) * 1.3, 0.1))
    axs[0].set_ylim(0, max(r["cakupan"] for r in h) * 1.2)
    axs[0].set_title("Cakupan SM per tahun", fontsize=9)
    axs[1].set_title("R kotor per rekomendasi SM per tahun", fontsize=9)
    tanda = {r > 0 for r in rr if r}
    bingkai(fig, f"Proses 6 · Seleksi model (δ = {angka(delta, 1)}R) per tahun OOF",
            f"Cakupan {persen(h[0]['cakupan'])} pada {h[0]['tahun']} dan "
            f"{persen(h[-1]['cakupan'])} pada {h[-1]['tahun']}; hasil per rekomendasi "
            + ("berganti tanda antartahun." if len(tanda) > 1 else "bertanda sama di semua tahun."),
            "Deskriptif; R kotor tanpa biaya spread. Data uji tidak dipakai.")
    return simpan(fig, path, plt)


DAFTAR = (("p5_1_logloss_tau.png", logloss_tau),
          ("p5_2_reliabilitas.png", reliabilitas),
          ("p5_3_sebaran_peluang.png", sebaran_peluang),
          ("p6_1_sebaran_erev.png", sebaran_erev),
          ("p6_2_cakupan_kinerja.png", cakupan_kinerja),
          ("p6_3_r_kumulatif.png", r_kumulatif),
          ("p6_4_sm_per_tahun.png", sm_per_tahun))
