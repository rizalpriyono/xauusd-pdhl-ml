"""Gambar Proses 3 (walk-forward) dan Proses 4 (model akhir d2_n100_m50)."""
import math

from . import data
from .gaya import (KATEGORI, PREDIKTOR_WARNA, SEKUENSIAL_BIRU, SURFACE, TEKS, TEKS_2, angka,
                   bingkai, bulat, format_sumbu, simpan)

NAMA_FITUR = {
    "f01_jenis_level": "f01 jenis level",
    "f02_rentang_hari_sebelumnya": "f02 rentang hari lalu",
    "f03_jarak_level_open_sesi": "f03 jarak level–open",
    "f04_sesi": "f04 sesi",
    "f05_hari": "f05 hari",
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


def heatmap_konfigurasi(D, plt, path):
    """P3: matriks log loss OOF 8 konfigurasi × 5 tahun + gabungan."""
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle
    wf = D["recon"]["walk_forward"]
    tahun = [str(t) for t in wf["tahun_validasi"]]
    cfgs = sorted(wf["per_tahun"], key=lambda c: wf["peringkat"][c])
    M = [[wf["per_tahun"][c][t] for t in tahun] + [wf["skor"][c]] for c in cfgs]
    vals = [v for r in M for v in r]
    lo, hi = min(vals), max(vals)
    rentang = (hi - lo) or 1.0
    cmap = LinearSegmentedColormap.from_list("biru", list(reversed(SEKUENSIAL_BIRU)))
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    im = ax.imshow(M, cmap=cmap, vmin=lo, vmax=hi, aspect="auto")
    terbaik = {j: min(range(len(cfgs)), key=lambda i: M[i][j]) for j in range(len(tahun))}
    for i, r in enumerate(M):
        for j, v in enumerate(r):
            gelap = (v - lo) / rentang < 0.45
            tanda = " ★" if terbaik.get(j) == i else ""
            ax.text(j, i, angka(v, 4) + tanda, ha="center", va="center", fontsize=8,
                    color="white" if gelap else TEKS,
                    fontweight="bold" if j == len(tahun) else None)
    ax.axvline(len(tahun) - 0.5, color=SURFACE, lw=5)
    baris_pilih = cfgs.index(wf["terpilih"]["id"])
    ax.add_patch(Rectangle((-0.5, baris_pilih - 0.5), len(tahun) + 1, 1, fill=False, ec=TEKS,
                           lw=1.5))
    ax.set_xticks(range(len(tahun) + 1), tahun + ["Gabungan"])
    ax.set_yticks(range(len(cfgs)), [f"{wf['peringkat'][c]}. {c}" for c in cfgs])
    ax.xaxis.tick_top()
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.8, label="Log loss OOF (lebih gelap = lebih baik)")
    cb.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: angka(v, 2)))
    bingkai(fig, "Proses 3 · Log loss OOF per konfigurasi dan tahun validasi",
            "★ = terbaik di tahun itu. Bingkai = konfigurasi terpilih. "
            "d (kedalaman), n (jumlah pohon), m (minimal sentuhan per daun).")
    return simpan(fig, path, plt)


def model_vs_pembanding(D, plt, path):
    """P3: log loss OOF per tahun, konfigurasi terpilih vs B0 vs B1."""
    b = D["recon"]["walk_forward"]["banding_pembanding"]
    tahun = [t["tahun"] for t in b["per_tahun"]]
    g = b["gabungan"]
    menang = sum(t["ll_model"] < min(t["ll_b0"], t["ll_b1"]) for t in b["per_tahun"])
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for nama, kolom in (("Model", "ll_model"), ("B0", "ll_b0"), ("B1", "ll_b1")):
        v = [t[kolom] for t in b["per_tahun"]]
        label = f"Model {D['recon']['walk_forward']['terpilih']['id']}" if nama == "Model" else nama
        ax.plot(tahun, v, marker="o", ms=8, mec=SURFACE, mew=2, color=PREDIKTOR_WARNA[nama],
                label=f"{label} (gabungan {angka(g[kolom], 4)})")
        ax.annotate(nama, (tahun[-1], v[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=8, color=TEKS)
    ax.set_xticks(tahun)
    format_sumbu(ax, 3)
    ax.set_ylabel("Log loss OOF (makin kecil makin baik)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", fontsize=7.5)
    bingkai(fig, "Proses 3 · Model terpilih vs pembanding per tahun",
            f"Model lebih baik daripada B0 dan B1 di {menang} dari {len(tahun)} tahun; selisih "
            f"gabungan model − B0 {angka(g['D_b0'], 4)}, model − B1 {angka(g['D_b1'], 4)}.")
    return simpan(fig, path, plt)


def konfigurasi_2x2x2(D, plt, path):
    """P3: dumbbell log loss gabungan; baris = kedalaman × jumlah pohon, titik = min. per daun."""
    wf = D["recon"]["walk_forward"]
    pilih = wf["terpilih"]["id"]
    baris = [(2, 100), (2, 300), (3, 100), (3, 300)]
    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    for y, (d, n) in enumerate(baris):
        v = {m: wf["skor"][f"d{d}_n{n}_m{m}"] for m in (20, 50)}
        ax.plot([v[20], v[50]], [y, y], color="#c3c2b7", lw=2, zorder=1)
        for m, warna in ((20, KATEGORI[1]), (50, KATEGORI[0])):
            ax.scatter(v[m], y, s=70, color=warna, edgecolor=SURFACE, linewidth=2, zorder=3,
                       label=f"minimal {m} sentuhan per daun" if y == 0 else None)
            kiri = v[m] <= min(v.values())
            ax.annotate(angka(v[m], 4), (v[m], y), xytext=(-9 if kiri else 9, 0),
                        textcoords="offset points", ha="right" if kiri else "left", va="center",
                        fontsize=7.5, color=TEKS)
            if f"d{d}_n{n}_m{m}" == pilih:
                ax.annotate("terpilih", (v[m], y), xytext=(0, -14), textcoords="offset points",
                            ha="center", fontsize=7.5, color=TEKS, fontweight="bold")
    ax.set_yticks(range(len(baris)), [f"kedalaman {d} · {n} pohon" for d, n in baris])
    ax.set_ylim(len(baris) - 0.4, -0.6)
    skor = list(wf["skor"].values())
    jarak = max(skor) - min(skor)
    ax.set_xlim(min(skor) - 0.12 * jarak, max(skor) + 0.12 * jarak)
    format_sumbu(ax, 2, "x")
    ax.set_xlabel("Log loss OOF gabungan 2019–2023 (makin kiri makin baik)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper right")
    s = wf["skor"]
    sederhana = all(s[f"d2_n{n}_m{m}"] < s[f"d3_n{n}_m{m}"]
                    and s[f"d{d}_n100_m{m}"] < s[f"d{d}_n300_m{m}"]
                    and s[f"d{d}_n{n}_m50"] < s[f"d{d}_n{n}_m20"]
                    for d in (2, 3) for n in (100, 300) for m in (20, 50))
    bingkai(fig, "Proses 3 · Log loss gabungan menurut kedalaman, jumlah pohon, dan isi daun",
            "Di setiap pasangan, pilihan yang lebih sederhana (kedalaman 2, 100 pohon, minimal 50 "
            "per daun) memberi log loss lebih kecil." if sederhana else
            "Pilihan yang lebih sederhana tidak selalu memberi log loss lebih kecil.")
    return simpan(fig, path, plt)


def _label_node(n, nama):
    """Teks simpul pemisah; kolom one-hot (0/1) ditulis sebagai pertanyaan 'bukan kategori'."""
    kolom = nama[n["split_feature"]]
    f = NAMA_FITUR.get(data.nama_dasar(kolom), kolom)
    if "=" in kolom:
        tanya = f"{f} bukan {kolom.split('=')[1]}?"
    else:
        tanya = f"{f} ≤ {format(n['threshold'], '.4g').replace('.', ',')}?"
    return f"{tanya}\nn = {bulat(n['internal_count'])}"


def pohon_pertama(D, plt, path):
    """P4: pohon pertama kelas Reversal pada model akhir, digambar dari dump LightGBM."""
    s = D["struktur"]
    akar, nama = s["pohon_pertama"], s["nama"]
    daun_x, pos, garis = [0], {}, []

    def tata(n, depth):
        if data.daun(n):
            pos[id(n)] = (daun_x[0], depth)
            daun_x[0] += 1
            return pos[id(n)][0]
        xs = [tata(n["left_child"], depth + 1), tata(n["right_child"], depth + 1)]
        garis.extend([(n, n["left_child"], "ya"), (n, n["right_child"], "tidak")])
        pos[id(n)] = (sum(xs) / 2, depth)
        return pos[id(n)][0]

    tata(akar, 0)
    fig, ax = plt.subplots(figsize=(10, 5.4))
    for a, b_, lbl in garis:
        (x0, y0), (x1, y1) = pos[id(a)], pos[id(b_)]
        ax.plot([x0, x1], [-y0, -y1], color="#c3c2b7", lw=1.5, zorder=1)
        ax.text((x0 + x1) / 2, -(y0 + y1) / 2, lbl, ha="center", va="center", fontsize=8,
                color=TEKS_2, bbox={"boxstyle": "round,pad=0.2", "fc": SURFACE, "ec": "none"})

    def gambar(n):
        x, y = pos[id(n)]
        if data.daun(n):
            teks, fc = f"skor {angka(n['leaf_value'], 3)}\nn = {bulat(n['leaf_count'])}", "#cde2fb"
        else:
            teks, fc = _label_node(n, nama), "#f0efec"
            gambar(n["left_child"])
            gambar(n["right_child"])
        ax.text(x, -y, teks, ha="center", va="center", fontsize=8.5, color=TEKS, zorder=3,
                bbox={"boxstyle": "round,pad=0.5", "fc": fc, "ec": "none"})

    gambar(akar)
    ax.set_xlim(-0.6, daun_x[0] - 0.4)
    ax.set_ylim(-2.45, 0.45)
    ax.axis("off")
    p_rev = D["recon"]["pembanding"]["akhir"]["B0"]["peluang"]["Reversal"]
    cfg = D["recon"]["model_akhir"]["konfigurasi"]
    bingkai(fig, f"Proses 4 · Pohon 1 dari {bulat(s['jumlah_pohon'])} pada model {cfg['id']} "
                 "(kelas Reversal)",
            f"Maksimal {2 ** cfg['max_depth']} daun. Skor daun ≈ ln peluang dasar Reversal "
            f"(ln {angka(p_rev, 3)} ≈ {angka(math.log(p_rev), 2)}): model berangkat dari B0, lalu "
            f"dikoreksi {cfg['n_estimators'] - 1} pohon berikutnya.")
    return simpan(fig, path, plt)


def gain_fitur(D, plt, path):
    """P4: sumbangan gain per fitur (one-hot digabung) pada 300 pohon model akhir."""
    s = D["struktur"]
    tot = sum(s["gain"].values())
    urut = sorted(s["gain"], key=s["gain"].get)
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    v = [s["gain"][f] / tot for f in urut]
    ax.barh(range(len(urut)), v, height=0.6, color=KATEGORI[0], edgecolor=SURFACE, linewidth=2)
    for y, f in enumerate(urut):
        ax.text(v[y] + 0.004, y, f"{angka(100 * v[y], 1)}%  ·  {s['split'][f]} pemisahan",
                va="center", fontsize=7.5, color=TEKS)
    ax.set_yticks(range(len(urut)), [NAMA_FITUR.get(f, f) for f in urut])
    ax.set_xlim(0, max(v) * 1.35)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: angka(100 * x, 0) + "%"))
    ax.set_xlabel("Sumbangan gain (persen dari total)")
    ax.grid(axis="y", visible=False)
    bingkai(fig, "Proses 4 · Fitur yang paling banyak dipakai model akhir",
            "Gain = penurunan kesalahan saat fitur dipakai memisah. Gambaran data dev, "
            "bukan pengganti analisis SHAP.")
    return simpan(fig, path, plt)


def isi_daun(D, plt, path):
    """P4: sebaran isi daun (jumlah sentuhan) seluruh pohon, dengan batas min_child_samples."""
    s = D["struktur"]
    isi = s["isi_daun"]
    batas = D["recon"]["model_akhir"]["konfigurasi"]["min_child_samples"]
    kecil = [c for c in isi if c < batas]
    hi, lo = max(isi) * 1.05, min(min(isi), batas) * 0.9
    bins = ([lo + (batas - lo) * i / 4 for i in range(4)]
            + [batas * (hi / batas) ** (i / 36) for i in range(37)])
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    ax.hist(isi, bins=bins, color=KATEGORI[0], edgecolor=SURFACE, linewidth=1)
    ax.set_xscale("log")
    ax.axvline(batas, color=TEKS, lw=1)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
    ax.text(batas * 1.04, ax.get_ylim()[1] * 0.97,
            f"min_child_samples = {batas}\n{len(kecil)} dari {bulat(len(isi))} daun < {batas} "
            f"(terkecil {min(isi)})", fontsize=8, color=TEKS, va="top")
    ticks = sorted({min(isi), batas, 100, 200, 500, 1000, max(isi)})
    ax.set_xticks(ticks, [bulat(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Isi daun (jumlah sentuhan latih, skala log)")
    ax.set_ylabel("Jumlah daun")
    ax.grid(axis="x", visible=False)
    bingkai(fig, f"Proses 4 · Isi daun pada {bulat(s['jumlah_pohon'])} pohon model akhir",
            f"Batas {batas} diperiksa LightGBM dengan perkiraan dari histogram, sehingga sebagian "
            f"kecil daun berisi sedikit di bawah {batas}.")
    return simpan(fig, path, plt)


DAFTAR = (("p3_1_heatmap_konfigurasi.png", heatmap_konfigurasi),
          ("p3_2_model_vs_pembanding.png", model_vs_pembanding),
          ("p3_3_konfigurasi_2x2x2.png", konfigurasi_2x2x2),
          ("p4_1_pohon_pertama.png", pohon_pertama),
          ("p4_2_gain_fitur.png", gain_fitur),
          ("p4_3_isi_daun.png", isi_daun))
