"""Gaya bersama gambar: palet tervalidasi, format angka Indonesia, dan penyimpanan PNG.

Palet mengikuti palet acuan skill dataviz (mode terang). Tiga slot kategori pertama
(biru, oranye, aqua) lolos validasi semua-pasangan; aqua di bawah kontras 3:1, sehingga gambar
yang memakainya selalu diberi label langsung dan tabel angkanya ada di dokumentasi.
"""
SURFACE = "#fcfcfb"
TEKS = "#0b0b0b"
TEKS_2 = "#52514e"
GRID = "#e4e3df"
NETRAL_ABU = "#f0efec"

KATEGORI = ("#2a78d6", "#eb6834", "#1baf7a")          # slot 1-3, urutan tetap
KELAS_WARNA = dict(zip(("Reversal", "Continuation", "Netral"), KATEGORI))
SPLIT_WARNA = {"dev": KATEGORI[0], "calib": KATEGORI[1]}
PREDIKTOR_WARNA = {"Model": KATEGORI[0], "B0": KATEGORI[1], "B1": KATEGORI[2]}
STRATEGI_WARNA = {"SM": KATEGORI[0], "S0": KATEGORI[1], "S1": KATEGORI[2]}
SEKUENSIAL_BIRU = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
DIVERGEN = ("#e34948", NETRAL_ABU, "#2a78d6")          # merah - abu - biru
CATATAN_DESKRIPTIF = ("Deskriptif; hanya data dev 2016–2023 dan calib 2024. Data uji 2025–2026 "
                      "tidak dipakai.")


def angka(x, nd=2):
    """Angka desimal gaya Indonesia: koma desimal, minus tipografis, tanpa '−0,00'."""
    x = round(x, nd) + 0.0
    s = f"{x:.{nd}f}".replace(".", ",")
    return s.replace("-", "−")


def bulat(n):
    """Bilangan bulat dengan titik pemisah ribuan (1.841)."""
    return f"{int(n):,}".replace(",", ".")


def persen(x, nd=1):
    return angka(100 * x, nd) + "%"


# Dipakai lewat `plt.rc_context(RC)`, sehingga rcParams global tidak berubah.
RC = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEKS_2, "axes.titlecolor": TEKS,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 9, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": TEKS_2, "ytick.color": TEKS_2, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 2,
    "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round", "font.size": 9,
    "axes.formatter.use_mathtext": False,
}


def format_sumbu(ax, nd=2, sumbu="y"):
    """Label sumbu dengan koma desimal."""
    from matplotlib.ticker import FuncFormatter
    f = FuncFormatter(lambda v, _: angka(v, nd))
    (ax.yaxis if sumbu == "y" else ax.xaxis).set_major_formatter(f)


def bingkai(fig, judul, sub=None, catatan=CATATAN_DESKRIPTIF):
    """Judul dan subjudul rata kiri di atas, catatan kaki di bawah, lalu tata letak rapat."""
    tinggi = fig.get_figheight()
    pita_atas = (0.62 if sub else 0.38) / tinggi
    pita_bawah = (0.40 if catatan else 0.08) / tinggi
    fig.text(0.012, 1 - 0.12 / tinggi, judul, ha="left", va="top", fontsize=12,
             fontweight="bold", color=TEKS)
    if sub:
        fig.text(0.012, 1 - 0.40 / tinggi, sub, ha="left", va="top", fontsize=8.5, color=TEKS_2)
    if catatan:
        fig.text(0.012, 0.08 / tinggi, catatan, ha="left", va="bottom", fontsize=7.5,
                 color=TEKS_2)
    fig.tight_layout(rect=(0, pita_bawah, 1, 1 - pita_atas))


def simpan(fig, path, plt):
    """PNG 150 dpi tanpa metadata versi pustaka; figur ditutup sesudah disimpan."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, metadata={"Software": None})
    plt.close(fig)
    return path
