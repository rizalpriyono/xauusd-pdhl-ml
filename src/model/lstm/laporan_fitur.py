"""Laporan deskriptif fitur sekuensial LSTM (RM-22) untuk data pengembangan dan kalibrasi.

Membaca `seq_meta_*.csv` (lewat loader ber-guard) dan `fitur_rekonsiliasi.json`, lalu menulis
`laporan_fitur.md`. Statistik dilaporkan pada bar keputusan t* dan sengaja tidak dikondisikan
pada label, sehingga tidak dipakai untuk mengubah rancangan fitur.
"""
import json
import math
from collections import Counter

from pdhl_data import common
from pdhl_data.common import SPLIT_END

from .fitur import FEATURE_NAMES

CORR_FLAG = 0.7


def quantile(sorted_xs, q):
    """Kuantil interpolasi linear (sama dengan numpy 'linear')."""
    if not sorted_xs:
        return math.nan
    pos = (len(sorted_xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * (pos - lo)


def describe(xs):
    xs = sorted(xs)
    n = len(xs)
    mu = sum(xs) / n if n else math.nan
    sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1)) if n > 1 else math.nan
    return {"n": n, "mean": mu, "sd": sd, "min": xs[0] if n else math.nan,
            "p05": quantile(xs, 0.05), "median": quantile(xs, 0.5), "p95": quantile(xs, 0.95),
            "max": xs[-1] if n else math.nan}


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return math.nan
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    return sxy / math.sqrt(sxx * syy) if sxx and syy else math.nan


def num(rows, col):
    return [float(r[col]) for r in rows if r[col] != ""]


def f(v, nd=3):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    return f"{v + 0.0:.{nd}f}".replace(f"-0.{'0' * nd}", f"0.{'0' * nd}")


def section_numeric(rows_by_split):
    out = [f"## 1. Statistik {len(FEATURE_NAMES)} fitur sekuensial pada bar keputusan (t*)", ""]
    for split, rows in rows_by_split.items():
        out += [f"### {split} (n = {len(rows)})", "",
                "| Fitur | kosong | mean | sd | min | p5 | median | p95 | max |",
                "|---|---|---|---|---|---|---|---|---|"]
        for c in FEATURE_NAMES:
            d = describe(num(rows, c))
            out.append(f"| {c} | {len(rows) - d['n']} | " + " | ".join(
                f(d[k]) for k in ("mean", "sd", "min", "p05", "median", "p95", "max")) + " |")
        out.append("")
    return out


def section_b1(rows_by_split):
    out = ["## 2. Flag pembanding B1 (wick penolakan ≥ θ)", "",
           "| Nilai | " + " | ".join(rows_by_split) + " |",
           "|---|" + "---|" * len(rows_by_split)]
    cats = sorted({r["b1_terpenuhi"] for rows in rows_by_split.values() for r in rows})
    for v in cats:
        cells = []
        for rows in rows_by_split.values():
            n = Counter(r["b1_terpenuhi"] for r in rows)[v]
            cells.append(f"{n} ({100 * n / len(rows):.1f}%)" if rows else "0 (–)")
        out.append(f"| {v} | " + " | ".join(cells) + " |")
    out.append("")
    return out


def section_corr(rows):
    cols = list(FEATURE_NAMES)
    full = [r for r in rows if all(r[c] != "" for c in cols)]
    vals = {c: [float(r[c]) for r in full] for c in cols}
    short = [f"s{i + 1:02d}" for i in range(len(cols))]
    out = ["## 3. Korelasi Pearson antarfitur pada bar keputusan (dev)", "",
           f"n = {len(full)}. Pasangan dengan |r| ≥ {CORR_FLAG} ditebalkan (catatan deskriptif).",
           "", "| | " + " | ".join(short) + " |", "|---|" + "---|" * len(cols)]
    flagged = []
    for i, a in enumerate(cols):
        cells = []
        for j, b in enumerate(cols):
            r = pearson(vals[a], vals[b])
            txt = f(r, 2)
            if i != j and abs(r) >= CORR_FLAG:
                txt = f"**{txt}**"
                if i < j:
                    flagged.append((a, b, r))
            cells.append(txt)
        out.append(f"| {short[i]} | " + " | ".join(cells) + " |")
    out += ["", "Keterangan: " + ", ".join(f"{s} = `{c}`" for s, c in zip(short, cols)) + ".",
            "Pasangan |r| ≥ 0,7: " + (", ".join(f"{a} × {b} ({r:.2f})" for a, b, r in flagged)
                                       if flagged else "tidak ada") + ".", ""]
    return out


def section_pit(recon):
    out = ["## 4. Uji point-in-time (tensor sekuens 3D)", "",
           f"- Guard `PointInTime`: {recon['guard_point_in_time_event']} event dihitung tanpa "
           "`LookAheadError` (seluruh jendela dibatasi sampai bar keputusan `t*`).",
           f"- Uji pemotongan (L = {recon['lookback_audit']}): bar dipotong tepat pada bar keputusan "
           f"suatu event, lalu konteks (EMA50, rezim ATR120, volume 20 sesi) dan seluruh sekuens 3D "
           f"(L × {len(FEATURE_NAMES)} fitur) dihitung ulang dan dibandingkan persis dengan data penuh.", "",
           "| Titik potong (NY) | bar_idx | event diperiksa | L uji | mismatch |",
           "|---|---|---|---|---|"]
    for r in recon["uji_pemotongan"]:
        out.append(f"| {r['cut_ts_ny']} | {r['cut_bar_idx']} | {r['n_event']} | "
                   f"{r['lookback_uji']} | {r['mismatch']} |")
    return out + [""]


def tulis_laporan_fitur(paths):
    """Tulis `laporan_fitur.md` dari keluaran tahap fitur LSTM; mengembalikan path laporan."""
    recon = json.loads((paths.output / "fitur_rekonsiliasi.json").read_text())
    rows_by_split = {s: common.load_model_csv(paths.data / f"seq_meta_{s}.csv") for s in SPLIT_END}
    lines = [
        "# Laporan Fitur Sekuensial LSTM (RM-22)", "",
        f"Sumber: `model.lstm.fitur`, Kunci-1 k = {recon['k']}, B1: f13 ≥ {recon['theta_b1']}, "
        f"audit L = {recon['lookback_audit']} (kandidat L = {recon['kandidat_lookback']}). "
        f"Data pipeline versi {recon.get('versi_pdhl_data', '–')}. Data uji tidak dimuat. "
        "Statistik tidak dikondisikan pada label.", "",
    ]
    lines += section_numeric(rows_by_split) + section_b1(rows_by_split)
    lines += section_corr(rows_by_split["dev"]) + section_pit(recon)
    out = paths.output / "laporan_fitur.md"
    out.write_text("\n".join(lines))
    return out
