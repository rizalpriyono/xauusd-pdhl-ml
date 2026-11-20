"""Laporan pelabelan triple barrier untuk data pengembangan dan kalibrasi.

Membaca `data/labels_*.csv` (lewat loader ber-guard) dan menulis `output/laporan_label.md`.
Rujukan pasal di teks laporan diambil dari `rujukan` pada konfigurasi.
"""
import datetime as dt
import json
import math
import random
from collections import Counter

from . import common
from .common import SPLIT_END
from .labels import LABELS

SAMPLE_N = 5


def pct(n, total):
    return f"{100 * n / total:.1f}%" if total else "–"


def wilson(n, total, z=1.96):
    if not total:
        return (math.nan, math.nan)
    p = n / total
    den = 1 + z * z / total
    mid = (p + z * z / (2 * total)) / den
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / den
    return mid - half, mid + half


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else math.nan


def num(x, nd=3):
    return f"{x:.{nd}f}".replace("-", "−")


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def load(data_dir):
    rows = []
    for split in SPLIT_END:
        for r in common.load_model_csv(data_dir / f"labels_{split}.csv"):
            for c in ("R_rev", "R_cont", "atr14_h1", "entry_price", "target_price", "stop_price",
                      "close_T"):
                r[c] = float(r[c])
            r["offset"] = int(r["barrier_bar_offset"]) if r["barrier_bar_offset"] else None
            r["both"] = common.parse_bool(r["both_same_bar"])
            rows.append(r)
    return rows


def sec_classes(rows, ref):
    body = []
    for split in SPLIT_END:
        for lv in ("PDH", "PDL", "Semua"):
            part = [r for r in rows if r["split"] == split and lv in (r["level_type"], "Semua")]
            c = Counter(r["label"] for r in part)
            body.append([split, lv, len(part)] + [f"{c[lab]} ({pct(c[lab], len(part))})"
                                                  for lab in LABELS])
    return ("## 1. Proporsi label per bagian dan level\n\n"
            + table(["Bagian", "Level", "Event", *LABELS], body)
            + "\n\nProporsi kalibrasi hanya dicatat; data 2024 tidak dipakai untuk keputusan apa pun "
              f"selain kalibrasi probabilitas ({ref['kalibrasi']}).\n")


def sec_same_bar(rows, ref):
    body = [[s, len(p := [r for r in rows if r["split"] == s]), sum(r["both"] for r in p),
             pct(sum(r["both"] for r in p), len(p))] for s in SPLIT_END]
    cases = [[r["event_id"], r["decision_ts_ny"], r["barrier_ts_ny"], r["offset"]]
             for r in rows if r["both"]]
    return ("## 2. Kedua barrier tersentuh dalam candle yang sama\n\n"
            "Urutannya tidak dapat diketahui dari data H1, sehingga kasus ini dilabeli "
            f"*Continuation* (penyelesaian konservatif terhadap posisi *reversal*, {ref['label']}).\n\n"
            + table(["Bagian", "Event", "Dua barrier satu candle", "Proporsi"], body) + "\n\n"
            + table(["event_id", "Keputusan (NY)", "Candle barrier (NY)", "Candle ke-"], cases)
            + "\n")


def sec_base_rate(rows, kunci, ref):
    a, b = kunci["a"], kunci["b"]
    body = []
    for split in SPLIT_END:
        part = [r for r in rows if r["split"] == split and r["label"] != "Netral"]
        n_rev = sum(r["label"] == "Reversal" for r in part)
        lo, hi = wilson(n_rev, len(part))
        body.append([split, len(part), n_rev, pct(n_rev, len(part)),
                     f"{100 * lo:.1f}–{100 * hi:.1f}%"])
    return ("## 3. Reversal di antara event yang menyentuh barrier\n\n"
            f"Pada gerak acak tanpa kecenderungan, peluang target tersentuh lebih dahulu kira-kira "
            f"b/(a + b) = {b:g}/({a:g} + {b:g}) = {100 * b / (a + b):.1f}% ({ref['label']}). Selisih dari "
            "angka ini bukan temuan; tabel hanya memeriksa geometri label.\n\n"
            + table(["Bagian", "Event tersentuh barrier", "Reversal", "Proporsi", "IK 95% (Wilson)"],
                    body) + "\n")


def sec_per_year(rows, ref):
    dev = [r for r in rows if r["split"] == "dev"]
    body = []
    for year in sorted({r["session_date"][:4] for r in dev}):
        part = [r for r in dev if r["session_date"][:4] == year]
        c = Counter(r["label"] for r in part)
        body.append([year, len(part)] + [pct(c[lab], len(part)) for lab in LABELS])
    return (f"## 4. Proporsi label per tahun (pengembangan, {ref['deskriptif']})\n\n"
            + table(["Tahun", "Event", *LABELS], body) + "\n")


def sec_time_to_barrier(rows, kunci, ref):
    dev = [r for r in rows if r["split"] == "dev"]
    body = []
    for j in range(1, kunci["T"] + 1):
        c = Counter(r["label"] for r in dev if r["offset"] == j)
        n = c["Reversal"] + c["Continuation"]
        body.append([j, c["Reversal"], c["Continuation"], n, pct(n, len(dev))])
    n_net = sum(r["label"] == "Netral" for r in dev)
    body.append([f"tidak tersentuh dalam {kunci['T']}", "–", "–", n_net, pct(n_net, len(dev))])
    return (f"## 5. Waktu hingga barrier pertama (pengembangan, {ref['deskriptif']})\n\n"
            + table(["Candle ke-", "Reversal", "Continuation", "Jumlah", "Proporsi"], body) + "\n")


def sec_r(rows, kunci, ref):
    a, b = kunci["a"], kunci["b"]
    body = []
    for split in SPLIT_END:
        part = [r for r in rows if r["split"] == split]
        body.append([split, len(part), num(mean(r["R_rev"] for r in part)),
                     num(mean(r["R_cont"] for r in part))])
    net = [r for r in rows if r["split"] == "dev" and r["label"] == "Netral"]
    by_class = [[lab, num(mean(r["R_rev"] for r in rows if r["split"] == "dev" and r["label"] == lab)),
                 num(mean(r["R_cont"] for r in rows if r["split"] == "dev" and r["label"] == lab))]
                for lab in LABELS]
    return (
        f"## 6. Hasil dalam satuan R ({ref['R']}, sebelum biaya)\n\n"
        "Rata-rata R per event bila **selalu** memilih satu aksi. Ini deskriptif, bukan klaim "
        "profitabilitas.\n\n"
        + table(["Bagian", "Event", "Rata-rata R selalu reversal", "Rata-rata R selalu continuation"],
                body)
        + "\n\nRata-rata R per kelas (pengembangan):\n\n"
        + table(["Label", "R reversal", "R continuation"], by_class)
        + f"\n\n**Pratinjau m_rev dan m_cont** (rata-rata R pada {len(net)} event Netral di data "
          f"pengembangan): m_rev = {num(mean(r['R_rev'] for r in net), 4)}, "
          f"m_cont = {num(mean(r['R_cont'] for r in net), 4)}. Nilai resmi ditetapkan pada aturan "
          f"keputusan ({ref['keputusan']}).\n\n"
        f"**Catatan untuk aturan keputusan ({ref['keputusan']}).** Karena posisi *continuation* memakai *barrier* "
        f"yang sama dalam peran terbalik, R_cont = −(b/a) × R_rev untuk setiap event "
        f"(+{a / b:g} ↔ −1, −1 ↔ +{b / a:g}, dan pada Netral kedua nilai berasal dari Δ yang sama). "
        f"Akibatnya m_cont = −(b/a) × m_rev dan E(continuation) = −(b/a) × E(reversal) untuk "
        f"probabilitas apa pun. Dengan δ = 0,1R, aturan keputusan setara dengan: *reversal* bila "
        f"E(reversal) ≥ δ, *continuation* bila E(reversal) ≤ −(a/b) × δ = −{a / b * 0.1:g}R, dan "
        "abstain di antaranya.\n")


def sec_purge(recon, ref):
    return (f"## 7. Purging dan embargo ({ref['pembagian']})\n\n"
            f"- Event dev yang jendela labelnya melewati akhir 2023: {recon['dev_purged']}.\n"
            f"- Event calib yang jendela labelnya melewati akhir 2024: {recon['calib_purged']}.\n"
            f"- Event yang jendela labelnya melintasi pergantian tahun (batas lipatan walk-forward): "
            f"{recon['label_crosses_year']}. Kolom `label_crosses_year` dipakai saat validasi "
            "walk-forward.\n")


def sec_sample(rows, seed):
    dev = sorted((r for r in rows if r["split"] == "dev"), key=lambda r: r["event_id"])
    pick = sorted(random.Random(seed).sample(dev, min(SAMPLE_N, len(dev))),
                  key=lambda r: r["event_id"])
    body = [[r["event_id"], r["decision_ts_ny"], num(r["entry_price"], 2), num(r["atr14_h1"], 3),
             num(r["target_price"], 3), num(r["stop_price"], 3), r["label"],
             r["barrier_ts_ny"] or "–", num(r["close_T"], 2)] for r in pick]
    return (f"## 8. Sampel untuk cek manual (seed {seed})\n\n"
            "Waktu adalah waktu **buka** candle H1 New York. Keputusan diambil saat candle keputusan "
            "ditutup; barrier diperiksa mulai candle berikutnya.\n\n"
            + table(["event_id", "Candle keputusan", "Entry", "ATR", "Target", "Stop", "Label",
                     "Candle barrier", "Close ke-T"], body) + "\n")


def write_report(cfg, paths):
    kunci, ref = cfg["label"], cfg["rujukan"]
    rows = load(paths.data)
    recon = json.loads((paths.output / "label_rekonsiliasi.json").read_text())
    head = (f"# Laporan Pelabelan Triple Barrier dan Satuan R — `{cfg['name']}`\n\n"
            f"Dibuat oleh `pdhl_data.labels_report` pada {dt.date.today().isoformat()}. "
            f"{cfg.get('judul', '')}. Parameter label: **a = {kunci['a']:g}**, "
            f"**b = {kunci['b']:g}**, **T = {kunci['T']}**, **k = {kunci['k']}** "
            f"(dikunci {kunci['tanggal_kunci']}). Hanya data pengembangan (2016–2023) dan "
            f"kalibrasi (2024); data uji terkunci di `kunci_uji/` sampai "
            f"{cfg['lock']['dibuka_pada']}.\n")
    parts = [head, sec_classes(rows, ref), sec_same_bar(rows, ref), sec_base_rate(rows, kunci, ref),
             sec_per_year(rows, ref), sec_time_to_barrier(rows, kunci, ref), sec_r(rows, kunci, ref),
             sec_purge(recon, ref), sec_sample(rows, cfg["report"]["sample_seed"]),
             "## 9. Angka rekonsiliasi\n\n"
             + table(["Kunci", "Nilai"], recon.items()) + "\n"]
    out = paths.output / "laporan_label.md"
    out.write_text("\n".join(parts))
    print(f"[laporan] {out}")
    return out
