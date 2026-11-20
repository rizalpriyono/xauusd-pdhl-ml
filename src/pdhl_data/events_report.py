"""Laporan kualitas data dan ringkasan event (Markdown) untuk tahap deteksi event."""
import datetime as dt
from collections import Counter

from .common import RESEARCH_END, RESEARCH_START, SPLITS, WARMUP_START
from .events import PARTIAL_SESSION_MIN_BARS

PROPOSAL_ESTIMATE = {"dev": 2200, "calib": 280, "test": 450}   # Tabel 4.4


def _table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _n_reason(events, reason):
    return sum(reason in e["exclusion_reason"].split(";") for e in events)


def _yn(flag):
    return "ya" if flag else ""


def _pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "-"


def write_report(path, k, horizon, bronze, bars, dq, break_by_year, sessions, events, dst_rows, recon):
    research = [s for s in sessions if s["split"]]
    valid = [e for e in events if e["is_valid"]]
    L = []
    add = L.append

    add("# Laporan Data Engineering: Deteksi Event PDH/PDL XAUUSD H1\n")
    add(f"Dibuat oleh `pdhl_data.events` pada {dt.date.today()}. Parameter: **k = {k}**, "
        f"**T = {horizon}**. Label triple barrier dihitung di tahap berikutnya "
        "(`pdhl_data.labels`).\n")

    add("## 1. Bronze (data mentah)\n")
    add(f"- Baris: {len(bronze):,}; rentang UTC {bronze[0]['ts_utc']:%Y-%m-%d %H:%M} s.d. "
        f"{bronze[-1]['ts_utc']:%Y-%m-%d %H:%M}.")
    add("- Header hanya 6 nama (`Time Open High Low Close Volume`), padahal tiap baris berisi 7 kolom. "
        "Skema yang dipakai: `Time, Open, High, Low, Close, BarMinutes, TickVolume`. "
        "`BarMinutes` = jumlah menit berisi tick dalam bar (umumnya 60).")
    add("- Zona waktu sumber: **UTC** (terverifikasi pada bagian 3).\n")

    add("## 2. Silver (pembersihan)\n")
    add(_table(["Langkah", "Bar dibuang"], [
        ("Duplikat timestamp", dq["duplikat"]),
        ("OHLC tidak konsisten (low > min(open, close) dst.)", dq["ohlc_tidak_valid"]),
        (f"Sesi sebelum {WARMUP_START} (di luar periode pemanasan)", dq["sebelum_warmup"]),
        ("Bar pada jam jeda 17.00 NY", dq["bar_jeda_17_ny"]),
    ]))
    add("\nBar jam 17.00 NY yang dibuang per tahun: "
        + ", ".join(f"{y}: {n}" for y, n in sorted(break_by_year.items()))
        + ". Hampir seluruhnya bar 1 menit (`BarMinutes` = 1); pada 2016–2017 seluruhnya tick datar "
        "(O = H = L = C) di dalam jeda harian, sehingga diperlakukan sebagai artefak. Jika dibiarkan, tick ini akan menjadi "
        "*candle* pertama sesi dan mengganggu aturan \"dibuka di luar level\".\n")
    add(f"Silver berisi **{len(bars):,} bar** ({bars[0]['ts_ny']:%Y-%m-%d %H:%M} s.d. "
        f"{bars[-1]['ts_ny']:%Y-%m-%d %H:%M} NY) dan **{len(sessions):,} sesi**. "
        "Data 2015 hanya dipakai untuk pemanasan ATR(14) H1 dan ATR(14) harian.\n")

    add("## 3. Verifikasi zona waktu (IV.II.1)\n")
    add("Timestamp dikonversi UTC → America/New_York (zoneinfo, DST otomatis). Untuk setiap pergantian "
        "DST di periode penelitian, minggu sebelum dan sesudahnya diperiksa. Kriteria PASS: setiap sesi "
        "ditutup paling lambat pada bar 16.00 NY dan tidak ada bar penuh (> 1 menit) di jam 17.00 NY "
        "pada data mentah. Pemeriksaan data mentah ini perlu karena pembersihan membuang bar 17.00, "
        "sehingga pergeseran zona waktu 1 jam hanya terlihat sebelum pembersihan. Sesi yang dibuka "
        "setelah 18.00 tidak menggagalkan jeda; itu bar hilang dan dicatat sebagai *data gap*.\n")
    add(_table(["Minggu DST", "Peran", "Offset UTC", "Sesi", "Buka 18.00", "Tutup ≤ 16.00",
                "Bar 17.00 mentah", "… di antaranya > 1 menit", "Status", "Catatan"],
               [(r["dst_change_week"], r["role"], r["utc_offset"], r["n_sessions"], r["first_bar_18"],
                 r["last_bar_le_16"], r["raw_bars_17"], r["raw_bars_17_full"], r["status"],
                 r["catatan"]) for r in dst_rows]))
    n_pass = sum(r["status"] == "PASS" for r in dst_rows)
    noted = [r for r in dst_rows if r["catatan"]]
    add(f"\n{n_pass} dari {len(dst_rows)} minggu PASS: jeda harian konsisten pada 17.00–18.00 NY di "
        f"seluruh {len(dst_rows) // 2} pergantian DST. "
        + (f"Pada {len(noted)} minggu ({', '.join(str(r['week_start']) for r in noted)}) penyedia data "
           "tidak memuat bar awal sesi Minggu malam. Kekosongan ini merupakan data hilang, bukan "
           "pergeseran zona waktu, dan event yang terdampak dikeluarkan dengan alasan `data_gap` "
           "(bagian 5).\n" if noted else "\n"))

    add("## 4. Sesi perdagangan\n")
    add("Satu sesi = 17.00 NY s.d. 17.00 NY keesokan harinya; `session_date` = tanggal NY saat sesi "
        "ditutup (bar Minggu 18.00 masuk sesi Senin). PDH/PDL = high/low sesi sebelumnya yang ada di data.\n")
    rows = []
    for name, _ in SPLITS:
        ss = [s for s in research if s["split"] == name]
        rows.append((name, len(ss), ss[0]["session_date"], ss[-1]["session_date"],
                     sum(s["is_partial"] for s in ss)))
    add(_table(["Bagian", "Sesi", "Pertama", "Terakhir", f"Sesi < {PARTIAL_SESSION_MIN_BARS} bar"], rows))
    sizes = Counter(s["n_bars"] for s in research)
    add("\nSebaran jumlah bar per sesi: " + ", ".join(f"{k} bar: {v}" for k, v in sorted(sizes.items()))
        + ". Sesi normal = 23 bar (18.00–16.00). Sesi 19–22 bar umumnya hari libur AS dengan penutupan "
        "lebih awal (13.00–14.30 NY).\n")
    partial = [s for s in research if s["is_partial"]]
    add(f"Sesi tidak lengkap (< {PARTIAL_SESSION_MIN_BARS} bar), hanya ditandai: "
        + ", ".join(f"{s['session_date']} ({s['n_bars']})" for s in partial) + ".\n")
    have = {s["session_date"] for s in research}
    missing, d = [], RESEARCH_START
    while d <= RESEARCH_END:
        if d.weekday() < 5 and d not in have:
            missing.append(d)
        d += dt.timedelta(days=1)
    add(f"Hari kerja tanpa sesi ({len(missing)}, libur Good Friday/Natal/Tahun Baru): "
        + ", ".join(str(m) for m in missing) + ". Sesi sesudahnya memakai PDH/PDL dari sesi terakhir "
        "sebelum libur.\n")
    gaps = [b for b in bars if b["hours_since_prev"] and b["hours_since_prev"] > 1
            and b["bar_in_session"] > 1 and b["session_date"] >= RESEARCH_START]
    add(f"Bar hilang di tengah sesi: {len(gaps)} kasus ("
        + ", ".join(f"sebelum {b['ts_ny']:%Y-%m-%d %H:%M} NY" for b in gaps) + "). Tidak diimputasi.\n")
    late = [s for s in research if bars[s["first_bar_idx"]]["missing_before"]]
    add(f"Sesi dibuka setelah 18.00 karena bar awal hilang: {len(late)} kasus ("
        + ", ".join(f"{s['session_date']} dibuka {s['first_ts_ny']:%a %H:%M}" for s in late)
        + "). Tidak diimputasi.\n")

    add("## 5. Event (IV.II.2)\n")
    add("Aturan: (1) *candle* H1 pertama dalam sesi dengan high ≥ PDH / low ≤ PDL; (2) tidak ada event "
        "untuk level itu bila *candle* pertama sesi dibuka > PDH / < PDL; (3) waktu keputusan = "
        "penutupan *candle* ke-k setelah *candle* sentuhan, harga masuk = close-nya; (4) event "
        "dikeluarkan bila T *candle* setelah waktu keputusan melintasi penutupan akhir pekan.\n")
    add("Aturan kualitas data tambahan (`data_gap`): event dikeluarkan bila ada bar hilang (a) di sesi "
        "event sebelum atau pada bar keputusan, (b) di sesi sebelumnya yang menjadi sumber PDH/PDL, atau "
        "(c) di dalam jendela label. Pada kondisi tersebut sentuhan pertama, PDH/PDL, atau label tidak "
        "dapat dipastikan. Aturan ini ditetapkan sebelum label dihitung.\n")
    add("### 5.1 Status setiap level per sesi (periode penelitian)\n")
    rows = []
    for lvl in ("pdh", "pdl"):
        c = Counter(s[f"{lvl}_status"] for s in research)
        rows.append((lvl.upper(), len(research), c["event"], c["opened_beyond"], c["not_touched"]))
    add(_table(["Level", "Sesi", "Event (sentuhan pertama)", "Dibuka di luar level", "Tidak tersentuh"],
               rows))
    add("\n### 5.2 Jumlah event per bagian data\n")
    rows = []
    for name, _ in SPLITS:
        ev = [e for e in events if e["split"] == name]
        va = [e for e in ev if e["is_valid"]]
        n_sessions = sum(s["split"] == name for s in research)
        rows.append((name, len(ev), _n_reason(ev, "window_crosses_week_close"), _n_reason(ev, "data_gap"),
                     len(va), sum(e["level_type"] == "PDH" for e in va),
                     sum(e["level_type"] == "PDL" for e in va),
                     f"{len(va) / n_sessions:.2f}", f"± {PROPOSAL_ESTIMATE[name]:,}"))
    rows.append(("total", len(events), _n_reason(events, "window_crosses_week_close"),
                 _n_reason(events, "data_gap"), len(valid),
                 sum(e["level_type"] == "PDH" for e in valid), sum(e["level_type"] == "PDL" for e in valid),
                 f"{len(valid) / len(research):.2f}", f"± {sum(PROPOSAL_ESTIMATE.values()):,}"))
    add(_table(["Bagian", "Event terdeteksi", "Keluar: lintas akhir pekan", "Keluar: data_gap",
                "Event valid", "PDH",
                "PDL", "Event valid/sesi", "Perkiraan proposal"], rows))
    add("\nSatu event dapat memiliki lebih dari satu alasan, sehingga kolom alasan bisa tumpang tindih.\n")
    gap_ev = [e for e in events if "data_gap" in e["exclusion_reason"].split(";")]
    add(_table(["Event `data_gap`", "Bagian", "Sentuhan (NY)", "(a) sebelum keputusan",
                "(b) sesi sebelumnya", "(c) jendela label", "Alasan lengkap"],
               [(e["event_id"], e["split"], f"{e['touch_ts_ny']:%Y-%m-%d %H:%M}",
                 _yn(e["gap_before_decision"]), _yn(e["gap_in_prev_session"]),
                 _yn(e["gap_in_label_window"]), e["exclusion_reason"]) for e in gap_ev]))
    add("\n### 5.3 Penanda tambahan (tidak mengeluarkan event)\n")
    add(_table(["Penanda", "Event valid", "Keterangan"], [
        ("both_levels_same_bar", sum(e["both_levels_same_bar"] for e in valid),
         "PDH dan PDL tersentuh pada candle yang sama"),
        ("prev_session_partial", sum(e["prev_session_partial"] for e in valid),
         f"PDH/PDL berasal dari sesi < {PARTIAL_SESSION_MIN_BARS} bar"),
        ("session_partial", sum(e["session_partial"] for e in valid),
         f"event terjadi di sesi < {PARTIAL_SESSION_MIN_BARS} bar"),
        ("label_window_crosses_split", sum(e["label_window_crosses_split"] for e in valid),
         "jendela label masuk ke bagian data berikutnya"),
        ("decision_in_next_session", sum(e["decision_in_next_session"] for e in valid),
         "candle keputusan (k > 0) jatuh di sesi berikutnya"),
        ("window_max_gap_h > 2", sum((e["window_max_gap_h"] or 0) > 2 for e in valid),
         "jendela label melewati libur/bar hilang (bukan akhir pekan)"),
    ]))
    hours = Counter(e["touch_ts_ny"].hour for e in valid)
    add("\nJam *candle* sentuhan (NY), event valid: "
        + ", ".join(f"{h:02d}: {hours[h]}" for h in sorted(hours)) + ".\n")

    add("## 6. Angka rekonsiliasi\n")
    add("Angka berikut dicocokkan dengan `references/<config>.json` oleh `pdhl_data verify` "
        f"(berlaku untuk k = {k}, T = {horizon}).\n")
    add(_table(["Kunci", "Nilai"], list(recon.items())))
    path.write_text("\n".join(L) + "\n")
