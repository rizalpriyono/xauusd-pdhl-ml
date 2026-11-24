"""Tahap 1: deteksi event PDH/PDL XAUUSD H1 (bronze -> silver -> gold).

Proposal Final v2, IV.II.1 (Data) dan IV.II.2 (Definisi Event). Logika dipindahkan dari
`Data Engineering PDH-PDL/pdhl_events_local.py`, yang sudah direkonsiliasi dengan implementasi
independen (notebook Databricks di folder lama dan pembangunan ulang dari spesifikasi) sebelum repo
dibuat.

Adendum 2026-11-24: bar sebelum WARMUP_START tidak lagi dibuang sebelum indikator dihitung, tetapi
menjadi riwayat TR/ATR(14) H1, ATR(14) harian, dan PDH/PDL sesi pertama. Dengan begitu tidak ada
nilai kosong sejak bar silver pertama. Keluarannya identik byte demi byte dengan pipeline lama,
kecuali berkas yang tercatat di `references/tesis.json` -> `adendum`.

Deteksi event tidak memerlukan label. Parameter k dan T hanya menentukan bar keputusan dan
pengecualian jendela label (akhir pekan, data hilang).
"""
import csv
import datetime as dt
import json
from collections import Counter, defaultdict
from zoneinfo import ZoneInfo

from .common import RESEARCH_END, RESEARCH_START, SPLITS, WARMUP_START

UTC = ZoneInfo("UTC")
NY = ZoneInfo("America/New_York")

ATR_N = 14
BREAK_HOUR_NY = 17                      # jeda harian 17.00-18.00 New York
SESSION_SHIFT = dt.timedelta(hours=24 - BREAK_HOUR_NY)   # 17.00 NY -> 00.00 tanggal sesi
SESSION_OPEN_HOUR_NY = (BREAK_HOUR_NY + 1) % 24          # sesi normal dibuka 18.00 NY
PARTIAL_SESSION_MIN_BARS = 20           # hanya penanda; proposal final tidak mengeluarkannya


def split_of_date(d):
    if d is None or d < RESEARCH_START or d > RESEARCH_END:
        return None
    return next(name for name, end in SPLITS if d <= end)


def wilder(values, n=ATR_N):
    """Wilder smoothing; nilai pertama = rata-rata n nilai pertama."""
    out = [None] * len(values)
    if len(values) < n:
        return out
    out[n - 1] = sum(values[:n]) / n
    for i in range(n, len(values)):
        out[i] = (out[i - 1] * (n - 1) + values[i]) / n
    return out


def true_range(high, low, prev_close):
    if prev_close is None:
        return high - low
    return max(high - low, abs(high - prev_close), abs(low - prev_close))


# ---------------------------------------------------------------- bronze
def read_bronze(path):
    """Header file hanya 6 nama, padahal tiap baris 7 kolom (dipisah tab):
    Time, Open, High, Low, Close, BarMinutes, TickVolume."""
    rows = []
    with open(path, newline="") as f:
        f.readline()
        for line_no, line in enumerate(f, start=2):
            p = line.rstrip("\r\n").split("\t")
            if len(p) != 7:
                raise ValueError(f"{path} baris {line_no}: {len(p)} kolom, seharusnya 7")
            rows.append({
                "ts_utc": dt.datetime.strptime(p[0], "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC),
                "open": float(p[1]), "high": float(p[2]), "low": float(p[3]), "close": float(p[4]),
                "bar_minutes": int(p[5]), "tick_volume": int(p[6]),
            })
    if not rows:
        raise ValueError(f"{path}: tidak ada baris data")
    return rows


# ---------------------------------------------------------------- silver: bar
def build_silver_bars(bronze, has_daily_break=True):
    """Bersihkan bronze menjadi bar silver (sesi >= WARMUP_START).

    Mengembalikan (bars, dq, break_by_year, warmup). `warmup` adalah bar bersih sebelum
    WARMUP_START. Bar ini tidak masuk silver, tetapi menjadi riwayat agar TR, `hours_since_prev`, dan
    ATR(14) H1 sudah terisi sejak bar silver pertama. Teruskan `warmup` ke `build_sessions` untuk PDH/PDL
    dan ATR harian. `bar_idx` hanya diberikan pada bar silver (mulai 0).

    has_daily_break=False untuk pasar tanpa jeda 17.00-18.00 NY (forex): candle 17.00 tidak
    dibuang dan sesi normal dibuka 17.00 NY."""
    open_hour = SESSION_OPEN_HOUR_NY if has_daily_break else BREAK_HOUR_NY
    dq = Counter()
    seen, clean = set(), []
    break_by_year = Counter()
    for r in sorted(bronze, key=lambda x: x["ts_utc"]):
        if r["ts_utc"] in seen:
            dq["duplikat"] += 1
            continue
        seen.add(r["ts_utc"])
        o, h, l, c = r["open"], r["high"], r["low"], r["close"]
        if not (0 < l <= min(o, c) and max(o, c) <= h):
            dq["ohlc_tidak_valid"] += 1
            continue
        ts_ny = r["ts_utc"].astimezone(NY)
        session_date = (ts_ny + SESSION_SHIFT).date()
        history = session_date < WARMUP_START
        if history:
            dq["sebelum_warmup"] += 1           # tidak masuk silver; hanya riwayat indikator
        if has_daily_break and ts_ny.hour == BREAK_HOUR_NY:
            if not history:
                dq["bar_jeda_17_ny"] += 1
                break_by_year[ts_ny.year] += 1
            continue
        clean.append({**r, "ts_ny": ts_ny, "ny_hour": ts_ny.hour,
                      "utc_offset_h": int(ts_ny.utcoffset().total_seconds() // 3600),
                      "session_date": session_date,
                      "week_start": session_date - dt.timedelta(days=session_date.weekday())})

    # Indikator dihitung atas riwayat + silver dalam satu urutan, baru kemudian riwayat dipotong.
    prev, in_session = None, Counter()
    for b in clean:
        in_session[b["session_date"]] += 1
        b["bar_in_session"] = in_session[b["session_date"]]
        b["hours_since_prev"] = None if prev is None else (b["ts_utc"] - prev["ts_utc"]).total_seconds() / 3600
        b["tr"] = true_range(b["high"], b["low"], None if prev is None else prev["close"])
        # Bar hilang tepat sebelum bar ini: celah > 1 jam di tengah sesi, atau sesi tidak dibuka 18.00.
        b["missing_before"] = (b["hours_since_prev"] > 1 if b["bar_in_session"] > 1
                               else b["ny_hour"] != open_hour)
        prev = b
    for b, atr in zip(clean, wilder([b["tr"] for b in clean])):
        b["atr14_h1"] = atr

    n_warmup = sum(b["session_date"] < WARMUP_START for b in clean)   # bar terurut: riwayat = prefiks
    warmup, bars = clean[:n_warmup], clean[n_warmup:]
    for i, b in enumerate(bars):
        b["bar_idx"] = i
    return bars, dq, break_by_year, warmup


# ---------------------------------------------------------------- silver: sesi
def _aggregate_sessions(bars):
    by_session = defaultdict(list)
    for b in bars:
        by_session[b["session_date"]].append(b)
    return [{
        "session_date": sd, "week_start": bs[0]["week_start"],
        "s_open": bs[0]["open"], "s_high": max(b["high"] for b in bs),
        "s_low": min(b["low"] for b in bs), "s_close": bs[-1]["close"],
        "n_bars": len(bs), "first_bar_idx": bs[0].get("bar_idx"), "last_bar_idx": bs[-1].get("bar_idx"),
        "first_ts_ny": bs[0]["ts_ny"], "last_ts_ny": bs[-1]["ts_ny"],
        "has_data_gap": any(b["missing_before"] for b in bs),
    } for sd, bs in sorted(by_session.items())]


def build_sessions(bars, warmup_bars=()):
    """Sesi silver dari `bars`. `warmup_bars` (keluaran ke-4 `build_silver_bars`) hanya menjadi
    riwayat: PDH/PDL sesi silver pertama dan ATR(14) harian dihitung dengan sesi-sesi tersebut,
    tetapi sesi riwayat tidak dikembalikan."""
    history = _aggregate_sessions(warmup_bars)
    sessions = _aggregate_sessions(bars)
    if history and sessions and history[-1]["session_date"] >= sessions[0]["session_date"]:
        raise ValueError("warmup_bars harus seluruhnya sebelum sesi silver pertama "
                         f"({history[-1]['session_date']} >= {sessions[0]['session_date']})")
    chain = history + sessions

    prev = None
    for s in chain:
        s["prev_session_date"] = prev and prev["session_date"]
        s["pdh"] = prev and prev["s_high"]
        s["pdl"] = prev and prev["s_low"]
        s["prev_n_bars"] = prev and prev["n_bars"]
        s["prev_has_data_gap"] = prev is not None and prev["has_data_gap"]
        s["tr_d"] = true_range(s["s_high"], s["s_low"], prev and prev["s_close"])
        prev = s
    atr_d = wilder([s["tr_d"] for s in chain])
    for i, s in enumerate(chain):
        s["atr14_d"] = atr_d[i]
        s["atr14_d_prev"] = atr_d[i - 1] if i > 0 else None   # hanya sesi yang sudah selesai

    for s in sessions:
        s["is_partial"] = s["n_bars"] < PARTIAL_SESSION_MIN_BARS
        s["split"] = split_of_date(s["session_date"])

        bs = bars[s["first_bar_idx"]: s["last_bar_idx"] + 1]
        for lvl, beyond, touched in (
            ("pdh", lambda: s["s_open"] > s["pdh"], lambda b: b["high"] >= s["pdh"]),
            ("pdl", lambda: s["s_open"] < s["pdl"], lambda b: b["low"] <= s["pdl"]),
        ):
            touch = None
            if s[lvl] is None:
                status = "no_prev_session"
            elif beyond():
                status = "opened_beyond"
            else:
                touch = next((b["bar_idx"] for b in bs if touched(b)), None)
                status = "not_touched" if touch is None else "event"
            s[f"{lvl}_status"] = status
            s[f"{lvl}_touch_idx"] = touch
    return sessions


# ---------------------------------------------------------------- gold: event
def detect_events(bars, sessions, k, horizon):
    week_last = {}
    for b in bars:
        week_last[b["week_start"]] = b["bar_idx"]
    last_idx = len(bars) - 1
    n_missing = [0]                     # n_missing[i] = jumlah bar missing_before dengan idx < i
    for b in bars:
        n_missing.append(n_missing[-1] + b["missing_before"])

    def any_missing(lo, hi):
        hi = min(hi, last_idx)
        return hi >= lo and n_missing[hi + 1] - n_missing[lo] > 0

    events = []
    for s in sessions:
        if s["split"] is None:
            continue
        for lvl, d in (("PDH", 1), ("PDL", -1)):
            touch_idx = s[f"{lvl.lower()}_touch_idx"]
            if touch_idx is None:
                continue
            t = bars[touch_idx]
            dec_idx = touch_idx + k
            dec = bars[dec_idx] if dec_idx <= last_idx else None
            wk_last = week_last[t["week_start"]]
            win = bars[dec_idx + 1: min(dec_idx + horizon, last_idx) + 1]
            win_end = bars[dec_idx + horizon] if dec_idx + horizon <= last_idx else None

            reasons = []
            if dec_idx + horizon > wk_last:
                reasons.append("window_crosses_week_close")
            elif dec_idx + horizon > last_idx:
                reasons.append("insufficient_future_bars")
            # data_gap: sentuhan pertama, PDH/PDL, atau label tidak bisa dipastikan karena bar hilang.
            gap_before = any_missing(s["first_bar_idx"], dec_idx)
            gap_window = any_missing(dec_idx + 1, dec_idx + horizon)
            if gap_before or gap_window or s["prev_has_data_gap"]:
                reasons.append("data_gap")

            other = "pdl" if lvl == "PDH" else "pdh"
            events.append({
                "event_id": f"{s['session_date']}_{lvl}",
                "session_date": s["session_date"], "split": s["split"],
                "level_type": lvl, "direction_d": d,
                "level_price": s[lvl.lower()], "pdh": s["pdh"], "pdl": s["pdl"],
                "prev_session_date": s["prev_session_date"],
                "prev_session_n_bars": s["prev_n_bars"], "session_n_bars": s["n_bars"],
                "session_open": s["s_open"],
                "touch_bar_idx": touch_idx, "touch_ts_utc": t["ts_utc"], "touch_ts_ny": t["ts_ny"],
                "touch_bar_in_session": t["bar_in_session"],
                "touch_open": t["open"], "touch_high": t["high"], "touch_low": t["low"],
                "touch_close": t["close"],
                "both_levels_same_bar": s[f"{other}_touch_idx"] == touch_idx,
                "decision_k": k, "decision_bar_idx": dec_idx,
                "decision_ts_ny": dec and dec["ts_ny"],
                "decision_in_next_session": dec is not None and dec["session_date"] != s["session_date"],
                "entry_price": dec and dec["close"], "atr14_h1": dec and dec["atr14_h1"],
                "atr14_d_prev": s["atr14_d_prev"],
                "horizon_T": horizon, "week_last_bar_idx": wk_last,
                "bars_to_week_close": wk_last - dec_idx,
                "window_max_gap_h": max((b["hours_since_prev"] for b in win), default=None),
                "window_end_session_date": win_end and win_end["session_date"],
                "label_window_crosses_split": (win_end is not None
                                               and split_of_date(win_end["session_date"]) != s["split"]),
                "prev_session_partial": s["prev_n_bars"] < PARTIAL_SESSION_MIN_BARS,
                "session_partial": s["is_partial"],
                "gap_before_decision": gap_before,
                "gap_in_prev_session": s["prev_has_data_gap"],
                "gap_in_label_window": gap_window,
                "exclusion_reason": ";".join(reasons),
                "is_valid": not reasons,
            })
    events.sort(key=lambda e: (e["touch_bar_idx"], e["level_type"]))
    return events


# ---------------------------------------------------------------- verifikasi zona waktu
def dst_verification(bronze, sessions):
    """Untuk setiap pergantian DST: minggu sebelum dan sesudahnya harus punya jeda harian
    17.00-18.00 NY. PASS bila setiap sesi ditutup paling lambat pada bar 16.00 dan tidak ada
    bar penuh (> 1 menit) di jam 17.00 pada data mentah. Pembersihan membuang bar 17.00, jadi
    pemeriksaan data mentah ini yang menangkap pergeseran zona waktu 1 jam ke arah mana pun.
    Sesi yang dibuka setelah 18.00 tidak menggagalkan jeda; itu bar hilang (data gap)."""
    raw17, raw17_full = Counter(), Counter()
    for r in bronze:
        ny = r["ts_utc"].astimezone(NY)
        if ny.hour == BREAK_HOUR_NY:
            sd = (ny + SESSION_SHIFT).date()
            week = sd - dt.timedelta(days=sd.weekday())
            raw17[week] += 1
            raw17_full[week] += r["bar_minutes"] > 1

    weeks = defaultdict(list)
    for s in sessions:
        weeks[s["week_start"]].append(s)
    ordered = sorted(weeks)
    offset = {w: weeks[w][0]["first_ts_ny"].utcoffset().total_seconds() / 3600 for w in ordered}

    rows = []
    for i in range(1, len(ordered)):
        before, after = ordered[i - 1], ordered[i]
        if offset[before] == offset[after] or after < RESEARCH_START:
            continue
        for role, w in (("sebelum", before), ("sesudah", after)):
            ss = weeks[w]
            n_first18 = sum(s["first_ts_ny"].hour == SESSION_OPEN_HOUR_NY for s in ss)
            n_last_ok = sum(s["last_ts_ny"].hour <= 16 for s in ss)
            rows.append({
                "dst_change_week": after, "role": role, "week_start": w,
                "utc_offset": int(offset[w]), "n_sessions": len(ss),
                "first_bar_18": n_first18, "last_bar_le_16": n_last_ok,
                "raw_bars_17": raw17[w], "raw_bars_17_full": raw17_full[w],
                "status": "PASS" if n_last_ok == len(ss) and raw17_full[w] == 0 else "GAGAL",
                "catatan": "; ".join(
                    f"bar hilang di awal sesi (data gap): sesi {s['session_date']} dibuka "
                    f"{s['first_ts_ny']:%a %H:%M}"
                    for s in ss if s["first_ts_ny"].hour != SESSION_OPEN_HOUR_NY),
            })
    return rows


# ---------------------------------------------------------------- keluaran
def fmt(v):
    if isinstance(v, dt.datetime):
        return v.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(v, bool):
        return "true" if v else "false"
    return "" if v is None else str(v)


def write_csv(path, rows, cols):
    """Format CSV keluaran data engineering (dialek excel, baris diakhiri CRLF)."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([fmt(r[c]) for c in cols])


BAR_COLS = ["bar_idx", "ts_utc", "ts_ny", "ny_hour", "utc_offset_h", "session_date", "week_start", "bar_in_session",
            "open", "high", "low", "close", "bar_minutes", "tick_volume", "hours_since_prev",
            "missing_before", "tr", "atr14_h1"]
SESSION_COLS = ["session_date", "split", "week_start", "s_open", "s_high", "s_low", "s_close",
                "n_bars", "is_partial", "has_data_gap", "prev_has_data_gap", "first_bar_idx", "last_bar_idx", "first_ts_ny", "last_ts_ny",
                "prev_session_date", "prev_n_bars", "pdh", "pdl", "tr_d", "atr14_d", "atr14_d_prev",
                "pdh_status", "pdh_touch_idx", "pdl_status", "pdl_touch_idx"]
EVENT_COLS = ["event_id", "session_date", "split", "level_type", "direction_d", "level_price",
              "pdh", "pdl", "prev_session_date", "prev_session_n_bars", "session_n_bars",
              "session_open", "touch_bar_idx", "touch_ts_utc", "touch_ts_ny", "touch_bar_in_session",
              "touch_open", "touch_high", "touch_low", "touch_close", "both_levels_same_bar",
              "decision_k", "decision_bar_idx", "decision_ts_ny", "decision_in_next_session",
              "entry_price", "atr14_h1", "atr14_d_prev", "horizon_T", "week_last_bar_idx",
              "bars_to_week_close", "window_max_gap_h", "window_end_session_date",
              "label_window_crosses_split", "prev_session_partial", "session_partial",
              "gap_before_decision", "gap_in_prev_session", "gap_in_label_window",
              "exclusion_reason", "is_valid"]
DST_COLS = ["dst_change_week", "role", "week_start", "utc_offset", "n_sessions", "first_bar_18",
            "last_bar_le_16", "raw_bars_17", "raw_bars_17_full", "status", "catatan"]


def reconciliation(bars, sessions, events, dst_rows):
    """Angka rekonsiliasi tahap events (dibandingkan dengan `references/tesis.json`)."""
    valid = [e for e in events if e["is_valid"]]
    out = {
        "silver_bars": len(bars),
        "silver_sessions": len(sessions),
        "research_sessions": sum(s["split"] is not None for s in sessions),
        "sum_tr_bars": round(sum(b["tr"] for b in bars), 4),
        "sum_atr14_h1_bars": round(sum(b["atr14_h1"] or 0 for b in bars), 4),
        "sum_atr14_d_sessions": round(sum(s["atr14_d"] or 0 for s in sessions), 4),
        "silver_bars_missing_before": sum(b["missing_before"] for b in bars),
        "sessions_with_data_gap": sum(s["has_data_gap"] for s in sessions),
        "dst_rows": len(dst_rows),
        "dst_pass": sum(r["status"] == "PASS" for r in dst_rows),
        "dst_rows_with_note": sum(bool(r["catatan"]) for r in dst_rows),
        "events_both_levels_same_bar": sum(e["both_levels_same_bar"] for e in events),
        "events_decision_in_next_session": sum(e["decision_in_next_session"] for e in events),
        "events_label_window_crosses_split": sum(e["label_window_crosses_split"] for e in events),
        "events_window_crosses_week_close": sum(
            "window_crosses_week_close" in e["exclusion_reason"].split(";") for e in events),
        "events_data_gap": sum("data_gap" in e["exclusion_reason"].split(";") for e in events),
        "sum_window_max_gap_h_all": round(sum(e["window_max_gap_h"] or 0 for e in events), 4),
    }
    out.update({f"valid_{s}_{l}": sum(e["split"] == s and e["level_type"] == l for e in valid)
                for s, _ in SPLITS for l in ("PDH", "PDL")})
    out["events_all"] = len(events)
    out["events_valid"] = len(valid)
    out["sum_touch_bar_idx_valid"] = sum(e["touch_bar_idx"] for e in valid)
    out["sum_entry_price_valid"] = round(sum(e["entry_price"] for e in valid), 4)
    # `or 0`: hanya relevan untuk data pendek tanpa pemanasan; data asli punya pemanasan 2015.
    out["sum_atr14_h1_valid"] = round(sum(e["atr14_h1"] or 0 for e in valid), 4)
    out["sum_atr14_d_prev_valid"] = round(sum(e["atr14_d_prev"] or 0 for e in valid), 4)
    return out


def run_events(raw_path, out_dir, k, horizon, write_report=True):
    """Jalankan bronze -> silver -> gold dan tulis keluarannya ke `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    bronze = read_bronze(raw_path)
    bars, dq, break_by_year, warmup = build_silver_bars(bronze)
    sessions = build_sessions(bars, warmup)
    events = detect_events(bars, sessions, k, horizon)
    dst_rows = dst_verification(bronze, sessions)

    write_csv(out_dir / "silver_xauusd_h1.csv", bars, BAR_COLS)
    write_csv(out_dir / "silver_sessions.csv", sessions, SESSION_COLS)
    write_csv(out_dir / "gold_pdhl_events.csv", events, EVENT_COLS)
    write_csv(out_dir / "dq_dst_verification.csv", dst_rows, DST_COLS)

    recon = reconciliation(bars, sessions, events, dst_rows)
    (out_dir / "de_rekonsiliasi.json").write_text(json.dumps(recon, indent=2) + "\n")
    if write_report:
        from .events_report import write_report as _write
        _write(out_dir / "laporan_data_engineering.md", k, horizon, bronze, bars, dq,
               break_by_year, sessions, events, dst_rows, recon)
    print(f"[events] bronze={len(bronze)} silver_bars={len(bars)} sessions={len(sessions)} "
          f"events={len(events)} valid={recon['events_valid']} (k={k}, T={horizon})")
    return recon
