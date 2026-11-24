"""Data sintetis bersama untuk tes pemodelan Gradient Boosting (bukan berkas tes)."""
import datetime as dt
import random

HOURS = list(range(18, 24)) + list(range(0, 17))   # 23 bar per sesi, tanpa jam 17
M = 200.0                                           # sumbu cermin harga
BAR_COLS = ["bar_idx", "ts_ny", "ny_hour", "session_date", "week_start", "bar_in_session",
            "open", "high", "low", "close", "tick_volume", "atr14_h1"]
EVENT_COLS = ["event_id", "split", "session_date", "level_type", "direction_d", "level_price",
              "pdh", "pdl", "session_open", "touch_bar_idx", "decision_bar_idx", "decision_ts_ny",
              "touch_bar_in_session", "touch_open", "touch_high", "touch_low", "touch_close",
              "entry_price", "atr14_h1", "atr14_d_prev"]
LABEL_COLS = ["event_id", "split", "session_date", "label", "label_end_session_date"]


def trading_days(start, n):
    out, day = [], start
    while len(out) < n:
        if day.weekday() < 5:
            out.append(day)
        day += dt.timedelta(days=1)
    return out


def make_bars(n_sessions=30, seed=7, start=dt.date(2016, 1, 4)):
    """Bar numerik sintetis berformat prepare_bars (bar_idx = indeks list)."""
    rnd, bars, price = random.Random(seed), [], 100.0
    for sd in trading_days(start, n_sessions):
        week_start = sd - dt.timedelta(days=sd.weekday())
        for j, h in enumerate(HOURS, start=1):
            day = sd - dt.timedelta(days=1) if h >= 18 else sd
            o = price
            c = o + rnd.uniform(-0.5, 0.5)
            bars.append({"bar_idx": len(bars), "ts_ny": f"{day} {h:02d}:00:00", "ny_hour": h,
                         "session_date": sd.isoformat(), "week_start": week_start.isoformat(),
                         "bar_in_session": j, "open": o, "close": c,
                         "high": max(o, c) + rnd.uniform(0.01, 0.3),
                         "low": min(o, c) - rnd.uniform(0.01, 0.3),
                         "tick_volume": 100.0 + 10 * h + rnd.uniform(0, 5),
                         "atr14_h1": 0.8 + rnd.uniform(0, 0.2)})
            price = c
    return bars


def make_event(bars, t, level_type="PDH", atr_d=5.0, split="dev"):
    b = bars[t]
    d = 1 if level_type == "PDH" else -1
    level = b["high"] - 0.1 if d == 1 else b["low"] + 0.1
    first = t - (b["bar_in_session"] - 1)
    pdh, pdl = (level, level - 3.0) if d == 1 else (level + 3.0, level)
    return {"event_id": f"{b['session_date']}_{level_type}", "split": split,
            "session_date": b["session_date"], "level_type": level_type, "direction_d": str(d),
            "level_price": repr(level), "pdh": repr(pdh), "pdl": repr(pdl),
            "session_open": repr(bars[first]["open"]), "touch_bar_idx": str(t),
            "decision_bar_idx": str(t), "decision_ts_ny": b["ts_ny"],
            "touch_bar_in_session": str(b["bar_in_session"]),
            "touch_open": repr(b["open"]), "touch_high": repr(b["high"]),
            "touch_low": repr(b["low"]), "touch_close": repr(b["close"]),
            "entry_price": repr(b["close"]), "atr14_h1": repr(b["atr14_h1"]),
            "atr14_d_prev": repr(atr_d)}


def mirror_bars(bars):
    return [{**b, "open": M - b["open"], "close": M - b["close"],
             "high": M - b["low"], "low": M - b["high"]} for b in bars]


def mirror_event(e):
    swap = {"touch_high": "touch_low", "touch_low": "touch_high"}
    out = {**e, "level_type": "PDL" if e["level_type"] == "PDH" else "PDH",
           "direction_d": str(-int(e["direction_d"])), "pdh": repr(M - float(e["pdl"])),
           "pdl": repr(M - float(e["pdh"]))}
    for k in ("level_price", "session_open", "touch_open", "touch_close", "entry_price"):
        out[k] = repr(M - float(e[k]))
    for k, src in swap.items():
        out[k] = repr(M - float(e[src]))
    return out


def make_run(paths, n_sessions, seed=11, labels=("Reversal", "Continuation", "Netral")):
    """Tulis run sintetis lengkap ke `paths.input`: bar, event dev/calib (satu per sesi mulai sesi
    ke-21), dan label acak deterministik. Mengembalikan (bars, events)."""
    from pdhl_data import common      # impor lokal: tes menambahkan src ke sys.path lebih dulu

    rnd = random.Random(seed)
    bars = make_bars(n_sessions, seed=seed)
    events, lab = [], []
    for s in range(20, n_sessions):
        t = 23 * s + rnd.choice((3, 9, 14))
        sd = bars[t]["session_date"]
        e = make_event(bars, t, rnd.choice(("PDH", "PDL")), split=common.split_of(sd))
        events.append(e)
        lab.append({"event_id": e["event_id"], "split": e["split"], "session_date": sd,
                    "label": rnd.choice(labels), "label_end_session_date": sd})
    common.write_csv(paths.input / "bars_devcalib.csv", bars, BAR_COLS)
    for split in ("dev", "calib"):
        common.write_csv(paths.input / f"events_{split}.csv",
                         [e for e in events if e["split"] == split], EVENT_COLS)
        common.write_csv(paths.input / f"labels_{split}.csv",
                         [r for r in lab if r["split"] == split], LABEL_COLS)
    return bars, events
