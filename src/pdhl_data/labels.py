"""Tahap 3: pelabelan triple barrier dan satuan R.

Proposal Final v2, IV.II.3 dan Tabel 4.3. Hanya data pengembangan (2016-2023) dan kalibrasi (2024)
yang dilabeli; data dimuat lewat `common.load_model_csv`, yang menolak data uji.

Setiap event diperlakukan sebagai kandidat posisi reversal (sell pada PDH, buy pada PDL) yang
dibuka pada close candle keputusan. Dengan d = +1 (PDH) / -1 (PDL):
    target = entry - d*a*ATR,  stop = entry + d*b*ATR
Barrier diperiksa mulai candle setelah keputusan sampai candle ke-T:
    hanya target tersentuh -> Reversal; stop tersentuh (termasuk keduanya dalam satu candle)
    -> Continuation; tidak ada yang tersentuh -> Netral.
Purging (IV.II.4): event yang jendela labelnya melewati akhir bagiannya dikeluarkan dari bagian itu.
"""
import json
import math

from . import common
from .common import CLASSES as LABELS
from .common import SPLIT_END
from .split import LockViolation, check_data_files, check_lock_params, read_manifest

LABEL_COLS = ["event_id", "split", "session_date", "level_type", "direction_d", "decision_bar_idx",
              "decision_ts_ny", "entry_price", "atr14_h1", "a", "b", "T", "k", "target_price",
              "stop_price", "label", "barrier_bar_offset", "barrier_ts_ny", "both_same_bar",
              "close_T", "label_end_bar_idx", "label_end_session_date", "label_crosses_year",
              "R_rev", "R_cont", "R_none"]
TOL = 1e-9


def barriers(entry, atr, d, a, b):
    """(target, stop) posisi reversal."""
    return entry - d * a * atr, entry + d * b * atr


def r_values(label, entry, close_t, atr, d, a, b):
    """(R_rev, R_cont) menurut Tabel 4.3; tidak bertransaksi selalu 0."""
    if label == "Reversal":
        return a / b, -1.0
    if label == "Continuation":
        return -1.0, b / a
    delta = close_t - entry
    return -d * delta / (b * atr), d * delta / (a * atr)


def label_event(window, entry, atr, d, a, b, T):
    """window: T candle setelah candle keputusan (dict berisi high, low, close bertipe float)."""
    if len(window) != T:
        raise ValueError(f"jendela label berisi {len(window)} candle, seharusnya T={T}")
    if not atr > 0:
        raise ValueError(f"ATR harus > 0, didapat {atr}")
    if d not in (1, -1):
        raise ValueError(f"faktor arah d harus +1 atau -1, didapat {d}")
    target, stop = barriers(entry, atr, d, a, b)
    label, offset, both = "Netral", None, False
    for j, bar in enumerate(window, start=1):
        if d == 1:
            hit_t, hit_s = bar["low"] <= target, bar["high"] >= stop
        else:
            hit_t, hit_s = bar["high"] >= target, bar["low"] <= stop
        if hit_t or hit_s:
            label = "Continuation" if hit_s else "Reversal"
            offset, both = j, hit_t and hit_s
            break
    close_t = window[-1]["close"]
    r_rev, r_cont = r_values(label, entry, close_t, atr, d, a, b)
    if label == "Netral" and not (-1 < r_rev < a / b and -1 < r_cont < b / a):
        raise AssertionError(f"R Netral di luar rentang: R_rev={r_rev}, R_cont={r_cont}")
    return {"label": label, "barrier_bar_offset": offset, "both_same_bar": both,
            "target_price": target, "stop_price": stop, "close_T": close_t,
            "R_rev": r_rev, "R_cont": r_cont, "R_none": 0.0}


def _num_bar(b):
    out = {"high": float(b["high"]), "low": float(b["low"]), "close": float(b["close"])}
    if not all(math.isfinite(v) for v in out.values()):
        raise ValueError(f"bar {b['bar_idx']}: OHLC tidak finite")
    return out


def _check_decision_bar(e, bars):
    """Entry dan ATR event harus sama persis (toleransi absolut 1e-9, seperti Spark) dengan bar
    keputusan. Diperiksa untuk semua event yang bar keputusannya tersedia, sebelum purging."""
    dec = int(e["decision_bar_idx"])
    if dec >= len(bars):
        return
    entry, atr = float(e["entry_price"]), float(e["atr14_h1"])
    if not math.isclose(float(bars[dec]["close"]), entry, rel_tol=0, abs_tol=TOL):
        raise ValueError(f"{e['event_id']}: entry_price tidak sama dengan close bar keputusan")
    if not math.isclose(float(bars[dec]["atr14_h1"]), atr, rel_tol=0, abs_tol=TOL):
        raise ValueError(f"{e['event_id']}: atr14_h1 tidak sama dengan ATR bar keputusan")


def label_events(events, bars, label_params, split_end=SPLIT_END):
    """Label setiap event dev/calib. Mengembalikan (baris label, event yang di-purge)."""
    a, b, T, k = (label_params[key] for key in ("a", "b", "T", "k"))
    common.assert_no_test_data(events, source="events")
    common.assert_no_test_data(bars, source="bars")
    if any(int(bar["bar_idx"]) != i for i, bar in enumerate(bars)):
        raise ValueError("bar harus prefiks bar_idx yang kontinu (indeks list = bar_idx)")
    rows, purged = [], []
    for e in events:
        if e["split"] not in split_end:
            raise common.TestDataLeak(f"{e['event_id']}: split {e['split']} tidak boleh dilabeli")
        if e["split"] != common.split_of(e["session_date"]):
            raise ValueError(f"{e['event_id']}: split {e['split']} tidak sesuai tanggal sesi")
        _check_decision_bar(e, bars)
        dec = int(e["decision_bar_idx"])
        end = dec + T
        if end >= len(bars):
            purged.append({**e, "purge_reason": "jendela_label_melewati_data_tersedia"})
            continue
        end_bar = bars[end]
        if end_bar["session_date"] > split_end[e["split"]]:
            purged.append({**e, "purge_reason": "jendela_label_melewati_akhir_bagian"})
            continue
        entry, atr = float(e["entry_price"]), float(e["atr14_h1"])
        window_raw = bars[dec + 1: end + 1]
        if any(common.parse_bool(w["missing_before"]) for w in window_raw):
            raise ValueError(f"{e['event_id']}: ada bar hilang di jendela label (cek data_gap)")
        res = label_event([_num_bar(w) for w in window_raw], entry, atr,
                          int(e["direction_d"]), a, b, T)
        off = res["barrier_bar_offset"]
        rows.append({
            **{c: e[c] for c in ("event_id", "split", "session_date", "level_type", "direction_d",
                                 "decision_bar_idx", "decision_ts_ny")},
            "entry_price": entry, "atr14_h1": atr, "a": a, "b": b, "T": T, "k": k, **res,
            "barrier_ts_ny": window_raw[off - 1]["ts_ny"] if off else None,
            "label_end_bar_idx": end, "label_end_session_date": end_bar["session_date"],
            "label_crosses_year": end_bar["session_date"][:4] != e["session_date"][:4],
        })
    return rows, purged


def reconciliation(rows, purged):
    out = {}
    for split in SPLIT_END:
        part = [r for r in rows if r["split"] == split]
        for lab in LABELS:
            out[f"{split}_{lab}"] = sum(r["label"] == lab for r in part)
        out[f"{split}_both_same_bar"] = sum(r["both_same_bar"] for r in part)
        out[f"{split}_sum_R_rev"] = round(sum(r["R_rev"] for r in part), 4)
        out[f"{split}_sum_R_cont"] = round(sum(r["R_cont"] for r in part), 4)
        out[f"{split}_sum_barrier_offset"] = sum(r["barrier_bar_offset"] or 0 for r in part)
        out[f"{split}_purged"] = sum(p["split"] == split for p in purged)
    out["label_crosses_year"] = sum(r["label_crosses_year"] for r in rows)
    return out


def run_labels(cfg, paths):
    """Label dev/calib. Parameter dan isi data/ dicek dulu terhadap manifest kunci uji."""
    manifest = read_manifest(cfg, paths)
    if manifest is None:
        raise LockViolation(f"Manifest kunci uji tidak ada di {paths.lock}; jalankan tahap split dulu")
    check_lock_params(cfg, manifest)
    check_data_files(manifest, paths.data)
    label = cfg["label"]
    bars = common.load_model_csv(paths.data / "bars_devcalib.csv")
    events = [e for name in ("events_dev.csv", "events_calib.csv")
              for e in common.load_model_csv(paths.data / name)]
    rows, purged = label_events(events, bars, label)
    if len(rows) + len(purged) != len(events):
        raise AssertionError("jumlah label + purge tidak sama dengan jumlah event")
    for split in SPLIT_END:
        common.write_csv(paths.data / f"labels_{split}.csv",
                         [r for r in rows if r["split"] == split], LABEL_COLS)
    common.write_csv(paths.output / "label_purged.csv", purged,
                     ["event_id", "split", "session_date", "purge_reason"])

    recon = reconciliation(rows, purged)
    (paths.output / "label_rekonsiliasi.json").write_text(json.dumps(recon, indent=2) + "\n")
    counts = " ".join(f"{s}={recon[f'{s}_Reversal']}/{recon[f'{s}_Continuation']}/"
                      f"{recon[f'{s}_Netral']}" for s in SPLIT_END)
    print(f"[labels] a={label['a']:g} b={label['b']:g} T={label['T']} k={label['k']}; "
          f"Rev/Cont/Net {counts}; purge {len(purged)}")
    return recon
