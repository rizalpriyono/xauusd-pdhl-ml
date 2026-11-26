"""Tahap 1 (RM-22 sekuensial v2): 13 fitur per-candle point-in-time untuk model LSTM.

Seluruh 13 fitur diturunkan dari gabungan tiga sumber (theory-driven, tanpa seleksi otomatis):
1. Proposal Final v2 (IV.II.5, Tabel 4.5): struktur level, body, wick, EMA50, rezim ATR120, sesi, hari.
2. Kuesioner RQ1 (9 responden lolos): wick penolakan (5/9), engulfing/body (3/9), lonjakan volume (3/9),
   sesi London/NY (median 4), arah tren H1 (median 5), volatilitas ATR (median 4).
3. Selected Papers (Grądzki et al. 2025; Jankowski & Weichbroth 2026; Enkhbayar & Ślepaczuk 2025;
   Moser & Brauneis 2026; Sokolovsky et al. 2023): fitur stasioner, simetris arah, tren MA/EMA,
   rezim volatilitas, volume relatif jam-sama, dan encoding siklik jam & hari (sin/cos).

Setiap event pada `decision_bar_idx = t*` membaca jendela sepanjang L bar [t* - L + 1 .. t*].
Pengaman (identik dengan jalur Gradient Boosting):
- Guard `load_model_csv` menolak folder `kunci_uji*` dan baris > 2024-12-31 (`TestDataLeak`).
- Guard `AtrKosong` menolak bar tanpa `atr14_h1` > 0 atau event tanpa `atr14_d_prev` > 0 (v1.1.0+).
- Guard `PointInTime` membatasi akses bar, EMA50, dan rezim volatilitas hingga indeks `t*`.
- `uji_pemotongan` memotong deret bar pada 4 titik waktu, membangun ulang konteks (EMA50, rezim
  ATR120, peta volume 20 sesi) serta sekuens 3D (L = 96), lalu membuktikan 0 perbedaan.

Modul ini hanya memakai pustaka standar Python (stdlib), tanpa PyTorch maupun NumPy.
"""
import json
import math
from types import MappingProxyType
from typing import NamedTuple

from pdhl_data import common
from pdhl_data.common import SPLIT_END

from .kunci import LOOKBACKS_TERKUNCI

FEATURE_NAMES = (
    "dist_level_dir",      # s01: d * (close - level) / ATR_t*
    "body_dir",            # s02: d * (close - open) / ATR_t*
    "range_atr",           # s03: (high - low) / ATR_t*
    "wick_against_dir",    # s04: wick searah reversal / ATR_t*
    "wick_with_dir",       # s05: wick searah continuation / ATR_t*
    "dist_ema50_dir",      # s06: d * (close - EMA50) / ATR_t*
    "vol_regime",          # s07: ATR_i / mean(ATR 120 bar s.d. i)
    "log_vol_rel_hour",    # s08: ln(V_i / mean V jam NY sama 20 sesi sebelumnya)
    "ny_hour_sin",         # s09: sin(2*pi * ny_hour / 24)
    "ny_hour_cos",         # s10: cos(2*pi * ny_hour / 24)
    "weekday_sin",         # s11: sin(2*pi * weekday / 5)  [Senin=0..Jumat=4]
    "weekday_cos",         # s12: cos(2*pi * weekday / 5)
    "log_gap_h",           # s13: ln(1 + hours_since_prev)
)
CYCLIC_INDICES = (8, 9, 10, 11)     # ny_hour_{sin,cos} & weekday_{sin,cos} sudah di [-1, 1]
META_COLS = [
    "event_id", "split", "session_date", "level_type", "direction_d",
    "decision_bar_idx", "decision_ts_ny",
]
OUT_COLS = META_COLS + list(FEATURE_NAMES) + ["f13_wick_penolakan", "b1_terpenuhi"]
EMA_N = 50
VOL_REGIME_N = 120
VOL_SESSIONS = 20
VOL_MIN_BARS = 10
TOL = 1e-9
PRICE_KEYS = ("open", "high", "low", "close", "tick_volume")
CUT_DATES = ("2018-03-01", "2020-06-17", "2023-11-01")
TWO_PI_24 = 2.0 * math.pi / 24.0
TWO_PI_5 = 2.0 * math.pi / 5.0


class LookAheadError(RuntimeError):
    """Perhitungan fitur sekuensial mencoba membaca bar sesudah bar keputusan."""


class AtrKosong(ValueError):
    """ATR kosong pada bar atau event: data berasal dari pipeline sebelum adendum ATR (v1.1.0)."""


class PointInTime:
    """Akses baca berindeks ke sebuah deret, dibatasi sampai indeks `limit`."""

    __slots__ = ("_seq", "_limit")

    def __init__(self, seq, limit):
        self._seq, self._limit = seq, limit

    @property
    def limit(self):
        return self._limit

    def __getitem__(self, i):
        if not isinstance(i, int) or isinstance(i, bool):
            raise TypeError("PointInTime hanya menerima indeks bilangan bulat (tanpa slice)")
        if i < 0 or i > self._limit:
            raise LookAheadError(f"indeks {i} di luar batas point-in-time 0..{self._limit}")
        return self._seq[i]


class Konteks(NamedTuple):
    ema50: tuple            # EMA(50) close H1 per bar_idx (kausal)
    vol_regime: tuple       # ATR_i / mean(ATR_{i-119..i}) per bar_idx (kausal)
    bar_session: tuple      # urutan sesi untuk setiap bar_idx
    session_hours: tuple    # per sesi: ny_hour -> bar_idx


# ---------------------------------------------------------------- fungsi dasar
def wick_penolakan(o, h, l, c, d):
    """Rasio wick searah reversal terhadap rentang candle sentuhan (identik f13 di GB)."""
    rng = h - l
    if rng < 0 or not l <= min(o, c) or not max(o, c) <= h:
        raise ValueError(f"OHLC tidak konsisten: o={o} h={h} l={l} c={c}")
    if rng == 0:
        return 0.0
    wick = h - max(o, c) if d == 1 else min(o, c) - l
    return wick / rng


def b1_terpenuhi(f13, theta):
    """Aturan B1 Kunci-1: wick penolakan >= theta (toleransi pembulatan theta di JSON)."""
    return f13 >= theta - TOL


def compute_ema(values, n=EMA_N):
    """EMA kausal dengan alpha = 2/(n+1); benih = rata-rata n nilai pertama; sebelum itu None."""
    out = [None] * len(values)
    if len(values) < n:
        return out
    alpha = 2.0 / (n + 1)
    ema = sum(values[:n]) / n
    out[n - 1] = ema
    for i in range(n, len(values)):
        ema = alpha * values[i] + (1.0 - alpha) * ema
        out[i] = ema
    return out


def compute_vol_regime(atrs, n=VOL_REGIME_N):
    """Rezim volatilitas kausal: ATR_i / mean(ATR_{i-n+1..i}); None bila i < n - 1."""
    out = [None] * len(atrs)
    if len(atrs) < n:
        return out
    rolling = sum(atrs[:n])
    out[n - 1] = atrs[n - 1] / (rolling / n)
    for i in range(n, len(atrs)):
        rolling += atrs[i] - atrs[i - n]
        out[i] = atrs[i] / (rolling / n)
    return out


def volume_relatif_jam(view, ctx, i):
    """Volume relatif kausal: V_i / rata-rata V jam NY sama pada 20 sesi sebelumnya (identik f15)."""
    bar = view[i]
    s, h = ctx.bar_session[i], bar["ny_hour"]
    hist = [
        view[ctx.session_hours[j][h]]["tick_volume"]
        for j in range(max(0, s - VOL_SESSIONS), s)
        if h in ctx.session_hours[j]
    ]
    if len(hist) < VOL_MIN_BARS:
        return None
    mean_v = sum(hist) / len(hist)
    if mean_v <= 0 or bar["tick_volume"] <= 0:
        return None
    return bar["tick_volume"] / mean_v


def weekday_idx(session_date):
    """Indeks hari perdagangan Senin=0 .. Jumat=4 dari `session_date`."""
    wd = common.parse_date(session_date).weekday()
    if wd > 4:
        raise ValueError(f"{session_date} bukan hari perdagangan Senin-Jumat")
    return wd


# ---------------------------------------------------------------- persiapan & konteks
def _atr_bar(r):
    raw = r.get("atr14_h1", "")
    if raw == "":
        raise AtrKosong(f"bar {r['bar_idx']} ({r['ts_ny']}): atr14_h1 kosong. Data ini berasal dari "
                        "pipeline sebelum v1.1.0 (adendum ATR 2026-11-24); jalankan ulang "
                        "`python -m pdhl_data run`.")
    atr = float(raw)
    if not (math.isfinite(atr) and atr > 0):
        raise ValueError(f"bar {r['bar_idx']}: atr14_h1 harus finite dan > 0, didapat {raw!r}")
    return atr


def _gap_jam(r):
    raw = r.get("hours_since_prev", "")
    if raw in ("", None):
        return 1.0
    v = float(raw)
    if not math.isfinite(v) or v < 0:
        raise ValueError(f"bar {r['bar_idx']}: hours_since_prev tidak valid ({raw!r})")
    return v


def prepare_bars(rows):
    """Bar CSV -> list dict numerik baru (baris asal tidak diubah). Setiap bar wajib ber-ATR."""
    out = []
    for r in rows:
        nb = {
            "bar_idx": int(r["bar_idx"]),
            "ts_ny": r["ts_ny"],
            "ny_hour": int(r["ny_hour"]),
            "session_date": r["session_date"],
            "week_start": r["week_start"],
            "bar_in_session": int(r["bar_in_session"]),
            **{k: float(r[k]) for k in PRICE_KEYS},
            "atr14_h1": _atr_bar(r),
            "hours_since_prev": _gap_jam(r),
        }
        if not all(math.isfinite(nb[k]) for k in PRICE_KEYS):
            raise ValueError(f"bar {nb['bar_idx']}: OHLC/tick volume tidak finite")
        if nb["tick_volume"] < 0:
            raise ValueError(f"bar {nb['bar_idx']}: tick_volume negatif ({nb['tick_volume']})")
        out.append(nb)
    return out


def build_context(bars):
    """Bangun konteks kausal (EMA50, rezim ATR120, peta sesi x jam NY) tanpa melihat masa depan."""
    bar_session, session_hours, prev_sd = [], [], None
    for b in bars:
        if b["session_date"] != prev_sd:
            if prev_sd is not None and b["session_date"] < prev_sd:
                raise ValueError(f"bar {b['bar_idx']}: session_date tidak berurutan")
            session_hours.append({})
            prev_sd = b["session_date"]
        if b["ny_hour"] in session_hours[-1]:
            raise ValueError(f"bar {b['bar_idx']}: jam {b['ny_hour']} ganda dalam satu sesi")
        session_hours[-1][b["ny_hour"]] = b["bar_idx"]
        bar_session.append(len(session_hours) - 1)
    return Konteks(
        ema50=tuple(compute_ema([b["close"] for b in bars])),
        vol_regime=tuple(compute_vol_regime([b["atr14_h1"] for b in bars])),
        bar_session=tuple(bar_session),
        session_hours=tuple(MappingProxyType(h) for h in session_hours),
    )


# ---------------------------------------------------------------- validasi & sekuens per event
def _num(e, key):
    v = float(e[key])
    if not math.isfinite(v):
        raise ValueError(f"{e['event_id']}: {key} tidak finite")
    return v


def _same(a, b):
    return math.isclose(a, b, rel_tol=0, abs_tol=TOL)


def _atr_harian(e):
    if e.get("atr14_d_prev", "") == "":
        raise AtrKosong(f"{e['event_id']}: atr14_d_prev kosong. Data ini berasal dari pipeline "
                        "sebelum v1.1.0 (adendum ATR 2026-11-24).")
    atr_d = _num(e, "atr14_d_prev")
    if atr_d <= 0:
        raise ValueError(f"{e['event_id']}: atr14_d_prev harus > 0, didapat {atr_d}")
    return atr_d


def _cek_event(e, view, tb, db, d):
    """Event harus konsisten dengan bar sentuhan, bar keputusan, dan pembukaan sesinya."""
    eid = e["event_id"]
    for key in ("open", "high", "low", "close"):
        if not _same(_num(e, f"touch_{key}"), tb[key]):
            raise ValueError(f"{eid}: touch_{key} tidak sama dengan bar sentuhan")
    if not _same(_num(e, "entry_price"), db["close"]):
        raise ValueError(f"{eid}: entry_price tidak sama dengan close bar keputusan")
    if not _same(_num(e, "atr14_h1"), db["atr14_h1"]):
        raise ValueError(f"{eid}: atr14_h1 tidak sama dengan ATR bar keputusan")
    level = _num(e, "level_price")
    if not _same(level, _num(e, "pdh" if d == 1 else "pdl")):
        raise ValueError(f"{eid}: level_price tidak sama dengan PDH/PDL")
    if (tb["high"] < level) if d == 1 else (tb["low"] > level):
        raise ValueError(f"{eid}: candle sentuhan tidak menyentuh level")
    if tb["session_date"] != e["session_date"] or tb["bar_in_session"] != int(e["touch_bar_in_session"]):
        raise ValueError(f"{eid}: sesi/urutan bar sentuhan tidak cocok")
    first = view[tb["bar_idx"] - (tb["bar_in_session"] - 1)]
    if first["session_date"] != e["session_date"] or first["bar_in_session"] != 1:
        raise ValueError(f"{eid}: bar pertama sesi tidak ditemukan")
    if not _same(_num(e, "session_open"), first["open"]):
        raise ValueError(f"{eid}: session_open tidak sama dengan open bar pertama sesi")


def hitung_sekuens_event(event, bars, lookback, ctx=None):
    """Hitung sekuens 2D (lookback × 13 fitur) untuk satu event; hanya membaca bar <= t*.

    Mengembalikan tuple `(seq, f13)` di mana `seq` adalah list sepanjang `lookback` berisi list
    13 float (urutan `FEATURE_NAMES`), dan `f13` adalah `wick_penolakan` pada bar sentuhan.
    """
    if lookback < 1:
        raise ValueError(f"lookback harus >= 1, didapat {lookback}")
    if ctx is None:
        ctx = build_context(bars)

    e = event
    d = int(e["direction_d"])
    if (e["level_type"], d) not in (("PDH", 1), ("PDL", -1)):
        raise ValueError(f"{e['event_id']}: level_type dan direction_d tidak cocok")
    touch, dec = int(e["touch_bar_idx"]), int(e["decision_bar_idx"])
    if not 0 <= touch <= dec:
        raise ValueError(f"{e['event_id']}: touch_bar_idx harus <= decision_bar_idx")
    start = dec - lookback + 1
    if start < 0:
        raise ValueError(
            f"{e['event_id']}: decision_bar_idx={dec} tidak cukup untuk lookback={lookback}"
        )

    view = PointInTime(bars, dec)
    ema_view = PointInTime(ctx.ema50, dec)
    vr_view = PointInTime(ctx.vol_regime, dec)
    tb, db = view[touch], view[dec]
    if tb["bar_idx"] != touch or db["bar_idx"] != dec:
        raise ValueError(f"{e['event_id']}: bar_idx tidak sama dengan indeks list")
    _cek_event(e, view, tb, db, d)
    _atr_harian(e)

    level = _num(e, "level_price")
    atr_dec = db["atr14_h1"]
    if not (math.isfinite(atr_dec) and atr_dec > 0):
        raise ValueError(f"{e['event_id']}: atr14_h1 harus > 0")

    seq = []
    for i in range(start, dec + 1):
        b = view[i]
        ema_i = ema_view[i]
        vr_i = vr_view[i]
        vrel_i = volume_relatif_jam(view, ctx, i)
        if ema_i is None or vr_i is None or vrel_i is None:
            raise ValueError(
                f"{e['event_id']}: riwayat tidak cukup pada bar {i} "
                f"(ema50={ema_i}, vol_regime={vr_i}, vol_rel={vrel_i})"
            )

        o, h, lo, c = b["open"], b["high"], b["low"], b["close"]
        dist_level_dir = d * (c - level) / atr_dec
        body_dir = d * (c - o) / atr_dec
        range_atr = (h - lo) / atr_dec
        upper_wick = (h - max(o, c)) / atr_dec
        lower_wick = (min(o, c) - lo) / atr_dec
        wick_against = upper_wick if d == 1 else lower_wick
        wick_with = lower_wick if d == 1 else upper_wick
        dist_ema50_dir = d * (c - ema_i) / atr_dec
        log_vol_rel_hour = math.log(vrel_i)

        h_angle = TWO_PI_24 * b["ny_hour"]
        wd_angle = TWO_PI_5 * weekday_idx(b["session_date"])
        log_gap_h = math.log1p(b.get("hours_since_prev", 1.0))

        row = [
            dist_level_dir,
            body_dir,
            range_atr,
            wick_against,
            wick_with,
            dist_ema50_dir,
            vr_i,
            log_vol_rel_hour,
            math.sin(h_angle),
            math.cos(h_angle),
            math.sin(wd_angle),
            math.cos(wd_angle),
            log_gap_h,
        ]
        if not all(math.isfinite(v) for v in row):
            raise ValueError(f"{e['event_id']}: ditemukan nilai fitur tidak berhingga pada bar {i}")
        seq.append(row)

    f13 = wick_penolakan(tb["open"], tb["high"], tb["low"], tb["close"], d)
    return seq, f13


def cek_input_bars_events(events, bars):
    """Validasi anti-kebocoran uji, kontinuitas bar_idx, dan konsistensi split event."""
    common.assert_no_test_data(events, source="events")
    common.assert_no_test_data(bars, source="bars")
    if any(b["bar_idx"] != i for i, b in enumerate(bars)):
        raise ValueError("bar harus prefiks bar_idx yang kontinu (indeks list = bar_idx)")
    for e in events:
        if e["split"] not in SPLIT_END:
            raise common.TestDataLeak(f"{e['event_id']}: split {e['split']} tidak boleh dihitung")
        if e["split"] != common.split_of(e["session_date"]):
            raise ValueError(f"{e['event_id']}: split {e['split']} tidak sesuai tanggal sesi")


def hitung_semua_sekuens(events, bars, lookback, theta, ctx=None):
    """Hitung sekuens 3D dan baris ringkasan bar keputusan t* untuk semua event.

    Mengembalikan `(seqs, meta_rows)`:
    - `seqs`: list sepanjang N berisi matriks (lookback × 13)
    - `meta_rows`: list dict berisi META_COLS + 13 fitur pada bar keputusan t* + f13 + b1_terpenuhi
    """
    cek_input_bars_events(events, bars)
    if ctx is None:
        ctx = build_context(bars)
    seqs, meta_rows = [], []
    for e in events:
        seq, f13 = hitung_sekuens_event(e, bars, lookback, ctx=ctx)
        last = seq[-1]      # nilai 13 fitur tepat pada bar keputusan t*
        row = {
            **{c: e[c] for c in META_COLS},
            **{name: last[i] for i, name in enumerate(FEATURE_NAMES)},
            "f13_wick_penolakan": f13,
            "b1_terpenuhi": b1_terpenuhi(f13, theta),
        }
        seqs.append(seq)
        meta_rows.append(row)
    return seqs, meta_rows


def cek_tanpa_kosong(meta_rows):
    """Pastikan seluruh nilai fitur ringkasan terisi dan finite."""
    cols = (*FEATURE_NAMES, "f13_wick_penolakan")
    buruk = [(r["event_id"], c) for r in meta_rows for c in cols
             if r[c] is None or not math.isfinite(r[c])]
    if buruk:
        contoh = ", ".join(f"{eid}:{c}" for eid, c in buruk[:5])
        raise ValueError(f"{len(buruk)} nilai fitur kosong/tidak finite (mis. {contoh})")


# ---------------------------------------------------------------- uji pemotongan
def uji_pemotongan(events, bars, cuts, lookback, theta):
    """Potong deret bar di `bar[:cut+1]`; bangun ulang konteks & sekuens 3D harus 100% identik."""
    full_seqs, full_meta = hitung_semua_sekuens(events, bars, lookback, theta)
    full_by_id = {m["event_id"]: (s, m) for s, m in zip(full_seqs, full_meta)}
    out = []
    for cut in cuts:
        part = bars[:cut + 1]
        sub = [e for e in events if int(e["decision_bar_idx"]) <= cut]
        # ctx=None sengaja agar EMA50, vol_regime, & peta volume dihitung ulang dari data terpotong
        sub_seqs, sub_meta = hitung_semua_sekuens(sub, part, lookback, theta, ctx=None)
        bad = [
            m["event_id"]
            for s, m in zip(sub_seqs, sub_meta)
            if (s, m) != full_by_id[m["event_id"]]
        ]
        out.append({
            "cut_bar_idx": cut,
            "cut_ts_ny": part[-1]["ts_ny"],
            "n_event": len(sub_meta),
            "lookback_uji": lookback,
            "mismatch": len(bad),
            "mismatch_ids": bad[:10],
        })
    return out


def pilih_titik_potong(events, bars, dates=CUT_DATES):
    """Bar keputusan event pertama pada/sesudah setiap tanggal, ditambah bar terakhir."""
    ordered = sorted(events, key=lambda e: int(e["decision_bar_idx"]))
    cuts = []
    for day in dates:
        e = next((e for e in ordered if e["session_date"] >= day), None)
        if e is not None:
            cuts.append(int(e["decision_bar_idx"]))
    return cuts + [len(bars) - 1]


# ---------------------------------------------------------------- rekonsiliasi
def reconciliation(seqs, meta_rows, lookback_audit):
    out = {
        "n_fitur_sekuensial": len(FEATURE_NAMES),
        "nama_fitur_sekuensial": list(FEATURE_NAMES),
        "lookback_audit": lookback_audit,
    }
    for split in SPLIT_END:
        idx = [i for i, r in enumerate(meta_rows) if r["split"] == split]
        part = [meta_rows[i] for i in idx]
        out[f"{split}_n"] = len(part)
        out[f"{split}_event_id_sha256"] = common.sha256_ids([r["event_id"] for r in part])
        for f_i, c in enumerate(FEATURE_NAMES):
            vals_last = [r[c] for r in part]
            vals_seq = [seqs[i][t][f_i] for i in idx for t in range(lookback_audit)]
            out[f"{split}_sum_tstar_{c}"] = round(sum(vals_last), 6)
            out[f"{split}_sum_seq_{c}"] = round(sum(vals_seq), 6)
        out[f"{split}_sum_f13_wick_penolakan"] = round(sum(r["f13_wick_penolakan"] for r in part), 6)
        out[f"{split}_b1_terpenuhi"] = sum(r["b1_terpenuhi"] for r in part)
    return out


# ---------------------------------------------------------------- tahap
def run_fitur(paths, kunci, sumber, lookback_audit=None):
    """Validasi & bangun 13 fitur sekuensial dev+calib, uji pemotongan, lalu tulis keluaran."""
    lb_audit = int(lookback_audit or max(kunci.lookbacks or LOOKBACKS_TERKUNCI))
    bars = prepare_bars(common.load_model_csv(paths.input / "bars_devcalib.csv"))
    events = [e for split in SPLIT_END
              for e in common.load_model_csv(paths.input / f"events_{split}.csv")]
    seqs, meta_rows = hitung_semua_sekuens(events, bars, lb_audit, kunci.theta)
    cek_tanpa_kosong(meta_rows)
    pit = uji_pemotongan(events, bars, pilih_titik_potong(events, bars), lb_audit, kunci.theta)
    if any(r["mismatch"] for r in pit):
        raise LookAheadError("uji pemotongan GAGAL: ada fitur sekuensial yang melihat masa depan")

    recon = {
        "rm": "RM-22",
        "versi_fitur": "v2_13fitur_theory_driven",
        "k": kunci.k,
        "theta_b1": kunci.theta,
        "kandidat_lookback": list(kunci.lookbacks),
        **reconciliation(seqs, meta_rows, lb_audit),
        "guard_point_in_time_event": len(meta_rows),
        "uji_pemotongan": pit,
        **sumber,
    }
    for split in SPLIT_END:
        common.write_csv(
            paths.data / f"seq_meta_{split}.csv",
            [r for r in meta_rows if r["split"] == split],
            OUT_COLS,
        )
    paths.output.mkdir(parents=True, exist_ok=True)
    (paths.output / "fitur_rekonsiliasi.json").write_text(
        json.dumps(recon, indent=2, ensure_ascii=False) + "\n"
    )
    return recon


def ringkas(recon):
    """Baris ringkasan untuk konsol."""
    lines = [
        f"Fitur sekuensial LSTM (v2): {recon['dev_n'] + recon['calib_n']} event "
        f"(dev {recon['dev_n']}, calib {recon['calib_n']}), "
        f"{recon['n_fitur_sekuensial']} fitur/bar, audit L={recon['lookback_audit']}, "
        f"theta B1 = {recon['theta_b1']}"
    ]
    lines += [
        f"  uji pemotongan {r['cut_ts_ny']}: {r['n_event']} event (L={r['lookback_uji']}), "
        f"mismatch {r['mismatch']}"
        for r in recon["uji_pemotongan"]
    ]
    return "\n".join(lines)
