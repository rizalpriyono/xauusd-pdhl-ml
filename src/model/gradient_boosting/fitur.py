"""Tahap 1 (RM-22): 15 fitur point-in-time untuk setiap event dev dan calib.

Fitur 1-12 mengikuti Proposal Final v2 IV.II.5 (Tabel 4.5) dan merupakan port setia dari
`Pemodelan PDH-PDL/fitur.py`. Fitur 13-15 mengikuti Kunci-1 Adendum 4 tanggal 25 November 2026
(`configs/kunci_1/kunci_1_adendum_4.json`): f13 rasio wick penolakan (tetap), f14 Chaikin Money
Flow 21 bar bertanda d, dan f15 permutation entropy atas 48 log return. Aturan B1 tetap dari
`kunci_1_fitur.json`. Jarak harga dinyatakan dalam ATR(14) H1 pada bar keputusan, kecuali fitur 2
dan 12 yang memakai ATR harian sesi sebelumnya. Faktor arah d = +1 (PDH) / -1 (PDL), sehingga nilai
positif berarti "menuju atau melewati level". Fitur 15 tidak bertanda d.

Point-in-time: setiap akses ke bar dan EMA lewat `PointInTime`, yang melempar `LookAheadError`
bila indeksnya melewati bar keputusan. `uji_pemotongan` membuktikan hal yang sama secara empiris:
fitur yang dihitung dari data yang dipotong pada bar keputusan harus identik.

Guard ATR (adendum ATR 2026-11-24): setiap bar `bars_devcalib.csv` wajib memiliki `atr14_h1` dan
setiap event wajib memiliki `atr14_d_prev` > 0. Data dari pipeline sebelum v1.1.0, yang ATR awal
2015-nya kosong, langsung ditolak. Fitur dev/calib yang kosong juga dianggap galat.

Guard Kunci-1: `cek_kunci_fitur` menolak berjalan bila nama atau jendela fitur 13-15 di berkas ini
berbeda dari Adendum 4.
"""
import json
import math
from collections import Counter
from types import MappingProxyType
from typing import NamedTuple

from pdhl_data import common
from pdhl_data.common import SPLIT_END

from .kunci import ADENDUM_FITUR_JSON, FITUR_DASAR, FITUR_JSON, KunciError

BERKAS_KUNCI_FITUR = (FITUR_JSON, ADENDUM_FITUR_JSON)   # hash keduanya dicatat di rekonsiliasi
F14, F15 = "f14_cmf21_d", "f15_pe48"
FEATURE_COLS = [
    "f01_jenis_level", "f02_rentang_hari_sebelumnya", "f03_jarak_level_open_sesi", "f04_sesi",
    "f05_hari", "f06_rezim_volatilitas", "f07_momentum_pendekatan", "f08_waktu_sejak_open_sesi",
    "f09_posisi_close", "f10_penetrasi", "f11_jarak_ema50", "f12_return_minggu",
    "f13_wick_penolakan", F14, F15,
]
CATEGORICAL = ("f04_sesi", "f05_hari")
NUMERIC = [c for c in FEATURE_COLS if c not in CATEGORICAL]
META_COLS = ["event_id", "split", "session_date", "level_type", "direction_d",
             "decision_bar_idx", "decision_ts_ny"]
OUT_COLS = META_COLS + FEATURE_COLS + ["b1_terpenuhi"]
SESI = ("Asia", "London", "NewYork")
HARI = ("Senin", "Selasa", "Rabu", "Kamis", "Jumat")
EMA_N, VOL_REGIME_N, MOMENTUM_LAG = 50, 120, 6
CMF_N = 21                          # fitur 14: jendela CMF, bar H1 sampai bar keputusan
PE_N, PE_M, PE_TUNDA = 48, 3, 1     # fitur 15: jumlah log return, orde, dan tunda
TOL = 1e-9
PRICE_KEYS = ("open", "high", "low", "close", "tick_volume")
CUT_DATES = ("2018-03-01", "2020-06-17", "2023-11-01")


class LookAheadError(RuntimeError):
    """Perhitungan fitur mencoba membaca bar sesudah bar keputusan."""


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
    ema50: tuple            # EMA(50) close H1 per bar_idx
    week_first: object      # week_start -> bar_idx pertama minggu itu


# ---------------------------------------------------------------- fungsi dasar
def sesi(ny_hour):
    """Fitur 4: sesi menurut jam New York candle sentuhan."""
    if not 0 <= ny_hour <= 23:
        raise ValueError(f"jam {ny_hour} tidak valid")
    if ny_hour >= 18 or ny_hour <= 2:
        return "Asia"
    if ny_hour <= 7:
        return "London"
    if ny_hour <= 16:
        return "NewYork"
    raise ValueError("jam 17.00 NY adalah jeda harian, tidak ada candle")


def hari(session_date):
    """Fitur 5: hari perdagangan sesi (sesi berakhir pukul 17.00 NY pada session_date)."""
    wd = common.parse_date(session_date).weekday()
    if wd > 4:
        raise ValueError(f"{session_date} bukan hari perdagangan Senin-Jumat")
    return HARI[wd]


def wick_penolakan(o, h, l, c, d):
    """Fitur 13: wick searah reversal dibagi rentang candle (0 bila rentang 0)."""
    rng = h - l
    if rng < 0 or not l <= min(o, c) or not max(o, c) <= h:
        raise ValueError(f"OHLC tidak konsisten: o={o} h={h} l={l} c={c}")
    if rng == 0:
        return 0.0
    wick = h - max(o, c) if d == 1 else min(o, c) - l
    return wick / rng


def money_flow_multiplier(h, l, c):
    """MFM = ((C - L) - (H - C)) / (H - L), rentang [-1, 1]; 0 bila H = L."""
    if not l <= c <= h:
        raise ValueError(f"HLC tidak konsisten: h={h} l={l} c={c}")
    rng = h - l
    return ((c - l) - (h - c)) / rng if rng > 0 else 0.0


def chaikin_money_flow(hlcv):
    """CMF = sum(MFM_i * V_i) / sum(V_i) atas deret (high, low, close, volume).

    Bernilai 0 bila sum(V) = 0."""
    hlcv = list(hlcv)
    if not hlcv:
        raise ValueError("CMF membutuhkan sekurang-kurangnya satu bar")
    if any(v < 0 for _, _, _, v in hlcv):
        raise ValueError("volume tidak boleh negatif")
    total = sum(v for _, _, _, v in hlcv)
    if total == 0:
        return 0.0
    return sum(money_flow_multiplier(h, l, c) * v for h, l, c, v in hlcv) / total


def cmf_d(view, t, d, n=CMF_N):
    """Fitur 14: d * CMF atas n bar terakhir sampai bar t (termasuk t); None bila riwayat < n bar.

    Positif berarti tekanan menuju atau melewati level (beli di PDH, jual di PDL). "Bar" adalah
    bar H1 yang tersedia, berurutan menurut waktu; bar yang hilang tidak ditambal."""
    if t < n - 1:
        return None
    bars = [view[i] for i in range(t - n + 1, t + 1)]
    cmf = chaikin_money_flow((b["high"], b["low"], b["close"], b["tick_volume"]) for b in bars)
    return d * cmf + 0.0                 # + 0.0: hindari -0.0 saat CMF tepat nol


def pola_ordinal(w):
    """Pola ordinal jendela: indeks diurutkan menurut nilai; nilai sama menurut kemunculan."""
    return tuple(sorted(range(len(w)), key=lambda j: (w[j], j)))


def permutation_entropy(xs, m=PE_M, tunda=PE_TUNDA):
    """Permutation entropy ternormalisasi (Bandt & Pompe, 2002), rentang [0, 1].

    PE = -sum(p * ln p) / ln(m!) atas pola ordinal semua jendela (x_i, x_{i+tunda}, ...,
    x_{i+(m-1)*tunda}). 0 berarti hanya satu pola (teratur); 1 berarti semua pola sama sering."""
    xs = list(xs)
    n_pola = len(xs) - (m - 1) * tunda
    if m < 2 or tunda < 1 or n_pola < 1:
        raise ValueError(f"deret {len(xs)} nilai terlalu pendek untuk orde {m} dan tunda {tunda}")
    if not all(math.isfinite(x) for x in xs):
        raise ValueError("permutation entropy membutuhkan nilai yang finite")
    pola = Counter(pola_ordinal([xs[i + j * tunda] for j in range(m)]) for i in range(n_pola))
    entropi = sum((c / n_pola) * math.log(n_pola / c) for c in pola.values())
    return entropi / math.log(math.factorial(m))


def pe_log_return(view, t, n=PE_N):
    """Fitur 15: permutation entropy atas n log return terakhir sampai bar t (n + 1 close).

    r_i = ln(C_i / C_{i-1}). None bila riwayat kurang dari n + 1 close. Tidak bertanda d: ini fitur
    keteraturan jalur harga, bukan fitur arah."""
    if t < n:
        return None
    close = [view[i]["close"] for i in range(t - n, t + 1)]
    if any(c <= 0 for c in close):
        raise ValueError(f"close harus > 0 untuk log return (bar {t - n}..{t})")
    return permutation_entropy([math.log(b / a) for a, b in zip(close, close[1:])])


def b1_terpenuhi(f13, theta):
    """Aturan B1: wick penolakan >= theta (toleransi pembulatan theta di JSON)."""
    return f13 >= theta - TOL


def compute_ema(values, n=EMA_N):
    """EMA dengan alpha = 2/(n+1); benih = rata-rata n nilai pertama; sebelum itu None."""
    out = [None] * len(values)
    if len(values) < n:
        return out
    alpha = 2 / (n + 1)
    ema = sum(values[:n]) / n
    out[n - 1] = ema
    for i in range(n, len(values)):
        ema = alpha * values[i] + (1 - alpha) * ema
        out[i] = ema
    return out


def cek_kunci_fitur(kunci):
    """Nama dan jendela fitur 13-15 di kode harus sama dengan Kunci-1 (Adendum 4)."""
    tambahan = list(kunci.fitur_tambahan)
    per_nama = {f.get("nama"): f for f in tambahan}
    f14, f15 = per_nama.get(F14, {}), per_nama.get(F15, {})
    cek = {
        f"nama fitur tambahan harus {FEATURE_COLS[FITUR_DASAR:]}":
            [f.get("nama") for f in tambahan] == FEATURE_COLS[FITUR_DASAR:],
        f"{F14}: jendela_bar harus {CMF_N} dan bertanda d":
            f14.get("jendela_bar") == CMF_N and f14.get("bertanda_d") is True,
        f"{F15}: jendela_return/orde_m/tunda harus {PE_N}/{PE_M}/{PE_TUNDA} dan tidak bertanda d":
            (f15.get("jendela_return"), f15.get("orde_m"), f15.get("tunda"))
            == (PE_N, PE_M, PE_TUNDA) and f15.get("bertanda_d") is False,
    }
    beda = [pesan for pesan, ok in cek.items() if not ok]
    if beda:
        raise KunciError("fitur di kode berbeda dari Kunci-1 Adendum 4: " + "; ".join(beda))


# ---------------------------------------------------------------- persiapan
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


def prepare_bars(rows):
    """Bar CSV -> dict numerik baru (baris asal tidak diubah). Setiap bar wajib ber-ATR."""
    out = []
    for r in rows:
        nb = {"bar_idx": int(r["bar_idx"]), "ts_ny": r["ts_ny"], "ny_hour": int(r["ny_hour"]),
              "session_date": r["session_date"], "week_start": r["week_start"],
              "bar_in_session": int(r["bar_in_session"]),
              **{k: float(r[k]) for k in PRICE_KEYS}, "atr14_h1": _atr_bar(r)}
        if not all(math.isfinite(nb[k]) for k in PRICE_KEYS):
            raise ValueError(f"bar {nb['bar_idx']}: OHLC/tick volume tidak finite")
        out.append(nb)
    return out


def build_context(bars):
    """EMA(50) dan bar pertama tiap minggu; menolak sesi yang mundur atau jam ganda dalam sesi."""
    week_first, jam_sesi, prev_sd = {}, set(), None
    for b in bars:
        week_first.setdefault(b["week_start"], b["bar_idx"])
        if b["session_date"] != prev_sd:
            if prev_sd is not None and b["session_date"] < prev_sd:
                raise ValueError(f"bar {b['bar_idx']}: session_date tidak berurutan")
            jam_sesi, prev_sd = set(), b["session_date"]
        if b["ny_hour"] in jam_sesi:
            raise ValueError(f"bar {b['bar_idx']}: jam {b['ny_hour']} ganda dalam satu sesi")
        jam_sesi.add(b["ny_hour"])
    return Konteks(ema50=tuple(compute_ema([b["close"] for b in bars])),
                   week_first=MappingProxyType(week_first))


# ---------------------------------------------------------------- fitur per event
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
    return _num(e, "atr14_d_prev")


def _cek_event(e, view, tb, db, d):
    """Event harus cocok dengan bar sentuhan, bar keputusan, dan pembukaan sesinya."""
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


def _rezim_volatilitas(view, dec, atr):
    if dec < VOL_REGIME_N - 1:
        return None
    hist = [view[i]["atr14_h1"] for i in range(dec - VOL_REGIME_N + 1, dec + 1)]
    return atr / (sum(hist) / VOL_REGIME_N)


def hitung_fitur(event, bars, ctx, theta):
    """15 fitur satu event. Hanya membaca bar <= bar keputusan; mengembalikan dict baru."""
    e = event
    d = int(e["direction_d"])
    if (e["level_type"], d) not in (("PDH", 1), ("PDL", -1)):
        raise ValueError(f"{e['event_id']}: level_type dan direction_d tidak cocok")
    touch, dec = int(e["touch_bar_idx"]), int(e["decision_bar_idx"])
    if not 0 <= touch <= dec:
        raise ValueError(f"{e['event_id']}: touch_bar_idx harus <= decision_bar_idx")
    view, ema = PointInTime(bars, dec), PointInTime(ctx.ema50, dec)
    tb, db = view[touch], view[dec]
    if tb["bar_idx"] != touch or db["bar_idx"] != dec:
        raise ValueError(f"{e['event_id']}: bar_idx tidak sama dengan indeks list")
    _cek_event(e, view, tb, db, d)

    level, atr, atr_d = _num(e, "level_price"), db["atr14_h1"], _atr_harian(e)
    if not (atr > 0 and atr_d > 0):
        raise ValueError(f"{e['event_id']}: ATR harus > 0")
    close = db["close"]
    extreme = tb["high"] if d == 1 else tb["low"]
    ema_dec = ema[dec]
    week_open = view[ctx.week_first[db["week_start"]]]["open"]
    f13 = wick_penolakan(tb["open"], tb["high"], tb["low"], tb["close"], d)
    return {
        **{c: e[c] for c in META_COLS},
        "f01_jenis_level": 1 if d == 1 else 0,
        "f02_rentang_hari_sebelumnya": (_num(e, "pdh") - _num(e, "pdl")) / atr_d,
        "f03_jarak_level_open_sesi": d * (level - _num(e, "session_open")) / atr,
        "f04_sesi": sesi(tb["ny_hour"]),
        "f05_hari": hari(e["session_date"]),
        "f06_rezim_volatilitas": _rezim_volatilitas(view, dec, atr),
        "f07_momentum_pendekatan": (d * (close - view[dec - MOMENTUM_LAG]["close"]) / atr
                                    if dec >= MOMENTUM_LAG else None),
        "f08_waktu_sejak_open_sesi": tb["bar_in_session"],
        "f09_posisi_close": d * (tb["close"] - level) / atr,
        "f10_penetrasi": d * (extreme - level) / atr,
        "f11_jarak_ema50": d * (close - ema_dec) / atr if ema_dec is not None else None,
        "f12_return_minggu": d * (close - week_open) / atr_d,
        "f13_wick_penolakan": f13,
        F14: cmf_d(view, dec, d),
        F15: pe_log_return(view, dec),
        "b1_terpenuhi": b1_terpenuhi(f13, theta),
    }


def hitung_semua(events, bars, theta):
    common.assert_no_test_data(events, source="events")
    common.assert_no_test_data(bars, source="bars")
    if any(b["bar_idx"] != i for i, b in enumerate(bars)):
        raise ValueError("bar harus prefiks bar_idx yang kontinu (indeks list = bar_idx)")
    for e in events:
        if e["split"] not in SPLIT_END:
            raise common.TestDataLeak(f"{e['event_id']}: split {e['split']} tidak boleh dihitung")
        if e["split"] != common.split_of(e["session_date"]):
            raise ValueError(f"{e['event_id']}: split {e['split']} tidak sesuai tanggal sesi")
    ctx = build_context(bars)
    return [hitung_fitur(e, bars, ctx, theta) for e in events]


def cek_tanpa_kosong(rows):
    """Fitur dev/calib tidak boleh kosong: riwayat 2015 cukup untuk semua fitur sejak 2016."""
    kosong = [(r["event_id"], c) for r in rows for c in FEATURE_COLS if r[c] is None]
    if kosong:
        contoh = ", ".join(f"{eid}:{c}" for eid, c in kosong[:5])
        raise ValueError(f"{len(kosong)} nilai fitur kosong (mis. {contoh}); periksa riwayat bar "
                         "dan ATR di data masukan")


# ---------------------------------------------------------------- uji pemotongan
def uji_pemotongan(events, bars, cuts, theta):
    """Hitung ulang fitur dari bar[:cut+1]; event dengan keputusan <= cut harus identik."""
    full = {r["event_id"]: r for r in hitung_semua(events, bars, theta)}
    out = []
    for cut in cuts:
        part = bars[:cut + 1]
        sub = [e for e in events if int(e["decision_bar_idx"]) <= cut]
        rows = hitung_semua(sub, part, theta)
        bad = [r["event_id"] for r in rows if r != full[r["event_id"]]]
        out.append({"cut_bar_idx": cut, "cut_ts_ny": part[-1]["ts_ny"], "n_event": len(rows),
                    "mismatch": len(bad), "mismatch_ids": bad[:10]})
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
def reconciliation(rows):
    out = {}
    for split in SPLIT_END:
        part = [r for r in rows if r["split"] == split]
        out[f"{split}_n"] = len(part)
        out[f"{split}_event_id_sha256"] = common.sha256_ids([r["event_id"] for r in part])
        for c in NUMERIC:
            vals = [r[c] for r in part if r[c] is not None]
            out[f"{split}_sum_{c}"] = round(sum(vals), 6)
            out[f"{split}_null_{c}"] = len(part) - len(vals)
        for c in CATEGORICAL:
            for v in sorted({r[c] for r in part}):
                out[f"{split}_{c}_{v}"] = sum(r[c] == v for r in part)
        out[f"{split}_b1_terpenuhi"] = sum(r["b1_terpenuhi"] for r in part)
    return out


# ---------------------------------------------------------------- tahap
def run_fitur(paths, kunci, sumber):
    """Hitung fitur dev+calib, uji pemotongan, lalu tulis CSV dan rekonsiliasi.

    `sumber` = hasil `jalur.info_sumber`. Semua berkas ditulis sesudah seluruh perhitungan dan
    pemeriksaan lolos. Mengembalikan isi rekonsiliasi."""
    cek_kunci_fitur(kunci)
    bars = prepare_bars(common.load_model_csv(paths.input / "bars_devcalib.csv"))
    events = [e for split in SPLIT_END
              for e in common.load_model_csv(paths.input / f"events_{split}.csv")]
    rows = hitung_semua(events, bars, kunci.theta)
    cek_tanpa_kosong(rows)
    pit = uji_pemotongan(events, bars, pilih_titik_potong(events, bars), kunci.theta)
    if any(r["mismatch"] for r in pit):
        raise LookAheadError("uji pemotongan GAGAL: ada fitur yang memakai data sesudah keputusan")

    recon = {"k": kunci.k, "theta_b1": kunci.theta, **reconciliation(rows),
             "guard_point_in_time_event": len(rows), "uji_pemotongan": pit,
             "kunci1_sha256_sesuai_manifest": {n: kunci.manifest_sha256[n]
                                               for n in BERKAS_KUNCI_FITUR},
             **sumber}
    for split in SPLIT_END:
        common.write_csv(paths.data / f"features_{split}.csv",
                         [r for r in rows if r["split"] == split], OUT_COLS)
    paths.output.mkdir(parents=True, exist_ok=True)
    (paths.output / "fitur_rekonsiliasi.json").write_text(
        json.dumps(recon, indent=2, ensure_ascii=False) + "\n")
    return recon


def ringkas(recon):
    """Baris ringkasan untuk konsol."""
    lines = [f"Fitur: {recon['dev_n'] + recon['calib_n']} event (dev {recon['dev_n']}, "
             f"calib {recon['calib_n']}), {len(FEATURE_COLS)} fitur, theta B1 = {recon['theta_b1']}"]
    lines += [f"  uji pemotongan {r['cut_ts_ny']}: {r['n_event']} event, mismatch {r['mismatch']}"
              for r in recon["uji_pemotongan"]]
    return "\n".join(lines)
