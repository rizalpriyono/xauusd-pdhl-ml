"""Tahap 3 (RM-24): validasi walk-forward LightGBM dan pemilihan konfigurasi.

Port setia dari `Pemodelan PDH-PDL/walk_forward.py`. Delapan konfigurasi LightGBM (grid Kunci-1,
`kunci_1_rq2.json` -> `teknis.lightgbm`) dilatih pada lima lipatan walk-forward berjangkar: latih
2016..y-1 (dengan purging), validasi tahun y, untuk y = 2019..2023 (Proposal v2 IV.II.4).
Konfigurasi dengan rata-rata log loss out-of-fold (OOF) terkecil dipilih; selisih <= toleransi
dianggap seri dan dimenangkan konfigurasi yang lebih sederhana (pohon lebih sedikit, lalu kedalaman
lebih kecil, lalu min_child_samples lebih besar).

Regresi logistik tidak lagi menjadi kandidat (keputusan 22 Nov 2026, menunggu Adendum 3), sehingga
LightGBM menjadi model akhir berdasarkan rancangan. Probabilitas walk-forward tidak dikalibrasi.
Prediksi OOF model terpilih dibandingkan secara deskriptif dengan OOF pembanding B0/B1 (tahap 2).
Hanya data dev yang dimuat. Butuh lightgbm dan numpy (`pip install -e '.[model]'`).
"""
import datetime as dt
import itertools
import json
import math
import platform

from pdhl_data import common
from pdhl_data.common import CLASSES

from . import pembanding
from .fitur import CATEGORICAL, FEATURE_COLS, HARI, SESI

if CLASSES != ("Reversal", "Continuation", "Netral"):
    raise RuntimeError("kolom p_rev/p_cont/p_net mengandaikan urutan kelas Kunci-1")

TAHUN_VALIDASI = pembanding.TAHUN_VALIDASI
# Grid dan parameter teknis yang dikunci (Proposal v2 IV.II.6; kunci_1_rq2.json → teknis).
GRID_TERKUNCI = {"max_depth": [2, 3], "n_estimators": [100, 300], "min_child_samples": [20, 50]}
TEKNIS_TERKUNCI = {"objective": "multiclass", "learning_rate": 0.05, "deterministic": True,
                   "num_threads": 1}
SEED_MODEL = 20261119
KATEGORI = {"f04_sesi": SESI, "f05_hari": HARI}
NUMERIK = [c for c in FEATURE_COLS if c not in CATEGORICAL]
OOF_COLS = ["event_id", "session_date", "tahun_validasi", "label", "konfigurasi",
            "p_rev", "p_cont", "p_net", "ll"]


# ---------------------------------------------------------------- fitur
def kolom_fitur():
    """Nama kolom matriks fitur: urutan FEATURE_COLS dengan f04/f05 dipecah one-hot."""
    cols = []
    for c in FEATURE_COLS:
        cols += [f"{c}={v}" for v in KATEGORI[c]] if c in KATEGORI else [c]
    return cols


def encode(fitur_row):
    """Satu baris fitur → vektor angka (kategori tetap, jadi tidak bergantung data latih)."""
    out = []
    for c in FEATURE_COLS:
        v = fitur_row[c]
        if c in KATEGORI:
            if v not in KATEGORI[c]:
                raise ValueError(f"{c}: kategori tidak dikenal {v!r}")
            out += [1.0 if v == k else 0.0 for k in KATEGORI[c]]
        else:
            x = float(v)
            if not math.isfinite(x):
                raise ValueError(f"{c}: nilai tidak berhingga {v!r}")
            out.append(x)
    return out


def muat_data(fitur_path, label_path, theta):
    """Muat fitur dan label dev (loader ber-guard), validasi lewat `pembanding.gabung`."""
    fitur_rows = common.load_model_csv(fitur_path)
    rows = pembanding.gabung(fitur_rows, common.load_model_csv(label_path), theta)
    return [{**r, "x": encode(f)} for r, f in zip(rows, fitur_rows)]


# ---------------------------------------------------------------- konfigurasi
def konfigurasi_grid(teknis):
    """Delapan konfigurasi; menolak berkas kunci yang menyimpang dari grid/parameter terkunci."""
    lg = teknis["lightgbm"]
    beda = [k for k, v in {**GRID_TERKUNCI, **TEKNIS_TERKUNCI}.items() if lg.get(k) != v]
    if teknis.get("seed_model") != SEED_MODEL:
        beda.append("seed_model")
    if beda:
        raise ValueError(f"teknis menyimpang dari Kunci-1: {', '.join(beda)}")
    return [{"id": f"d{d}_n{n}_m{m}", "max_depth": d, "n_estimators": n, "min_child_samples": m}
            for d, n, m in itertools.product(lg["max_depth"], lg["n_estimators"],
                                             lg["min_child_samples"])]


def params_lgbm(cfg, teknis):
    """Parameter LightGBM terkunci; parameter lain bawaan. Jumlah pohon = num_boost_round."""
    lg = teknis["lightgbm"]
    return {"objective": lg["objective"], "num_class": len(CLASSES),
            "learning_rate": lg["learning_rate"], "max_depth": cfg["max_depth"],
            "min_child_samples": cfg["min_child_samples"], "seed": teknis["seed_model"],
            "deterministic": lg["deterministic"], "num_threads": lg["num_threads"],
            "verbose": -1}


def kesederhanaan(cfg):
    """Kunci urut: makin kecil makin sederhana."""
    return (cfg["n_estimators"], cfg["max_depth"], -cfg["min_child_samples"])


def himpunan_seri(skor, cfgs, tol):
    """Konfigurasi yang selisihnya dengan skor terbaik ≤ tol (termasuk yang terbaik)."""
    terbaik = min(skor[c["id"]] for c in cfgs)
    return [c for c in cfgs if skor[c["id"]] - terbaik <= tol]


def pilih_konfigurasi(skor, cfgs, tol):
    return min(himpunan_seri(skor, cfgs, tol), key=kesederhanaan)


def ringkasan_pemilihan(skor, cfgs, tol):
    """Peringkat, pilihan, himpunan seri, dan konfigurasi terdekat untuk laporan."""
    terpilih = pilih_konfigurasi(skor, cfgs, tol)
    urut = sorted(cfgs, key=lambda c: (skor[c["id"]], kesederhanaan(c)))
    lain = [c for c in urut if c["id"] != terpilih["id"]]
    return {"terpilih": terpilih,
            "peringkat": {c["id"]: i for i, c in enumerate(urut, 1)},
            "seri": [c["id"] for c in himpunan_seri(skor, cfgs, tol)],
            "terdekat": lain[0]["id"] if lain else None,
            "selisih_ke_terdekat": skor[lain[0]["id"]] - skor[terpilih["id"]] if lain else None}


# ---------------------------------------------------------------- LightGBM
def _impor_lgbm():
    try:
        import lightgbm
        import numpy
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(f"{exc.name} belum terpasang. Pasang dependensi pemodelan: "
                                  "`pip install -e '.[model]'` atau "
                                  "`pip install -r requirements-model.txt`") from exc
    return lightgbm, numpy


def latih_lgbm(X, y, cfg, teknis):
    lgb, np = _impor_lgbm()
    ds = lgb.Dataset(np.asarray(X, dtype=float), label=np.asarray(y, dtype=int))
    return lgb.train(params_lgbm(cfg, teknis), ds, num_boost_round=cfg["n_estimators"])


def prediksi_lgbm(model, X):
    _, np = _impor_lgbm()
    return [[float(v) for v in p] for p in model.predict(np.asarray(X, dtype=float))]


# ---------------------------------------------------------------- walk-forward
def _rerata(xs):
    return sum(xs) / len(xs) if xs else None


def walk_forward_model(rows, cfgs, teknis, tahun, clip, latih=latih_lgbm, prediksi=prediksi_lgbm):
    """Prediksi OOF setiap konfigurasi pada setiap lipatan; data latih memakai purging."""
    oof, lipatan = [], []
    for y in tahun:
        data, purged = pembanding.data_latih(rows, dt.date(y, 1, 1))
        if not data:
            raise ValueError(f"lipatan {y}: tidak ada data latih")
        val = [r for r in rows if int(r["session_date"][:4]) == y]
        if not val:
            raise ValueError(f"lipatan {y}: tidak ada event validasi")
        X, Y = [r["x"] for r in data], [CLASSES.index(r["label"]) for r in data]
        for cfg in cfgs:
            model = latih(X, Y, cfg, teknis)
            preds = prediksi(model, [r["x"] for r in val])
            if len(preds) != len(val):
                raise ValueError(f"lipatan {y}: {len(preds)} prediksi untuk {len(val)} event")
            lls = []
            for r, p in zip(val, preds):
                if len(p) != len(CLASSES) or abs(sum(p) - 1.0) > 1e-6 or min(p) < 0:
                    raise ValueError(f"{r['event_id']}: peluang tidak valid {p}")
                ll = pembanding.log_loss(p, r["label"], clip)
                oof.append({"event_id": r["event_id"], "session_date": r["session_date"],
                            "tahun_validasi": y, "label": r["label"], "konfigurasi": cfg["id"],
                            "p_rev": p[0], "p_cont": p[1], "p_net": p[2], "ll": ll})
                lls.append(ll)
            lipatan.append({"tahun_validasi": y, "konfigurasi": cfg["id"], "n_latih": len(data),
                            "n_purged": len(purged), "n_validasi": len(val),
                            "log_loss": _rerata(lls)})
    return oof, lipatan


def skor_konfigurasi(oof, cfgs):
    """Rata-rata log loss OOF per event (semua tahun validasi digabung) untuk setiap konfigurasi."""
    return {c["id"]: _rerata([r["ll"] for r in oof if r["konfigurasi"] == c["id"]]) for c in cfgs}


def banding_pembanding(oof_terpilih, pembanding_rows):
    """Selisih log loss OOF model − B0/B1 per tahun dan gabungan (deskriptif)."""
    pb = {r["event_id"]: r for r in pembanding_rows}
    if {r["event_id"] for r in oof_terpilih} != set(pb) or len(pb) != len(oof_terpilih):
        raise ValueError("event OOF model dan pembanding tidak sama")
    for r in oof_terpilih:
        b = pb[r["event_id"]]
        if ("label" in b and b["label"] != r["label"]) or \
                ("tahun_validasi" in b and int(b["tahun_validasi"]) != r["tahun_validasi"]):
            raise ValueError(f"{r['event_id']}: label/tahun berbeda dengan berkas pembanding")

    def ringkas(rs):
        m = [r["ll"] for r in rs]
        b0 = [float(pb[r["event_id"]]["ll_b0"]) for r in rs]
        b1 = [float(pb[r["event_id"]]["ll_b1"]) for r in rs]
        return {"n": len(rs), "ll_model": _rerata(m), "ll_b0": _rerata(b0), "ll_b1": _rerata(b1),
                "D_b0": _rerata([a - b for a, b in zip(m, b0)]),
                "D_b1": _rerata([a - b for a, b in zip(m, b1)])}

    tahun = sorted({r["tahun_validasi"] for r in oof_terpilih})
    return {"per_tahun": [{"tahun": t, **ringkas([r for r in oof_terpilih
                                                  if r["tahun_validasi"] == t])} for t in tahun],
            "gabungan": ringkas(oof_terpilih)}


# ---------------------------------------------------------------- laporan
def _f(x):
    return "–" if x is None else f"{x:.4f}"


def _d(x):
    if x is None:
        return "–"
    x = round(x, 4)
    return f"{x + 0.0:+.4f}" if x else "0.0000"


def laporan(recon):
    cfgs, skor, per = recon["konfigurasi"], recon["skor"], recon["per_tahun"]
    pilih, peringkat = recon["terpilih"]["id"], recon["peringkat"]
    tahun = recon["tahun_validasi"]
    env = recon["lingkungan"]
    lines = [
        "# Laporan Validasi Walk-Forward LightGBM (RM-24)", "",
        "Sumber: `model.gradient_boosting.walk_forward`, data pengembangan saja. Lingkungan: Python "
        f"{env['python']}, LightGBM {env['lightgbm']}, NumPy {env['numpy']} ({env['platform']}).",
        "", "Probabilitas walk-forward **tidak dikalibrasi** (kalibrasi hanya memakai data 2024). "
        f"Catatan: {recon['catatan']}", "",
        "## 1. Log loss OOF per konfigurasi", "",
        "Rata-rata log loss per event (makin kecil makin baik). Gabungan = rata-rata seluruh "
        f"{recon['oof_n_per_konfigurasi']} event validasi {tahun[0]}–{tahun[-1]}, dasar "
        "pemilihan.", "",
        "| # | Konfigurasi | max_depth | n_estimators | min_child_samples | "
        + " | ".join(str(t) for t in tahun) + " | Gabungan | Peringkat |",
        "|" + "---|" * (7 + len(tahun)),
    ]
    for i, c in enumerate(cfgs, 1):
        tanda = "**" if c["id"] == pilih else ""
        lines.append(f"| {i} | {tanda}{c['id']}{tanda} | {c['max_depth']} | {c['n_estimators']} | "
                     f"{c['min_child_samples']} | "
                     + " | ".join(_f(per[c["id"]].get(str(t))) for t in tahun)
                     + f" | {tanda}{_f(skor[c['id']])}{tanda} | {peringkat[c['id']]} |")
    p = recon["pemilihan"]
    lines += ["", "## 2. Konfigurasi terpilih", "",
              f"**{pilih}** (max_depth {recon['terpilih']['max_depth']}, n_estimators "
              f"{recon['terpilih']['n_estimators']}, min_child_samples "
              f"{recon['terpilih']['min_child_samples']}), log loss OOF {_f(skor[pilih])}.", "",
              f"- Konfigurasi terdekat ({p['terdekat'] or '–'}): selisih log loss "
              f"{_d(p['selisih_ke_terdekat'])} terhadap konfigurasi terpilih.",
              f"- Himpunan seri dengan skor terbaik (selisih ≤ {recon['toleransi_seri']}): "
              f"{', '.join(p['seri'])}"
              + (" (hanya konfigurasi terpilih, tidak ada seri)." if len(p["seri"]) == 1
                 else "; aturan kesederhanaan dipakai."), "",
              "## 3. Kestabilan terhadap pembanding (deskriptif, bukan bukti keunggulan)", "",
              "Selisih log loss OOF model terpilih − pembanding (B0/B1 diestimasi ulang per "
              "lipatan, RM-23). Negatif = model lebih baik.", "",
              "| Tahun | n | LightGBM | B0 | B1 | Model − B0 | Model − B1 |",
              "|---|---|---|---|---|---|---|"]
    b = recon["banding_pembanding"]
    for t in b["per_tahun"]:
        lines.append(f"| {t['tahun']} | {t['n']} | {_f(t['ll_model'])} | {_f(t['ll_b0'])} | "
                     f"{_f(t['ll_b1'])} | {_d(t['D_b0'])} | {_d(t['D_b1'])} |")
    g = b["gabungan"]
    lines += [f"| **Gabungan** | {g['n']} | {_f(g['ll_model'])} | {_f(g['ll_b0'])} | "
              f"{_f(g['ll_b1'])} | {_d(g['D_b0'])} | {_d(g['D_b1'])} |", ""]
    return "\n".join(lines)


def _lingkungan():
    """Versi Python dan pustaka (None bila pustaka tidak terpasang, mis. tes dengan pelatih palsu)."""
    def versi(nama):
        try:
            return __import__(nama).__version__
        except ModuleNotFoundError:
            return None
    return {"python": platform.python_version(), "lightgbm": versi("lightgbm"),
            "numpy": versi("numpy"), "platform": platform.platform()}


# ---------------------------------------------------------------- tahap
CATATAN = ("regresi logistik tidak lagi menjadi kandidat (keputusan 22 Nov 2026, menunggu "
           "Adendum 3); LightGBM menjadi model akhir berdasarkan rancangan.")


def cek_pembanding_segar(paths, fp, lp):
    """OOF pembanding harus dihitung dari features_dev.csv dan labels_dev.csv yang sama."""
    rp = paths.output / "pembanding_rekonsiliasi.json"
    dicatat = json.loads(rp.read_text()).get("sumber_sha256", {}) if rp.exists() else {}
    if any(dicatat.get(p.name) != common.sha256_file(p) for p in (fp, lp)):
        raise ValueError(f"{paths.data / 'pembanding_oof_dev.csv'} tidak dihitung dari fitur/label "
                         "yang sekarang; jalankan ulang tahap `pembanding`")


def run_walk_forward(paths, kunci, sumber, cfgs=None, tahun_validasi=TAHUN_VALIDASI,
                     latih=latih_lgbm, prediksi=prediksi_lgbm):
    """Walk-forward semua konfigurasi, pilih konfigurasi, bandingkan dengan B0/B1, tulis keluaran.

    `latih`/`prediksi` dapat diganti pelatih palsu untuk tes. Mengembalikan rekonsiliasi."""
    teknis, clip, tol = kunci.teknis, kunci.clip, kunci.toleransi_seri
    fp, lp = paths.data / "features_dev.csv", paths.input / "labels_dev.csv"
    pp = paths.data / "pembanding_oof_dev.csv"
    for p, tahap in ((fp, "fitur"), (pp, "pembanding")):
        if not p.exists():
            raise FileNotFoundError(f"{p} belum ada; jalankan tahap `{tahap}` dulu")
    cek_pembanding_segar(paths, fp, lp)
    rows = muat_data(fp, lp, kunci.theta)
    cfgs = cfgs or konfigurasi_grid(teknis)

    oof, lipatan = walk_forward_model(rows, cfgs, teknis, tahun_validasi, clip,
                                      latih=latih, prediksi=prediksi)
    skor = skor_konfigurasi(oof, cfgs)
    pilihan = ringkasan_pemilihan(skor, cfgs, tol)
    terpilih, peringkat = pilihan["terpilih"], pilihan["peringkat"]
    banding = banding_pembanding([r for r in oof if r["konfigurasi"] == terpilih["id"]],
                                 common.load_model_csv(pp))

    recon = {
        "rm": "RM-24", "kelas": list(CLASSES), "kolom_fitur": kolom_fitur(),
        "lingkungan": _lingkungan(),
        "params_tetap": {k: v for k, v in params_lgbm(cfgs[0], teknis).items()
                         if k not in ("max_depth", "min_child_samples")},
        "clip": clip, "toleransi_seri": tol, "tahun_validasi": list(tahun_validasi),
        "sumber_sha256": {p.name: common.sha256_file(p) for p in (fp, lp, pp)},
        "catatan": CATATAN,
        "konfigurasi": cfgs, "skor": skor, "peringkat": peringkat,
        "per_tahun": {c["id"]: {str(f["tahun_validasi"]): f["log_loss"] for f in lipatan
                                if f["konfigurasi"] == c["id"]} for c in cfgs},
        "lipatan": [{k: f[k] for k in ("tahun_validasi", "n_latih", "n_purged", "n_validasi")}
                    for f in lipatan if f["konfigurasi"] == cfgs[0]["id"]],
        "terpilih": terpilih,
        "pemilihan": {k: pilihan[k] for k in ("seri", "terdekat", "selisih_ke_terdekat")},
        "oof_n_per_konfigurasi": sum(r["konfigurasi"] == cfgs[0]["id"] for r in oof),
        "banding_pembanding": banding,
        "versi_pdhl_data": sumber.get("versi_pdhl_data"), "commit": sumber.get("commit"),
        "sumber_pipeline_sha256": sumber.get("sumber_sha256"),
    }
    teks = laporan(recon)                     # semua isi siap sebelum ada berkas yang ditulis
    common.write_csv(paths.data / "walk_forward_oof_dev.csv", oof, OOF_COLS)
    paths.output.mkdir(parents=True, exist_ok=True)
    (paths.output / "walk_forward_rekonsiliasi.json").write_text(
        json.dumps(recon, indent=2, ensure_ascii=False) + "\n")
    (paths.output / "laporan_walk_forward.md").write_text(teks)
    return recon


def ringkas(recon):
    """Baris ringkasan untuk konsol."""
    skor, peringkat = recon["skor"], recon["peringkat"]
    lines = [f"Walk-forward: {len(recon['konfigurasi'])} konfigurasi × "
             f"{len(recon['tahun_validasi'])} lipatan, {recon['oof_n_per_konfigurasi']} event OOF "
             "per konfigurasi"]
    for c in sorted(recon["konfigurasi"], key=lambda c: peringkat[c["id"]]):
        lines.append(f"  {peringkat[c['id']]}. {c['id']:<12} log loss OOF {skor[c['id']]:.4f}")
    g = recon["banding_pembanding"]["gabungan"]
    lines.append(f"Terpilih: {recon['terpilih']['id']} | model − B0 {g['D_b0']:+.4f}, "
                 f"model − B1 {g['D_b1']:+.4f}")
    return "\n".join(lines)
