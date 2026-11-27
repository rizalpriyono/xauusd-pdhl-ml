"""Tahap 4 (RM-25 s.d. RM-27): model akhir, kalibrasi temperature scaling, parameter keputusan.

Port setia dari `Pemodelan PDH-PDL/model_akhir.py`.

1. Konfigurasi terpilih walk-forward (RM-24, `walk_forward_rekonsiliasi.json`) dilatih sekali pada
   seluruh data pengembangan 2016–2023 (purging di batas 2024-01-01) dengan parameter terkunci
   (Proposal v2 IV.II.6; `kunci_1_rq2.json` -> teknis). Model disimpan sebagai teks LightGBM beserta
   SHA-256-nya untuk Kunci-2.
2. Temperature scaling (Guo dkk., 2017): p = softmax(z / τ) atas skor mentah LightGBM. τ
   meminimalkan log loss data kalibrasi 2024, dicari dengan golden-section pada rentang terkunci
   (RENCANA_ANALISIS_RQ2.md §5). Kalibrasi dilaporkan sebelum/sesudah τ: log loss, Brier, ECE per
   kelas, dan diagram reliabilitas (§2.3). B0/B1 pada 2024 ditampilkan sebagai acuan.
3. m_rev dan m_cont = rata-rata R posisi reversal dan continuation pada event Netral dev
   (Proposal v2 IV.II.8). δ diambil dari kunci. Pratinjau aturan keputusan pada 2024 bersifat
   deskriptif (τ diestimasi pada data yang sama) dan tidak dipakai untuk memilih apa pun.

Hanya data dev dan calib yang dimuat (lewat `load_model_csv`); data uji tidak disentuh. Butuh
lightgbm dan numpy (`pip install -e '.[model]'`). Laporan dan gambar ada di `laporan_model_akhir`.
"""
import datetime as dt
import json
import math

from pdhl_data import common
from pdhl_data.common import CALIB_END, CLASSES, DEV_END, RESEARCH_START

from . import kunci as kunci_mod
from . import pembanding
from . import walk_forward as wf
from .laporan_model_akhir import gambar_reliabilitas, laporan

LEVEL = ("PDH", "PDL")
RENTANG_SPLIT = {"dev": (RESEARCH_START, DEV_END),
                 "calib": (DEV_END + dt.timedelta(days=1), CALIB_END)}
BATAS_DEV = DEV_END + dt.timedelta(days=1)
BERKAS_KUNCI1 = (kunci_mod.LABEL_JSON, kunci_mod.FITUR_JSON, kunci_mod.ADENDUM_FITUR_JSON,
                 kunci_mod.RQ2_JSON)
TOL_TAU = 1e-8                      # toleransi lebar selang golden-section (implementasi)
N_GRID_CEK = 400                    # grid log-spasi untuk memastikan minimum global τ
TOL_R = 1e-8                        # R di CSV dibulatkan 10 desimal
TOL_PREDIKSI = 1e-9                 # softmax(skor mentah) vs predict LightGBM
ABSTAIN = "Abstain"
PHI = (math.sqrt(5) - 1) / 2
SINGKAT = pembanding.SINGKAT
MODEL_BERKAS = "model_akhir_lgbm.txt"
CALIB_COLS = (["event_id", "session_date", "label"]
              + [f"z_{SINGKAT[k]}" for k in CLASSES]
              + [f"p_{SINGKAT[k]}_mentah" for k in CLASSES] + [f"p_{SINGKAT[k]}" for k in CLASSES]
              + ["ll_mentah", "ll", "E_rev", "E_cont", "rekomendasi", "R_rev", "R_cont", "R_kotor"])


# ---------------------------------------------------------------- data dan konfigurasi
def cek_rentang(rows, split):
    """Semua session_date harus berada di dalam rentang split (dev 2016–2023, calib 2024)."""
    if not rows:
        raise ValueError(f"split {split}: tidak ada event")
    awal, akhir = RENTANG_SPLIT[split]
    for r in rows:
        if not awal <= common.parse_date(r["session_date"]) <= akhir:
            raise ValueError(f"{r['event_id']}: session_date {r['session_date']} di luar {split}")


def muat_split(fitur_path, label_path, split, theta):
    """Fitur + label satu split lewat loader ber-guard; tambah vektor fitur dan R per event."""
    fitur_rows = common.load_model_csv(fitur_path)
    label_rows = common.load_model_csv(label_path)
    rows = pembanding.gabung(fitur_rows, label_rows, theta)
    out = []
    for r, f, lab in zip(rows, fitur_rows, label_rows):
        r_rev, r_cont = float(lab["R_rev"]), float(lab["R_cont"])
        if not (math.isfinite(r_rev) and math.isfinite(r_cont)):
            raise ValueError(f"{r['event_id']}: R tidak berhingga")
        if f["level_type"] not in LEVEL:
            raise ValueError(f"{r['event_id']}: level_type tidak dikenal {f['level_type']!r}")
        out.append({**r, "level": f["level_type"], "x": wf.encode(f), "R_rev": r_rev,
                    "R_cont": r_cont})
    cek_rentang(out, split)
    return out


def konfigurasi_terpilih(recon_wf, teknis, sha_dev):
    """Konfigurasi pemenang RM-24: anggota grid, konsisten dengan skornya, data dev tetap."""
    cfg, grid = recon_wf["terpilih"], wf.konfigurasi_grid(teknis)
    if cfg not in grid:
        raise ValueError(f"konfigurasi {cfg} bukan anggota grid Kunci-1")
    if "skor" in recon_wf:
        ulang = wf.pilih_konfigurasi(recon_wf["skor"], grid,
                                     float(teknis["toleransi_seri_logloss"]))
        if ulang != cfg:
            raise ValueError(f"skor walk-forward memilih {ulang['id']}, bukan {cfg['id']}")
    lama = recon_wf["sumber_sha256"]
    beda = [n for n, h in sha_dev.items() if lama.get(n) != h]
    if beda:
        raise ValueError(f"data dev berubah sejak walk-forward: {', '.join(beda)}; jalankan ulang "
                         "tahap `walk-forward`")
    return cfg


# ---------------------------------------------------------------- LightGBM
def latih_lgbm(X, y, cfg, teknis, nama_fitur):
    """Sama dengan `walk_forward.latih_lgbm`, ditambah nama fitur di berkas model."""
    lgb, np = wf._impor_lgbm()
    ds = lgb.Dataset(np.asarray(X, dtype=float), label=np.asarray(y, dtype=int),
                     feature_name=list(nama_fitur))
    return lgb.train(wf.params_lgbm(cfg, teknis), ds, num_boost_round=cfg["n_estimators"])


def skor_mentah(model, X):
    """Logit (raw score) per kelas; softmax-nya sama dengan `model.predict`."""
    _, np = wf._impor_lgbm()
    return [[float(v) for v in z] for z in model.predict(np.asarray(X, dtype=float),
                                                         raw_score=True)]


def latih_akhir(rows, cfg, teknis, nama_fitur, latih=latih_lgbm):
    """Latih sekali pada seluruh dev; event yang jendela labelnya mencapai 2024 di-purge."""
    data, purged = pembanding.data_latih(rows, BATAS_DEV)
    if not data:
        raise ValueError("tidak ada data latih dev")
    model = latih([r["x"] for r in data], [CLASSES.index(r["label"]) for r in data], cfg, teknis,
                  nama_fitur)
    return model, data, purged


# ---------------------------------------------------------------- temperature scaling
def softmax(z, tau=1.0):
    if not tau > 0:
        raise ValueError(f"τ harus > 0, didapat {tau}")
    s = [v / tau for v in z]
    m = max(s)
    e = [math.exp(v - m) for v in s]
    tot = sum(e)
    return [v / tot for v in e]


def log_loss_tau(Z, labels, tau, clip):
    return sum(pembanding.log_loss(softmax(z, tau), y, clip) for z, y in zip(Z, labels)) / len(Z)


def golden_section(f, lo, hi, tol=TOL_TAU, maks_iter=500):
    """Minimum fungsi unimodal pada [lo, hi]; kedua ujung ikut dibandingkan di akhir."""
    if not (lo < hi and tol > 0):
        raise ValueError(f"rentang/toleransi tidak valid: [{lo}, {hi}], tol={tol}")
    a, b = lo, hi
    c, d = b - PHI * (b - a), a + PHI * (b - a)
    fc, fd = f(c), f(d)
    it = 0
    while b - a > tol:
        if it >= maks_iter:
            raise RuntimeError(f"golden-section belum konvergen setelah {maks_iter} iterasi")
        if fc <= fd:
            b, d, fd = d, c, fc
            c = b - PHI * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + PHI * (b - a)
            fd = f(d)
        it += 1
    x = (a + b) / 2
    fx, x = min((f(x), x), (f(lo), lo), (f(hi), hi))
    return x, fx, it


def cari_tau(Z, labels, rentang, clip, tol=TOL_TAU):
    """τ peminimal log loss; dicek terhadap grid log-spasi agar bukan minimum lokal."""
    if not Z or len(Z) != len(labels):
        raise ValueError("butuh logit dan label dengan jumlah sama (> 0)")
    lo, hi = float(rentang[0]), float(rentang[1])

    def f(t):
        return log_loss_tau(Z, labels, t, clip)

    tau, ll, it = golden_section(f, lo, hi, tol)
    grid = [lo * (hi / lo) ** (i / N_GRID_CEK) for i in range(N_GRID_CEK + 1)]
    ll_grid, tau_grid = min((f(t), t) for t in grid)
    if ll > ll_grid + 1e-9:
        raise RuntimeError(f"golden-section (τ={tau}, {ll}) kalah dari grid (τ={tau_grid}, "
                           f"{ll_grid}): log loss tidak unimodal?")
    return {"tau": tau, "log_loss": ll, "iterasi": it, "rentang": [lo, hi], "toleransi": tol,
            "di_batas": min(tau - lo, hi - tau) <= 10 * tol,
            "cek_grid": {"n": N_GRID_CEK + 1, "tau": tau_grid, "log_loss": ll_grid}}


# ---------------------------------------------------------------- metrik peluang
def brier(P, labels):
    """Brier multikelas (1/N) Σ_i Σ_c (p_ic − y_ic)², rentang 0–2."""
    return sum(sum((p[j] - (CLASSES[j] == y)) ** 2 for j in range(len(CLASSES)))
               for p, y in zip(P, labels)) / len(P)


def reliabilitas(P, labels, n_bin):
    """Per kelas (satu-lawan-semua): n, peluang rata-rata, dan frekuensi teramati per bin."""
    out = {}
    for j, k in enumerate(CLASSES):
        isi = [[] for _ in range(n_bin)]
        for p, y in zip(P, labels):
            isi[min(int(p[j] * n_bin), n_bin - 1)].append((p[j], y == k))
        out[k] = [{"bin": i, "batas": [i / n_bin, (i + 1) / n_bin], "n": len(b),
                   "p_rerata": sum(v for v, _ in b) / len(b) if b else None,
                   "frek": sum(o for _, o in b) / len(b) if b else None}
                  for i, b in enumerate(isi)]
    return out


def ece(P, labels, n_bin, rel=None):
    """ECE per kelas: Σ_bin (n_bin/N)·|p̄ − frekuensi|."""
    rel = rel or reliabilitas(P, labels, n_bin)
    return {k: sum(b["n"] * abs(b["p_rerata"] - b["frek"]) for b in rel[k] if b["n"]) / len(P)
            for k in CLASSES}


def ringkas_peluang(P, labels, clip, n_bin, dengan_reliabilitas=True):
    rel = reliabilitas(P, labels, n_bin)
    out = {"n": len(P),
           "log_loss": sum(pembanding.log_loss(p, y, clip) for p, y in zip(P, labels)) / len(P),
           "brier": brier(P, labels), "ece": ece(P, labels, n_bin, rel)}
    return {**out, "reliabilitas": rel} if dengan_reliabilitas else out


def log_loss_per_level(rows, prediksi, clip):
    """Log loss per jenis level (PDH/PDL) untuk setiap prediktor {nama: daftar peluang}."""
    out = {}
    for lv in LEVEL:
        idx = [i for i, r in enumerate(rows) if r["level"] == lv]
        out[lv] = {"n": len(idx), **{
            nm: (sum(pembanding.log_loss(P[i], rows[i]["label"], clip) for i in idx) / len(idx)
                 if idx else None) for nm, P in prediksi.items()}}
    return out


# ---------------------------------------------------------------- aturan keputusan
def cek_r(r, a, b):
    """R mengikuti Tabel 4.3: barrier -> nilai tetap; Netral -> di dalam rentang dan konsisten."""
    r_rev, r_cont = r["R_rev"], r["R_cont"]
    if r["label"] == "Reversal":
        ok = abs(r_rev - a / b) <= TOL_R and abs(r_cont + 1) <= TOL_R
    elif r["label"] == "Continuation":
        ok = abs(r_rev + 1) <= TOL_R and abs(r_cont - b / a) <= TOL_R
    else:
        ok = (-1 < r_rev < a / b and -1 < r_cont < b / a
              and abs(r_cont + r_rev * b / a) <= TOL_R)
    if not ok:
        raise ValueError(f"{r['event_id']}: R tidak sesuai Tabel 4.3 ({r['label']}, "
                         f"R_rev={r_rev}, R_cont={r_cont})")


def m_netral(rows, a, b):
    """m_rev, m_cont = rata-rata R reversal/continuation atas event Netral (Proposal IV.II.8)."""
    for r in rows:
        cek_r(r, a, b)
    net = [r for r in rows if r["label"] == "Netral"]
    if not net:
        raise ValueError("tidak ada event Netral untuk menghitung m_rev/m_cont")
    return {"n_netral": len(net), "m_rev": sum(r["R_rev"] for r in net) / len(net),
            "m_cont": sum(r["R_cont"] for r in net) / len(net)}


def nilai_harapan(p, a, b, m):
    """E(rev) = p_rev·a/b − p_cont + p_net·m_rev; E(cont) = p_cont·b/a − p_rev + p_net·m_cont."""
    p_rev, p_cont, p_net = p
    return (p_rev * a / b - p_cont + p_net * m["m_rev"],
            p_cont * b / a - p_rev + p_net * m["m_cont"])


def rekomendasi(e_rev, e_cont, delta):
    """Aksi dengan E tertinggi bila ≥ δ; abstain bila keduanya < δ atau seri persis."""
    if max(e_rev, e_cont) < delta or e_rev == e_cont:
        return ABSTAIN
    return "Reversal" if e_rev > e_cont else "Continuation"


def r_kotor(row, rek):
    return {"Reversal": row["R_rev"], "Continuation": row["R_cont"], ABSTAIN: 0.0}[rek]


def _ringkas_strategi(rows, reks):
    rs = [r_kotor(r, k) for r, k in zip(rows, reks)]
    n_arah = sum(k != ABSTAIN for k in reks)
    return {"n": len(rows), "n_rev": reks.count("Reversal"), "n_cont": reks.count("Continuation"),
            "n_abstain": reks.count(ABSTAIN), "cakupan": n_arah / len(rows),
            "r_per_event": sum(rs) / len(rows),
            "r_per_rekomendasi": sum(rs) / n_arah if n_arah else None}


def pratinjau_keputusan(rows, P, a, b, m, deltas):
    """SM per δ: cakupan dan R kotor (tanpa biaya), abstain = 0."""
    E = [nilai_harapan(p, a, b, m) for p in P]
    return [{"delta": d, **_ringkas_strategi(rows, [rekomendasi(*e, d) for e in E])}
            for d in deltas]


def gabung_oof(oof_rows, dev_rows, cfg_id):
    """Prediksi OOF walk-forward konfigurasi terpilih, sejajar dengan baris dev (label sama)."""
    by_id = {r["event_id"]: r for r in dev_rows}
    rows, P = [], []
    for o in oof_rows:
        if o["konfigurasi"] != cfg_id:
            continue
        r = by_id.get(o["event_id"])
        if r is None or r["label"] != o["label"]:
            raise ValueError(f"{o['event_id']}: OOF tidak cocok dengan data dev")
        rows.append(r)
        P.append([float(o["p_rev"]), float(o["p_cont"]), float(o["p_net"])])
    if not rows:
        raise ValueError(f"tidak ada prediksi OOF untuk {cfg_id}")
    if len({r["event_id"] for r in rows}) != len(rows):
        raise ValueError("event OOF ganda")
    return rows, P


def strategi_acuan(rows):
    """S0 = semua sentuhan reversal; S1 = reversal hanya bila wick ≥ 2/3 (R kotor)."""
    return {"S0": _ringkas_strategi(rows, ["Reversal"] * len(rows)),
            "S1": _ringkas_strategi(rows, ["Reversal" if r["b1"] else ABSTAIN for r in rows])}


# ---------------------------------------------------------------- tahap
CATATAN_M = ("event Netral data latih dev sesudah purging di 2024-01-01 (sama dengan seluruh dev "
             "bila purging 0)")
ATURAN_SERI = ("Karena R_cont = −(b/a)·R_rev di setiap event, E(cont) = −(b/a)·E(rev); seri hanya "
               "terjadi bila keduanya 0 dan diperlakukan sebagai abstain "
               "(hanya relevan untuk δ = 0).")
CATATAN_CALIB = ("Deskriptif saja: τ diestimasi pada data 2024 yang sama, R kotor tanpa biaya "
                 "spread, n kecil. Tidak dipakai untuk memilih δ atau parameter lain; evaluasi RO2 "
                 "resmi hanya pada data uji.")
CATATAN_OOF = ("Peluang OOF walk-forward konfigurasi terpilih 2019–2023 (tanpa τ, model per "
               "lipatan); m_rev/m_cont dari seluruh Netral dev ikut memuat tahun validasi. "
               "Deskriptif, R kotor tanpa biaya.")


def _masukan(paths):
    """Berkas masukan tahap ini; tolak bila keluaran tahap sebelumnya belum ada."""
    p = {"features_dev.csv": paths.data / "features_dev.csv",
         "labels_dev.csv": paths.input / "labels_dev.csv",
         "features_calib.csv": paths.data / "features_calib.csv",
         "labels_calib.csv": paths.input / "labels_calib.csv",
         "walk_forward_oof_dev.csv": paths.data / "walk_forward_oof_dev.csv",
         "walk_forward_rekonsiliasi.json": paths.output / "walk_forward_rekonsiliasi.json"}
    pembuat = {"features_dev.csv": "tahap `fitur`", "features_calib.csv": "tahap `fitur`",
               "labels_dev.csv": "pipeline data (`python -m pdhl_data run`)",
               "labels_calib.csv": "pipeline data (`python -m pdhl_data run`)",
               "walk_forward_oof_dev.csv": "tahap `walk-forward`",
               "walk_forward_rekonsiliasi.json": "tahap `walk-forward`"}
    for nama, oleh in pembuat.items():
        if not p[nama].exists():
            raise FileNotFoundError(f"{p[nama]} belum ada; jalankan {oleh} dulu")
    return p


def _skor_oof(recon_wf, cfg):
    """Log loss OOF konfigurasi terpilih dari rekonsiliasi walk-forward."""
    skor = recon_wf.get("skor", {}).get(cfg["id"])
    if skor is None:
        raise ValueError(f"walk_forward_rekonsiliasi.json tidak memuat skor {cfg['id']}; jalankan "
                         "ulang tahap `walk-forward`")
    return skor


def _cek_prediksi(P_mentah, P_lgbm):
    selisih = max(abs(x - y) for p, q in zip(P_mentah, P_lgbm) for x, y in zip(p, q))
    if selisih > TOL_PREDIKSI:
        raise RuntimeError(f"softmax(skor mentah) ≠ predict LightGBM (selisih {selisih})")


def _baris_calib(r, z, pm, p, e, rk, clip):
    return {"event_id": r["event_id"], "session_date": r["session_date"], "label": r["label"],
            **{f"z_{SINGKAT[k]}": z[j] for j, k in enumerate(CLASSES)},
            **{f"p_{SINGKAT[k]}_mentah": pm[j] for j, k in enumerate(CLASSES)},
            **{f"p_{SINGKAT[k]}": p[j] for j, k in enumerate(CLASSES)},
            "ll_mentah": pembanding.log_loss(pm, r["label"], clip),
            "ll": pembanding.log_loss(p, r["label"], clip), "E_rev": e[0], "E_cont": e[1],
            "rekomendasi": rk, "R_rev": r["R_rev"], "R_cont": r["R_cont"],
            "R_kotor": r_kotor(r, rk)}


def run_model_akhir(paths, kunci, sumber, latih=latih_lgbm, skor=skor_mentah,
                    prediksi=wf.prediksi_lgbm):
    """Model akhir + τ + parameter keputusan; tulis model, CSV calib, rekonsiliasi, laporan, gambar.

    `latih`/`skor`/`prediksi` dapat diganti versi palsu untuk tes. Model harus punya
    `model_to_string()` dan `num_trees()`. Mengembalikan rekonsiliasi."""
    teknis, clip, n_bin, a, b = kunci.teknis, kunci.clip, kunci.ece_bin, kunci.a, kunci.b
    p = _masukan(paths)
    sha = {n: common.sha256_file(x) for n, x in p.items()}
    recon_wf = json.loads(p["walk_forward_rekonsiliasi.json"].read_text(encoding="utf-8"))
    cfg = konfigurasi_terpilih(recon_wf, teknis,
                               {n: sha[n] for n in ("features_dev.csv", "labels_dev.csv")})
    skor_oof = _skor_oof(recon_wf, cfg)

    dev = muat_split(p["features_dev.csv"], p["labels_dev.csv"], "dev", kunci.theta)
    cal = muat_split(p["features_calib.csv"], p["labels_calib.csv"], "calib", kunci.theta)
    nama = wf.kolom_fitur()
    model, latih_rows, purged = latih_akhir(dev, cfg, teknis, nama, latih=latih)
    teks_model = model.model_to_string()

    # Skor mentah dan konsistensi dengan predict bawaan LightGBM.
    Xc, Yc = [r["x"] for r in cal], [r["label"] for r in cal]
    Zc = skor(model, Xc)
    P_mentah = [softmax(z) for z in Zc]
    _cek_prediksi(P_mentah, prediksi(model, Xc))
    ll_dev = log_loss_tau(skor(model, [r["x"] for r in latih_rows]),
                          [r["label"] for r in latih_rows], 1.0, clip)

    tau = cari_tau(Zc, Yc, teknis["temperature_rentang"], clip)
    P = [softmax(z, tau["tau"]) for z in Zc]
    pemb = pembanding.model_akhir(dev, kunci.laplace)
    P_b = [pembanding.prediksi(pemb, r) for r in cal]
    for r in dev + cal:
        cek_r(r, a, b)
    m = m_netral(latih_rows, a, b)          # Netral dev sesudah purging (tanpa harga 2024)
    E = [nilai_harapan(pp, a, b, m) for pp in P]
    rek = [rekomendasi(*e, kunci.delta) for e in E]
    oof_rows, P_oof = gabung_oof(common.load_model_csv(p["walk_forward_oof_dev.csv"]), dev,
                                 cfg["id"])

    recon = {
        "rm": "RM-25", "kelas": list(CLASSES), "lingkungan": wf._lingkungan(),
        "konfigurasi": cfg, "params": wf.params_lgbm(cfg, teknis),
        "jumlah_pohon": model.num_trees(), "kolom_fitur": nama,
        "sumber_sha256": sha,
        "kunci1_sha256_sesuai_manifest": {n: kunci.manifest_sha256[n] for n in BERKAS_KUNCI1},
        "model_berkas": MODEL_BERKAS,
        "model_sha256": common.sha256_bytes(teks_model.encode()),
        "n_latih": len(latih_rows), "n_purged": len(purged), "n_calib": len(cal),
        "dev_dalam_sampel": {"log_loss": ll_dev, "oof_walk_forward": skor_oof},
        "kalibrasi": {"metode": "temperature scaling, p = softmax(z/τ)", **tau,
                      "sebelum": ringkas_peluang(P_mentah, Yc, clip, n_bin),
                      "sesudah": ringkas_peluang(P, Yc, clip, n_bin)},
        "pembanding_calib": {nm: ringkas_peluang([x[nm] for x in P_b], Yc, clip, n_bin, False)
                             for nm in ("B0", "B1")},
        "per_level_calib": log_loss_per_level(
            cal, {"model_sebelum": P_mentah, "model_sesudah": P,
                  "B0": [x["B0"] for x in P_b], "B1": [x["B1"] for x in P_b]}, clip),
        "pembanding_dev": {"B0": pemb["B0"], "B1": pemb["B1"]},
        "aturan_keputusan": {"a": a, "b": b, **m, "sumber_m": CATATAN_M, "delta": kunci.delta,
                             "delta_grid": list(kunci.delta_grid), "aturan_seri": ATURAN_SERI},
        "pratinjau_calib": {"catatan": CATATAN_CALIB,
                            "SM": pratinjau_keputusan(cal, P, a, b, m, kunci.delta_grid),
                            **strategi_acuan(cal)},
        "cakupan_oof_dev": {"catatan": CATATAN_OOF,
                            "SM": pratinjau_keputusan(oof_rows, P_oof, a, b, m, kunci.delta_grid),
                            **strategi_acuan(oof_rows)},
        "versi_pdhl_data": sumber.get("versi_pdhl_data"), "commit": sumber.get("commit"),
        "sumber_pipeline_sha256": sumber.get("sumber_sha256"),
    }
    pred = [_baris_calib(r, z, pm, pp, e, rk, clip)
            for r, z, pm, pp, e, rk in zip(cal, Zc, P_mentah, P, E, rek)]
    teks = laporan(recon)
    svg = gambar_reliabilitas(recon["kalibrasi"]["sebelum"]["reliabilitas"],
                              recon["kalibrasi"]["sesudah"]["reliabilitas"], len(cal), tau["tau"])

    # Semua isi (termasuk teks JSON dan CSV) siap sebelum ada berkas yang ditulis.
    teks_recon = json.dumps(recon, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    teks_calib = common.csv_text(pred, CALIB_COLS)
    paths.model.mkdir(parents=True, exist_ok=True)
    (paths.model / MODEL_BERKAS).write_bytes(teks_model.encode())     # = byte yang di-hash
    paths.data.mkdir(parents=True, exist_ok=True)
    (paths.data / "model_akhir_calib.csv").write_text(teks_calib)
    paths.output.mkdir(parents=True, exist_ok=True)
    (paths.output / "model_akhir_rekonsiliasi.json").write_text(teks_recon, encoding="utf-8")
    (paths.output / "laporan_model_akhir.md").write_text(teks, encoding="utf-8")
    (paths.output / "gambar_reliabilitas_2024.svg").write_text(svg, encoding="utf-8")
    return recon


def ringkas(recon):
    """Baris ringkasan untuk konsol."""
    cfg, k, m = recon["konfigurasi"], recon["kalibrasi"], recon["aturan_keputusan"]
    pc = recon["pembanding_calib"]
    return "\n".join([
        f"Model akhir {cfg['id']}: {recon['n_latih']} event dev (purged {recon['n_purged']}), "
        f"{recon['jumlah_pohon']} pohon, SHA-256 {recon['model_sha256'][:16]}",
        f"τ = {k['tau']:.4f} | log loss 2024 {k['sebelum']['log_loss']:.4f} → "
        f"{k['sesudah']['log_loss']:.4f} | B0 {pc['B0']['log_loss']:.4f}, "
        f"B1 {pc['B1']['log_loss']:.4f}",
        f"m_rev {m['m_rev']:+.4f}R, m_cont {m['m_cont']:+.4f}R (n Netral {m['n_netral']}), "
        f"δ = {m['delta']:g}R",
    ])
