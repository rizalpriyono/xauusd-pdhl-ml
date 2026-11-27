"""Kurva belajar per iterasi boosting (Proses 3): log loss dan akurasi, latih vs validasi.

Pada gradient boosting, padanan "epoch" adalah iterasi boosting (jumlah pohon per kelas). Model
setiap lipatan walk-forward dilatih ulang untuk konfigurasi terpilih (kedalaman dan minimal isi
daun), dengan parameter Kunci-1 yang sama dan mode deterministik, sampai jumlah pohon terbesar di
grid. LightGBM mencatat log loss dan galat klasifikasi pada data latih dan validasi di setiap
iterasi; metrik evaluasi tidak memengaruhi pelatihan.

Pengaman: log loss validasi gabungan pada setiap jumlah pohon di grid (100 dan 300) harus sama
dengan skor walk-forward di `walk_forward_rekonsiliasi.json`. Bila berbeda, proses berhenti.

Kurva ini deskriptif. Ia tidak dipakai untuk memilih jumlah pohon (early stopping), karena grid dan
konfigurasi sudah terkunci di Kunci-1. Hanya data dev yang dipakai.
"""
import datetime as dt

from pdhl_data.common import CLASSES

from .. import pembanding
from .. import walk_forward as wf

TOL = 1e-8


def _argmax(p):
    return max(range(len(p)), key=p.__getitem__)


def akurasi_pembanding(latih, val, laplace):
    """Akurasi B0 dan B1 (kelas dengan peluang tertinggi) pada data validasi satu lipatan."""
    model = pembanding.latih_pembanding(latih, laplace)
    benar = {"B0": 0, "B1": 0}
    for r in val:
        p = pembanding.prediksi(model, r)
        for nama in benar:
            benar[nama] += CLASSES[_argmax(p[nama])] == r["label"]
    return {nama: n / len(val) for nama, n in benar.items()}


def hitung_kurva(rows, kunci, tahun, max_depth, min_child, n_iter):
    """Metrik per iterasi untuk setiap lipatan: log loss dan akurasi latih/validasi, plus B0/B1."""
    lgb, np = wf._impor_lgbm()
    cfg = {"id": "kurva", "max_depth": max_depth, "n_estimators": n_iter,
           "min_child_samples": min_child}
    params = {**wf.params_lgbm(cfg, kunci.teknis), "metric": ["multi_logloss", "multi_error"]}
    lipatan = []
    for y in tahun:
        latih, _ = pembanding.data_latih(rows, dt.date(y, 1, 1))
        val = [r for r in rows if int(r["session_date"][:4]) == y]
        if not latih or not val:
            raise ValueError(f"lipatan {y}: data latih atau validasi kosong")
        set_ = {nama: lgb.Dataset(np.asarray([r["x"] for r in d], dtype=float),
                                  label=np.asarray([CLASSES.index(r["label"]) for r in d]))
                for nama, d in (("latih", latih), ("val", val))}
        ev = {}
        lgb.train(params, set_["latih"], num_boost_round=n_iter,
                  valid_sets=[set_["latih"], set_["val"]], valid_names=["latih", "val"],
                  callbacks=[lgb.record_evaluation(ev)])
        lipatan.append({
            "tahun": y, "n_latih": len(latih), "n_val": len(val),
            "ll_latih": list(ev["latih"]["multi_logloss"]),
            "ll_val": list(ev["val"]["multi_logloss"]),
            "akurasi_latih": [1 - e for e in ev["latih"]["multi_error"]],
            "akurasi_val": [1 - e for e in ev["val"]["multi_error"]],
            "akurasi_pembanding": akurasi_pembanding(latih, val, kunci.laplace)})
    return lipatan


def gabung(lipatan, metrik, bobot):
    """Rata-rata per iterasi atas semua lipatan, dibobot jumlah sentuhan (`n_latih`/`n_val`)."""
    total = sum(f[bobot] for f in lipatan)
    n_iter = len(lipatan[0][metrik])
    return [sum(f[metrik][i] * f[bobot] for f in lipatan) / total for i in range(n_iter)]


def cek_cocok(val_gabungan, skor_wf, max_depth, min_child, grid_n):
    """Log loss validasi gabungan pada jumlah pohon grid harus sama dengan skor walk-forward."""
    for n in grid_n:
        cfg_id = f"d{max_depth}_n{n}_m{min_child}"
        if abs(val_gabungan[n - 1] - skor_wf[cfg_id]) > TOL * max(1.0, abs(skor_wf[cfg_id])):
            raise ValueError(f"kurva iterasi {n} ({val_gabungan[n - 1]}) tidak sama dengan skor "
                             f"walk-forward {cfg_id} ({skor_wf[cfg_id]})")


def kurva_terpilih(D):
    """Kurva untuk kedalaman dan isi daun terpilih, iterasi 1 s.d. jumlah pohon terbesar grid."""
    rek = D["recon"]["walk_forward"]
    t = rek["terpilih"]
    grid_n = sorted(D["kunci"].teknis["lightgbm"]["n_estimators"])
    lipatan = hitung_kurva(D["dev"], D["kunci"], rek["tahun_validasi"], t["max_depth"],
                           t["min_child_samples"], max(grid_n))
    val = gabung(lipatan, "ll_val", "n_val")
    cek_cocok(val, rek["skor"], t["max_depth"], t["min_child_samples"], grid_n)
    n_val = sum(f["n_val"] for f in lipatan)
    return {"lipatan": lipatan, "grid_n": grid_n, "terpilih": t,
            "ll_latih": gabung(lipatan, "ll_latih", "n_latih"), "ll_val": val,
            "akurasi_latih": gabung(lipatan, "akurasi_latih", "n_latih"),
            "akurasi_val": gabung(lipatan, "akurasi_val", "n_val"),
            "akurasi_pembanding": {nm: sum(f["akurasi_pembanding"][nm] * f["n_val"]
                                          for f in lipatan) / n_val for nm in ("B0", "B1")}}
