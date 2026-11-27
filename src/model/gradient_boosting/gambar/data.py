"""Data gambar: hanya keluaran tahap sebelumnya (dev dan calib), dimuat lewat `load_model_csv`.

Tidak ada perhitungan model baru. Nilai harapan, rekomendasi, dan ringkasan strategi memakai
fungsi `model_akhir` yang sama dengan tahap 4, sehingga angka di gambar sama dengan rekonsiliasi.
"""
import json
from collections import Counter

from pdhl_data import common
from pdhl_data.common import CLASSES

from .. import model_akhir as ma
from ..fitur import CATEGORICAL, FEATURE_COLS

NUMERIK = tuple(c for c in FEATURE_COLS if c not in CATEGORICAL)
REKON = ("pembanding", "walk_forward", "model_akhir")


def _wajib(paths):
    """Berkas yang dibutuhkan beserta tahap pembuatnya."""
    return {paths.data / "features_dev.csv": "fitur", paths.data / "features_calib.csv": "fitur",
            paths.input / "labels_dev.csv": "pipeline data",
            paths.input / "labels_calib.csv": "pipeline data",
            paths.output / "pembanding_rekonsiliasi.json": "pembanding",
            paths.data / "walk_forward_oof_dev.csv": "walk-forward",
            paths.output / "walk_forward_rekonsiliasi.json": "walk-forward",
            paths.data / "model_akhir_calib.csv": "model-akhir",
            paths.output / "model_akhir_rekonsiliasi.json": "model-akhir",
            paths.model / ma.MODEL_BERKAS: "model-akhir"}


def muat(paths, kunci):
    """Semua masukan gambar dalam satu dict. Menolak bila keluaran tahap sebelumnya belum ada."""
    for p, tahap in _wajib(paths).items():
        if not p.exists():
            raise FileNotFoundError(f"{p} belum ada; jalankan tahap `{tahap}` dulu")
    dev = ma.muat_split(paths.data / "features_dev.csv", paths.input / "labels_dev.csv", "dev",
                        kunci.theta)
    cal = ma.muat_split(paths.data / "features_calib.csv", paths.input / "labels_calib.csv",
                        "calib", kunci.theta)
    recon = {n: json.loads((paths.output / f"{n}_rekonsiliasi.json").read_text(encoding="utf-8"))
             for n in REKON}
    pred = common.load_model_csv(paths.data / "model_akhir_calib.csv")
    if [r["event_id"] for r in pred] != [r["event_id"] for r in cal]:
        raise ValueError("model_akhir_calib.csv tidak sejajar dengan data calib; jalankan ulang "
                         "tahap `model-akhir`")
    model_path = paths.model / ma.MODEL_BERKAS
    if common.sha256_file(model_path) != recon["model_akhir"]["model_sha256"]:
        raise ValueError(f"{model_path} berbeda dari model_sha256 di "
                         "model_akhir_rekonsiliasi.json; "
                         "jalankan ulang tahap `model-akhir`")
    cfg_id = recon["model_akhir"]["konfigurasi"]["id"]
    oof_rows, P_oof = ma.gabung_oof(common.load_model_csv(paths.data / "walk_forward_oof_dev.csv"),
                                    dev, cfg_id)
    urut = sorted(range(len(oof_rows)), key=lambda i: (oof_rows[i]["session_date"],
                                                       oof_rows[i]["event_id"]))
    return {"dev": dev, "calib": cal, "pred_calib": pred, "recon": recon, "kunci": kunci,
            "fitur": {s: common.load_model_csv(paths.data / f"features_{s}.csv")
                      for s in ("dev", "calib")},
            "oof_rows": [oof_rows[i] for i in urut], "P_oof": [P_oof[i] for i in urut],
            "model_path": model_path}


# ---------------------------------------------------------------- ringkasan untuk gambar
def label_per_tahun(rows):
    """{tahun: Counter(kelas)} dari baris berlabel."""
    out = {}
    for r in rows:
        out.setdefault(int(r["session_date"][:4]), Counter())[r["label"]] += 1
    return {t: {k: out[t].get(k, 0) for k in CLASSES} for t in sorted(out)}


def parameter_keputusan(D):
    a = D["recon"]["model_akhir"]["aturan_keputusan"]
    return a["a"], a["b"], {"m_rev": a["m_rev"], "m_cont": a["m_cont"]}, a["delta"]


def peluang_calib(D, terkalibrasi=True):
    akhir = "" if terkalibrasi else "_mentah"
    return [[float(r[f"p_{s}{akhir}"]) for s in ("rev", "cont", "net")] for r in D["pred_calib"]]


def e_rev(P, a, b, m):
    return [ma.nilai_harapan(p, a, b, m)[0] for p in P]


def rekomendasi_sm(P, a, b, m, delta):
    return [ma.rekomendasi(*ma.nilai_harapan(p, a, b, m), delta) for p in P]


def rekomendasi_strategi(rows, P, a, b, m, delta):
    """{SM, S0, S1: daftar rekomendasi per sentuhan}."""
    return {"SM": rekomendasi_sm(P, a, b, m, delta),
            "S0": ["Reversal"] * len(rows),
            "S1": ["Reversal" if r["b1"] else ma.ABSTAIN for r in rows]}


def r_kumulatif(rows, reks):
    total, out = 0.0, []
    for r, k in zip(rows, reks):
        total += ma.r_kotor(r, k)
        out.append(total)
    return out


def sm_per_tahun(rows, P, a, b, m, delta):
    """Cakupan dan R per rekomendasi SM per tahun (pratinjau_keputusan per kelompok tahun)."""
    tahun = sorted({int(r["session_date"][:4]) for r in rows})
    out = []
    for t in tahun:
        idx = [i for i, r in enumerate(rows) if int(r["session_date"][:4]) == t]
        h = ma.pratinjau_keputusan([rows[i] for i in idx], [P[i] for i in idx], a, b, m,
                                   [delta])[0]
        out.append({"tahun": t, **h})
    return out


def korelasi(rows, cols):
    """Matriks korelasi Pearson antarfitur (numpy)."""
    import numpy as np
    X = np.asarray([[float(r[c]) for c in cols] for r in rows])
    return np.corrcoef(X, rowvar=False)


def daun(n):
    """Simpul daun LightGBM: tanpa `split_feature` (pohon satu-daun juga tanpa `leaf_index`)."""
    return "split_feature" not in n


def nama_dasar(nama_kolom):
    """Nama fitur dasar dari kolom model (one-hot `f04_sesi=Asia` -> `f04_sesi`)."""
    return nama_kolom.split("=")[0]


def struktur_model(model_path):
    """Baca berkas model LightGBM lalu ringkas strukturnya (lihat `ringkas_pohon`)."""
    from .. import walk_forward as wf
    lgb, _ = wf._impor_lgbm()
    return ringkas_pohon(lgb.Booster(model_file=str(model_path)).dump_model())


def ringkas_pohon(dump):
    """Pohon pertama, isi daun, jumlah pemisahan, dan gain per fitur dasar (one-hot digabung)."""
    nama = dump["feature_names"]
    split, gain, isi = Counter(), Counter(), []

    def jalan(n):
        if daun(n):
            isi.append(n["leaf_count"])
            return
        f = nama_dasar(nama[n["split_feature"]])
        split[f] += 1
        gain[f] += n["split_gain"]
        jalan(n["left_child"])
        jalan(n["right_child"])

    for t in dump["tree_info"]:
        jalan(t["tree_structure"])
    return {"nama": nama, "pohon_pertama": dump["tree_info"][0]["tree_structure"],
            "jumlah_pohon": len(dump["tree_info"]), "isi_daun": isi, "split": dict(split),
            "gain": dict(gain)}
