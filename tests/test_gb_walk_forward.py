"""Tes tahap 3 validasi walk-forward LightGBM dan pemilihan konfigurasi (RM-24; IV.II.4, IV.II.6).

Port dari `Pemodelan PDH-PDL/tests/test_walk_forward.py`. Tes tanpa LightGBM memakai pelatih palsu
(stdlib) sehingga bisa jalan di Python sistem/CI. Tes yang butuh LightGBM dilewati bila pustakanya
tidak ada (`pip install -e '.[model]'`).
"""
import importlib.util
import json
import math
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gb_sintetis import make_run  # noqa: E402
from model.gradient_boosting import jalur, kunci, pembanding  # noqa: E402
from model.gradient_boosting import walk_forward as wf  # noqa: E402
from model.gradient_boosting.fitur import run_fitur  # noqa: E402
from pdhl_data import common  # noqa: E402
from pdhl_data.config import load_config  # noqa: E402

R, C, N = common.CLASSES
ADA_LGBM = importlib.util.find_spec("lightgbm") is not None
KUNCI = kunci.muat_kunci(load_config(ROOT / "configs" / "tesis.json"), ROOT / "configs" / "kunci_1")
TEKNIS = KUNCI.teknis


def baris(i, tanggal, label, x=None, akhir=None):
    return {"event_id": f"e{i}", "session_date": tanggal, "label": label,
            "label_end_session_date": akhir or tanggal, "f13": 0.1, "b1": False,
            "x": x if x is not None else [float(i % 3), float(i % 5)]}


def latih_frekuensi(X, y, cfg, teknis):
    """Pelatih palsu: 'model' = frekuensi kelas data latih (setara B0)."""
    c = Counter(y)
    return [c.get(k, 0) / len(y) for k in range(len(common.CLASSES))]


def prediksi_frekuensi(model, X):
    return [list(model) for _ in X]


class TestGrid(unittest.TestCase):
    def test_delapan_konfigurasi_dalam_urutan_dokumen(self):
        cfgs = wf.konfigurasi_grid(TEKNIS)
        self.assertEqual(len(cfgs), 8)
        self.assertEqual(cfgs[0], {"id": "d2_n100_m20", "max_depth": 2, "n_estimators": 100,
                                   "min_child_samples": 20})
        self.assertEqual(cfgs[-1]["id"], "d3_n300_m50")
        self.assertEqual(len({c["id"] for c in cfgs}), 8)

    def test_parameter_lgbm_mengikuti_kunci(self):
        p = wf.params_lgbm(wf.konfigurasi_grid(TEKNIS)[5], TEKNIS)
        self.assertEqual(p["objective"], "multiclass")
        self.assertEqual(p["num_class"], 3)
        self.assertEqual(p["learning_rate"], 0.05)
        self.assertEqual((p["max_depth"], p["min_child_samples"]), (3, 50))
        self.assertEqual(p["seed"], 20261119)
        self.assertTrue(p["deterministic"])
        self.assertEqual(p["num_threads"], 1)
        self.assertNotIn("n_estimators", p)        # jumlah pohon lewat num_boost_round

    def test_grid_ditolak_bila_kunci_berubah(self):
        lg = TEKNIS["lightgbm"]
        rusak = {"learning_rate": {**TEKNIS, "lightgbm": {**lg, "learning_rate": 0.1}},
                 "kedalaman": {**TEKNIS, "lightgbm": {**lg, "max_depth": [2, 4]}},
                 "deterministik": {**TEKNIS, "lightgbm": {**lg, "deterministic": False}},
                 "thread": {**TEKNIS, "lightgbm": {**lg, "num_threads": 4}},
                 "seed": {**TEKNIS, "seed_model": 1}}
        for nama, t in rusak.items():
            with self.subTest(nama), self.assertRaises(ValueError):
                wf.konfigurasi_grid(t)


class TestPemilihan(unittest.TestCase):
    def setUp(self):
        self.cfgs = wf.konfigurasi_grid(TEKNIS)

    def test_memilih_log_loss_terkecil(self):
        skor = {c["id"]: 1.0 for c in self.cfgs}
        skor["d3_n300_m20"] = 0.9
        self.assertEqual(wf.pilih_konfigurasi(skor, self.cfgs, 1e-6)["id"], "d3_n300_m20")

    def test_seri_memilih_pohon_lebih_sedikit(self):
        skor = {c["id"]: 1.0 for c in self.cfgs}
        skor["d3_n300_m20"] = 0.9
        skor["d3_n100_m20"] = 0.9 + 5e-7
        self.assertEqual(wf.pilih_konfigurasi(skor, self.cfgs, 1e-6)["id"], "d3_n100_m20")

    def test_seri_lalu_kedalaman_lalu_daun_lebih_besar(self):
        skor = {c["id"]: 0.9 for c in self.cfgs}
        self.assertEqual(wf.pilih_konfigurasi(skor, self.cfgs, 1e-6)["id"], "d2_n100_m50")

    def test_ringkasan_saat_pemenang_bukan_peringkat_satu(self):
        skor = {c["id"]: 1.0 for c in self.cfgs}
        skor["d3_n300_m20"] = 0.9
        skor["d2_n100_m50"] = 0.9 + 5e-7               # seri, lebih sederhana → menang
        r = wf.ringkasan_pemilihan(skor, self.cfgs, 1e-6)
        self.assertEqual(r["terpilih"]["id"], "d2_n100_m50")
        self.assertEqual(r["peringkat"]["d3_n300_m20"], 1)
        self.assertEqual(sorted(r["seri"]), ["d2_n100_m50", "d3_n300_m20"])
        self.assertEqual(r["terdekat"], "d3_n300_m20")
        self.assertLess(r["selisih_ke_terdekat"], 0)

    def test_ringkasan_tanpa_seri(self):
        skor = {c["id"]: 1.0 + i / 100 for i, c in enumerate(self.cfgs)}
        r = wf.ringkasan_pemilihan(skor, self.cfgs, 1e-6)
        self.assertEqual(r["seri"], ["d2_n100_m20"])
        self.assertEqual(r["terdekat"], "d2_n100_m50")
        self.assertAlmostEqual(r["selisih_ke_terdekat"], 0.01)

    def test_selisih_di_atas_toleransi_tidak_dianggap_seri(self):
        skor = {c["id"]: 1.0 for c in self.cfgs}
        skor["d3_n300_m20"] = 0.9
        skor["d2_n100_m50"] = 0.9 + 2e-6
        self.assertEqual(wf.pilih_konfigurasi(skor, self.cfgs, 1e-6)["id"], "d3_n300_m20")


class TestEncode(unittest.TestCase):
    def _fitur(self, **ubah):
        base = {c: "0.5" for c in wf.NUMERIK}
        base.update({"f01_jenis_level": "1", "f04_sesi": "London", "f05_hari": "Rabu"})
        base.update(ubah)
        return base

    def test_one_hot_sesi_dan_hari(self):
        cols = wf.kolom_fitur()
        x = wf.encode(self._fitur())
        self.assertEqual(len(cols), 21)
        self.assertEqual(len(x), 21)
        d = dict(zip(cols, x))
        self.assertEqual((d["f04_sesi=Asia"], d["f04_sesi=London"], d["f04_sesi=NewYork"]),
                         (0.0, 1.0, 0.0))
        self.assertEqual(sum(v for k, v in d.items() if k.startswith("f05_hari=")), 1.0)
        self.assertEqual(d["f05_hari=Rabu"], 1.0)
        self.assertEqual(d["f01_jenis_level"], 1.0)

    def test_kategori_asing_ditolak(self):
        with self.assertRaises(ValueError):
            wf.encode(self._fitur(f04_sesi="Sydney"))

    def test_nilai_tidak_berhingga_ditolak(self):
        with self.assertRaises(ValueError):
            wf.encode(self._fitur(f06_rezim_volatilitas="nan"))


class TestWalkForwardPalsu(unittest.TestCase):
    def setUp(self):
        self.rows, i = [], 0
        for tahun in range(2016, 2024):
            for bulan, lab in zip((2, 5, 8, 11), (R, C, C, N if tahun % 2 else R)):
                self.rows.append(baris(i, f"{tahun}-{bulan:02d}-03", lab))
                i += 1
        self.cfgs = wf.konfigurasi_grid(TEKNIS)[:2]

    def test_data_latih_hanya_masa_lalu(self):
        dilihat = []

        def latih(X, y, cfg, teknis):
            dilihat.append(len(X))
            return latih_frekuensi(X, y, cfg, teknis)

        oof, lipatan = wf.walk_forward_model(self.rows, self.cfgs, TEKNIS, (2019, 2020), 1e-15,
                                             latih=latih, prediksi=prediksi_frekuensi)
        self.assertEqual(dilihat, [12, 12, 16, 16])        # 2 konfigurasi × 2 lipatan
        self.assertEqual(len(oof), 2 * 2 * 4)
        self.assertEqual(len(lipatan), 4)
        self.assertTrue(all(int(r["session_date"][:4]) == r["tahun_validasi"] for r in oof))

    def test_pelatih_frekuensi_sama_dengan_b0_pembanding(self):
        oof, _ = wf.walk_forward_model(self.rows, self.cfgs[:1], TEKNIS, pembanding.TAHUN_VALIDASI,
                                       1e-15, latih=latih_frekuensi, prediksi=prediksi_frekuensi)
        b0, _ = pembanding.walk_forward(self.rows, pembanding.TAHUN_VALIDASI, laplace=1, clip=1e-15)
        self.assertEqual([r["event_id"] for r in oof], [r["event_id"] for r in b0])
        for a, b in zip(oof, b0):
            self.assertAlmostEqual(a["ll"], b["ll_b0"])

    def test_ringkasan_rata_rata_per_event(self):
        oof, lipatan = wf.walk_forward_model(self.rows, self.cfgs, TEKNIS, (2019, 2020), 1e-15,
                                             latih=latih_frekuensi, prediksi=prediksi_frekuensi)
        skor = wf.skor_konfigurasi(oof, self.cfgs)
        for c in self.cfgs:
            lls = [r["ll"] for r in oof if r["konfigurasi"] == c["id"]]
            self.assertAlmostEqual(skor[c["id"]], sum(lls) / len(lls))

    def test_peluang_tidak_valid_ditolak(self):
        with self.assertRaises(ValueError):
            wf.walk_forward_model(self.rows, self.cfgs[:1], TEKNIS, (2019,), 1e-15,
                                  latih=latih_frekuensi,
                                  prediksi=lambda m, X: [[0.5, 0.5, 0.5] for _ in X])

    def test_lipatan_tanpa_data_latih_ditolak(self):
        rows = [baris(0, "2019-03-01", R)]
        with self.assertRaises(ValueError):
            wf.walk_forward_model(rows, self.cfgs[:1], TEKNIS, (2019,), 1e-15,
                                  latih=latih_frekuensi, prediksi=prediksi_frekuensi)

    def test_purging_dan_tanpa_event_validasi_di_data_latih(self):
        rows = [baris(0, "2017-03-01", R, x=[0.0]), baris(1, "2018-06-01", C, x=[1.0]),
                baris(2, "2018-12-28", N, x=[2.0], akhir="2019-01-03"),   # di-purge
                baris(3, "2019-02-01", C, x=[3.0]), baris(4, "2019-07-01", R, x=[4.0])]
        dilihat = []

        def latih(X, y, cfg, teknis):
            dilihat.append(sorted(x[0] for x in X))
            return latih_frekuensi(X, y, cfg, teknis)

        oof, lipatan = wf.walk_forward_model(rows, self.cfgs[:1], TEKNIS, (2019,), 1e-15,
                                             latih=latih, prediksi=prediksi_frekuensi)
        self.assertEqual(dilihat, [[0.0, 1.0]])
        self.assertEqual((lipatan[0]["n_latih"], lipatan[0]["n_purged"]), (2, 1))
        self.assertEqual([r["event_id"] for r in oof], ["e3", "e4"])

    def test_jumlah_prediksi_tidak_cocok_ditolak(self):
        with self.assertRaises(ValueError):
            wf.walk_forward_model(self.rows, self.cfgs[:1], TEKNIS, (2019,), 1e-15,
                                  latih=latih_frekuensi,
                                  prediksi=lambda m, X: [list(m) for _ in X[:-1]])

    def test_tahun_validasi_kosong_ditolak(self):
        rows = [r for r in self.rows if not r["session_date"].startswith("2020")]
        with self.assertRaises(ValueError):
            wf.walk_forward_model(rows, self.cfgs[:1], TEKNIS, (2019, 2020), 1e-15,
                                  latih=latih_frekuensi, prediksi=prediksi_frekuensi)


class TestBandingPembanding(unittest.TestCase):
    def _oof(self):
        return [{"event_id": "a", "tahun_validasi": 2019, "ll": 0.9},
                {"event_id": "b", "tahun_validasi": 2020, "ll": 1.1}]

    def _pemb(self, ids=("a", "b")):
        return [{"event_id": e, "ll_b0": "1.0", "ll_b1": "0.95"} for e in ids]

    def test_selisih_per_tahun_dan_gabungan(self):
        hasil = wf.banding_pembanding(self._oof(), self._pemb())
        self.assertAlmostEqual(hasil["gabungan"]["D_b0"], 0.0)
        self.assertAlmostEqual(hasil["gabungan"]["D_b1"], 0.05)
        self.assertAlmostEqual(hasil["per_tahun"][0]["D_b0"], -0.1)
        self.assertEqual([t["tahun"] for t in hasil["per_tahun"]], [2019, 2020])

    def test_event_tidak_cocok_ditolak(self):
        with self.assertRaises(ValueError):
            wf.banding_pembanding(self._oof(), self._pemb(("a", "c")))

    def test_label_atau_tahun_berbeda_ditolak(self):
        oof = [dict(r, label=R) for r in self._oof()]
        pemb = [dict(p, label=R, tahun_validasi=str(o["tahun_validasi"]))
                for p, o in zip(self._pemb(), oof)]
        wf.banding_pembanding(oof, pemb)                    # konsisten → lolos
        with self.assertRaises(ValueError):
            wf.banding_pembanding(oof, [dict(pemb[0], label=C), pemb[1]])
        with self.assertRaises(ValueError):
            wf.banding_pembanding(oof, [pemb[0], dict(pemb[1], tahun_validasi="2021")])


@unittest.skipUnless(ADA_LGBM, "lightgbm tidak terpasang (pip install -e '.[model]')")
class TestLightGBM(unittest.TestCase):
    def setUp(self):
        self.X = [[(i * 7) % 11 / 10, (i * 3) % 5 / 4, float(i % 2)] for i in range(300)]
        self.y = [0 if x[0] > 0.6 else (1 if x[1] > 0.4 else 2) for x in self.X]
        self.cfg = {"id": "d2_n100_m20", "max_depth": 2, "n_estimators": 100,
                    "min_child_samples": 20}

    def test_peluang_berjumlah_satu_dan_jumlah_pohon(self):
        model = wf.latih_lgbm(self.X, self.y, self.cfg, TEKNIS)
        self.assertEqual(model.num_trees(), 100 * 3)
        for p in wf.prediksi_lgbm(model, self.X[:20]):
            self.assertAlmostEqual(sum(p), 1.0)

    def test_parameter_sampai_ke_booster(self):
        cfg = {"id": "d3_n100_m50", "max_depth": 3, "n_estimators": 100, "min_child_samples": 50}
        p = wf.latih_lgbm(self.X, self.y, cfg, TEKNIS).params
        self.assertEqual((p["max_depth"], p["min_child_samples"]), (3, 50))
        self.assertEqual((p["learning_rate"], p["num_threads"]), (0.05, 1))
        self.assertTrue(p["deterministic"])
        self.assertEqual(p["seed"], 20261119)

    def test_deterministik(self):
        p1 = wf.prediksi_lgbm(wf.latih_lgbm(self.X, self.y, self.cfg, TEKNIS), self.X)
        p2 = wf.prediksi_lgbm(wf.latih_lgbm(self.X, self.y, self.cfg, TEKNIS), self.X)
        self.assertEqual(p1, p2)

    def test_kedalaman_pohon_dibatasi(self):
        model = wf.latih_lgbm(self.X, self.y, self.cfg, TEKNIS)

        def dalam(node):
            if "leaf_index" in node:
                return 0
            return 1 + max(dalam(node["left_child"]), dalam(node["right_child"]))

        trees = model.dump_model()["tree_info"]
        self.assertLessEqual(max(dalam(t["tree_structure"]) for t in trees), 2)


class TestRunWalkForward(unittest.TestCase):
    """Alur fitur -> pembanding -> walk-forward pada run sintetis (pelatih palsu)."""

    def _siapkan(self, tmp):
        paths = jalur.ModelPaths(Path(tmp) / "uji")
        make_run(paths, n_sessions=400)                        # 2016-01 s.d. 2017-07
        sumber = jalur.info_sumber(jalur.cek_data_input(paths, None))
        run_fitur(paths, KUNCI, sumber)
        return paths, sumber

    def test_menulis_keluaran(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, sumber = self._siapkan(tmp)
            pembanding.run_pembanding(paths, KUNCI, sumber, tahun_validasi=(2017,))
            cfgs = wf.konfigurasi_grid(TEKNIS)[:2]
            recon = wf.run_walk_forward(paths, KUNCI, sumber, cfgs=cfgs, tahun_validasi=(2017,),
                                        latih=latih_frekuensi, prediksi=prediksi_frekuensi)
            # pelatih frekuensi = B0, jadi kedua konfigurasi seri -> yang lebih sederhana menang
            self.assertEqual(recon["terpilih"]["id"], "d2_n100_m50")
            self.assertAlmostEqual(recon["banding_pembanding"]["gabungan"]["D_b0"], 0.0)
            self.assertEqual(len(recon["kolom_fitur"]), 21)
            oof = common.load_model_csv(paths.data / "walk_forward_oof_dev.csv")
            self.assertEqual(len(oof), 2 * recon["oof_n_per_konfigurasi"])
            saved = json.loads((paths.output / "walk_forward_rekonsiliasi.json").read_text())
            self.assertEqual(saved["terpilih"]["id"], "d2_n100_m50")
            self.assertIn("d2_n100_m50", (paths.output / "laporan_walk_forward.md").read_text())
            self.assertIn("Terpilih: d2_n100_m50", wf.ringkas(recon))

    def test_pembanding_basi_ditolak(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, sumber = self._siapkan(tmp)
            pembanding.run_pembanding(paths, KUNCI, sumber, tahun_validasi=(2017,))
            rp = paths.output / "pembanding_rekonsiliasi.json"
            recon = json.loads(rp.read_text())
            recon["sumber_sha256"]["features_dev.csv"] = "0" * 64   # fitur dihitung ulang sesudahnya
            rp.write_text(json.dumps(recon))
            with self.assertRaises(ValueError) as ctx:
                wf.run_walk_forward(paths, KUNCI, sumber, latih=latih_frekuensi,
                                    prediksi=prediksi_frekuensi)
            self.assertIn("pembanding", str(ctx.exception))
            self.assertFalse((paths.data / "walk_forward_oof_dev.csv").exists())

    def test_tanpa_pembanding_ditolak(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, sumber = self._siapkan(tmp)
            with self.assertRaises(FileNotFoundError):
                wf.run_walk_forward(paths, KUNCI, sumber, latih=latih_frekuensi,
                                    prediksi=prediksi_frekuensi)

    @unittest.skipUnless(ADA_LGBM, "lightgbm tidak terpasang (pip install -e '.[model]')")
    def test_lightgbm_sungguhan_pada_run_sintetis(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, sumber = self._siapkan(tmp)
            pembanding.run_pembanding(paths, KUNCI, sumber, tahun_validasi=(2017,))
            cfgs = wf.konfigurasi_grid(TEKNIS)[:1]
            recon = wf.run_walk_forward(paths, KUNCI, sumber, cfgs=cfgs, tahun_validasi=(2017,))
            self.assertTrue(math.isfinite(recon["skor"]["d2_n100_m20"]))
            self.assertIsNotNone(recon["lingkungan"]["lightgbm"])


if __name__ == "__main__":
    unittest.main()
