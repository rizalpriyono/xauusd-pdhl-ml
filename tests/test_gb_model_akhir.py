"""Tes tahap 4: model akhir, kalibrasi temperature scaling, parameter keputusan (RM-25 s.d. 27).

Port dari `Pemodelan PDH-PDL/tests/test_model_akhir.py` (Proposal v2 IV.II.6–IV.II.8;
RENCANA_ANALISIS_RQ2.md §2.1, §2.3, §4.3, §5). Tes tanpa LightGBM memakai model palsu (stdlib)
sehingga bisa jalan di CI; tes LightGBM dilewati bila pustakanya tidak ada, dan tes data asli
dilewati bila `runs/tesis` belum dibuat atau fiturnya dihitung dengan daftar fitur lain.
"""
import csv
import datetime as dt
import importlib.util
import json
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model.gradient_boosting import jalur, kunci  # noqa: E402
from model.gradient_boosting import model_akhir as ma  # noqa: E402
from model.gradient_boosting import walk_forward as wf  # noqa: E402
from model.gradient_boosting.fitur import FEATURE_COLS, HARI, SESI, b1_terpenuhi  # noqa: E402
from pdhl_data import common  # noqa: E402
from pdhl_data.config import load_config  # noqa: E402
from pdhl_data.pipeline import load_reference  # noqa: E402

R, C, N = common.CLASSES
ADA_LGBM = importlib.util.find_spec("lightgbm") is not None
KUNCI = kunci.muat_kunci(load_config(ROOT / "configs" / "tesis.json"), ROOT / "configs" / "kunci_1")
TEKNIS = KUNCI.teknis
A, B = KUNCI.a, KUNCI.b
LAMA = ROOT / "references" / "gradient_boosting_lama" / "model_akhir_rekonsiliasi.json"
RUN_ASLI = jalur.model_paths(load_config(ROOT / "configs" / "tesis.json"), ROOT / "runs")


def ll_manual(p, label):
    return -math.log(p[common.CLASSES.index(label)])


# ---------------------------------------------------------------- softmax dan temperature
class TestSoftmax(unittest.TestCase):
    def test_tau_satu_sama_dengan_softmax_biasa(self):
        z = [1.0, 0.0, -1.0]
        e = [math.exp(v) for v in z]
        for a, b in zip(ma.softmax(z), [v / sum(e) for v in e]):
            self.assertAlmostEqual(a, b, places=15)

    def test_stabil_untuk_logit_besar(self):
        p = ma.softmax([1000.0, 999.0, -1000.0])
        self.assertAlmostEqual(sum(p), 1.0)
        self.assertTrue(all(math.isfinite(v) for v in p))

    def test_tau_besar_mendekati_seragam_dan_urutan_tetap(self):
        z = [2.0, 0.5, -1.0]
        p1, p20 = ma.softmax(z, 1.0), ma.softmax(z, 20.0)
        self.assertGreater(min(p20), 0.3)
        urut = [sorted(range(3), key=p.__getitem__) for p in (p1, p20)]
        self.assertEqual(urut[0], urut[1])

    def test_tau_kecil_mempertajam(self):
        z = [2.0, 0.5, -1.0]
        self.assertGreater(ma.softmax(z, 0.5)[0], ma.softmax(z, 1.0)[0])

    def test_tau_tidak_positif_ditolak(self):
        for tau in (0.0, -1.0, float("nan")):
            with self.subTest(tau=tau), self.assertRaises(ValueError):
                ma.softmax([0.0, 0.0, 0.0], tau)


class TestGoldenSection(unittest.TestCase):
    def test_minimum_interior(self):
        x, fx, it = ma.golden_section(lambda t: (t - 3.0) ** 2, 0.05, 20.0, tol=1e-9)
        self.assertAlmostEqual(x, 3.0, places=7)
        self.assertAlmostEqual(fx, 0.0, places=12)
        self.assertGreater(it, 10)

    def test_fungsi_monoton_berhenti_di_batas(self):
        x, _, _ = ma.golden_section(lambda t: t, 0.05, 20.0, tol=1e-9)
        self.assertAlmostEqual(x, 0.05, places=7)
        x, _, _ = ma.golden_section(lambda t: -t, 0.05, 20.0, tol=1e-9)
        self.assertAlmostEqual(x, 20.0, places=7)

    def test_rentang_tidak_valid_ditolak(self):
        with self.assertRaises(ValueError):
            ma.golden_section(lambda t: t, 5.0, 1.0)
        with self.assertRaises(ValueError):
            ma.golden_section(lambda t: t, 1.0, 5.0, tol=0)

    def test_iterasi_maksimum(self):
        with self.assertRaises(RuntimeError):
            ma.golden_section(lambda t: t * t, -1.0, 1.0, tol=1e-12, maks_iter=5)


class TestCariTau(unittest.TestCase):
    def _data(self, skala):
        """Logit terlalu yakin: label mengikuti softmax(z0), tetapi model melaporkan skala·z0."""
        Z, Y = [], []
        pola = [([1.0, 0.2, -0.5], [R] * 5 + [C] * 3 + [N] * 2),
                ([-0.3, 1.1, 0.1], [R] * 2 + [C] * 6 + [N] * 2),
                ([0.0, 0.4, 0.6], [R] * 3 + [C] * 3 + [N] * 4)]
        for z0, labels in pola:
            for lab in labels:
                Z.append([skala * v for v in z0])
                Y.append(lab)
        return Z, Y

    def test_model_terlalu_yakin_diberi_tau_lebih_dari_satu(self):
        Z, Y = self._data(4.0)
        hasil = ma.cari_tau(Z, Y, (0.05, 20.0), 1e-15)
        self.assertGreater(hasil["tau"], 1.0)
        self.assertLess(hasil["log_loss"], ma.log_loss_tau(Z, Y, 1.0, 1e-15))
        self.assertFalse(hasil["di_batas"])

    def test_sama_dengan_minimum_grid(self):
        Z, Y = self._data(3.0)
        hasil = ma.cari_tau(Z, Y, (0.05, 20.0), 1e-15)
        grid = [0.05 * (400 ** (i / 2000)) for i in range(2001)]
        terbaik = min(ma.log_loss_tau(Z, Y, t, 1e-15) for t in grid)
        self.assertLessEqual(hasil["log_loss"], terbaik + 1e-12)
        self.assertLessEqual(hasil["log_loss"], hasil["cek_grid"]["log_loss"] + 1e-12)

    def test_log_loss_tau_manual(self):
        Z, Y = [[0.0, 0.0, 0.0], [math.log(2), 0.0, 0.0]], [R, R]
        self.assertAlmostEqual(ma.log_loss_tau(Z, Y, 1.0, 1e-15),
                               (math.log(3) + math.log(2)) / 2)

    def test_minimum_di_batas_ditandai(self):
        # Label selalu kelas dengan logit terbesar → τ sekecil mungkin.
        Z, Y = [[1.0, 0.0, 0.0]] * 5, [R] * 5
        hasil = ma.cari_tau(Z, Y, (0.05, 20.0), 1e-15)
        self.assertTrue(hasil["di_batas"])
        self.assertAlmostEqual(hasil["tau"], 0.05, places=5)

    def test_data_kosong_ditolak(self):
        with self.assertRaises(ValueError):
            ma.cari_tau([], [], (0.05, 20.0), 1e-15)

    def test_hasil_lebih_buruk_dari_grid_ditolak(self):
        Z, Y = self._data(3.0)
        buruk = (19.0, ma.log_loss_tau(Z, Y, 19.0, 1e-15), 1)   # seolah terjebak minimum lokal
        with mock.patch.object(ma, "golden_section", return_value=buruk), \
                self.assertRaises(RuntimeError):
            ma.cari_tau(Z, Y, (0.05, 20.0), 1e-15)


# ---------------------------------------------------------------- metrik peluang
class TestMetrik(unittest.TestCase):
    P = [[0.05, 0.9, 0.05], [0.15, 0.8, 0.05], [0.12, 0.08, 0.8], [1.0, 0.0, 0.0]]
    Y = [R, C, N, R]

    def test_brier_sempurna_dan_terburuk(self):
        self.assertEqual(ma.brier([[1.0, 0.0, 0.0]], [R]), 0.0)
        self.assertEqual(ma.brier([[0.0, 1.0, 0.0]], [R]), 2.0)

    def test_brier_manual(self):
        # (0.5−1)² + 0.3² + 0.2² = 0.38
        self.assertAlmostEqual(ma.brier([[0.5, 0.3, 0.2]], [R]), 0.38)

    def test_reliabilitas_bin_dan_jumlah(self):
        rel = ma.reliabilitas(self.P, self.Y, 10)
        self.assertEqual(len(rel[R]), 10)
        for k in common.CLASSES:
            self.assertEqual(sum(b["n"] for b in rel[k]), 4)
        self.assertEqual(rel[R][9]["n"], 1)                    # p = 1,0 masuk bin terakhir
        self.assertEqual(rel[R][1]["n"], 2)                    # 0,15 dan 0,12
        self.assertAlmostEqual(rel[R][1]["p_rerata"], 0.135)
        self.assertAlmostEqual(rel[R][1]["frek"], 0.0)
        self.assertIsNone(rel[R][5]["p_rerata"])               # bin kosong

    def test_ece_manual(self):
        # Kelas R: bin0 {0,05 → y=1} |0,05−1|=0,95; bin1 {0,15 → 0, 0,12 → 0} |0,135−0|; bin9 0.
        self.assertAlmostEqual(ma.ece(self.P, self.Y, 10)[R], (1 * 0.95 + 2 * 0.135 + 0.0) / 4)

    def test_ece_nol_bila_terkalibrasi_sempurna(self):
        P, Y = [[0.5, 0.5, 0.0]] * 4, [R, C, R, C]
        self.assertAlmostEqual(ma.ece(P, Y, 10)[R], 0.0)
        self.assertAlmostEqual(ma.ece(P, Y, 10)[N], 0.0)

    def test_log_loss_per_level(self):
        rows = [{"label": R, "level": "PDH"}, {"label": C, "level": "PDL"},
                {"label": N, "level": "PDL"}]
        P = [[0.5, 0.3, 0.2], [0.2, 0.6, 0.2], [0.25, 0.25, 0.5]]
        h = ma.log_loss_per_level(rows, {"m": P}, 1e-15)
        self.assertEqual((h["PDH"]["n"], h["PDL"]["n"]), (1, 2))
        self.assertAlmostEqual(h["PDH"]["m"], -math.log(0.5))
        self.assertAlmostEqual(h["PDL"]["m"], (-math.log(0.6) - math.log(0.5)) / 2)

    def test_ringkas_peluang(self):
        P, Y = [[0.5, 0.3, 0.2], [0.2, 0.6, 0.2]], [R, C]
        s = ma.ringkas_peluang(P, Y, 1e-15, 10)
        self.assertEqual(s["n"], 2)
        self.assertAlmostEqual(s["log_loss"], (ll_manual(P[0], R) + ll_manual(P[1], C)) / 2)
        self.assertAlmostEqual(s["brier"], (0.38 + 0.24) / 2)
        self.assertEqual(set(s["ece"]), set(common.CLASSES))
        self.assertNotIn("reliabilitas", ma.ringkas_peluang(P, Y, 1e-15, 10, False))


# ---------------------------------------------------------------- m_rev, m_cont, aturan keputusan
def baris_r(label, r_rev, r_cont, i=0, tanggal="2020-01-06"):
    return {"event_id": f"e{i}", "session_date": tanggal, "label": label,
            "R_rev": r_rev, "R_cont": r_cont}


class TestMNetral(unittest.TestCase):
    def test_rata_rata_hanya_event_netral(self):
        rows = [baris_r(R, 2.0, -1.0), baris_r(C, -1.0, 0.5),
                baris_r(N, 0.4, -0.2), baris_r(N, -0.6, 0.3)]
        m = ma.m_netral(rows, A, B)
        self.assertEqual(m["n_netral"], 2)
        self.assertAlmostEqual(m["m_rev"], -0.1)
        self.assertAlmostEqual(m["m_cont"], 0.05)

    def test_r_netral_tidak_konsisten_ditolak(self):
        # R_cont harus = −R_rev · b/a untuk event Netral.
        with self.assertRaises(ValueError):
            ma.m_netral([baris_r(N, 0.4, 0.2)], A, B)

    def test_r_netral_di_luar_rentang_ditolak(self):
        with self.assertRaises(ValueError):
            ma.m_netral([baris_r(N, 2.2, -1.1)], A, B)

    def test_r_barrier_tidak_sesuai_tabel_ditolak(self):
        with self.assertRaises(ValueError):
            ma.m_netral([baris_r(R, 1.5, -1.0), baris_r(N, 0.4, -0.2)], A, B)
        with self.assertRaises(ValueError):
            ma.m_netral([baris_r(C, -1.0, 0.4), baris_r(N, 0.4, -0.2)], A, B)

    def test_tanpa_netral_ditolak(self):
        with self.assertRaises(ValueError):
            ma.m_netral([baris_r(R, 2.0, -1.0)], A, B)


class TestKeputusan(unittest.TestCase):
    M = {"m_rev": -0.1, "m_cont": 0.05}

    def test_nilai_harapan_manual(self):
        e_rev, e_cont = ma.nilai_harapan([0.4, 0.4, 0.2], A, B, self.M)
        self.assertAlmostEqual(e_rev, 0.4 * 2 - 0.4 + 0.2 * -0.1)     # 0,38
        self.assertAlmostEqual(e_cont, 0.4 * 0.5 - 0.4 + 0.2 * 0.05)  # −0,19

    def test_e_cont_sama_dengan_minus_b_per_a_kali_e_rev(self):
        # R_cont = −(b/a)·R_rev di setiap event; bila m_cont = −(b/a)·m_rev,
        # maka E(cont) = −(b/a)·E(rev).
        m = {"m_rev": 0.5627, "m_cont": -0.5627 * B / A}
        for p in ([0.2249, 0.566, 0.2091], [0.5, 0.2, 0.3], [0.1, 0.8, 0.1]):
            e_rev, e_cont = ma.nilai_harapan(p, A, B, m)
            self.assertAlmostEqual(e_cont, -B / A * e_rev, places=12)

    def test_base_rate_memberi_abstain_pada_delta_utama(self):
        # Base rate dev ≈ (0,225; 0,566; 0,209) → E(rev) ≈ −0,13, E(cont) ≈ 0,07: keduanya < 0,1.
        e = ma.nilai_harapan([0.2249, 0.5660, 0.2091], A, B, self.M)
        self.assertEqual(ma.rekomendasi(*e, KUNCI.delta), ma.ABSTAIN)

    def test_rekomendasi(self):
        self.assertEqual(ma.rekomendasi(0.38, -0.19, 0.1), R)
        self.assertEqual(ma.rekomendasi(-0.5, 0.12, 0.1), C)
        self.assertEqual(ma.rekomendasi(0.1, 0.05, 0.1), R)            # tepat δ → rekomendasi
        self.assertEqual(ma.rekomendasi(0.09, 0.05, 0.1), ma.ABSTAIN)
        self.assertEqual(ma.rekomendasi(0.2, 0.2, 0.1), ma.ABSTAIN)    # seri persis → abstain

    def test_r_kotor(self):
        row = baris_r(N, 0.4, -0.2)
        self.assertEqual(ma.r_kotor(row, R), 0.4)
        self.assertEqual(ma.r_kotor(row, C), -0.2)
        self.assertEqual(ma.r_kotor(row, ma.ABSTAIN), 0.0)

    def test_pratinjau_keputusan(self):
        rows = [baris_r(R, 2.0, -1.0, 0), baris_r(C, -1.0, 0.5, 1), baris_r(N, 0.4, -0.2, 2)]
        P = [[0.6, 0.2, 0.2], [0.05, 0.9, 0.05], [0.2, 0.6, 0.2]]
        h = {x["delta"]: x for x in ma.pratinjau_keputusan(rows, P, A, B, self.M, [0.0, 0.1])}
        # δ = 0,1: e0 E(rev)=0,98 → R (+2); e1 E(cont)=0,4 → C (+0,5); e2 E(cont)=0,11 → C (−0,2).
        self.assertEqual((h[0.1]["n_rev"], h[0.1]["n_cont"], h[0.1]["n_abstain"]), (1, 2, 0))
        self.assertAlmostEqual(h[0.1]["r_per_event"], (2.0 + 0.5 - 0.2) / 3)
        self.assertAlmostEqual(h[0.1]["r_per_rekomendasi"], (2.0 + 0.5 - 0.2) / 3)
        self.assertAlmostEqual(h[0.1]["cakupan"], 1.0)

    def test_pratinjau_tanpa_rekomendasi(self):
        hasil = ma.pratinjau_keputusan([baris_r(N, 0.4, -0.2)], [[0.2249, 0.566, 0.2091]],
                                       A, B, self.M, [0.1])
        self.assertEqual(hasil[0]["cakupan"], 0.0)
        self.assertIsNone(hasil[0]["r_per_rekomendasi"])

    def test_gabung_oof(self):
        dev = [baris_r(R, 2.0, -1.0, 0), baris_r(C, -1.0, 0.5, 1), baris_r(N, 0.4, -0.2, 2)]
        oof = [{"event_id": "e2", "label": N, "konfigurasi": "k", "p_rev": "0.2", "p_cont": "0.5",
                "p_net": "0.3"},
               {"event_id": "e0", "label": R, "konfigurasi": "lain", "p_rev": "1", "p_cont": "0",
                "p_net": "0"},
               {"event_id": "e0", "label": R, "konfigurasi": "k", "p_rev": "0.6", "p_cont": "0.3",
                "p_net": "0.1"}]
        rows, P = ma.gabung_oof(oof, dev, "k")
        self.assertEqual([r["event_id"] for r in rows], ["e2", "e0"])
        self.assertEqual(P, [[0.2, 0.5, 0.3], [0.6, 0.3, 0.1]])
        with self.assertRaises(ValueError):                       # label berbeda
            ma.gabung_oof([dict(oof[0], label=R)], dev, "k")
        with self.assertRaises(ValueError):                       # event tidak dikenal
            ma.gabung_oof([dict(oof[0], event_id="zz")], dev, "k")
        with self.assertRaises(ValueError):                       # konfigurasi tidak ada
            ma.gabung_oof(oof, dev, "tidak_ada")
        with self.assertRaises(ValueError):                       # ganda
            ma.gabung_oof([oof[0], oof[0]], dev, "k")

    def test_strategi_acuan_s0_s1(self):
        rows = [dict(baris_r(R, 2.0, -1.0, 0), b1=True), dict(baris_r(C, -1.0, 0.5, 1), b1=False),
                dict(baris_r(N, 0.4, -0.2, 2), b1=True)]
        s = ma.strategi_acuan(rows)
        self.assertAlmostEqual(s["S0"]["r_per_event"], (2.0 - 1.0 + 0.4) / 3)
        self.assertEqual(s["S1"]["n_rev"], 2)
        self.assertAlmostEqual(s["S1"]["r_per_event"], (2.0 + 0.4) / 3)
        self.assertAlmostEqual(s["S1"]["r_per_rekomendasi"], 1.2)


# ---------------------------------------------------------------- data dan konfigurasi
class TestKonfigurasiTerpilih(unittest.TestCase):
    SHA = {"features_dev.csv": "aa", "labels_dev.csv": "bb"}

    def _recon(self, **ubah):
        return {"terpilih": {"id": "d2_n100_m50", "max_depth": 2, "n_estimators": 100,
                             "min_child_samples": 50},
                "sumber_sha256": {**self.SHA, "pembanding_oof_dev.csv": "cc"}, **ubah}

    def test_konfigurasi_dari_walk_forward(self):
        self.assertEqual(ma.konfigurasi_terpilih(self._recon(), TEKNIS, self.SHA)["id"],
                         "d2_n100_m50")

    def test_konfigurasi_di_luar_grid_ditolak(self):
        recon = self._recon(terpilih={"id": "d4_n100_m50", "max_depth": 4, "n_estimators": 100,
                                      "min_child_samples": 50})
        with self.assertRaises(ValueError):
            ma.konfigurasi_terpilih(recon, TEKNIS, self.SHA)

    def test_id_tidak_sesuai_parameter_ditolak(self):
        recon = self._recon(terpilih={"id": "d2_n100_m50", "max_depth": 3, "n_estimators": 100,
                                      "min_child_samples": 50})
        with self.assertRaises(ValueError):
            ma.konfigurasi_terpilih(recon, TEKNIS, self.SHA)

    def test_skor_harus_memilih_konfigurasi_yang_sama(self):
        skor = {c["id"]: 1.0 for c in wf.konfigurasi_grid(TEKNIS)}
        skor["d2_n100_m50"] = 0.9
        ma.konfigurasi_terpilih(self._recon(skor=skor), TEKNIS, self.SHA)
        skor["d3_n300_m20"] = 0.8
        with self.assertRaises(ValueError):
            ma.konfigurasi_terpilih(self._recon(skor=skor), TEKNIS, self.SHA)

    def test_data_dev_berubah_sejak_walk_forward_ditolak(self):
        with self.assertRaises(ValueError):
            ma.konfigurasi_terpilih(self._recon(), TEKNIS, {**self.SHA, "labels_dev.csv": "zz"})


class TestRentangSplit(unittest.TestCase):
    def test_dev_dan_calib(self):
        ma.cek_rentang([{"event_id": "a", "session_date": "2016-01-04"},
                        {"event_id": "b", "session_date": "2023-12-29"}], "dev")
        ma.cek_rentang([{"event_id": "c", "session_date": "2024-01-02"}], "calib")

    def test_baris_di_luar_split_ditolak(self):
        with self.assertRaises(ValueError):
            ma.cek_rentang([{"event_id": "a", "session_date": "2024-01-02"}], "dev")
        with self.assertRaises(ValueError):
            ma.cek_rentang([{"event_id": "a", "session_date": "2023-12-29"}], "calib")

    def test_split_kosong_ditolak(self):
        with self.assertRaises(ValueError):
            ma.cek_rentang([], "calib")

    def test_data_uji_ditolak_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "features_calib.csv"
            p.write_text("event_id,session_date\ne1,2025-01-06\n")
            with self.assertRaises(common.TestDataLeak):
                ma.muat_split(p, p, "calib", KUNCI.theta)

    def test_data_uji_di_berkas_label_ditolak(self):
        with tempfile.TemporaryDirectory() as tmp:
            fit, lab = Path(tmp) / "features_calib.csv", Path(tmp) / "labels_calib.csv"
            fit.write_text("event_id,session_date\ne1,2024-01-02\n")
            lab.write_text("event_id,session_date\ne1,2025-01-06\n")
            with self.assertRaises(common.TestDataLeak):
                ma.muat_split(fit, lab, "calib", KUNCI.theta)

    def test_folder_kunci_uji_ditolak(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "kunci_uji" / "events_test.csv"
            p.parent.mkdir()
            p.write_text("event_id,session_date\ne1,2024-01-02\n")
            with self.assertRaises(common.TestDataLeak):
                ma.muat_split(p, p, "calib", KUNCI.theta)


class TestLatihAkhir(unittest.TestCase):
    def test_purging_di_batas_dev_kalibrasi(self):
        rows = [{"event_id": "a", "session_date": "2023-12-28", "label": R,
                 "label_end_session_date": "2023-12-29", "x": [0.0]},
                {"event_id": "b", "session_date": "2023-12-29", "label": C,
                 "label_end_session_date": "2024-01-02", "x": [1.0]}]
        dipanggil = {}

        def latih(X, y, cfg, teknis, nama):
            dipanggil.update(X=X, y=y, nama=nama)
            return "model"

        model, latih_rows, purged = ma.latih_akhir(rows, {"id": "x"}, TEKNIS, ["f"], latih=latih)
        self.assertEqual(model, "model")
        self.assertEqual([r["event_id"] for r in latih_rows], ["a"])
        self.assertEqual([r["event_id"] for r in purged], ["b"])
        self.assertEqual(dipanggil["y"], [0])
        self.assertEqual(dipanggil["nama"], ["f"])


# ---------------------------------------------------------------- alur tahap dengan model palsu
CFG_ID = "d2_n100_m50"
FITUR_COLS = ["event_id", "session_date", "level_type"] + FEATURE_COLS + ["b1_terpenuhi"]
LABEL_COLS = ["event_id", "session_date", "label", "label_end_session_date", "R_rev", "R_cont"]


TANGGAL_PURGE = dt.date(2023, 12, 29)        # jendela label event ini melewati 2024-01-01


def tanggal_event(awal=dt.date(2016, 1, 4), akhir=dt.date(2024, 12, 27), langkah=9):
    """Hari kerja kira-kira tiap `langkah` hari, ditambah TANGGAL_PURGE."""
    out, d = {TANGGAL_PURGE}, awal
    while d <= akhir:
        out.add(d if d.weekday() < 5 else d + dt.timedelta(days=7 - d.weekday()))
        d += dt.timedelta(days=langkah)
    return sorted(out)


def buat_run_sintetis(paths):
    """Masukan tahap model akhir tanpa menjalankan tahap sebelumnya: fitur, label ber-R, OOF
    walk-forward, dan rekonsiliasi walk-forward. Satu event di 2023-12-29 di-purge."""
    fitur, label = {"dev": [], "calib": []}, {"dev": [], "calib": []}
    for i, d in enumerate(tanggal_event()):
        sd, lv = d.isoformat(), ("PDH", "PDL")[i % 2]
        f13 = (i * 0.37) % 1
        row = {"event_id": f"{sd}_{lv}", "session_date": sd, "level_type": lv,
               **{c: float((i * 7 + j) % 11) / 10 for j, c in enumerate(FEATURE_COLS)},
               "f01_jenis_level": 1 if lv == "PDH" else 0, "f04_sesi": SESI[i % 3],
               "f05_hari": HARI[d.weekday()], "f13_wick_penolakan": f13,
               "b1_terpenuhi": b1_terpenuhi(f13, KUNCI.theta)}
        kelas = (R, C, C, N, C, R, N)[i % 7]
        if kelas == R:
            r_rev, r_cont = A / B, -1.0
        elif kelas == C:
            r_rev, r_cont = -1.0, B / A
        else:                                      # Netral: −1 < R_rev < a/b, R_cont = −(b/a)·R_rev
            r_rev = ((i * 0.53) % 2.6) - 0.95
            r_cont = -r_rev * B / A
        akhir = "2024-01-02" if d == TANGGAL_PURGE else sd
        split = common.split_of(sd)
        fitur[split].append(row)
        label[split].append({"event_id": row["event_id"], "session_date": sd, "label": kelas,
                             "label_end_session_date": akhir, "R_rev": r_rev, "R_cont": r_cont})
    for s in ("dev", "calib"):
        common.write_csv(paths.data / f"features_{s}.csv", fitur[s], FITUR_COLS)
        common.write_csv(paths.input / f"labels_{s}.csv", label[s], LABEL_COLS)
    oof = []
    for lab in label["dev"]:
        if lab["session_date"] >= "2019-01-01":
            for cfg in (CFG_ID, "d2_n100_m20"):
                oof.append({"event_id": lab["event_id"], "session_date": lab["session_date"],
                            "tahun_validasi": int(lab["session_date"][:4]), "label": lab["label"],
                            "konfigurasi": cfg, "p_rev": 0.3, "p_cont": 0.5, "p_net": 0.2,
                            "ll": 1.0})
    common.write_csv(paths.data / "walk_forward_oof_dev.csv", oof, wf.OOF_COLS)
    cfgs = wf.konfigurasi_grid(TEKNIS)
    skor = {c["id"]: 1.0 for c in cfgs}
    skor[CFG_ID] = 0.9
    sha = {"features_dev.csv": common.sha256_file(paths.data / "features_dev.csv"),
           "labels_dev.csv": common.sha256_file(paths.input / "labels_dev.csv")}
    recon = {"terpilih": next(c for c in cfgs if c["id"] == CFG_ID), "skor": skor,
             "sumber_sha256": sha}
    paths.output.mkdir(parents=True, exist_ok=True)
    (paths.output / "walk_forward_rekonsiliasi.json").write_text(json.dumps(recon))
    return len(label["dev"]), len(label["calib"]), sum(o["konfigurasi"] == CFG_ID for o in oof)


class ModelPalsu:
    """Pengganti Booster: logit = log frekuensi kelas × 2 + sedikit pengaruh fitur pertama."""

    def __init__(self, y):
        self.freq = [(y.count(k) + 1) / (len(y) + 3) for k in range(len(common.CLASSES))]

    def model_to_string(self):
        return "model palsu " + json.dumps(self.freq)

    def num_trees(self):
        return 0


def latih_palsu(X, y, cfg, teknis, nama):
    return ModelPalsu(list(y))


def skor_palsu(model, X):
    return [[2 * math.log(f) + (0.3 * x[0] if j == 0 else 0.0) for j, f in enumerate(model.freq)]
            for x in X]


def prediksi_palsu(model, X):
    return [ma.softmax(z) for z in skor_palsu(model, X)]


class TestRunModelAkhir(unittest.TestCase):
    SUMBER = {"versi_pdhl_data": "uji", "commit": None, "sumber_sha256": {}}

    def _jalankan(self, paths, **kw):
        opsi = {"latih": latih_palsu, "skor": skor_palsu, "prediksi": prediksi_palsu, **kw}
        return ma.run_model_akhir(paths, KUNCI, self.SUMBER, **opsi)

    def test_menulis_keluaran_konsisten(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = jalur.ModelPaths(Path(tmp) / "uji")
            n_dev, n_cal, n_oof = buat_run_sintetis(paths)
            recon = self._jalankan(paths)
            self.assertEqual(recon["konfigurasi"]["id"], CFG_ID)
            self.assertEqual((recon["n_latih"], recon["n_purged"], recon["n_calib"]),
                             (n_dev - 1, 1, n_cal))
            self.assertEqual(recon["kunci1_sha256_sesuai_manifest"],
                             {n: KUNCI.manifest_sha256[n] for n in ma.BERKAS_KUNCI1})
            m = recon["aturan_keputusan"]
            self.assertEqual((m["a"], m["b"], m["delta"]), (2.0, 1.0, 0.1))
            self.assertEqual(m["delta_grid"], [0.0, 0.1, 0.2, 0.3])
            self.assertAlmostEqual(m["m_cont"], -m["m_rev"] * B / A, places=9)
            self.assertEqual(recon["cakupan_oof_dev"]["S0"]["n"], n_oof)
            self.assertEqual([h["delta"] for h in recon["pratinjau_calib"]["SM"]],
                             [0.0, 0.1, 0.2, 0.3])
            k = recon["kalibrasi"]
            self.assertLessEqual(k["sesudah"]["log_loss"], k["sebelum"]["log_loss"])

            teks_model = (paths.model / ma.MODEL_BERKAS).read_bytes()
            self.assertEqual(common.sha256_bytes(teks_model), recon["model_sha256"])
            saved = json.loads((paths.output / "model_akhir_rekonsiliasi.json").read_text())
            self.assertEqual(saved["model_sha256"], recon["model_sha256"])
            self.assertIn("τ", (paths.output / "laporan_model_akhir.md").read_text())
            self.assertTrue((paths.output / "gambar_reliabilitas_2024.svg").read_text()
                            .startswith("<svg"))
            self.assertIn(f"Model akhir {CFG_ID}", ma.ringkas(recon))

            rows = common.load_model_csv(paths.data / "model_akhir_calib.csv")
            self.assertEqual(len(rows), n_cal)
            cal = ma.muat_split(paths.data / "features_calib.csv", paths.input / "labels_calib.csv",
                                "calib", KUNCI.theta)
            for r, row in zip(cal, rows):
                z = [float(row[f"z_{s}"]) for s in ("rev", "cont", "net")]
                p = ma.softmax(z, k["tau"])
                self.assertAlmostEqual(float(row["p_rev"]), p[0], places=9)
                e = ma.nilai_harapan(p, A, B, m)
                self.assertAlmostEqual(float(row["E_rev"]), e[0], places=8)
                self.assertEqual(row["rekomendasi"], ma.rekomendasi(*e, m["delta"]))
                self.assertAlmostEqual(float(row["R_kotor"]), ma.r_kotor(r, row["rekomendasi"]),
                                       places=9)

    def test_tanpa_walk_forward_ditolak(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = jalur.ModelPaths(Path(tmp) / "uji")
            buat_run_sintetis(paths)
            (paths.output / "walk_forward_rekonsiliasi.json").unlink()
            with self.assertRaises(FileNotFoundError) as ctx:
                self._jalankan(paths)
            self.assertIn("walk-forward", str(ctx.exception))
            self.assertFalse(paths.model.exists())

    def test_data_dev_berubah_ditolak_tanpa_menulis(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = jalur.ModelPaths(Path(tmp) / "uji")
            buat_run_sintetis(paths)
            rp = paths.output / "walk_forward_rekonsiliasi.json"
            recon = json.loads(rp.read_text())
            recon["sumber_sha256"]["features_dev.csv"] = "0" * 64
            rp.write_text(json.dumps(recon))
            with self.assertRaises(ValueError):
                self._jalankan(paths)
            self.assertFalse(paths.model.exists())
            self.assertFalse((paths.output / "model_akhir_rekonsiliasi.json").exists())

    def test_prediksi_tidak_konsisten_ditolak(self):
        def prediksi_geser(model, X):
            return [[p[0] + 1e-6, p[1] - 1e-6, p[2]] for p in prediksi_palsu(model, X)]

        with tempfile.TemporaryDirectory() as tmp:
            paths = jalur.ModelPaths(Path(tmp) / "uji")
            buat_run_sintetis(paths)
            with self.assertRaises(RuntimeError):
                self._jalankan(paths, prediksi=prediksi_geser)
            self.assertFalse(paths.model.exists())

    def test_r_tidak_sesuai_tabel_ditolak(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = jalur.ModelPaths(Path(tmp) / "uji")
            buat_run_sintetis(paths)
            p = paths.input / "labels_calib.csv"
            rows = common.read_csv(p)
            i = next(j for j, r in enumerate(rows) if r["label"] == R)
            rows[i]["R_rev"] = "1.5"
            common.write_csv(p, rows, LABEL_COLS)
            with self.assertRaises(ValueError):
                self._jalankan(paths)
            self.assertFalse(paths.model.exists())
            self.assertFalse((paths.output / "model_akhir_rekonsiliasi.json").exists())

    def test_skor_walk_forward_hilang_ditolak(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = jalur.ModelPaths(Path(tmp) / "uji")
            buat_run_sintetis(paths)
            rp = paths.output / "walk_forward_rekonsiliasi.json"
            recon = json.loads(rp.read_text())
            del recon["skor"]
            rp.write_text(json.dumps(recon))
            with self.assertRaises(ValueError) as ctx:
                self._jalankan(paths)
            self.assertIn("skor", str(ctx.exception))
            self.assertFalse(paths.model.exists())

    def test_label_hilang_memberi_petunjuk_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = jalur.ModelPaths(Path(tmp) / "uji")
            buat_run_sintetis(paths)
            (paths.input / "labels_calib.csv").unlink()
            with self.assertRaises(FileNotFoundError) as ctx:
                self._jalankan(paths)
            self.assertIn("pdhl_data run", str(ctx.exception))


# ---------------------------------------------------------------- LightGBM
@unittest.skipUnless(ADA_LGBM, "lightgbm tidak terpasang (pip install -e '.[model]')")
class TestLightGBM(unittest.TestCase):
    def setUp(self):
        self.X = [[(i * 7) % 11 / 10, (i * 3) % 5 / 4, float(i % 2)] for i in range(300)]
        self.y = [0 if x[0] > 0.6 else (1 if x[1] > 0.4 else 2) for x in self.X]
        self.cfg = {"id": "d2_n100_m50", "max_depth": 2, "n_estimators": 100,
                    "min_child_samples": 50}
        self.nama = ["a=1", "b", "c"]

    def test_nama_fitur_tidak_mengubah_model(self):
        m1 = wf.latih_lgbm(self.X, self.y, self.cfg, TEKNIS)
        m2 = ma.latih_lgbm(self.X, self.y, self.cfg, TEKNIS, self.nama)
        self.assertEqual(wf.prediksi_lgbm(m1, self.X), wf.prediksi_lgbm(m2, self.X))
        self.assertEqual(m2.feature_name(), self.nama)

    def test_softmax_skor_mentah_sama_dengan_predict(self):
        m = ma.latih_lgbm(self.X, self.y, self.cfg, TEKNIS, self.nama)
        for z, p in zip(ma.skor_mentah(m, self.X), wf.prediksi_lgbm(m, self.X)):
            for a, b in zip(ma.softmax(z), p):
                self.assertAlmostEqual(a, b, places=12)

    def test_simpan_muat_model_identik(self):
        import lightgbm as lgb
        m = ma.latih_lgbm(self.X, self.y, self.cfg, TEKNIS, self.nama)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "m.txt"
            p.write_text(m.model_to_string())
            m2 = lgb.Booster(model_file=str(p))
            self.assertEqual(ma.skor_mentah(m, self.X), ma.skor_mentah(m2, self.X))
            self.assertEqual(m2.feature_name(), self.nama)

    def test_teks_model_deterministik(self):
        t1 = ma.latih_lgbm(self.X, self.y, self.cfg, TEKNIS, self.nama).model_to_string()
        t2 = ma.latih_lgbm(self.X, self.y, self.cfg, TEKNIS, self.nama).model_to_string()
        self.assertEqual(t1, t2)

    def test_lightgbm_sungguhan_pada_run_sintetis(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = jalur.ModelPaths(Path(tmp) / "uji")
            n_dev, _, _ = buat_run_sintetis(paths)
            recon = ma.run_model_akhir(paths, KUNCI, TestRunModelAkhir.SUMBER)
            self.assertEqual(recon["jumlah_pohon"], 100 * 3)
            self.assertEqual(recon["n_latih"], n_dev - 1)
            self.assertIsNotNone(recon["lingkungan"]["lightgbm"])


# ---------------------------------------------------------------- data asli
BERKAS_ASLI = (RUN_ASLI.data / "features_dev.csv", RUN_ASLI.data / "features_calib.csv",
               RUN_ASLI.data / "walk_forward_oof_dev.csv",
               RUN_ASLI.output / "walk_forward_rekonsiliasi.json",
               RUN_ASLI.input / "labels_dev.csv", RUN_ASLI.input / "labels_calib.csv")


def fitur_asli_sesuai_kode():
    """`features_dev.csv` di runs/tesis memuat semua kolom fitur yang dipakai kode sekarang."""
    with open(BERKAS_ASLI[0], newline="") as f:
        header = next(csv.reader(f), [])
    return set(FEATURE_COLS) <= set(header)


def run_asli_sesuai_acuan():
    """runs/tesis lengkap sampai walk-forward, data pipelinenya = references/tesis.json, dan
    fiturnya dihitung dengan daftar fitur kode sekarang.

    Bila data dibuat ulang dengan versi lain, atau fitur di runs/tesis masih dari daftar fitur lama
    (sebelum Kunci-1 Adendum 4) karena tahap fitur belum dijalankan ulang, tes data asli dilewati,
    bukan gagal."""
    if not all(p.exists() for p in BERKAS_ASLI):
        return False
    try:
        jalur.cek_data_input(RUN_ASLI, load_reference(ROOT / "references" / "tesis.json"))
    except (FileNotFoundError, jalur.DataInputBerbeda):
        return False
    return fitur_asli_sesuai_kode()


@unittest.skipUnless(ADA_LGBM and run_asli_sesuai_acuan(),
                     "butuh lightgbm dan runs/tesis (sesuai references/tesis.json, fitur sesuai "
                     "kode sekarang) sampai tahap walk-forward")
class TestDataAsli(unittest.TestCase):
    """Model akhir dari data asli (salinan di folder sementara) dibandingkan dengan hasil lama.

    Catatan Adendum 4: angka acuan di kelas ini (konfigurasi terpilih, τ, log loss) berasal dari
    run 24 November 2026 dengan fitur 14-15 lama. Setelah tahap fitur sampai model akhir dijalankan
    ulang dengan fitur Adendum 4, angka yang bergantung pada fitur perlu diperbarui dari hasil baru;
    m_rev, m_cont, dan jumlah event tidak bergantung pada fitur."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.paths = jalur.ModelPaths(Path(cls.tmp.name) / "tesis")
        for p in BERKAS_ASLI:
            tujuan = cls.paths.run_root / p.relative_to(RUN_ASLI.run_root)
            tujuan.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, tujuan)
        cls.recon = ma.run_model_akhir(cls.paths, KUNCI, TestRunModelAkhir.SUMBER)
        cls.lama = json.loads(LAMA.read_text())

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_model_dan_data(self):
        r = self.recon
        self.assertEqual(r["konfigurasi"]["id"], "d2_n100_m50")
        self.assertEqual((r["n_latih"], r["n_purged"], r["n_calib"]), (1841, 0, 227))
        self.assertEqual(r["jumlah_pohon"], 300)

    def test_angka_utama_sama_dengan_hasil_lama(self):
        r, lama = self.recon, self.lama
        for jalan in (("kalibrasi", "tau"), ("kalibrasi", "sesudah", "log_loss"),
                      ("aturan_keputusan", "m_rev"), ("aturan_keputusan", "m_cont"),
                      ("dev_dalam_sampel", "log_loss")):
            a, b = r, lama
            for k in jalan:
                a, b = a[k], b[k]
            self.assertAlmostEqual(a, b, delta=1e-9 * max(1.0, abs(b)), msg=".".join(jalan))
        self.assertEqual(r["aturan_keputusan"]["n_netral"], 385)

    def test_berkas_model_dapat_dimuat_ulang(self):
        import lightgbm as lgb
        p = self.paths.model / ma.MODEL_BERKAS
        self.assertEqual(common.sha256_file(p), self.recon["model_sha256"])
        cal = ma.muat_split(self.paths.data / "features_calib.csv",
                            self.paths.input / "labels_calib.csv", "calib", KUNCI.theta)
        Z = ma.skor_mentah(lgb.Booster(model_file=str(p)), [r["x"] for r in cal])
        rows = common.load_model_csv(self.paths.data / "model_akhir_calib.csv")
        self.assertEqual(len(rows), 227)
        tau, m = self.recon["kalibrasi"]["tau"], self.recon["aturan_keputusan"]
        for r, z, row in zip(cal, Z, rows):
            self.assertEqual((row["event_id"], row["label"]), (r["event_id"], r["label"]))
            pm, p = ma.softmax(z), ma.softmax(z, tau)
            for j, s in enumerate(("rev", "cont", "net")):
                self.assertAlmostEqual(float(row[f"z_{s}"]), z[j], places=9)
                self.assertAlmostEqual(float(row[f"p_{s}_mentah"]), pm[j], places=9)
                self.assertAlmostEqual(float(row[f"p_{s}"]), p[j], places=9)
            e = ma.nilai_harapan(p, A, B, m)
            self.assertAlmostEqual(float(row["E_rev"]), e[0], places=9)
            self.assertEqual(row["rekomendasi"], ma.rekomendasi(*e, m["delta"]))

    def test_cakupan_oof_dev(self):
        c = self.recon["cakupan_oof_dev"]
        self.assertEqual(c["S0"]["n"], 1152)
        self.assertEqual(c["SM"][0]["cakupan"], 1.0)
        cak = [h["cakupan"] for h in c["SM"]]
        self.assertEqual(cak, sorted(cak, reverse=True))          # δ naik → cakupan turun


if __name__ == "__main__":
    unittest.main()
