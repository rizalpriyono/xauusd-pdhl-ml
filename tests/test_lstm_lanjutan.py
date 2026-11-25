"""Unit test tahap lanjutan LSTM: RM-25 (model akhir), RM-26 (kalibrasi τ), RM-27 (keputusan), & Kunci-2.

Seluruh tes logika matematika, aturan keputusan, golden-section search, dan segel Kunci-2 berjalan
murni dengan stdlib Python. Tes serialisasi checkpoint PyTorch `.pt` berjalan otomatis bila `torch`
terpasang (`pip install -e '.[lstm]'`).
"""
import datetime as dt
import importlib.util
import json
import math
import os
import random
import stat
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gb_sintetis import BAR_COLS, EVENT_COLS, make_bars, make_event  # noqa: E402
from model.lstm import (  # noqa: E402
    arsitektur,
    fitur,
    jalur,
    kalibrasi,
    keputusan,
    kunci,
    kunci_2,
    laporan_fitur,
    model_akhir,
    pembanding,
    walk_forward,
)
from pdhl_data import common  # noqa: E402
from pdhl_data.config import load_config  # noqa: E402

ADA_TORCH = importlib.util.find_spec("torch") is not None
CFG = load_config(ROOT / "configs" / "tesis.json")
KUNCI = kunci.muat_kunci(CFG, ROOT / "configs" / "kunci_1")

LABEL_R_COLS = [
    "event_id", "split", "session_date", "level_type", "label",
    "label_end_session_date", "R_rev", "R_cont", "R_none",
]


def _make_run_dev_calib_r(paths, seed=17):
    """Bangun run sintetis lengkap dev (2018-2023) + calib (2024) dengan kolom R_rev/R_cont/R_none."""
    rnd = random.Random(seed)
    # Mulai 2023-09-01 (~110 hari trading -> mencakup akhir 2023 & awal 2024)
    bars = make_bars(n_sessions=115, seed=seed, start=dt.date(2023, 9, 1))
    events_dev, events_calib = [], []
    labels_dev, labels_calib = [], []

    # Pisahkan menjadi 2023 (dev) dan 2024 (calib) setelah 22 sesi pemanasan volume
    for s in range(22, len(bars) // 23):
        t = 23 * s + rnd.choice((4, 10, 15))
        sd = bars[t]["session_date"]
        sp = common.split_of(sd)
        if sp not in ("dev", "calib"):
            continue
        lv = rnd.choice(("PDH", "PDL"))
        e = make_event(bars, t, lv, split=sp)
        lab = rnd.choice(("Reversal", "Continuation", "Netral"))
        if lab == "Reversal":
            rr, rc = 2.0, -1.0
        elif lab == "Continuation":
            rr, rc = -1.0, 0.5
        else:
            rr = round(rnd.uniform(-0.8, 1.4), 10)
            rc = round(-0.5 * rr, 10)
        row_lab = {
            "event_id": e["event_id"],
            "split": sp,
            "session_date": sd,
            "level_type": lv,
            "label": lab,
            "label_end_session_date": sd,
            "R_rev": rr,
            "R_cont": rc,
            "R_none": 0.0,
        }
        if sp == "dev":
            events_dev.append(e)
            labels_dev.append(row_lab)
        else:
            events_calib.append(e)
            labels_calib.append(row_lab)

    # Pastikan tiap kelas muncul di dev agar Netral/m_rev valid
    assert any(r["label"] == "Netral" for r in labels_dev)
    assert len(events_dev) >= 24 and len(events_calib) >= 12

    common.write_csv(paths.input / "bars_devcalib.csv", bars, BAR_COLS)
    common.write_csv(paths.input / "events_dev.csv", events_dev, EVENT_COLS)
    common.write_csv(paths.input / "events_calib.csv", events_calib, EVENT_COLS)
    common.write_csv(paths.input / "labels_dev.csv", labels_dev, LABEL_R_COLS)
    common.write_csv(paths.input / "labels_calib.csv", labels_calib, LABEL_R_COLS)
    return bars, events_dev, events_calib


def _cleanup_ro(root):
    p = Path(root)
    if not p.exists():
        return
    for item in [p, *p.rglob("*")]:
        try:
            os.chmod(item, stat.S_IRWXU)
        except OSError:
            pass


class TestKalibrasiMatematika(unittest.TestCase):
    def test_parameter_kunci1_lanjutan_terbaca_benar(self):
        self.assertEqual((KUNCI.a, KUNCI.b, KUNCI.T, KUNCI.k), (2.0, 1.0, 8, 0))
        self.assertEqual(KUNCI.temperature_rentang, (0.05, 20.0))
        self.assertEqual(KUNCI.ece_bin, 10)
        self.assertAlmostEqual(KUNCI.delta_utama, 0.1)
        self.assertEqual(KUNCI.delta_grid, (0.0, 0.1, 0.2, 0.3))

    def test_golden_section_redam_overconfident_dan_pertahankan_ranking(self):
        # Logits sangat ekstrem (overconfident) dengan 30% label salah -> tau optimal harus > 1
        logits = []
        labels = []
        for i in range(60):
            if i % 3 == 0:
                logits.append([4.0, 0.0, -1.0])
                labels.append("Continuation" if i % 9 == 0 else "Reversal")
            elif i % 3 == 1:
                logits.append([0.0, 4.0, -1.0])
                labels.append("Netral" if i % 9 == 1 else "Continuation")
            else:
                logits.append([-1.0, 0.0, 4.0])
                labels.append("Reversal" if i % 9 == 2 else "Netral")

        ll_raw = kalibrasi.log_loss_logits(logits, labels, tau=1.0, clip=1e-15)
        opt = kalibrasi.cari_tau_golden_section(logits, labels, rentang=(0.05, 20.0), clip=1e-15)
        self.assertGreater(opt["tau"], 1.2)
        self.assertLess(opt["log_loss"], ll_raw)

        # Argmax (urutan peringkat prediksi) tidak boleh berubah setelah temperature scaling
        for z in logits:
            p_raw = kalibrasi.softmax_row(z, tau=1.0)
            p_cal = kalibrasi.softmax_row(z, tau=opt["tau"])
            self.assertEqual(p_raw.index(max(p_raw)), p_cal.index(max(p_cal)))
            self.assertAlmostEqual(sum(p_cal), 1.0, places=12)

    def test_golden_section_tajamkan_underconfident(self):
        # Logits terlalu datar untuk prediksi yang 100% akurat -> tau optimal harus < 1
        logits = [[0.2, -0.1, -0.1]] * 20 + [[-0.1, 0.2, -0.1]] * 20 + [[-0.1, -0.1, 0.2]] * 20
        labels = ["Reversal"] * 20 + ["Continuation"] * 20 + ["Netral"] * 20
        ll_raw = kalibrasi.log_loss_logits(logits, labels, tau=1.0, clip=1e-15)
        opt = kalibrasi.cari_tau_golden_section(logits, labels, rentang=(0.05, 20.0), clip=1e-15)
        self.assertLess(opt["tau"], 0.5)
        self.assertLess(opt["log_loss"], ll_raw)

    def test_evaluasi_probabilitas_ece_dan_reliabilitas(self):
        probs = [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7]]
        labels = ["Reversal", "Continuation", "Netral"]
        levels = ["PDH", "PDL", "PDH"]
        ev = kalibrasi.evaluasi_probabilitas(probs, labels, levels, clip=1e-15, n_bins=10)
        self.assertEqual(ev["n"], 3)
        self.assertEqual(len(ev["reliabilitas_top"]), 10)
        self.assertEqual(set(ev["ece_per_kelas"]), set(common.CLASSES))
        self.assertEqual(ev["log_loss_per_level"]["PDH"]["n"], 2)
        self.assertEqual(ev["log_loss_per_level"]["PDL"]["n"], 1)


class TestKeputusanRM27(unittest.TestCase):
    def test_estimasi_m_netral_dan_identitas_aljabar(self):
        rows = [
            {"session_date": "2023-10-01", "label_end_session_date": "2023-10-01",
             "label": "Netral", "R_rev": "0.8", "R_cont": "-0.4"},
            {"session_date": "2023-10-02", "label_end_session_date": "2023-10-02",
             "label": "Netral", "R_rev": "0.2", "R_cont": "-0.1"},
            {"session_date": "2023-12-31", "label_end_session_date": "2024-01-02",
             "label": "Netral", "R_rev": "1.6", "R_cont": "-0.8"},   # harus di-purge!
            {"session_date": "2023-10-05", "label_end_session_date": "2023-10-05",
             "label": "Reversal", "R_rev": "2.0", "R_cont": "-1.0"},
        ]
        est = keputusan.estimasi_m_netral(rows, a=2.0, b=1.0)
        self.assertEqual(est["n_dev_latih"], 3)
        self.assertEqual(est["n_dev_purged"], 1)
        self.assertEqual(est["n_netral_dev"], 2)
        self.assertAlmostEqual(est["m_rev"], 0.5, places=12)
        self.assertAlmostEqual(est["m_cont"], -0.25, places=12)

    def test_batas_keputusan_dan_realized_r(self):
        m_rev, m_cont = 0.5626710944, -0.2813355472
        # Kasus 1: E_rev > +0.1 -> Reversal
        er1, ec1 = keputusan.nilai_harapan(0.45, 0.35, 0.20, m_rev, m_cont, a=2.0, b=1.0)
        self.assertAlmostEqual(ec1, -0.5 * er1, places=10)
        self.assertGreaterEqual(er1, 0.1)
        self.assertEqual(keputusan.putuskan(er1, ec1, delta=0.1), "Reversal")
        self.assertEqual(keputusan.realized_r("Reversal", 2.0, -1.0, 0.0), 2.0)

        # Kasus 2: E_rev <= -0.2 (E_cont >= +0.1) -> Continuation
        er2, ec2 = keputusan.nilai_harapan(0.15, 0.70, 0.15, m_rev, m_cont, a=2.0, b=1.0)
        self.assertAlmostEqual(ec2, -0.5 * er2, places=10)
        self.assertLessEqual(er2, -0.2)
        self.assertEqual(keputusan.putuskan(er2, ec2, delta=0.1), "Continuation")
        self.assertEqual(keputusan.realized_r("Continuation", -1.0, 0.5, 0.0), 0.5)

        # Kasus 3: -0.2 < E_rev < +0.1 -> Abstain (0R)
        er3, ec3 = keputusan.nilai_harapan(0.25, 0.55, 0.20, m_rev, m_cont, a=2.0, b=1.0)
        self.assertTrue(-0.2 < er3 < 0.1)
        self.assertEqual(keputusan.putuskan(er3, ec3, delta=0.1), "Abstain")
        self.assertEqual(keputusan.realized_r("Abstain", 2.0, -1.0, 0.0), 0.0)

        # Kasus batas delta=0 dan E_rev == 0 -> Abstain
        self.assertEqual(keputusan.putuskan(0.0, 0.0, delta=0.0), "Abstain")
        self.assertEqual(keputusan.putuskan(0.01, -0.005, delta=0.0), "Reversal")
        self.assertEqual(keputusan.putuskan(-0.01, 0.005, delta=0.0), "Continuation")


class TestAlurEndToEndKunci2Stdlib(unittest.TestCase):
    def test_rantai_lengkap_hingga_kunci2_dan_deteksi_tamper(self):
        tmp = tempfile.mkdtemp()
        try:
            paths = jalur.model_paths(CFG, tmp)
            _make_run_dev_calib_r(paths, seed=21)
            sumber = jalur.info_sumber(jalur.cek_data_input(paths, None))

            # RM-22 & RM-23
            fitur.run_fitur(paths, KUNCI, sumber, lookback_audit=12)
            laporan_fitur.tulis_laporan_fitur(paths)
            # Pembanding akhir + OOF minimal 1 tahun sintetis (2023)
            meta_dev = common.load_model_csv(paths.data / "seq_meta_dev.csv")
            lab_dev = common.load_model_csv(paths.input / "labels_dev.csv")
            gab = pembanding.gabung(meta_dev, lab_dev, KUNCI.theta)
            akhir_pb = pembanding.model_akhir(gab, KUNCI.laplace)
            # Tulis rekonsiliasi & OOF pembanding sintetis yang sah secara hash
            oof_pb = [{
                "event_id": r["event_id"], "session_date": r["session_date"],
                "tahun_validasi": 2023, "label": r["label"], "b1_terpenuhi": r["b1"],
                "b0_p_rev": 0.3, "b0_p_cont": 0.5, "b0_p_net": 0.2,
                "b1_p_rev": 0.3, "b1_p_cont": 0.5, "b1_p_net": 0.2,
                "ll_b0": 1.0, "ll_b1": 1.0,
            } for r in gab]
            common.write_csv(paths.data / "pembanding_oof_dev.csv", oof_pb, pembanding.OOF_COLS)
            pb_recon = {
                "rm": "RM-23", "kelas": list(common.CLASSES), "theta_b1": KUNCI.theta,
                "laplace": KUNCI.laplace, "clip": KUNCI.clip,
                "sumber_sha256": {
                    "seq_meta_dev.csv": common.sha256_file(paths.data / "seq_meta_dev.csv"),
                    "labels_dev.csv": common.sha256_file(paths.input / "labels_dev.csv"),
                },
                "akhir": akhir_pb, "walk_forward": [], "oof": {"n": len(oof_pb)},
            }
            paths.output.mkdir(parents=True, exist_ok=True)
            (paths.output / "pembanding_rekonsiliasi.json").write_text(json.dumps(pb_recon))

            # Tulis walk_forward_rekonsiliasi & OOF sintetis (L12_h16 terpilih)
            common.write_csv(paths.data / "walk_forward_oof_dev.csv", [], walk_forward.OOF_COLS)
            wf_recon = {
                "rm": "RM-24",
                "terpilih": {"id": "L12_h16", "lookback_L": 12, "hidden_size": 16},
                "sumber_sha256": {
                    "seq_meta_dev.csv": common.sha256_file(paths.data / "seq_meta_dev.csv"),
                    "labels_dev.csv": common.sha256_file(paths.input / "labels_dev.csv"),
                    "pembanding_oof_dev.csv": common.sha256_file(paths.data / "pembanding_oof_dev.csv"),
                },
            }
            (paths.output / "walk_forward_rekonsiliasi.json").write_text(json.dumps(wf_recon))

            # RM-25 memakai stub stdlib (menguji simpan, roundtrip, hash, laporan)
            def fake_latih(X, y, cfg, teknis):
                n_es = max(4, len(X) // 5)
                n_in = len(X) - n_es
                return {
                    "cfg": dict(cfg),
                    "members": [
                        {"seed": s, "best_epoch": 5 + i, "best_es_loss": 0.91 + 0.001 * i,
                         "n_in_train": n_in, "n_early_stop": n_es}
                        for i, s in enumerate(KUNCI.seeds)
                    ],
                }

            def fake_simpan(bundle, ckpt_dir, n_fit):
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                sp = ckpt_dir / f"scaler_{bundle['cfg']['id']}.json"
                sp.write_text(json.dumps({
                    "cfg_id": bundle["cfg"]["id"], "lookback_L": bundle["cfg"]["lookback_L"],
                    "n_fit_events": n_fit, "feature_names": list(fitur.FEATURE_NAMES),
                    "skip_indices": list(fitur.CYCLIC_INDICES), "clip_val": 10.0,
                    "mean": [0.0] * len(fitur.FEATURE_NAMES),
                    "std": [1.0] * len(fitur.FEATURE_NAMES),
                }))
                cps = []
                for m in bundle["members"]:
                    p = ckpt_dir / f"lstm_{bundle['cfg']['id']}_seed{m['seed']}.pt"
                    p.write_bytes(f"FAKE_PT_{m['seed']}".encode())
                    cps.append(p)
                return {"scaler_path": sp, "checkpoint_paths": cps}

            def fake_muat(cpaths, spath, cfg, teknis):
                return {"reloaded": True}

            def fake_logits(bundle, X):
                return [[0.8, -0.3, -0.4] if i % 2 == 0 else [-0.6, 1.2, -0.5] for i in range(len(X))]

            ma = model_akhir.run_model_akhir(
                paths, KUNCI, sumber,
                latih=fake_latih, simpan=fake_simpan, muat=fake_muat,
                prediksi_logits=fake_logits, verbose=False,
            )
            self.assertTrue(ma["uji_roundtrip"]["lulus"])
            self.assertEqual(len(ma["artefak"]["checkpoints"]), 5)

            # RM-26 (kalibrasi τ)
            kal = kalibrasi.run_kalibrasi(
                paths, KUNCI, sumber,
                prediksi_logits=lambda X: fake_logits(None, X),
            )
            self.assertTrue(0.05 <= kal["tau"] <= 20.0)
            self.assertLessEqual(kal["metrik_cal"]["log_loss"], kal["metrik_raw"]["log_loss"] + 1e-12)
            self.assertTrue((paths.data / "calib_pred_lstm.csv").exists())

            # RM-27 (keputusan)
            kep = keputusan.run_keputusan(paths, KUNCI, sumber)
            self.assertEqual(len(kep["kurva_delta_cal"]), len(KUNCI.delta_grid))
            self.assertTrue((paths.data / "keputusan_calib_lstm.csv").exists())

            # Kunci-2 (segel & idempotensi)
            k2_a = kunci_2.run_kunci_2(paths, KUNCI, sumber, CFG["name"])
            self.assertTrue(k2_a["baru_disegel"])
            k2_b = kunci_2.run_kunci_2(paths, KUNCI, sumber, CFG["name"])
            self.assertFalse(k2_b["baru_disegel"])

            # Verifikasi segel Kunci-2 harus OK
            ver = kunci_2.verifikasi_kunci_2(paths, KUNCI, sumber)
            self.assertTrue(ver["ok"])
            self.assertEqual(ver["n_berkas_beku"], 8)  # 5 .pt + scaler + json + md

            # Tamper 1 byte pada berkas beku -> wajib ditolak Kunci2Error
            os.chmod(paths.kunci_2, stat.S_IRWXU)
            target = paths.kunci_2 / "kunci_2_lstm.json"
            os.chmod(target, stat.S_IRWXU)
            target.write_text(target.read_text() + " ")
            with self.assertRaises(kunci_2.Kunci2Error):
                kunci_2.verifikasi_kunci_2(paths, KUNCI, sumber)
        finally:
            _cleanup_ro(tmp)


@unittest.skipUnless(ADA_TORCH, "PyTorch belum terpasang (pip install -e '.[lstm]')")
class TestPyTorchSimpanMuatDanKunci2(unittest.TestCase):
    def test_simpan_dan_muat_bundle_roundtrip_eksak(self):
        import numpy as np

        n_f = len(fitur.FEATURE_NAMES)
        rng = np.random.default_rng(19)
        X_tr = rng.normal(size=(40, 12, n_f)).astype(np.float32)
        y_tr = np.array([i % 3 for i in range(40)], dtype=np.int64)
        X_cal = rng.normal(size=(10, 12, n_f)).astype(np.float32)

        teknis_cepat = {**KUNCI.lstm_teknis, "max_epochs": 4, "patience": 2, "batch_size": 16}
        cfg = {"id": "L12_h8", "lookback_L": 12, "hidden_size": 8}
        seeds = (20261119, 20261120)

        bundle = arsitektur.latih_lstm(X_tr, y_tr, cfg, teknis_cepat, seeds=seeds)
        logits_before = arsitektur.prediksi_logits_lstm(bundle, X_cal)

        with tempfile.TemporaryDirectory() as tmp:
            saved = arsitektur.simpan_bundle(bundle, Path(tmp) / "ckpts")
            reloaded = arsitektur.muat_bundle(
                saved["checkpoint_paths"], saved["scaler_path"], cfg, teknis_cepat, seeds=seeds
            )
            logits_after = arsitektur.prediksi_logits_lstm(reloaded, X_cal)
            np.testing.assert_allclose(logits_before, logits_after, rtol=0.0, atol=1e-12)

            # Suhu != 1 menghasilkan probabilitas valid berbeda namun argmax identik
            p_raw = arsitektur.prediksi_lstm(reloaded, X_cal, temperature=1.0)
            p_hot = arsitektur.prediksi_lstm(reloaded, X_cal, temperature=2.0)
            for a, b in zip(p_raw, p_hot):
                self.assertAlmostEqual(sum(b), 1.0, places=6)
                self.assertEqual(a.index(max(a)), b.index(max(b)))


if __name__ == "__main__":
    unittest.main()
