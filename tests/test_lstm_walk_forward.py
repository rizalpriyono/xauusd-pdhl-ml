"""Tes tahap 3 validasi walk-forward LSTM (RM-24) dan arsitektur PyTorch.

Tes tanpa PyTorch memakai pelatih palsu (stdlib) sehingga selalu berjalan di Python sistem/CI.
Tes yang membutuhkan PyTorch dilewati otomatis bila `torch` belum terpasang.
"""
import importlib.util
import json
import sys
import tempfile
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gb_sintetis import make_bars, make_event  # noqa: E402
from model.lstm import arsitektur, fitur, jalur, kunci, pembanding  # noqa: E402
from model.lstm import walk_forward as wf  # noqa: E402
from pdhl_data import common  # noqa: E402
from pdhl_data.config import load_config  # noqa: E402

ADA_TORCH = importlib.util.find_spec("torch") is not None
CFG = load_config(ROOT / "configs" / "tesis.json")
KUNCI = kunci.muat_kunci(CFG, ROOT / "configs" / "kunci_1")


def latih_frekuensi(X, y, cfg, teknis):
    """Pelatih palsu (stdlib): kembalikan frekuensi kelas latih."""
    c = Counter(y)
    return [c.get(k, 0) / len(y) for k in range(len(common.CLASSES))]


def prediksi_frekuensi(model, X):
    return [list(model) for _ in X]


def _baris(i, tanggal, label, akhir=None):
    return {
        "event_id": f"e{i}",
        "session_date": tanggal,
        "label": label,
        "label_end_session_date": akhir or tanggal,
        "f13": 0.1,
        "b1": False,
        "x": [[0.0] * len(fitur.FEATURE_NAMES) for _ in range(4)],
    }


class TestGridDanPemilihan(unittest.TestCase):
    def test_sepuluh_konfigurasi_terkunci(self):
        cfgs = wf.konfigurasi_grid(KUNCI)
        self.assertEqual(len(cfgs), 10)
        self.assertEqual(cfgs[0], {"id": "L23_h16", "lookback_L": 23, "hidden_size": 16})
        self.assertEqual(cfgs[-1], {"id": "L96_h32", "lookback_L": 96, "hidden_size": 32})

    def test_grid_menyimpang_ditolak(self):
        rusak = replace(KUNCI, lookbacks=(10, 20))
        with self.assertRaises(ValueError):
            wf.konfigurasi_grid(rusak)

    def test_tie_breaker_pilih_model_lebih_sederhana(self):
        cfgs = [
            {"id": "L40_h32", "lookback_L": 40, "hidden_size": 32},
            {"id": "L60_h16", "lookback_L": 60, "hidden_size": 16},
            {"id": "L23_h16", "lookback_L": 23, "hidden_size": 16},
        ]
        # Selisih semua <= 1e-6 -> harus menang L23_h16 (hidden terkecil, lalu L terkecil)
        skor = {"L40_h32": 0.9300000, "L60_h16": 0.9300004, "L23_h16": 0.9300008}
        pilih = wf.pilih_konfigurasi(skor, cfgs, tol=1e-6)
        self.assertEqual(pilih["id"], "L23_h16")

    def test_walk_forward_purging_dengan_pelatih_palsu(self):
        rows = [
            _baris(1, "2018-06-01", "Reversal"),
            _baris(2, "2018-07-01", "Continuation"),
            _baris(3, "2018-12-31", "Netral", akhir="2019-01-02"),   # harus di-purge untuk 2019
            _baris(4, "2019-02-01", "Reversal"),
            _baris(5, "2019-06-01", "Continuation"),
            _baris(6, "2020-03-01", "Netral"),
        ]
        cfgs = [{"id": "L4_h16", "lookback_L": 4, "hidden_size": 16}]
        oof, lipatan = wf.walk_forward_model(
            {4: rows}, cfgs, KUNCI.lstm_teknis, KUNCI.seeds, (2019, 2020), KUNCI.clip,
            latih=latih_frekuensi, prediksi=prediksi_frekuensi, verbose=False,
        )
        self.assertEqual(len(oof), 3)
        self.assertEqual(lipatan[0]["n_latih"], 2)
        self.assertEqual(lipatan[0]["n_purged"], 1)
        self.assertEqual(lipatan[1]["n_latih"], 5)
        self.assertEqual(lipatan[1]["n_purged"], 0)

    def test_kesegaran_pembanding_menolak_hash_lama(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = jalur.model_paths(CFG, tmp)
            paths.data.mkdir(parents=True)
            paths.input.mkdir(parents=True)
            paths.output.mkdir(parents=True)
            mp = paths.data / "seq_meta_dev.csv"
            lp = paths.input / "labels_dev.csv"
            mp.write_text("a\n1\n")
            lp.write_text("b\n2\n")
            (paths.output / "pembanding_rekonsiliasi.json").write_text(
                json.dumps({"sumber_sha256": {mp.name: "0" * 64, lp.name: "0" * 64}})
            )
            with self.assertRaises(ValueError):
                wf.cek_pembanding_segar(paths, mp, lp)


@unittest.skipUnless(ADA_TORCH, "PyTorch belum terpasang (pip install -e '.[lstm]')")
class TestPyTorchLSTM(unittest.TestCase):
    def test_scaler_3d_tidak_bocor_dan_lewati_siklik(self):
        import numpy as np

        n_f = len(fitur.FEATURE_NAMES)
        X_tr = np.zeros((20, 8, n_f), dtype=np.float32)
        X_tr[:, :, 0] = np.linspace(-2.0, 2.0, 20)[:, None]
        for idx, val in zip(fitur.CYCLIC_INDICES, (0.5, -0.5, 0.3, -0.3)):
            X_tr[:, :, idx] = val

        X_val = np.full((5, 8, n_f), 100.0, dtype=np.float32)
        for idx, val in zip(fitur.CYCLIC_INDICES, (0.25, -0.75, 0.6, -0.8)):
            X_val[:, :, idx] = val

        sc = arsitektur.SequenceScaler3D(clip_val=10.0)
        sc.fit(X_tr)
        mean_before = sc.mean_.copy()
        out_val = sc.transform(X_val)

        # Transformasi val tidak boleh mengubah statistik train
        np.testing.assert_allclose(sc.mean_, mean_before)
        # Fitur non-siklik dipotong di +10.0; 4 fitur siklik (8..11) tidak disentuh
        np.testing.assert_allclose(out_val[:, :, 0], 10.0)
        for idx, val in zip(fitur.CYCLIC_INDICES, (0.25, -0.75, 0.6, -0.8)):
            np.testing.assert_allclose(out_val[:, :, idx], val)

    def test_determinisme_seed_dan_bentuk_probabilitas(self):
        import numpy as np

        n_f = len(fitur.FEATURE_NAMES)
        rng = np.random.default_rng(7)
        X_tr = rng.normal(size=(40, 12, n_f)).astype(np.float32)
        y_tr = np.array([i % 3 for i in range(40)], dtype=np.int64)
        X_val = rng.normal(size=(8, 12, n_f)).astype(np.float32)

        teknis_cepat = {**KUNCI.lstm_teknis, "max_epochs": 4, "patience": 2, "batch_size": 16}
        cfg = {"id": "L12_h8", "lookback_L": 12, "hidden_size": 8}

        b1 = arsitektur.latih_lstm(X_tr, y_tr, cfg, teknis_cepat, seeds=(20261119, 20261120))
        b2 = arsitektur.latih_lstm(X_tr, y_tr, cfg, teknis_cepat, seeds=(20261119, 20261120))
        p1 = arsitektur.prediksi_lstm(b1, X_val)
        p2 = arsitektur.prediksi_lstm(b2, X_val)

        np.testing.assert_allclose(p1, p2, atol=1e-6)
        self.assertEqual(len(p1), 8)
        for row in p1:
            self.assertEqual(len(row), 3)
            self.assertAlmostEqual(sum(row), 1.0, places=6)
            self.assertTrue(all(0.0 <= v <= 1.0 for v in row))


if __name__ == "__main__":
    unittest.main()
