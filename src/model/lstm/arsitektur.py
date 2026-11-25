"""Arsitektur LSTM PyTorch, SequenceScaler3D anti-leakage, dan pelatihan 5-seed.

Modul ini diimpor secara malas (lazy) hanya saat tahap `walk-forward` dijalankan dengan pelatihan
PyTorch nyata, sehingga tahap `fitur`, `pembanding`, serta unit test stdlib tetap berjalan tanpa
PyTorch maupun NumPy.
"""
import copy
import random

from .fitur import CYCLIC_INDICES


def _impor_torch():
    try:
        import numpy as np
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"{exc.name} belum terpasang. Pasang dependensi LSTM: `pip install -e '.[lstm]'`"
        ) from exc
    return np, torch, nn, DataLoader, TensorDataset


def set_deterministic_seed(seed):
    """Kunci seed Python, NumPy, dan PyTorch agar pelatihan 100% deterministik."""
    np, torch, _, _, _ = _impor_torch()
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)


def softmax_np(logits):
    """Softmax stabil secara numerik sepanjang sumbu terakhir."""
    np, _, _, _, _ = _impor_torch()
    z = np.asarray(logits, dtype=np.float64)
    z_shift = z - np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z_shift)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


def log_loss_np(probs, y_true, clip=1e-15):
    """Rata-rata −ln p(y) dengan clip [clip, 1] lalu normalisasi ulang (identik Kunci-1)."""
    np, _, _, _, _ = _impor_torch()
    p = np.clip(np.asarray(probs, dtype=np.float64), clip, 1.0)
    p = p / np.sum(p, axis=1, keepdims=True)
    y = np.asarray(y_true, dtype=np.int64)
    return float(-np.mean(np.log(p[np.arange(len(y)), y])))


class SequenceScaler3D:
    """Standardisasi z-score per fitur pada tensor 3D (N, L, F) hanya dari data latih.

    Fitur siklik (`ny_hour_sin`, `ny_hour_cos`) dilewati karena sudah berada pada [-1, 1].
    Setelah scaling, fitur non-siklik dipotong ke [-clip_val, +clip_val] untuk meredam outlier.
    """

    def __init__(self, clip_val=10.0, skip_indices=CYCLIC_INDICES):
        self.clip_val = float(clip_val)
        self.skip_indices = tuple(skip_indices)
        self.mean_ = None
        self.std_ = None

    def fit(self, X_train):
        np, _, _, _, _ = _impor_torch()
        arr = np.asarray(X_train, dtype=np.float64)
        if arr.ndim != 3:
            raise ValueError(f"SequenceScaler3D membutuhkan tensor 3D (N, L, F), didapat {arr.shape}")
        flat = arr.reshape(-1, arr.shape[-1])
        mean = flat.mean(axis=0)
        std = flat.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        for idx in self.skip_indices:
            mean[idx] = 0.0
            std[idx] = 1.0
        self.mean_ = mean
        self.std_ = std
        return self

    def transform(self, X):
        np, _, _, _, _ = _impor_torch()
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("SequenceScaler3D belum di-fit sebelum transform")
        arr = np.asarray(X, dtype=np.float64)
        out = (arr - self.mean_) / self.std_
        if self.clip_val is not None:
            mask = np.ones(arr.shape[-1], dtype=bool)
            for idx in self.skip_indices:
                mask[idx] = False
            out[..., mask] = np.clip(out[..., mask], -self.clip_val, self.clip_val)
        return out.astype(np.float32)

    def fit_transform(self, X_train):
        return self.fit(X_train).transform(X_train)


def build_classifier(n_features, hidden_size, num_layers=1, dropout=0.3, n_classes=3):
    """Bangun modul PyTorch 1-layer LSTM -> hidden state terakhir -> Dropout -> Linear(3)."""
    _, _, nn, _, _ = _impor_torch()

    class PDHLLSTMClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=int(n_features),
                hidden_size=int(hidden_size),
                num_layers=int(num_layers),
                batch_first=True,
                dropout=float(dropout) if int(num_layers) > 1 else 0.0,
            )
            self.drop = nn.Dropout(p=float(dropout))
            self.head = nn.Linear(int(hidden_size), int(n_classes))

        def forward(self, x):
            _, (h_n, _) = self.lstm(x)
            return self.head(self.drop(h_n[-1]))

    return PDHLLSTMClassifier()


def _predict_logits(model, X_scaled, device):
    np, torch, _, _, _ = _impor_torch()
    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(X_scaled).to(device)
        return model(xb).cpu().numpy().astype(np.float64)


def latih_satu_seed(X_tr_raw, y_tr, *, hidden_size, teknis, seed, clip=1e-15, device="cpu"):
    """Latih 1 seed LSTM dengan early stopping kronologis 15% di ujung data latih."""
    np, torch, nn, DataLoader, TensorDataset = _impor_torch()
    set_deterministic_seed(seed)

    X_arr = np.asarray(X_tr_raw, dtype=np.float32)
    y_arr = np.asarray(y_tr, dtype=np.int64)
    n_tr = len(X_arr)
    n_es = max(16, int(round(n_tr * float(teknis["early_stop_frac"]))))
    if n_es >= n_tr - 16:
        n_es = max(4, n_tr // 5)
    split_at = n_tr - n_es

    X_in_tr, y_in_tr = X_arr[:split_at], y_arr[:split_at]
    X_in_es, y_in_es = X_arr[split_at:], y_arr[split_at:]

    scaler = SequenceScaler3D(clip_val=teknis.get("clip_features", 10.0))
    X_in_tr_s = scaler.fit_transform(X_in_tr)
    X_in_es_s = scaler.transform(X_in_es)

    dev = torch.device(device)
    model = build_classifier(
        n_features=X_arr.shape[-1],
        hidden_size=int(hidden_size),
        num_layers=int(teknis.get("num_layers", 1)),
        dropout=float(teknis["dropout"]),
    ).to(dev)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(teknis["lr"]),
        weight_decay=float(teknis["weight_decay"]),
    )

    gen = torch.Generator()
    gen.manual_seed(int(seed))
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_in_tr_s), torch.from_numpy(y_in_tr)),
        batch_size=int(teknis["batch_size"]),
        shuffle=True,
        generator=gen,
    )

    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 1
    no_improve = 0
    patience = int(teknis["patience"])
    max_epochs = int(teknis["max_epochs"])
    grad_clip = float(teknis["grad_clip"])

    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        es_logits = _predict_logits(model, X_in_es_s, dev)
        es_loss = log_loss_np(softmax_np(es_logits), y_in_es, clip=clip)
        if es_loss < best_loss - 1e-6:
            best_loss = es_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return {
        "model": model,
        "scaler": scaler,
        "best_epoch": best_epoch,
        "best_es_loss": best_loss,
        "seed": int(seed),
        "device": dev,
    }


def latih_lstm(X_tr, y_tr, cfg, teknis, *, seeds, clip=1e-15, device="cpu"):
    """Latih ensemble 5 seed LSTM untuk satu konfigurasi (L, hidden) pada satu lipatan."""
    members = [
        latih_satu_seed(
            X_tr, y_tr,
            hidden_size=cfg["hidden_size"],
            teknis=teknis,
            seed=s,
            clip=clip,
            device=device,
        )
        for s in seeds
    ]
    return {
        "members": members,
        "best_epochs": [m["best_epoch"] for m in members],
        "cfg": dict(cfg),
    }


def prediksi_lstm(bundle, X_val):
    """Rata-ratakan logits lintas 5 seed lalu terapkan softmax -> [[p_rev, p_cont, p_net], ...]."""
    np, _, _, _, _ = _impor_torch()
    X_arr = np.asarray(X_val, dtype=np.float32)
    seed_logits = []
    for m in bundle["members"]:
        X_s = m["scaler"].transform(X_arr)
        seed_logits.append(_predict_logits(m["model"], X_s, m["device"]))
    mean_logits = np.mean(seed_logits, axis=0)
    probs = softmax_np(mean_logits)
    return [[float(v) for v in row] for row in probs]
