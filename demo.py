import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ---------- Paths (assume script is run from its directory) ----------
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
NPZ_PATH = os.path.join(DATA_DIR, "cev_20221214_windows_npz.npz")
MANIFEST_PATH = os.path.join(DATA_DIR, "windows_A_manifest.json")
NORM_STATS_PATH = os.path.join(DATA_DIR, "cev_20221214_norm_stats.json")

# ---------- Dataset ----------
class WindowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ---------- Model ----------
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        # NOTE: no Sigmoid here; we will use BCEWithLogitsLoss for stability
    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        last = self.dropout(hn[-1])
        logit = self.fc(last).squeeze(-1)
        return logit

# ---------- Utils ----------
def pick_device():
    # Prefer Apple Silicon MPS when available (M1/M2/M3), then CUDA, else CPU
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def load_artifacts(npz_path, manifest_path, norm_path):
    d = np.load(npz_path, allow_pickle=True)
    X = d["X"]           # (n_windows, seq_len, n_features)
    y = d["y"]           # (n_windows,)
    feature_cols = d["feature_cols"].tolist()
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    with open(norm_path, "r") as f:
        norm = json.load(f)
    return X, y, feature_cols, manifest, norm

def get_indices_and_weights(manifest, y):
    # Support both manifest schemas
    if "indices" in manifest:
        train_idx = np.array(manifest["indices"].get("train", []), dtype=int)
        val_idx = np.array(manifest["indices"].get("val", []), dtype=int)
        if "class_weights" in manifest and "fault" in manifest["class_weights"]:
            w_fault = float(manifest["class_weights"]["fault"])
        else:
            # compute from train only
            y_tr = y[train_idx]
            n_fault = max(1, int((y_tr == 1).sum()))
            n_norm = max(1, int((y_tr == 0).sum()))
            w_fault = n_norm / n_fault
    else:
        # Fallback keys (train_idx/val_idx and class_weights list)
        train_idx = np.array(manifest.get("train_idx", []), dtype=int)
        val_idx = np.array(manifest.get("val_idx", []), dtype=int)
        cw = manifest.get("class_weights")
        if isinstance(cw, list) and len(cw) == 2:
            # list is [w_normal, w_fault] or similar; assume index 1 is positive
            w_fault = float(cw[1])
        else:
            y_tr = y[train_idx]
            n_fault = max(1, int((y_tr == 1).sum()))
            n_norm = max(1, int((y_tr == 0).sum()))
            w_fault = n_norm / n_fault
    return train_idx, val_idx, w_fault

def drop_all_nan_features(X, feature_cols):
    keep = []
    for j in range(X.shape[2]):
        if not np.all(np.isnan(X[:, :, j])):
            keep.append(j)
    X2 = X[:, :, keep]
    cols2 = [feature_cols[j] for j in keep]
    return X2, cols2, keep

def standardise(X, feature_cols, norm):
    # norm has mu/sigma keyed by feature name
    mu = norm.get("mu", {})
    sigma = norm.get("sigma", {})
    means = np.array([mu.get(name, 0.0) for name in feature_cols], dtype=np.float32)
    stds = np.array([sigma.get(name, 1.0) for name in feature_cols], dtype=np.float32)
    stds = np.where((stds == 0) | ~np.isfinite(stds), 1.0, stds)
    # broadcast over time dimension
    X = (X - means) / stds
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X

def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            logits = model(Xb)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= threshold).astype(int)
            y_true.append(yb.numpy())
            y_pred.append(preds)
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()

    X, y, feature_cols, manifest, norm = load_artifacts(NPZ_PATH, MANIFEST_PATH, NORM_STATS_PATH)

    # Drop all-NaN features (e.g., IA(A) in this event)
    X, feature_cols, kept_idx = drop_all_nan_features(X, feature_cols)

    # Standardise with train-set stats (mu/sigma are global, but that’s fine for this single-file demo)
    X = standardise(X, feature_cols, norm)

    # Splits and class weights
    train_idx, val_idx, w_fault = get_indices_and_weights(manifest, y)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val,   y_val   = X[val_idx],   y[val_idx]

    train_loader = DataLoader(WindowDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(WindowDataset(X_val,   y_val),   batch_size=args.batch_size, shuffle=False)

    device = pick_device()
    print(f"Using device: {device}")

    model = LSTMModel(input_size=X.shape[2], hidden_size=args.hidden_size, dropout=args.dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # Positive class weighting for imbalance
    pos_weight = torch.tensor([w_fault], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_f1 = -1.0
    patience_ctr = 0
    history = {"train": [], "val": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for Xb, yb in train_loader:
            Xb = Xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Evaluate
        train_metrics = evaluate(model, train_loader, device)
        val_metrics   = evaluate(model, val_loader, device)
        history["train"].append({"loss": float(np.mean(train_losses)), **train_metrics})
        history["val"].append(val_metrics)
        print(f"Epoch {epoch:03d} | Train F1={train_metrics['f1']:.3f} | Val F1={val_metrics['f1']:.3f} | Val Rec={val_metrics['recall']:.3f}")

        if val_metrics["f1"] > best_f1 + 1e-4:
            best_f1 = val_metrics["f1"]
            patience_ctr = 0
            torch.save({
                "model_state": model.state_dict(),
                "feature_cols": feature_cols,
                "hidden_size": args.hidden_size,
                "dropout": args.dropout,
            }, os.path.join(DATA_DIR, "model_lstm_v1.pt"))
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print("Early stopping.")
                break

    with open(os.path.join(DATA_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"Best Val F1: {best_f1:.3f}")

if __name__ == "__main__":
    main()