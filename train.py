"""
Offline training script for EEG Motor Imagery Classification.
Dataset: PhysioNet EEG Motor Movement/Imagery Dataset
Task: T1 (imagined left fist) vs T2 (imagined right fist)
Runs used: 4, 8, 12

Feature extraction: CSP (Common Spatial Patterns) + band power
CSP is the standard feature extraction method for motor imagery BCI.
Evaluation: Per-subject 5-fold stratified cross-validation.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import mne
from mne.datasets import eegbci
from mne.decoding import CSP
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

RUNS = [4, 8, 12]
EVENT_ID = {"T1": 2, "T2": 3}
TMIN, TMAX = 0.5, 3.5
L_FREQ, H_FREQ = 8.0, 30.0
MOTOR_CHANNELS = ["C3", "Cz", "C4", "FC3", "FC4", "CP3", "CP4"]
SUBJECTS = list(range(1, 51))
N_CSP = 4  # number of CSP components


def load_subject_epochs(subject):
    raw_files = eegbci.load_data(subject, RUNS, path="data/")
    raws = [mne.io.read_raw_edf(f, preload=True, verbose=False) for f in raw_files]
    raw = mne.concatenate_raws(raws)
    mne.datasets.eegbci.standardize(raw)
    available = [ch for ch in MOTOR_CHANNELS if ch in raw.ch_names]
    raw.pick_channels(available if available else raw.ch_names[:10])
    raw.filter(L_FREQ, H_FREQ, fir_design="firwin", verbose=False)
    events, _ = mne.events_from_annotations(raw, verbose=False)
    try:
        epochs = mne.Epochs(raw, events, EVENT_ID, tmin=TMIN, tmax=TMAX,
                            baseline=None, preload=True, verbose=False)
    except Exception:
        return None
    return epochs if len(epochs) > 0 else None


def make_pipelines():
    return {
        "Random Forest": Pipeline([
            ("csp", CSP(n_components=N_CSP, reg=None, log=True, norm_trace=False)),
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=300, max_depth=8,
                                           min_samples_leaf=2, random_state=42, n_jobs=-1))
        ]),
        "MLP": Pipeline([
            ("csp", CSP(n_components=N_CSP, reg=None, log=True, norm_trace=False)),
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500,
                                  random_state=42, early_stopping=True,
                                  validation_fraction=0.15, learning_rate_init=0.001))
        ])
    }


def train():
    print(f"\nLoading data for {len(SUBJECTS)} subjects...")
    subject_data = {}

    for s in SUBJECTS:
        print(f"  Subject {s:02d}...", end=" ", flush=True)
        epochs = load_subject_epochs(s)
        if epochs is None:
            print("skipped")
            continue
        # Store raw epoch data (n_epochs, n_channels, n_times) and labels
        X = epochs.get_data()
        y = (epochs.events[:, 2] == 3).astype(int)
        subject_data[s] = (X, y)
        print(f"{len(y)} epochs, {X.shape[1]} channels")

    print(f"\nLoaded {len(subject_data)} subjects")

    os.makedirs("models", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    results = {}
    # Store all data for final model training
    all_X = np.vstack([v[0] for v in subject_data.values()])
    all_y = np.concatenate([v[1] for v in subject_data.values()])
    print(f"Total epochs: {len(all_y)} | Left: {np.sum(all_y==0)} | Right: {np.sum(all_y==1)}")

    for name, _ in make_pipelines().items():
        print(f"\nEvaluating {name} with CSP (per-subject 5-fold CV)...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        subject_accs, subject_f1s = [], []
        all_true, all_pred = [], []

        for s, (Xs, ys) in subject_data.items():
            if len(np.unique(ys)) < 2 or len(ys) < 10:
                continue
            fold_preds = np.zeros(len(ys), dtype=int)
            for train_idx, test_idx in cv.split(Xs, ys):
                pipe = make_pipelines()[name]
                pipe.fit(Xs[train_idx], ys[train_idx])
                fold_preds[test_idx] = pipe.predict(Xs[test_idx])
            subject_accs.append(accuracy_score(ys, fold_preds))
            subject_f1s.append(f1_score(ys, fold_preds, zero_division=0))
            all_true.extend(ys)
            all_pred.extend(fold_preds)

        mean_acc = np.mean(subject_accs)
        std_acc = np.std(subject_accs)
        mean_f1 = np.mean(subject_f1s)
        cm = confusion_matrix(all_true, all_pred)

        print(f"  Accuracy: {mean_acc:.3f} ± {std_acc:.3f} | F1: {mean_f1:.3f}")

        # Confusion matrix plot
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Left (T1)", "Right (T2)"])
        ax.set_yticklabels(["Left (T1)", "Right (T2)"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(f"{name} — Confusion Matrix")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max()/2 else "black",
                        fontweight="bold")
        plt.tight_layout()
        safe = name.lower().replace(" ", "_")
        fig.savefig(f"results/figures/{safe}_cm.png", dpi=100)
        plt.close()

        results[name] = {
            "accuracy": round(mean_acc, 4),
            "accuracy_std": round(std_acc, 4),
            "f1_score": round(mean_f1, 4),
            "confusion_matrix": cm.tolist(),
            "n_subjects": len(subject_data),
            "n_epochs": int(len(all_y)),
            "n_csp_components": N_CSP,
            "channels": MOTOR_CHANNELS,
            "filter": f"{L_FREQ}-{H_FREQ} Hz",
            "features": f"CSP ({N_CSP} components, log variance)",
            "evaluation": "Per-subject 5-fold stratified CV",
            "epoch_window": f"{TMIN}-{TMAX}s"
        }

        # Train final model on all data
        final_pipe = make_pipelines()[name]
        final_pipe.fit(all_X, all_y)
        joblib.dump(final_pipe, f"models/{safe}.pkl")
        print(f"  Saved models/{safe}.pkl")

    pd.DataFrame(results).T.to_csv("results/offline_model_results.csv")
    with open("models/model_metadata.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Final Results ===")
    for name, r in results.items():
        print(f"{name}: Accuracy={r['accuracy']:.3f} ± {r['accuracy_std']:.3f} | F1={r['f1_score']:.3f}")
    print("\nAll done.")


if __name__ == "__main__":
    train()
