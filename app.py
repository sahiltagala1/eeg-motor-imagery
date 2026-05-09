"""
EEG Motor Imagery Prediction Application
PhysioNet EEG Motor Movement/Imagery Dataset
Task: Left-hand (T1) vs Right-hand (T2) motor imagery classification
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import mne
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

EVENT_ID = {"T1": 2, "T2": 3}
TMIN, TMAX = 0.5, 3.5
L_FREQ, H_FREQ = 8.0, 30.0
MOTOR_CHANNELS = ["C3", "Cz", "C4", "FC3", "FC4", "CP3", "CP4"]
LABEL_MAP = {0: "Left (T1)", 1: "Right (T2)"}
COLOR_MAP = {"Left (T1)": "#4C9BE8", "Right (T2)": "#E8704C"}

st.set_page_config(page_title="EEG Motor Imagery Predictor", page_icon="🧠", layout="wide")

st.markdown("""
    <style>
    .section-header {
        font-size: 1.1rem; font-weight: 600; color: #1e5aa0;
        border-bottom: 2px solid #1e5aa0; padding-bottom: 4px; margin-bottom: 12px;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    models, metadata = {}, {}
    model_dir = "models"
    if not os.path.exists(model_dir):
        return models, metadata
    for fname in os.listdir(model_dir):
        if fname.endswith(".pkl"):
            name = fname.replace(".pkl", "").replace("_", " ").title()
            models[name] = joblib.load(os.path.join(model_dir, fname))
    meta_path = os.path.join(model_dir, "model_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            metadata = json.load(f)
    return models, metadata


def process_edf(uploaded_file):
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
    mne.datasets.eegbci.standardize(raw)

    raw_full = raw.copy()
    available = [ch for ch in MOTOR_CHANNELS if ch in raw.ch_names]
    raw.pick_channels(available if available else raw.ch_names[:10])

    raw_filtered = raw.copy().filter(L_FREQ, H_FREQ, fir_design="firwin", verbose=False)
    events, _ = mne.events_from_annotations(raw_filtered, verbose=False)

    try:
        epochs = mne.Epochs(raw_filtered, events, EVENT_ID, tmin=TMIN, tmax=TMAX,
                            baseline=None, preload=True, verbose=False)
    except Exception as e:
        st.error(f"Could not extract epochs: {e}")
        return raw_full, raw_filtered, None, events

    os.unlink(tmp_path)
    return raw_full, raw_filtered, epochs, events


# ── Sidebar ────────────────────────────────────────────────────────────────
models, metadata = load_models()

with st.sidebar:
    st.markdown("### Upload EDF File")
    uploaded = st.file_uploader("Supported: PhysioNet EDF (Runs 4, 8, 12)", type=["edf"])
    st.markdown("---")
    st.markdown("**Ground Truth Labels**")
    st.markdown("- T1 = Imagined left fist")
    st.markdown("- T2 = Imagined right fist")
    if models:
        selected_model = st.selectbox("Model", list(models.keys()))
        channel = st.selectbox("Channel to visualise", MOTOR_CHANNELS, index=0)
        time_range = st.select_slider("Time range (s)",
                                      options=["0-10", "0-20", "0-30", "0-60", "0-120"],
                                      value="0-30")
    else:
        st.warning("No trained models found. Run train.py first.")
        selected_model = None

# ── Header ─────────────────────────────────────────────────────────────────
st.title("🧠 EEG Motor Imagery Prediction")
st.caption("PhysioNet EEG Motor Movement/Imagery Dataset · Left vs Right Hand Motor Imagery")

# ── Offline model summary (always shown) ───────────────────────────────────
if metadata:
    st.markdown('<p class="section-header">Offline Model Training Summary</p>', unsafe_allow_html=True)
    cols = st.columns(len(metadata))
    for col, (mname, mdata) in zip(cols, metadata.items()):
        with col:
            st.markdown(f"**{mname}**")
            m1, m2 = st.columns(2)
            m1.metric("Accuracy", f"{mdata['accuracy']:.2%}")
            m2.metric("F1 Score", f"{mdata['f1_score']:.2%}")
            std = mdata.get("accuracy_std", "")
            st.caption(f"Std: ±{std}" if std else "")
            st.caption(f"Subjects: {mdata.get('n_subjects','?')} · Epochs: {mdata.get('n_epochs','?')}")
            st.caption(f"Features: {mdata.get('features','?')}")
            st.caption(f"Filter: {mdata.get('filter','?')}")
            st.caption(f"Eval: {mdata.get('evaluation','?')}")
            safe = mname.lower().replace(" ", "_")
            cm_path = f"results/figures/{safe}_cm.png"
            if os.path.exists(cm_path):
                st.image(cm_path, caption="Confusion Matrix")

if uploaded is None:
    st.info("Upload an EDF file from PhysioNet runs 4, 8, or 12 to begin analysis.")
    st.stop()

# ── Process file ───────────────────────────────────────────────────────────
with st.spinner("Reading and processing EDF file..."):
    raw_full, raw_filtered, epochs, events = process_edf(uploaded)

# ── Section 1: File Info ───────────────────────────────────────────────────
st.markdown("---")
st.markdown('<p class="section-header">1. EDF File Information</p>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Sampling Frequency", f"{raw_filtered.info['sfreq']:.0f} Hz")
c2.metric("Channels (motor)", len(raw_filtered.ch_names))
c3.metric("Duration", f"{raw_filtered.times[-1]:.1f} s")
annotations = sorted(set(raw_full.annotations.description))
c4.metric("Annotations", ", ".join(annotations))
with st.expander("Channel names"):
    st.write(raw_filtered.ch_names)

# ── Section 2: Signal Visualisation ───────────────────────────────────────
st.markdown('<p class="section-header">2. EEG Signal Visualisation</p>', unsafe_allow_html=True)

t_end = int(time_range.split("-")[1])
sfreq = raw_filtered.info["sfreq"]
if channel not in raw_filtered.ch_names:
    channel = raw_filtered.ch_names[0]

ch_idx = raw_filtered.ch_names.index(channel)
times = raw_filtered.times
mask = times <= t_end
t_plot = times[mask]

raw_ch = raw_full.get_data()[raw_full.ch_names.index(channel)][mask] * 1e6 \
    if channel in raw_full.ch_names else np.zeros(mask.sum())
filt_ch = raw_filtered.get_data()[ch_idx][mask] * 1e6

col1, col2, col3 = st.columns([2, 1.5, 1.5])

with col1:
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(t_plot, filt_ch, color="#1e5aa0", lw=0.7)
    for ev in events:
        ev_t = ev[0] / sfreq
        if ev_t <= t_end:
            color = "#4C9BE8" if ev[2] == 2 else "#E8704C"
            ax.axvline(ev_t, color=color, alpha=0.8, lw=1.2,
                       label="T1 (Left)" if ev[2] == 2 else "T2 (Right)")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=7)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude (µV)")
    ax.set_title(f"Filtered EEG - Channel {channel}")
    st.pyplot(fig, use_container_width=True)
    plt.close()

with col2:
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(t_plot, raw_ch, color="#aaa", lw=0.5, label="Raw", alpha=0.8)
    ax.plot(t_plot, filt_ch, color="#1e5aa0", lw=0.7, label="Filtered (8-30Hz)")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("µV")
    ax.set_title("Raw vs Filtered")
    ax.legend(fontsize=7)
    st.pyplot(fig, use_container_width=True)
    plt.close()

with col3:
    freqs_psd, psd = welch(filt_ch, fs=sfreq, nperseg=256)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.semilogy(freqs_psd, psd, color="#1e5aa0", lw=1)
    ax.axvspan(8, 13, alpha=0.15, color="green", label="Mu (8-13Hz)")
    ax.axvspan(13, 30, alpha=0.15, color="orange", label="Beta (13-30Hz)")
    ax.set_xlim(0, 50); ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (µV²/Hz)"); ax.set_title("PSD Plot")
    ax.legend(fontsize=7)
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ── Section 3: Preprocessing & Pipeline ────────────────────────────────────
st.markdown('<p class="section-header">3. Preprocessing & Feature Extraction</p>', unsafe_allow_html=True)
st.markdown("""
**Pipeline:**  
`Band-pass filter (8–30 Hz)` → `Extract T1/T2 events` → `Create epochs (0.5–3.5s)` → `CSP spatial filtering (4 components)` → `Pre-trained classifier`

**CSP (Common Spatial Patterns)** finds linear combinations of EEG channels that maximise variance differences between left and right hand classes. 
This captures the contralateral motor cortex activation pattern more effectively than individual channel analysis.
""")

if epochs is None or len(epochs) == 0:
    st.warning("No T1/T2 epochs found. Make sure you uploaded a run 4, 8, or 12 file.")
    st.stop()

n_left = np.sum(epochs.events[:, 2] == 2)
n_right = np.sum(epochs.events[:, 2] == 3)
st.success(f"Found **{len(epochs)}** epochs - {n_left} Left (T1) · {n_right} Right (T2)")

# ── Section 4: Predictions ─────────────────────────────────────────────────
st.markdown('<p class="section-header">4. Prediction Results & Evaluation</p>', unsafe_allow_html=True)

if selected_model and selected_model in models:
    model = models[selected_model]
    X_raw = epochs.get_data()  # CSP is inside the pipeline, just pass raw epochs
    y_true = (epochs.events[:, 2] == 3).astype(int)

    try:
        y_pred = model.predict(X_raw)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    y_true_labels = [LABEL_MAP[v] for v in y_true]
    y_pred_labels = [LABEL_MAP[v] for v in y_pred]
    correct = [gt == pr for gt, pr in zip(y_true_labels, y_pred_labels)]

    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy", f"{acc:.2%}")
    m2.metric("F1 Score", f"{f1:.2%}")
    m3.metric("Correct Epochs", f"{sum(correct)} / {len(correct)}")

    # Epoch table
    epoch_times = epochs.events[:, 0] / sfreq
    df = pd.DataFrame({
        "Epoch": range(1, len(y_true) + 1),
        "Start (s)": epoch_times.round(1),
        "End (s)": (epoch_times + (TMAX - TMIN)).round(1),
        "Ground Truth": y_true_labels,
        "Prediction": y_pred_labels,
        "Correct": ["✓" if c else "✗" for c in correct]
    })
    st.dataframe(df, use_container_width=True)

    # Prediction timeline
    st.markdown("**Prediction Timeline**")
    fig, axes = plt.subplots(2, 1, figsize=(10, 3), sharex=True)
    for i, (labels, title) in enumerate([(y_true_labels, "Ground Truth"),
                                          (y_pred_labels, f"Prediction ({selected_model})")]):
        for label, t in zip(labels, epoch_times):
            axes[i].barh(0, TMAX - TMIN, left=t, color=COLOR_MAP[label],
                         edgecolor="white", height=0.5)
            axes[i].text(t + (TMAX - TMIN) / 2, 0, label.split()[0],
                         ha="center", va="center", fontsize=7,
                         color="white", fontweight="bold")
        axes[i].set_ylabel(title, fontsize=9)
        axes[i].set_yticks([])
    axes[1].set_xlabel("Time (s)")
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=l) for l, c in COLOR_MAP.items()]
    axes[0].legend(handles=legend_elements, loc="upper right", fontsize=7)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()
