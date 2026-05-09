# EEG Motor Imagery Prediction Application

A Streamlit application for analysing and classifying EEG motor imagery signals using the [PhysioNet EEG Motor Movement/Imagery Dataset](https://www.physionet.org/content/eegmmidb/1.0.0/).

**ARISE Summer Research Internship - Technical Challenge**  
**Task:** Left-hand (T1) vs Right-hand (T2) imagined movement classification

---

## Live Application

[View on Streamlit Community Cloud](https://eeg-motor-imagery.streamlit.app/)

---

## Project Structure

```
eeg-motor-imagery/
├── app.py                  # Main Streamlit application
├── train.py                # Offline model training script
├── requirements.txt
├── README.md
├── models/
│   ├── random_forest.pkl
│   ├── mlp.pkl
│   └── model_metadata.json
├── results/
│   ├── offline_model_results.csv
│   └── figures/
│       ├── random_forest_cm.png
│       └── mlp_cm.png
└── screenshots/
    └── app_screenshot.png
```

---

## Setup & Running Locally

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download PhysioNet data

The training script auto-downloads the data via MNE:

```bash
python train.py
```

This downloads runs 4, 8, and 12 for subjects 1-50 automatically into a local data/ folder.

### 3. Run the app

```bash
streamlit run app.py
```

Then upload any EDF file from runs 4, 8, or 12 to analyse.

---

## Offline Training Details

### Dataset
- **Source:** PhysioNet EEG Motor Movement/Imagery Dataset
- **Subjects used:** S001-S050 (50 subjects)
- **Runs:** 4, 8, 12 (imagined movement runs only)
- **Total epochs:** 2,250 (Left: 1,134 | Right: 1,116)
- **Task:** Binary classification - T1 (imagined left fist) vs T2 (imagined right fist)

### Channels
Motor cortex channels: C3, Cz, C4, FC3, FC4, CP3, CP4

These channels were selected for their known relevance to motor imagery. C3 and C4 sit directly over the left and right motor cortex respectively and show the strongest contralateral activation patterns during imagined hand movement.

### Preprocessing
1. **Band-pass filtering:** 8-30 Hz (FIR, firwin design) - covers mu (8-13 Hz) and beta (13-30 Hz) rhythms associated with motor imagery
2. **Event extraction:** T1 and T2 annotations only (T0/rest excluded from classification)
3. **Epoch creation:** 0.5-3.5 seconds post-stimulus - edges trimmed to avoid movement onset artefacts

### Feature Extraction
CSP (Common Spatial Patterns) with 4 components.

CSP finds linear combinations of EEG channels that maximise the variance difference between left and right hand classes. Rather than treating each channel independently, CSP identifies the spatial filter that best separates the two conditions - directly capturing the contralateral activation pattern in motor cortex. Log-variance of the CSP-filtered signal is used as the feature vector (4 values per epoch).

CSP is the standard feature extraction method for motor imagery BCI and consistently outperforms simple band power approaches on this type of data.

### Models

| Model | Architecture | Accuracy | F1 Score | Evaluation |
|-------|-------------|----------|----------|------------|
| Random Forest | 300 trees, max_depth=8 | 0.600 +/- 0.151 | 0.596 | Per-subject 5-fold CV |
| MLP | 256->128->64 units, early stopping | 0.566 +/- 0.115 | 0.633 | Per-subject 5-fold CV |

**Evaluation strategy:** Per-subject 5-fold stratified cross-validation. Each subject's epochs are evaluated independently and results are averaged across all 50 subjects. This avoids inter-subject data leakage and reflects realistic performance - mixing subjects inflates accuracy because the model memorises individual brain signal patterns rather than learning a general classifier.

The accuracy range of 57-60% is consistent with published baselines for this dataset using spatial filtering and linear classifiers. Inter-subject variability is a known challenge - each person's EEG patterns differ significantly, making cross-subject generalisation difficult without subject-specific calibration data.

---

## Application Components

1. **EDF Upload & Data Inspection** - sampling frequency, channel names, recording duration, available annotations
2. **EEG Signal Visualisation** - filtered signal with event markers, raw vs filtered comparison, power spectral density plot
3. **Preprocessing & Feature Extraction** - pipeline summary, CSP spatial filtering description
4. **Offline Model Summary** - accuracy, F1 score, and confusion matrix for both models
5. **Prediction Results** - epoch-level table with ground truth vs predicted labels, colour-coded prediction timeline

---

## Notes
- The raw EDF dataset is not included in this repository - download directly from PhysioNet
- Model training is performed entirely offline; the app loads pre-trained .pkl files for inference
- Tested on runs 4, 8, and 12 across multiple subjects
- The CSP transform is fitted inside the sklearn pipeline, so the app passes raw epoch data directly to the model
