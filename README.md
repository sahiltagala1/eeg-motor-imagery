# EEG Motor Imagery Prediction Application

A Streamlit application for analysing and classifying EEG motor imagery signals using the [PhysioNet EEG Motor Movement/Imagery Dataset](https://www.physionet.org/content/eegmmidb/1.0.0/).

**ARISE Summer Research Internship — Technical Challenge**  
**Task:** Left-hand (T1) vs Right-hand (T2) imagined movement classification

---

## Live Application

[View on Streamlit Community Cloud](#) *(link added after deployment)*

---

## Project Structure

```
eeg_prediction_application/
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

Download EDF files from: https://www.physionet.org/content/eegmmidb/1.0.0/

Or let the training script auto-download via MNE:

```bash
python train.py --subjects 1 20
```

To use local files:

```bash
python train.py --data_dir /path/to/physionet --subjects 1 20
```

### 3. Run the app

```bash
streamlit run app.py
```

Then upload any EDF file from **runs 4, 8, or 12** to analyse.

---

## Offline Training Details

### Dataset
- **Source:** PhysioNet EEG Motor Movement/Imagery Dataset
- **Subjects used:** S001–S020 (20 subjects)
- **Runs:** 4, 8, 12 (imagined movement runs)
- **Task:** Binary classification — T1 (imagined left fist) vs T2 (imagined right fist)

### Channels
Motor cortex channels: `C3, Cz, C4, FC3, FC4, CP3, CP4`  
These are selected for their relevance to motor imagery, particularly the contralateral activation pattern associated with left/right hand movements.

### Preprocessing
1. **Band-pass filtering:** 8–30 Hz (FIR, firwin design)  
   Covers mu (8–13 Hz) and beta (13–30 Hz) rhythms associated with motor imagery
2. **Event extraction:** T1 and T2 annotations only (T0/rest excluded from classification)
3. **Epoch creation:** 0–4 seconds post-stimulus, no baseline correction

### Feature Extraction
**Band power features** computed via FFT for each epoch:
- **Mu band:** 8–13 Hz mean power per channel
- **Beta band:** 13–30 Hz mean power per channel
- **Total features:** 2 × 7 channels = 14 features per epoch

Band power was chosen because mu and beta rhythms show event-related desynchronisation (ERD) contralateral to the imagined movement hand, making them the most established features for motor imagery BCI.

### Models

| Model | Architecture | Accuracy | F1 Score |
|-------|-------------|----------|----------|
| Random Forest | 200 trees, max_depth=10 | ~0.82 | ~0.81 |
| MLP | 128→64 hidden units, early stopping | ~0.79 | ~0.78 |

- **Evaluation:** 5-fold stratified cross-validation
- **Preprocessing in pipeline:** StandardScaler applied before each classifier

---

## Application Components

1. **EDF Upload & Data Inspection** — sampling frequency, channels, duration, annotations
2. **EEG Signal Visualisation** — raw vs filtered, PSD plot, event markers on timeline
3. **Preprocessing & Feature Extraction** — pipeline summary, band power bar chart per channel
4. **Offline Model Summary** — accuracy, F1, confusion matrices for both models
5. **Prediction Results** — epoch-level table + colour-coded prediction timeline vs ground truth

---

## Notes
- The full dataset is not included in this repository (files are large; download from PhysioNet directly)
- Model training is performed entirely offline; the app only loads pre-trained `.pkl` files for inference
- Tested on runs 4, 8, and 12 from multiple subjects
