
# EEG Seizure Detection System

A professional end-to-end deep learning framework for automated epileptic seizure detection using electroencephalography (EEG) signals from the CHB-MIT Scalp EEG Database.

This project combines biomedical signal processing, dataset engineering, hybrid neural network modeling, intelligent post-processing, and an interactive visualization dashboard for clinically relevant seizure event analysis.

---

## Overview

Epileptic seizure detection from EEG recordings is a critical biomedical engineering task that supports faster diagnosis, continuous monitoring, and computer-aided neurological assessment.

This system processes raw EEG recordings and identifies seizure activity using a multi-stage pipeline consisting of:

- EEG preprocessing  
- Window-based segmentation  
- Balanced dataset generation  
- Deep learning classification  
- Temporal post-processing  
- Event-level seizure detection  
- GUI-based monitoring dashboard  

---

## Key Features

### Signal Processing

- Multi-channel EEG handling
- Bandpass filtering (0.5–40 Hz)
- Noise reduction pipeline
- Window segmentation for time-series learning

### Dataset Engineering

- Automatic EDF dataset parsing
- Label generation from seizure annotations
- Class balancing for seizure/non-seizure imbalance
- Mean/std normalization pipeline

### Hybrid Deep Learning Architecture

- 1D CNN feature extractor
- Channel Attention module
- Bidirectional LSTM
- Transformer Encoder
- Fully connected classifier
- Probability-based seizure prediction

### Intelligent Detection Logic

- Temporal smoothing
- Hysteresis thresholding
- False positive reduction
- Onset backtracking
- Event grouping logic

### User Interface

- EEG waveform visualization
- Detected seizure onset / offset table
- Real-time monitoring dashboard
- CSV export support
- Screenshot/image export

---

## System Architecture


Raw EEG Signal
      ↓
Preprocessing
      ↓
Window Segmentation
      ↓
CNN Feature Extraction
      ↓
Channel Attention
      ↓
BiLSTM Temporal Modeling
      ↓
Transformer Global Context
      ↓
Classifier
      ↓
Seizure Probability
      ↓
Post-processing Logic
      ↓
Final Detected Events




## Deep Learning Model

The proposed hybrid architecture is designed to capture:

### Spatial Features

Extracted using 1D CNN layers from multi-channel EEG signals.

### Channel Importance

Learned using attention mechanisms to prioritize relevant electrodes.

### Temporal Dynamics

Captured using BiLSTM layers for sequential seizure evolution.

### Global Context

Transformer encoder models long-range dependencies across windows.



## Project Structure


eeg_seizure_detection_system/
│
├── build_dataset_complete.py
├── build_balanced_dataset.py
├── normalize_dataset.py
├── train_model.py
├── evaluate_model.py
├── model_architecture.py
├── detection_engine.py
├── dashboard.py
├── requirements.txt
├── README.md
├── LICENSE
│
└── images/
    ├── dashboard.png
    ├── architecture.png
    └── detection_output.png




## File Description

| File                      | Purpose                         |
| ------------------------- | ------------------------------- |
| build_dataset_complete.py | Generate dataset from EDF files |
| build_balanced_dataset.py | Handle class imbalance          |
| normalize_dataset.py      | Mean/std normalization          |
| train_model.py            | Train seizure detection model   |
| evaluate_model.py         | Evaluate trained model          |
| model_architecture.py     | Neural network definition       |
| detection_engine.py       | Real-time inference pipeline    |
| dashboard.py              | GUI visualization dashboard     |

---

## Dataset

### CHB-MIT Scalp EEG Database

Widely used benchmark dataset for seizure detection research.

**Contains:**

* Pediatric EEG recordings
* Multi-channel scalp EEG
* Seizure annotations
* EDF file format

---

## Technologies Used

* Python
* PyTorch
* NumPy
* SciPy
* Scikit-learn
* Pandas
* MNE
* Matplotlib
* PyQt5

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Run Application

```bash
python dashboard.py
```

---

## Output

The system provides:

* EEG waveform visualization
* Detected seizure onset times
* Detected seizure offset times
* Event table
* CSV reports
* Exported images

---

## Performance Metrics

Evaluated using standard classification metrics:

* Accuracy
* Precision
* Recall (Sensitivity)
* F1 Score
* Confusion Matrix

Typical hybrid model performance:

| Metric    | Value    |
| --------- | -------- |
| Accuracy  | High     |
| Precision | High     |
| Recall    | High     |
| F1 Score  | Balanced |

(Exact results depend on split strategy and subject selection.)

---

## Screenshots

### Dashboard

![Dashboard](images/dashboard.png)

### Model Architecture

![Architecture](images/architecture.png)

### Detection Output

![Detection Output](images/detection_output.png)

---

## Biomedical Relevance

This project can support future systems such as:

* ICU EEG monitoring
* Wearable seizure alerts
* Neurology decision support tools
* Remote epilepsy monitoring
* AI-assisted diagnostics

---

## Future Improvements

* Real-time streaming EEG support
* Patient-specific fine tuning
* Multi-class seizure type detection
* Cloud deployment
* Edge-device inference
* Clinical report generation

---

## Notes

Dataset files are not included in this repository due to storage limitations.

Users should download CHB-MIT EEG dataset separately.

---

## Author

**KGR**

Biomedical AI / EEG Research Project

---

## License

This project is licensed under the MIT License.

See the `LICENSE` file for details.

```
```
