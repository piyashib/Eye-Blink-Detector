# Eye Blink Artifact Detection from Single-Channel EEG

This repository provides a semi-supervised pipeline to create datasets and detect/classify eye blink artifacts in single-channel EEG recordings using statistical features, dimensionality reduction, and unsupervised clustering.

## Overview

Traditional multi-channel EEG setups can use ICA for artifact rejection, but this approach is not feasible for single-channel recordings (e.g., Cz, located centrally, away from the forehead). This workflow enables automated artifact labeling and blink detection for such cases using data-driven, semi-supervised methods.

### Data

- 25-channel EEG, 5 minutes per 19 subjects, already notch filtered (60 Hz), high-pass filtered (0.1 Hz), and re-referenced using common average referencing (CAR).

## Notebooks

### 1. CreateAllDatasets.ipynb

**Purpose:** Prepare training, validation, and test datasets for the eye-blink detection model.

**Key Steps:**
- Load multi-channel EEG and extract relevant channels (Cz for modeling, Fp1 for blink reference).
- Visualize differences between channels (blink artifact is prominent on Fp1, subtle on Cz).
- Segment data into fixed-size epochs (e.g., 0.5-sec).
- Assign preliminary blink/no-blink labels using the Fp1 channel and thresholding.
- Split into train/val/test sets, with minimal manual labeling required.
- Save processed data to CSV for downstream modeling.

### 2. ModelEyeBlinkDetector.ipynb

**Purpose:** Develop a machine learning model to classify eye-blink contaminated epochs using semi-supervised clustering.

**Key Steps:**
- Extract waveform statistical features from each epoch: mean, median, std, variance, skew, kurtosis, range, peak/width, low/high frequency power, Hjorth mobility/complexity.
- Standardize features and apply PCA for dimensionality reduction.
- Use k-means clustering on features (or PCA representation) to find blink/no-blink groupings.
- Map cluster labels to actual artifact classes using labeled training subset.
- Evaluate performance on validation/test data with metrics: accuracy, F1, precision, recall.
- (Explored) Time-series k-means clustering with DTW, but computationally intensive.
- Visualize cluster separation and waveform properties.

## Dependencies

- Python 3.8+
    - numpy, pandas, scipy, matplotlib, seaborn
    - scikit-learn
    - h5py
    - Optionally: dtaidistance, tslearn (for DTW experiments)
- Custom helpers (you must provide): signalprocessinghelper.py and mlmodelhelper.py

## Usage

- Place raw EEG data (.mat files), channel info, and these notebooks in your working directory.
- Run CreateAllDatasets.ipynb to preprocess, epoch, and label your data. It outputs labeled/unlabeled dataset CSVs.
- Run ModelEyeBlinkDetector.ipynb to build and evaluate the ML pipeline.
- Adjust paths and parameters as necessary for your directory or analysis tweaks.
- Inspect confusion matrices and feature plots to refine model choices.

## Notes

- Labels for training are largely bootstrapped via the Fp1 channel and thresholding; the aim is to minimize manual annotation.
- The pipeline is extensible to other channels and waveforms; feature functions and cluster number can be modified.
- Dynamic Time Warping and clustering on raw time-series were not tested due to computational challenges.