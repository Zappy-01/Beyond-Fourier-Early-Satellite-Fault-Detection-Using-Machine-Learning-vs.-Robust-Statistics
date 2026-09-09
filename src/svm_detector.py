"""
SVM Fault Detection Module - Week 9

This module implements the Support Vector Machine (SVM) based fault detection system.
It uses a digital twin approach where expected healthy features are subtracted from
actual features to create residuals, which are then classified by an SVM.

Based on research: "Beyond Fourier: Early Satellite Fault Detection Using Machine Learning vs. Robust Statistics"
"""

import os
import glob
import numpy as np
import pandas as pd
import pywt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold


# Configuration constants
DT = 0.1  # Sampling period
WINDOW_SIZE = 128  # 12.8 second windows
STEP = 10  # 1 second step
CWT_SCALES = np.arange(1, 32)  # Scales for CWT (Morlet)
NUM_FEATURES_PER_CHANNEL = 6  # Band1-4, Entropy, Centroid
NUM_CHANNELS = 4  # 4 motor current channels


def compute_cwt_features(window_data):
    """
    Compute 6 CWT-based features for a single channel window.
    
    Features:
    - Band1-Band4: Energy fractions in 4 scale bands
    - Entropy: Spectral entropy
    - Centroid: Spectral centroid
    
    Parameters
    ----------
    window_data : array-like
        Signal window (128 samples)
        
    Returns
    -------
    list
        6 features: [band1, band2, band3, band4, entropy, centroid]
    """
    # Perform Continuous Wavelet Transform with Morlet wavelet
    cwtmatr, _ = pywt.cwt(window_data, CWT_SCALES, 'morl', sampling_period=DT)
    abs_cwt = np.abs(cwtmatr)
    
    # Compute energy per scale
    scale_energies = np.sum(abs_cwt**2, axis=1)
    total_energy = np.sum(scale_energies) + 1e-12
    
    # Normalize energies
    norm_energies = scale_energies / total_energy
    
    # Four frequency bands (8 scales each, last band gets remainder)
    band1 = np.sum(norm_energies[0:8])
    band2 = np.sum(norm_energies[8:16])
    band3 = np.sum(norm_energies[16:24])
    band4 = np.sum(norm_energies[24:])
    
    # Spectral entropy
    probs = norm_energies + 1e-12
    entropy_val = -np.sum(probs * np.log2(probs))
    
    # Spectral centroid
    indices = np.arange(1, len(scale_energies) + 1)
    centroid_val = np.sum(indices * norm_energies)
    
    return [band1, band2, band3, band4, entropy_val, centroid_val]


def extract_svm_features_for_file(filepath):
    """
    Extract SVM features from a single data file.
    
    Parameters
    ----------
    filepath : str
        Path to CSV file
        
    Returns
    -------
    tuple
        (features, labels, file_num, true_onset, is_faulty)
    """
    name_no_ext = os.path.splitext(os.path.basename(filepath))[0]
    parts = name_no_ext.split('_')
    
    if not parts[0].isdigit():
        return None
    
    file_num = int(parts[0])
    scenario = int(parts[1])
    
    # Determine fault status and onset time
    if scenario == 0:
        true_onset = np.inf
        is_faulty = 0
    else:
        kt_inc = float(parts[4])
        vbus_inc = float(parts[5])
        onsets = []
        if int(parts[2]) == 1:
            onsets.append(kt_inc)
        if int(parts[3]) == 1:
            onsets.append(vbus_inc)
        true_onset = min(onsets) if onsets else np.inf
        is_faulty = 1
    
    # Load data
    df = pd.read_csv(filepath)
    
    # Get column names for motor currents
    i_healthy_cols = [f'wheel{w}_i_healthy' for w in range(1, 5)]
    i_faulty_cols = [f'wheel{w}_i_faulty' for w in range(1, 5)]
    
    # Check required columns exist
    if not all(col in df.columns for col in i_healthy_cols + i_faulty_cols):
        return None
    
    i_expected = df[i_healthy_cols].values
    i_actual = df[i_faulty_cols].values if is_faulty else df[i_healthy_cols].values
    n_samples = len(i_actual)
    
    # Extract features for sliding windows
    features_list = []
    labels_list = []
    end_times_list = []
    
    for start_idx in range(0, n_samples - WINDOW_SIZE + 1, STEP):
        end_idx = start_idx + WINDOW_SIZE
        end_time = end_idx * DT
        
        # Label: 1 if window end is at or after fault onset
        label = 1 if end_time >= true_onset else 0
        
        # Extract features for each channel
        expected_window = i_expected[start_idx:end_idx, :]
        actual_window = i_actual[start_idx:end_idx, :]
        
        feats_expected = []
        feats_actual = []
        
        for col_idx in range(4):
            feats_expected.extend(compute_cwt_features(expected_window[:, col_idx]))
            feats_actual.extend(compute_cwt_features(actual_window[:, col_idx]))
        
        # Residual features (actual - expected)
        residual_features = np.array(feats_actual) - np.array(feats_expected)
        
        features_list.append(residual_features)
        labels_list.append(label)
        end_times_list.append(end_time)
    
    return {
        'features': np.array(features_list),
        'labels': np.array(labels_list),
        'file_num': file_num,
        'true_onset': true_onset,
        'is_faulty': is_faulty,
        'end_times': np.array(end_times_list)
    }


def extract_full_dataset(raw_dir, cache_path=None):
    """
    Extract SVM features for all files in the dataset.
    
    Parameters
    ----------
    raw_dir : str
        Directory containing CSV files
    cache_path : str, optional
        Path to save/load cached features
        
    Returns
    -------
    tuple
        (X, y, groups, end_times, true_onsets, is_faulty)
    """
    if cache_path and os.path.exists(cache_path):
        print("Loading cached SVM features...")
        data = np.load(cache_path)
        return (data['X'], data['y'], data['groups'], 
                data['end_times'], data['true_onsets'], data['is_faulty'])
    
    print("Extracting CWT features for all files (This may take 10-20 mins)...")
    file_paths = glob.glob(os.path.join(raw_dir, "*.csv"))
    
    X_list, y_list, group_list = [], [], []
    end_times_list, true_onsets_list, is_faulty_list = [], [], []
    
    for idx, filepath in enumerate(file_paths):
        result = extract_svm_features_for_file(filepath)
        if result is None:
            continue
        
        n_windows = len(result['features'])
        X_list.append(result['features'])
        y_list.append(result['labels'])
        group_list.extend([result['file_num']] * n_windows)
        end_times_list.append(result['end_times'])
        true_onsets_list.extend([result['true_onset']] * n_windows)
        is_faulty_list.extend([result['is_faulty']] * n_windows)
        
        if (idx + 1) % 20 == 0:
            print(f"  ...{idx + 1}/{len(file_paths)} files")
    
    # Flatten arrays
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    groups = np.array(group_list)
    end_times = np.concatenate(end_times_list)
    true_onsets = np.array(true_onsets_list)
    is_faulty = np.array(is_faulty_list)
    
    # Cache results
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path,
                           X=X, y=y, groups=groups,
                           end_times=end_times, true_onsets=true_onsets,
                           is_faulty=is_faulty)
        print("Extraction complete and cached.")
    
    return X, y, groups, end_times, true_onsets, is_faulty


def run_5fold_cv(X, y, groups, end_times, true_onsets, is_faulty):
    """
    Run 5-fold group cross-validation for SVM.
    
    Each file appears exactly once in a test fold, ensuring out-of-sample evaluation.
    
    Parameters
    ----------
    X : ndarray
        Feature matrix
    y : ndarray
        Labels
    groups : ndarray
        Group assignments (file numbers)
    end_times : ndarray
        End times for each window
    true_onsets : ndarray
        True fault onset times
    is_faulty : ndarray
        Fault status per window
        
    Returns
    -------
    list
        List of per-file results dictionaries
    """
    print("Executing 5-Pass Cross-Validation (Out-of-Fold Predictions)...")
    
    gkf = GroupKFold(n_splits=5)
    all_preds = np.zeros(len(y), dtype=int)
    
    # SVM pipeline with standardization
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(C=0.1, gamma='scale', kernel='rbf', random_state=42))
    ])
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        print(f"  Pass {fold+1}/5: Training on 80 files, predicting 20 files...")
        pipeline.fit(X[train_idx], y[train_idx])
        all_preds[test_idx] = pipeline.predict(X[test_idx])
    
    # Aggregate results per file
    results = []
    for file_num in np.unique(groups):
        file_mask = (groups == file_num)
        preds = all_preds[file_mask]
        et = end_times[file_mask]
        to = true_onsets[file_mask][0]
        is_f = is_faulty[file_mask][0]
        
        # Calculate FPR on healthy windows
        healthy_mask = et < to
        fp_count = int(np.sum(preds[healthy_mask] == 1))
        n_healthy = int(np.sum(healthy_mask))
        fpr = (fp_count / n_healthy * 100) if n_healthy > 0 else 0.0
        
        # Calculate detection latency for faulty files
        latency = np.nan
        if is_f:
            valid_flags = et[(preds == 1) & (et >= to - 10.0)]
            if len(valid_flags) > 0:
                latency = valid_flags[0] - to
        
        results.append({
            'File': int(file_num),
            'Faulty': bool(is_f),
            'Latency': latency,
            'FPR': fpr
        })
    
    return results


def train_svm_model(X_train, y_train):
    """
    Train an SVM model on provided data.
    
    Parameters
    ----------
    X_train : ndarray
        Training features
    y_train : ndarray
        Training labels
        
    Returns
    -------
    Pipeline
        Trained SVM pipeline
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(C=0.1, gamma='scale', kernel='rbf', random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def predict_with_svm(pipeline, X):
    """
    Make predictions using a trained SVM pipeline.
    
    Parameters
    ----------
    pipeline : Pipeline
        Trained SVM pipeline
    X : ndarray
        Features to classify
        
    Returns
    -------
    ndarray
        Binary predictions
    """
    return pipeline.predict(X)


def calculate_metrics(results):
    """
    Calculate aggregate metrics from per-file results.
    
    Parameters
    ----------
    results : list
        List of per-file result dictionaries
        
    Returns
    -------
    dict
        Dictionary with aggregate metrics
    """
    results_array = np.array(results)
    
    # Detection rate
    faulty_files = [r for r in results if r['Faulty']]
    detected_files = [r for r in faulty_files if not np.isnan(r['Latency'])]
    detection_rate = len(detected_files) / len(faulty_files) if faulty_files else 0.0
    
    # Latency statistics
    latencies = [r['Latency'] for r in results if not np.isnan(r['Latency'])]
    mean_latency = np.mean(latencies) if latencies else np.nan
    std_latency = np.std(latencies, ddof=1) if len(latencies) > 1 else 0.0
    early_detections = sum(1 for l in latencies if l < 0)
    
    # FPR statistics
    fprs = [r['FPR'] for r in results]
    mean_fpr = np.mean(fprs)
    max_fpr = np.max(fprs)
    
    return {
        'detection_rate': detection_rate,
        'total_faulty': len(faulty_files),
        'detected': len(detected_files),
        'mean_latency': mean_latency,
        'std_latency': std_latency,
        'early_detections': early_detections,
        'mean_fpr': mean_fpr,
        'max_fpr': max_fpr
    }


class SVMDetector:
    """
    SVM-based fault detector with digital twin residual input.
    
    This class provides a scikit-learn compatible interface for the SVM detector.
    """
    
    def __init__(self, C=0.1, kernel='rbf', random_state=42):
        self.C = C
        self.kernel = kernel
        self.random_state = random_state
        self.pipeline = None
    
    def fit(self, X, y):
        """Train the SVM model."""
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(C=self.C, gamma='scale', kernel=self.kernel, 
                       random_state=self.random_state))
        ])
        self.pipeline.fit(X, y)
        return self
    
    def predict(self, X):
        """Predict fault labels."""
        if self.pipeline is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.pipeline.predict(X)
    
    def predict_proba(self, X):
        """Predict fault probabilities."""
        if self.pipeline is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.pipeline.predict_proba(X)
