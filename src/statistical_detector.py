"""
Statistical Detector Module - Week 7

This module implements rule-based statistical detectors for satellite fault detection.
It includes four detector architectures (Primary, Consensus, Robust, Ratio) and 
extracts dense features using the Stationary Wavelet Transform (SWT).

Based on research: "Beyond Fourier: Early Satellite Fault Detection Using Machine Learning vs. Robust Statistics"
"""

import numpy as np
import pywt
from scipy import stats


# Configuration constants
DT = 0.1  # Sampling period
WINDOW_SIZE = 128  # 12.8 seconds at DT=0.1
STEP = 10  # 1 second step
K = 3  # Threshold multiplier (3-sigma)
PERSISTENCE, MERGE_GAP, MIN_DURATION = 1, 1, 1  # Event consolidation parameters


def extract_dense_features(signal):
    """
    Extract 7 statistical features from a signal using level-3 Haar SWT.
    
    Features are extracted for each sliding window across the signal.
    Including the approximation coefficient (cA3) is critical for detecting gradual drifts.
    
    Parameters
    ----------
    signal : array-like
        Input signal (motor current or angular velocity)
        
    Returns
    -------
    ndarray
        Array of shape (num_windows, 7) with features:
        [Energy, Entropy, Variance, RMS, P2P, Kurtosis, Skewness]
    """
    signal = np.array(signal, dtype=float, copy=True)
    
    # Pad signal to be divisible by 2^3 for SWT
    required = 2 ** 3
    n = len(signal)
    if n % required != 0:
        signal = np.pad(signal, (0, required - (n % required)), mode='edge')
    
    # Perform level-3 Stationary Wavelet Transform with Haar wavelet
    coeffs = pywt.swt(signal, wavelet='haar', level=3, trim_approx=False)
    detail_coeffs = [level[1] for level in coeffs]  # Detail coefficients (high-frequency)
    approx_coeff = coeffs[-1][0]  # Approximation coefficient (low-frequency trend)
    
    # Calculate number of windows
    num_windows = ((n - WINDOW_SIZE) // STEP) + 1
    features = np.zeros((num_windows, 7))
    
    for i in range(num_windows):
        s0, e0 = i * STEP, i * STEP + WINDOW_SIZE
        
        # Extract windowed coefficients
        detail_slices = [cd[s0:e0] for cd in detail_coeffs]
        approx_slice = approx_coeff[s0:e0]
        
        # Compute energy per detail level
        energies = [np.sum(np.square(s)) for s in detail_slices]
        E_total = sum(energies)
        
        # Feature 0: Total Energy
        features[i, 0] = E_total
        
        # Feature 1: Entropy (if energy > 0)
        if E_total > 1e-12:
            probs = np.array(energies) / E_total
            probs = probs[probs > 0]
            features[i, 1] = -np.sum(probs * np.log2(probs))
        
        # Flatten all coefficients for remaining features
        flat = np.concatenate([approx_slice] + detail_slices)
        
        # Feature 2: Variance
        features[i, 2] = np.var(flat, ddof=1)
        
        # Feature 3: RMS (Root Mean Square)
        features[i, 3] = np.sqrt(np.mean(flat ** 2))
        
        # Feature 4: Peak-to-Peak
        features[i, 4] = np.max(flat) - np.min(flat)
        
        # Feature 5: Kurtosis
        features[i, 5] = stats.kurtosis(flat, fisher=False, nan_policy='omit')
        
        # Feature 6: Skewness
        features[i, 6] = stats.skew(flat, nan_policy='omit')
    
    return features


def consolidate_events(binary_flags):
    """
    Consolidate binary flags into events with persistence and merge gap rules.
    
    Parameters
    ----------
    binary_flags : array-like
        Binary array indicating flagged windows
        
    Returns
    -------
    list
        List of event dictionaries with 'start' and 'end' indices
    """
    events, in_event, start = [], False, 0
    
    for i, flag in enumerate(binary_flags):
        if flag and not in_event:
            in_event, start = True, i
        elif not flag and in_event:
            in_event = False
            if i - start >= PERSISTENCE:
                events.append({'start': start, 'end': i - 1})
    
    # Handle event extending to end of signal
    if in_event and len(binary_flags) - start >= PERSISTENCE:
        events.append({'start': start, 'end': len(binary_flags) - 1})
    
    if not events:
        return []
    
    # Merge nearby events
    merged = [events[0]]
    for ev in events[1:]:
        if ev['start'] - merged[-1]['end'] - 1 <= MERGE_GAP:
            merged[-1]['end'] = ev['end']
        else:
            merged.append(ev)
    
    # Filter by minimum duration
    return [ev for ev in merged if ev['end'] - ev['start'] + 1 >= MIN_DURATION]


def compute_z_scores(features):
    """
    Compute robust z-scores using median and MAD (Median Absolute Deviation).
    
    The MAD is scaled by 1.4826 to be comparable to standard deviation for 
    normally distributed data.
    
    Parameters
    ----------
    features : ndarray
        Feature matrix of shape (n_samples, n_features)
        
    Returns
    -------
    ndarray
        Z-score matrix of same shape
    """
    z = np.zeros_like(features)
    for j in range(features.shape[1]):
        col = features[:, j]
        med = np.median(col)
        mad = np.median(np.abs(col - med))
        mad_s = (mad * 1.4826) if mad > 1e-12 else 1e-12
        z[:, j] = (col - med) / mad_s
    return z


def detect_anomalies_z(z_scores, threshold=K):
    """
    Detect anomalies where any feature exceeds the threshold in absolute z-score.
    
    Parameters
    ----------
    z_scores : ndarray
        Z-score matrix
    threshold : float
        Threshold multiplier (default: 3 for 3-sigma)
        
    Returns
    -------
    ndarray
        Binary array indicating anomalous windows
    """
    return (np.abs(z_scores) > threshold).any(axis=1)


def calculate_fpr(binary_flags):
    """
    Calculate False Positive Rate from binary flags.
    
    FPR is computed as the fraction of flagged windows, accounting for 
    event consolidation (persistence, merge gap, minimum duration).
    
    Parameters
    ----------
    binary_flags : array-like
        Binary array indicating flagged windows
        
    Returns
    -------
    float
        False Positive Rate (0.0 to 1.0)
    """
    evs = consolidate_events(binary_flags)
    flagged = sum(int(binary_flags[ev['start']:ev['end'] + 1].sum()) for ev in evs)
    return flagged / len(binary_flags) if len(binary_flags) > 0 else 0.0


def fpr_robust(features_healthy):
    """
    Calculate FPR for the Robust detector on healthy data.
    
    This is the complete pipeline: compute z-scores, detect anomalies,
    consolidate events, and calculate FPR.
    
    Parameters
    ----------
    features_healthy : ndarray
        Feature matrix from healthy data only
        
    Returns
    -------
    float
        False Positive Rate (as fraction, not percentage)
    """
    z = compute_z_scores(features_healthy)
    binary = detect_anomalies_z(z, threshold=K)
    evs = consolidate_events(binary)
    flagged = sum(int(binary[ev['start']:ev['end'] + 1].sum()) for ev in evs)
    return flagged / len(binary) if len(binary) > 0 else 0.0


class StatisticalDetector:
    """
    Base class for statistical fault detectors.
    
    Subclasses implement different decision rules (Primary, Consensus, Robust, Ratio).
    """
    
    def __init__(self, threshold=K):
        self.threshold = threshold
        self.reference_stats = None
    
    def fit(self, features_healthy):
        """
        Fit the detector on healthy reference data.
        
        Parameters
        ----------
        features_healthy : ndarray
            Feature matrix from healthy data
        """
        raise NotImplementedError
    
    def predict(self, features):
        """
        Predict anomaly flags for input features.
        
        Parameters
        ----------
        features : ndarray
            Feature matrix to classify
            
        Returns
        -------
        ndarray
            Binary array indicating detected anomalies
        """
        raise NotImplementedError
    
    def get_detection_latency(self, predictions, true_onset, dt=DT):
        """
        Calculate detection latency relative to true fault onset.
        
        Parameters
        ----------
        predictions : ndarray
            Binary prediction array
        true_onset : float
            True fault injection time
        dt : float
            Time per sample
            
        Returns
        -------
        float or None
            Detection latency in seconds, or None if no detection
        """
        detected_indices = np.where(predictions)[0]
        if len(detected_indices) == 0:
            return None
        detection_time = detected_indices[0] * dt
        return detection_time - true_onset


class RobustDetector(StatisticalDetector):
    """
    Robust statistical detector using median and MAD.
    
    This detector is resistant to contamination from faulty data in the reference,
    making it suitable when only healthy data may not be guaranteed.
    """
    
    def fit(self, features_healthy):
        """Compute median and MAD for each feature."""
        self.medians = np.median(features_healthy, axis=0)
        mads = np.median(np.abs(features_healthy - self.medians), axis=0)
        self.mad_scaled = (mads * 1.4826)
        self.mad_scaled[self.mad_scaled < 1e-12] = 1e-12
    
    def predict(self, features):
        """Flag windows where any feature exceeds threshold in robust z-score."""
        z_scores = (features - self.medians) / self.mad_scaled
        return (np.abs(z_scores) > self.threshold).any(axis=1)


class PrimaryDetector(StatisticalDetector):
    """
    Primary detector using mean and standard deviation.
    
    This is the classical approach but is sensitive to outliers in the reference data.
    """
    
    def fit(self, features_healthy):
        """Compute mean and standard deviation for each feature."""
        self.means = np.mean(features_healthy, axis=0)
        self.stds = np.std(features_healthy, axis=0, ddof=1)
        self.stds[self.stds < 1e-12] = 1e-12
    
    def predict(self, features):
        """Flag windows where any feature exceeds threshold in z-score."""
        z_scores = (features - self.means) / self.stds
        return (np.abs(z_scores) > self.threshold).any(axis=1)


class ConsensusDetector(StatisticalDetector):
    """
    Consensus detector requiring multiple features to exceed threshold simultaneously.
    
    Designed for sharp spikes that affect multiple features at once.
    Uses Peak-to-Peak, Variance, and Entropy features (indices 4, 2, 1).
    """
    
    CONSENSUS_FEATURES = [1, 2, 4]  # Entropy, Variance, P2P
    MIN_CONSENSUS = 2  # Require 2 out of 3 features to exceed threshold
    
    def fit(self, features_healthy):
        """Compute median and MAD for consensus features."""
        self.medians = np.median(features_healthy, axis=0)
        mads = np.median(np.abs(features_healthy - self.medians), axis=0)
        self.mad_scaled = (mads * 1.4826)
        self.mad_scaled[self.mad_scaled < 1e-12] = 1e-12
    
    def predict(self, features):
        """Flag windows where at least MIN_CONSENSUS features exceed threshold."""
        z_scores = (features - self.medians) / self.mad_scaled
        exceeded = np.abs(z_scores[:, self.CONSENSUS_FEATURES]) > self.threshold
        return exceeded.sum(axis=1) >= self.MIN_CONSENSUS


class RatioDetector(StatisticalDetector):
    """
    Ratio detector monitoring Kurtosis/Energy ratio.
    
    Designed for faults that appear as short spikes changing the distribution shape.
    """
    
    KURTOSIS_IDX = 5
    ENERGY_IDX = 0
    
    def fit(self, features_healthy):
        """Compute reference statistics for the ratio."""
        ratios = features_healthy[:, self.KURTOSIS_IDX] / (features_healthy[:, self.ENERGY_IDX] + 1e-12)
        self.ratio_med = np.median(ratios)
        ratio_mad = np.median(np.abs(ratios - self.ratio_med)) * 1.4826
        self.ratio_mad = ratio_mad if ratio_mad > 1e-12 else 1e-12
    
    def predict(self, features):
        """Flag windows where ratio exceeds threshold."""
        ratios = features[:, self.KURTOSIS_IDX] / (features[:, self.ENERGY_IDX] + 1e-12)
        z_scores = (ratios - self.ratio_med) / self.ratio_mad
        return np.abs(z_scores) > self.threshold


def evaluate_detector(detector, features_healthy, features_faulty, true_onset):
    """
    Evaluate a detector's performance on a file.
    
    Parameters
    ----------
    detector : StatisticalDetector
        Fitted detector instance
    features_healthy : ndarray
        Features from healthy portion
    features_faulty : ndarray
        Features from faulty portion
    true_onset : float
        True fault injection time
        
    Returns
    -------
    dict
        Dictionary with 'fpr', 'latency', 'detected' keys
    """
    # Calculate FPR on healthy data
    pred_healthy = detector.predict(features_healthy)
    fpr = calculate_fpr(pred_healthy)
    
    # Calculate detection latency on full data
    features_full = np.vstack([features_healthy, features_faulty])
    pred_full = detector.predict(features_full)
    latency = None
    
    # Find first detection after onset
    healthy_windows = len(features_healthy)
    onset_window = int(true_onset / DT)
    
    for i in range(max(0, onset_window - int(10/DT)), len(pred_full)):
        if pred_full[i]:
            detection_time = i * DT
            latency = detection_time - true_onset
            break
    
    return {
        'fpr': fpr,
        'latency': latency,
        'detected': latency is not None
    }
