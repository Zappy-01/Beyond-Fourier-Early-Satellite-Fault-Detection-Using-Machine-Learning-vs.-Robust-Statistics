"""
Wavelet Benchmarking Module - Phase 1

This module handles the benchmarking of different mother wavelets (Morlet, Mexican Hat, Haar)
for satellite fault detection. It computes wavelet features and calculates Fisher Discriminant
Ratios to determine which wavelet best separates healthy from faulty signal windows.

Based on research: "Beyond Fourier: Early Satellite Fault Detection Using Machine Learning vs. Robust Statistics"
"""

import numpy as np
import pywt
from scipy import stats


# Configuration constants
SCALES = np.arange(1, 64)  # Scales for CWT
NUM_WINDOWS = 20  # Number of windows to divide signal into
DT = 0.1  # Sampling period
FEATURE_NAMES = ['Energy', 'Entropy', 'Variance', 'RMS', 'P2P', 'Kurtosis', 'Skewness']


def compute_wavelet_features(segment, wavelet, scales, dt):
    """
    Compute 7 statistical features from a signal segment using a specified wavelet.
    
    Parameters
    ----------
    segment : array-like
        Input signal segment
    wavelet : str
        Wavelet type ('morl', 'mexh', or 'haar')
    scales : array-like
        Scales for CWT (ignored for Haar DWT)
    dt : float
        Sampling period
        
    Returns
    -------
    ndarray
        Array of 7 features: [Energy, Entropy, Variance, RMS, P2P, Kurtosis, Skewness]
    """
    segment = np.array(segment, dtype=float, copy=True)
    
    if wavelet == 'haar':
        # Use discrete wavelet decomposition for Haar
        max_level = pywt.dwt_max_level(len(segment), filter_len=pywt.Wavelet('haar').dec_len)
        if max_level == 0:
            return np.zeros(7)
        coeffs = pywt.wavedec(segment, wavelet, level=max_level)
        flat = np.concatenate(coeffs)
        e_per_scale = np.array([np.sum(np.abs(c)**2) for c in coeffs])
    else:
        # Use continuous wavelet transform for Morlet and Mexican Hat
        coeffs, _ = pywt.cwt(segment, scales, wavelet, sampling_period=dt)
        flat = coeffs.flatten()
        e_per_scale = np.sum(np.abs(coeffs)**2, axis=1)
        
    if len(flat) < 2:
        return np.zeros(7)
    
    # Compute statistical features
    var = np.var(flat, ddof=1)
    rms = np.sqrt(np.mean(flat**2))
    p2p = np.max(flat) - np.min(flat)
    kurt = stats.kurtosis(flat, fisher=False, nan_policy='omit') if var > 0 else 0
    skew = stats.skew(flat, nan_policy='omit') if var > 0 else 0
    
    # Compute energy and entropy
    total_e = np.sum(e_per_scale)
    if total_e > 0:
        probs = e_per_scale / total_e
        probs = probs[probs > 0]
        ent = -np.sum(probs * np.log2(probs))
    else:
        ent = 0
    
    return np.array([total_e, ent, var, rms, p2p, kurt, skew])


def calculate_fdr(feat_h, feat_f):
    """
    Calculate Fisher Discriminant Ratio between healthy and faulty feature sets.
    
    FDR = (μ_faulty − μ_healthy)² / (σ²_healthy + σ²_faulty)
    
    A large FDR indicates the feature separates healthy from faulty well.
    
    Parameters
    ----------
    feat_h : array-like
        Features from healthy windows
    feat_f : array-like
        Features from faulty windows
        
    Returns
    -------
    float
        Fisher Discriminant Ratio
    """
    mu_h, mu_f = np.mean(feat_h), np.mean(feat_f)
    var_h, var_f = np.var(feat_h, ddof=1), np.var(feat_f, ddof=1)
    denom = var_h + var_f
    return ((mu_f - mu_h)**2) / denom if denom > 0 else 0.0


def benchmark_wavelets(signal_data, wavelets=['morl', 'mexh', 'haar']):
    """
    Benchmark multiple wavelets on provided signal data.
    
    Parameters
    ----------
    signal_data : dict
        Dictionary with keys 'hc', 'hv', 'fc', 'fv' containing healthy/faulty signals
    wavelets : list
        List of wavelet names to benchmark
        
    Returns
    -------
    dict
        Dictionary mapping wavelet names to their total separability scores
    """
    wavelet_scores = {w: [] for w in wavelets}
    
    for w in wavelets:
        total_fdr = 0
        for col_h, col_f in [('hc', 'fc'), ('hv', 'fv')]:
            sig_h = signal_data[col_h].to_numpy(copy=True)
            sig_f = signal_data[col_f].to_numpy(copy=True)
            win_size = len(sig_h) // NUM_WINDOWS
            
            feats_h = {k: [] for k in FEATURE_NAMES}
            feats_f = {k: [] for k in FEATURE_NAMES}
            
            for i in range(NUM_WINDOWS):
                s, e = i * win_size, (i + 1) * win_size
                vals_h = compute_wavelet_features(sig_h[s:e], w, SCALES, DT)
                vals_f = compute_wavelet_features(sig_f[s:e], w, SCALES, DT)
                for k, v_h, v_f in zip(FEATURE_NAMES, vals_h, vals_f):
                    feats_h[k].append(v_h)
                    feats_f[k].append(v_f)
            
            for k in FEATURE_NAMES:
                total_fdr += calculate_fdr(np.array(feats_h[k]), np.array(feats_f[k]))
        
        wavelet_scores[w].append(total_fdr)
    
    # Return aggregated scores
    agg_scores = {w: np.mean(s) for w, s in wavelet_scores.items()}
    return agg_scores


def generate_wavelet_shapes(t_range=(-5, 5), num_points=200):
    """
    Generate sample wavelet shapes for visualization.
    
    Parameters
    ----------
    t_range : tuple
        Time range (min, max)
    num_points : int
        Number of points to sample
        
    Returns
    -------
    dict
        Dictionary mapping wavelet names to their (t, psi) arrays
    """
    t = np.linspace(t_range[0], t_range[1], num_points)
    
    wavelets = {}
    
    # Morlet: damped oscillation
    morlet = np.real(np.exp(1j * 5.0 * t)) * np.exp(-0.5 * t ** 2)
    wavelets['morl'] = (t, morlet)
    
    # Mexican Hat: bell-shaped pulse (second derivative of Gaussian)
    mexh = (1 - t ** 2) * np.exp(-0.5 * t ** 2)
    wavelets['mexh'] = (t, mexh)
    
    # Haar: step function
    t_h = np.linspace(0, 1, num_points)
    haar = np.where(t_h < 0.5, 1.0, -1.0)
    wavelets['haar'] = (t_h, haar)
    
    return wavelets
