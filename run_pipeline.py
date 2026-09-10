"""
Main Pipeline - Satellite Fault Detection

This script orchestrates the complete fault detection pipeline, integrating:
1. Wavelet Benchmarking (Phase 1)
2. Statistical Detector Evaluation (Week 7)
3. SVM Training and Evaluation (Week 9)
4. Visualization Generation

Based on research: "Beyond Fourier: Early Satellite Fault Detection Using Machine Learning vs. Robust Statistics"

Usage:
    python run_pipeline.py --raw_dir <path> --summary_file <path> --out_dir <path>
"""

import os
import argparse
import numpy as np
import pandas as pd

from src.wavelet_benchmark import benchmark_wavelets, generate_wavelet_shapes
from src.statistical_detector import extract_dense_features, fpr_robust
from src.svm_detector import extract_full_dataset, run_5fold_cv, calculate_metrics
from src.visualization import generate_all_figures, get_font_sizes


# Default configuration
DEFAULT_CONFIG = {
    'X_LABEL_SCALE': 6,
    'Y_LABEL_SCALE': 6,
    'DT': 0.1,
    'WINDOW_SIZE': 128,
    'STEP': 10,
    'K': 3,
    'COL_I': 8,   # Column index for wheel_i signal
    'COL_J': 9,   # Column index for wheel_w signal
    'COL_X': 23,  # Column index for expected current
    'COL_Y': 24,  # Column index for expected velocity
}

# Target files for wavelet benchmarking
TARGET_FILES = [
    "55_12_1_1_19.3007_23.0744_52.2731_38.3541_0.030153_5.6374.csv",
    "56_12_1_1_15.0004_14.5984_52.6181_45.2467_0.029682_6.4313.csv",
    "94_12_1_1_18.5066_5.1343_45.0543_39.6693_0.031413_5.4014.csv",
    "40_13_1_1_15.6565_12.0145_53.475_51.8986_0.029291_6.147.csv",
    "7_13_1_1_23.6799_18.5747_48.9435_48.5783_0.028375_6.1866.csv",
]

# Detector column mappings
DET_COLS = {
    'Primary': 'latency_system_pri_sec',
    'Consensus': 'latency_system_con_sec',
    'Robust': 'latency_system_rob_sec',
    'Ratio': 'latency_system_rat_sec'
}

DET_ABBR = {
    'Primary': 'P',
    'Consensus': 'C',
    'Robust': 'Robust',
    'Ratio': 'Ra'
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Satellite Fault Detection Pipeline'
    )
    parser.add_argument(
        '--raw_dir', type=str, required=True,
        help='Directory containing raw CSV telemetry files'
    )
    parser.add_argument(
        '--summary_file', type=str, required=True,
        help='Path to Week 7 summary CSV with detector latencies'
    )
    parser.add_argument(
        '--out_dir', type=str, default='./results',
        help='Output directory for figures and stats'
    )
    parser.add_argument(
        '--cache_dir', type=str, default='./cache',
        help='Directory for caching intermediate results'
    )
    return parser.parse_args()


def load_detector_summary(summary_path):
    """
    Load and process the Week 7 detector summary file.
    
    Returns statistics for each detector architecture.
    """
    df = pd.read_csv(summary_path)
    
    # Identify faulty files (scenario != 0)
    faulty_mask = np.array([int(f.split('_')[1]) != 0 for f in df['filename']])
    
    det_stats = {}
    for name, col in DET_COLS.items():
        vals = df.loc[faulty_mask, col].dropna().to_numpy(dtype=float)
        n_faulty = int(faulty_mask.sum())
        
        det_stats[name] = {
            'mean': float(np.mean(vals)) if vals.size else np.nan,
            'std': float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
            'rate': 100.0 * vals.size / n_faulty if n_faulty else 0.0,
            'vals': vals,
            'early': int(np.sum(vals < 0))
        }
    
    return det_stats, df, faulty_mask


def compute_robust_fpr(raw_dir, col_i=8, col_j=9):
    """
    Compute honest Robust detector FPR across all files.
    
    This evaluates the detector on healthy portions only.
    """
    print("Computing honest Robust (Median/MAD) FPR across all files...")
    rob_fpr_list = []
    
    files = [f for f in sorted(os.listdir(raw_dir)) 
             if f.endswith('.csv') and f.lower() != 'input.csv']
    
    for idx, fname in enumerate(files):
        df = pd.read_csv(os.path.join(raw_dir, fname), header=0)
        
        fi = extract_dense_features(df.iloc[:, col_i].to_numpy(dtype=float))
        fw = extract_dense_features(df.iloc[:, col_j].to_numpy(dtype=float))
        
        ni, nw = len(fi), len(fw)
        fpr_i = fpr_robust(fi)
        fpr_w = fpr_robust(fw)
        
        # Weighted average FPR
        rob_fpr_list.append(100.0 * (fpr_i * ni + fpr_w * nw) / (ni + nw))
        
        if (idx + 1) % 20 == 0:
            print(f"  ...{idx + 1}/{len(files)} files")
    
    return np.array(rob_fpr_list)


def extract_robust_latencies(df, faulty_mask):
    """Extract robust detector per-file latency data."""
    valid_mask = faulty_mask & df['latency_system_rob_sec'].notna()
    robust_latencies = df.loc[valid_mask, 'latency_system_rob_sec'].to_numpy(dtype=float)
    robust_files = df.loc[valid_mask, 'filename'].apply(
        lambda x: int(x.split('_')[0])
    ).to_numpy()
    return robust_latencies, robust_files


def run_wavelet_benchmark(raw_dir, target_files, col_i=8, col_j=9, col_x=23, col_y=24):
    """
    Run Phase 1 wavelet benchmarking on target files.
    
    Returns aggregated separability scores for each wavelet.
    """
    print("Running Phase 1 Wavelet Benchmarking...")
    from src.wavelet_benchmark import FEATURE_NAMES, SCALES, DT, NUM_WINDOWS
    from src.wavelet_benchmark import compute_wavelet_features, calculate_fdr
    
    wavelets = ['morl', 'mexh', 'haar']
    wavelet_scores = {w: [] for w in wavelets}
    
    for fname in target_files:
        path = os.path.join(raw_dir, fname)
        if not os.path.exists(path):
            print(f"  Warning: {fname} not found, skipping...")
            continue
        
        d = pd.read_csv(path, usecols=[col_i, col_j, col_x, col_y], header=0)
        d.columns = ['hc', 'hv', 'fc', 'fv']
        d = d.fillna(0)
        
        for w in wavelets:
            total_fdr = 0
            for col_h, col_f in [('hc', 'fc'), ('hv', 'fv')]:
                sig_h = d[col_h].to_numpy(copy=True)
                sig_f = d[col_f].to_numpy(copy=True)
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
    
    agg_scores = {w: np.mean(s) for w, s in wavelet_scores.items()}
    return agg_scores


def write_summary(det_stats, svm_results, rob_fpr_vals, svm_all_fprs, 
                  agg_scores, out_dir):
    """Write text summary of all results."""
    out_txt = []
    out_txt.append("================ FINAL POSTER NUMBERS (100 FILES) ================")
    
    out_txt.append("\n[DETECTORS - LATENCY & EARLY DETECTION]")
    for n in DET_COLS:
        s = det_stats[n]
        out_txt.append(
            f"{n:10s}: Rate={s['rate']:5.1f}% | Mean={s['mean']:+7.3f}s | "
            f"Std Dev={s['std']:6.3f}s | Early={s['early']}"
        )
    
    out_txt.append("\n[SVM 5-FOLD CROSS VALIDATION (100 FILES)]")
    svm_lats = [r['Latency'] for r in svm_results if not np.isnan(r['Latency'])]
    faulty_count = sum(1 for r in svm_results if r['Faulty'])
    svm_det_rate = 100.0 * len(svm_lats) / faulty_count if faulty_count else 0
    svm_early = int(np.sum(np.array(svm_lats) < 0))
    
    out_txt.append(f"Detection Rate: {len(svm_lats)}/{faulty_count} ({svm_det_rate:.1f}%)")
    out_txt.append(
        f"Mean Latency  : {np.mean(svm_lats):+.3f}s | "
        f"Std Dev={np.std(svm_lats, ddof=1):.3f}s | Early={svm_early}"
    )
    
    out_txt.append("\n[FALSE POSITIVE RATES (FPR) - ALL 100 FILES]")
    out_txt.append(f"Robust (Median/MAD): Mean={np.mean(rob_fpr_vals):.2f}% | Max={np.max(rob_fpr_vals):.2f}%")
    out_txt.append(f"SVM (Digital Twin) : Mean={np.mean(svm_all_fprs):.2f}% | Max={np.max(svm_all_fprs):.2f}%")
    
    out_txt.append("\n[PHASE 1 WAVELET BENCHMARKING]")
    wnames = ['Morlet', 'Mexican Hat', 'Haar']
    wkeys = ['morl', 'mexh', 'haar']
    for w, k in zip(wnames, wkeys):
        out_txt.append(f"  {w:12s}: {agg_scores[k]:.2f}")
    
    out_txt.append("==================================================================")
    
    text_summary = "\n".join(out_txt)
    print(text_summary)
    
    with open(os.path.join(out_dir, "poster_stats.txt"), 'w') as f:
        f.write(text_summary)
    
    return text_summary


def main():
    """Main pipeline execution."""
    args = parse_args()
    
    # Create output directories
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    
    # Get font sizes
    F = get_font_sizes(DEFAULT_CONFIG['X_LABEL_SCALE'], DEFAULT_CONFIG['Y_LABEL_SCALE'])
    
    print("=" * 70)
    print("SATELLITE FAULT DETECTION PIPELINE")
    print("=" * 70)
    
    # ===== Step 1: Load Detector Summary =====
    print("\n[Step 1/5] Loading detector summary...")
    det_stats, df, faulty_mask = load_detector_summary(args.summary_file)
    print(f"  Loaded {len(df)} files, {faulty_mask.sum()} faulty")
    
    # ===== Step 2: Extract Robust Latencies =====
    print("\n[Step 2/5] Extracting robust detector latencies...")
    robust_latencies, robust_files = extract_robust_latencies(df, faulty_mask)
    print(f"  Extracted {len(robust_latencies)} latency values")
    
    # ===== Step 3: Compute Robust FPR =====
    print("\n[Step 3/5] Computing robust detector FPR...")
    rob_fpr_vals = compute_robust_fpr(args.raw_dir, 
                                       DEFAULT_CONFIG['COL_I'], 
                                       DEFAULT_CONFIG['COL_J'])
    print(f"  Mean FPR: {np.mean(rob_fpr_vals):.2f}%")
    
    # ===== Step 4: Run SVM 5-Fold CV =====
    print("\n[Step 4/5] Running SVM 5-fold cross-validation...")
    cache_path = os.path.join(args.cache_dir, 'full_svm_features_5fold.npz')
    
    X, y, groups, end_times, true_onsets, is_faulty = extract_full_dataset(
        args.raw_dir, cache_path
    )
    
    svm_results = run_5fold_cv(X, y, groups, end_times, true_onsets, is_faulty)
    svm_metrics = calculate_metrics(svm_results)
    print(f"  Detection rate: {svm_metrics['detection_rate']*100:.1f}%")
    print(f"  Mean latency: {svm_metrics['mean_latency']:+.3f}s")
    print(f"  Mean FPR: {svm_metrics['mean_fpr']:.2f}%")
    
    # ===== Step 5: Run Wavelet Benchmark =====
    print("\n[Step 5/5] Running wavelet benchmark...")
    agg_scores = run_wavelet_benchmark(
        args.raw_dir, TARGET_FILES,
        DEFAULT_CONFIG['COL_I'], DEFAULT_CONFIG['COL_J'],
        DEFAULT_CONFIG['COL_X'], DEFAULT_CONFIG['COL_Y']
    )
    best_wavelet = max(agg_scores, key=agg_scores.get)
    print(f"  Best wavelet: {best_wavelet} (score: {agg_scores[best_wavelet]:.2f})")
    
    # ===== Generate Figures =====
    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)
    
    # Prepare SVM data for visualization
    svm_all_files = np.array([r['File'] for r in svm_results])
    svm_all_fprs = np.array([r['FPR'] for r in svm_results])
    
    generate_all_figures(
        det_stats=det_stats,
        rob_fpr_vals=rob_fpr_vals,
        svm_results=svm_results,
        robust_latencies=robust_latencies,
        robust_files=robust_files,
        agg_scores=agg_scores,
        out_dir=args.out_dir,
        F=F
    )
    
    # ===== Write Summary =====
    print("\n" + "=" * 70)
    print("WRITING SUMMARY")
    print("=" * 70)
    write_summary(det_stats, svm_results, rob_fpr_vals, svm_all_fprs, 
                  agg_scores, args.out_dir)
    
    print(f"\n[SUCCESS] Pipeline complete!")
    print(f"Figures and stats written to: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
