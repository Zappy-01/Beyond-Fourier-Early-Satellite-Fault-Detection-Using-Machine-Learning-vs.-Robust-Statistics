import os
import glob
import numpy as np
import pandas as pd
import pywt
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================= CONFIG ======================================
# INDEPENDENT AXIS LABEL SCALING (1 to 10)
X_LABEL_SCALE = 6  
Y_LABEL_SCALE = 6

RAW_DIR     = r"C:\Users\mcfor\Documents\Python Projs\Beyond The Fourier\Week 4"
W7_SUMMARY  = r"C:\Users\mcfor\Documents\Python Projs\Beyond The Fourier\Updated week 7\generated\updated_week7_summary.csv"
OUT_DIR     = r"C:\Users\mcfor\Documents\Python Projs\Beyond The Fourier\poster_figures"
SVM_CACHE   = r"C:\Users\mcfor\Documents\Python Projs\Beyond The Fourier\Week 9\generated\full_svm_features_5fold.npz"

# Strict Color Coding
COLORS = {
    'Robust': '#2ecc71',    # Green
    'SVM': '#e67e22',       # Orange
    'Primary': '#3498db',   # Blue
    'Consensus': '#9b59b6', # Purple
    'Ratio': '#e74c3c',     # Red
    'Morlet': '#3498db',    # Blue
    'Mexh': '#9b59b6',      # Purple
    'Haar': '#1abc9c'       # Teal
}

COL_I, COL_J, COL_X, COL_Y = 8, 9, 23, 24
DT = 0.1
WINDOW_SIZE = 128
STEP = 10
K = 3
PERSISTENCE, MERGE_GAP, MIN_DURATION = 1, 1, 1
FEATURE_NAMES = ['Energy', 'Entropy', 'Variance', 'RMS', 'P2P', 'Kurtosis', 'Skewness']
DET_COLS = {'Primary': 'latency_system_pri_sec', 'Consensus': 'latency_system_con_sec',
            'Robust': 'latency_system_rob_sec', 'Ratio': 'latency_system_rat_sec'}
DET_ABBR = {'Primary': 'P', 'Consensus': 'C', 'Robust': 'Robust', 'Ratio': 'Ra'}

TARGET_FILES = [
    "55_12_1_1_19.3007_23.0744_52.2731_38.3541_0.030153_5.6374.csv",
    "56_12_1_1_15.0004_14.5984_52.6181_45.2467_0.029682_6.4313.csv",
    "94_12_1_1_18.5066_5.1343_45.0543_39.6693_0.031413_5.4014.csv",
    "40_13_1_1_15.6565_12.0145_53.475_51.8986_0.029291_6.147.csv",
    "7_13_1_1_23.6799_18.5747_48.9435_48.5783_0.028375_6.1866.csv",
]
SCALES = np.arange(1, 64)
CWT_SCALES = np.arange(1, 32)
NUM_WINDOWS = 20
# ===========================================================================

def get_font_sizes():
    x_s = int(min(10, max(1, X_LABEL_SCALE)))
    y_s = int(min(10, max(1, Y_LABEL_SCALE)))
    return {
        "x_label": 6 + 2 * x_s,
        "y_label": 6 + 2 * y_s,
        "tick": 4 + 1.5 * max(x_s, y_s),
        "title": 8 + 2 * max(x_s, y_s),
        "legend": 4 + 1.5 * max(x_s, y_s),
        "annot": 4 + 1.5 * max(x_s, y_s)
    }

def apply_axes(ax, F, xlabel=None, ylabel=None, title=None, legend=False):
    ax.tick_params(labelsize=F["tick"])
    if xlabel: ax.set_xlabel(xlabel, fontsize=F["x_label"])
    if ylabel: ax.set_ylabel(ylabel, fontsize=F["y_label"])
    if title: ax.set_title(title, fontsize=F["title"])
    if legend: ax.legend(fontsize=F["legend"])

# ------------------- Week 7 Pipeline Functions -----------------------------
def extract_dense_features(signal):
    signal = np.array(signal, dtype=float, copy=True)
    required = 2 ** 3
    n = len(signal)
    if n % required != 0:
        signal = np.pad(signal, (0, required - (n % required)), mode='edge')
    coeffs = pywt.swt(signal, wavelet='haar', level=3, trim_approx=False)
    detail_coeffs = [level[1] for level in coeffs]
    approx_coeff = coeffs[-1][0]
    num_windows = ((n - WINDOW_SIZE) // STEP) + 1
    features = np.zeros((num_windows, 7))
    for i in range(num_windows):
        s0, e0 = i * STEP, i * STEP + WINDOW_SIZE
        detail_slices = [cd[s0:e0] for cd in detail_coeffs]
        approx_slice = approx_coeff[s0:e0]
        energies = [np.sum(np.square(s)) for s in detail_slices]
        E_total = sum(energies)
        features[i, 0] = E_total
        if E_total > 1e-12:
            probs = np.array(energies) / E_total
            probs = probs[probs > 0]
            features[i, 1] = -np.sum(probs * np.log2(probs))
        flat = np.concatenate([approx_slice] + detail_slices)
        features[i, 2] = np.var(flat, ddof=1)
        features[i, 3] = np.sqrt(np.mean(flat ** 2))
        features[i, 4] = np.max(flat) - np.min(flat)
        features[i, 5] = stats.kurtosis(flat, fisher=False, nan_policy='omit')
        features[i, 6] = stats.skew(flat, nan_policy='omit')
    return features

def consolidate_events(binary_flags):
    events, in_event, start = [], False, 0
    for i, flag in enumerate(binary_flags):
        if flag and not in_event:
            in_event, start = True, i
        elif not flag and in_event:
            in_event = False
            if i - start >= PERSISTENCE:
                events.append({'start': start, 'end': i - 1})
    if in_event and len(binary_flags) - start >= PERSISTENCE:
        events.append({'start': start, 'end': len(binary_flags) - 1})
    if not events: return []
    merged = [events[0]]
    for ev in events[1:]:
        if ev['start'] - merged[-1]['end'] - 1 <= MERGE_GAP:
            merged[-1]['end'] = ev['end']
        else:
            merged.append(ev)
    return [ev for ev in merged if ev['end'] - ev['start'] + 1 >= MIN_DURATION]

def fpr_robust(features_healthy):
    z = np.zeros_like(features_healthy)
    for j in range(features_healthy.shape[1]):
        col = features_healthy[:, j]
        med = np.median(col)
        mad = np.median(np.abs(col - med))
        mad_s = (mad * 1.4826) if mad > 1e-12 else 1e-12
        z[:, j] = (col - med) / mad_s
    binary = (np.abs(z) > K).any(axis=1)
    evs = consolidate_events(binary)
    flagged = sum(int(binary[ev['start']:ev['end'] + 1].sum()) for ev in evs)
    return flagged / len(binary) if len(binary) > 0 else 0.0

# ------------------- Phase 1 Wavelet Benchmarking --------------------------
def compute_wavelet_features(segment, wavelet, scales, dt):
    segment = np.array(segment, dtype=float, copy=True)
    if wavelet == 'haar':
        max_level = pywt.dwt_max_level(len(segment), filter_len=pywt.Wavelet('haar').dec_len)
        if max_level == 0: return np.zeros(7)
        coeffs = pywt.wavedec(segment, wavelet, level=max_level)
        flat = np.concatenate(coeffs)
        e_per_scale = np.array([np.sum(np.abs(c)**2) for c in coeffs])
    else:
        coeffs, _ = pywt.cwt(segment, scales, wavelet, sampling_period=dt)
        flat = coeffs.flatten()
        e_per_scale = np.sum(np.abs(coeffs)**2, axis=1)
        
    if len(flat) < 2: return np.zeros(7)
    var = np.var(flat, ddof=1)
    rms = np.sqrt(np.mean(flat**2))
    p2p = np.max(flat) - np.min(flat)
    kurt = stats.kurtosis(flat, fisher=False, nan_policy='omit') if var > 0 else 0
    skew = stats.skew(flat, nan_policy='omit') if var > 0 else 0
    
    total_e = np.sum(e_per_scale)
    if total_e > 0:
        probs = e_per_scale / total_e
        probs = probs[probs > 0]
        ent = -np.sum(probs * np.log2(probs))
    else:
        ent = 0
    return np.array([total_e, ent, var, rms, p2p, kurt, skew])

def calculate_fdr(feat_h, feat_f):
    mu_h, mu_f = np.mean(feat_h), np.mean(feat_f)
    var_h, var_f = np.var(feat_h, ddof=1), np.var(feat_f, ddof=1)
    denom = var_h + var_f
    return ((mu_f - mu_h)**2) / denom if denom > 0 else 0.0

# ------------------- Week 9 SVM 5-Fold Extraction & CV ---------------------
def compute_cwt_features(window_data):
    cwtmatr, _ = pywt.cwt(window_data, CWT_SCALES, 'morl', sampling_period=DT)
    abs_cwt = np.abs(cwtmatr)
    scale_energies = np.sum(abs_cwt**2, axis=1)
    total_energy = np.sum(scale_energies) + 1e-12
    norm_energies = scale_energies / total_energy
    band1 = np.sum(norm_energies[0:8])
    band2 = np.sum(norm_energies[8:16])
    band3 = np.sum(norm_energies[16:24])
    band4 = np.sum(norm_energies[24:])
    probs = norm_energies + 1e-12
    entropy_val = -np.sum(probs * np.log2(probs))
    indices = np.arange(1, len(scale_energies) + 1)
    centroid_val = np.sum(indices * norm_energies)
    return [band1, band2, band3, band4, entropy_val, centroid_val]

def extract_full_svm_dataset():
    print("Extracting CWT features for all 100 files (This may take 10-20 mins)...")
    file_paths = glob.glob(os.path.join(RAW_DIR, "*.csv"))
    Xi_feat_list, y_list, group_list = [], [], []
    end_times_list, true_onsets_list, is_faulty_list = [], [], []
    
    for filepath in file_paths:
        name_no_ext = os.path.splitext(os.path.basename(filepath))[0]
        parts = name_no_ext.split('_')
        if not parts[0].isdigit(): continue
        file_num = int(parts[0])
        scenario = int(parts[1])
        if scenario == 0:
            true_onset = np.inf
            is_faulty = 0
        else:
            kt_inc = float(parts[4])
            vbus_inc = float(parts[5])
            onsets = []
            if int(parts[2]) == 1: onsets.append(kt_inc)
            if int(parts[3]) == 1: onsets.append(vbus_inc)
            true_onset = min(onsets) if onsets else np.inf
            is_faulty = 1
            
        df = pd.read_csv(filepath)
        i_healthy_cols = [f'wheel{w}_i_healthy' for w in range(1, 5)]
        i_faulty_cols = [f'wheel{w}_i_faulty' for w in range(1, 5)]
        if not all(col in df.columns for col in i_healthy_cols + i_faulty_cols): continue
        
        i_expected = df[i_healthy_cols].values
        i_actual = df[i_faulty_cols].values if is_faulty else df[i_healthy_cols].values
        n_samples = len(i_actual)
        
        for start_idx in range(0, n_samples - WINDOW_SIZE + 1, STEP):
            end_idx = start_idx + WINDOW_SIZE
            end_time = end_idx * DT
            label = 1 if end_time >= true_onset else 0
            expected_window = i_expected[start_idx:end_idx, :]
            actual_window = i_actual[start_idx:end_idx, :]
            feats_expected, feats_actual = [], []
            for col_idx in range(4):
                feats_expected.extend(compute_cwt_features(expected_window[:, col_idx]))
                feats_actual.extend(compute_cwt_features(actual_window[:, col_idx]))
            residual_features = np.array(feats_actual) - np.array(feats_expected)
            Xi_feat_list.append(residual_features)
            y_list.append(label)
            group_list.append(file_num)
            end_times_list.append(end_time)
            true_onsets_list.append(true_onset)
            is_faulty_list.append(is_faulty)
            
    np.savez_compressed(SVM_CACHE,
        X=np.array(Xi_feat_list), y=np.array(y_list), groups=np.array(group_list),
        end_times=np.array(end_times_list), true_onsets=np.array(true_onsets_list),
        is_faulty=np.array(is_faulty_list))
    print("Extraction complete and cached.")
    return np.array(Xi_feat_list), np.array(y_list), np.array(group_list), np.array(end_times_list), np.array(true_onsets_list), np.array(is_faulty_list)

def run_5fold_svm():
    if not os.path.exists(SVM_CACHE):
        X, y, groups, end_times, true_onsets, is_faulty = extract_full_svm_dataset()
    else:
        print("Loading cached SVM features...")
        data = np.load(SVM_CACHE)
        X, y, groups = data['X'], data['y'], data['groups']
        end_times, true_onsets, is_faulty = data['end_times'], data['true_onsets'], data['is_faulty']

    print("Executing 5-Pass Cross-Validation (Out-of-Fold Predictions)...")
    gkf = GroupKFold(n_splits=5)
    all_preds = np.zeros(len(y), dtype=int)
    
    pipeline = Pipeline([('scaler', StandardScaler()), ('svm', SVC(C=0.1, gamma='scale', kernel='rbf', random_state=42))])
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        print(f"  Pass {fold+1}/5: Training on 80 files, predicting 20 files...")
        pipeline.fit(X[train_idx], y[train_idx])
        all_preds[test_idx] = pipeline.predict(X[test_idx])
        
    results = []
    for file_num in np.unique(groups):
        file_mask = (groups == file_num)
        preds = all_preds[file_mask]
        et = end_times[file_mask]
        to = true_onsets[file_mask][0]
        is_f = is_faulty[file_mask][0]
        
        healthy_mask = et < to
        fp_count = int(np.sum(preds[healthy_mask] == 1))
        n_healthy = int(np.sum(healthy_mask))
        fpr = (fp_count / n_healthy * 100) if n_healthy > 0 else 0.0
        
        latency = np.nan
        if is_f:
            valid_flags = et[(preds == 1) & (et >= to - 10.0)]
            if len(valid_flags) > 0:
                latency = valid_flags[0] - to
                
        results.append({'File': int(file_num), 'Faulty': bool(is_f), 'Latency': latency, 'FPR': fpr})
    return results

# ------------------- Main Execution ----------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    F = get_font_sizes()
    
    # 1. Week 7 Detector Latencies
    df = pd.read_csv(W7_SUMMARY)
    faulty_mask = np.array([int(f.split('_')[1]) != 0 for f in df['filename']])
    det_stats = {}
    for name, col in DET_COLS.items():
        vals = df.loc[faulty_mask, col].dropna().to_numpy(dtype=float)
        n_faulty = int(faulty_mask.sum())
        det_stats[name] = {
            'mean': float(np.mean(vals)) if vals.size else np.nan,
            'std': float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
            'rate': 100.0 * vals.size / n_faulty if n_faulty else 0.0,
            'vals': vals, 'early': int(np.sum(vals < 0))
        }
    rob = det_stats['Robust']

    # Extract Robust Latency Per-File Data
    valid_rob_mask = faulty_mask & df['latency_system_rob_sec'].notna()
    robust_latencies = df.loc[valid_rob_mask, 'latency_system_rob_sec'].to_numpy(dtype=float)
    robust_files = df.loc[valid_rob_mask, 'filename'].apply(lambda x: int(x.split('_')[0])).to_numpy()

    # 2. Honest Robust FPR
    print("Computing honest Robust (Median/MAD) FPR across all files...")
    rob_fpr_list = []
    files = [f for f in sorted(os.listdir(RAW_DIR)) if f.endswith('.csv') and f.lower() != 'input.csv']
    for idx, fname in enumerate(files):
        dfr = pd.read_csv(os.path.join(RAW_DIR, fname), header=0)
        fi = extract_dense_features(dfr.iloc[:, COL_I].to_numpy(dtype=float))
        fw = extract_dense_features(dfr.iloc[:, COL_J].to_numpy(dtype=float))
        ni, nw = len(fi), len(fw)
        fpr_i, fpr_w = fpr_robust(fi), fpr_robust(fw)
        rob_fpr_list.append(100.0 * (fpr_i * ni + fpr_w * nw) / (ni + nw))
        if (idx + 1) % 20 == 0: print(f"  ...{idx + 1}/{len(files)} files")
    rob_fpr_vals = np.array(rob_fpr_list)

    # 3. Week 9 SVM (5-Fold CV)
    svm_results = run_5fold_svm()
    
    # Extract SVM Per-File Data
    svm_all_files = np.array([r['File'] for r in svm_results])
    svm_all_fprs = np.array([r['FPR'] for r in svm_results])
    
    svm_faulty_mask = np.array([r['Faulty'] for r in svm_results])
    svm_latencies_raw = np.array([r['Latency'] for r in svm_results])
    valid_svm_lat_mask = svm_faulty_mask & ~np.isnan(svm_latencies_raw)
    svm_lats = svm_latencies_raw[valid_svm_lat_mask]
    svm_files = svm_all_files[valid_svm_lat_mask]

    # 4. Phase 1 Wavelet Benchmarking
    wavelets = ['morl', 'mexh', 'haar']
    wavelet_scores = {w: [] for w in wavelets}
    print("Running Phase 1 Wavelet Benchmarking...")
    for fname in TARGET_FILES:
        path = os.path.join(RAW_DIR, fname)
        if not os.path.exists(path): continue
        d = pd.read_csv(path, usecols=[COL_I, COL_J, COL_X, COL_Y], header=0)
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
                        feats_h[k].append(v_h); feats_f[k].append(v_f)
                for k in FEATURE_NAMES:
                    total_fdr += calculate_fdr(np.array(feats_h[k]), np.array(feats_f[k]))
            wavelet_scores[w].append(total_fdr)
    agg_scores = {w: np.mean(s) for w, s in wavelet_scores.items()}

    # ================= CHART GENERATION =================
    rng = np.random.default_rng(42)
    
    # Plot 1: Detector Benchmark
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(DET_COLS.keys())
    xs = np.arange(len(names))
    means = [det_stats[n]['mean'] if not np.isnan(det_stats[n]['mean']) else 0.0 for n in names]
    stds = [det_stats[n]['std'] for n in names]
    ax.bar(xs, means, yerr=stds, capsize=10, ecolor='black',
           color=[COLORS[n] for n in names], edgecolor='black', linewidth=1.0)
    for i, n in enumerate(names):
        if np.isnan(det_stats[n]['mean']):
            ax.annotate(f"{det_stats[n]['rate']:.0f}%", (xs[i], 0), ha='center', va='bottom',
                        fontsize=F['annot'], color='red', fontweight='bold')
        else:
            ax.annotate(f"{means[i]:+.2f}s", (xs[i], means[i] + stds[i]), ha='center',
                        va='bottom', fontsize=F['annot'], fontweight='bold')
    ax.set_xticks(xs); ax.set_xticklabels([DET_ABBR[n] for n in names])
    ax.axhline(0, color='black', lw=0.8)
    apply_axes(ax, F, xlabel="Detector Architecture", ylabel="Latency (s)", title="Detector Benchmark")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "detector_benchmark.png"), dpi=200); plt.close(fig)

    # Plot 2: FPR Trade-Off
    fig, ax = plt.subplots(figsize=(8, 5))
    x_rob = 0 + rng.uniform(-0.1, 0.1, len(rob_fpr_vals))
    x_svm = 1 + rng.uniform(-0.1, 0.1, len(svm_all_fprs))
    ax.scatter(x_rob, rob_fpr_vals, color=COLORS['Robust'], alpha=0.6, label='Per-File')
    ax.scatter(x_svm, svm_all_fprs, color=COLORS['SVM'], alpha=0.6, label='Per-File')
    ax.scatter([0], [np.mean(rob_fpr_vals)], color=COLORS['Robust'], s=150, marker='s', edgecolor='black', zorder=5)
    ax.scatter([1], [np.mean(svm_all_fprs)], color=COLORS['SVM'], s=150, marker='s', edgecolor='black', zorder=5)
    ax.axhline(1.0, color='red', linestyle='--', linewidth=2, label='1% Aerospace Target')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Robust', 'SVM'])
    apply_axes(ax, F, xlabel="Architecture", ylabel="False Positive Rate (%)", title="FPR Trade-Off", legend=True)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "fpr_comparison.png"), dpi=200); plt.close(fig)

    # Plot 3: SVM Per-File Latency
    if svm_lats.size:
        fig, ax = plt.subplots(figsize=(12, 5))
        order = np.argsort(svm_files)
        colors_svm = ['#d35400' if l < 0 else COLORS['SVM'] for l in svm_lats[order]]
        ax.bar(np.arange(svm_lats.size), svm_lats[order], color=colors_svm, edgecolor='black', linewidth=0.6)
        ax.axhline(0, color='black', lw=1.2, ls='--')
        step = max(1, len(svm_files) // 20)
        ax.set_xticks(np.arange(0, len(svm_files), step))
        ax.set_xticklabels([str(svm_files[order][i]) for i in range(0, len(svm_files), step)], rotation=45)
        apply_axes(ax, F, xlabel="Test File (Sorted)", ylabel="Latency (s)", title="SVM Per-File Latency (100 Files)")
        fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "svm_per_file_latency.png"), dpi=200); plt.close(fig)

    # Plot 4: Latency & Standard Deviation Comparison
    if svm_lats.size and not np.isnan(rob['mean']):
        fig, ax = plt.subplots(figsize=(7, 5.5))
        means_hs = [rob['mean'], np.mean(svm_lats)]
        stds_hs = [rob['std'], np.std(svm_lats, ddof=1)]
        ax.bar([0, 1], means_hs, yerr=stds_hs, capsize=14, width=0.5, ecolor='black',
               color=[COLORS['Robust'], COLORS['SVM']], edgecolor='black', linewidth=1.0)
        ax.scatter(rng.uniform(-0.08, 0.08, rob['vals'].size), rob['vals'],
                   s=16, color='black', alpha=0.45, zorder=3, label='Robust per-file')
        ax.scatter(1 + rng.uniform(-0.08, 0.08, svm_lats.size), svm_lats,
                   s=16, color='black', alpha=0.45, zorder=3, label='SVM per-file')
        ax.axhline(0, color='grey', lw=1, ls='--')
        for x, m, s in zip([0, 1], means_hs, stds_hs):
            ax.annotate(f"{m:+.2f} ± {s:.2f}s", (x, m + s), ha='center', va='bottom',
                        fontsize=F['annot'], fontweight='bold')
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Robust", "SVM"])
        lo = min(0.0, float(np.min(rob['vals'])), float(np.min(svm_lats))) - 1.0
        hi = max([m + s for m, s in zip(means_hs, stds_hs)] +
                 [float(np.max(rob['vals'])), float(np.max(svm_lats))]) + 1.5
        ax.set_ylim(lo, hi)
        apply_axes(ax, F, xlabel="Architecture", ylabel="Latency (s)",
                   title="Latency & Standard Deviation Comparison\n(Error bars = ±1 Standard Deviation)", legend=True)
        fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "latency_spread_comparison.png"), dpi=200); plt.close(fig)

    # Plot 5: Mother Wavelets
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    t = np.linspace(-5, 5, 200)
    axs[0].plot(t, np.real(np.exp(1j * 5.0 * t)) * np.exp(-0.5 * t ** 2), lw=2, color=COLORS['Morlet'])
    axs[1].plot(t, (1 - t ** 2) * np.exp(-0.5 * t ** 2), lw=2, color=COLORS['Mexh'])
    t_h = np.linspace(0, 1, 200)
    axs[2].step(t_h, np.where(t_h < 0.5, 1.0, -1.0), lw=2, color=COLORS['Haar'], where='post')
    for a, ttl in zip(axs, ["Morlet", "Mexican Hat", "Haar"]):
        apply_axes(a, F, xlabel="t", ylabel="psi(t)", title=ttl)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "mother_wavelet_shapes.png"), dpi=200); plt.close(fig)

    # Plot 6: Wavelet Benchmark
    fig, ax = plt.subplots(figsize=(8, 5))
    wnames = ['Morlet', 'Mexican Hat', 'Haar']
    wkeys = ['morl', 'mexh', 'haar']
    vals = [agg_scores[k] for k in wkeys]
    cols = [COLORS['Morlet'], COLORS['Mexh'], COLORS['Haar']]
    ax.bar(wnames, vals, color=cols, edgecolor='black', linewidth=0.8)
    apply_axes(ax, F, xlabel="Mother Wavelet", ylabel="Separability Score", title="Phase 1: Mother Wavelet Benchmarking")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "wavelet_benchmark_stats.png"), dpi=200); plt.close(fig)

    # Plot 7: Robust FPR Per-File (FIXED TITLE)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(np.arange(len(rob_fpr_vals)), rob_fpr_vals, color=COLORS['Robust'], edgecolor='black', linewidth=0.5)
    ax.axhline(1.0, color='red', lw=2, label='1% target')
    step = max(1, len(rob_fpr_vals) // 20)
    ax.set_xticks(np.arange(0, len(rob_fpr_vals), step))
    ax.set_xticklabels([str(i) for i in range(0, len(rob_fpr_vals), step)], rotation=45)
    apply_axes(ax, F, xlabel="File Index", ylabel="False Positive Rate (%)", title="Robust FPR Per-File", legend=True)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "robust_fpr_per_file.png"), dpi=200); plt.close(fig)

    # Plot 8: SVM FPR Per-File (FIXED TITLE)
    fig, ax = plt.subplots(figsize=(12, 5))
    order = np.argsort(svm_all_files)
    ax.bar(np.arange(len(svm_all_fprs)), svm_all_fprs[order], color=COLORS['SVM'], edgecolor='black', linewidth=0.5)
    ax.axhline(1.0, color='red', lw=2, label='1% target')
    step = max(1, len(svm_all_files) // 20)
    ax.set_xticks(np.arange(0, len(svm_all_files), step))
    ax.set_xticklabels([str(svm_all_files[order][i]) for i in range(0, len(svm_all_files), step)], rotation=45)
    apply_axes(ax, F, xlabel="File Index (Sorted)", ylabel="False Positive Rate (%)", title="SVM FPR Per-File", legend=True)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "svm_fpr_per_file.png"), dpi=200); plt.close(fig)

    # Plot 9: Robust Latency Per-File (FIXED TITLE)
    fig, ax = plt.subplots(figsize=(12, 5))
    order = np.argsort(robust_files)
    colors_rob = ['#145a32' if l < 0 else COLORS['Robust'] for l in robust_latencies[order]] 
    ax.bar(np.arange(len(robust_latencies)), robust_latencies[order], color=colors_rob, edgecolor='black', linewidth=0.6)
    ax.axhline(0, color='black', lw=1.2, ls='--')
    step = max(1, len(robust_files) // 20)
    ax.set_xticks(np.arange(0, len(robust_files), step))
    ax.set_xticklabels([str(robust_files[order][i]) for i in range(0, len(robust_files), step)], rotation=45)
    apply_axes(ax, F, xlabel="Test File (Sorted)", ylabel="Latency (s)", title="Robust Per-File Latency")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "robust_per_file_latency.png"), dpi=200); plt.close(fig)

    # ================= TEXT OUTPUT =================
    out_txt = []
    out_txt.append("================ FINAL POSTER NUMBERS (100 FILES) ================")
    out_txt.append("\n[DETECTORS - LATENCY & EARLY DETECTION]")
    for n in DET_COLS:
        s = det_stats[n]
        out_txt.append(f"{n:10s}: Rate={s['rate']:5.1f}% | Mean={s['mean']:+7.3f}s | Std Dev={s['std']:6.3f}s | Early={s['early']}")
        
    out_txt.append("\n[SVM 5-FOLD CROSS VALIDATION (100 FILES)]")
    svm_det_rate = 100.0 * len(svm_lats) / sum(1 for r in svm_results if r['Faulty'])
    svm_early = int(np.sum(svm_lats < 0))
    out_txt.append(f"Detection Rate: {len(svm_lats)}/{sum(1 for r in svm_results if r['Faulty'])} ({svm_det_rate:.1f}%)")
    out_txt.append(f"Mean Latency  : {np.mean(svm_lats):+.3f}s | Std Dev={np.std(svm_lats, ddof=1):.3f}s | Early={svm_early}")
        
    out_txt.append("\n[FALSE POSITIVE RATES (FPR) - ALL 100 FILES]")
    out_txt.append(f"Robust (Median/MAD): Mean={np.mean(rob_fpr_vals):.2f}% | Max={np.max(rob_fpr_vals):.2f}%")
    out_txt.append(f"SVM (Digital Twin) : Mean={np.mean(svm_all_fprs):.2f}% | Max={np.max(svm_all_fprs):.2f}%")
        
    out_txt.append("\n[PHASE 1 WAVELET BENCHMARKING]")
    for w, k in zip(wnames, wkeys):
        out_txt.append(f"  {w:12s}: {agg_scores[k]:.2f}")
    out_txt.append("==================================================================")
    
    text_summary = "\n".join(out_txt)
    print(text_summary)
    with open(os.path.join(OUT_DIR, "poster_stats.txt"), 'w') as f:
        f.write(text_summary)
    print(f"\n[SUCCESS] 9 Figures and stats written to: {os.path.abspath(OUT_DIR)}")

if __name__ == "__main__":
    main()