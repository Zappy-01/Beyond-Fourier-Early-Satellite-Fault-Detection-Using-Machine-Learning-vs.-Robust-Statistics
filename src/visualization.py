"""
Visualization Module

This module generates all figures for the research paper.
It creates publication-quality plots comparing detector architectures,
wavelet benchmarks, and performance metrics.

Based on research: "Beyond Fourier: Early Satellite Fault Detection Using Machine Learning vs. Robust Statistics"
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Strict Color Coding (matching paper)
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


def get_font_sizes(x_label_scale=6, y_label_scale=6):
    """
    Calculate font sizes based on independent axis label scaling.
    
    Parameters
    ----------
    x_label_scale : int
        X-axis label scale (1-10)
    y_label_scale : int
        Y-axis label scale (1-10)
        
    Returns
    -------
    dict
        Font sizes for different plot elements
    """
    x_s = int(min(10, max(1, x_label_scale)))
    y_s = int(min(10, max(1, y_label_scale)))
    return {
        "x_label": 6 + 2 * x_s,
        "y_label": 6 + 2 * y_s,
        "tick": 4 + 1.5 * max(x_s, y_s),
        "title": 8 + 2 * max(x_s, y_s),
        "legend": 4 + 1.5 * max(x_s, y_s),
        "annot": 4 + 1.5 * max(x_s, y_s)
    }


def apply_axes(ax, F, xlabel=None, ylabel=None, title=None, legend=False):
    """
    Apply consistent formatting to axes.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to format
    F : dict
        Font sizes dictionary
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    title : str, optional
        Plot title
    legend : bool
        Whether to show legend
    """
    ax.tick_params(labelsize=F["tick"])
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=F["x_label"])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=F["y_label"])
    if title:
        ax.set_title(title, fontsize=F["title"])
    if legend:
        ax.legend(fontsize=F["legend"])


def plot_detector_benchmark(det_stats, out_path, F=None):
    """
    Plot 1: Detector Benchmark - Mean latency with std bars for each detector.
    
    Parameters
    ----------
    det_stats : dict
        Dictionary with detector statistics (mean, std, rate)
    out_path : str
        Output file path
    F : dict, optional
        Font sizes
    """
    if F is None:
        F = get_font_sizes()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(det_stats.keys())
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
    
    DET_ABBR = {'Primary': 'P', 'Consensus': 'C', 'Robust': 'Robust', 'Ratio': 'Ra'}
    ax.set_xticks(xs)
    ax.set_xticklabels([DET_ABBR[n] for n in names])
    ax.axhline(0, color='black', lw=0.8)
    
    apply_axes(ax, F, xlabel="Detector Architecture", ylabel="Latency (s)", title="Detector Benchmark")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_fpr_comparison(rob_fpr_vals, svm_all_fprs, out_path, F=None):
    """
    Plot 2: FPR Trade-Off - Scatter plot of per-file FPR for both architectures.
    
    Parameters
    ----------
    rob_fpr_vals : ndarray
        Robust detector per-file FPR values
    svm_all_fprs : ndarray
        SVM per-file FPR values
    out_path : str
        Output file path
    F : dict, optional
        Font sizes
    """
    if F is None:
        F = get_font_sizes()
    
    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x_rob = 0 + rng.uniform(-0.1, 0.1, len(rob_fpr_vals))
    x_svm = 1 + rng.uniform(-0.1, 0.1, len(svm_all_fprs))
    
    ax.scatter(x_rob, rob_fpr_vals, color=COLORS['Robust'], alpha=0.6, label='Per-File')
    ax.scatter(x_svm, svm_all_fprs, color=COLORS['SVM'], alpha=0.6, label='Per-File')
    
    # Mean markers
    ax.scatter([0], [np.mean(rob_fpr_vals)], color=COLORS['Robust'], s=150, 
               marker='s', edgecolor='black', zorder=5, label='Mean')
    ax.scatter([1], [np.mean(svm_all_fprs)], color=COLORS['SVM'], s=150, 
               marker='s', edgecolor='black', zorder=5)
    
    ax.axhline(1.0, color='red', linestyle='--', linewidth=2, label='1% Aerospace Target')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Robust', 'SVM'])
    
    apply_axes(ax, F, xlabel="Architecture", ylabel="False Positive Rate (%)", 
               title="FPR Trade-Off", legend=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_svm_per_file_latency(svm_files, svm_lats, out_path, F=None):
    """
    Plot 3: SVM Per-File Latency - Bar chart of latency for each faulty file.
    
    Parameters
    ----------
    svm_files : ndarray
        File numbers
    svm_lats : ndarray
        Latency values
    out_path : str
        Output file path
    F : dict, optional
        Font sizes
    """
    if F is None or svm_lats.size == 0:
        if svm_lats.size == 0:
            print("No SVM latency data to plot.")
            return
        F = get_font_sizes()
    
    fig, ax = plt.subplots(figsize=(12, 5))
    order = np.argsort(svm_files)
    colors_svm = ['#d35400' if l < 0 else COLORS['SVM'] for l in svm_lats[order]]
    
    ax.bar(np.arange(svm_lats.size), svm_lats[order], color=colors_svm, 
           edgecolor='black', linewidth=0.6)
    ax.axhline(0, color='black', lw=1.2, ls='--')
    
    step = max(1, len(svm_files) // 20)
    ax.set_xticks(np.arange(0, len(svm_files), step))
    ax.set_xticklabels([str(svm_files[order][i]) for i in range(0, len(svm_files), step)], 
                       rotation=45)
    
    apply_axes(ax, F, xlabel="Test File (Sorted)", ylabel="Latency (s)", 
               title="SVM Per-File Latency (100 Files)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_latency_spread_comparison(rob_means, rob_std, rob_vals, svm_lats, out_path, F=None):
    """
    Plot 4: Latency & Standard Deviation Comparison - Side-by-side comparison.
    
    Parameters
    ----------
    rob_means : float
        Robust detector mean latency
    rob_std : float
        Robust detector std deviation
    rob_vals : ndarray
        Robust detector per-file latencies
    svm_lats : ndarray
        SVM per-file latencies
    out_path : str
        Output file path
    F : dict, optional
        Font sizes
    """
    if F is None:
        F = get_font_sizes()
    
    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    
    means_hs = [rob_means, np.mean(svm_lats)]
    stds_hs = [rob_std, np.std(svm_lats, ddof=1)]
    
    ax.bar([0, 1], means_hs, yerr=stds_hs, capsize=14, width=0.5, ecolor='black',
           color=[COLORS['Robust'], COLORS['SVM']], edgecolor='black', linewidth=1.0)
    
    # Scatter individual points
    ax.scatter(rng.uniform(-0.08, 0.08, rob_vals.size), rob_vals,
               s=16, color='black', alpha=0.45, zorder=3, label='Robust per-file')
    ax.scatter(1 + rng.uniform(-0.08, 0.08, svm_lats.size), svm_lats,
               s=16, color='black', alpha=0.45, zorder=3, label='SVM per-file')
    
    ax.axhline(0, color='grey', lw=1, ls='--')
    
    # Annotate mean ± std
    for x, m, s in zip([0, 1], means_hs, stds_hs):
        ax.annotate(f"{m:+.2f} ± {s:.2f}s", (x, m + s), ha='center', va='bottom',
                    fontsize=F['annot'], fontweight='bold')
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Robust", "SVM"])
    
    # Set y-limits
    lo = min(0.0, float(np.min(rob_vals)), float(np.min(svm_lats))) - 1.0
    hi = max([m + s for m, s in zip(means_hs, stds_hs)] +
             [float(np.max(rob_vals)), float(np.max(svm_lats))]) + 1.5
    ax.set_ylim(lo, hi)
    
    apply_axes(ax, F, xlabel="Architecture", ylabel="Latency (s)",
               title="Latency & Standard Deviation Comparison\n(Error bars = ±1 Standard Deviation)", 
               legend=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_mother_wavelets(out_path, F=None):
    """
    Plot 5: Mother Wavelet Shapes - Visualize the three candidate wavelets.
    
    Parameters
    ----------
    out_path : str
        Output file path
    F : dict, optional
        Font sizes
    """
    if F is None:
        F = get_font_sizes()
    
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    t = np.linspace(-5, 5, 200)
    
    # Morlet
    axs[0].plot(t, np.real(np.exp(1j * 5.0 * t)) * np.exp(-0.5 * t ** 2), 
                lw=2, color=COLORS['Morlet'])
    # Mexican Hat
    axs[1].plot(t, (1 - t ** 2) * np.exp(-0.5 * t ** 2), 
                lw=2, color=COLORS['Mexh'])
    # Haar
    t_h = np.linspace(0, 1, 200)
    axs[2].step(t_h, np.where(t_h < 0.5, 1.0, -1.0), 
                lw=2, color=COLORS['Haar'], where='post')
    
    for a, ttl in zip(axs, ["Morlet", "Mexican Hat", "Haar"]):
        apply_axes(a, F, xlabel="t", ylabel="psi(t)", title=ttl)
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_wavelet_benchmark(agg_scores, out_path, F=None):
    """
    Plot 6: Wavelet Benchmark Stats - Bar chart of separability scores.
    
    Parameters
    ----------
    agg_scores : dict
        Aggregated scores per wavelet
    out_path : str
        Output file path
    F : dict, optional
        Font sizes
    """
    if F is None:
        F = get_font_sizes()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    wnames = ['Morlet', 'Mexican Hat', 'Haar']
    wkeys = ['morl', 'mexh', 'haar']
    vals = [agg_scores[k] for k in wkeys]
    cols = [COLORS['Morlet'], COLORS['Mexh'], COLORS['Haar']]
    
    ax.bar(wnames, vals, color=cols, edgecolor='black', linewidth=0.8)
    
    apply_axes(ax, F, xlabel="Mother Wavelet", ylabel="Separability Score", 
               title="Phase 1: Mother Wavelet Benchmarking")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_robust_fpr_per_file(rob_fpr_vals, out_path, F=None):
    """
    Plot 7: Robust FPR Per-File - Bar chart of FPR for each file.
    
    Parameters
    ----------
    rob_fpr_vals : ndarray
        Per-file FPR values
    out_path : str
        Output file path
    F : dict, optional
        Font sizes
    """
    if F is None:
        F = get_font_sizes()
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(np.arange(len(rob_fpr_vals)), rob_fpr_vals, color=COLORS['Robust'], 
           edgecolor='black', linewidth=0.5)
    ax.axhline(1.0, color='red', lw=2, label='1% target')
    
    step = max(1, len(rob_fpr_vals) // 20)
    ax.set_xticks(np.arange(0, len(rob_fpr_vals), step))
    ax.set_xticklabels([str(i) for i in range(0, len(rob_fpr_vals), step)], rotation=45)
    
    apply_axes(ax, F, xlabel="File Index", ylabel="False Positive Rate (%)", 
               title="Robust FPR Per-File", legend=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_svm_fpr_per_file(svm_all_files, svm_all_fprs, out_path, F=None):
    """
    Plot 8: SVM FPR Per-File - Bar chart of FPR for each file (sorted).
    
    Parameters
    ----------
    svm_all_files : ndarray
        File numbers
    svm_all_fprs : ndarray
        Per-file FPR values
    out_path : str
        Output file path
    F : dict, optional
        Font sizes
    """
    if F is None:
        F = get_font_sizes()
    
    fig, ax = plt.subplots(figsize=(12, 5))
    order = np.argsort(svm_all_files)
    
    ax.bar(np.arange(len(svm_all_fprs)), svm_all_fprs[order], color=COLORS['SVM'], 
           edgecolor='black', linewidth=0.5)
    ax.axhline(1.0, color='red', lw=2, label='1% target')
    
    step = max(1, len(svm_all_files) // 20)
    ax.set_xticks(np.arange(0, len(svm_all_files), step))
    ax.set_xticklabels([str(svm_all_files[order][i]) for i in range(0, len(svm_all_files), step)], 
                       rotation=45)
    
    apply_axes(ax, F, xlabel="File Index (Sorted)", ylabel="False Positive Rate (%)", 
               title="SVM FPR Per-File", legend=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_robust_per_file_latency(robust_files, robust_latencies, out_path, F=None):
    """
    Plot 9: Robust Per-File Latency - Bar chart of latency for each faulty file.
    
    Parameters
    ----------
    robust_files : ndarray
        File numbers
    robust_latencies : ndarray
        Latency values
    out_path : str
        Output file path
    F : dict, optional
        Font sizes
    """
    if F is None:
        F = get_font_sizes()
    
    fig, ax = plt.subplots(figsize=(12, 5))
    order = np.argsort(robust_files)
    colors_rob = ['#145a32' if l < 0 else COLORS['Robust'] for l in robust_latencies[order]]
    
    ax.bar(np.arange(len(robust_latencies)), robust_latencies[order], color=colors_rob, 
           edgecolor='black', linewidth=0.6)
    ax.axhline(0, color='black', lw=1.2, ls='--')
    
    step = max(1, len(robust_files) // 20)
    ax.set_xticks(np.arange(0, len(robust_files), step))
    ax.set_xticklabels([str(robust_files[order][i]) for i in range(0, len(robust_files), step)], 
                       rotation=45)
    
    apply_axes(ax, F, xlabel="Test File (Sorted)", ylabel="Latency (s)", 
               title="Robust Per-File Latency")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def generate_all_figures(det_stats, rob_fpr_vals, svm_results, robust_latencies, 
                         robust_files, agg_scores, out_dir, F=None):
    """
    Generate all 9 figures for the research paper.
    
    Parameters
    ----------
    det_stats : dict
        Detector statistics
    rob_fpr_vals : ndarray
        Robust detector per-file FPR
    svm_results : list
        SVM results from 5-fold CV
    robust_latencies : ndarray
        Robust per-file latencies
    robust_files : ndarray
        Robust file numbers
    agg_scores : dict
        Wavelet benchmark scores
    out_dir : str
        Output directory
    F : dict, optional
        Font sizes
    """
    if F is None:
        F = get_font_sizes()
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Extract SVM data
    svm_all_files = np.array([r['File'] for r in svm_results])
    svm_all_fprs = np.array([r['FPR'] for r in svm_results])
    svm_faulty_mask = np.array([r['Faulty'] for r in svm_results])
    svm_latencies_raw = np.array([r['Latency'] for r in svm_results])
    valid_svm_lat_mask = svm_faulty_mask & ~np.isnan(svm_latencies_raw)
    svm_lats = svm_latencies_raw[valid_svm_lat_mask]
    svm_files = svm_all_files[valid_svm_lat_mask]
    
    # Get robust stats
    rob = det_stats['Robust']
    
    # Generate all plots
    print("Generating figures...")
    
    plot_detector_benchmark(det_stats, os.path.join(out_dir, "detector_benchmark.png"), F)
    print("  1/9: detector_benchmark.png")
    
    plot_fpr_comparison(rob_fpr_vals, svm_all_fprs, 
                        os.path.join(out_dir, "fpr_comparison.png"), F)
    print("  2/9: fpr_comparison.png")
    
    if svm_lats.size:
        plot_svm_per_file_latency(svm_files, svm_lats, 
                                  os.path.join(out_dir, "svm_per_file_latency.png"), F)
        print("  3/9: svm_per_file_latency.png")
    
    if svm_lats.size and not np.isnan(rob['mean']):
        plot_latency_spread_comparison(rob['mean'], rob['std'], rob['vals'], svm_lats,
                                       os.path.join(out_dir, "latency_spread_comparison.png"), F)
        print("  4/9: latency_spread_comparison.png")
    
    plot_mother_wavelets(os.path.join(out_dir, "mother_wavelet_shapes.png"), F)
    print("  5/9: mother_wavelet_shapes.png")
    
    plot_wavelet_benchmark(agg_scores, os.path.join(out_dir, "wavelet_benchmark_stats.png"), F)
    print("  6/9: wavelet_benchmark_stats.png")
    
    plot_robust_fpr_per_file(rob_fpr_vals, os.path.join(out_dir, "robust_fpr_per_file.png"), F)
    print("  7/9: robust_fpr_per_file.png")
    
    plot_svm_fpr_per_file(svm_all_files, svm_all_fprs, 
                          os.path.join(out_dir, "svm_fpr_per_file.png"), F)
    print("  8/9: svm_fpr_per_file.png")
    
    plot_robust_per_file_latency(robust_files, robust_latencies, 
                                 os.path.join(out_dir, "robust_per_file_latency.png"), F)
    print("  9/9: robust_per_file_latency.png")
    
    print(f"\n[SUCCESS] All figures written to: {os.path.abspath(out_dir)}")
