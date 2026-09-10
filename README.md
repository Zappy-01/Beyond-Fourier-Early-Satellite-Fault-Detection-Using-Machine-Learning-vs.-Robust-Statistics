# Satellite Fault Detection Research

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Research Internship Project: Early Satellite Fault Detection Using Machine Learning vs. Robust Statistics**

This repository contains the complete codebase for the research paper *"Beyond Fourier: Early Satellite Fault Detection Using Machine Learning vs. Robust Statistics"*. The project compares two fault detection architectures on simulated satellite reaction wheel telemetry:

1. **Statistical Detector (Week 7)**: Rule-based detection using wavelet features and robust statistics (median/MAD)
2. **SVM Detector (Week 9)**: Support Vector Machine trained on digital-twin residual spectra

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Repository Modules](#repository-modules)
- [Key Findings](#key-findings)
- [Citation](#citation)

## Overview

### Problem Statement
Reaction wheels are the primary actuators for satellite attitude control. Mechanical or electrical faults in these components first appear as slow drifts in motor current and angular velocity telemetry. This work aims to detect faults shortly after they begin, while the drift is still small, so operators can act before any protection limit is reached.

### Approach
Both architectures use **wavelet transforms**, which record *when* a signal change occurs as well as *which* frequencies change — a property Fourier analysis does not provide.

| Aspect | Statistical Detector | SVM Detector |
|--------|---------------------|--------------|
| **Input** | Raw wavelet features | Digital-twin residuals |
| **Decision Rule** | Median/MAD threshold | RBF-SVM boundary |
| **Training** | None (unsupervised) | 5-fold CV on 80 files |
| **Dependency** | Standalone | Requires simulation |

### Dataset
- 100 simulated telemetry files
- 54 faulty files (drift in torque constant, bus voltage, or both)
- Signals: motor current (*i*) and angular velocity (*w*)
- Both expected (healthy) and actual values recorded

## Project Structure

```
satellite-fault-detection/
├── README.md                 # This file (full paper content)
├── run_pipeline.py           # Main pipeline orchestrator
├── src/                      # Source modules
│   ├── __init__.py
│   ├── wavelet_benchmark.py  # Phase 1: Wavelet selection
│   ├── statistical_detector.py # Week 7: Rule-based detectors
│   ├── svm_detector.py       # Week 9: SVM training & evaluation
│   └── visualization.py      # Figure generation (9 plots)
├── data/                     # Raw telemetry data (not included)
├── results/                  # Generated figures and stats
├── cache/                    # Cached intermediate results
├── docs/                     # Additional documentation
└── config/                   # Configuration files
```

## Installation

### Requirements
- Python 3.8+
- NumPy
- Pandas
- SciPy
- PyWavelets
- scikit-learn
- Matplotlib

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd satellite-fault-detection

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install numpy pandas scipy pywavelets scikit-learn matplotlib
```

## Usage

### Full Pipeline Execution

Run the complete analysis pipeline:

```bash
python run_pipeline.py \
    --raw_dir /path/to/telemetry/csv/files \
    --summary_file /path/to/week7_summary.csv \
    --out_dir ./results \
    --cache_dir ./cache
```

### Individual Module Usage

#### 1. Wavelet Benchmarking (Phase 1)

```python
from src.wavelet_benchmark import benchmark_wavelets, compute_wavelet_features

# Benchmark multiple wavelets
scores = benchmark_wavelets(signal_data, wavelets=['morl', 'mexh', 'haar'])
print(f"Best wavelet: {max(scores, key=scores.get)}")

# Compute features for a single segment
features = compute_wavelet_features(segment, wavelet='haar', scales=np.arange(1,64), dt=0.1)
```

#### 2. Statistical Detector (Week 7)

```python
from src.statistical_detector import (
    extract_dense_features, 
    RobustDetector, 
    PrimaryDetector
)

# Extract features from signal
features = extract_dense_features(motor_current_signal)

# Use Robust detector (recommended)
detector = RobustDetector(threshold=3)
detector.fit(features_healthy)
predictions = detector.predict(features_test)

# Calculate FPR
from src.statistical_detector import calculate_fpr
fpr = calculate_fpr(predictions)
```

#### 3. SVM Detector (Week 9)

```python
from src.svm_detector import (
    extract_full_dataset,
    run_5fold_cv,
    SVMDetector
)

# Extract features from all files
X, y, groups, end_times, true_onsets, is_faulty = extract_full_dataset(
    raw_dir='/path/to/data',
    cache_path='./cache/svm_features.npz'
)

# Run 5-fold cross-validation
results = run_5fold_cv(X, y, groups, end_times, true_onsets, is_faulty)

# Or train custom model
model = SVMDetector(C=0.1, kernel='rbf')
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

#### 4. Visualization

```python
from src.visualization import generate_all_figures, get_font_sizes

F = get_font_sizes(x_label_scale=6, y_label_scale=6)

generate_all_figures(
    det_stats=detector_stats,
    rob_fpr_vals=robust_fpr_values,
    svm_results=svm_cv_results,
    robust_latencies=robust_latency_values,
    robust_files=robust_file_numbers,
    agg_scores=wavelet_scores,
    out_dir='./results',
    F=F
)
```

## Repository Modules

### `src/wavelet_benchmark.py`
**Purpose**: Phase 1 - Select the optimal mother wavelet for fault detection.

**Key Functions**:
- `compute_wavelet_features()`: Extract 7 statistical features using specified wavelet
- `calculate_fdr()`: Compute Fisher Discriminant Ratio for separability scoring
- `benchmark_wavelets()`: Compare multiple wavelets on provided data
- `generate_wavelet_shapes()`: Generate wavelet template visualizations

**Finding**: Haar wavelet scored 0.28 vs 0.13 (Mexican Hat) and 0.12 (Morlet), making it approximately twice as effective for step-like faults.

### `src/statistical_detector.py`
**Purpose**: Week 7 - Implement and evaluate rule-based statistical detectors.

**Classes**:
- `RobustDetector`: Uses median and MAD (recommended)
- `PrimaryDetector`: Uses mean and standard deviation
- `ConsensusDetector`: Requires multiple features to exceed threshold
- `RatioDetector`: Monitors Kurtosis/Energy ratio

**Key Functions**:
- `extract_dense_features()`: Extract features using level-3 Haar SWT
- `consolidate_events()`: Group flagged windows into events
- `fpr_robust()`: Calculate honest FPR on healthy data

**Finding**: Robust detector achieved 100% detection rate with mean latency of -2.70s ± 4.90s, but FPR of 29.24%.

### `src/svm_detector.py`
**Purpose**: Week 9 - Implement SVM-based detection with digital twin residuals.

**Key Functions**:
- `compute_cwt_features()`: Extract 6 CWT-based features per channel
- `extract_full_dataset()`: Process all files and extract residual features
- `run_5fold_cv()`: Execute 5-fold group cross-validation
- `calculate_metrics()`: Aggregate performance metrics

**Finding**: SVM matched robust detector in detection rate (100%) and latency (-1.31s ± 5.59s), with similar mean FPR (~29%) but quantized per-file behavior.

### `src/visualization.py`
**Purpose**: Generate all 9 publication-quality figures.

**Figures**:
1. Detector benchmark (latency comparison)
2. FPR trade-off (scatter plot)
3. SVM per-file latency
4. Latency spread comparison
5. Mother wavelet shapes
6. Wavelet benchmark stats
7. Robust FPR per-file
8. SVM FPR per-file
9. Robust per-file latency

## Key Findings

### Detection Performance
| Metric | Robust Detector | SVM Detector |
|--------|----------------|--------------|
| Detection Rate | 100% | 100% |
| Mean Latency | -2.70s | -1.31s |
| Std Deviation | 4.90s | 5.59s |
| Mean FPR | 29.24% | ~29% |
| Max FPR | 44.79% | ~100% (quantized) |

### Critical Insights

1. **Wavelet Selection**: The Haar wavelet is unambiguously superior for step-like parameter drifts, scoring approximately double the separability of alternatives.

2. **Early Detection Trade-off**: Both architectures detected all 54 faulty files, with many detections occurring *before* the documented injection time. However, neither satisfies the 1% FPR aerospace requirement.

3. **Architectural Differences**::
   - **Robust**: No training required, no simulation dependency, continuous FPR distribution
   - **SVM**: Requires digital twin, quantized FPR behavior (near 0% or 100% per file)

4. **FPR Behavior**: The narrow reference band that enables early detection in the robust rule also causes false positives on healthy tail values. The SVM's global boundary separates files rather than windows.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{beyond_fourier_2026,
  title={Beyond Fourier: Early Satellite Fault Detection Using Machine Learning vs. Robust Statistics},
  author={Your Name},
  year={2026},
  publisher={Elevate 2026 Engineering Internship Research}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Elevate 2026 Engineering Internship Program
- Research mentorship and guidance
- Simulated telemetry dataset providers

---

**Note**: The full research paper content is available in this README. For detailed methodology, mathematical definitions, and comprehensive discussion, refer to the sections above.
