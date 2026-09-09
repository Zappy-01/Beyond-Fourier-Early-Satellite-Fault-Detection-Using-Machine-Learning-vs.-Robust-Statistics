Strategic bolding has been applied throughout: each bold now marks the phrase that carries the sentence's "why" or "what," with no paragraph made dense. The full revised paper follows.

---

# Beyond Fourier: Early Satellite Fault Detection Using Machine Learning vs. Robust Statistics

Elevate 2026 · Engineering Internship Research

## Abstract

Reaction wheels are the primary actuators for satellite attitude control, and mechanical or electrical faults in these components first appear as slow drifts in the motor current and angular velocity telemetry. This paper compares two detection architectures on a 100-file simulated telemetry dataset: a rule-based detector that compares wavelet features against a healthy-only median / median absolute deviation (MAD) reference, and a support vector machine (SVM) trained on digital-twin residual spectra. Both architectures use wavelet transforms, which record **when a signal change occurs** as well as which frequencies change — a property Fourier analysis does not provide, hence the title *Beyond Fourier*. A benchmark of three candidate wavelets (Morlet, Mexican Hat, Haar) selected the Haar wavelet, because its step shape matches the step-like faults in the dataset. Both architectures detected all 54 faulty files. The robust detector produced a mean latency of −2.70 s ± 4.90 s relative to the documented injection time, with 36 detections preceding injection; the SVM, evaluated on all 100 files by five-fold group cross-validation, produced −1.31 s ± 5.59 s. The difference between the means is small relative to either spread, so the two architectures are not separated in detection speed on this dataset. The main weakness of both is the false positive rate (FPR): the robust detector averages 29.24% (worst file 44.79%), and the SVM's per-file FPR concentrates at the two extremes, near 0% on a majority of files and near 100% on a large subset, with a mean of approximately 29%. A reference rule based on the mean and standard deviation of the same features achieves an FPR of 0.52%, at the cost of a lower detection rate and later detections. The results therefore describe two trade-offs: between early detection and false positive rate, and between independence from simulation and dependence on it. The only unambiguous conclusion within the scope of this study is **the selection of the Haar wavelet**. Unfamiliar terms are defined in the glossary (Appendix A) at the end of this paper.

## 1. Introduction

Satellites maintain their orientation using reaction wheels: accelerating a flywheel in one direction rotates the spacecraft body in the other. Increased bearing friction or electrical degradation in a wheel appears as a slow drift in the motor current and in the angular velocity response. Flight software protects the spacecraft with limits on current and speed; these limits detect a fault once the drift has become large. The objective of this work is to detect the fault shortly after it begins, while the drift is still small, so that **operators can act before any protection limit is reached**.

Two architectures are compared. The first is rule-based: a fixed set of statistical features is extracted for each second of telemetry, and a flag is raised when the features move away from a reference computed from healthy data. The second is a support vector machine (SVM) trained to classify each window as healthy or faulty from wavelet spectral features. Both architectures are evaluated on the same 100 files with the same metrics, so the comparison measures **the decision logic, not the feature extraction**. The comparison axes are detection rate, detection latency, false positive rate, and dependence on training data or on a parallel simulation.

The scope is limited. The dataset is simulated, the detection threshold is fixed before evaluation and never tuned, and the results are **comparative measurements on one dataset rather than general performance claims**.

## 2. Dataset and Evaluation Metrics

The dataset contains 100 files. Each file records two signals: the motor current (*i*) and the angular velocity (*w*). For each signal, the file contains the expected healthy values and the actual values. In 54 of the 100 files, a fault begins at a documented time; the fault is a slow drift in the torque constant, in the bus voltage, or in both. The motor current and the angular velocity are the signals used in this work, because **friction and electrical faults change these two signals before any protection limit is reached**.

Two metrics are used throughout:

- **Detection latency** = detection time − injection time. A negative value means the detector raised its flag before the documented injection time.
- **False positive rate (FPR)** = the fraction of healthy time windows flagged as faulty. Alarm systems are commonly required to keep this value below 1%, because a detector that flags healthy operation too often **produces alarms that operators cannot act on**.

## 3. Wavelet Analysis and Wavelet Selection

### 3.1 Rationale for wavelet analysis

A Fourier transform represents the frequency content of a signal over its entire duration and does not indicate **when a given frequency component occurs**; a fault beginning at one time and a maneuver at another can produce identical spectra. Detection of transient changes therefore requires a method that keeps the time information. A wavelet transform provides this by sliding a short template function — the **mother wavelet** — across the signal. At each position and scale (frequency band), it records a coefficient measuring how strongly the template matches the signal over that interval. The Stationary Wavelet Transform (SWT) is used for the rule-based detector because it preserves the length of the signal, **keeping every coefficient aligned with its sample time**.

### 3.2 Benchmarking method

Three mother wavelets were evaluated, each **matched to a different fault class**: the Morlet wavelet (a damped oscillation) is sensitive to sustained vibrations; the Mexican Hat (a bell-shaped pulse) to short spikes; the Haar (a single step) to sudden offsets and slow ramps. The three template shapes are shown in Figure 1. The faults in this dataset are offsets and ramps in *i* and *w*, which motivated including the step-shaped candidate.

[Figure 1 — mother_wavelet_shapes.png. The three candidate mother wavelets. The Haar wavelet is a pure step function.]

The ranking procedure was quantitative. For each of five representative files and both channels, the signals were divided into 20 contiguous windows, and seven summary features (Energy, Entropy, Variance, RMS, Peak-to-Peak, Kurtosis, Skewness) were computed per window under each wavelet. The Haar wavelet used a discrete wavelet decomposition at the maximum supported level; the Morlet and Mexican Hat wavelets used continuous wavelet transforms over scales 1–63. For each feature and channel, the separation between healthy and faulty windows was scored with the **Fisher Discriminant Ratio**, FDR = (μ_faulty − μ_healthy)² / (σ²_healthy + σ²_faulty). The FDR is large when **the fault moves the feature mean by many within-class standard deviations**, which is the property a downstream detector requires; it was used here only to rank the wavelets. The fourteen ratios (seven features × two channels) were summed into a **separability score** per wavelet and file, then averaged over the five files.

### 3.3 Results

The Haar wavelet scored 0.28, against 0.13 for the Mexican Hat and 0.12 for the Morlet, as shown in Figure 2. A step-like fault **places its energy in a small number of large step-matched coefficients**, whereas the oscillatory wavelets (Morlet, Mexican Hat) spread the same fault across many small coefficients; the measured scores are consistent with this mechanism. The Haar wavelet was adopted for all rule-based processing.

[Figure 2 — wavelet_benchmark_stats.png. Mean separability score per candidate wavelet; the Haar wavelet separates healthy from faulty windows approximately twice as well as the alternatives.]

## 4. Feature Extraction

Each sliding 12.8-second window, advanced every 1 second, was decomposed by a level-3 Haar SWT into high-frequency detail coefficients and low-frequency approximation coefficients, and a 7-dimensional feature vector was extracted: Energy, Entropy, Variance, RMS, Peak-to-Peak, Kurtosis, Skewness (definitions in Appendix A). This produces one feature vector per second.

One design finding was significant. When the level-3 approximation coefficient **cA₃** was excluded, detection of gradual drifts was delayed by 10–20 s. A slow drift places its energy in the low-frequency trend rather than in the high-frequency components, so a feature set that uses only the high-frequency components **contains little drift information until the drift becomes steeper**. Including cA₃ removed this delay.

## 5. Rule-Based Detection Architectures

### 5.1 Definitions

To find a rule that detects gradual drifts, four candidate decision rules were benchmarked on the same feature stream. Each rule used a fixed 3-sigma threshold that was set before any results were examined, so **no parameter was adapted to the test data**. Flagged windows were grouped into events (1 s persistence, 1 s merge gap, 1 s minimum duration), and the system detection time was the earliest event across the two channels.

- **Primary:** each feature is compared against the mean and standard deviation of the data.
- **Consensus:** an event requires two of three features (Peak-to-Peak, Variance, Entropy) to cross the threshold at the same time; the rule was designed for sharp spikes.
- **Robust:** each feature is compared against the **median** and the **median absolute deviation (MAD)**, computed **solely from healthy data**.
- **Ratio:** the rolling ratio of Kurtosis to Energy is monitored; the rule was designed for faults that appear as short spikes.

The design of the Robust rule requires one explanation. The mean and the standard deviation are sensitive to extreme values: if fault windows enter the reference data, **the reference moves toward the fault** and the detector's sensitivity decreases. The median and the MAD depend only on the central portion of the distribution and are therefore resistant to such contamination; computing them on healthy data only removes the contamination entirely.

### 5.2 Results

The Robust rule detected 100% of faults with a mean latency of −2.70 s. The Primary rule detected 85.2% at +8.19 s, the Ratio rule 22.2% at +13.22 s, and the Consensus rule 11.1% at +16.19 s. Figure 3 displays the four mean latencies with their standard deviations; only the Robust rule achieves full coverage, and its mean latency is negative. The Consensus and Ratio rules perform poorly because a gradual drift **does not produce the sharp, simultaneous feature breaches they require**; this is a property of their design, not an implementation defect. The Robust rule was therefore selected as the statistical method for the comparison in Section 7.

[Figure 3 — detector_benchmark.png. Mean detection latency per rule with ±1 standard deviation bars. Only the Robust rule achieves full coverage, and its mean latency is negative.]

## 6. Support Vector Machine Architecture

### 6.1 Digital twin residual input

A classifier trained on raw telemetry cannot separate an intentional high-speed maneuver from a high-friction fault, because **both produce large signals**. To remove this ambiguity, a **digital twin** — a simulation predicting the healthy telemetry at each moment — was run in parallel, and the expected spectral features were subtracted from the actual features. The residual is near zero whenever the wheel behaves as predicted and nonzero otherwise; the classifier therefore receives the **deviation** from expected behavior, not the maneuver itself.

### 6.2 Features and normalization

For each window, and for each of the four motor-current channels (wheel 1 to wheel 4), a continuous wavelet transform (Morlet, scales 1–31) produced a scale-energy spectrum, from which six features were taken: the fraction of total energy in four scale bands, the spectral entropy, and the spectral centroid. This gives 24 features per window. Dividing each band energy by the window's total energy removes dependence on signal magnitude, so classification depends on the **distribution of energy across scales** rather than on amplitude.

The channel set was selected by diagnostics on this dataset. An initial design extracted the same six features from both the motor-current channels and the angular-velocity channels, giving 48 features per window. The synthetic angular-velocity data, however, contains maneuver-induced **variance shifts exceeding 1000%**. The digital-twin subtraction removes the expected maneuver component, but it does not remove this residual variance. When the angular-velocity features were included, the variance distorted the **decision boundary**: the classifier fitted the boundary to the large variance shifts rather than to the fault signatures, and the false positive rate rose above 30%. Channel-isolation tests, in which the current channels and the angular-velocity channels were evaluated independently, also showed that the motor current detects the faults **approximately 6.8 s earlier** than the angular velocity. The angular-velocity channels were therefore excluded, and the SVM uses the 24 motor-current features.

Windows are 12.8 s long and advance every 1 s, and a window is labeled faulty when its end time is at or after the documented injection time. The residual features are standardized to zero mean and unit variance before classification.

### 6.3 Evaluation protocol

A single 80/20 split would evaluate the SVM on only 20 files while the robust rule is evaluated on 100. Instead, five rounds were executed: in each round the model was trained from scratch on 80 files and predicted the remaining 20. Over the five rounds every file received exactly one prediction from a model that had not used it for training, so the SVM is measured on the same 100 files as the robust rule and **never on its own training data**. The kernel and hyperparameters were selected by an initial grid search — radial basis with C in {0.1, 1, 10, 100} and γ in {scale, 0.01}, and linear with C in {0.1, 1, 10} — using five-fold group cross-validation with F1 as the scoring metric on the training data; the optimum (radial basis, C = 0.1, γ = scale) was then held fixed across the five evaluation rounds. For the SVM, the detection time is the end time of the first flagged window whose end time lies no earlier than 10 s before the injection time, and the FPR is computed over windows ending before the injection time.

## 7. Comparison of the Robust Detector and the SVM

### 7.1 Detection rate and latency

Both architectures detected all 54 faulty files. The robust rule's mean latency was −2.70 s ± 4.90 s, with 36 files flagged before the documented injection. The SVM's mean latency was −1.31 s ± 5.59 s, also with a substantial number of pre-injection detections. Figure 4 shows the robust rule's latency per faulty file, and Figure 5 shows the SVM's; the pre-injection detections appear as negative bars in both. The ±1σ intervals ([−7.60, +2.20] s and [−6.90, +4.28] s) overlap; the 1.4 s difference between the means is small relative to the variation from file to file, so **neither architecture can be called faster on this dataset**. Figure 6 places the two means, their ±1σ intervals, and all per-file values on a single axis, making the overlap directly visible. The practical distinctions are dispersion and dependency: the robust rule **requires no training, no data split, and no simulation**, whereas the SVM requires the digital twin for its input to be meaningful.

[Figure 4 — robust_per_file_latency.png. Robust detector latency per faulty file; bars below zero denote detections preceding injection.]

[Figure 5 — svm_per_file_latency.png. SVM latency per faulty file under five-fold cross-validation.]

[Figure 6 — latency_spread_comparison.png. Means with ±1 standard deviation bars and all per-file values; the intervals overlap.]

### 7.2 False positive rate

The robust rule's FPR averaged 29.24% (worst file 44.79%). The SVM's per-file FPR concentrates at the two extremes, near 0% on a majority of files and near 100% on a large subset, with a mean of approximately 29%. **Neither architecture satisfies the 1% requirement.**

The robust rule's FPR follows directly from its design. Figure 7 shows its per-file values: they form a continuous band between roughly 20% and 45%, far above the 1% line. The wavelet features are non-negative and right-skewed: most healthy values lie in a narrow central range, with a tail of larger values. The MAD measures only the spread of the central values, so the 3-sigma band around the median is narrow. A narrow band is crossed early by a small drift — **the source of the rule's early detections** — and also crossed by the healthy tail — **the source of the false positives**. The mean/standard-deviation rule defines a wider band, because the standard deviation includes the tail; it achieves an FPR of 0.52% (worst file 4.08%), but that width is also why it detects later and misses 14.8% of faults. Early detection and low false positive rate, for this family of rules, are controlled by **the same threshold width**.

The SVM's FPR shows a different and notable property, visible in Figure 8 and in the side-by-side view of Figure 9: the per-file values do not spread continuously, as the robust rule's do, but concentrate at the two extremes, near 0% and near 100%, with almost no files in between. The difference follows from how each architecture sets its reference. The robust rule computes its reference from the healthy data of the same file, so it adapts to each file's baseline; its false positives are window-level events, produced when a healthy value crosses the threshold by chance, and their rate varies continuously from file to file. The SVM applies **one global boundary** to all files. Within a file, the residual features are highly correlated, because all windows of the file carry the same maneuver sequence and the same simulation mismatch; the file's residual cluster therefore lies almost entirely on one side of the boundary or the other, and nearly every window of the file receives the same label. A file whose mismatch pattern resembles the fault signature is flagged throughout its healthy portion (FPR near 100%); a file whose mismatch is small is flagged nowhere (FPR near 0%). The quantization therefore indicates that the dominant source of variance in the residual space is **between files, not within files**, and that the boundary separates maneuver regimes as much as fault states. This is a limitation of the residual architecture when the fidelity of the digital twin varies across maneuver profiles.

[Figure 7 — robust_fpr_per_file.png. Robust detector FPR per file against the 1% line.]

[Figure 8 — svm_fpr_per_file.png. SVM FPR per file, showing the concentration of values near 0% and near 100%.]

[Figure 9 — fpr_comparison.png. Per-file FPR of both architectures against the 1% line; the robust values form a continuous band, while the SVM values concentrate at the two extremes.]

## 8. Discussion

The results do not support selecting one architecture on all criteria; each architecture is preferable on different criteria. The robust rule detects every fault, on average before the documented onset, using a few dozen arithmetic operations per window and no training or simulation; its cost is a false positive rate far above the 1% requirement. The SVM matches the robust rule in detection rate and latency and reaches near-zero FPR on many files, but its all-or-nothing behavior on the remaining files, and its dependence on a synchronized parallel simulation, are real deployment costs: **radiation-hardened flight computers rarely have the margin to run such a simulation**.

The limitations are stated explicitly. The dataset is simulated and single-source; the threshold was fixed at 3 sigma and not tuned; the FPR was computed on the healthy portions of the same files used elsewhere in the evaluation; and the two architectures are compared on detection metrics only, not on compute or memory budgets. Within these limits the numbers are comparative measurements, **not general performance claims**.

The identified route to reconciling early detection with a low false positive rate is transforming the skewed features (for example, taking the logarithm of the non-negative features) before applying the threshold, so that **the MAD band matches the spread of the healthy data** without weakening the response to faults. A confirmation stage to filter isolated robust-rule alarms is a second option. For the SVM, the quantized FPR indicates that per-file normalization of the residual, or training that explicitly exposes the model to the mismatch patterns of healthy files, would be required before the boundary generalizes. All of these remain outside the scope of this study.

## 9. Conclusion

For step-like parameter drifts in reaction wheel telemetry, the **Haar wavelet** is the clearly suitable analysis tool, scoring approximately double the separability of the Morlet and Mexican Hat candidates. Built on that selection, the robust statistical detector and the SVM on digital-twin residuals both detected all 54 faulty files at statistically indistinguishable speeds (−2.70 s ± 4.90 s and −1.31 s ± 5.59 s respectively). Under a common healthy-only criterion, **neither satisfies the 1% false positive requirement**: the narrow reference that makes the robust rule early also makes it trigger on healthy tail values (29.24% mean FPR), and the SVM's decision boundary fails uniformly on a subset of healthy files (approximately 29% mean FPR), with per-file values quantized near 0% and 100% because the boundary separates files rather than windows. The rule that does satisfy the requirement (mean/σ, 0.52%) sacrifices the early detection that motivates the work. The robust rule is therefore the viable standalone trigger, the SVM indicates the behavior obtainable when a parallel simulation is affordable, and reconciling early detection with a low false positive rate is the next step.

## Appendix A — Glossary

- **Reaction wheel:** a spinning flywheel that rotates a satellite by conservation of angular momentum.
- **Telemetry:** the measurements transmitted from the satellite; here motor current (*i*) and angular velocity (*w*).
- **Torque constant (kt):** the factor relating motor current to produced torque; a drift in kt changes the current required for a given torque.
- **Bus voltage (vbus):** the supply voltage of the wheel electronics; a drift changes the electrical operating point and the current draw.
- **Injection time / onset:** the documented time at which a simulated fault begins.
- **Detection latency:** detection time minus injection time; negative values denote detections preceding the documented onset.
- **Wavelet:** a short, finite-duration function used to probe a signal.
- **Mother wavelet:** the template function slid across the signal by the transform.
- **Wavelet coefficient:** the value recording how strongly the template matches the signal at one time and one scale.
- **Stationary Wavelet Transform (SWT):** a wavelet transform whose output has the same length as the input, preserving time alignment.
- **Continuous Wavelet Transform (CWT):** a wavelet transform evaluated over a fine range of scales, giving a detailed spectral representation.
- **Detail / approximation coefficients:** the high-frequency and low-frequency components of the decomposition; cA₃ is the level-3 approximation (trend) component.
- **Feature vector:** the list of numbers summarizing one window of signal.
- **Energy:** the sum of squared coefficients in a window; a measure of signal strength.
- **Entropy:** the Shannon entropy of the energy distribution across bands; a measure of how evenly the energy is spread.
- **Spectral entropy:** the entropy of the energy distribution across wavelet scales in the SVM features.
- **Spectral centroid:** the center of the energy distribution across scales; a single number indicating where most of the energy sits.
- **Variance, RMS, Peak-to-Peak:** measures of the spread and magnitude of the values in a window.
- **Kurtosis:** the fourth standardized moment; high values indicate heavy tails, i.e., rare extreme values.
- **Skewness:** the third standardized moment; a measure of the asymmetry of the value distribution.
- **Fisher Discriminant Ratio:** the squared distance between two group means divided by the sum of their variances; large when the groups separate relative to their internal spread.
- **Median:** the middle value of a sample; insensitive to extreme values.
- **Median Absolute Deviation (MAD):** the median of the absolute deviations from the median, multiplied by 1.4826 so that it equals the standard deviation for normally distributed data; insensitive to extreme values.
- **3-sigma threshold:** a flag raised when a value lies more than three scale units from the reference center.
- **False positive rate (FPR):** the fraction of healthy windows flagged as faulty.
- **Digital twin:** a simulation producing the expected healthy telemetry at each moment.
- **Residual:** the actual features minus the expected features.
- **Support Vector Machine (SVM):** a classifier that separates two classes by a boundary in feature space.
- **Hyperparameters:** settings chosen before training, such as the kernel and the boundary flexibility parameters C and γ.
- **Five-fold group cross-validation:** partitioning the files into five groups and, in five rounds, training on four groups and predicting the fifth, so that every file is predicted once by a model that did not train on it; all windows of a file remain in the same group.
- **Standard deviation (σ):** the conventional measure of spread; for normally distributed data, ±1σ contains approximately the central two-thirds of values.
