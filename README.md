# MyoControl Suite v0.5 — Production-Ready EMG AI Platform

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21940052.svg)](https://doi.org/10.5281/zenodo.21940052)

🚀 **[Live Interactive Demo](https://myocontrol-qussai-bme.streamlit.app/)** — try the classifier in your browser, no install needed## 🔧 Audit Log — bugs found and fixed

This repository went through a thorough correctness audit (feature
extraction, classifier training/explainability, edge deployment,
real-time inference, and documentation accuracy). Listed here for full
transparency rather than buried in commit history — every item below
was independently reproduced, fixed, and locked in with a regression
test before being marked done.

| # | Bug | Where | Impact |
|---|---|---|---|
| 1 | WAMP/SSC/MYOP/ZCR computed with a hardcoded `threshold=0`, saturating them to a near-constant value carrying ~zero information | `features.py`, `features_extended.py`, `features_fast.py` (3 separate implementations, same bug in each) | These 4 features contributed nothing to any classifier trained before this fix |
| 2 | `ar_coefficients()` called `scipy.signal.lpc`, which does not exist in any released scipy version — silently returned all-zeros on every call | `features_extended.py` | AR features were dead weight in every LOSO run to date; likely explains the "AR ≈0% SHAP importance" observation that motivated dropping them from `features_fast.py`'s 308-feature set |
| 3 | SHAP output-shape handling assumed an old shap API convention, breaking on current shap/xgboost versions | `classifier.py` | `explain()` failed outright for XGBoost; fixed with format-agnostic shape handling + an automatic fallback to a model-agnostic explainer if the fast path still fails |
| 4 | `cross_val_score`'s per-fold NaN (a degenerate LOSO fold) silently poisoned the overall mean into an unexplained NaN | `classifier.py` | LOSO accuracy showed "NaN" with zero indication of why; now reports valid-fold count, failed-fold count, and the reason |
| 5 | README's Scientific Validation numbers didn't match any script/log in the repo, **and** an earlier audit pass wrongly assumed the underlying validation used the UCI dataset (it uses NinaPro DB2/DB3/DB7) | `README.md` | See the Scientific Validation section below for the correction and the real, verified numbers |
| 6 | `DANClassifier` crashed with a raw `NameError` if PyTorch wasn't installed, despite the docstring claiming a NumPy fallback existed (it never did) | `dan_baseline.py` | Now raises a clear `ImportError` at construction time instead |
| 7 | ONNX export failed outright for `LinearSVC` (`skl2onnx`'s default `zipmap` option isn't supported for classifiers without a native `predict_proba`) | `edge/onnx_export.py` | One of 4 supported classifier types couldn't be deployed to ONNX at all; fixed with a `nocl=True` fallback |
| 8 | **`RealtimeEMGEngine` used `features_fast.py`'s ~38-feature/channel schema, while every model actually trained through this product (`EMGClassifier`) uses `features.py`'s 7-feature schema** | `edge/realtime_inference.py` | Real-time inference could not work with any model trained through the normal product workflow — crashed with an ONNX shape-mismatch error on every call. This is the most severe item on this list. Fixed by making the real-time engine use the same feature pipeline models are actually trained with |
| 9 | `ShapReportGenerator` mishandled the same modern-shap 3D output shape as #3, in a third independent implementation — silently returned a meaningless, sometimes-duplicated "top features" list instead of erroring | `edge/shap_reports.py` | Per-prediction transparency reports could show the same feature 2-4× while omitting genuinely important ones |
| 10 | Lite-DAN's "52,649 parameters" claim didn't match its own Quick Start example's configuration | `README.md` | Cosmetic — corrected to 52,880, the actual count for the documented example |

Items 1-4 and 6-9 are locked in with regression tests (`tests/test_*.py`,
169 passing). Items 5 and 10 are documentation corrections. See the
Scientific Validation section for how bug #2 was checked against the
separate codebase behind the paper's real numbers — resolved, not
applicable there (different implementation).

## 📦 Installation

```bash
# Clone
git clone https://github.com/Qussai-BME/myocontrol.git
cd myocontrol

# Quick install (all features)
pip install -r requirements.txt

# Or install as editable package with optional extras:
pip install -e .                          # core only (lightweight)
pip install -e ".[ui,api]"                # + dashboard + API
pip install -e ".[ui,api,onnx,torch]"     # + ONNX edge + Lite-DAN
pip install -e ".[all]"                   # everything

# Run the dashboard
streamlit run src/ui/app.py

# Run the API server
uvicorn src.api.server:app --reload --port 8000
```

## 🆕 What's New in v0.5 (vs v0.4)

### ⚡ Performance (4× faster)
- **Numba JIT acceleration**: 1,600+ windows/sec (vs 584 in v0.4)
- **308 features per window** (vs 618) — AR coefficients removed based on an observed "SHAP ≈ 0" finding that turned out to be a bug, not a real result (see Scientific Validation below); re-adding a correct implementation is planned, not yet done
- **Manual DFT** for PSD (Numba-compatible)
- **CPU-only**: works on Intel Core i3 / 4GB RAM

### 🚀 Edge Deployment (NEW)
- **ONNX export**: works for all 4 classifier types (XGBoost, LDA, RandomForest, LinearSVC — LinearSVC export was broken before this audit, see Audit Log #7). Throughput is hardware- and classifier-dependent: this audit measured 46K samples/sec (XGBoost) to 5.7M samples/sec (LinearSVC) on its own test machine — re-benchmark on your actual target hardware before quoting a specific number to anyone; don't reuse "117K samples/sec" as a fixed claim
- **Real-time capable**: sub-millisecond per-sample latency measured across all 4 classifier types in this audit's environment, comfortably under any real-time EMG control loop's budget
- **Cross-platform**: Python, C++, JavaScript, Rust via ONNX Runtime
- **Lite-DAN**: 52K parameters (vs SOTA's millions)

### 🎯 Scientific Validation

**Correction:** this section previously cited "UCI EMG (6 subjects)" with
XGBoost 70.14% / LDA 69.03% / RF 70.59% and Friedman p=0.6065. Those
numbers were wrong, but not for the reason an earlier audit of this repo
concluded. That audit assumed the underlying validation used the UCI
"EMG data for gestures" dataset, built a reproduction script for it, got
very different (much lower) numbers, and concluded the original claim
was likely never computed. **That assumption was wrong** — the real
validation, provided by the author (now maintained in its own
repository, [sEMG-Zero-Calibration-LOSO-Benchmark][validation-repo], the
full LOSO benchmark behind the paper cited below), uses three
**NinaPro** databases, not UCI. The corrected numbers below are read
directly from that repository's own result tables
(`validation/results/Table2_main_results.csv`, `TableS3_friedman_test.csv`)
and match the paper's Abstract and Results sections exactly.

[validation-repo]: https://github.com/Qussai-BME/sEMG-Zero-Calibration-LOSO-Benchmark

**Real data LOSO cross-validation** (Leave-One-Subject-Out, zero
subject-specific calibration), 420-dimensional hand-crafted feature set,
4 classifiers, across three NinaPro databases:

| Database | Population | N subjects | XGBoost accuracy | XGBoost macro-F1 | Friedman p (4 classifiers) |
|---|---|---|---|---|---|
| DB7 | mixed | 22 | 65.96% ± 6.01% | 27.14% | 0.179 (n.s.) |
| DB2 | intact-limb | 40 | 54.64% ± 7.84% | 21.89% | <0.001 (sig.) |
| DB3 | transradial amputee | 11 | 43.46% ± 10.73% | 4.01% | <0.001 (sig.) |

Key findings (see the manuscript, cited below, for full methodology,
limitations, and statistical corrections):
- On DB7 only, all four classifiers perform statistically equivalently
  (Friedman p=0.179) — the feature space, not classifier choice, is the
  binding constraint there. DB2 and DB3 both show *significant*
  inter-classifier differences, so this is not a universal finding.
- DB3's 43.46% accuracy vs. 4.01% macro-F1 is a **39.86 percentage-point
  Rest-class inflation gap** — the paper's central methodological point:
  overall accuracy alone badly overstates real gesture-recognition
  ability on amputee data.
- A naive CNN-1D baseline (no transfer learning, no pretraining) scored
  21.60% / 15.58% / 4.69% on DB7/DB2/DB3 — 38-44pp below the classical
  pipeline.
- The paper's own §5.6 Limitations (10 items, unusually candid) and §5.7
  Future Directions are worth reading before quoting any number here —
  including that Euclidean Alignment's independent contribution was not
  separately ablated, and that "AR coefficients ≈0% SHAP importance" is
  explicitly reported as "a SHAP-based inference only," not yet an
  ablated, verified result.

**A separate question this correction raised, now checked and resolved:**
an earlier pass on *this product's own code*
(`src/core/features_extended.py`, `features_fast.py`) found and fixed a
real, confirmed bug — `scipy.signal.lpc` does not exist in any released
scipy version, so AR-coefficient features silently returned all-zeros
in *this* codebase. That bug is real and independent of the paper's
numbers. It was found in a different codebase than the one behind the
paper: the paper's validation imports `EMGConfig` /
`EMGFeatureExtractor` from `src/core_engine.py`, which at the time
wasn't available to check. It has since been reviewed directly, with
two findings: (1) `core_engine.py`'s `EMGFeatureExtractor` doesn't
compute AR coefficients at all — no `scipy.signal.lpc` call anywhere in
it; (2) the paper's actual AR feature is computed by an independent,
self-contained function in `process_engine.py`
(`_ar_autocorr`, normalized-autocorrelation coefficients), unrelated to
`core_engine.py` and unaffected by the `scipy.signal.lpc` bug. So bug #2
does not carry over to the paper's pipeline. This doesn't change the
paper's own treatment of AR importance as an unverified, SHAP-based
observation rather than an ablated result (§5.7) — that caveat is
independent of this bug question and still stands.

**Reproducing these numbers:** the full analysis code, checkpoints, and
result tables are maintained in the dedicated
[sEMG-Zero-Calibration-LOSO-Benchmark][validation-repo] repository (see
link above), including `src/core_engine.py`.
`scripts/reproduce_loso_validation.py` in *this* repository targets the
UCI dataset instead and does **not** correspond to this paper's claims —
see the warning at the top of that script. NinaPro DB2/DB3/DB7 are
available at http://ninaweb.hevs.ch (registration required); the
paper's full code/data archive is at
https://doi.org/10.5281/zenodo.21940345.

### 🔍 Transparency (NEW)
- **SHAP-based transparency reports** for every prediction
- **Per-channel contribution** breakdown
- **Per-feature-group importance**
- **Trust score** calibration
- **Designed with the transparency principles the EU AI Act requires** (explainability, traceability) — this is a design intent, not an independent legal compliance audit; don't represent it as one without actual legal review

### 🧠 Lite-DAN (NEW)
- **Domain Adversarial Network**: 52,880 parameters (for the Quick Start example's config: n_features=308, n_classes=6, n_domains=10 — exact count scales with those, see `DANClassifier.count_parameters()`)
- **Gradient Reversal Layer** for subject-invariant features
- **CPU-only training**: 10 minutes on Intel Core i3 *(not independently re-timed in this audit — verify on your own target hardware before quoting this to anyone)*
- **Expected LOSO improvement**: +15-25% over XGBoost baseline *("Expected" = a projection, not a measured result — no ablation for this exists yet, see paper §5.6/§5.7)*
- **Requires PyTorch** (`pip install torch`) — there is no NumPy-only fallback; `DANClassifier` raises a clear `ImportError` at construction time if PyTorch isn't installed, rather than the confusing crash it used to produce

## 📦 New Files

```
src/core/
├── features_fast.py      # Numba-accelerated (4× faster)
├── dan_baseline.py       # Lite-DAN (NEW)
└── (existing files from v0.4)

src/edge/                 # NEW directory
├── onnx_export.py        # ONNX export + inference
├── realtime_inference.py # Real-time engine
└── shap_reports.py       # Transparency reports
```

**Audit note:** none of `features_fast.py`, `dan_baseline.py`,
`onnx_export.py`, `realtime_inference.py`, or `shap_reports.py` had any
test coverage before this audit. Actually running each end-to-end (not
just importing it) found and fixed real bugs in every one of them:
- `dan_baseline.py`: `DANClassifier` crashed with a raw `NameError` if
  PyTorch wasn't installed, despite the docstring claiming a NumPy
  fallback existed (it didn't). Now raises a clear `ImportError` at
  construction time instead.
- `onnx_export.py`: exporting a `LinearSVC`-based classifier failed
  outright (skl2onnx's default `zipmap` option isn't supported for
  LinearSVC). Added a `nocl=True` fallback that actually works for it.
- `realtime_inference.py`: **`RealtimeEMGEngine` used a completely
  different, incompatible feature set internally
  (`features_fast.py`, ~38 features/channel) than the one any model
  trained through this product actually uses (`EMGClassifier`,
  `features.py`'s 7-feature `TIME_FEATURES`)** — real-time inference
  crashed with an ONNX shape-mismatch error on every call against any
  normally-trained model. Fixed to use the same feature pipeline
  `EMGClassifier` trains on.
- `shap_reports.py`: modern shap's 3D `(samples, features, classes)`
  output for multi-class trees wasn't handled, causing `top_features`
  in transparency reports to show the same feature name repeated
  multiple times instead of a genuinely ranked list. Fixed with the
  same shape-handling logic already applied to
  `EMGClassifier.explain()`.

All four now have real test coverage (`tests/test_dan_baseline.py`,
`tests/test_onnx_export.py`, `tests/test_realtime_inference.py`,
`tests/test_shap_reports.py`) that exercises them against an actual
`EMGClassifier`-trained model, not just mocks.

## 🚀 Quick Start

```python
# 1. Train XGBoost classifier
from src.core.classifier import EMGClassifier
clf = EMGClassifier(model_type='xgboost')
clf.fit(X_train, y_train)

# 2. Export to ONNX (for Edge deployment)
from src.edge.onnx_export import OnnxExporter
OnnxExporter.export(clf, 'model.onnx', n_features=X.shape[1])

# 3. Real-time inference (< 5ms)
from src.edge.onnx_export import OnnxInferenceEngine
engine = OnnxInferenceEngine('model.onnx')
predictions = engine.predict(X_new)  # sub-millisecond per sample (measured; exact throughput is hardware/classifier-dependent)

# 4. Generate SHAP transparency report
from src.edge.shap_reports import ShapReportGenerator
gen = ShapReportGenerator(clf)
report = gen.explain_prediction(X[:1], predicted_class='rest', confidence=0.95)
print(report.explanation)

# 5. Train Lite-DAN (Domain Adversarial)
from src.core.dan_baseline import DANClassifier
dan = DANClassifier(n_features=308, n_classes=6, n_domains=10, n_epochs=50)
dan.fit(X, y, domains=subject_ids)  # subject-invariant features
```

## 📊 Performance Comparison

| Metric | v0.3 | v0.4 | **v0.5** |
|---|---|---|---|
| Features per window | 10 | 618 | **308** |
| Feature extraction speed | 200 win/sec | 584 win/sec | **1,600+ win/sec** |
| ONNX export | ✗ | ✗ | **✓ (46K-5.7M samples/sec, classifier-dependent — re-verify on your hardware)** |
| Lite-DAN | ✗ | ✗ | **✓ (52K params)** |
| SHAP reports | partial | partial | **✓ (full transparency)** |
| Real data LOSO | ✗ | ✗ | **✓ (65.96% on NinaPro DB7, 22 subjects — see Scientific Validation)** |
| Statistical tests | ✗ | ✓ | **✓** |
| CPU-only training | ✓ | ✓ | **✓** |
| Edge deployment | ✗ | ✗ | **✓** |

## 📄 Citation

```bibtex
@article{adlbi2026restclass,
  title={Rest-Class Metric Inflation in Zero-Calibration
         Cross-Subject sEMG: A Three-Database LOSO Benchmark},
  author={Adlbi, Q. and Darwich, M. A.},
  journal={Biomedical Signal Processing and Control (under review)},
  year={2026}
}
```

## 👤 Author

**Qussai Adlbi** — Biomedical Engineering Student
- Al-Andalus University for Medical Sciences (Syria) · Pázmány Péter Catholic University (Hungary)
- qussai.adlbi@au.edu.sy · GitHub: @Qussai-BME

## 📄 License

MIT
