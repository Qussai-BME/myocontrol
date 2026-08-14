# validation/ — moved to its own repository

The paper validation pipeline that used to live in this folder (NinaPro
DB2/DB3/DB7 LOSO benchmark, `process_engine.py`, `validate_engine.py`,
the `paper1_results/` tables, and the manuscript `.docx` files) has been
extracted into its own, purpose-built, standalone repository:

**https://github.com/Qussai-BME/sEMG-Zero-Calibration-LOSO-Benchmark**

It is a **separate codebase** from the `src/core/` package the rest of
this repository (and the Streamlit app) uses — related in spirit, not
shared code. Keeping it in its own repo means it can carry its own
citation, its own Zenodo DOI, and its own dependency list
(`statsmodels`, `lightgbm`, `PyWavelets`, ...) without those bleeding
into this product's `requirements.txt`.

## What used to be here, and where it went

| Was here | Now here |
|---|---|
| `process_engine.py`, `validate_engine.py`, `data_loaders.py`, `metrics.py`, `shap_analysis.py`, `statistical_reporter.py`, `cnn_baseline.py`, `report_generator.py`, `config.yaml`, and the numbered pipeline scripts (`01a_run_loso_benchmark.py`, etc.) | Same files, professionally renamed and reorganized, in `validation/` of the linked repo |
| `paper1_results/` (Table2_main_results.csv, TableS3_friedman_test.csv, confusion matrices, SHAP tables, ...) | `validation/results/` in the linked repo — byte-identical data, verified during the move |
| The manuscript `.docx` and `Supplementary.docx` | Not part of either public repo; see the linked repo's citation / DOI for the published paper once available |

## The "still-open question" from the earlier audit — now resolved

A previous audit of *this product's own* feature-extraction code
(`src/core/features_extended.py`, `features_fast.py`) found a real bug:
`ar_coefficients()` called `scipy.signal.lpc`, which does not exist in
any released scipy version, so AR-coefficient features silently
returned all-zeros. That earlier note flagged it as unverified whether
the paper's own pipeline (which imports `EMGConfig` /
`EMGFeatureExtractor` from `src/core_engine.py`) shared the same bug,
since that file wasn't available to check at the time.

It has since been reviewed directly. Two independent findings:

1. **`src/core_engine.py` does not compute AR coefficients at all** —
   its `EMGFeatureExtractor` only extracts MAV/RMS/ZCR/WL/SSC
   (time-domain) and MDF/MNF (frequency-domain) features. It has no
   `scipy.signal.lpc` call and no AR-related code.
2. **The paper's actual AR feature** is computed by an independent,
   self-contained function inside `process_engine.py` itself
   (`_ar_autocorr`, normalized-autocorrelation coefficients) — it does
   not call `scipy.signal.lpc` and does not depend on `core_engine.py`
   in any way.

So bug #2 in this product's own code does not carry over to the paper's
pipeline — different implementation, unaffected. Separately, AR
features were disabled in the paper's default configuration
(`compute_ar: false`, "no benefit, 3x slower") and the paper itself
already treats "AR ≈0% SHAP importance" as an unverified,
SHAP-based observation rather than an ablated result (§5.7) — that
caveat still stands and is unrelated to this bug question.

## Reproducing the paper's numbers

See the linked repository's own README for installation and the full
`01a`→`06b` pipeline. NinaPro DB2/DB3/DB7 are third-party datasets
(registration required at http://ninaweb.hevs.ch) and are not
redistributed there either. The full code/data archive, including
pre-computed feature matrices and result JSONs, is at
https://doi.org/10.5281/zenodo.20982280.
