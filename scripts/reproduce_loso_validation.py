#!/usr/bin/env python3
"""
scripts/reproduce_loso_validation.py

*** CORRECTION NOTICE (read this first) ***
This script was originally written under the assumption that
README.md's "Scientific Validation" numbers came from the UCI "EMG data
for gestures" dataset. That assumption was WRONG. The actual validation
behind those numbers uses three NinaPro databases (DB2/DB3/DB7) and
lives in its own repository,
https://github.com/Qussai-BME/sEMG-Zero-Calibration-LOSO-Benchmark
(a separate codebase, `src/core_engine.py`, not the `src.core.*`
package this script imports from) — see that repository's
`validation/results/` for the real result tables and README.md's
Scientific Validation section for the corrected numbers.

This script is still a legitimate, working LOSO validator for the UCI
dataset against *this repository's* `src/core/lodo_cv.py` pipeline — 
useful if you want to validate that pipeline specifically — but it does
NOT reproduce, and was never going to reproduce, the NinaPro-based paper
numbers in README.md. Don't point this script at that comparison again;
use the dedicated repository above for that.
*** END CORRECTION NOTICE ***

Runs a LOSO validation of src/core/lodo_cv.py + features_extended.py
(this repository's own pipeline) against the UCI "EMG data for gestures"
dataset. Useful as a standalone sanity check of this repo's pipeline on
a real, different dataset than NinaPro -- not as a reproduction of the
paper's own results (see notice above).

ORIGINAL RATIONALE (kept for context; the dataset assumption below was
the error -- the rest of the reasoning, about needing a real
reproducible script rather than hand-typed numbers, still stands and is
why this script exists at all)
-----------------------------------------------------------------------
As of this script's creation, the numbers in README.md (e.g. "XGBoost:
70.14% +/- 8.81%", "Friedman test p=0.6065") did not correspond to any
script, log, or results file anywhere in this repository as it stood at
the time -- that's a real gap between "the code has a test suite" and
"these specific numbers are substantiated," independent of which
dataset turned out to be the right one to check against. This script
was written to close that gap:

  - Run it against the UCI dataset for a sanity check of THIS repo's
    own `src/core/lodo_cv.py` pipeline.
  - For the actual paper numbers, use the dedicated
    sEMG-Zero-Calibration-LOSO-Benchmark repository instead -- see the
    correction notice above.

A NOTE ON WHY EXACT REPRODUCTION MAY NOT BE POSSIBLE EVEN WITH THE RIGHT
DATA: two real bugs were found and fixed in the feature-extraction code
after this project's README was written (ZCR/SSC/WAMP/MYOP were computed
with a hardcoded threshold=0, which saturates them to a near-constant
value carrying ~zero information -- see EMGConfig.feature_threshold_multiplier
and the TD_FEATURES comment in features_extended.py). If whatever
produced the original README numbers used the pre-fix code, those numbers
reflect a measurably different (and less correct) feature space than the
one this script now uses. A mismatch against the README is therefore not
necessarily a red flag on its own -- read the log's "vs README" section
for the actual comparison and don't just eyeball the headline accuracy.

DATA YOU NEED
-------------
UCI "EMG data for gestures" (Lobov et al. 2018, CC BY 4.0):
  https://archive.ics.uci.edu/dataset/481/emg+data+for+gestures
or the dataset authors' own GitHub mirror:
  git clone https://github.com/UNNLobachevsky/EMG_data_for_gestures.git

Expected --data-dir layout: one subdirectory per subject ("01", "02",
...), each containing one or more "*.txt" files in the UCI raw format
(10 tab-separated columns: time[ms], channel1..channel8, class[0-7]).

USAGE
-----
    python scripts/reproduce_loso_validation.py \\
        --data-dir /path/to/EMG_data_for_gestures \\
        --subjects 01,02,03,04,05,06 \\
        --out-dir results/

    # Quick smoke test (small k, no EA, ~1 min) before committing to a
    # full run:
    python scripts/reproduce_loso_validation.py \\
        --data-dir /path/to/EMG_data_for_gestures \\
        --subjects 01,02 --k-features 50 --no-euclidean-alignment \\
        --classifiers lda

Produces a timestamped .log file (human-readable) and a .json file
(machine-readable) in --out-dir.
"""
import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.dataset_loader import robust_read_table  # noqa: E402
from src.core.lodo_cv import LOSOCrossValidator  # noqa: E402
from src.core.evaluation import friedman_test, wilcoxon_pairwise, per_subject_accuracy  # noqa: E402

CLASS_NAMES = {
    0: 'unmarked',
    1: 'rest',
    2: 'fist',
    3: 'wrist_flexion',
    4: 'wrist_extension',
    5: 'radial_deviation',
    6: 'ulnar_deviation',
    7: 'extended_palm',
}

# These are the RETRACTED numbers that used to be in README.md's
# Scientific Validation section, kept here only so this script's log can
# show what was retracted and why -- NOT a target to reproduce or tune
# toward. They were sourced from the wrong dataset entirely (see the
# correction notice at the top of this file): the real, current numbers
# are NinaPro-based and live in the dedicated
# sEMG-Zero-Calibration-LOSO-Benchmark repository, not here.
RETRACTED_README_CLAIMS = {
    'n_subjects': 6,
    'per_classifier': {
        'XGBoost': {'accuracy_mean': 0.7014, 'accuracy_std': 0.0881, 'macro_f1_mean': 0.6918},
        'LDA':     {'accuracy_mean': 0.6903, 'accuracy_std': 0.0981, 'macro_f1_mean': 0.6777},
        'RF':      {'accuracy_mean': 0.7059, 'accuracy_std': 0.1015, 'macro_f1_mean': 0.6958},
    },
    'friedman_p': 0.6065,
}

CLASSIFIER_FACTORIES = {
    'xgboost': lambda: __import__('xgboost').XGBClassifier(
        n_estimators=200, max_depth=4, eval_metric='mlogloss',
        use_label_encoder=False, verbosity=0),
    'lda': lambda: __import__('sklearn.discriminant_analysis', fromlist=['LinearDiscriminantAnalysis']).LinearDiscriminantAnalysis(),
    'rf': lambda: __import__('sklearn.ensemble', fromlist=['RandomForestClassifier']).RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=0, n_jobs=-1),
}
CLASSIFIER_DISPLAY_NAMES = {'xgboost': 'XGBoost', 'lda': 'LDA', 'rf': 'RF'}


def load_subject(subject_dir: Path, drop_unmarked: bool = True) -> dict:
    """Load every raw_data_*.txt file in one subject folder into a single
    (signal, labels, label_times) record, in the format
    LOSOCrossValidator.run_loso() expects.

    UCI files are one row per SAMPLE with a dense per-sample class column
    -- not the sparse label_times format some other loaders in this repo
    use. We pass label_times as literally every sample's own timestamp;
    run_loso's per-window "last label at or before window time" lookup
    handles a dense timeline correctly, just less efficiently than a
    sparse one (fine at this data scale).

    XGBoost's own docstring for fs: this dataset's time column is in ms
    and is explicitly documented as ~1 kHz logging by prior work using
    this exact dataset -- see the module docstring. Multiple consecutive
    identical readings are common (Bluetooth packet holds), which affects
    the reliability of frequency-domain features on this data; that's a
    property of the dataset, not a bug in this script.
    """
    files = sorted(Path(subject_dir).glob("*.txt"))
    if not files:
        raise FileNotFoundError(
            f"No .txt files found in {subject_dir} -- see this script's "
            "module docstring for where to get the dataset and the "
            "expected directory layout.")

    signals, labels, times = [], [], []
    t_offset = 0.0
    for f in files:
        df = robust_read_table(f)
        df.columns = [c.strip().lower() for c in df.columns]
        missing = [c for c in ['time', 'class'] + [f'channel{i}' for i in range(1, 9)]
                   if c not in df.columns]
        if missing:
            raise ValueError(f"{f} is missing expected columns: {missing}")

        t = df['time'].to_numpy(dtype=float) / 1000.0 + t_offset  # ms -> s
        sig = df[[f'channel{i}' for i in range(1, 9)]].to_numpy(dtype=float)
        cls = df['class'].to_numpy(dtype=int)

        if drop_unmarked:
            mask = cls != 0
            t, sig, cls = t[mask], sig[mask], cls[mask]

        signals.append(sig)
        labels.append(cls)
        times.append(t)
        t_offset = float(t[-1]) + 0.001 if len(t) else t_offset

    signal = np.concatenate(signals, axis=0)
    label_ints = np.concatenate(labels, axis=0)
    label_times = np.concatenate(times, axis=0)
    label_strs = np.array([CLASS_NAMES.get(int(c), str(c)) for c in label_ints])

    return {
        'signal': signal,
        'labels': label_strs,
        'label_times': label_times,
        'subject_id': Path(subject_dir).name,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce the LOSO validation numbers claimed in README.md",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument('--data-dir', required=True, type=Path,
                        help="Path to the extracted UCI EMG data for gestures dataset "
                             "(folder containing subject subdirectories '01', '02', ...)")
    parser.add_argument('--subjects', default='01,02,03,04,05,06',
                        help="Comma-separated subject folder names (default: first 6, "
                             "matching README.md's claimed n=6)")
    parser.add_argument('--classifiers', default='xgboost,lda,rf',
                        help="Comma-separated: xgboost,lda,rf")
    parser.add_argument('--window-ms', type=int, default=400)
    parser.add_argument('--overlap', type=float, default=0.5)
    parser.add_argument('--k-features', type=int, default=420,
                        help="SelectKBest feature count (README/paper default: 420)")
    parser.add_argument('--fs', type=int, default=1000,
                        help="Sampling rate for windowing (this dataset's time column "
                             "is in ms at ~1 kHz logging resolution -- see module docstring)")
    parser.add_argument('--no-euclidean-alignment', action='store_true',
                        help="Disable Euclidean Alignment domain adaptation (faster, "
                             "but not what the README describes)")
    parser.add_argument('--keep-unmarked', action='store_true',
                        help="Keep class 0 ('unmarked'/transition periods) instead of "
                             "dropping it before windowing")
    parser.add_argument('--out-dir', type=Path, default=Path('results'))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = args.out_dir / f'loso_validation_{timestamp}.log'
    json_path = args.out_dir / f'loso_validation_{timestamp}.json'

    logging.basicConfig(
        level=logging.INFO, format='%(message)s',
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)])
    log = logging.getLogger('reproduce_loso_validation')

    log.info("=" * 78)
    log.info("LOSO VALIDATION REPRODUCTION RUN")
    log.info(f"Started: {datetime.now().isoformat()}")
    log.info("=" * 78)

    subjects_list = [s.strip() for s in args.subjects.split(',')]
    classifiers_list = [c.strip() for c in args.classifiers.split(',')]
    for c in classifiers_list:
        if c not in CLASSIFIER_FACTORIES:
            log.error(f"Unknown classifier '{c}'. Choose from: {list(CLASSIFIER_FACTORIES)}")
            sys.exit(1)

    log.info(f"\nConfig: window_ms={args.window_ms} overlap={args.overlap} "
             f"k_features={args.k_features} fs={args.fs} "
             f"euclidean_alignment={not args.no_euclidean_alignment} "
             f"drop_unmarked={not args.keep_unmarked}")
    log.info(f"Subjects requested: {subjects_list}")
    log.info(f"Classifiers: {classifiers_list}")

    # ---- Load data ----
    log.info("\n" + "-" * 78)
    log.info("LOADING DATA")
    log.info("-" * 78)
    subjects_data = []
    t0 = time.time()
    for s in subjects_list:
        subj_dir = args.data_dir / s
        if not subj_dir.exists():
            log.error(f"Subject directory not found: {subj_dir}")
            log.error("See this script's module docstring for how to get the dataset.")
            sys.exit(1)
        try:
            rec = load_subject(subj_dir, drop_unmarked=not args.keep_unmarked)
        except Exception as e:
            log.error(f"Failed to load subject '{s}': {e}")
            sys.exit(1)
        classes_present = sorted(set(rec['labels'].tolist()))
        log.info(f"  Subject {s}: {rec['signal'].shape[0]} samples, "
                 f"{rec['signal'].shape[1]} channels, classes={classes_present}")
        subjects_data.append(rec)
    log.info(f"Loaded {len(subjects_data)} subjects in {time.time() - t0:.1f}s")

    # ---- Run LOSO per classifier ----
    cv = LOSOCrossValidator(
        fs=args.fs, window_ms=args.window_ms, overlap=args.overlap,
        k_features=args.k_features, include_freq=True, include_inter_channel=True,
        use_euclidean_alignment=not args.no_euclidean_alignment, verbose=True)

    all_results = {}
    per_fold_accuracy = {}
    per_fold_macro_f1 = {}
    per_subject_acc = {}

    for clf_key in classifiers_list:
        display_name = CLASSIFIER_DISPLAY_NAMES[clf_key]
        log.info("\n" + "-" * 78)
        log.info(f"RUNNING LOSO: {display_name}")
        log.info("-" * 78)
        t0 = time.time()
        result = cv.run_loso(subjects_data, CLASSIFIER_FACTORIES[clf_key],
                             classifier_name=display_name)
        elapsed = time.time() - t0
        log.info(f"{display_name} LOSO complete in {elapsed:.1f}s")
        log.info(f"  accuracy_mean={result['accuracy_mean']:.4f} "
                 f"accuracy_std={result['accuracy_std']:.4f}")
        log.info(f"  macro_f1_mean={result['macro_f1_mean']:.4f}")

        all_results[display_name] = result
        per_fold_accuracy[display_name] = [f['accuracy'] for f in result['per_fold']]
        per_fold_macro_f1[display_name] = [f['macro_f1'] for f in result['per_fold']]
        per_subject_acc[display_name] = per_fold_accuracy[display_name]

    # ---- Statistical tests across classifiers ----
    log.info("\n" + "-" * 78)
    log.info("STATISTICAL TESTS ACROSS CLASSIFIERS")
    log.info("-" * 78)
    friedman_result = None
    wilcoxon_result = None
    if len(classifiers_list) >= 2:
        friedman_result = friedman_test(per_fold_accuracy)
        log.info(f"Friedman test: chi2={friedman_result.get('chi2'):.4f} "
                 f"p={friedman_result.get('p_value'):.4f} "
                 f"({friedman_result.get('decision')})")
        wilcoxon_result = wilcoxon_pairwise(per_fold_accuracy)
        for pair, res in wilcoxon_result.items():
            log.info(f"  Wilcoxon {pair}: p={res['p_value']:.4f} "
                     f"adj_p={res['adjusted_p']:.4f} "
                     f"significant={res['significant']} "
                     f"(Cohen's d={res['cohens_d']:.3f}, {res['interpretation']})")
    else:
        log.info("Only one classifier run -- skipping Friedman/Wilcoxon (need >=2).")

    subj_summary = per_subject_accuracy(per_subject_acc)
    log.info("\nPer-subject accuracy range by classifier:")
    for name, s in subj_summary.items():
        log.info(f"  {name}: min={s['min']:.3f} max={s['max']:.3f} "
                 f"range={s['range_pp']:.3f}")

    # ---- Compare against the RETRACTED numbers (for context only) ----
    log.info("\n" + "=" * 78)
    log.info("CONTEXT: comparison against the RETRACTED (wrong-dataset) numbers")
    log.info("that used to be in README.md -- NOT a target, NOT the paper's real")
    log.info("numbers. See correction notice at the top of this script.")
    log.info("=" * 78)
    if len(subjects_list) != RETRACTED_README_CLAIMS['n_subjects']:
        log.info(f"NOTE: this run used {len(subjects_list)} subjects; the retracted "
                 f"claim used n={RETRACTED_README_CLAIMS['n_subjects']}. Subject count "
                 "alone can shift these numbers substantially -- treat any comparison "
                 "below as indicative, not a strict apples-to-apples check, and "
                 "remember the retracted numbers were from a different dataset anyway.")

    for name, result in all_results.items():
        claimed = RETRACTED_README_CLAIMS['per_classifier'].get(name)
        if claimed is None:
            continue
        acc_diff = result['accuracy_mean'] - claimed['accuracy_mean']
        log.info(f"{name}: this run={result['accuracy_mean']*100:.2f}% "
                 f"(+/-{result['accuracy_std']*100:.2f}) vs retracted claim={claimed['accuracy_mean']*100:.2f}% "
                 f"(+/-{claimed['accuracy_std']*100:.2f})  diff={acc_diff*100:+.2f}pp")

    if friedman_result and 'p_value' in friedman_result:
        p_diff = friedman_result['p_value'] - RETRACTED_README_CLAIMS['friedman_p']
        log.info(f"\nFriedman p: this run={friedman_result['p_value']:.4f} "
                 f"vs retracted claim={RETRACTED_README_CLAIMS['friedman_p']:.4f}  diff={p_diff:+.4f}")

    log.info("\n" + "=" * 78)
    log.info("VERDICT: this run validates THIS repository's own src/core/lodo_cv.py "
             "pipeline on the UCI dataset as a standalone sanity check. It does NOT "
             "confirm or refute the paper's real NinaPro-based numbers -- for that, "
             "see the sEMG-Zero-Calibration-LOSO-Benchmark repository and "
             "README.md's Scientific Validation section.")
    log.info("=" * 78)

    # ---- Save machine-readable summary ----
    summary = {
        'timestamp': timestamp,
        'config': vars(args) | {'data_dir': str(args.data_dir), 'out_dir': str(args.out_dir)},
        'subjects_loaded': [s['subject_id'] for s in subjects_data],
        'results': {name: {
            'accuracy_mean': r['accuracy_mean'],
            'accuracy_std': r['accuracy_std'],
            'macro_f1_mean': r['macro_f1_mean'],
            'per_fold_accuracy': per_fold_accuracy[name],
        } for name, r in all_results.items()},
        'friedman': friedman_result,
        'wilcoxon': wilcoxon_result,
        'per_subject_summary': subj_summary,
        'retracted_readme_claims': RETRACTED_README_CLAIMS,
    }

    def _json_default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, Path):
            return str(o)
        return str(o)

    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2, default=_json_default)

    log.info(f"\nLog written to: {log_path}")
    log.info(f"JSON summary written to: {json_path}")


if __name__ == '__main__':
    main()
