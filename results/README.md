# Validation run logs — CORRECTED, read this first

**These logs are from `scripts/reproduce_loso_validation.py` run against
the UCI "EMG data for gestures" dataset — a different dataset than the
one actually behind README.md's Scientific Validation numbers.** An
earlier audit pass assumed the README's "6 subjects" claim referred to
this UCI dataset. That assumption was wrong. The real validation uses
three NinaPro databases (DB2/DB3/DB7) and is maintained in its own
repository,
[sEMG-Zero-Calibration-LOSO-Benchmark](https://github.com/Qussai-BME/sEMG-Zero-Calibration-LOSO-Benchmark)
— see that repository's `validation/results/` for the real result
tables and this repo's `README.md`'s Scientific Validation section for
the corrected numbers.

**What the logs in this folder actually show:** a standalone sanity
check of *this repository's own* `src/core/lodo_cv.py` pipeline against
the UCI dataset — legitimate on its own terms, just not a check of the
paper's claims (which use different code, `src/core_engine.py` in the
linked repository, entirely).

- **`loso_validation_20260724_070311.*`** — 6 subjects (01-06), LDA,
  k_features=420, Euclidean Alignment disabled. Result: 21.4% accuracy
  on this repo's own pipeline against UCI data.
- **`loso_validation_20260724_070420.*`** — 3 subjects (01-03), LDA,
  k_features=420, Euclidean Alignment enabled. Result: 15.5% accuracy.

These numbers are not wrong, but they answer a different question than
"does the paper's 65.96% on NinaPro DB7 hold up" — they don't, because
they're a different dataset and a different feature-extraction codebase.
If you want to sanity-check *this repo's* pipeline again, these logs
(and the script that produced them) are still useful for that. If you
want to check the paper's actual numbers, see the
[sEMG-Zero-Calibration-LOSO-Benchmark](https://github.com/Qussai-BME/sEMG-Zero-Calibration-LOSO-Benchmark)
repository instead.

**Before citing any accuracy number for this project, check which
pipeline and which dataset produced it** — this repo (`src/core/`) and
the paper's validation (`src/core_engine.py` in the linked repository)
are two different, related-but-separate implementations.
