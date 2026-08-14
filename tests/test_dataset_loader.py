"""
test_dataset_loader.py - Tests for src/core/dataset_loader.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.core.dataset_loader import (
    load_features_csv, load_raw_emg_and_extract, load_raw_labeled_table,
    load_ninapro_mat, inspect_mat_fields, load_mat_numeric_field,
    concatenate_subjects,
)
from src.core.classifier import EMGClassifier
from src.core.simulator import EMGSimulator
from src.core.features import TIME_FEATURES


@pytest.fixture
def feature_csv(tmp_path):
    rng = np.random.RandomState(0)
    df = pd.DataFrame({f: rng.rand(60) for f in TIME_FEATURES})
    df['label'] = rng.choice(['fist', 'open'], 60)
    df['subject'] = rng.choice(['s1', 's2', 's3'], 60)
    path = tmp_path / "features.csv"
    df.to_csv(path, index=False)
    return path


class TestLoadFeaturesCSV:
    def test_loads_expected_shape(self, feature_csv):
        ds = load_features_csv(feature_csv, group_col='subject')
        assert len(ds['features']) == 60
        assert len(ds['labels']) == 60
        assert set(np.unique(ds['groups'])) == {'s1', 's2', 's3'}
        assert set(ds['features'][0].keys()) == set(TIME_FEATURES)

    def test_missing_label_col_raises(self, feature_csv):
        with pytest.raises(ValueError, match="label"):
            load_features_csv(feature_csv, label_col='does_not_exist')

    def test_missing_feature_cols_raises(self, tmp_path):
        df = pd.DataFrame({'MAV': [1, 2], 'label': ['a', 'b']})
        path = tmp_path / "incomplete.csv"
        df.to_csv(path, index=False)
        with pytest.raises(ValueError, match="missing expected feature columns"):
            load_features_csv(path)

    def test_no_group_col_returns_none_groups(self, tmp_path):
        rng = np.random.RandomState(1)
        df = pd.DataFrame({f: rng.rand(10) for f in TIME_FEATURES})
        df['label'] = rng.choice(['a', 'b'], 10)
        path = tmp_path / "no_groups.csv"
        df.to_csv(path, index=False)
        ds = load_features_csv(path, group_col='subject')
        assert ds['groups'] is None

    def test_feeds_classifier_end_to_end(self, feature_csv):
        ds = load_features_csv(feature_csv, group_col='subject')
        clf = EMGClassifier(model_type='random_forest')
        X = clf.prepare_features(ds['features'])
        metrics = clf.fit(X, ds['labels'], groups=ds['groups'])
        assert 'loso_accuracy_mean' in metrics
        assert metrics['loso_n_folds'] == 3


class TestLoadRawEMGAndExtract:
    def test_windows_and_labels_align(self):
        sim = EMGSimulator(fs=2000, n_channels=2)
        raw = sim.generate_contraction(4.0, 'grip', intensity_scale=1.0)
        labels = np.array(['rest', 'grip'])
        label_times = np.array([0.0, 1.0])
        ds = load_raw_emg_and_extract(raw, labels, label_times, fs=2000,
                                       channel=0, threshold=0.01, subject_id='subjA')
        assert len(ds['features']) == len(ds['labels'])
        assert set(np.unique(ds['groups'])) == {'subjA'}
        # first windows (before t=1.0) should be labeled 'rest'
        assert ds['labels'][0] == 'rest'

    def test_no_subject_id_gives_none_groups(self):
        sim = EMGSimulator(fs=2000, n_channels=1)
        raw = sim.generate_contraction(2.0, 'grip', intensity_scale=1.0)
        labels = np.array(['rest'])
        label_times = np.array([0.0])
        ds = load_raw_emg_and_extract(raw, labels, label_times, fs=2000)
        assert ds['groups'] is None


class TestLoadRawLabeledTable:
    """Covers the UCI 'EMG data for gestures' format (Krilova et al. 2018):
    a header row, tab-delimited, time in ms, 8 channel columns, a dense
    per-sample class column — the exact format that first exposed the
    fs-unit bug this test suite locks in.
    """

    @staticmethod
    def _write_uci_style_file(tmp_path, n_rows=300, fs_hz=1000, label=0):
        rng = np.random.RandomState(0)
        dt_ms = 1000.0 / fs_hz
        times = (np.arange(n_rows) * dt_ms).round().astype(int)
        data = {'time': times}
        for ch in range(1, 9):
            data[f'channel{ch}'] = rng.normal(0, 1e-5, n_rows)
        data['class'] = label
        df = pd.DataFrame(data)
        path = tmp_path / "emg_raw.txt"
        df.to_csv(path, sep='\t', index=False)
        return path

    def test_fs_autodetected_correctly_from_ms_time_column(self, tmp_path):
        path = self._write_uci_style_file(tmp_path, fs_hz=1000)
        ds = load_raw_labeled_table(path, label_col='class', time_col='time',
                                     time_unit='ms', window_ms=100, overlap=0.5)
        # Must be close to the true 1000 Hz, NOT 1 Hz (the ms-vs-seconds bug)
        assert 900 <= ds['fs_estimated'] <= 1100

    def test_wrong_time_unit_raises_instead_of_silently_producing_1hz(self, tmp_path):
        path = self._write_uci_style_file(tmp_path, fs_hz=1000)
        with pytest.raises(ValueError, match="outside a plausible"):
            load_raw_labeled_table(path, label_col='class', time_col='time',
                                    time_unit='s')  # wrong unit on purpose

    def test_explicit_fs_bypasses_autodetection(self, tmp_path):
        path = self._write_uci_style_file(tmp_path, fs_hz=1000)
        ds = load_raw_labeled_table(path, label_col='class', time_col='time',
                                     fs=2000, window_ms=100, overlap=0.5)
        assert ds['fs_estimated'] == 2000

    def test_majority_vote_labeling_across_transition(self, tmp_path):
        rng = np.random.RandomState(1)
        n_rows = 200
        times = np.arange(n_rows)  # 1 ms steps -> 1000 Hz
        data = {'time': times}
        for ch in range(1, 9):
            data[f'channel{ch}'] = rng.normal(0, 1e-5, n_rows)
        # first 60% rest (0), rest gesture (2)
        labels = np.array([0] * 120 + [2] * 80)
        data['class'] = labels
        df = pd.DataFrame(data)
        path = tmp_path / "transition.txt"
        df.to_csv(path, sep='\t', index=False)

        ds = load_raw_labeled_table(path, label_col='class', time_col='time',
                                     time_unit='ms', fs=1000, window_ms=50, overlap=0.0)
        assert set(np.unique(ds['labels'])) == {0, 2}

    def test_feeds_classifier_end_to_end(self, tmp_path):
        # two "subjects" with different label mixes so LOSO has something to fold over
        rng = np.random.RandomState(2)
        paths = []
        for i in range(3):
            n_rows = 400
            times = np.arange(n_rows)
            data = {'time': times}
            for ch in range(1, 9):
                data[f'channel{ch}'] = rng.normal(0, 1e-5, n_rows)
            data['class'] = ([0] * 200 + [2] * 200)
            df = pd.DataFrame(data)
            path = tmp_path / f"subj{i}.txt"
            df.to_csv(path, sep='\t', index=False)
            paths.append(path)

        datasets = [
            load_raw_labeled_table(p, label_col='class', time_col='time',
                                    time_unit='ms', fs=1000, window_ms=50,
                                    overlap=0.5, subject_id=f"s{i}")
            for i, p in enumerate(paths)
        ]
        merged = concatenate_subjects(datasets)
        clf = EMGClassifier(model_type='lda')
        X = clf.prepare_features(merged['features'])
        metrics = clf.fit(X, merged['labels'], groups=merged['groups'])
        assert metrics['loso_n_folds'] == 3

    def test_missing_label_col_raises(self, tmp_path):
        path = self._write_uci_style_file(tmp_path)
        with pytest.raises(ValueError, match="not found"):
            load_raw_labeled_table(path, label_col='does_not_exist', time_unit='ms')


class TestLoadNinaproMat:
    @staticmethod
    def _write_fake_ninapro(tmp_path, name="fake.mat", label_field='restimulus',
                             n=4000, n_channels=10, fs=100):
        from scipy.io import savemat
        rng = np.random.RandomState(0)
        emg = rng.normal(0, 1e-4, (n, n_channels))
        labels = np.zeros((n, 1), dtype=int)
        labels[500:1500] = 1
        labels[2500:3500] = 2
        path = tmp_path / name
        savemat(path, {'emg': emg, label_field: labels, 'repetition': np.ones((n, 1))})
        return path

    def test_loads_with_restimulus(self, tmp_path):
        path = self._write_fake_ninapro(tmp_path)
        ds = load_ninapro_mat(path, fs=100, subject_id='s1')
        assert ds['label_field_used'] == 'restimulus'
        assert set(np.unique(ds['labels'])) == {0, 1, 2}
        assert ds['n_channels_used'] == 10
        assert set(np.unique(ds['groups'])) == {'s1'}

    def test_falls_back_to_stimulus(self, tmp_path):
        path = self._write_fake_ninapro(tmp_path, label_field='stimulus')
        ds = load_ninapro_mat(path, fs=100)
        assert ds['label_field_used'] == 'stimulus'

    def test_missing_emg_field_raises(self, tmp_path):
        from scipy.io import savemat
        path = tmp_path / "no_emg.mat"
        savemat(path, {'notemg': np.zeros((10, 5)), 'stimulus': np.zeros((10, 1))})
        with pytest.raises(ValueError, match="'emg' not found"):
            load_ninapro_mat(path, fs=100)

    def test_missing_both_label_fields_raises(self, tmp_path):
        from scipy.io import savemat
        path = tmp_path / "no_labels.mat"
        savemat(path, {'emg': np.zeros((10, 5))})
        with pytest.raises(ValueError, match="Neither"):
            load_ninapro_mat(path, fs=100)

    def test_non_numeric_field_gives_clear_diagnostic_not_cryptic_numpy_error(self, tmp_path):
        """Regression test for the exact bug reported: a MATLAB struct/cell
        field silently became a string like '[0E0,0E0,0E0,0E0,0E0]' and
        blew up deep inside SHAP with no useful context. inspect_mat_fields
        / load_mat_numeric_field must catch this at the loader boundary
        with a message that names the field and explains why."""
        from scipy.io import savemat
        path = tmp_path / "struct_field.mat"
        struct_field = np.array([('x', np.zeros(5))], dtype=[('name', 'U10'), ('data', 'f8', (5,))])
        savemat(path, {'emg': np.zeros((100, 5)), 'restimulus': np.zeros((100, 1)), 'weird': struct_field})

        info = inspect_mat_fields(path)
        assert info['emg']['is_numeric'] is True
        assert info['weird']['is_numeric'] is False

        with pytest.raises(ValueError, match="isn't a plain numeric matrix"):
            load_mat_numeric_field(path, 'weird')

        # the presence of a non-numeric field elsewhere in the file must
        # NOT break loading the fields that ARE numeric
        ds = load_ninapro_mat(path, fs=100)
        assert len(ds['features']) > 0

    def test_inspect_mat_fields_reports_all_top_level_keys(self, tmp_path):
        from scipy.io import savemat
        path = tmp_path / "multi_field.mat"
        savemat(path, {'emg': np.zeros((50, 8)), 'restimulus': np.zeros((50, 1)),
                        'repetition': np.ones((50, 1))})
        info = inspect_mat_fields(path)
        assert set(info.keys()) == {'emg', 'restimulus', 'repetition'}
        assert info['emg']['shape'] == (50, 8)

    def test_feeds_classifier_with_loso_across_subjects(self, tmp_path):
        paths = [self._write_fake_ninapro(tmp_path, name=f"s{i}.mat") for i in range(3)]
        datasets = [load_ninapro_mat(p, fs=100, subject_id=f"s{i}")
                    for i, p in enumerate(paths)]
        merged = concatenate_subjects(datasets)
        clf = EMGClassifier(model_type='lda')
        X = clf.prepare_features(merged['features'])
        metrics = clf.fit(X, merged['labels'], groups=merged['groups'])
        assert metrics['loso_n_folds'] == 3


class TestConcatenateSubjects:
    def test_merges_and_assigns_groups(self):
        sim = EMGSimulator(fs=2000, n_channels=1)
        datasets = []
        for sid in ['A', 'B']:
            raw = sim.generate_contraction(2.0, 'grip', intensity_scale=1.0)
            labels, label_times = np.array(['rest']), np.array([0.0])
            datasets.append(load_raw_emg_and_extract(
                raw, labels, label_times, fs=2000, subject_id=sid))
        merged = concatenate_subjects(datasets)
        assert len(merged['features']) == len(datasets[0]['features']) + len(datasets[1]['features'])
        assert set(np.unique(merged['groups'])) == {'A', 'B'}

    def test_auto_assigns_groups_when_missing(self):
        ds_no_group = {'features': [{'MAV': 0.1}] * 5, 'labels': np.array(['a'] * 5), 'groups': None}
        merged = concatenate_subjects([ds_no_group])
        assert set(np.unique(merged['groups'])) == {'subject_0'}
