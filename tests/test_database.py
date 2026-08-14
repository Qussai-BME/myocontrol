"""
test_database.py - Tests for session persistence (src/core/database.py)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from src.core.database import save_session, list_sessions, load_session, delete_session
from src.core.config import EMGConfig
from src.core.engine import EMGEngine
from src.core.simulator import EMGSimulator


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test_sessions.db"


@pytest.fixture
def sample_result():
    config = EMGConfig(sampling_rate=2000, window_size=200, overlap=0.5)
    engine = EMGEngine(config)
    sim = EMGSimulator(fs=2000, n_channels=4)
    raw = sim.generate_contraction(3.0, sim.list_gestures()[1], intensity_scale=1.0)
    result = engine.process(raw, selected_channel=0)
    return result, config


class TestSessionPersistence:
    def test_save_and_list(self, db_path, sample_result):
        result, config = sample_result
        sid = save_session(result, config, acquisition_info={'source': 'Simulation', 'gesture': 'fist', 'intensity': 1.0}, db_path=db_path)
        assert isinstance(sid, int) and sid > 0

        sessions = list_sessions(db_path=db_path)
        assert len(sessions) == 1
        assert sessions[0]['id'] == sid
        assert sessions[0]['gesture'] == 'fist'
        assert sessions[0]['snr_quality'] == result['signal_quality']['snr_quality']

    def test_load_roundtrip(self, db_path, sample_result):
        result, config = sample_result
        sid = save_session(result, config, db_path=db_path)
        loaded = load_session(sid, db_path=db_path)

        assert loaded is not None
        assert loaded['result']['metadata']['n_samples'] == result['metadata']['n_samples']
        assert loaded['config']['sampling_rate'] == config.sampling_rate

    def test_load_missing_returns_none(self, db_path, sample_result):
        result, config = sample_result
        save_session(result, config, db_path=db_path)
        assert load_session(9999, db_path=db_path) is None

    def test_delete(self, db_path, sample_result):
        result, config = sample_result
        sid = save_session(result, config, db_path=db_path)
        assert delete_session(sid, db_path=db_path) is True
        assert list_sessions(db_path=db_path) == []
        assert delete_session(sid, db_path=db_path) is False

    def test_list_ordering_and_limit(self, db_path, sample_result):
        result, config = sample_result
        ids = [save_session(result, config, db_path=db_path) for _ in range(5)]
        sessions = list_sessions(limit=3, db_path=db_path)
        assert len(sessions) == 3
        # newest first
        assert sessions[0]['id'] == ids[-1]

    def test_multiple_sessions_isolated_dbs(self, tmp_path, sample_result):
        result, config = sample_result
        db_a = tmp_path / "a.db"
        db_b = tmp_path / "b.db"
        save_session(result, config, db_path=db_a)
        assert list_sessions(db_path=db_a) != []
        assert list_sessions(db_path=db_b) == []
