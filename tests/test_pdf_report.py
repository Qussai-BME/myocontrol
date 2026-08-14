"""
test_pdf_report.py - Tests for src/core/pdf_report.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import io
import pytest

from src.core.pdf_report import generate_pdf_report
from src.core.config import EMGConfig
from src.core.engine import EMGEngine
from src.core.simulator import EMGSimulator
from src.core.classifier import EMGClassifier, generate_synthetic_emg_dataset


@pytest.fixture
def engine_result():
    config = EMGConfig(sampling_rate=2000, window_size=200, overlap=0.5)
    engine = EMGEngine(config)
    sim = EMGSimulator(fs=2000, n_channels=2)
    raw = sim.generate_contraction(3.0, 'grip', intensity_scale=1.0)
    filtered = engine.preprocess(raw)
    result = engine.process(raw, selected_channel=0)
    return result, raw, filtered


class TestGeneratePdfReport:
    def test_minimal_report_is_valid_pdf(self, engine_result):
        result, raw, filtered = engine_result
        pdf_bytes = generate_pdf_report(result)
        assert pdf_bytes[:4] == b"%PDF"
        assert len(pdf_bytes) > 500

    def test_report_with_signal_and_classifier(self, engine_result):
        result, raw, filtered = engine_result
        dataset = generate_synthetic_emg_dataset(n_classes=3, n_samples_per_class=30)
        clf = EMGClassifier(model_type='lda')
        X = clf.prepare_features(dataset['features'])
        metrics = clf.fit(X, dataset['labels'], groups=dataset['groups'])

        pdf_bytes = generate_pdf_report(
            result, raw_signal=raw, filtered_signal=filtered,
            acquisition_info={'source': 'Simulation', 'gesture': 'grip', 'intensity': 1.0},
            classifier_metrics=metrics, author="Test Author",
        )
        assert pdf_bytes[:4] == b"%PDF"

        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(pdf_bytes))
        assert len(reader.pages) >= 1
        text = "".join(p.extract_text() for p in reader.pages)
        assert "Test Author" in text
        assert "LOSO" in text
        assert "Signal Quality" in text or "Signal quality" in text.lower() or "quality" in text.lower()

    def test_report_without_optional_sections_still_valid(self, engine_result):
        result, raw, filtered = engine_result
        # No raw_signal, no classifier_metrics, no acquisition_info
        pdf_bytes = generate_pdf_report(result)
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(pdf_bytes))
        assert len(reader.pages) >= 1
