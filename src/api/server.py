"""
server.py - FastAPI server for MyoControl Suite v0.3
Endpoints:
    GET  /health
    POST /v1/analyze         — analyze EMG file
    POST /v1/classify        — classify gestures from EMG file
    POST /v1/extract-features — extract features only
    GET  /v1/models          — list available models
    POST /v1/train           — train a new model (demo)

Run:
    uvicorn src.api.server:app --reload --port 8000
"""
import io
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Optional, List

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.core.config import EMGConfig
from src.core.engine import EMGEngine
from src.core.classifier import EMGClassifier, generate_synthetic_emg_dataset
from src.core.features import extract_features_stream
from src.core.simulator import EMGSimulator

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="MyoControl Suite API",
    version="0.3.0",
    description="EMG analysis + gesture classification API. IEEE/ISEK compliant.",
)

# CORS for browser-based clients (Streamlit, Next.js, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Models storage (in production: replace with S3/database)
# ============================================================
MODELS_DIR = Path(tempfile.gettempdir()) / "myocontrol_models"
MODELS_DIR.mkdir(exist_ok=True)


# ============================================================
# Schemas
# ============================================================
class AnalyzeResponse(BaseModel):
    metadata: dict
    signal_quality: dict
    clinical_interpretation: dict
    summary_statistics: dict


class ClassifyRequest(BaseModel):
    features: List[dict]
    model_id: str = "default"


# ============================================================
# Helpers
# ============================================================
def load_signal_from_bytes(contents: bytes, filename: str) -> np.ndarray:
    """Load EMG signal from CSV, TXT, or NPY bytes."""
    name = filename.lower()
    if name.endswith('.csv'):
        return np.loadtxt(io.BytesIO(contents), delimiter=',')
    elif name.endswith('.txt'):
        return np.loadtxt(io.BytesIO(contents))
    elif name.endswith('.npy'):
        return np.load(io.BytesIO(contents))
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {filename}. Use CSV, TXT, or NPY.")


def get_or_train_default_model() -> EMGClassifier:
    """Get the default classifier, training it if needed (synthetic data)."""
    model_path = MODELS_DIR / "default.pkl"
    if model_path.exists():
        return EMGClassifier.load(model_path)

    # Train on synthetic data for demo purposes
    logger.info("Training default classifier on synthetic data...")
    dataset = generate_synthetic_emg_dataset(
        n_classes=5, n_samples_per_class=50, n_channels=4)
    clf = EMGClassifier(model_type='xgboost')
    X = clf.prepare_features(dataset['features'])
    clf.fit(X, dataset['labels'])
    clf.save(model_path)
    return clf


# ============================================================
# Endpoints
# ============================================================
@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok", "version": "0.3.0"}


@app.post("/v1/analyze")
async def analyze(
    file: UploadFile = File(...),
    sampling_rate: int = Form(2000),
    cutoff_low: float = Form(20.0),
    cutoff_high: float = Form(450.0),
    filter_order: int = Form(4),
    notch_freq: float = Form(50.0),
    window_ms: int = Form(100),
    overlap: float = Form(0.5),
    filter_type: str = Form("butterworth"),
    noise_method: str = Form("percentile"),
    psd_method: str = Form("welch"),
    compute_freq: bool = Form(False),
    selected_channel: int = Form(0),
):
    """
    Upload an EMG file and receive full analysis (filtering + features + stats).
    """
    contents = await file.read()
    try:
        data = load_signal_from_bytes(contents, file.filename)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load file: {e}")

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    config = EMGConfig(
        sampling_rate=sampling_rate,
        cutoff_low=cutoff_low,
        cutoff_high=cutoff_high,
        filter_order=filter_order,
        notch_freq=notch_freq,
        window_size=int(window_ms * sampling_rate / 1000),
        overlap=overlap,
        filter_type=filter_type,
        noise_estimation_method=noise_method,
        psd_method=psd_method,
        compute_frequency_features=compute_freq,
    )

    engine = EMGEngine(config)
    try:
        result = engine.process(
            data, selected_channel=selected_channel,
            measure_time=True, compute_freq_features=compute_freq)
        return JSONResponse(result)
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/extract-features")
async def extract_features(
    file: UploadFile = File(...),
    sampling_rate: int = Form(2000),
    window_ms: int = Form(100),
    overlap: float = Form(0.5),
    include_freq: bool = Form(False),
):
    """Extract features from EMG file (no filtering, no stats)."""
    contents = await file.read()
    try:
        data = load_signal_from_bytes(contents, file.filename)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load file: {e}")

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    # Apply standard filtering for fair feature extraction
    config = EMGConfig(sampling_rate=sampling_rate,
                       window_size=int(window_ms * sampling_rate / 1000),
                       overlap=overlap)
    engine = EMGEngine(config)
    filtered = engine.preprocess(data)
    feats = extract_features_stream(
        filtered, sampling_rate,
        window_size=config.window_size,
        overlap=overlap,
        include_freq=include_freq,
    )
    return JSONResponse({
        'n_channels': data.shape[1],
        'n_windows': len(feats[0]) if feats else 0,
        'features': feats,
    })


@app.post("/v1/classify")
async def classify(
    file: UploadFile = File(...),
    sampling_rate: int = Form(2000),
    window_ms: int = Form(100),
    overlap: float = Form(0.5),
    model_id: str = Form("default"),
):
    """
    Classify gestures from an EMG file.

    Returns per-window predictions + confidence + summary.
    """
    contents = await file.read()
    try:
        data = load_signal_from_bytes(contents, file.filename)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load file: {e}")

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    # Load model
    try:
        clf = get_or_train_default_model() if model_id == "default" \
            else EMGClassifier.load(MODELS_DIR / f"{model_id}.pkl")
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' not found: {e}")

    # Process signal
    config = EMGConfig(sampling_rate=sampling_rate,
                       window_size=int(window_ms * sampling_rate / 1000),
                       overlap=overlap)
    engine = EMGEngine(config)
    filtered = engine.preprocess(data)
    feats = extract_features_stream(
        filtered, sampling_rate,
        window_size=config.window_size,
        overlap=overlap, include_freq=False)

    # Use channel 0 features for classification
    if not feats or not feats[0]:
        raise HTTPException(status_code=400, detail="No windows extracted.")

    results = clf.predict_with_confidence(feats[0])

    # Aggregate predictions
    predictions = [r.predicted_class for r in results]
    unique, counts = np.unique(predictions, return_counts=True)
    summary = {str(u): int(c) for u, c in zip(unique, counts)}
    dominant_gesture = str(unique[np.argmax(counts)]) if len(unique) > 0 else None

    return JSONResponse({
        'n_windows': len(results),
        'predictions': predictions,
        'dominant_gesture': dominant_gesture,
        'gesture_distribution': summary,
        'mean_confidence': float(np.mean([r.confidence for r in results])),
        'per_window_results': [r.to_dict() for r in results],
        'model_id': model_id,
        'classes': clf.classes_,
    })


@app.get("/v1/models")
async def list_models():
    """List available trained models."""
    models = []
    for p in MODELS_DIR.glob("*.pkl"):
        try:
            clf = EMGClassifier.load(p)
            models.append({
                'id': p.stem,
                'model_type': clf.model_type,
                'classes': clf.classes_,
                'n_features': len(clf.feature_names),
                'training_metadata': clf.training_metadata,
            })
        except Exception as e:
            logger.warning(f"Failed to load {p}: {e}")
    return {'models': models}


@app.post("/v1/train")
async def train_demo_model(
    n_classes: int = Form(5),
    n_samples_per_class: int = Form(50),
    model_type: str = Form("xgboost"),
    model_id: str = Form(None),
):
    """
    Train a demo classifier on synthetic data.
    (In production: replace with real dataset upload endpoint.)
    """
    try:
        dataset = generate_synthetic_emg_dataset(
            n_classes=n_classes,
            n_samples_per_class=n_samples_per_class)
        clf = EMGClassifier(model_type=model_type)
        X = clf.prepare_features(dataset['features'])
        metrics = clf.fit(X, dataset['labels'], groups=dataset['groups'])

        mid = model_id or f"demo_{model_type}_{n_classes}c"
        clf.save(MODELS_DIR / f"{mid}.pkl")

        return JSONResponse({
            'model_id': mid,
            'training_metrics': metrics,
            'classes': dataset['classes'],
        })
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/explain")
async def explain(
    file: UploadFile = File(...),
    sampling_rate: int = Form(2000),
    window_ms: int = Form(100),
    overlap: float = Form(0.5),
    model_id: str = Form("default"),
):
    """
    Compute SHAP-based feature importance for an uploaded EMG file.
    """
    try:
        import shap  # noqa
    except ImportError:
        raise HTTPException(
            status_code=400,
            detail="shap package not installed on server.")

    contents = await file.read()
    data = load_signal_from_bytes(contents, file.filename)
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    clf = get_or_train_default_model() if model_id == "default" \
        else EMGClassifier.load(MODELS_DIR / f"{model_id}.pkl")

    config = EMGConfig(sampling_rate=sampling_rate,
                       window_size=int(window_ms * sampling_rate / 1000),
                       overlap=overlap)
    engine = EMGEngine(config)
    filtered = engine.preprocess(data)
    feats = extract_features_stream(
        filtered, sampling_rate,
        window_size=config.window_size,
        overlap=overlap, include_freq=False)

    X = clf.prepare_features(feats[0])
    explanation = clf.explain(X)
    return JSONResponse(explanation or {'error': 'explanation failed'})


# ============================================================
# Main entry
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
