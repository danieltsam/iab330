import os
from datetime import datetime
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "gestureio")

_client: Optional[MongoClient] = None
_db = None

def get_db():
    """
    Lazily create and cache the DB connection.
    Raises if the URI is invalid or server is unreachable.
    """
    global _client, _db
    if _db is not None:
        return _db
    _client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    _client.admin.command("ping")  # sanity check connection
    _db = _client[MONGO_DB]
    return _db

def save_training_metrics(model_name: str, metrics: Dict[str, Any], params: Dict[str, Any]):
    """
    Store one record per training run.
    Example metrics: {"accuracy": 0.94, "loss": 0.21, "val_accuracy": 0.91, "val_loss": 0.28}
    Example params:  {"epochs": 20, "batch_size": 64, "window_size": 128, "features": [...]}
    """
    db = get_db()
    doc = {
        "_kind": "training_run",
        "model_name": model_name,
        "metrics": metrics,
        "params": params,
        "created_at": datetime.utcnow(),
    }
    db.training_runs.insert_one(doc)
    return doc

def save_prediction(session_id: str, gesture: str, score: float, meta: Dict[str, Any]):
    """
    Save a single live inference result.
    meta can include device_id, sampling_rate, window_length, etc.
    """
    db = get_db()
    doc = {
        "_kind": "prediction",
        "session_id": session_id,
        "gesture": gesture,
        "score": score,
        "meta": meta,
        "created_at": datetime.utcnow(),
    }
    db.predictions.insert_one(doc)
    return doc


def get_recent_predictions(limit: int = 20):
    db = get_db()
    return list(db.predictions.find().sort("created_at", -1).limit(limit))