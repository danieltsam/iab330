"""
Live 1D-CNN inference for IMU streams.

Loads exports created by model_training.py:
  exports/
    metadata.json
    cnn_1d/
      model.keras
      preprocessor.joblib

Usage:
    infer = LiveCNNInference(export_dir="exports/2025-10-16")
    out = infer.update([ax, ay, az, gx, gy, gz])  # returns (label, probs_dict) or None
"""

from __future__ import annotations
from pathlib import Path
from collections import deque
import json
from typing import Dict, List, Optional, Tuple
from db import save_prediction
import uuid

import numpy as np
import joblib

try:
    from tensorflow import keras
except Exception:
    keras = None


class LiveCNNInference:
    def __init__(self, export_dir: str | Path, *, log_to_db: bool = True, min_confidence: float = 0.0):
        export_dir = Path(export_dir)
        meta = json.loads((export_dir / "metadata.json").read_text())
        self.window_size: int = int(meta["window_size"])
        self.step_size: int = int(meta["step_size"])
        self.num_channels: int = int(meta["num_channels"])
        self.labels: List[str] = list(meta["labels"])
        self.channel_order: List[str] = list(meta.get("channel_order", []))

        model_dir = export_dir / "cnn_1d"
        if not model_dir.exists():
            raise FileNotFoundError(f"cnn_1d export not found in {export_dir}")

        if keras is None:
            raise RuntimeError("TensorFlow/Keras not available. Install tensorflow to use cnn_1d inference.")

   
        self.model = keras.models.load_model(model_dir / "model.keras")
        self.preprocessor = joblib.load(model_dir / "preprocessor.joblib")

   
        self.buf: deque[np.ndarray] = deque(maxlen=self.window_size)
        self._since_last_emit = 0

      
        self.session_id = str(uuid.uuid4())

       
        self.log_to_db = log_to_db
        self.min_confidence = float(min_confidence)

    def reset(self) -> None:
        self.buf.clear()
        self._since_last_emit = 0
        

    def update(self, sample: List[float] | np.ndarray) -> Optional[Tuple[str, Dict[str, float]]]:
        """
        Push one IMU sample [ax, ay, az, gx, gy, gz].
        Returns (label, probs_dict) every step_size samples once window is full, else None.
        """
        arr = np.asarray(sample, dtype=float).reshape(-1)
        if arr.size != self.num_channels:
            raise ValueError(f"Expected {self.num_channels} channels, got {arr.size}")

        self.buf.append(arr)
        self._since_last_emit += 1

        if len(self.buf) < self.window_size or self._since_last_emit < self.step_size:
            return None

        self._since_last_emit = 0

        window = np.stack(self.buf, axis=0)  
        T, C = window.shape
        xf = window.reshape(1, T * C)
        xf = self.preprocessor.transform(xf)     
        xseq = xf.reshape(1, T, C)

        probs = self.model.predict(xseq, verbose=0)[0]
        idx = int(np.argmax(probs))
        label = self.labels[idx]
        probs_dict = {self.labels[i]: float(p) for i, p in enumerate(probs)}

        score = float(np.max(probs))

       
        if self.log_to_db and score >= self.min_confidence:
            meta = {
                "window_size": self.window_size,
                "num_channels": self.num_channels,
                "step_size": self.step_size,
                "note": "live inference update()",
            }
          
            try:
                save_prediction(session_id=self.session_id, gesture=label, score=score, meta=meta)
            except Exception as e:
               
                print(f"[DB] save_prediction failed: {e}")

        return label, probs_dict
