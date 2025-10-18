"""Train gesture recognition models from CSV recordings and export for live inference.

- Each CSV column is one full recording; row 0 is a human-readable title.
- Rows 1..end contain comma-delimited tokens like: timestamp, ax, ay, az, gx, gy, gz
- Timestamps are ignored; the last 6 numeric tokens per row are used (ax..gz).

Models:
- Decision Tree, Random Forest, RBF-SVM (with probabilities), KNN, optional 1D CNN.
- Classical models use flattened windows; CNN uses sequential windows.
- Group-aware split ensures entire recordings do not leak between train and test.

Exports:
- metadata.json (window/step/channels/labels)
- One directory per model:
    - classical: pipeline.joblib (preprocessor + classifier)
    - cnn_1d:    model.keras  + preprocessor.joblib (scaler pipeline)
"""

import argparse, dataclasses, json, math, pathlib, warnings, joblib
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from db import save_training_metrics


import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    from tensorflow import keras
    from tensorflow.keras import layers

    _HAS_TENSORFLOW = True
except Exception:
    keras = None
    layers = None
    _HAS_TENSORFLOW = False

# Configuration / constants
CHANNEL_ORDER = ["ax", "ay", "az", "gx", "gy", "gz"]
NUM_CHANNELS = len(CHANNEL_ORDER)
RANDOM_STATE = 42

@dataclasses.dataclass
class TrainingDataset:
    flat_windows: np.ndarray
    sequence_windows: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    label_encoder: LabelEncoder


@dataclasses.dataclass
class ModelTrainingResult:
    model: Any
    preprocessor: Optional[Pipeline]
    label_encoder: LabelEncoder
    accuracy: float

def _parse_numeric_tokens(cell_value: str) -> List[float]:
    values: List[float] = []
    for token in str(cell_value).split(","):
        token = token.strip().strip('"')
        if not token:
            continue
        try:
            values.append(float(token))
        except ValueError:
            continue
    return values

"""Return list of (group_key, samples) from a CSV where each column is one recording.

    Row 0 = title (used as the group key; taken as-is).
    Rows 1..N = samples (comma-delimited). We keep the last 6 numeric columns (ax..gz).
    """
def _read_recordings_with_titles(csv_path: pathlib.Path) -> List[Tuple[str, np.ndarray]]:
    df = pd.read_csv(csv_path, header=None, engine="python", dtype=str)
    df = df.replace({"": np.nan, " ": np.nan}).dropna(axis=0, how="all").dropna(axis=1, how="all")
    if df.empty:
        raise ValueError(f"CSV file '{csv_path}' has no data")

    outputs: List[Tuple[str, np.ndarray]] = []
    for col in df.columns:
        series = df[col].dropna()
        if series.empty:
            continue

        title = str(series.iloc[0]).strip()
        group_key = f"{csv_path.stem}::{title}"

        data_series = series.iloc[1:]
        parsed_rows = [_parse_numeric_tokens(v) for v in data_series]
        parsed_rows = [r for r in parsed_rows if r]
        if not parsed_rows:
            continue

        width = max(len(r) for r in parsed_rows)
        M = np.full((len(parsed_rows), width), np.nan, dtype=float)
        for i, r in enumerate(parsed_rows):
            M[i, : len(r)] = r

        if M.shape[1] > NUM_CHANNELS:
            M = M[:, -NUM_CHANNELS:]

        M = M[np.all(np.isfinite(M), axis=1)]
        if M.shape[0] == 0 or M.shape[1] < 3:
            continue

        outputs.append((group_key, M))

    if not outputs:
        raise ValueError(f"No usable recordings in '{csv_path}'")
    return outputs

# Create sliding windows (T, C) -> (num_windows, T, C). Pads tail if needed.
def _create_windows(samples: np.ndarray, window_size: int, step_size: Optional[int]) -> np.ndarray:
    if window_size <= 0:
        raise ValueError("window_size must be positive")

    step = step_size or window_size
    if step <= 0:
        raise ValueError("step_size must be positive")

    if samples.shape[0] < window_size:
        pad = np.tile(samples[-1], (window_size - samples.shape[0], 1))
        samples = np.vstack([samples, pad])

    windows = [
        samples[start : start + window_size]
        for start in range(0, samples.shape[0] - window_size + 1, step)
    ]
    if not windows:
        windows.append(samples[-window_size:])
    return np.stack(windows)

# Raises if any group id appears in both train and test, or test has one class.
def _assert_no_leak(groups_all, train_idx, test_idx, y) -> None:
    g_all = np.asarray(groups_all, dtype=object)
    g_train = set(g_all[train_idx])
    g_test = set(g_all[test_idx])
    inter = g_train & g_test
    if inter:
        raise AssertionError(f"Group leakage detected: {list(inter)[:5]}")
    if np.unique(y[test_idx]).size < 2:
        raise AssertionError("Test fold has a single class; use a group-stratified split.")

# Train/test split; prefers group-aware holdout. Includes leak tripwire.
def _train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    *,
    groups: Optional[np.ndarray] = None,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X)
    y = np.asarray(y)

    if y.shape[0] < 2:
        warnings.warn("Not enough samples for a train/test split; using the full dataset", RuntimeWarning)
        return X, X, y, y

    if groups is not None:
        groups = np.asarray(groups, dtype=object)
        if len(groups) != len(y):
            raise ValueError("groups must have the same length as y")

        unique_groups = np.unique(groups)
        if unique_groups.size < 2:
            warnings.warn("Not enough unique recordings for a group-aware split; using the full dataset",
                          RuntimeWarning)
            return X, X, y, y

        n_groups = unique_groups.size
        n_test_groups = int(math.ceil(test_size * n_groups)) if test_size < 1.0 else int(test_size)
        n_test_groups = min(max(1, n_test_groups), n_groups - 1)
        adjusted_test_size = n_test_groups / n_groups

        splitter = GroupShuffleSplit(n_splits=1, test_size=adjusted_test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(X, y, groups))
        _assert_no_leak(groups, train_idx, test_idx, y)
        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

    counts = np.bincount(y)
    stratify = y if np.all(counts >= 2) else None
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)

def load_datasets(
    file_label_pairs: Sequence[Tuple[str, str]],
    window_size: int = 64,
    step_size: Optional[int] = None,
) -> TrainingDataset:
    flat_windows: List[np.ndarray] = []
    seq_windows: List[np.ndarray] = []
    labels: List[str] = []
    groups: List[str] = []

    for file_path, label in file_label_pairs:
        path = pathlib.Path(file_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"CSV file '{file_path}' was not found")

        for group_key, recording in _read_recordings_with_titles(path):
            windows = _create_windows(recording, window_size, step_size)
            seq_windows.append(windows)
            flat_windows.append(windows.reshape(windows.shape[0], -1))
            labels.extend([label] * windows.shape[0])
            groups.extend([group_key] * windows.shape[0])

    if not flat_windows:
        raise ValueError("No usable recordings were loaded")

    X_flat = np.concatenate(flat_windows, axis=0)
    X_seq = np.concatenate(seq_windows, axis=0)

    le = LabelEncoder()
    y = le.fit_transform(labels)

    return TrainingDataset(
        flat_windows=X_flat,
        sequence_windows=X_seq,
        labels=y,
        groups=np.asarray(groups, dtype=object),
        label_encoder=le,
    )

# Fit a scikit-learn estimator; optionally add StandardScaler. Returns accuracy.
def _fit_classical_model(
    estimator: BaseEstimator,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    *,
    scale: bool = False,
) -> Tuple[BaseEstimator, Optional[Pipeline], float]:
    pre: Optional[Pipeline] = None
    if scale:
        pre = Pipeline([("scaler", StandardScaler())])
        X_train_proc = pre.fit_transform(X_train)
        X_test_proc = pre.transform(X_test)
    else:
        X_train_proc, X_test_proc = X_train, X_test

    estimator.fit(X_train_proc, y_train)
    y_pred = estimator.predict(X_test_proc)
    return estimator, pre, float(accuracy_score(y_test, y_pred))


def train_decision_tree(data: TrainingDataset) -> ModelTrainingResult:
    Xtr, Xte, ytr, yte = _train_test_split(data.flat_windows, data.labels, groups=data.groups)
    model, pre, acc = _fit_classical_model(
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        Xtr, Xte, ytr, yte,
    )
    return ModelTrainingResult(model, pre, data.label_encoder, acc)


def train_random_forest(data: TrainingDataset) -> ModelTrainingResult:
    Xtr, Xte, ytr, yte = _train_test_split(data.flat_windows, data.labels, groups=data.groups)
    model, pre, acc = _fit_classical_model(
        RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        Xtr, Xte, ytr, yte,
    )
    return ModelTrainingResult(model, pre, data.label_encoder, acc)


def train_rbf_svm(data: TrainingDataset) -> ModelTrainingResult:
    Xtr, Xte, ytr, yte = _train_test_split(data.flat_windows, data.labels, groups=data.groups)
    model, pre, acc = _fit_classical_model(
        SVC(kernel="rbf", gamma="scale", probability=True),
        Xtr, Xte, ytr, yte,
        scale=True,
    )
    return ModelTrainingResult(model, pre, data.label_encoder, acc)


def train_knn(data: TrainingDataset) -> ModelTrainingResult:
    Xtr, Xte, ytr, yte = _train_test_split(data.flat_windows, data.labels, groups=data.groups)
    model, pre, acc = _fit_classical_model(
        KNeighborsClassifier(n_neighbors=5),
        Xtr, Xte, ytr, yte,
        scale=True,
    )
    return ModelTrainingResult(model, pre, data.label_encoder, acc)


def train_cnn(
    data: TrainingDataset,
    *,
    epochs: int = 120,
    batch_size: int = 32,
    validation_split: float = 0.1,
) -> ModelTrainingResult:
    if not _HAS_TENSORFLOW:
        raise RuntimeError("TensorFlow is required for the CNN model")

    Xtr, Xte, ytr, yte = _train_test_split(data.sequence_windows, data.labels, groups=data.groups)

    T, C = Xtr.shape[1], Xtr.shape[2]
    Xtr_flat = Xtr.reshape(Xtr.shape[0], -1)
    Xte_flat = Xte.reshape(Xte.shape[0], -1)
    scaler = StandardScaler().fit(Xtr_flat)
    Xtr = scaler.transform(Xtr_flat).reshape(-1, T, C)
    Xte = scaler.transform(Xte_flat).reshape(-1, T, C)

    model = keras.Sequential(
        [
            layers.Input(shape=(T, C)),
            layers.Conv1D(64, 5, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Conv1D(128, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Conv1D(128, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling1D(),
            layers.Dropout(0.35),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.35),
            layers.Dense(len(data.label_encoder.classes_), activation="softmax"),
        ]
    )

    opt = keras.optimizers.Adam(learning_rate=3e-4)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5),
    ]
    val_split = validation_split if Xtr.shape[0] >= 200 else 0.0

    model.fit(Xtr, ytr, epochs=epochs, batch_size=batch_size, validation_split=val_split, callbacks=callbacks, verbose=0)
    _, test_acc = model.evaluate(Xte, yte, verbose=0)

    train_acc = float(history.history['accuracy'][-1])
    train_loss = float(history.history['loss'][-1])


    val_acc = float(history.history['val_accuracy'][-1]) if 'val_accuracy' in history.history else None
    val_loss = float(history.history['val_loss'][-1]) if 'val_loss' in history.history else None

    metrics = {
        "accuracy": train_acc,
        "loss": train_loss,
        "val_accuracy": val_acc,
        "val_loss": val_loss,
    }

    params = {
        "epochs": EPOCHS if 'EPOCHS' in globals() else None,
        "batch_size": BATCH_SIZE if 'BATCH_SIZE' in globals() else None,
        "window_size": window_size if 'window_size' in globals() else None,
        "features": ['ax', 'ay', 'az', 'gx', 'gy', 'gz'],
    }


    save_training_metrics(model_name="gestureio_model_v1", metrics=metrics, params=params)




    pre = Pipeline([
        ("scaler", ColumnTransformer([("standard", StandardScaler(), list(range(T * C)))], remainder="drop"))
    ])
    pre.named_steps["scaler"].fit(Xtr_flat)

    return ModelTrainingResult(model, pre, data.label_encoder, float(test_acc))


MODEL_REGISTRY = {
    "decision_tree": train_decision_tree,
    "random_forest": train_random_forest,
    "rbf_svm": train_rbf_svm,
    "knn": train_knn,
    "cnn_1d": train_cnn,
}

def export_for_inference(
    results: Dict[str, ModelTrainingResult],
    export_dir: pathlib.Path,
    *,
    window_size: int,
    step_size: Optional[int],
    label_encoder: LabelEncoder,
    num_channels: int,
) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "window_size": int(window_size),
        "step_size": int(step_size or window_size),
        "num_channels": int(num_channels),
        "labels": list(label_encoder.classes_),
        "channel_order": CHANNEL_ORDER,
        "version": 1,
    }
    (export_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    for name, r in results.items():
        mdir = export_dir / name
        mdir.mkdir(exist_ok=True)
        if hasattr(r.model, "save") and callable(getattr(r.model, "save")):
            r.model.save(mdir / "model.keras")
            joblib.dump(r.preprocessor, mdir / "preprocessor.joblib")
        else:
            steps = []
            if r.preprocessor is not None:
                steps.append(("pre", r.preprocessor))
            steps.append(("clf", r.model))
            pipe = Pipeline(steps)
            joblib.dump(pipe, mdir / "pipeline.joblib")

# Train requested models and return their fitted instances.
def train_models(
    file_label_pairs: Sequence[Tuple[str, str]],
    *,
    window_size: int = 64,
    step_size: Optional[int] = None,
    models: Optional[Iterable[str]] = None,
    cnn_epochs: int = 25,
    cnn_batch_size: int = 32,
) -> Dict[str, ModelTrainingResult]:
    requested = list(models) if models is not None else list(MODEL_REGISTRY.keys())
    if not _HAS_TENSORFLOW and "cnn_1d" in requested:
        warnings.warn("TensorFlow not available; skipping cnn_1d model", RuntimeWarning)
        requested = [m for m in requested if m != "cnn_1d"]
    if not requested:
        raise RuntimeError("No models available to train with the current configuration")

    data = load_datasets(file_label_pairs, window_size=window_size, step_size=step_size)

    results: Dict[str, ModelTrainingResult] = {}
    for name in requested:
        trainer = MODEL_REGISTRY.get(name)
        if trainer is None:
            raise ValueError(f"Unknown model '{name}'")
        if name == "cnn_1d":
            results[name] = trainer(data, epochs=cnn_epochs, batch_size=cnn_batch_size)
        else:
            results[name] = trainer(data)
    return results


def _parse_cli_inputs(inputs: Iterable[str]) -> List[Tuple[str, str]]:
    parsed: List[Tuple[str, str]] = []
    for entry in inputs:
        if ":" not in entry:
            raise ValueError(f"Invalid input '{entry}'. Expected 'path/to/file.csv:LABEL'")
        file_path, label = entry.split(":", 1)
        parsed.append((file_path, label))
    return parsed


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train gesture recognition models")
    parser.add_argument("inputs", nargs="+", help="CSV training data as 'path/to/file.csv:LABEL'")
    parser.add_argument("--window-size", type=int, default=64, help="Samples per window (default: 64)")
    parser.add_argument("--step-size", type=int, default=None, help="Stride between windows (default: window size)")
    parser.add_argument("--models", nargs="*", choices=list(MODEL_REGISTRY.keys()), help="Subset of models to train")
    parser.add_argument("--cnn-epochs", type=int, default=25, help="Epochs for the 1D CNN")
    parser.add_argument("--cnn-batch-size", type=int, default=32, help="Batch size for the 1D CNN")
    parser.add_argument("--export-dir", type=pathlib.Path, help="Directory to save trained models + metadata for live inference")

    args = parser.parse_args(argv)

    try:
        file_label_pairs = _parse_cli_inputs(args.inputs)
        results = train_models(
            file_label_pairs,
            window_size=args.window_size,
            step_size=args.step_size,
            models=args.models,
            cnn_epochs=args.cnn_epochs,
            cnn_batch_size=args.cnn_batch_size,
        )
    except Exception as exc:
        parser.error(str(exc))

    print("Training complete. Accuracy scores:")
    for name, result in results.items():
        print(f"  {name}: {result.accuracy:.3f}")

    if args.export_dir:
        any_result = next(iter(results.values()))
        export_for_inference(
            results,
            args.export_dir,
            window_size=args.window_size,
            step_size=args.step_size,
            label_encoder=any_result.label_encoder,
            num_channels=NUM_CHANNELS,
        )
        print(f"Exported models to: {args.export_dir}")

    if "cnn_1d" in (args.models or MODEL_REGISTRY.keys()) and not _HAS_TENSORFLOW:
        print("Note: TensorFlow not available; cnn_1d was skipped.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())