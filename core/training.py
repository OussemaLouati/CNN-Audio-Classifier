import hashlib
import json
import logging
import sys
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import tensorflow as tf
import numpy as np
import mlflow
import mlflow.tensorflow
import tensorflow as tf
import argparse
import os

from core.config import PipelineConfig
logger = logging.getLogger(__name__)


@dataclass
class DatasetSplit:
    train_indices: List[int]
    val_indices: List[int]
    test_indices: List[int]


@dataclass
class TrainingResult:
    model_version: str
    model_path: str
    run_id: str
    history: Dict[str, List[float]]
    epochs_completed: int
    final_val_loss: Optional[float]
    final_val_accuracy: Optional[float]


def split_dataset(
    num_samples: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> DatasetSplit:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_samples).tolist()

    train_end = int(num_samples * train_ratio)
    val_end = train_end + int(num_samples * val_ratio)

    return DatasetSplit(
        train_indices=indices[:train_end],
        val_indices=indices[train_end:val_end],
        test_indices=indices[val_end:],
    )


def compute_config_hash(config: Dict[str, Any]) -> str:
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()


def generate_model_version() -> str:
    return f"v-{uuid.uuid4().hex[:8]}"


def build_cnn_model(input_shape: Tuple[int, int, int], num_classes: int = 3):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])
    return model


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    input_shape: Tuple[int, int, int],
    num_classes: int = 3,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    model_save_path: str = "model_artefact",
    dataset_version: str = "unknown",
    config_hash: str = "",
    commit_sha: str = "unknown",
    config=None,
):
    try:
        
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
        mlflow.set_experiment(config.mlflow.experiment_name)

        model_name = config.mlflow.model_name

        with mlflow.start_run() as run:
            run_id = run.info.run_id

            # Log parameters
            mlflow.log_params({
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "num_classes": num_classes,
                "input_shape": str(list(input_shape)),
                "dataset_version": dataset_version,
                "config_hash": config_hash,
                "commit_sha": commit_sha,
            })

            # Build and compile the model
            model = build_cnn_model(input_shape=input_shape, num_classes=num_classes)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss="categorical_crossentropy",
                metrics=["accuracy"],
            )

            history = model.fit(
                x_train,
                y_train,
                validation_data=(x_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                verbose=1,
            )

            # Extract training history
            history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
            final_val_loss = history_dict.get("val_loss", [None])[-1]
            final_val_accuracy = history_dict.get("val_accuracy", [None])[-1]

            # Log metrics
            mlflow.log_metrics({
                "val_loss": final_val_loss if final_val_loss is not None else 0.0,
                "val_accuracy": final_val_accuracy if final_val_accuracy is not None else 0.0,
                "epochs_completed": len(history_dict.get("loss", [])),
            })

            save_path = model_save_path
            if not save_path.endswith((".keras", ".h5", ".hdf5")):
                save_path = f"{save_path}.keras"
            model.save(save_path)
            logger.info(f"Model saved to {save_path}")

            # Log model artifact to MLflow registry
            mlflow.tensorflow.log_model(
                model,
                "model",
                registered_model_name=model_name,
            )

            # Generate a unique model version
            model_version = generate_model_version()

            logger.info(f"Training complete. MLflow run_id={run_id}, model_version={model_version}")

            return TrainingResult(
                model_version=model_version,
                model_path=save_path,
                run_id=run_id,
                history=history_dict,
                epochs_completed=len(history_dict.get("loss", [])),
                final_val_loss=final_val_loss,
                final_val_accuracy=final_val_accuracy,
            )

    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":   
    parser = argparse.ArgumentParser(description="Run training step")
    parser.add_argument("--config", required=True, help="Path to pipeline_config.yaml")
    parser.add_argument("--data-dir", required=True, help="Directory with extracted spectrograms")
    parser.add_argument("--output-dir", required=True, help="Directory to write model artifacts")
    args = parser.parse_args()

    config = PipelineConfig.from_yaml(args.config)

    # Load extracted data
    spectrograms_path = os.path.join(args.data_dir, "spectrograms.npy")
    labels_path = os.path.join(args.data_dir, "labels.npy")

    if not os.path.exists(spectrograms_path) or not os.path.exists(labels_path):
        logger.error(f"Missing spectrograms.npy or labels.npy in {args.data_dir}")
        with open("/tmp/run-id.txt", "w") as f:
            f.write(f"failed-{uuid.uuid4().hex[:8]}")
        sys.exit(1)

    X = np.load(spectrograms_path)  # shape: (N, 128, max_frames)
    y_indices = np.load(labels_path)  # shape: (N,) with values 0, 1, 2

    logger.info(f"Loaded {len(X)} samples, shape={X.shape}")

    # Add channel dimension: (N, 128, frames) -> (N, 128, frames, 1)
    X = X[..., np.newaxis]

    # One-hot encode labels
    y = tf.keras.utils.to_categorical(y_indices, num_classes=3)

    # Split dataset
    split = split_dataset(
        num_samples=len(X),
        train_ratio=config.training.train_split,
        val_ratio=config.training.val_split,
        test_ratio=config.training.test_split,
        seed=config.training.random_seed,
    )

    x_train = X[split.train_indices]
    y_train = y[split.train_indices]
    x_val = X[split.val_indices]
    y_val = y[split.val_indices]
    x_test = X[split.test_indices]
    y_test = y[split.test_indices]

    logger.info(f"Split: train={len(x_train)}, val={len(x_val)}, test={len(x_test)}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Compute config hash
    config_dict = {
        "epochs": config.training.epochs,
        "batch_size": config.training.batch_size,
        "learning_rate": config.training.learning_rate,
        "random_seed": config.training.random_seed,
    }

    # Train model
    result = train_model(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        input_shape=x_train.shape[1:],  # (128, frames, 1)
        num_classes=3,
        epochs=config.training.epochs,
        batch_size=config.training.batch_size,
        learning_rate=config.training.learning_rate,
        model_save_path=os.path.join(args.output_dir, "model"),
        dataset_version=f"n={len(X)}",
        config_hash=compute_config_hash(config_dict),
        commit_sha=os.environ.get("COMMIT_SHA", "unknown"),
        config=config,
    )

    # Save test data for evaluation step
    np.save(os.path.join(args.data_dir, "x_test.npy"), x_test)
    np.save(os.path.join(args.data_dir, "y_test.npy"), y_test)

    # Write run-id for Argo output parameter
    with open("/tmp/run-id.txt", "w") as f:
        f.write(result.run_id)

    logger.info(
        f"Training complete. run_id={result.run_id}, model_version={result.model_version}, val_loss={result.final_val_loss or 0:.4f}, val_acc={result.final_val_accuracy or 0:.4f}"
    )
    sys.exit(0)