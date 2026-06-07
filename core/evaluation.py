import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import argparse
import mlflow
import json
import os
import sys
import tensorflow as tf
from core.config import PipelineConfig
from core.constants import CLASS_NAMES

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    accuracy: float
    per_class_precision: Dict[str, float]
    per_class_recall: Dict[str, float]
    macro_f1: float
    approved: bool
    failed_metrics: List[Tuple[str, float, float]]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0.0)),
    }

    precision_per_class = precision_score(
        y_true, y_pred, average=None, labels=range(len(CLASS_NAMES)), zero_division=0.0
    )
    recall_per_class = recall_score(
        y_true, y_pred, average=None, labels=range(len(CLASS_NAMES)), zero_division=0.0
    )

    for i, name in enumerate(CLASS_NAMES):
        metrics[f"precision_{name}"] = float(precision_per_class[i])
        metrics[f"recall_{name}"] = float(recall_per_class[i])

    return metrics


def evaluate_against_thresholds(
    metrics,
    thresholds,
):
    failed = []
    for metric_name, threshold_value in thresholds.items():
        actual = metrics.get(metric_name)
        if actual is not None and actual < threshold_value:
            failed.append((metric_name, actual, threshold_value))
    approved = len(failed) == 0
    return approved, failed


def build_thresholds_from_config(evaluation_config):
    thresholds = {
        "accuracy": evaluation_config.accuracy_threshold,
        "macro_f1": evaluation_config.f1_threshold,
    }
    for name in CLASS_NAMES:
        thresholds[f"precision_{name}"] = evaluation_config.precision_threshold
        thresholds[f"recall_{name}"] = evaluation_config.recall_threshold
    return thresholds


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    thresholds: Dict[str, float],
    run_id: str = "",
    config=None,
):
    metrics = compute_metrics(y_true, y_pred)
    approved, failed_metrics = evaluate_against_thresholds(metrics, thresholds)

    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({f"eval_{k}": v for k, v in metrics.items()})
        mlflow.log_metrics(
            {
                "eval_approved": 1.0 if approved else 0.0,
            }
        )

    per_class_precision = {name: metrics[f"precision_{name}"] for name in CLASS_NAMES}
    per_class_recall = {name: metrics[f"recall_{name}"] for name in CLASS_NAMES}

    return EvaluationResult(
        accuracy=metrics["accuracy"],
        per_class_precision=per_class_precision,
        per_class_recall=per_class_recall,
        macro_f1=metrics["macro_f1"],
        approved=approved,
        failed_metrics=failed_metrics,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run evaluation step")
    parser.add_argument("--config", required=True, help="Path to pipeline_config.yaml")
    parser.add_argument(
        "--model-dir", required=True, help="Directory with trained model"
    )
    parser.add_argument("--data-dir", required=True, help="Directory with test data")
    parser.add_argument("--run-id", default="", help="MLflow run ID from training")
    args = parser.parse_args()

    config = PipelineConfig.from_yaml(args.config)

    # Load test data
    x_test_path = os.path.join(args.data_dir, "x_test.npy")
    y_test_path = os.path.join(args.data_dir, "y_test.npy")

    if not os.path.exists(x_test_path) or not os.path.exists(y_test_path):
        logger.error(f"Missing x_test.npy or y_test.npy in {args.data_dir}")
        sys.exit(1)

    x_test = np.load(x_test_path)
    y_test = np.load(y_test_path)  # one-hot encoded

    # Load model
    model_path = os.path.join(args.model_dir, "model.keras")
    if not os.path.exists(model_path):
        # Try without .keras extension
        model_path = args.model_dir
    model = tf.keras.models.load_model(model_path)
    logger.info(f"Model loaded from {model_path}")

    # Run predictions
    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Compute metrics and evaluate against thresholds
    thresholds = build_thresholds_from_config(config.evaluation)
    result = evaluate_model(
        y_true, y_pred, thresholds, run_id=args.run_id, config=config
    )

    logger.info(
        f"Evaluation: accuracy={result.accuracy:.4f}, macro_f1={result.macro_f1:.4f}, approved={result.approved}"
    )
    if result.failed_metrics:
        for name, actual, threshold in result.failed_metrics:
            logger.warning(
                f"  FAILED: {name} = {actual:.4f} (threshold: {threshold:.4f})"
            )

    # Save metrics to JSON for the promotion step
    metrics = compute_metrics(y_true, y_pred)
    metrics_file = os.path.join(args.data_dir, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_file}")
