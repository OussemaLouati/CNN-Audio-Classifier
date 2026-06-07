import logging
from typing import Dict
import argparse
import json
import os
import sys
from mlflow.tracking import MlflowClient
from core.config import PipelineConfig
from core.evaluation import evaluate_against_thresholds
from core.evaluation import build_thresholds_from_config
logger = logging.getLogger(__name__)


def get_model_version_from_run(client, model_name: str, run_id: str):
    try:
        versions = client.search_model_versions(f"name='{model_name}' and run_id='{run_id}'")
        if versions:
            return versions[0].version
    except Exception as e:
        logger.warning(f"Failed to find model version for run {run_id}: {e}")
    return None


def promote_model(
    run_id: str,
    metrics: Dict[str, float],
    thresholds: Dict[str, float],
    config,
) -> bool:
    model_name = config.mlflow.model_name
    client = MlflowClient(tracking_uri=config.mlflow.tracking_uri)

    # Check if metrics pass thresholds
    approved, failed = evaluate_against_thresholds(metrics, thresholds)
    if not approved:
        logger.info(
            f"Model from run {run_id} does not pass thresholds. Failed metrics: {[(name, actual, thresh) for name, actual, thresh in failed]}"
        )
        return False

    # Get the model version for this run
    version = get_model_version_from_run(client, model_name, run_id)
    if version is None:
        logger.error(f"No registered model version found for run {run_id}")
        return False

    # Compare against current Production model
    try:
        prod_versions = []
        try:
            prod_version_info = client.get_model_version_by_alias(model_name, "champion")
            if prod_version_info:
                prod_versions = [prod_version_info]
        except Exception:
            pass


        if prod_versions:
            prod_run_id = prod_versions[0].run_id
            prod_run = client.get_run(prod_run_id)
            prod_metrics = prod_run.data.metrics

            # Compare: new model must have higher macro_f1
            new_f1 = metrics.get("macro_f1", 0)
            prod_f1 = prod_metrics.get("eval_macro_f1", prod_metrics.get("macro_f1", 0))

            if new_f1 <= prod_f1:
                logger.info(
                    f"New model (macro_f1={new_f1:.4f}) does not beat production model (macro_f1={prod_f1:.4f}). Keeping as Staging."
                )
                return False

            logger.info(
                f"New model (macro_f1={new_f1:.4f}) beats production model (macro_f1={prod_f1:.4f}). Promoting."
            )
    except Exception as e:
        logger.info(f"No existing production model found ({e}). Promoting new model.")

    # Promote to Production using aliases
    try:
        client.set_registered_model_alias(model_name, "champion", version)
        logger.info(
            f"Model version {version} (run {run_id}) promoted: alias 'champion' set."
        )
    except Exception as e:
        logger.warning(f"Alias API failed ({e}), trying legacy stages...")
        try:
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Production",
                archive_existing_versions=True,
            )
            logger.info(
                f"Model version {version} (run {run_id}) promoted to Production stage."
            )
        except Exception as e2:
            logger.error(f"Failed to promote model: {e2}")
            return False

    client.set_model_version_tag(model_name, version, "status", "production")
    client.set_model_version_tag(model_name, version, "promoted_by", "pipeline")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model promotion step")
    parser.add_argument("--config", required=True, help="Path to pipeline_config.yaml")
    parser.add_argument("--run-id", required=True, help="MLflow run ID from training")
    parser.add_argument("--metrics-file", required=True, help="JSON file with evaluation metrics")
    args = parser.parse_args()


    config = PipelineConfig.from_yaml(args.config)

    metrics = {}
    if os.path.exists(args.metrics_file):
        with open(args.metrics_file, "r") as f:
            metrics = json.load(f)

    thresholds = build_thresholds_from_config(config.evaluation)

    promoted = promote_model(
        run_id=args.run_id,
        metrics=metrics,
        thresholds=thresholds,
        config=config,
    )

    # Write result for Argo output parameter
    with open("/tmp/promoted.txt", "w") as f:
        f.write("true" if promoted else "false")

    if promoted:
        logger.info("Model promotion succeeded.")
    else:
        logger.info("Model was not promoted.")
