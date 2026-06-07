import argparse
import logging
import os

import tensorflow as tf
import numpy as np
from flask import Flask, jsonify, request

logger = logging.getLogger(__name__)

app = Flask(__name__)
_model = None


def load_model(model_path: str):
    global _model

    candidates = [
        os.path.join(model_path, "data", "model.keras"),
        os.path.join(model_path, "model.keras"),
        model_path,
    ]

    for path in candidates:
        if os.path.exists(path) and path.endswith(".keras"):
            _model = tf.keras.models.load_model(path)
            logger.info(f"Model loaded from {path}")
            return

    if os.path.isdir(model_path):
        _model = tf.keras.layers.TFSMLayer(model_path, call_endpoint="serving_default")
        logger.info(f"Model loaded as TFSMLayer from {model_path}")
        return

    raise ValueError(f"No loadable model found at {model_path}")


@app.route("/health/status", methods=["GET"])
def health():
    if _model is None:
        return jsonify({"status": "unhealthy"}), 503
    return jsonify({"status": "healthy"}), 200


@app.route("/api/v1.0/predictions", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "data" not in data or "ndarray" not in data["data"]:
        return jsonify({"error": "Expected {data: {ndarray: [...]}}"}), 400

    try:
        input_array = np.array(data["data"]["ndarray"], dtype=np.float32)

        probs = _model.predict(input_array, verbose=0)
        return jsonify({"data": {"ndarray": probs.tolist()}}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def main():
    parser = argparse.ArgumentParser(description="Model serving endpoint")
    parser.add_argument(
        "--model-path", default="/mnt/models", help="Path to model directory"
    )
    parser.add_argument("--port", type=int, default=8080, help="Port")
    parser.add_argument("--host", default="0.0.0.0", help="Host")
    args = parser.parse_args()

    load_model(args.model_path)
    logger.info(f"Starting server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
