import argparse
import json
import logging
import sys
import urllib.request
from dataclasses import dataclass
from typing import List, Tuple
import librosa
import numpy as np

from core.config import PipelineConfig
from core.extraction import compute_mel_spectrogram
from core.constants import CLASS_NAMES

logger = logging.getLogger(__name__)


@dataclass
class PredictionEntry:
    start_time: float
    end_time: float
    label: str
    confidence: float


def format_label_file(
    predictions: List[PredictionEntry],
    confidence_threshold: float,
    review_marker: str = "[review]",
) -> str:
    lines = []
    for pred in predictions:
        marker = review_marker if pred.confidence < confidence_threshold else ""
        line = f"{marker}{pred.start_time:.6f}\t{pred.end_time:.6f}\t{pred.label}\t{pred.confidence:.4f}"
        lines.append(line)
    return "\n".join(lines)


def parse_label_file_with_confidence(text: str) -> List[PredictionEntry]:
    entries = []
    for line in text.strip().split("\n"):
        if not line:
            continue
        if line.startswith("[review]"):
            line = line[1:]
        parts = line.split("\t")
        entries.append(PredictionEntry(
            start_time=float(parts[0]),
            end_time=float(parts[1]),
            label=parts[2],
            confidence=float(parts[3]),
        ))
    return entries


def validate_labels(labels: List[str]) -> List[Tuple[int, str]]:
    """Validate that all labels are one of the valid classes."""
    valid = {"b", "mb", "h"}
    return [(i, label) for i, label in enumerate(labels) if label not in valid]


def segment_audio(
    audio: np.ndarray,
    sample_rate: int,
    segment_duration: float = 1.0,
    hop_duration: float = 0.5,
) -> List[tuple]:
    segment_samples = int(segment_duration * sample_rate)
    hop_samples = int(hop_duration * sample_rate)
    total_samples = len(audio)

    segments = []
    start = 0
    while start + segment_samples <= total_samples:
        end = start + segment_samples
        start_time = start / sample_rate
        end_time = end / sample_rate
        segment = audio[start:end]
        segments.append((start_time, end_time, segment))
        start += hop_samples

    return segments


def predict_via_endpoint(spectrogram: np.ndarray, endpoint_url: str) -> np.ndarray:
    model_input = spectrogram[:, :, np.newaxis].tolist()  # (128, frames, 1)

    payload = json.dumps({"data": {"ndarray": [model_input]}}).encode()
    req = urllib.request.Request(
        f"{endpoint_url}/api/v1.0/predictions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    resp = urllib.request.urlopen(req, timeout=30)
    result = json.loads(resp.read())

    # Seldon returns {"data": {"ndarray": [[p1, p2, p3]]}}
    probs = np.array(result["data"]["ndarray"][0])
    return probs


def run_inference(
    audio_path: str,
    endpoint_url: str,
    config: PipelineConfig,
) -> List[PredictionEntry]:
   
    audio, _ = librosa.load(audio_path, sr=config.extraction.sample_rate)

    segments = segment_audio(
        audio,
        sample_rate=config.extraction.sample_rate,
        segment_duration=1.0,
        hop_duration=0.5,
    )

    if not segments:
        return []

    predictions = []
    for start_time, end_time, segment in segments:
        spectrogram = compute_mel_spectrogram(
            audio_segment=segment,
            sample_rate=config.extraction.sample_rate,
            n_mels=config.extraction.mel_bands,
            hop_length=config.extraction.hop_length,
            n_fft=config.extraction.n_fft,
        )

        try:
            probs = predict_via_endpoint(spectrogram, endpoint_url)
        except Exception as e:
            logger.warning(f"Prediction failed for segment {start_time:.2f}-{end_time:.2f}: {e}")
            continue

        class_idx = int(np.argmax(probs))
        confidence = float(probs[class_idx])
        label = CLASS_NAMES[class_idx]

        predictions.append(PredictionEntry(
            start_time=start_time,
            end_time=end_time,
            label=label,
            confidence=confidence,
        ))

    return predictions


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate draft labels from model inference on audio files."
    )
    parser.add_argument("--input", required=True, help="Path to the input audio file.")
    parser.add_argument("--config", required=True, help="Path to pipeline_config.yaml.")
    parser.add_argument("--output", required=True, help="Path for the output label file.")
    parser.add_argument("--endpoint", default="http://localhost:9500",
                        help="Seldon prediction endpoint URL (default: http://localhost:9500)")
    args = parser.parse_args()

    try:
        config = PipelineConfig.from_yaml(args.config)
    except (FileNotFoundError, OSError) as e:
        logger.error(f"Failed to load config: {e}")
        return 1


    try:
        predictions = run_inference(args.input, args.endpoint, config)
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return 1

    if not predictions:
        logger.warning(f"No predictions generated for {args.input}")

    # Format output
    output_text = format_label_file(
        predictions=predictions,
        confidence_threshold=config.pre_labeller.confidence_threshold,
        review_marker=config.pre_labeller.review_marker,
    )

    # Write output
    try:
        with open(args.output, "w") as f:
            f.write(output_text)
            if output_text:
                f.write("\n")
    except OSError as e:
        logger.error(f"Failed to write output file: {e}")
        return 1

    logger.info(f"Pre-labelling complete: {len(predictions)} predictions written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())