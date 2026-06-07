import logging
from dataclasses import dataclass
from typing import List

import numpy as np
import librosa
import argparse
import glob
import os
import sys

from core.config import PipelineConfig
from core.constants import CLASS_NAMES

logger = logging.getLogger(__name__)


@dataclass
class LabelEntry:
    start_time: float  
    end_time: float  
    label: str  
    source_file: str


def parse_label_file(file_path: str, source_audio_path: str):
    entries = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            start_time = float(parts[0])
            end_time = float(parts[1])
            label = parts[2]
            entries.append(
                LabelEntry(
                    start_time=start_time,
                    end_time=end_time,
                    label=label,
                    source_file=source_audio_path,
                )
            )
    return entries


def filter_short_segments(
    entries: List[LabelEntry], min_duration_ms: float = 50.0
):

    valid = []
    discarded = []
    for entry in entries:
        duration_ms = (entry.end_time - entry.start_time) * 1000.0
        if duration_ms < min_duration_ms:
            logger.warning(
                f"Discarding short segment ({duration_ms}ms < {min_duration_ms}ms): "
                f"source={entry.source_file}, start={entry.start_time}, end={entry.end_time}, label={entry.label}"
            )
            discarded.append(entry)
        else:
            valid.append(entry)
    return valid, discarded


def load_audio_segment(
    file_path: str,
    start_time: float,
    end_time: float,
    sample_rate: int,
) -> np.ndarray:
    duration = end_time - start_time
    audio, _ = librosa.load(
        file_path,
        sr=sample_rate,
        offset=start_time,
        duration=duration,
    )
    return audio


def compute_mel_spectrogram(
    audio_segment: np.ndarray,
    sample_rate: int,
    n_mels: int,
    hop_length: int,
    n_fft: int,
) -> np.ndarray:
    spectrogram = librosa.feature.melspectrogram(
        y=audio_segment,
        sr=sample_rate,
        n_mels=n_mels,
        hop_length=hop_length,
        n_fft=n_fft,
    )
    return librosa.power_to_db(spectrogram, ref=np.max)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run extraction step")
    parser.add_argument("--config", required=True, help="Path to pipeline_config.yaml")
    parser.add_argument("--input-dir", required=True, help="Directory with audio + label files")
    parser.add_argument("--output-dir", required=True, help="Directory to write spectrograms")
    args = parser.parse_args()

    config = PipelineConfig.from_yaml(args.config)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Find all .txt label files in input directory
    label_files = glob.glob(os.path.join(args.input_dir, "*.txt"))
    if not label_files:
        logger.error(f"No .txt label files found in {args.input_dir}")
        sys.exit(1)

    all_spectrograms = []
    all_labels = []
    total_extracted = 0
    total_discarded = 0

    for label_file in sorted(label_files):
        # Find matching audio file (.wav)
        base_name = os.path.splitext(label_file)[0]
        audio_file = base_name + ".wav"
        if not os.path.exists(audio_file):
            logger.warning(f"No matching .wav for {label_file}, skipping")
            continue

        logger.info(f"Processing: {os.path.basename(label_file)}")

        # Parse label file
        entries = parse_label_file(label_file, audio_file)

        # Filter to only valid labels (b, mb, h)
        entries = [e for e in entries if e.label in CLASS_NAMES]

        # Filter short segments
        valid_entries, discarded = filter_short_segments(
            entries, min_duration_ms=config.extraction.min_segment_duration_ms
        )
        total_discarded += len(discarded)

        # Extract spectrograms for each valid segment
        for entry in valid_entries:
            try:
                audio = load_audio_segment(
                    file_path=entry.source_file,
                    start_time=entry.start_time,
                    end_time=entry.end_time,
                    sample_rate=config.extraction.sample_rate,
                )

                # Skip if audio is too short for FFT
                if len(audio) < config.extraction.n_fft:
                    logger.warning(
                        f"Audio segment too short for FFT ({len(audio)} < {config.extraction.n_fft} samples), skipping"
                    )
                    continue

                spectrogram = compute_mel_spectrogram(
                    audio_segment=audio,
                    sample_rate=config.extraction.sample_rate,
                    n_mels=config.extraction.mel_bands,
                    hop_length=config.extraction.hop_length,
                    n_fft=config.extraction.n_fft,
                )

                all_spectrograms.append(spectrogram)
                # Map labels to indices: b=0, mb=1, h=2
                label_map = {"b": 0, "mb": 1, "h": 2}
                all_labels.append(label_map[entry.label])
                total_extracted += 1

            except Exception as e:
                logger.warning(f"Failed to extract segment: {e}")
                continue

    # Pad spectrograms to uniform width (max time frames)
    max_frames = max(s.shape[1] for s in all_spectrograms)
    padded = np.zeros((len(all_spectrograms), config.extraction.mel_bands, max_frames), dtype=np.float32)
    for i, s in enumerate(all_spectrograms):
        padded[i, :, :s.shape[1]] = s

    # Save as numpy arrays
    np.save(os.path.join(args.output_dir, "spectrograms.npy"), padded)
    np.save(os.path.join(args.output_dir, "labels.npy"), np.array(all_labels, dtype=np.int32))

    logger.info(
        f"Extraction complete: {total_extracted} spectrograms extracted, {total_discarded} discarded, shape={padded.shape}"
    )
