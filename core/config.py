from dataclasses import dataclass, field
import yaml
import os


@dataclass
class ExtractionConfig:
    mel_bands: int = 128
    min_segment_duration_ms: float = 50.0
    sample_rate: int = 22050
    hop_length: int = 512
    n_fft: int = 2048


@dataclass
class TrainingConfig:
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 0.001
    random_seed: int = 42
    train_split: float = 0.70
    val_split: float = 0.15
    test_split: float = 0.15


@dataclass
class EvaluationConfig:
    accuracy_threshold: float = 0.80
    precision_threshold: float = 0.75
    recall_threshold: float = 0.75
    f1_threshold: float = 0.78


@dataclass
class PreLabellerConfig:
    confidence_threshold: float = 0.7
    review_marker: str = "[review]"


@dataclass
class MLflowConfig:
    tracking_uri: str = "http://mlflow.audio-classifier:5000"
    experiment_name: str = "audio-classification"
    model_name: str = "audio-classifier-cnn"


@dataclass
class StorageConfig:
    backend: str = "minio"
    minio_endpoint: str = "minio.audio-classifier:9000"
    minio_bucket: str = "mlflow-artifacts"
    minio_access_key: str = ""
    minio_secret_key: str = ""


@dataclass
class PipelineConfig:
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    pre_labeller: PreLabellerConfig = field(default_factory=PreLabellerConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        extraction = ExtractionConfig(**data.pop("extraction", {}))
        training = TrainingConfig(**data.pop("training", {}))
        evaluation = EvaluationConfig(**data.pop("evaluation", {}))
        pre_labeller = PreLabellerConfig(**data.pop("pre_labeller", {}))

        mlflow_data = data.pop("mlflow", {})
        mlflow_config = MLflowConfig(**mlflow_data)

        storage_data = data.pop("storage", {})
        for key, value in storage_data.items():
            if (
                isinstance(value, str)
                and value.startswith("${")
                and value.endswith("}")
            ):
                env_var = value[2:-1]
                storage_data[key] = os.environ.get(env_var, "")
        storage_config = StorageConfig(**storage_data)

        return cls(
            extraction=extraction,
            training=training,
            evaluation=evaluation,
            pre_labeller=pre_labeller,
            mlflow=mlflow_config,
            storage=storage_config,
            **data,
        )
