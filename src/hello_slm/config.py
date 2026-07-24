from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonschema  # type: ignore[import-untyped]

PARAMETER_CAP = 500_000_000


class ConfigError(ValueError):
    """Raised when a user-provided experiment config is invalid."""


def repo_root(start: Path | None = None) -> Path:
    path = (start or Path.cwd()).resolve()
    for candidate in (path, *path.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "schemas").exists():
            return candidate
    raise ConfigError(f"could not find repository root from {path}")


def resolve_repo_path(path: str | Path, root: Path | None = None) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value.resolve()
    return (root or repo_root()) / value


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ExperimentConfig:
    path: Path
    root: Path
    data: dict[str, Any]
    effective_hash: str
    parameter_count: int

    @property
    def artifact_dir(self) -> Path:
        return resolve_repo_path(self.data["run"]["artifact_dir"], self.root)

    @property
    def manifest_path(self) -> Path:
        return resolve_repo_path(self.data["corpus"]["manifest_path"], self.root)

    def resolve_path(self, value: str | Path) -> Path:
        return resolve_repo_path(value, self.root)


def load_toml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_json_schema(instance: Any, schema_path: str | Path) -> None:
    schema = load_json(schema_path)
    validator = jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker())
    errors = sorted(validator.iter_errors(instance), key=lambda error: list(error.path))
    if errors:
        message = "; ".join(
            f"{'/'.join(map(str, error.absolute_path)) or '<root>'}: {error.message}"
            for error in errors
        )
        raise ConfigError(message)


def estimate_transformer_parameters(model: dict[str, Any]) -> int:
    vocab = int(model["vocab_size"])
    d_model = int(model["d_model"])
    n_layers = int(model["n_layers"])
    n_heads = int(model["n_heads"])
    n_kv_heads = int(model["n_kv_heads"])
    d_ff = int(model["d_ff"])

    if d_model % n_heads != 0:
        raise ConfigError("model.d_model must be divisible by model.n_heads")
    if n_heads % n_kv_heads != 0:
        raise ConfigError("model.n_heads must be divisible by model.n_kv_heads")
    if n_kv_heads != n_heads:
        raise ConfigError("format_version=1 supports only n_kv_heads == n_heads")

    head_dim = d_model // n_heads
    kv_dim = n_kv_heads * head_dim

    token_embedding = vocab * d_model
    attention = (d_model * d_model) + (2 * d_model * kv_dim) + (d_model * d_model)
    swiglu = 3 * d_model * d_ff
    per_layer_norms = 2 * d_model
    layers = n_layers * (attention + swiglu + per_layer_norms)
    final_norm = d_model
    output_embedding = 0 if model["tie_embeddings"] else vocab * d_model
    return token_embedding + layers + final_norm + output_embedding


def validate_parameter_cap(config: dict[str, Any]) -> int:
    model = config["model"]
    if model["vocab_size"] != config["tokenizer"]["vocab_size"]:
        raise ConfigError("model.vocab_size must equal tokenizer.vocab_size")
    if model["max_seq_len"] != config["dataset"]["max_seq_len"]:
        raise ConfigError("model.max_seq_len must equal dataset.max_seq_len")
    if config["tokenizer"]["vocab_size"] >= 50_000:
        raise ConfigError("tokenizer.vocab_size must be less than 50000")

    parameter_count = estimate_transformer_parameters(model)
    declared_cap = min(int(model["parameter_cap"]), PARAMETER_CAP)
    if parameter_count >= declared_cap:
        raise ConfigError(
            f"estimated parameter count {parameter_count} violates cap {declared_cap}"
        )
    return parameter_count


def load_experiment_config(path: str | Path, root: Path | None = None) -> ExperimentConfig:
    config_path = Path(path).resolve()
    root_path = root or repo_root(config_path.parent)
    data = load_toml(config_path)
    validate_json_schema(data, root_path / "schemas" / "experiment.schema.json")
    parameter_count = validate_parameter_cap(data)
    return ExperimentConfig(
        path=config_path,
        root=root_path,
        data=data,
        effective_hash=canonical_sha256(data),
        parameter_count=parameter_count,
    )
