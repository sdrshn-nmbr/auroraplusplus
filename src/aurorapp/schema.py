from pathlib import Path
from typing import Any

import orjson
from pydantic import BaseModel

from aurorapp.models import (
    ArtifactRef,
    CandidateCheckpoint,
    DataManifest,
    EvaluatorBundle,
    Event,
    ExperimentDraft,
    ExperimentSpec,
    HumanDecision,
    JudgeCheck,
    JudgeResult,
    PhysicalEvalReport,
    PromotionDecision,
    TraceRecord,
    TrainingBatch,
)

SCHEMA_TYPES: tuple[type[BaseModel], ...] = (
    ExperimentDraft,
    ExperimentSpec,
    DataManifest,
    EvaluatorBundle,
    TraceRecord,
    TrainingBatch,
    CandidateCheckpoint,
    PhysicalEvalReport,
    JudgeCheck,
    JudgeResult,
    HumanDecision,
    PromotionDecision,
    Event,
    ArtifactRef,
)


def write_schemas(directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for model_type in SCHEMA_TYPES:
        schema: dict[str, Any] = model_type.model_json_schema()
        path = directory / f"{model_type.__name__}.schema.json"
        encoded = orjson.dumps(schema, option=orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2)
        path.write_bytes(encoded + b"\n")
        paths.append(path)
    return paths
