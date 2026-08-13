from pathlib import Path

import pytest
from pydantic import ValidationError

from aurorapp.canonical import canonical_sha256
from aurorapp.models import ExperimentDraft, SystemMode
from aurorapp.schema import write_schemas

FIXTURE = Path(__file__).parents[2] / "configs" / "laguna_dflash_int4.json"


def test_pinned_experiment_has_stable_identity() -> None:
    spec = ExperimentDraft.model_validate_json(FIXTURE.read_text())

    assert spec.target.revision == "4b7e28abdc0a8b121def816b89d631750bc53c92"
    assert spec.drafter.revision == "630267be6a2ad870ca80fc3930bde87d6ad7bb24"
    assert spec.initial_mode is SystemMode.OFF
    assert canonical_sha256(spec) == canonical_sha256(spec.model_dump(mode="json"))


def test_contract_rejects_unknown_fields() -> None:
    payload = ExperimentDraft.model_validate_json(FIXTURE.read_text()).model_dump(mode="json")
    payload["unreviewed_override"] = True

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ExperimentDraft.model_validate(payload)


def test_checked_in_schemas_match_types(tmp_path: Path) -> None:
    generated = write_schemas(tmp_path)
    repository_schemas = Path(__file__).parents[2] / "schemas"

    for path in generated:
        assert (repository_schemas / path.name).read_bytes() == path.read_bytes()
