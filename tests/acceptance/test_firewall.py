import pytest

from aurorapp.canonical import canonical_sha256
from aurorapp.firewall import DataFirewall, FirewallViolation
from aurorapp.models import (
    ArtifactRef,
    DataManifest,
    DataPartition,
    DataRecord,
    PartitionMembership,
    SealedSourceRef,
)

REVISION = "b76504d85f7f14ef4b1fad81f136f638f2ce625b"


def membership(name: DataPartition, row_id: str) -> PartitionMembership:
    return PartitionMembership(
        name=name,
        ordered_row_ids=(row_id,),
        ordered_row_ids_hash=canonical_sha256((row_id,)),
        count=1,
        source_revision=REVISION,
        group_assignment_hash="4" * 64,
        build_report=ArtifactRef(
            content_hash="5" * 64,
            size=1,
            storage_path="objects/55/" + "5" * 64 + "/report.json",
            producer="test",
            validation_result="valid",
        ),
    )


@pytest.fixture
def firewall() -> DataFirewall:
    partitions = (
        membership(DataPartition.TRAIN, "train-1"),
        membership(DataPartition.EVALUATOR_DEVELOPMENT, "dev-1"),
        membership(DataPartition.PROMOTION_SAME_FAMILY, "same-1"),
        membership(DataPartition.PROMOTION_UNSEEN_FAMILY, "unseen-1"),
    )
    manifest = DataManifest(
        source="GPUMODE/KernelBook",
        revision=REVISION,
        approved_row_ids=("train-1", "dev-1", "same-1", "unseen-1"),
        exclusions={},
        partitions=partitions,
        license_policy_version="permissive-v1",
        duplicate_check_version="v1",
        sealed_sources=(
            SealedSourceRef(
                repository="ScalingIntelligence/KernelBench",
                revision="sealed",
                manifest_hash="6" * 64,
            ),
        ),
    )
    return DataFirewall(manifest)


def test_public_audit_record_cannot_enter_training(firewall: DataFirewall) -> None:
    record = DataRecord(
        record_id="kb-1",
        source="ScalingIntelligence/KernelBench",
        source_revision="sealed",
        partition=DataPartition.PUBLIC_AUDIT,
        prompt_hash="0" * 64,
        content_hash="1" * 64,
    )

    with pytest.raises(FirewallViolation, match="sealed source"):
        firewall.authorize_training(record)


def test_judge_cannot_see_internal_promotion_holdout(firewall: DataFirewall) -> None:
    record = DataRecord(
        record_id="unseen-1",
        source="GPUMODE/KernelBook",
        source_revision="b76504d85f7f14ef4b1fad81f136f638f2ce625b",
        partition=DataPartition.PROMOTION_UNSEEN_FAMILY,
        prompt_hash="2" * 64,
        content_hash="3" * 64,
    )

    with pytest.raises(FirewallViolation, match="judge"):
        firewall.authorize_judge(record)


def test_forged_train_label_cannot_hide_sealed_source(firewall: DataFirewall) -> None:
    record = DataRecord(
        record_id="train-1",
        source="ScalingIntelligence/KernelBench",
        source_revision="sealed",
        partition=DataPartition.TRAIN,
        prompt_hash="7" * 64,
        content_hash="8" * 64,
    )

    with pytest.raises(FirewallViolation, match="sealed source"):
        firewall.authorize_training(record)


def test_forged_partition_label_cannot_cross_manifest(firewall: DataFirewall) -> None:
    record = DataRecord(
        record_id="unseen-1",
        source="GPUMODE/KernelBook",
        source_revision=REVISION,
        partition=DataPartition.TRAIN,
        prompt_hash="9" * 64,
        content_hash="a" * 64,
    )

    with pytest.raises(FirewallViolation, match="disagrees"):
        firewall.authorize_training(record)
