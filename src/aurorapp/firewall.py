from typing import ClassVar

from aurorapp.models import DataManifest, DataPartition, DataRecord


class FirewallViolation(ValueError):
    pass


class DataFirewall:
    _judge_partitions: ClassVar[set[DataPartition]] = {
        DataPartition.TRAIN,
        DataPartition.EVALUATOR_DEVELOPMENT,
    }

    def __init__(self, manifest: DataManifest) -> None:
        self.manifest = manifest
        self._membership = {
            row_id: partition.name
            for partition in manifest.partitions
            for row_id in partition.ordered_row_ids
        }
        self._sealed_sources = {source.repository for source in manifest.sealed_sources}

    def _trusted_partition(self, record: DataRecord) -> DataPartition:
        if record.source in self._sealed_sources:
            raise FirewallViolation(
                f"sealed source {record.source} is unavailable to the normal loop"
            )
        source_mismatch = record.source != self.manifest.source
        revision_mismatch = record.source_revision != self.manifest.revision
        if source_mismatch or revision_mismatch:
            raise FirewallViolation("record source identity does not match the signed manifest")
        partition = self._membership.get(record.record_id)
        if partition is None:
            raise FirewallViolation("record is not a member of the signed manifest")
        if partition is not record.partition:
            raise FirewallViolation("caller-supplied partition disagrees with signed membership")
        return partition

    def authorize_training(self, record: DataRecord) -> None:
        partition = self._trusted_partition(record)
        if partition is DataPartition.PUBLIC_AUDIT:
            raise FirewallViolation("public audit data cannot enter training")
        if partition is not DataPartition.TRAIN:
            raise FirewallViolation(f"{partition.value} data cannot enter training")

    def authorize_judge(self, record: DataRecord) -> None:
        partition = self._trusted_partition(record)
        if partition not in self._judge_partitions:
            raise FirewallViolation(f"{partition.value} data is unavailable to the judge")

    def authorize_promotion(self, record: DataRecord) -> None:
        allowed = {
            DataPartition.PROMOTION_SAME_FAMILY,
            DataPartition.PROMOTION_UNSEEN_FAMILY,
        }
        partition = self._trusted_partition(record)
        if partition not in allowed:
            raise FirewallViolation(f"{partition.value} data cannot enter ordinary promotion")
