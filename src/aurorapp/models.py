from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    computed_field,
    model_validator,
)

from aurorapp.canonical import canonical_sha256

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
GitRevision = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class SystemMode(StrEnum):
    OFF = "off"
    STARTING = "starting"
    SHADOW = "shadow"
    AUTO_DRAFTER = "auto-drafter"
    DRAINING = "draining"
    FAILED = "failed"


class EvidenceLevel(StrEnum):
    CODE_EXISTS = "code-exists"
    LOCAL_TEST = "local-test"
    DETERMINISTIC_SIMULATION = "deterministic-simulation"
    REAL_PROCESS = "real-process"
    PHYSICAL_GPU = "physical-gpu"
    ALWAYS_RUNNING = "always-running"
    AUTOMATIC_PROMOTION_ROLLBACK = "automatic-promotion-rollback"


class DataPartition(StrEnum):
    TRAIN = "train"
    EVALUATOR_DEVELOPMENT = "evaluator-development"
    PROMOTION_SAME_FAMILY = "promotion-same-family"
    PROMOTION_UNSEEN_FAMILY = "promotion-unseen-family"
    PUBLIC_AUDIT = "public-audit"
    QUARANTINE = "quarantine"


class EventType(StrEnum):
    MODE_REQUESTED = "mode-requested"
    RECOVERED_TO_SHADOW = "recovered-to-shadow"
    FENCE_ACQUIRED = "fence-acquired"
    LAST_GOOD_SET = "last-good-set"
    CANDIDATE_PROMOTED = "candidate-promoted"
    PROMOTION_REJECTED = "promotion-rejected"
    LEARNING_FAILED = "learning-failed"
    DRAIN_STARTED = "drain-started"
    DRAIN_COMPLETED = "drain-completed"
    ROLLED_BACK = "rolled-back"


class ModelRef(StrictModel):
    repository: str
    revision: GitRevision
    file_hashes: dict[str, Sha256] = Field(min_length=1)
    tokenizer_hash: Sha256
    chat_template_hash: Sha256
    precision: str
    license: str


class HardwareRequest(StrictModel):
    provider: Literal["modal"]
    gpu: Literal["H100", "L40S", "B200"]
    count: int = Field(ge=1)
    memory_gib_minimum: int = Field(ge=1)
    allow_substitution: Literal[False] = False


class ObservedHardware(StrictModel):
    provider: Literal["modal"]
    sku: str
    count: int = Field(ge=1)
    gpu_uuids: list[str] = Field(min_length=1)
    topology_hash: Sha256
    driver: str
    cuda: str
    clock_policy: str


class RuntimeRef(StrictModel):
    image_digest: str
    python: str
    packages: dict[str, str]
    cuda: str
    driver: str
    sglang_revision: GitRevision
    specforge_revision: GitRevision
    kernel_revisions: dict[str, GitRevision]
    environment: dict[str, str]
    hardware: HardwareRequest
    observed_hardware: ObservedHardware


class DataSplitDraft(StrictModel):
    name: DataPartition
    ratio: float = Field(ge=0, le=1)
    row_ids_hash: Sha256 | None = None


class PartitionMembership(StrictModel):
    name: DataPartition
    ordered_row_ids: tuple[str, ...] = Field(min_length=1)
    ordered_row_ids_hash: Sha256
    count: int = Field(ge=1)
    source_revision: GitRevision
    group_assignment_hash: Sha256
    build_report: "ArtifactRef"

    @model_validator(mode="after")
    def membership_is_self_consistent(self) -> "PartitionMembership":
        if self.count != len(self.ordered_row_ids):
            raise ValueError("partition count does not match ordered row IDs")
        if self.ordered_row_ids_hash != canonical_sha256(self.ordered_row_ids):
            raise ValueError("partition row ID hash does not match ordered row IDs")
        if len(set(self.ordered_row_ids)) != len(self.ordered_row_ids):
            raise ValueError("partition contains duplicate row IDs")
        return self


class SealedSourceRef(StrictModel):
    repository: str
    revision: str
    manifest_hash: Sha256


class DataManifest(StrictModel):
    source: str
    revision: GitRevision
    approved_row_ids: tuple[str, ...] = Field(min_length=1)
    exclusions: dict[str, str]
    partitions: tuple[PartitionMembership, ...]
    license_policy_version: str
    duplicate_check_version: str
    sealed_sources: tuple[SealedSourceRef, ...]

    @model_validator(mode="after")
    def validate_membership(self) -> "DataManifest":
        required = {
            DataPartition.TRAIN,
            DataPartition.EVALUATOR_DEVELOPMENT,
            DataPartition.PROMOTION_SAME_FAMILY,
            DataPartition.PROMOTION_UNSEEN_FAMILY,
        }
        names = [partition.name for partition in self.partitions]
        if set(names) != required or len(names) != len(required):
            raise ValueError("data manifest must contain each internal partition exactly once")
        all_partition_ids = [
            row for partition in self.partitions for row in partition.ordered_row_ids
        ]
        if len(set(all_partition_ids)) != len(all_partition_ids):
            raise ValueError("row IDs cross partition boundaries")
        if set(all_partition_ids) != set(self.approved_row_ids):
            raise ValueError("partitions must be exhaustive over approved row IDs")
        return self


class PromotionLimits(StrictModel):
    matched_requests_per_arm: int = Field(default=256, ge=1)
    output_rate_lcb_minimum: float = 0.03
    p95_latency_upper_regression_maximum: float = 0.01
    p99_latency_upper_regression_maximum: float = 0.02
    cold_loads_required: int = Field(default=3, ge=1)
    bootstrap_resamples: int = Field(default=10_000, ge=100)
    bootstrap_seed: int = 20260812
    required_concurrencies: tuple[int, ...] = (1, 4)

    @model_validator(mode="after")
    def concurrencies_are_unique(self) -> "PromotionLimits":
        if not self.required_concurrencies:
            raise ValueError("at least one required concurrency is necessary")
        if any(value < 1 for value in self.required_concurrencies):
            raise ValueError("required concurrencies must be positive")
        if len(set(self.required_concurrencies)) != len(self.required_concurrencies):
            raise ValueError("required concurrencies must be unique")
        return self


class HumanApproval(StrictModel):
    payload_hash: Sha256
    signature: str
    public_key: str
    signer: str
    algorithm: Literal["ssh-ed25519"] = "ssh-ed25519"
    signed_at: datetime


class EvaluatorBundle(StrictModel):
    version: str
    binary_check_versions: list[str]
    judge_prompt_versions: list[str]
    physical_test_version: str
    workload_weights: dict[str, float]
    promotion_limits: PromotionLimits
    coverage_floors: dict[str, int]
    approval: HumanApproval | None = None


class TrainingRule(StrictModel):
    trainer: Literal["specforge"]
    mode: Literal["online_dflash"]
    learning_rate: float = Field(gt=0)
    batch_size: int = Field(ge=1)
    update_every_traces: int = Field(ge=1)
    maximum_policy_lag: int = Field(ge=0, le=1)
    loss_selection: Literal["all_positions"] = "all_positions"


class StopRules(StrictModel):
    maximum_updates: int = Field(ge=1)
    stop_on_parity_mismatch: bool = True
    stop_on_missing_observability: bool = True
    stop_on_artifact_failure: bool = True


class ExperimentSpec(StrictModel):
    schema_version: Literal["1"]
    name: str
    target: ModelRef
    drafter: ModelRef
    training: TrainingRule
    runtime: RuntimeRef
    data: DataManifest
    evaluator: EvaluatorBundle
    seeds: dict[str, int]
    stop_rules: StopRules
    initial_mode: Literal[SystemMode.OFF] = SystemMode.OFF

    def effective_payload(self) -> dict[str, Any]:
        return self.model_dump(
            mode="json",
            exclude_computed_fields=True,
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def run_id(self) -> str:
        return canonical_sha256(self.effective_payload())


class ActivatedExperiment(StrictModel):
    spec: ExperimentSpec
    approval: HumanApproval

    @model_validator(mode="after")
    def signature_targets_spec(self) -> "ActivatedExperiment":
        if self.approval.payload_hash != self.spec.run_id:
            raise ValueError("approval does not target the experiment spec")
        return self


class DraftModelRef(StrictModel):
    repository: str
    revision: GitRevision
    expected_precision: str


class DraftRuntimeRef(StrictModel):
    sglang_revision: GitRevision
    specforge_revision: GitRevision
    requested_hardware: HardwareRequest


class DraftDataManifest(StrictModel):
    source: str
    revision: GitRevision
    split_ratios: dict[DataPartition, float]
    license_policy_version: str
    duplicate_check_version: str
    sealed_sources: tuple[str, ...]

    @model_validator(mode="after")
    def ratios_sum_to_one(self) -> "DraftDataManifest":
        if abs(sum(self.split_ratios.values()) - 1.0) > 1e-9:
            raise ValueError("draft split ratios must sum to one")
        return self


class ExperimentDraft(StrictModel):
    schema_version: Literal["1"]
    name: str
    target: DraftModelRef
    drafter: DraftModelRef
    training: TrainingRule
    runtime: DraftRuntimeRef
    data: DraftDataManifest
    evaluator: EvaluatorBundle
    seeds: dict[str, int]
    stop_rules: StopRules
    initial_mode: Literal[SystemMode.OFF] = SystemMode.OFF


class ArtifactRef(StrictModel):
    content_hash: Sha256
    size: int = Field(ge=0)
    storage_path: str
    producer: str
    validation_result: Literal["pending", "valid", "invalid"]


class TraceRecord(StrictModel):
    trace_id: str
    request_id: str
    experiment_id: Sha256
    raw_request: dict[str, Any]
    target_response: str
    target_token_ids: list[int]
    active_drafter: Sha256
    parent_drafter: Sha256 | None
    timing_ns: dict[str, int]
    accepted_spans: list[int]
    rejected_spans: list[int]
    health: dict[str, Any]
    provenance: dict[str, str]
    artifact_refs: list[ArtifactRef]


class TrainingBatch(StrictModel):
    batch_id: str
    trace_ids: list[str]
    policy_lag: int = Field(ge=0, le=1)
    ordering_hash: Sha256
    masks_hash: Sha256
    trainer_configuration_hash: Sha256


class CandidateCheckpoint(StrictModel):
    candidate_id: Sha256
    parent_id: Sha256
    optimizer: ArtifactRef
    random_state: ArtifactRef
    training_cursor: str
    weights: list[ArtifactRef]
    effective_configuration_hash: Sha256
    loader_verified: bool


class OracleResult(StrictModel):
    passed: bool
    version: str
    evidence: list[ArtifactRef]
    failures: list[str]

    @model_validator(mode="after")
    def passing_result_has_valid_evidence(self) -> "OracleResult":
        if self.passed and self.failures:
            raise ValueError("passing oracle cannot contain failures")
        if self.passed and any(item.validation_result != "valid" for item in self.evidence):
            raise ValueError("passing oracle requires valid evidence")
        return self


class PhysicalEvalReport(StrictModel):
    report_id: Sha256
    baseline_id: Sha256
    challenger_id: Sha256
    correctness: OracleResult
    lossless: OracleResult
    performance: OracleResult
    health: OracleResult
    environment_identity: Sha256
    profiled: bool
    evidence_level: EvidenceLevel


class JudgeCheck(StrictModel):
    check_id: str
    question: str
    required_evidence: list[str]
    version: str
    expected_answer: Literal["pass", "fail"] | None = None


class JudgeResult(StrictModel):
    check_id: str
    answer: Literal["pass", "fail", "unclear"]
    reason: str
    uncertainty: float = Field(ge=0, le=1)
    repeated_answers: tuple[str, str, str]
    source_evidence_hashes: list[Sha256]


class HumanDecision(StrictModel):
    decision_id: str
    question: str
    answer: str
    reason: str
    evidence_hashes: list[Sha256]
    reviewer: str
    approval: HumanApproval

    def approval_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"approval"})

    @model_validator(mode="after")
    def approval_targets_decision(self) -> "HumanDecision":
        if self.approval.payload_hash != canonical_sha256(self.approval_payload()):
            raise ValueError("approval does not target the human decision")
        if self.approval.signer != self.reviewer:
            raise ValueError("approval signer does not match reviewer")
        return self


class ConcurrencyResult(StrictModel):
    concurrency: int = Field(ge=1)
    matched_requests: int = Field(ge=0)
    output_rate_lcb_improvement: float
    p95_latency_upper_regression: float
    p99_latency_upper_regression: float
    parity_mismatches: int = Field(ge=0)
    correctness_regressions: int = Field(ge=0)
    cold_load_successes: int = Field(ge=0)
    health_failures: list[str]


class PairedEvaluation(StrictModel):
    results: list[ConcurrencyResult]


class PromotionEligibility(StrictModel):
    eligible: bool
    reasons: list[str]


class PromotionDecision(StrictModel):
    decision_id: str
    baseline_id: Sha256
    challenger_id: Sha256
    evaluator_bundle_hash: Sha256
    paired_statistics: PairedEvaluation
    eligible: bool
    reasons: list[str]
    rollback_parent: Sha256


class Event(StrictModel):
    sequence: int = Field(ge=1)
    experiment_id: str
    type: EventType
    idempotency_key: str
    command_hash: Sha256
    fencing_number: int = Field(ge=0)
    monotonic_ns: int = Field(ge=0)
    wall_time: datetime
    payload: dict[str, Any]


class DataRecord(StrictModel):
    record_id: str
    source: str
    source_revision: str
    partition: DataPartition
    prompt_hash: Sha256
    content_hash: Sha256


class AuthorizedPromotion(StrictModel):
    candidate_id: Sha256
    parent_id: Sha256
    experiment_id: Sha256
    evaluator_bundle_hash: Sha256
    physical_report_hash: Sha256
    artifacts_verified: Literal[True]
    loader_verified: Literal[True]
    evidence_level: Literal[EvidenceLevel.PHYSICAL_GPU]
    all_oracles_passed: Literal[True]
    paired_gate_eligible: Literal[True]
    approval: HumanApproval

    def approval_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"approval"})

    @model_validator(mode="after")
    def approval_targets_promotion(self) -> "AuthorizedPromotion":
        if self.approval.payload_hash != canonical_sha256(self.approval_payload()):
            raise ValueError("approval does not target the promotion authorization")
        return self


class ActivationAuthorization(StrictModel):
    experiment_id: Sha256
    compatibility_report_hash: Sha256
    runtime_identity_hash: Sha256
    data_manifest_hash: Sha256
    evaluator_bundle_hash: Sha256
    repository_revision: GitRevision
    compatibility_eligible: Literal[True]
    artifacts_verified: Literal[True]
    approval: HumanApproval

    def approval_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"approval"})

    @model_validator(mode="after")
    def approval_targets_activation(self) -> "ActivationAuthorization":
        if self.approval.payload_hash != canonical_sha256(self.approval_payload()):
            raise ValueError("approval does not target the activation authorization")
        return self


class AuthorizedBootstrap(StrictModel):
    experiment_id: Sha256
    drafter_id: Sha256
    parent_id: Sha256
    artifact_manifest_hash: Sha256
    activation_hash: Sha256
    compatibility_report_hash: Sha256
    loader_verified: Literal[True]
    approval: HumanApproval

    def approval_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"approval"})

    @model_validator(mode="after")
    def approval_targets_bootstrap(self) -> "AuthorizedBootstrap":
        if self.approval.payload_hash != canonical_sha256(self.approval_payload()):
            raise ValueError("approval does not target the bootstrap authorization")
        return self
