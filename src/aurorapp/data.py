import ast
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import polars as pl
from huggingface_hub import hf_hub_download
from pydantic import Field

from aurorapp.canonical import canonical_sha256, file_sha256
from aurorapp.models import DataPartition, StrictModel

KERNELBOOK_REPOSITORY = "GPUMODE/KernelBook"
KERNELBOOK_REVISION = "b76504d85f7f14ef4b1fad81f136f638f2ce625b"
KERNELBOOK_FILE = "dataset_permissive.parquet"
KERNELBOOK_FILE_SHA256 = "64af2baa3c9a835dac85a5d0c772cc698610bccd5c63c44a56ebe0afaf5d9ed2"

PERMISSIVE_LICENSES = {
    "Apache-2.0",
    "MIT",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "ISC",
}


class KernelBookRow(StrictModel):
    uuid: int
    repo_name: str
    sha: str
    licenses: tuple[str, ...] = Field(min_length=1)
    entry_point: str
    python_code: str
    triton_code: str
    repo_link: str


class PartitionPlan(StrictModel):
    source_revision: str
    seed: int
    membership: dict[DataPartition, tuple[str, ...]]
    quarantine: dict[str, str]
    row_fingerprints: dict[str, str]
    operation_family_hashes: dict[str, str]
    group_assignment_hash: str
    static_duplicate_check_version: str = "python-ast-opgraph-v2"
    semantic_execution_status: str = "pending-physical-gpu-job"


class _UnionFind:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            keep, merge = sorted((left_root, right_root))
            self.parent[merge] = keep


def _ast_and_operations(source: str) -> tuple[str, str]:
    tree = ast.parse(source)
    normalized = ast.dump(tree, annotate_fields=True, include_attributes=False)
    operations: list[str] = []
    node_types: Counter[str] = Counter()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Load, ast.Store, ast.Del, ast.Constant, ast.Name)):
            node_types[type(node).__name__] += 1
        if isinstance(node, ast.Call):
            operations.append(ast.dump(node.func, annotate_fields=True, include_attributes=False))
        elif isinstance(node, (ast.BinOp, ast.UnaryOp)):
            operations.append(type(node.op).__name__)
    operation_graph = {
        "operations": sorted(operations),
        "node_types": sorted(node_types.items()),
    }
    return canonical_sha256(normalized), canonical_sha256(operation_graph)


def _bucket(value: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 100


def build_partition_plan(rows: list[KernelBookRow], seed: int) -> PartitionPlan:
    accepted: dict[str, KernelBookRow] = {}
    quarantine: dict[str, str] = {}
    fingerprints: dict[str, tuple[str, str]] = {}
    row_fingerprints: dict[str, str] = {}
    for row in sorted(rows, key=lambda item: item.uuid):
        row_id = str(row.uuid)
        rejected = sorted(set(row.licenses) - PERMISSIVE_LICENSES)
        if rejected:
            quarantine[row_id] = f"license-not-approved:{','.join(rejected)}"
            continue
        if not row.repo_name or len(row.sha) != 40 or row.sha not in row.repo_link:
            quarantine[row_id] = "source-identity-invalid"
            continue
        try:
            ast_hash, operation_hash = _ast_and_operations(row.python_code)
            ast.parse(row.triton_code)
        except SyntaxError as error:
            quarantine[row_id] = f"python-parse-failed:{error.msg}"
            continue
        accepted[row_id] = row
        fingerprints[row_id] = (ast_hash, operation_hash)
        row_fingerprints[row_id] = canonical_sha256(
            {
                "source": row.repo_name,
                "revision": row.sha,
                "python_ast": ast_hash,
                "operation_graph": operation_hash,
                "entry_point": row.entry_point,
            }
        )

    union = _UnionFind(list(accepted))
    equivalence: dict[tuple[str, str], str] = {}
    for row_id, row in accepted.items():
        for key in (
            ("repo", row.repo_name),
            ("ast", fingerprints[row_id][0]),
            ("operation-family", fingerprints[row_id][1]),
        ):
            previous = equivalence.get(key)
            if previous is not None:
                union.union(previous, row_id)
            equivalence[key] = row_id

    groups: dict[str, list[str]] = defaultdict(list)
    for row_id in accepted:
        groups[union.find(row_id)].append(row_id)

    membership: dict[DataPartition, list[str]] = {
        DataPartition.TRAIN: [],
        DataPartition.EVALUATOR_DEVELOPMENT: [],
        DataPartition.PROMOTION_SAME_FAMILY: [],
        DataPartition.PROMOTION_UNSEEN_FAMILY: [],
    }
    for group_id, row_ids in sorted(groups.items()):
        if len(accepted) < 20:
            membership[DataPartition.TRAIN].extend(row_ids)
            continue
        family_hashes = sorted({fingerprints[row_id][1] for row_id in row_ids})
        family_bucket = _bucket(family_hashes[0], seed)
        group_bucket = _bucket(group_id, seed)
        if family_bucket < 10:
            partition = DataPartition.PROMOTION_UNSEEN_FAMILY
        elif group_bucket < 11:
            partition = DataPartition.EVALUATOR_DEVELOPMENT
        elif group_bucket < 22:
            partition = DataPartition.PROMOTION_SAME_FAMILY
        else:
            partition = DataPartition.TRAIN
        membership[partition].extend(row_ids)

    for row_ids in membership.values():
        row_ids.sort(key=int)
    frozen = {partition: tuple(row_ids) for partition, row_ids in membership.items()}
    assignment_hash = canonical_sha256(
        {partition.value: row_ids for partition, row_ids in frozen.items()}
    )
    return PartitionPlan(
        source_revision=KERNELBOOK_REVISION,
        seed=seed,
        membership=frozen,
        quarantine=quarantine,
        row_fingerprints=row_fingerprints,
        operation_family_hashes={
            row_id: fingerprint[1] for row_id, fingerprint in fingerprints.items()
        },
        group_assignment_hash=assignment_hash,
    )


def download_kernelbook(cache_directory: Path) -> Path:
    path = Path(
        hf_hub_download(
            KERNELBOOK_REPOSITORY,
            KERNELBOOK_FILE,
            repo_type="dataset",
            revision=KERNELBOOK_REVISION,
            cache_dir=cache_directory,
        )
    )
    actual_hash = file_sha256(path)
    if actual_hash != KERNELBOOK_FILE_SHA256:
        raise RuntimeError(
            f"KernelBook file hash mismatch: expected {KERNELBOOK_FILE_SHA256}, got {actual_hash}"
        )
    return path


def load_kernelbook(path: Path) -> list[KernelBookRow]:
    columns = [
        "uuid",
        "repo_name",
        "sha",
        "licenses",
        "entry_point",
        "python_code",
        "triton_code",
        "repo_link",
    ]
    frame = pl.read_parquet(path, columns=columns)
    return [KernelBookRow.model_validate(row) for row in frame.iter_rows(named=True)]


def write_partition_plan(plan: PartitionPlan, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = plan.model_dump(mode="json")
    output.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
