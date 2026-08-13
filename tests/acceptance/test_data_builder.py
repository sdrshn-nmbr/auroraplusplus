from aurorapp.data import KernelBookRow, build_partition_plan
from aurorapp.models import DataPartition


def row(
    row_id: int,
    repository: str,
    license_name: str = "MIT",
    python_code: str | None = None,
) -> KernelBookRow:
    return KernelBookRow(
        uuid=row_id,
        repo_name=repository,
        sha=f"{row_id:040x}",
        licenses=(license_name,),
        entry_point=f"Kernel{row_id}",
        python_code=python_code or f"def kernel_{row_id}(x):\n    return x + {row_id}\n",
        triton_code=f"def kernel_{row_id}(x):\n    return x\n",
        repo_link=f"https://github.com/{repository}/tree/{row_id:040x}",
    )


def test_partition_build_is_deterministic_and_exhaustive() -> None:
    rows = [row(index, f"org/repo-{index}") for index in range(100)]

    first = build_partition_plan(rows, seed=17)
    second = build_partition_plan(list(reversed(rows)), seed=17)

    assert first == second
    assigned = [row_id for values in first.membership.values() for row_id in values]
    assert sorted(assigned, key=int) == [str(index) for index in range(100)]
    assert len(assigned) == len(set(assigned))


def test_source_repository_never_crosses_a_partition() -> None:
    rows = [row(index, "org/shared") for index in range(10)]
    rows.extend(row(index + 10, f"org/other-{index}") for index in range(20))

    plan = build_partition_plan(rows, seed=19)

    locations = {
        partition
        for partition, row_ids in plan.membership.items()
        if any(str(index) in row_ids for index in range(10))
    }
    assert len(locations) == 1


def test_near_duplicate_ast_never_crosses_a_partition() -> None:
    equivalent = "def kernel(x):\n    return x + 1\n"
    rows = [
        row(1, "org/one", python_code=equivalent),
        row(2, "org/two", python_code=equivalent),
    ]
    rows.extend(row(index, f"org/repo-{index}") for index in range(3, 30))

    plan = build_partition_plan(rows, seed=23)

    locations = [
        partition
        for partition, row_ids in plan.membership.items()
        if "1" in row_ids or "2" in row_ids
    ]
    assert len(locations) == 1


def test_operation_family_never_crosses_unseen_family_boundary() -> None:
    rows = []
    for index in range(100):
        operation = index % 20
        source = f"def kernel_{index}(x):\n    return torch.operation_{operation}(x)\n"
        rows.append(row(index, f"org/repo-{index}", python_code=source))

    plan = build_partition_plan(rows, seed=31)
    unseen_ids = set(plan.membership[DataPartition.PROMOTION_UNSEEN_FAMILY])
    unseen_families = {plan.operation_family_hashes[row_id] for row_id in unseen_ids}
    ordinary_ids = {
        row_id
        for partition, row_ids in plan.membership.items()
        if partition is not DataPartition.PROMOTION_UNSEEN_FAMILY
        for row_id in row_ids
    }
    ordinary_families = {plan.operation_family_hashes[row_id] for row_id in ordinary_ids}

    assert unseen_families
    assert unseen_families.isdisjoint(ordinary_families)


def test_non_permissive_and_invalid_rows_are_quarantined() -> None:
    rows = [
        row(1, "org/good"),
        row(2, "org/gpl", license_name="GPL-3.0"),
        row(3, "org/broken", python_code="not valid python ("),
    ]

    plan = build_partition_plan(rows, seed=29)

    assert plan.quarantine["2"] == "license-not-approved:GPL-3.0"
    assert plan.quarantine["3"].startswith("python-parse-failed:")
    assert plan.membership[DataPartition.TRAIN] == ("1",)
