from aurorapp.source_probe import EXPECTED_REJECTED_FILES, parse_rejected_files


def test_patch_failure_must_match_exact_expected_rejected_file_set() -> None:
    diagnostic = "\n".join(
        f"error: {name}: patch does not apply" for name in EXPECTED_REJECTED_FILES
    )

    assert parse_rejected_files(diagnostic) == EXPECTED_REJECTED_FILES
    assert parse_rejected_files(diagnostic + "\nerror: extra.py: patch does not apply") != (
        EXPECTED_REJECTED_FILES
    )
