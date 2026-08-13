import subprocess

from aurorapp.source_probe import (
    EXPECTED_REJECTED_FILES,
    check_patch_application,
    parse_rejected_files,
)


def test_patch_failure_must_match_exact_expected_rejected_file_set() -> None:
    diagnostic = "\n".join(
        f"error: {name}: patch does not apply" for name in EXPECTED_REJECTED_FILES
    )

    assert parse_rejected_files(diagnostic) == EXPECTED_REJECTED_FILES
    assert parse_rejected_files(diagnostic + "\nerror: extra.py: patch does not apply") != (
        EXPECTED_REJECTED_FILES
    )


def test_ported_patch_gate_executes_git_apply_against_exact_tree(tmp_path) -> None:
    repository = tmp_path / "upstream"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    source = repository / "source.py"
    source.write_text("value = 'old'\n", encoding="utf-8")
    subprocess.run(["git", "add", "source.py"], cwd=repository, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Aurora++ Test",
            "-c",
            "user.email=aurorapp@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=repository,
        check=True,
    )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    patch = tmp_path / "capture.patch"
    patch.write_text(
        """diff --git a/source.py b/source.py
index 90e19bd..f49e80e 100644
--- a/source.py
+++ b/source.py
@@ -1 +1 @@
-value = 'old'
+value = 'captured'
""",
        encoding="utf-8",
    )

    result = check_patch_application(repository, revision, patch)

    assert result.applies_cleanly is True
    assert result.exit_code == 0
    assert result.rejected_files == ()
    assert source.read_text(encoding="utf-8") == "value = 'old'\n"
