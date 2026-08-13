import argparse
import hashlib
import inspect
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from sglang.srt.managers.tp_worker import TpModelWorker
from transformers import AutoTokenizer

from aurorapp.canonical import canonical_sha256, file_sha256
from aurorapp.model_port import TokenizerTemplateIdentityResult
from aurorapp.sglang_contract import LAGUNA_TOKENIZER_FILE_HASHES

TOKENIZER_FILES = tuple(LAGUNA_TOKENIZER_FILE_HASHES)
TOKENIZER_OVERRIDE_NAMES = frozenset(
    {
        "chat_template.jinja",
        "merges.txt",
        "sentencepiece.bpe.model",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "vocab.json",
        "vocab.txt",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-repository", required=True)
    parser.add_argument("--target-revision", required=True)
    parser.add_argument("--draft-repository", required=True)
    parser.add_argument("--draft-revision", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api = HfApi()
    target_files = tuple(
        sorted(api.list_repo_files(args.target_repository, revision=args.target_revision))
    )
    draft_files = tuple(
        sorted(api.list_repo_files(args.draft_repository, revision=args.draft_revision))
    )
    missing = sorted(set(TOKENIZER_FILES) - set(target_files))
    if missing:
        raise ValueError(f"target repository is missing tokenizer files: {missing}")

    target_hashes = {
        name: file_sha256(
            Path(
                hf_hub_download(
                    args.target_repository,
                    name,
                    revision=args.target_revision,
                )
            )
        )
        for name in TOKENIZER_FILES
    }
    tokenizer = AutoTokenizer.from_pretrained(
        args.target_repository,
        revision=args.target_revision,
        trust_remote_code=True,
    )
    if not isinstance(tokenizer.chat_template, str) or not tokenizer.chat_template:
        raise ValueError("loaded target tokenizer has no chat template")
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Add two tensors with matching shapes."}],
        tokenize=False,
        add_generation_prompt=True,
    )
    rendered_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Add two tensors with matching shapes."}],
        tokenize=True,
        add_generation_prompt=True,
    )
    source_path = Path(inspect.getsourcefile(TpModelWorker) or "")
    source = source_path.read_text(encoding="utf-8")
    draft_worker_skips_tokenizer = (
        "server_args.skip_tokenizer_init or self.is_draft_worker" in source
        and "self.tokenizer = self.processor = None" in source
    )
    special_token_ids = {
        name: value
        for name in (
            "bos_token_id",
            "eos_token_id",
            "pad_token_id",
            "unk_token_id",
            "mask_token_id",
        )
        if isinstance((value := getattr(tokenizer, name, None)), int)
    }
    result = TokenizerTemplateIdentityResult(
        target_repository=args.target_repository,
        target_revision=args.target_revision,
        parent_draft_repository=args.draft_repository,
        parent_draft_revision=args.draft_revision,
        target_tokenizer_file_hashes=target_hashes,
        expected_target_tokenizer_file_hashes=LAGUNA_TOKENIZER_FILE_HASHES,
        draft_repository_files=draft_files,
        draft_tokenizer_overrides=tuple(
            sorted(name for name in draft_files if name in TOKENIZER_OVERRIDE_NAMES)
        ),
        loaded_chat_template_hash=hashlib.sha256(
            tokenizer.chat_template.encode("utf-8")
        ).hexdigest(),
        vocabulary_hash=canonical_sha256(tokenizer.get_vocab()),
        rendered_prompt_hash=hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        rendered_token_ids_hash=canonical_sha256(rendered_ids),
        vocabulary_size=tokenizer.vocab_size,
        tokenizer_length=len(tokenizer),
        special_token_ids=special_token_ids,
        runtime_tokenizer_path=args.target_repository,
        draft_worker_skips_tokenizer=draft_worker_skips_tokenizer,
        sglang_tokenizer_source_hash=file_sha256(source_path),
    )
    print(
        "AURORAPP_TOKENIZER_RESULT=" + json.dumps(result.model_dump(mode="json"), sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
