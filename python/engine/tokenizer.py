"""Tokenizer abstractions: byte-level and BPE."""

import os
from pathlib import Path


class ByteTokenizer:
    """Byte-level tokenizer (vocab_size=256). No external dependencies."""

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, token_ids: list[int]) -> str:
        out = []
        for b in token_ids:
            if 32 <= b < 127:
                out.append(chr(b))
            elif b == 10:
                out.append("\n")
            else:
                out.append("?")
        return "".join(out)


class BpeTokenizer:
    """BPE tokenizer loaded from a tokenizers JSON file."""

    def __init__(self, path: str):
        try:
            from tokenizers import Tokenizer
        except ImportError as e:
            raise ImportError(
                "tokenizers package is required to load BPE tokenizers; "
                "install with: pip install tokenizers"
            ) from e
        self._tok = Tokenizer.from_file(path)

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def decode(self, token_ids: list[int]) -> str:
        return self._tok.decode(token_ids, skip_special_tokens=False)


def load_tokenizer(tokenizer_path: str | None = None,
                   data_dir: str | None = None) -> ByteTokenizer | BpeTokenizer:
    """Load the appropriate tokenizer. BPE if path provided, else byte-level."""
    if tokenizer_path and os.path.exists(tokenizer_path):
        return BpeTokenizer(tokenizer_path)
    if data_dir:
        bpe_path = Path(data_dir) / "tokenizer.json"
        if bpe_path.exists():
            return BpeTokenizer(str(bpe_path))
    return ByteTokenizer()
