"""
NL-Hecate engine: shared machinery for build + generation.

All math stays in Rust. This package is pure orchestration (CS-18).
No train/eval distinction — same forward path everywhere (CS-10).
"""

from engine.config import BuildConfig, cosine_lr
from engine.tokenizer import ByteTokenizer, BpeTokenizer, load_tokenizer
from engine.data import MmapTokenStream, BpeDataLoader, load_binary_tokens, DEMO_TEXT
from engine.generation import (
    generate, generate_cached, generate_learning, IM_START, IM_END, PAD,
    chatml_encode_turn, chatml_encode_prompt,
)
from engine.evaluation import (
    evaluate, evaluate_numpy, print_level_metrics,
    eval_coherence_samples, generate_samples,
    full_snapshot, full_restore,
    probe_within_generation, probe_cross_exposure, probe_context_value,
    SAMPLE_PROMPTS, EVAL_PROMPTS,
)
from engine.logging_utils import JSONLLogger, rss_mb
from engine.loop import run_build
from engine.chat import run_chat

__all__ = [
    # config
    "BuildConfig", "cosine_lr",
    # tokenizer
    "ByteTokenizer", "BpeTokenizer", "load_tokenizer",
    # data
    "MmapTokenStream", "BpeDataLoader", "load_binary_tokens", "DEMO_TEXT",
    # generation
    "generate", "generate_cached", "generate_learning",
    "IM_START", "IM_END", "PAD",
    "chatml_encode_turn", "chatml_encode_prompt",
    # evaluation
    "evaluate", "evaluate_numpy", "print_level_metrics",
    "eval_coherence_samples", "generate_samples",
    "full_snapshot", "full_restore",
    "probe_within_generation", "probe_cross_exposure", "probe_context_value",
    "SAMPLE_PROMPTS", "EVAL_PROMPTS",
    # logging
    "JSONLLogger", "rss_mb",
    # loops
    "run_build", "run_chat",
]
