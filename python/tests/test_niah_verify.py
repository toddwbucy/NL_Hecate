"""Unit tests for NIAH verification tool — pure logic, no GPU required."""

import math
import random
import sys
import os

import pytest

import importlib.util

# Import niah_verify directly (bypasses engine/__init__.py which needs nl_hecate)
_niah_path = os.path.join(os.path.dirname(__file__), "..", "engine", "niah_verify.py")
_spec = importlib.util.spec_from_file_location("niah_verify", _niah_path,
                                                submodule_search_locations=[])
_mod = importlib.util.module_from_spec(_spec)
# Stub out nl_hecate and engine.* so the module loads without CUDA
_stub_nl = type(sys)("nl_hecate")
sys.modules["nl_hecate"] = _stub_nl

_stub_data = type(sys)("engine.data")
_stub_data.BpeTokenStream = None  # not used in pure-logic tests
sys.modules["engine"] = type(sys)("engine")
sys.modules["engine.data"] = _stub_data

_stub_tok = type(sys)("engine.tokenizer")
_stub_tok.load_tokenizer = None
sys.modules["engine.tokenizer"] = _stub_tok
_spec.loader.exec_module(_mod)

NEEDLES = _mod.NEEDLES
_logprob_at_position = _mod._logprob_at_position
_build_trial_sequence = _mod._build_trial_sequence


# ── _logprob_at_position tests ──────────────────────────────────────

class TestLogprobAtPosition:
    """Test log-softmax scoring at a specific position."""

    def test_uniform_logits(self):
        """Uniform logits → log(1/vocab) for any token."""
        vocab = 10
        logits = [0.0] * (3 * vocab)  # 3 positions, uniform
        lp = _logprob_at_position(logits, 1, 5, vocab)
        assert abs(lp - math.log(1.0 / vocab)) < 1e-6

    def test_dominant_token(self):
        """One token has much higher logit → near-zero logprob."""
        vocab = 10
        row = [0.0] * vocab
        row[3] = 100.0  # token 3 dominates
        logits = [0.0] * vocab + row + [0.0] * vocab  # 3 positions
        lp = _logprob_at_position(logits, 1, 3, vocab)
        assert lp > -0.01  # nearly 0 (probability ≈ 1.0)

    def test_suppressed_token(self):
        """One token has much lower logit → very negative logprob."""
        vocab = 10
        row = [10.0] * vocab
        row[7] = -100.0  # token 7 suppressed
        logits = row  # position 0
        lp = _logprob_at_position(logits, 0, 7, vocab)
        assert lp < -50.0

    def test_out_of_range_token(self):
        """Token ID >= vocab returns -inf."""
        vocab = 10
        logits = [1.0] * vocab
        lp = _logprob_at_position(logits, 0, 15, vocab)
        assert lp == float("-inf")

    def test_negative_token(self):
        """Negative token ID returns -inf."""
        vocab = 10
        logits = [1.0] * vocab
        lp = _logprob_at_position(logits, 0, -1, vocab)
        assert lp == float("-inf")

    def test_logprobs_sum_to_one(self):
        """All logprobs at a position should sum to ~1.0 in probability."""
        vocab = 8
        rng = random.Random(99)
        row = [rng.gauss(0, 2) for _ in range(vocab)]
        logits = row
        total_prob = sum(
            math.exp(_logprob_at_position(logits, 0, t, vocab))
            for t in range(vocab)
        )
        assert abs(total_prob - 1.0) < 1e-5

    def test_multiple_positions(self):
        """Scoring at different positions yields different results."""
        vocab = 4
        # Position 0: token 0 dominant. Position 1: token 3 dominant.
        row0 = [10.0, 0.0, 0.0, 0.0]
        row1 = [0.0, 0.0, 0.0, 10.0]
        logits = row0 + row1

        lp0_t0 = _logprob_at_position(logits, 0, 0, vocab)
        lp1_t0 = _logprob_at_position(logits, 1, 0, vocab)
        lp1_t3 = _logprob_at_position(logits, 1, 3, vocab)

        assert lp0_t0 > -0.01   # token 0 dominant at pos 0
        assert lp1_t0 < -5.0    # token 0 suppressed at pos 1
        assert lp1_t3 > -0.01   # token 3 dominant at pos 1


# ── _build_trial_sequence tests ─────────────────────────────────────

class TestBuildTrialSequence:
    """Test haystack construction with randomized needle depth."""

    def _make_corpus(self, n: int) -> list[int]:
        """Create a dummy corpus of sequential token IDs."""
        return list(range(n))

    def test_basic_construction(self):
        """Sequence is built with correct structure."""
        corpus = self._make_corpus(20000)
        needle = [9000, 9001, 9002]
        query = [8000, 8001]
        rng = random.Random(42)

        result = _build_trial_sequence(
            corpus, needle, query,
            retrieval_distance=4096,
            haystack_size=8192,
            min_prefix=512,
            min_suffix=256,
            rng=rng,
            corpus_offset=0)

        assert result is not None
        seq = result["sequence"]
        depth = result["needle_depth"]

        # Needle should appear at the correct depth
        assert seq[depth:depth + 3] == needle
        # Query should appear after distance
        query_start = depth + len(needle) + 4096
        assert seq[query_start:query_start + 2] == query
        # answer_pos is last token of query
        assert result["query_answer_pos"] == query_start + len(query) - 1

    def test_needle_depth_randomized(self):
        """Different seeds produce different needle depths."""
        corpus = self._make_corpus(20000)
        needle = [9000]
        query = [8000]

        depths = set()
        for seed in range(20):
            rng = random.Random(seed)
            result = _build_trial_sequence(
                corpus, needle, query,
                retrieval_distance=2048,
                haystack_size=8192,
                min_prefix=512,
                min_suffix=256,
                rng=rng,
                corpus_offset=0)
            assert result is not None
            depths.add(result["needle_depth"])

        # With 20 different seeds, we should get multiple distinct depths
        assert len(depths) >= 5

    def test_needle_never_at_zero(self):
        """Needle depth is always >= min_prefix."""
        corpus = self._make_corpus(20000)
        needle = [9000]
        query = [8000]

        for seed in range(50):
            rng = random.Random(seed)
            result = _build_trial_sequence(
                corpus, needle, query,
                retrieval_distance=2048,
                haystack_size=8192,
                min_prefix=512,
                min_suffix=256,
                rng=rng,
                corpus_offset=0)
            assert result is not None
            assert result["needle_depth"] >= 512

    def test_suffix_present(self):
        """Sequence extends beyond query (query is not terminal)."""
        corpus = self._make_corpus(20000)
        needle = [9000]
        query = [8000]
        rng = random.Random(42)

        result = _build_trial_sequence(
            corpus, needle, query,
            retrieval_distance=2048,
            haystack_size=8192,
            min_prefix=512,
            min_suffix=256,
            rng=rng,
            corpus_offset=0)

        assert result is not None
        seq = result["sequence"]
        qap = result["query_answer_pos"]
        # There should be tokens after the query answer position
        assert len(seq) > qap + 1

    def test_corpus_too_short(self):
        """Returns None when corpus can't satisfy the requirements."""
        corpus = self._make_corpus(100)  # way too short
        needle = [9000]
        query = [8000]
        rng = random.Random(42)

        result = _build_trial_sequence(
            corpus, needle, query,
            retrieval_distance=4096,
            haystack_size=8192,
            min_prefix=512,
            min_suffix=256,
            rng=rng,
            corpus_offset=0)

        assert result is None

    def test_haystack_too_small_for_distance(self):
        """Returns None when haystack_size < distance + overhead."""
        corpus = self._make_corpus(20000)
        needle = [9000]
        query = [8000]
        rng = random.Random(42)

        # haystack_size=1000 but distance=4096 — impossible
        result = _build_trial_sequence(
            corpus, needle, query,
            retrieval_distance=4096,
            haystack_size=1000,
            min_prefix=512,
            min_suffix=256,
            rng=rng,
            corpus_offset=0)

        assert result is None

    def test_different_corpus_offsets(self):
        """Different offsets produce sequences from different parts of corpus."""
        corpus = self._make_corpus(50000)
        needle = [9000]
        query = [8000]

        results = []
        for offset in [0, 10000, 20000]:
            rng = random.Random(42)  # same seed
            result = _build_trial_sequence(
                corpus, needle, query,
                retrieval_distance=2048,
                haystack_size=4096,
                min_prefix=256,
                min_suffix=128,
                rng=rng,
                corpus_offset=offset)
            assert result is not None
            results.append(result)

        # Prefix tokens should differ (they come from different corpus regions)
        assert results[0]["sequence"][0] != results[1]["sequence"][0]
        assert results[1]["sequence"][0] != results[2]["sequence"][0]


# ── NEEDLES validation ──────────────────────────────────────────────

class TestNeedles:
    """Validate needle definitions."""

    def test_five_needles(self):
        assert len(NEEDLES) == 5

    def test_needle_structure(self):
        for stmt, query, answer in NEEDLES:
            assert isinstance(stmt, str)
            assert isinstance(query, str)
            assert isinstance(answer, str)
            # Answer is a 4-digit number
            assert len(answer) == 4
            assert answer.isdigit()
            # Statement contains the answer
            assert answer in stmt
            # Query asks about the same concept as the statement
            # (crude check: they share at least 3 words)
            stmt_words = set(stmt.lower().split())
            query_words = set(query.lower().split())
            assert len(stmt_words & query_words) >= 3

    def test_answers_unique(self):
        answers = [a for _, _, a in NEEDLES]
        assert len(set(answers)) == len(answers)
