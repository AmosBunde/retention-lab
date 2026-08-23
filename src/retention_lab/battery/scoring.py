"""Task scoring: deterministic, per-item, no sampling anywhere.

Three scorers cover the whole battery:

- multiple choice: pick the choice with the highest byte-length-normalized
  log-likelihood (the acc_norm convention), so long answers are not punished
  for having more tokens;
- greedy continuation: correct when every continuation token is the model's
  greedy choice (the LAMBADA convention);
- bits per byte: total negative log-likelihood over documents divided by
  their UTF-8 byte count, in bits.

Every scorer returns per-item values alongside the aggregate, because the
noise-band bootstrap resamples items, not summaries.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from retention_lab.battery.protocol import LM, Tokenizer


@dataclass(frozen=True)
class MCItem:
    context: str
    choices: tuple[str, ...]
    gold: int


@dataclass(frozen=True)
class ContextChoiceItem:
    """Choice-dependent contexts with one shared continuation (Winogrande)."""

    contexts: tuple[str, ...]
    continuation: str
    gold: int


@dataclass(frozen=True)
class GreedyItem:
    context: str
    continuation: str


@dataclass(frozen=True)
class TaskResult:
    task: str
    metric: str  # "accuracy" or "bpb"
    value: float
    chance: float  # score of a chance-level model on the same items
    n_items: int
    per_item: tuple[float, ...]


def score_multiple_choice(lm: LM, task: str, items: Sequence[MCItem]) -> TaskResult:
    per_item = []
    for item in items:
        if not 0 <= item.gold < len(item.choices):
            raise ValueError(f"{task}: gold index {item.gold} out of range")
        normed = []
        for choice in item.choices:
            n_bytes = len(choice.encode("utf-8"))
            if n_bytes == 0:
                raise ValueError(f"{task}: empty choice")
            result = lm.loglikelihood(item.context, choice)
            normed.append(result.logprob / n_bytes)
        best = max(range(len(normed)), key=lambda i: (normed[i], -i))
        per_item.append(1.0 if best == item.gold else 0.0)
    chance = _mean([1.0 / len(item.choices) for item in items])
    return TaskResult(task, "accuracy", _mean(per_item), chance, len(per_item), tuple(per_item))


def score_context_choice(
    lm: LM, task: str, items: Sequence[ContextChoiceItem]
) -> TaskResult:
    """Pick the context under which the shared continuation is most likely.

    No length normalization is needed: the continuation is identical across
    the candidate contexts, so raw log-likelihoods are directly comparable.
    """
    per_item = []
    for item in items:
        if not 0 <= item.gold < len(item.contexts):
            raise ValueError(f"{task}: gold index {item.gold} out of range")
        scores = [lm.loglikelihood(ctx, item.continuation).logprob for ctx in item.contexts]
        best = max(range(len(scores)), key=lambda i: (scores[i], -i))
        per_item.append(1.0 if best == item.gold else 0.0)
    chance = _mean([1.0 / len(item.contexts) for item in items])
    return TaskResult(task, "accuracy", _mean(per_item), chance, len(per_item), tuple(per_item))


def score_greedy(lm: LM, task: str, items: Sequence[GreedyItem]) -> TaskResult:
    """Exact-continuation scoring; a chance-level model scores zero."""
    per_item = []
    for item in items:
        result = lm.loglikelihood(item.context, item.continuation)
        per_item.append(1.0 if result.greedy else 0.0)
    return TaskResult(task, "accuracy", _mean(per_item), 0.0, len(per_item), tuple(per_item))


def score_bits_per_byte(
    lm: LM, task: str, docs: Sequence[str], tokenizer: Tokenizer, vocab_size: int
) -> TaskResult:
    """Bits per byte over documents; per-item values are per-document BPB.

    The chance level is the bits per byte of a uniform distribution over the
    vocabulary, computed with exactly the same token accounting as the model
    score, so chance and value are comparable term for term.
    """
    per_item = []
    total_nll = 0.0
    total_bytes = 0
    total_scored_tokens = 0
    for doc in docs:
        tokens = tokenizer.encode(doc)
        if len(tokens) < 2:
            raise ValueError(f"{task}: document too short to score")
        nll = -lm.loglikelihood_tokens(tokens)
        n_bytes = len(doc.encode("utf-8"))
        per_item.append(nll / (math.log(2) * n_bytes))
        total_nll += nll
        total_bytes += n_bytes
        total_scored_tokens += len(tokens) - 1
    value = total_nll / (math.log(2) * total_bytes)
    chance = (total_scored_tokens * math.log2(vocab_size)) / total_bytes
    return TaskResult(task, "bpb", value, chance, len(per_item), tuple(per_item))


def group_by_capability(
    results: Sequence[TaskResult], capabilities: dict[str, list[str]]
) -> dict[str, list[TaskResult]]:
    """Group task results by the capability map from the battery config."""
    by_name = {r.task: r for r in results}
    grouped: dict[str, list[TaskResult]] = {}
    for capability, tasks in capabilities.items():
        missing = [t for t in tasks if t not in by_name]
        if missing:
            raise ValueError(f"capability {capability!r} missing task results: {missing}")
        grouped[capability] = [by_name[t] for t in tasks]
    return grouped


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average zero items")
    return sum(values) / len(values)
