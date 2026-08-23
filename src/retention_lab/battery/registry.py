"""Item builders and pinned dataset loading for the battery.

Builders are pure functions from a dataset row (a plain dict) to a scoring
item, so unit tests cover them without any network. Loading resolves each
dataset at its exact pinned revision; slice selection is a permutation prefix
seeded per task from the battery seed, and the CI slice is a prefix of the
full evaluation slice by construction.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any

from retention_lab.battery.scoring import ContextChoiceItem, GreedyItem, MCItem
from retention_lab.utils.seeding import numpy_rng

ARC_LABEL_SETS = ("ABCDEFGH", "12345678")


def build_sciq(row: dict[str, Any], prompt: str) -> MCItem:
    support = row["support"].strip()
    context = prompt.format(
        support_block=f"{support}\n" if support else "", question=row["question"]
    )
    choices = (row["distractor1"], row["distractor2"], row["distractor3"], row["correct_answer"])
    return MCItem(context, tuple(f" {c}" for c in choices), gold=3)


def build_arc(row: dict[str, Any], prompt: str) -> MCItem:
    labels = list(row["choices"]["label"])
    texts = list(row["choices"]["text"])
    gold = labels.index(row["answerKey"])
    return MCItem(
        prompt.format(question=row["question"]),
        tuple(f" {t}" for t in texts),
        gold=gold,
    )


def _hellaswag_clean(text: str) -> str:
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def build_hellaswag(row: dict[str, Any], prompt: str) -> MCItem:
    ctx = row["ctx_a"] + " " + row["ctx_b"].capitalize()
    context = _hellaswag_clean(prompt.format(activity_label=row["activity_label"], ctx=ctx))
    endings = tuple(" " + _hellaswag_clean(e) for e in row["endings"])
    return MCItem(context, endings, gold=int(row["label"]))


def build_piqa(row: dict[str, Any], prompt: str) -> MCItem:
    return MCItem(
        prompt.format(goal=row["goal"]),
        (f" {row['sol1']}", f" {row['sol2']}"),
        gold=int(row["label"]),
    )


def build_winogrande(row: dict[str, Any], prompt: str | None) -> ContextChoiceItem:
    sentence = row["sentence"]
    if sentence.count("_") != 1:
        raise ValueError(f"winogrande sentence needs exactly one blank: {sentence!r}")
    prefix, suffix = sentence.split("_")
    contexts = (prefix + row["option1"], prefix + row["option2"])
    return ContextChoiceItem(contexts, suffix, gold=int(row["answer"]) - 1)


def build_lambada(row: dict[str, Any], prompt: str | None) -> GreedyItem:
    context, _, last_word = row["text"].strip().rpartition(" ")
    if not context:
        raise ValueError("lambada text has no final-word split")
    return GreedyItem(context, f" {last_word}")


def build_boolq(row: dict[str, Any], prompt: str) -> MCItem:
    context = prompt.format(passage=row["passage"], question=row["question"])
    return MCItem(context, (" no", " yes"), gold=int(bool(row["answer"])))


BUILDERS: dict[str, Callable[..., Any]] = {
    "sciq": build_sciq,
    "arc": build_arc,
    "hellaswag": build_hellaswag,
    "piqa": build_piqa,
    "winogrande": build_winogrande,
    "lambada": build_lambada,
    "boolq": build_boolq,
}


def build_wikitext_docs(rows: Sequence[str], doc_chars: int) -> list[str]:
    """Join the line-level rows and cut fixed-size character documents."""
    text = "".join(rows)
    docs = [text[i : i + doc_chars] for i in range(0, len(text), doc_chars)]
    return [d for d in docs if len(d) >= 64]


def select_indices(n_available: int, cap: int, seed: int, task: str) -> list[int]:
    """Deterministic evaluation slice: a seeded permutation prefix.

    The CI slice takes a shorter prefix of the same permutation, so CI items
    are always a subset of the full evaluation slice.
    """
    rng = numpy_rng(seed, f"battery-slice:{task}")
    return [int(i) for i in rng.permutation(n_available)[:cap]]


def load_rows(source: dict[str, Any]) -> Any:
    """Load the pinned split for a task source; returns a datasets Dataset."""
    from datasets import load_dataset

    if "parquet_path" in source:
        url = (
            f"https://huggingface.co/datasets/{source['repo']}/resolve/"
            f"{source['revision']}/{source['parquet_path']}"
        )
        return load_dataset("parquet", data_files={"split": url}, split="split")
    return load_dataset(
        source["repo"],
        source["config"],
        split=source["split"],
        revision=source["revision"],
    )
