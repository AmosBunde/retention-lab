"""Run the battery against a model.

The CI slice (``--slice ci``) scores a short prefix of each task's frozen
evaluation slice with the toy model and the byte tokenizer; the full slice
(``--slice full``) is what teacher, control, and student scoring use. Both
paths share every line of scoring code.
"""

from __future__ import annotations

import argparse
import json

from retention_lab.battery.protocol import LM, ByteTokenizer, Tokenizer, TorchCausalLM
from retention_lab.battery.registry import (
    BUILDERS,
    build_wikitext_docs,
    load_rows,
    select_indices,
)
from retention_lab.battery.scoring import (
    TaskResult,
    group_by_capability,
    score_bits_per_byte,
    score_context_choice,
    score_greedy,
    score_multiple_choice,
)
from retention_lab.utils.config import load_config

KIND_SCORERS = {
    "multiple-choice": score_multiple_choice,
    "context-choice": score_context_choice,
    "greedy": score_greedy,
}


def run_task(
    lm: LM, tokenizer: Tokenizer, vocab_size: int, name: str, task_cfg: dict, cap: int, seed: int
) -> TaskResult:
    rows = load_rows(task_cfg["source"])
    if task_cfg["kind"] == "bits-per-byte":
        docs = build_wikitext_docs(rows["text"], int(task_cfg["doc_chars"]))
        idx = select_indices(len(docs), cap, seed, name)
        return score_bits_per_byte(lm, name, [docs[i] for i in idx], tokenizer, vocab_size)
    builder = BUILDERS[task_cfg["builder"]]
    idx = select_indices(len(rows), cap, seed, name)
    items = [builder(rows[i], task_cfg["prompt"]) for i in idx]
    return KIND_SCORERS[task_cfg["kind"]](lm, name, items)


def run_battery(
    lm: LM, tokenizer: Tokenizer, vocab_size: int, cfg: dict, slice_name: str
) -> dict[str, list[TaskResult]]:
    seed = int(cfg["seed"])
    results = []
    for name, task_cfg in cfg["tasks"].items():
        cap = int(cfg["ci_items_per_task"]) if slice_name == "ci" else int(task_cfg["eval_cap"])
        results.append(run_task(lm, tokenizer, vocab_size, name, task_cfg, cap, seed))
    return group_by_capability(results, cfg["capabilities"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the battery")
    parser.add_argument("--config", required=True)
    parser.add_argument("--slice", choices=["ci", "full"], default="ci")
    parser.add_argument("--toy", action="store_true", help="score the CI toy model")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if not args.toy:
        raise SystemExit(
            "only the toy model path exists yet; real model scoring arrives "
            "with the teacher harness"
        )

    from retention_lab.models.toy import build_toy_lm
    from retention_lab.utils.seeding import torch_generator

    # seq_len 768: the longest battery continuation observed is ~310 bytes
    # (HellaSwag and PIQA solutions), and contexts truncate left, so 768
    # byte-tokens hold any continuation with room for conditioning context.
    block = {"vocab_size": 64, "d_model": 64, "n_layer": 2, "n_head": 2, "seq_len": 768}
    model = build_toy_lm(block, torch_generator(int(cfg["seed"]), "battery-toy"))
    tokenizer = ByteTokenizer(block["vocab_size"])
    lm = TorchCausalLM(model, tokenizer, max_len=block["seq_len"])

    grouped = run_battery(lm, tokenizer, block["vocab_size"], cfg, args.slice)
    payload = {
        cap: [
            {"task": r.task, "metric": r.metric, "value": round(r.value, 6),
             "chance": round(r.chance, 6), "n": r.n_items}
            for r in results
        ]
        for cap, results in grouped.items()
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        for cap, rows in payload.items():
            for r in rows:
                print(
                    f"battery[{cap}] {r['task']}: {r['metric']}={r['value']} "
                    f"chance={r['chance']} n={r['n']}"
                )
    print("battery: OK")


if __name__ == "__main__":
    main()
