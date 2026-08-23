"""Score a pinned model on the battery and write a tracker record.

This is the harness the teacher denominator run uses, and later every
control and student checkpoint. It refuses to write a record without GPU
hours and cost, because budget honesty is a schema property, not a habit.
The record carries the scoreboard hash it was produced under, so any
consumer can verify the scores belong to the frozen battery.

The full-slice run on real hardware executes only after the owner approves
the posted config and hour estimate; nothing here simulates a result.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from retention_lab.battery.freeze import repo_root, scoreboard_hash
from retention_lab.battery.protocol import LM, Tokenizer, TorchCausalLM
from retention_lab.battery.run import run_battery
from retention_lab.battery.scoring import TaskResult
from retention_lab.utils.config import load_config


@dataclass(frozen=True)
class ScoreRecord:
    run_id: str
    kind: str  # "teacher" | "control" | "student" | "int8"
    model: str
    revision: str
    battery_hash: str
    slice: str
    gpu_hours: float
    cost_usd: float
    instance: str
    image_tag: str
    wall_seconds: float
    scores: dict


def results_payload(grouped: dict[str, list[TaskResult]]) -> dict:
    return {
        cap: [
            {
                "task": r.task,
                "metric": r.metric,
                "value": r.value,
                "chance": r.chance,
                "n": r.n_items,
                "per_item": list(r.per_item),
            }
            for r in results
        ]
        for cap, results in grouped.items()
    }


def score_and_record(
    lm: LM,
    tokenizer: Tokenizer,
    vocab_size: int,
    battery_cfg: dict,
    slice_name: str,
    meta: dict,
) -> ScoreRecord:
    """Score and assemble the record; hours are measured, cost is derived.

    GPU hours come from the measured wall time of the scoring itself and the
    cost from the supplied marketplace rate, so neither number can be typed
    in wrong or invented; the rate and the instance name are the only human
    inputs.
    """
    started = time.monotonic()
    grouped = run_battery(lm, tokenizer, vocab_size, battery_cfg, slice_name)
    wall = time.monotonic() - started
    gpu_hours = round(wall / 3600.0, 4)
    rate = float(meta["hourly_rate_usd"])
    return ScoreRecord(
        run_id=meta["run_id"],
        kind=meta["kind"],
        model=meta["model"],
        revision=meta["revision"],
        battery_hash=scoreboard_hash(repo_root()),
        slice=slice_name,
        gpu_hours=gpu_hours,
        cost_usd=round(gpu_hours * rate, 4),
        instance=meta["instance"],
        image_tag=meta["image_tag"],
        wall_seconds=round(wall, 3),
        scores=results_payload(grouped),
    )


def write_record(record: ScoreRecord, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(asdict(record), fh, indent=2, sort_keys=True)
        fh.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a pinned model on the battery")
    parser.add_argument("--config", default="configs/battery/battery.yaml")
    parser.add_argument("--model", required=True, help="Hugging Face repo id")
    parser.add_argument("--revision", required=True, help="full 40-hex commit SHA")
    parser.add_argument("--kind", required=True, choices=["teacher", "control", "student", "int8"])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--slice", choices=["ci", "full"], default="full")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument(
        "--hourly-rate-usd",
        type=float,
        required=True,
        help="marketplace rate of the instance; hours are measured, cost is derived",
    )
    parser.add_argument("--instance", required=True)
    parser.add_argument("--image-tag", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from retention_lab.models.pinned import load_pinned_causal_lm

    model, tokenizer, max_len = load_pinned_causal_lm(
        args.model, args.revision, device=args.device, dtype=args.dtype
    )
    lm = TorchCausalLM(model, tokenizer, max_len=max_len, device=args.device)
    battery_cfg = load_config(args.config)
    meta = {
        "run_id": args.run_id,
        "kind": args.kind,
        "model": args.model,
        "revision": args.revision,
        "hourly_rate_usd": args.hourly_rate_usd,
        "instance": args.instance,
        "image_tag": args.image_tag,
    }
    record = score_and_record(
        lm, tokenizer, tokenizer.vocab_size, battery_cfg, args.slice, meta
    )
    write_record(record, Path(args.out))
    print(
        f"score_model: wrote {args.out} (battery_hash={record.battery_hash[:12]}, "
        f"gpu_hours={record.gpu_hours}, cost_usd={record.cost_usd})"
    )


if __name__ == "__main__":
    main()
