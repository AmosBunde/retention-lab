"""The real-run entry point: config to trained, scored, recorded student.

This is the command the approved GPU runs execute. It refuses to invent
anything: assets must verify, the config resolves through the audited
inheritance chain, hours are measured, cost derives from the supplied
rate, reclaims are counted by the resume machinery, and the final tracker
record carries the frozen scoreboard hash the battery ran under. Nothing
here executes without the owner-approved instance; on this CPU-only
machine the module exists to be reviewed and to serve the GPU command
line.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from retention_lab.battery.protocol import TorchCausalLM
from retention_lab.battery.score_model import score_and_record
from retention_lab.data.packing import MixtureStream
from retention_lab.kd.losses import build_objective
from retention_lab.models.pinned import HFLogitsAdapter, HFTokenizerAdapter
from retention_lab.train.checkpoint import resume_if_available, save_checkpoint
from retention_lab.train.loop import Trainer
from retention_lab.train.pack_corpus import pack_generated, pack_split
from retention_lab.utils.config import config_hash, load_config
from retention_lab.utils.seeding import set_global_determinism, stream_seed


def build_student(cfg: dict, assets_root: Path, run_seed: int, device: str):
    from transformers import AutoConfig, AutoModelForCausalLM

    model_dir = assets_root / "models" / "student-pretrained"
    if cfg["init"]["source"] == "pretrained":
        model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
    elif cfg["init"]["source"] == "scratch":
        torch.manual_seed(stream_seed(run_seed, "model-init"))
        config = AutoConfig.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_config(config)
    else:
        raise ValueError(f"unknown init source {cfg['init']['source']!r}")
    return HFLogitsAdapter(model).to(device)


def build_teacher(assets_root: Path, device: str, dtype: str):
    from transformers import AutoModelForCausalLM

    model_dir = assets_root / "models" / "teacher"
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=getattr(torch, dtype))
    return HFLogitsAdapter(model).to(device).eval()


def build_stream(cfg: dict, assets_root: Path, tokenizer, generated_dir: Path | None):
    assets = load_config("configs/assets.yaml")
    corpus_name = cfg["data"]["corpus"]
    corpus_spec = assets["corpus"][corpus_name]
    block_len = int(cfg["data"]["block_len"])
    # GPT-NeoX uses token id 0 as endoftext; fixed here rather than probed so
    # the packed cache is independent of tokenizer wrapper details.
    eos_id = 0
    packed = {
        "corpus": pack_split(
            assets_root, corpus_name, corpus_spec, tokenizer.encode, eos_id, block_len, False
        )
    }
    proportions = dict(cfg["mixture"]["sources"])
    if "teacher_generated" in proportions:
        if generated_dir is None:
            raise FileNotFoundError(
                "mixture demands teacher_generated data; pass --generated-dir"
            )
        packed["teacher_generated"] = pack_generated(
            assets_root, generated_dir, tokenizer.encode, eos_id, block_len
        )
    return MixtureStream(packed, proportions, seed=int(cfg["seed"]))


def count_reclaim(out_dir: Path, resumed: bool) -> int:
    marker = out_dir / "reclaims.txt"
    count = int(marker.read_text()) if marker.exists() else 0
    if resumed:
        count += 1
        marker.write_text(str(count))
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Train, score, and record one arm")
    parser.add_argument("--config", required=True, help="a configs/variants/*.yaml file")
    parser.add_argument("--seed", type=int, required=True, help="per-run init seed (1 or 2)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--assets-root", default="assets")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--generated-dir", default=None)
    parser.add_argument("--bf16", action="store_true", help="autocast bf16 on CUDA")
    parser.add_argument("--hourly-rate-usd", type=float, required=True)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--image-tag", required=True)
    parser.add_argument("--steps-override", type=int, default=None,
                        help="calibration segments only; the record marks the short run")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    started = time.monotonic()
    cfg = load_config(args.config)
    resolved_hash = config_hash(cfg)
    set_global_determinism(args.seed)
    assets_root = Path(args.assets_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = HFTokenizerAdapter(
        AutoTokenizer.from_pretrained(assets_root / "models" / "student-pretrained")
    )
    student = build_student(cfg, assets_root, args.seed, args.device)
    objective = build_objective(cfg["loss"])
    teacher = None
    if objective.name != "lm_only":
        teacher = build_teacher(assets_root, args.device, "float32")
    stream = build_stream(
        cfg, assets_root, tokenizer, Path(args.generated_dir) if args.generated_dir else None
    )

    trainer = Trainer(
        model=student,
        objective=objective,
        stream=stream,
        train_cfg=cfg["train"],
        teacher=teacher,
        device=args.device,
        autocast_bf16=args.bf16,
    )
    resumed_step = resume_if_available(trainer, out_dir / "ckpt")
    reclaims = count_reclaim(out_dir, resumed_step is not None)

    total_steps = args.steps_override or int(cfg["train"]["steps"])
    checkpoint_every = int(cfg["train"]["checkpoint_every"])
    metrics_path = out_dir / "metrics.jsonl"
    with open(metrics_path, "a") as metrics_file:
        while trainer.step < total_steps:
            record = trainer.train_step()
            metrics_file.write(json.dumps(record) + "\n")
            if trainer.step % 50 == 0:
                metrics_file.flush()
                print(
                    f"train: step={trainer.step}/{total_steps} "
                    f"total={record['total']:.4f} lr={record['lr']:.2e}"
                )
            if trainer.step % checkpoint_every == 0:
                save_checkpoint(trainer, out_dir / "ckpt")
    save_checkpoint(trainer, out_dir / "ckpt")

    variant = Path(args.config).stem
    run_id = f"{variant}-seed{args.seed}"
    kind = "control" if variant == "control" else "student"
    lm = TorchCausalLM(student, tokenizer, max_len=2048, device=args.device)
    battery_cfg = load_config("configs/battery/battery.yaml")
    meta = {
        "run_id": run_id,
        "kind": kind,
        "model": f"{cfg['model']['config_repo']} ({cfg['init']['source']})",
        "revision": cfg["model"]["revision"],
        "hourly_rate_usd": args.hourly_rate_usd,
        "instance": args.instance,
        "image_tag": args.image_tag,
    }
    record = score_and_record(
        lm, tokenizer, tokenizer.vocab_size, battery_cfg, "full", meta
    )
    # The record's measured hours cover only the scoring wall time; training
    # dominates, so replace with the whole-process measurement and re-derive.
    total_hours = round((time.monotonic() - started) / 3600.0, 4)
    from dataclasses import asdict

    payload = {
        **asdict(record),
        "gpu_hours": total_hours,
        "cost_usd": round(total_hours * args.hourly_rate_usd, 4),
        "config_hash": resolved_hash,
        "seed": args.seed,
        "tokens_trained": trainer.tokens_seen,
        "reclaims": reclaims,
        "steps_override": args.steps_override,
    }
    out_path = out_dir / f"{run_id}.json"
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"run_training: wrote {out_path} (gpu_hours={total_hours}, reclaims={reclaims})")


if __name__ == "__main__":
    main()
