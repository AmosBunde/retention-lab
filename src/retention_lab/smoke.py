"""The progressive smoke entry point (ADR-0004), now at its M3 form.

The smoke run is a real two-phase distillation in miniature: a toy teacher
is first trained on the synthetic corpus with plain LM loss, then a smaller
toy student distills from it with the configured KD objective through the
same Trainer, stream, and objective code the real runs use. Both loss
traces must decrease, and the whole run is a pure function of the config
and the shared seed, which the determinism test asserts bit for bit.
"""

from __future__ import annotations

import argparse
import json

from retention_lab.data.packing import MixtureStream
from retention_lab.data.synthetic import synthetic_tokens
from retention_lab.kd.losses import build_objective
from retention_lab.models.toy import build_toy_lm
from retention_lab.train.loop import Trainer
from retention_lab.utils.config import config_hash, load_config
from retention_lab.utils.seeding import set_global_determinism, torch_generator


def _stream(cfg: dict, block_len: int) -> MixtureStream:
    tokens = synthetic_tokens(
        int(cfg["seed"]), int(cfg["model"]["vocab_size"]), int(cfg["data"]["n_tokens"])
    )
    n_blocks = tokens.shape[0] // block_len
    packed = tokens[: n_blocks * block_len].reshape(n_blocks, block_len).numpy()
    return MixtureStream({"corpus": packed}, {"corpus": 1.0}, seed=int(cfg["seed"]))


def run_smoke(config_path: str) -> dict[str, list[float]]:
    cfg = load_config(config_path)
    seed = int(cfg["seed"])
    set_global_determinism(seed)
    block_len = int(cfg["model"]["seq_len"]) + 1
    stream = _stream(cfg, block_len)

    teacher = build_toy_lm(cfg["teacher_model"], torch_generator(seed, "teacher-init"))
    teacher_metrics = Trainer(
        model=teacher,
        objective=build_objective({"name": "lm_only"}),
        stream=stream,
        train_cfg=cfg["teacher_train"],
    ).train(int(cfg["teacher_train"]["steps"]))

    student = build_toy_lm(cfg["model"], torch_generator(seed, "model-init"))
    student_metrics = Trainer(
        model=student,
        objective=build_objective(cfg["loss"]),
        stream=stream,
        train_cfg=cfg["train"],
        teacher=teacher,
    ).train(int(cfg["train"]["steps"]))

    return {
        "teacher_loss": [m["total"] for m in teacher_metrics],
        "student_total": [m["total"] for m in student_metrics],
        "student_ce": [m["ce"] for m in student_metrics],
        "student_kd": [m["kd"] for m in student_metrics],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Retention Lab tiny KD smoke run")
    parser.add_argument("--config", required=True)
    parser.add_argument("--json", action="store_true", help="emit the traces as JSON")
    args = parser.parse_args()

    cfg = load_config(args.config)
    traces = run_smoke(args.config)
    if args.json:
        print(json.dumps({"config_hash": config_hash(cfg), "traces": traces}))
    else:
        print(f"smoke: config_hash={config_hash(cfg)[:12]}")
        for name, series in traces.items():
            print(f"smoke: {name} first={series[0]:.6f} last={series[-1]:.6f}")
    failures = [
        name
        for name in ("teacher_loss", "student_total")
        if not traces[name][-1] < traces[name][0]
    ]
    if failures:
        raise SystemExit(f"smoke: FAIL, no decrease in {failures}")
    print("smoke: OK, teacher and distilled student both learned")


if __name__ == "__main__":
    main()
