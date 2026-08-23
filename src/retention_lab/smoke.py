"""The progressive smoke entry point (ADR-0004).

M0 scope: build the toy model, pack the synthetic corpus deterministically,
train a few plain language-model steps on CPU, and verify the loss decreased.
The trace is a pure function of the config and the shared seed, which the
determinism test asserts bit for bit.
"""

from __future__ import annotations

import argparse
import json

import torch
from torch.nn import functional as F

from retention_lab.data.synthetic import pack_batches, synthetic_tokens
from retention_lab.models.toy import build_toy_lm
from retention_lab.utils.config import config_hash, load_config
from retention_lab.utils.seeding import set_global_determinism, torch_generator


def run_smoke(config_path: str) -> list[float]:
    cfg = load_config(config_path)
    seed = int(cfg["seed"])
    set_global_determinism(seed)

    model = build_toy_lm(cfg["model"], torch_generator(seed, "model-init"))
    stream = synthetic_tokens(
        seed, int(cfg["model"]["vocab_size"]), int(cfg["data"]["n_tokens"])
    )
    batches = pack_batches(
        stream, int(cfg["model"]["seq_len"]), int(cfg["train"]["batch_size"]), seed
    )
    steps = int(cfg["train"]["steps"])
    if len(batches) < steps:
        raise ValueError(f"corpus supplies {len(batches)} batches, config asks {steps} steps")

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]))
    losses: list[float] = []
    model.train()
    for step in range(steps):
        batch = batches[step]
        logits = model(batch[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def main() -> None:
    parser = argparse.ArgumentParser(description="Retention Lab progressive smoke run")
    parser.add_argument("--config", required=True)
    parser.add_argument("--json", action="store_true", help="emit the loss trace as JSON")
    args = parser.parse_args()

    cfg = load_config(args.config)
    losses = run_smoke(args.config)
    first, last = losses[0], losses[-1]
    if args.json:
        print(json.dumps({"config_hash": config_hash(cfg), "losses": losses}))
    else:
        print(f"smoke: config_hash={config_hash(cfg)[:12]} steps={len(losses)}")
        print(f"smoke: first_loss={first:.6f} last_loss={last:.6f}")
    if not last < first:
        raise SystemExit("smoke: FAIL, loss did not decrease")
    print("smoke: OK, loss decreased")


if __name__ == "__main__":
    main()
