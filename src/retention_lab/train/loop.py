"""Single-device training loop: exact on CPU, efficient on one GPU.

The same loop trains the control and every distillation arm; the only
difference between arms is the objective built from the config, which is
what makes comparisons one-variable at the code level. The teacher forward
runs under no_grad in eval mode (the objective deliberately does not
detach, see #17), logits are sliced to the tokenizer vocabulary before the
objective so padded embedding rows cannot leak signal, and the reported
loss of an accumulated step is the mean over its equal-size micro-batches,
which equals the full-batch value up to float association.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from retention_lab.kd.losses import KDObjective
from retention_lab.train.schedule import lr_at


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        objective: KDObjective,
        stream,
        train_cfg: dict,
        teacher: torch.nn.Module | None = None,
        device: str = "cpu",
        autocast_bf16: bool = False,
    ):
        self.model = model.to(device)
        self.objective = objective.to(device)
        self.teacher = teacher.to(device).eval() if teacher is not None else None
        self.stream = stream
        self.device = device
        self.autocast_bf16 = autocast_bf16 and device == "cuda"
        self.batch_size = int(train_cfg["batch_size"])
        self.accum = int(train_cfg.get("accum_steps", 1))
        if self.batch_size % self.accum != 0:
            raise ValueError("batch_size must divide evenly into accumulation micro-batches")
        self.base_lr = float(train_cfg["lr"])
        self.warmup_steps = int(train_cfg.get("warmup_steps", 0) or 1)
        self.total_steps = int(train_cfg["steps"])
        self.min_lr_ratio = float(train_cfg.get("min_lr_ratio", 0.1))
        self.clip_norm = float(train_cfg.get("clip_norm", 1.0))
        self.vocab_limit = int(train_cfg.get("vocab_limit", 0)) or None
        params = list(self.model.parameters()) + list(self.objective.parameters())
        self.optimizer = torch.optim.AdamW(
            params,
            lr=self.base_lr,
            weight_decay=float(train_cfg.get("weight_decay", 0.1)),
            betas=(0.9, 0.95),
        )
        self.step = 0
        self.tokens_seen = 0

    def _forward_micro(self, micro: torch.Tensor) -> dict[str, torch.Tensor]:
        inputs, targets = micro[:, :-1], micro[:, 1:]
        with torch.autocast("cuda", torch.bfloat16, enabled=self.autocast_bf16):
            student_logits = self.model(inputs)
            teacher_logits = None
            if self.teacher is not None:
                with torch.no_grad():
                    teacher_logits = self.teacher(inputs)
        student_logits = student_logits.float()
        if teacher_logits is not None:
            teacher_logits = teacher_logits.float()
        if self.vocab_limit:
            student_logits = student_logits[..., : self.vocab_limit]
            if teacher_logits is not None:
                teacher_logits = teacher_logits[..., : self.vocab_limit]
        return self.objective(student_logits, teacher_logits, targets)

    def train_step(self) -> dict[str, float]:
        batch = torch.from_numpy(
            np.ascontiguousarray(self.stream.batch(self.step, self.batch_size))
        ).to(self.device)
        lr = lr_at(self.step, self.base_lr, self.warmup_steps, self.total_steps, self.min_lr_ratio)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.zero_grad(set_to_none=True)
        micros = batch.chunk(self.accum, dim=0)
        totals: dict[str, float] = {}
        for micro in micros:
            parts = self._forward_micro(micro)
            (parts["total"] / self.accum).backward()
            for key, value in parts.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach()) / self.accum
        torch.nn.utils.clip_grad_norm_(
            [p for g in self.optimizer.param_groups for p in g["params"]], self.clip_norm
        )
        self.optimizer.step()
        self.tokens_seen += batch.shape[0] * (batch.shape[1] - 1)
        self.step += 1
        return {**totals, "lr": lr, "step": self.step, "tokens_seen": self.tokens_seen}

    def train(
        self, n_steps: int, on_step: Callable[[dict[str, float]], None] | None = None
    ) -> list[dict[str, float]]:
        metrics = []
        for _ in range(n_steps):
            record = self.train_step()
            metrics.append(record)
            if on_step is not None:
                on_step(record)
        return metrics
