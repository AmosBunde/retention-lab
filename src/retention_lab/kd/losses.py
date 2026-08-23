"""The KD objective zoo behind one interface.

Every objective is constructed purely from the config ``loss`` block, which
is the single block the E1 experiment overrides. The total loss composes a
language-model cross-entropy with a distillation term:

    total = (1 - alpha) * ce + alpha * kd_term

with ``alpha = 0`` (or name ``lm_only``) giving the control's objective
exactly. Temperature-scaled KL terms carry the standard T-squared factor so
gradient magnitude stays comparable across temperatures, which keeps
temperature a one-variable experiment instead of a hidden learning-rate
change. Optional intermediate layer matching adds learned projections from
student to teacher width; those parameters belong to the objective module
and are trained with the student.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

LOGIT_OBJECTIVES = ("lm_only", "forward_kl", "reverse_kl", "logit_mse")


def _ce(student_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        student_logits.reshape(-1, student_logits.shape[-1]), targets.reshape(-1)
    )


def _forward_kl(student: torch.Tensor, teacher: torch.Tensor, t: float) -> torch.Tensor:
    """KL(teacher_T || student_T) * T^2, mean over token positions."""
    s = F.log_softmax(student / t, dim=-1)
    p = F.softmax(teacher / t, dim=-1)
    log_p = F.log_softmax(teacher / t, dim=-1)
    kl = (p * (log_p - s)).sum(dim=-1)
    return kl.mean() * (t * t)


def _reverse_kl(student: torch.Tensor, teacher: torch.Tensor, t: float) -> torch.Tensor:
    """KL(student_T || teacher_T) * T^2: mode-seeking direction."""
    log_q = F.log_softmax(student / t, dim=-1)
    q = log_q.exp()
    log_p = F.log_softmax(teacher / t, dim=-1)
    kl = (q * (log_q - log_p)).sum(dim=-1)
    return kl.mean() * (t * t)


def _logit_mse(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(student, teacher)


class KDObjective(nn.Module):
    """Config-built objective; the training loop knows nothing else.

    ``forward`` consumes student logits, teacher logits (``None`` for the
    control), targets, and optional hidden-state lists, returning a dict
    with ``total`` plus every component for the tracker.
    """

    def __init__(self, loss_cfg: dict):
        super().__init__()
        name = loss_cfg["name"]
        if name not in LOGIT_OBJECTIVES:
            raise ValueError(f"unknown KD objective {name!r}; known: {LOGIT_OBJECTIVES}")
        self.name = name
        self.temperature = float(loss_cfg.get("temperature", 1.0))
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        self.alpha = float(loss_cfg.get("alpha", 0.5)) if name != "lm_only" else 0.0
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")
        match_cfg = loss_cfg.get("layer_match")
        self.layer_weight = 0.0
        self.projections = None
        if match_cfg:
            self.layer_weight = float(match_cfg["weight"])
            self.projections = nn.ModuleList(
                nn.Linear(int(match_cfg["student_dim"]), int(match_cfg["teacher_dim"]))
                for _ in match_cfg["layers"]
            )
            self.match_layers = [tuple(pair) for pair in match_cfg["layers"]]

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor | None,
        targets: torch.Tensor,
        student_hidden: list[torch.Tensor] | None = None,
        teacher_hidden: list[torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        ce = _ce(student_logits, targets)
        parts: dict[str, torch.Tensor] = {"ce": ce}
        if self.name == "lm_only":
            parts["total"] = ce
            return parts
        if teacher_logits is None:
            raise ValueError(f"objective {self.name} requires teacher logits")
        if self.name == "forward_kl":
            kd = _forward_kl(student_logits, teacher_logits, self.temperature)
        elif self.name == "reverse_kl":
            kd = _reverse_kl(student_logits, teacher_logits, self.temperature)
        else:
            kd = _logit_mse(student_logits, teacher_logits)
        parts["kd"] = kd
        total = (1.0 - self.alpha) * ce + self.alpha * kd
        if self.projections is not None:
            if student_hidden is None or teacher_hidden is None:
                raise ValueError("layer matching requires hidden states from both models")
            match = student_logits.new_zeros(())
            for proj, (s_idx, t_idx) in zip(
                self.projections, self.match_layers, strict=True
            ):
                match = match + F.mse_loss(proj(student_hidden[s_idx]), teacher_hidden[t_idx])
            parts["layer_match"] = match
            total = total + self.layer_weight * match
        parts["total"] = total
        return parts


def build_objective(loss_cfg: dict) -> KDObjective:
    return KDObjective(loss_cfg)
