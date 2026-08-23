import math

import pytest
import torch
from torch.nn import functional as F

from retention_lab.kd.losses import build_objective

B, S, V = 2, 3, 5


def logits(seed):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(B, S, V, generator=gen)


def targets():
    gen = torch.Generator().manual_seed(99)
    return torch.randint(0, V, (B, S), generator=gen)


def test_identical_logits_give_zero_kd_for_every_family():
    x = logits(1)
    for name in ("forward_kl", "reverse_kl", "logit_mse"):
        obj = build_objective({"name": name, "temperature": 2.0, "alpha": 1.0})
        parts = obj(x, x.clone(), targets())
        assert parts["kd"].item() == pytest.approx(0.0, abs=1e-6), name


def test_lm_only_is_exactly_cross_entropy_and_ignores_teacher():
    obj = build_objective({"name": "lm_only"})
    x, t = logits(1), targets()
    parts = obj(x, None, t)
    expected = F.cross_entropy(x.reshape(-1, V), t.reshape(-1))
    assert parts["total"].item() == pytest.approx(expected.item(), rel=1e-6)
    assert "kd" not in parts


def test_forward_kl_matches_hand_computation_at_t1():
    obj = build_objective({"name": "forward_kl", "temperature": 1.0, "alpha": 1.0})
    s, t = logits(1), logits(2)
    parts = obj(s, t, targets())
    p = F.softmax(t, dim=-1)
    manual = (p * (F.log_softmax(t, -1) - F.log_softmax(s, -1))).sum(-1).mean()
    assert parts["kd"].item() == pytest.approx(manual.item(), rel=1e-6)


def test_temperature_carries_t_squared_factor():
    s, t = logits(1), logits(2)
    obj_t2 = build_objective({"name": "forward_kl", "temperature": 2.0, "alpha": 1.0})
    p = F.softmax(t / 2.0, dim=-1)
    manual = (p * (F.log_softmax(t / 2.0, -1) - F.log_softmax(s / 2.0, -1))).sum(-1).mean()
    assert obj_t2(s, t, targets())["kd"].item() == pytest.approx(4.0 * manual.item(), rel=1e-6)


def test_forward_and_reverse_differ_on_asymmetric_case():
    s, t = logits(1), logits(2)
    fwd = build_objective({"name": "forward_kl", "temperature": 1.0, "alpha": 1.0})
    rev = build_objective({"name": "reverse_kl", "temperature": 1.0, "alpha": 1.0})
    a = fwd(s, t, targets())["kd"].item()
    b = rev(s, t, targets())["kd"].item()
    assert a != pytest.approx(b, rel=1e-3)
    assert a > 0 and b > 0


def test_alpha_composes_ce_and_kd():
    s, t, y = logits(1), logits(2), targets()
    obj = build_objective({"name": "logit_mse", "alpha": 0.25})
    parts = obj(s, t, y)
    expected = 0.75 * parts["ce"] + 0.25 * parts["kd"]
    assert parts["total"].item() == pytest.approx(expected.item(), rel=1e-6)


def test_layer_match_adds_trainable_projections():
    cfg = {
        "name": "forward_kl",
        "temperature": 2.0,
        "alpha": 0.5,
        "layer_match": {"weight": 0.1, "student_dim": 4, "teacher_dim": 6, "layers": [[0, 1]]},
    }
    obj = build_objective(cfg)
    assert sum(p.numel() for p in obj.parameters()) == 4 * 6 + 6
    s, t, y = logits(1), logits(2), targets()
    sh = [torch.randn(B, S, 4)]
    th = [torch.randn(B, S, 6), torch.randn(B, S, 6)]
    parts = obj(s, t, y, student_hidden=sh, teacher_hidden=th)
    assert parts["layer_match"].item() > 0
    with pytest.raises(ValueError, match="hidden states"):
        obj(s, t, y)


def test_config_validation():
    with pytest.raises(ValueError, match="unknown"):
        build_objective({"name": "soft_targets"})
    with pytest.raises(ValueError, match="temperature"):
        build_objective({"name": "forward_kl", "temperature": 0})
    with pytest.raises(ValueError, match="alpha"):
        build_objective({"name": "forward_kl", "alpha": 1.5})
    with pytest.raises(ValueError, match="teacher logits"):
        build_objective({"name": "forward_kl"})(logits(1), None, targets())


def test_gradients_flow_to_student_only_through_kd():
    s = logits(1).requires_grad_(True)
    t = logits(2).requires_grad_(True)
    obj = build_objective({"name": "forward_kl", "temperature": 2.0, "alpha": 1.0})
    obj(s, t, targets())["kd"].backward()
    assert s.grad is not None and s.grad.abs().sum() > 0
    # The teacher term receives gradient mathematically, but the training
    # loop runs the teacher under no_grad; this test documents that the
    # objective itself does not detach, so the loop's no_grad is load-bearing.
    assert t.grad is not None


def test_uniform_teacher_forward_kl_equals_ce_minus_entropy_shape():
    # Sanity anchor: against a uniform teacher, forward KL at T=1 equals
    # log(V) minus the mean student log-probability, computable by hand.
    s = logits(3)
    t = torch.zeros(B, S, V)
    obj = build_objective({"name": "forward_kl", "temperature": 1.0, "alpha": 1.0})
    kd = obj(s, t, targets())["kd"].item()
    manual = (-F.log_softmax(s, -1).mean(-1) - math.log(V)).mean()
    assert kd == pytest.approx(manual.item(), rel=1e-5)
