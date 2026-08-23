import numpy as np
import pytest
import torch

from retention_lab.data.packing import MixtureStream
from retention_lab.kd.losses import build_objective
from retention_lab.models.toy import build_toy_lm
from retention_lab.train.loop import Trainer
from retention_lab.train.schedule import lr_at
from retention_lab.utils.seeding import numpy_rng, set_global_determinism, torch_generator

BLOCK = {"vocab_size": 16, "d_model": 32, "n_layer": 1, "n_head": 2, "seq_len": 16}
TRAIN = {"steps": 6, "batch_size": 4, "lr": 0.005, "warmup_steps": 2}


def make_stream(seed=3, n_blocks=64, data_vocab=16):
    tokens = numpy_rng(seed, "loop-test").integers(0, data_vocab, size=n_blocks * 17)
    packed = tokens.reshape(n_blocks, 17)
    return MixtureStream({"corpus": packed}, {"corpus": 1.0}, seed=seed)


def make_trainer(train_cfg=None, with_teacher=True, seed=5, data_vocab=16):
    set_global_determinism(seed)
    student = build_toy_lm(BLOCK, torch_generator(seed, "student"))
    teacher = None
    objective = build_objective({"name": "lm_only"})
    if with_teacher:
        teacher = build_toy_lm({**BLOCK, "d_model": 48}, torch_generator(seed, "teacher"))
        objective = build_objective({"name": "forward_kl", "temperature": 2.0, "alpha": 0.5})
    return Trainer(
        model=student,
        objective=objective,
        stream=make_stream(data_vocab=data_vocab),
        train_cfg=train_cfg or TRAIN,
        teacher=teacher,
    )


def test_schedule_shape():
    assert lr_at(0, 1.0, 10, 100, 0.1) == pytest.approx(0.1)
    assert lr_at(9, 1.0, 10, 100, 0.1) == pytest.approx(1.0)
    assert lr_at(100, 1.0, 10, 100, 0.1) == pytest.approx(0.1)
    mid = lr_at(55, 1.0, 10, 100, 0.1)
    assert 0.1 < mid < 1.0
    with pytest.raises(ValueError):
        lr_at(0, 1.0, 100, 100, 0.1)


def test_loop_is_bit_identical_across_instantiations():
    a = make_trainer().train(4)
    b = make_trainer().train(4)
    assert [m["total"] for m in a] == [m["total"] for m in b]


def test_accumulation_matches_full_batch_within_float_tolerance():
    full = make_trainer({**TRAIN, "accum_steps": 1}).train(4)
    accum = make_trainer({**TRAIN, "accum_steps": 2}).train(4)
    for x, y in zip(full, accum, strict=True):
        assert x["total"] == pytest.approx(y["total"], rel=1e-5)


def test_accumulation_must_divide_batch():
    with pytest.raises(ValueError, match="divide"):
        make_trainer({**TRAIN, "accum_steps": 3})


def test_teacher_receives_no_gradient_and_never_moves():
    trainer = make_trainer()
    before = [p.detach().clone() for p in trainer.teacher.parameters()]
    trainer.train(3)
    for param, snapshot in zip(trainer.teacher.parameters(), before, strict=True):
        assert param.grad is None
        assert torch.equal(param, snapshot)


def test_vocab_limit_slices_before_the_objective():
    captured = {}

    class Probe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = build_objective({"name": "forward_kl", "temperature": 1.0})

        def forward(self, s, t, y, **kw):
            captured["shapes"] = (s.shape[-1], t.shape[-1])
            return self.inner(s, t, y, **kw)

    # Mirrors reality: the model's embedding table (16) is padded past the
    # tokenizer vocabulary (12), and every data token is below the limit.
    set_global_determinism(5)
    trainer = make_trainer({**TRAIN, "vocab_limit": 12}, data_vocab=12)
    trainer.objective = Probe()
    trainer.train(1)
    assert captured["shapes"] == (12, 12)


def test_tokens_seen_accounting():
    trainer = make_trainer()
    trainer.train(2)
    assert trainer.tokens_seen == 2 * TRAIN["batch_size"] * BLOCK["seq_len"]


def test_control_and_kd_share_the_loop():
    control = make_trainer(with_teacher=False).train(2)
    assert "kd" not in control[0]
    kd = make_trainer(with_teacher=True).train(2)
    assert "kd" in kd[0]


def test_stream_batches_are_reproducible_tensors():
    a = make_stream().batch(0, 4)
    b = make_stream().batch(0, 4)
    assert np.array_equal(a, b)
