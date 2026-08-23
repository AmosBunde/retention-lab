import pytest
import torch

from retention_lab.battery.protocol import ByteTokenizer, TorchCausalLM
from retention_lab.deploy.int8 import measure_latency, quantize_dynamic_int8
from retention_lab.models.toy import build_toy_lm
from retention_lab.utils.seeding import torch_generator

BLOCK = {"vocab_size": 32, "d_model": 32, "n_layer": 1, "n_head": 2, "seq_len": 64}


def toy():
    return build_toy_lm(BLOCK, torch_generator(9, "int8-test"))


def test_quantization_replaces_linears_and_still_scores():
    model = toy()
    quantized = quantize_dynamic_int8(model)
    kinds = {type(m).__name__ for m in quantized.modules()}
    assert "LinearPackedParams" in kinds, "no dynamically quantized linear found"
    assert not any(type(m) is torch.nn.Linear for m in quantized.modules()), (
        "a float Linear survived quantization"
    )
    lm = TorchCausalLM(quantized, ByteTokenizer(32), max_len=64)
    result = lm.loglikelihood("the corpus", " slice")
    assert result.n_tokens == 6
    assert result.logprob < 0


def test_quantized_scores_stay_close_on_a_toy_model():
    model = toy()
    lm_fp = TorchCausalLM(model, ByteTokenizer(32), max_len=64)
    fp = lm_fp.loglikelihood("a deterministic context", " tail").logprob
    lm_q = TorchCausalLM(quantize_dynamic_int8(toy()), ByteTokenizer(32), max_len=64)
    q = lm_q.loglikelihood("a deterministic context", " tail").logprob
    assert q == pytest.approx(fp, rel=0.05)


def test_latency_protocol_shape_and_determinism_of_fields():
    model = quantize_dynamic_int8(toy())
    report = measure_latency(model, prompt_tokens=[1, 2, 3], generate_tokens=4, repeats=2, warmup=1)
    assert set(report) == {
        "median_seconds",
        "tokens_per_second",
        "resident_memory_mb",
        "prompt_tokens",
        "generate_tokens",
        "repeats",
    }
    assert report["tokens_per_second"] > 0
    assert report["resident_memory_mb"] > 0
    assert report["prompt_tokens"] == 3


def test_quantization_refuses_non_cpu_models():
    model = toy()
    if torch.cuda.is_available():
        model = model.cuda()
        with pytest.raises(ValueError, match="CPU"):
            quantize_dynamic_int8(model)
    else:
        quantize_dynamic_int8(model)  # on CPU machines this simply succeeds
