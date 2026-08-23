"""INT8 quantization and the fixed latency-memory measurement protocol.

Dynamic INT8 quantization replaces every Linear layer's weights with int8
and quantizes activations at run time; it runs on CPU, which makes the
deployment question honest on commodity hardware. The measurement protocol
is fixed here so every row of the tradeoff table is produced identically:
single stream, fixed prompt and continuation lengths, warmup iterations
excluded, medians reported, and resident memory read from the process
after generation.
"""

from __future__ import annotations

import statistics
import time

import torch


def quantize_dynamic_int8(model: torch.nn.Module) -> torch.nn.Module:
    """Dynamic INT8 over all Linear layers; the model must live on CPU."""
    if any(p.device.type != "cpu" for p in model.parameters()):
        raise ValueError("dynamic INT8 quantization requires a CPU model")
    model.eval()
    return torch.ao.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )


def resident_memory_mb() -> float:
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    raise RuntimeError("VmRSS not found")


@torch.no_grad()
def measure_latency(
    model: torch.nn.Module,
    prompt_tokens: list[int],
    generate_tokens: int,
    repeats: int = 5,
    warmup: int = 2,
) -> dict[str, float]:
    """Greedy single-stream generation timing under the fixed protocol.

    Returns median wall seconds, tokens per second over the generated span,
    and resident memory after generation. Greedy decoding keeps the
    measurement deterministic; no KV cache is assumed, so the numbers are a
    conservative floor comparable across every model in the table.
    """
    model.eval()
    timings = []
    for iteration in range(warmup + repeats):
        tokens = list(prompt_tokens)
        start = time.monotonic()
        for _ in range(generate_tokens):
            ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
            logits = model(ids)
            tokens.append(int(logits[0, -1].argmax()))
        elapsed = time.monotonic() - start
        if iteration >= warmup:
            timings.append(elapsed)
    median = statistics.median(timings)
    return {
        "median_seconds": round(median, 4),
        "tokens_per_second": round(generate_tokens / median, 2),
        "resident_memory_mb": round(resident_memory_mb(), 1),
        "prompt_tokens": len(prompt_tokens),
        "generate_tokens": generate_tokens,
        "repeats": repeats,
    }
