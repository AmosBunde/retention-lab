"""A minimal causal transformer language model.

This is the CPU-sized stand-in used by the tiny path, the smoke run, and the
unit tests. It exists so that every pipeline component can be exercised end to
end on a laptop; the real students are pinned Pythia architectures loaded in
later milestones. Attention is written with explicit matrix products so that
``torch.use_deterministic_algorithms(True)`` holds on CPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ToyConfig:
    vocab_size: int
    d_model: int
    n_layer: int
    n_head: int
    seq_len: int

    @classmethod
    def from_dict(cls, block: dict) -> ToyConfig:
        return cls(
            vocab_size=int(block["vocab_size"]),
            d_model=int(block["d_model"]),
            n_layer=int(block["n_layer"]),
            n_head=int(block["n_head"]),
            seq_len=int(block["seq_len"]),
        )


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ToyConfig):
        super().__init__()
        if cfg.d_model % cfg.n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        self.n_head = cfg.n_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        mask = torch.triu(torch.ones(cfg.seq_len, cfg.seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq, dim = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        shape = (bsz, seq, self.n_head, self.head_dim)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.mask[:seq, :seq], float("-inf"))
        out = torch.softmax(scores, dim=-1) @ v
        out = out.transpose(1, 2).reshape(bsz, seq, dim)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, cfg: ToyConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class ToyLM(nn.Module):
    def __init__(self, cfg: ToyConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        seq = tokens.shape[1]
        pos = torch.arange(seq, device=tokens.device)
        x = self.tok_emb(tokens) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))


def build_toy_lm(block: dict, generator: torch.Generator) -> ToyLM:
    """Construct a ToyLM with weights initialized from a dedicated stream."""
    cfg = ToyConfig.from_dict(block)
    model = ToyLM(cfg)
    with torch.no_grad():
        for param in model.parameters():
            if param.dim() >= 2:
                nn.init.normal_(param, mean=0.0, std=0.02, generator=generator)
            else:
                nn.init.zeros_(param)
    return model
