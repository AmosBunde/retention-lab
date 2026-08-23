# ADR-0003: Pythia family for teacher and students

## Status

Accepted (2026-08-23).

## Context

Logit-level distillation objectives (forward KL, reverse KL, logit MSE) require the teacher and the student to share a vocabulary, or else an alignment layer that becomes a confound. The study also needs permissive licensing, research-grade revision pinning, and model sizes that fit a 50 to 80 GPU hour budget on a single rented card.

## Decision

The teacher is EleutherAI pythia-1.4b at an exact revision pinned in `configs/assets.yaml`. Students and the control use the pythia-160m architecture. All Pythia sizes share the GPT-NeoX tokenizer, so every logit-level objective is vocabulary-aligned by construction. All models are Apache-2.0.

## Consequences

Distillation losses need no vocabulary mapping, removing a whole class of confounds, and the roughly 9x parameter gap gives retention room to vary. The cost is that the teacher is a base model of moderate ability, so the battery is restricted to zero-shot log-likelihood tasks where such models score above chance; instruction-following capabilities are out of scope for this study.
