# ADR-0006: One variable per experiment, enforced by config inheritance

## Status

Accepted (2026-08-23).

## Context

Experiments that vary several things at once produce results nobody can attribute. Discipline by convention decays under deadline pressure; discipline must therefore live in the test suite.

## Decision

`configs/baseline.yaml` is the single parent. Every experiment variant is a file in `configs/variants/` that names its parent and overrides exactly one top-level configuration block (for example `loss`, `temperature`, `mixture`, or `init`). A resolver materializes the full config, and a config-diff test fails any variant whose resolved diff against the baseline touches zero blocks or more than one block. The shared seed pins data order, so paired arms consume identical batches, and every variant runs at two seeds minimum.

## Consequences

Attribution of an effect to its variable is mechanical, and reviewers can audit an experiment by reading one small override file. The cost is that a genuinely coupled change (for example a loss that requires a different learning rate) must be decomposed into two registered experiments or a documented joint block, which is slower and intentionally so.
