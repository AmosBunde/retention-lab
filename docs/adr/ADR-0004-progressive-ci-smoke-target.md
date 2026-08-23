# ADR-0004: Progressive CI smoke target

## Status

Accepted (2026-08-23).

## Context

The build contract requires two things that collide in time: CI trains a tiny knowledge-distillation smoke run on every pull request, and no knowledge-distillation code exists before the scoreboard freeze at the end of M1. A literal reading would require KD code in M0, which would gut the scoreboard-first rule.

## Decision

`make smoke` is a single stable CI entry point whose scope grows with the project. At M0 it runs the tiny end-to-end package path on `configs/tiny.yaml`. At M1 it additionally scores the CPU battery slice and verifies the freeze hash. At M3, once the distillation core exists, it becomes the tiny knowledge-distillation smoke run required by the contract and remains so for the rest of the project. The Makefile target name and the CI job never change; only the work behind them deepens.

## Consequences

CI enforces the strongest contract available at every milestone without violating the freeze ordering, and pull requests never need to rewire CI when a milestone lands. The cost is that between M0 and M3 the smoke run is weaker than its final form, which is acceptable because the gap is documented here and closed on a fixed schedule.
