# ADR-0002: The scoreboard is frozen before the distillation core

## Status

Accepted (2026-08-23).

## Context

A measurement instrument built after the thing it measures invites tuning the instrument to flatter the result. Distillation studies are particularly exposed: the choice of tasks, prompts, and normalization can move retention numbers by large margins.

## Decision

The task battery, the retention metric with control attribution, the teacher denominators, and the noise-band policy are built during milestone M1 and frozen by a sha256 content hash before any knowledge-distillation code exists. CI recomputes the hash on every pull request and fails on drift. A defect discovered after the freeze is raised as an issue labeled `frozen-battery` and resolved through a visible unfreeze record; it is never patched silently. The teacher is scored exactly once under the frozen battery, and stored denominators serve all later retention computations.

## Consequences

The scoreboard cannot be tuned to the results, and every retention number is comparable across the whole study. The cost is rigidity: a genuinely broken task stays visibly broken until formally unfrozen, and the milestone order forbids starting distillation work early even when it would be convenient.
