# ADR-0005: The control student is trained first and is the number to beat

## Status

Accepted (2026-08-23).

## Context

A distilled student that scores well proves little by itself: a student of the same size trained on the same tokens with plain language-model loss might score just as well. Without a control, retention numbers conflate teacher signal with ordinary training.

## Decision

Before any distillation variant runs, a control student of identical architecture, data mixture, token budget, and seed schedule is trained with no teacher term, at two seeds, and scored on the frozen battery. Every result table reports retention against the teacher and delta against the control side by side, and the attribution ratio `A_c = (S_c - C_c) / (T_c - C_c)` quantifies the fraction of the teacher-control gap closed by distillation. A variant that does not beat the control is merged as a documented negative result and labeled `negative-result`.

## Consequences

Every claim of distillation benefit is causally grounded, and negative results have a first-class home instead of disappearing. The cost is two full training runs spent on a model whose purpose is to be beaten, roughly 6 GPU hours of the budget, which is the price of the attribution column.
