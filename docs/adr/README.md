# Architecture decision records

Every load-bearing decision in this repository is recorded here as a numbered, immutable record. A record is never edited after acceptance; a reversal is a new record that supersedes the old one and links back to it.

Each record carries four sections: **Status** (accepted, superseded by ADR-NNNN), **Context** (the forces that made a decision necessary), **Decision** (what was decided, in the active voice), and **Consequences** (what becomes easier, what becomes harder, and what is now forbidden).

## Index

| Record | Title |
|---|---|
| [ADR-0001](ADR-0001-record-architecture-decisions.md) | Record architecture decisions |
| [ADR-0002](ADR-0002-scoreboard-frozen-before-distillation-core.md) | The scoreboard is frozen before the distillation core |
| [ADR-0003](ADR-0003-pythia-family-for-teacher-and-students.md) | Pythia family for teacher and students |
| [ADR-0004](ADR-0004-progressive-ci-smoke-target.md) | Progressive CI smoke target |
| [ADR-0005](ADR-0005-control-first-attribution.md) | The control student is trained first and is the number to beat |
| [ADR-0006](ADR-0006-one-variable-experiments-by-config-inheritance.md) | One variable per experiment, enforced by config inheritance |
| [ADR-0007](ADR-0007-file-based-run-tracker.md) | File-based run tracker committed to the repository |
