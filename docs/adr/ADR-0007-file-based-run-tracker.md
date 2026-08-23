# ADR-0007: File-based run tracker committed to the repository

## Status

Accepted (2026-08-23).

## Context

The study must record GPU hours and marketplace cost for every run, and results tables must be reproducible by anyone who clones the repository. A hosted experiment tracker adds an account dependency, an availability dependency, and an escape hatch for numbers that never faced review.

## Decision

The tracker is a directory, `tracker/runs/`, of schema-validated JSON records committed through the same pull request as the experiment they belong to. The schema requires `gpu_hours` and `cost_usd` on every record, together with the config hash, the battery hash, the seed, the image tag, and the raw scores. Results tables are generated exclusively from these records. Only real executions produce records; nothing in the repository simulates or fabricates a run, a curve, or a result.

## Consequences

Every number in every table is reviewable in the same diff that introduced it, and a clone of the repository is the complete evidentiary record. The costs are that large artifacts (checkpoints, logs) stay outside git on the run volume, referenced by hash, and that writing a record is a manual, reviewed act rather than an automatic upload, which is exactly the friction the contract wants.
