# ADR-0001: Record architecture decisions

## Status

Accepted (2026-08-23).

## Context

This repository is built by a single builder against a written contract, and it must remain publicly reviewable long after the build. Decisions made in commit messages or pull request threads scatter; a reviewer reconstructing why the code is shaped a certain way should not need to excavate the history.

## Decision

Every decision that shapes a component boundary, a scientific control, a pinned dependency choice, or a governance rule is recorded in `docs/adr/` as a numbered record with Status, Context, Decision, and Consequences. Records are immutable after acceptance; reversals are new records that supersede old ones.

## Consequences

Reviewers read intent in one place. The cost is a small documentation burden on every structural change, which is acceptable because structural changes in this study are rare by design.
