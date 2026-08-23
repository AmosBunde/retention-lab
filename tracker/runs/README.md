# Tracker records

One JSON file per real run. Records are schema-validated in CI (see
`src/retention_lab/tracker/schema.py`): GPU hours, cost, instance, and
image tag are mandatory on every record, and scoring records additionally
carry the model revision, the scoreboard hash they ran under, and full
per-item scores.

Provenance rule: only real executions produce records. Every record enters
this directory through a pull request that quotes the run it came from, and
results tables are generated exclusively from these files by
`make results`. Nothing in this repository simulates, mocks, or fabricates
a run, a curve, or a result.
