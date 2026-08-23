# BREAKDOWN: issue-level plan

This file is the contract for how the work in [`README.md`](README.md) is decomposed into milestones, issues, branches, and pull requests. One issue equals one branch equals one pull request; a pull request past roughly 400 reviewable lines is split into two issues rather than merged large.

## Labels

| Label | Meaning |
|---|---|
| `type:infra` | Infrastructure, tooling, training code |
| `type:experiment` | Pre-registered experiment with hypothesis and GPU runs |
| `type:data` | Datasets, assets, licenses, pinning |
| `type:docs` | Documentation, diagrams, report pages |
| `needs-gpu` | Requires an approved run on the rented GPU |
| `negative-result` | Hypothesis did not hold; documented as such |
| `frozen-battery` | Touches the frozen scoreboard; requires a raised defect issue, never a silent patch |
| `blocked` | Waiting on owner approval or an external dependency |

## Milestones

| Milestone | Gate to exit |
|---|---|
| M0 Scaffold | `make quickstart` green in CI on a clean runner; CUDA image published to GHCR |
| M1 Retention scoreboard | Battery, metric, band policy frozen by hash; teacher denominators stored from one approved GPU run; freeze check in CI |
| M2 Data and assets | Pinned revisions with licenses and sha256 manifests; deterministic packing tested |
| M3 Distillation core | KD loss zoo behind one interface; kill-and-resume test green; control trained at two seeds and scored |
| M4 Experiment series | Four one-variable experiments at two seeds each, merged with band-verdict tables |
| M5 Handoff and report | INT8 pass, tradeoff table, final report with spend against budget |

## Issue conventions

- Infrastructure and data issues carry **Problem, Proposal, Acceptance criteria**.
- Experiment issues carry **Hypothesis, Design, Acceptance criteria, Risks**, with the hypothesis stated in the issue before any run launches.
- Branches are created with `gh issue develop <n> --name <type>/<n>-<slug> --checkout`.
- Commits are conventional, why-first, with a `Refs #<n>` footer and never a closing keyword; the pull request body carries `Closes #<n>`.
- Every issue receives a closing comment after merge stating the outcome (for experiments: whether the hypothesis held).

## M0 Scaffold

### I-01 Bootstrap: technical specification, issue-level plan, ADR directory, governance (`type:docs`)

README.md as full specification, BREAKDOWN.md as this plan, `docs/adr/` with the load-bearing decisions, `docs/architecture.html`, LICENSE, .gitignore. Acceptance: documents merged through a linked pull request; no contraction, em dash, or unfilled placeholder in any file.

### I-02 Package scaffold, Makefile, quickstart tiny path (`type:infra`)

Problem: nothing is installable or runnable. Proposal: `retention_lab` package under `src/`, `pyproject.toml` with pinned tooling, `uv.lock`, Makefile targets (`quickstart`, `test`, `lint`, `smoke`), `configs/tiny.yaml`, a bundled synthetic corpus generator for the tiny path, seeding and config-inheritance utilities with unit tests. Acceptance: `make quickstart` succeeds on a clean CPU machine; tiny path runs end to end in minutes; tests and lint green.

### I-03 CI: lint, tests, and the progressive smoke run on every pull request (`type:infra`)

Problem: nothing enforces the quickstart contract. Proposal: `.github/workflows/ci.yml` running `make quickstart` on ubuntu-latest for every pull request and push to main; smoke scope is progressive per ADR-0004 (M0 tiny path, M1 adds battery slice and freeze check, M3 upgrades to tiny KD run). Acceptance: CI green on this pull request; branch protection expectation documented.

### I-04 CUDA training image published to GHCR (`type:infra`)

Problem: GPU runs need a pinned, reproducible environment. Proposal: `docker/Dockerfile.cuda` with pinned CUDA, Python, and locked dependencies; `.github/workflows/cuda-image.yml` building and pushing `ghcr.io/amosbunde/retention-lab-cuda` on changes to the image inputs; image tag recorded in tracker records. Acceptance: image builds in CI and is pullable from GHCR.

## M1 Retention scoreboard

### I-05 Task battery: capability groups and task registry (`type:infra`)

Problem: no frozen definition of what is measured. Proposal: `battery/` package with a task registry for SciQ, ARC-Easy, HellaSwag, PIQA, Winogrande, LAMBADA, BoolQ, and held-out BPB; zero-shot prompt formats; length-normalized log-likelihood scoring; `configs/battery/battery.yaml` as the single definition; a CPU-sized battery slice for CI. Acceptance: battery runs on a toy model on CPU; per-task and per-capability scores reproducible bit for bit across two invocations.

### I-06 Retention metric with control attribution and chance adjustment (`type:infra`)

Problem: scores without denominators are not retention. Proposal: implement `R_c` and `A_c` exactly as specified in README section 3, including the BPB orientation; table renderer that never prints retention without the control columns. Acceptance: unit tests cover accuracy and BPB orientations, above-teacher and below-chance cases; renderer refuses a table missing control data.

### I-07 Noise-band policy: pooled seed variance plus bootstrap floor (`type:infra`)

Problem: seed noise can masquerade as effect. Proposal: implement the band exactly as README section 3.4 (pooled paired seed differences, `1.96 * sqrt(2) * sigma_c`, 10,000-resample bootstrap floor with a recorded bootstrap seed); verdict function returning `no effect`, `above band`, or `below band`. Acceptance: property tests on synthetic score sets; verdicts are pure functions of stored data.

### I-08 Battery freeze: content hash, CI check, and defect protocol (`type:infra`, `frozen-battery`)

Problem: a scoreboard that can drift silently proves nothing. Proposal: canonical serialization and sha256 of the battery definition (tasks, slices, prompts, metrics, bootstrap seed); `make battery-hash`; CI job failing on drift; documented defect protocol (raise an issue with `frozen-battery`, never patch silently). Acceptance: hash stable across machines; CI fails on any battery edit without an accompanying unfreeze record.

### I-09 Teacher scoring harness and denominator storage (`type:infra`, `needs-gpu`)

Problem: retention needs teacher denominators produced once under the frozen battery. Proposal: harness that loads the pinned teacher, scores the full battery, and writes the denominator record with battery hash; run config and GPU-hour estimate posted for owner approval; the real run executes only after approval. Acceptance: denominator record in `tracker/runs/` from the real run; CPU tiny-model path covered by tests before the GPU run.

### I-10 Freeze commit and scoreboard documentation (`type:docs`, `frozen-battery`)

Problem: the freeze must be a visible, referencable event. Proposal: record the frozen battery hash in the repository, update README and `docs/architecture.html` freeze boundary, document the frozen values and the defect protocol. Acceptance: freeze hash committed and checked in CI; M1 closes only after the teacher denominator record exists.

## M2 Data and assets

### I-11 Asset pinning: teacher and student revisions with licenses (`type:data`)

Problem: unpinned assets make the study unreproducible. Proposal: `configs/assets.yaml` with exact Hugging Face revision SHAs for pythia-1.4b and pythia-160m, license strings, and download scripts verifying sha256; weights git-ignored. Acceptance: `make assets` downloads and verifies on a clean machine; no weight file tracked by git.

### I-12 Corpus slice: pinned download with hash manifest (`type:data`)

Problem: the training corpus must be exact. Proposal: FineWeb-Edu sample shard list with per-shard sha256, ODC-By 1.0 recorded, held-out evaluation slice split off deterministically for BPB. Acceptance: two independent downloads produce identical hashes; held-out slice disjoint from training tokens by construction.

### I-13 Teacher-generation pipeline (`type:data`, `needs-gpu`)

Problem: the mixture experiment needs teacher-generated data that is honestly labeled. Proposal: pipeline that samples prompts from the corpus slice and generates continuations with the pinned teacher under a fixed sampling config; output labeled `source: teacher_generated`; generation run config and GPU-hour estimate posted for approval before executing. Acceptance: generated shards carry provenance labels and hashes; tiny CPU path tested.

### I-14 Mixture configuration and deterministic packing (`type:data`)

Problem: token streams must be a pure function of config and seed. Proposal: mixture configs by source with exact proportions; packing into fixed-length sequences with a seeded, recorded permutation; data cursor addressable by step for resume. Acceptance: same config and seed produce byte-identical batches across processes and machines.

### I-15 Data determinism tests (`type:infra`)

Problem: determinism claims need adversarial tests. Proposal: tests that reorder shard arrival, vary worker counts, and restart mid-epoch, asserting identical token streams; CI-runnable on the tiny corpus. Acceptance: all determinism tests green in CI.

## M3 Distillation core

### I-16 KD objective interface and loss zoo (`type:infra`)

Problem: comparable objectives need one contract. Proposal: `KDObjective` interface consuming student outputs, teacher outputs, and batch, returning a loss dictionary; implementations for forward KL with temperature, reverse KL, logit MSE, and optional intermediate layer matching with projections; toy-model gradient tests for each. Acceptance: all objectives interchangeable behind the interface; unit tests verify limiting behaviors (temperature 1.0, identical logits give zero loss).

### I-17 Training loop: AMP, accumulation, schedule (`type:infra`)

Problem: the loop must be efficient on one GPU yet exact on CPU. Proposal: single-device loop with optional bf16 autocast, gradient accumulation, clipping, warmup-cosine schedule, step-cadence checkpointing, and tracker field emission; tiny CPU config exercised in CI. Acceptance: tiny run reproduces loss curve bit for bit across two invocations at the same seed.

### I-18 Checkpoint, resume, and the kill-and-resume test (`type:infra`)

Problem: reclaimable instances require provable resume. Proposal: checkpoints carrying model, optimizer, scheduler, RNG streams (CPU and CUDA), and data cursor; kill-and-resume test that trains N steps with a mid-run kill and asserts bit-identical continuation against an uninterrupted reference. Acceptance: the test is in CI on the tiny config and green.

### I-19 Config inheritance and the config-diff test (`type:infra`)

Problem: one variable per experiment must be machine-enforced. Proposal: variant configs inherit `baseline.yaml` and override exactly one top-level block; a resolver plus a test that fails any variant differing in zero or multiple blocks; shared seed pins data order across paired arms. Acceptance: config-diff test green and wired into CI.

### I-20 Tracker: run records with mandatory cost fields (`type:infra`)

Problem: cost honesty requires schema enforcement. Proposal: JSON schema for `tracker/runs/` records with `gpu_hours` and `cost_usd` required; validation in CI; table generator reading only tracker records. Acceptance: a record missing cost fields fails validation; results tables build from tracker files alone.

### I-21 Control student: trained at two seeds and scored (`type:experiment`, `needs-gpu`)

Hypothesis: a from-scratch pythia-160m trained on the 500M-token budget with plain LM loss lands well below the teacher on every capability, establishing non-trivial room for teacher signal. Design: two seeds, baseline data mixture, no teacher term; scored on the frozen battery; band seed component initialized from these paired runs. Acceptance: two tracker records with costs; control scores stored as the attribution reference. Risks: budget overrun if throughput is misestimated; mitigated by a short calibration segment in the first run.

## M4 Experiment series

### I-22 E1 loss design: forward KL baseline versus reverse KL (`type:experiment`, `needs-gpu`)

Hypothesis: forward KL retains broad-coverage capabilities (core-lm, comprehension) better, while reverse KL, being mode-seeking, retains at most parity on multiple-choice capabilities and loses BPB. Design: baseline arm (forward KL, T=2.0) and reverse KL arm differ only in the loss block; two seeds each; the baseline arm becomes the shared reference for E2 through E4. Acceptance: retention table with control columns and band verdicts; one results page. Risks: reverse KL instability early in training.

### I-23 E2 temperature: T=2.0 versus T=4.0 (`type:experiment`, `needs-gpu`)

Hypothesis: T=4.0 softens teacher targets enough to improve retention on commonsense capabilities for a small student, at a small cost to core-lm BPB. Design: temperature block is the single override; two seeds. Acceptance and risks: as E1.

### I-24 E3 data mixture: 0 versus 30 percent teacher-generated (`type:experiment`, `needs-gpu`)

Hypothesis: 30 percent teacher-generated continuations improve retention on recall, where teacher phrasing carries facts, with no effect elsewhere. Design: mixture block is the single override; teacher data labeled by source; two seeds. Acceptance and risks: as E1, plus provenance labels verified.

### I-25 E4 initialization: from scratch versus pretrained pythia-160m (`type:experiment`, `needs-gpu`)

Hypothesis: pretrained initialization dominates from-scratch at this token budget on every capability, and the gap exceeds the noise band everywhere except possibly recall. Design: initialization block is the single override; two seeds. Acceptance and risks: as E1.

### I-26 Experiment series rollup (`type:docs`)

Problem: per-experiment pages need a single verdict view. Proposal: `docs/results/` index aggregating all arms, every delta with its band verdict, negative results labeled; README results section replaced with the real table. Acceptance: rollup generated from tracker records only.

## M5 Handoff and report

### I-27 INT8 quantization of the best student (`type:infra`)

Problem: the deployment question is whether retention survives quantization. Proposal: PyTorch dynamic INT8 quantization of the best-attributed student; full battery rescored on the quantized model; tracker record for the evaluation. Acceptance: INT8 battery scores stored with cost fields.

### I-28 Retention-latency-memory tradeoff table (`type:docs`)

Problem: the handoff needs one honest table. Proposal: table over teacher, control, best student, and INT8 student with per-capability retention, single-stream latency, and resident memory measured under a fixed protocol. Acceptance: all numbers from real measurement records; protocol documented.

### I-29 Final report: collected log and spend against budget (`type:docs`)

Problem: the study must close with an account of what happened. Proposal: report collecting all experiment pages, negative results, band verdicts, total GPU hours and cost against the 50 to 80 hour budget, and an overrun explanation if any. Acceptance: every claim traceable to a tracker record or a merged pull request.

### I-30 Archive pass: README results table, final diagrams (`type:docs`)

Problem: the repository must be legible standing alone. Proposal: README results section carries the final table; `docs/architecture.html` and the Mermaid summaries reflect the final shape; definition-of-done checklist verified item by item. Acceptance: all definition-of-done items in the build contract check out.

## Budget plan (estimates, refined before each approval request)

| Stage | Runs | Est. GPU hours |
|---|---|---|
| Teacher battery scoring | 1 | 2 |
| Teacher generation (E3 data) | 1 | 3 |
| Control | 2 | 6 |
| E1 arms (baseline fKL, reverse KL) | 4 | 16 |
| E2 arm (T=4.0) | 2 | 8 |
| E3 arm (30 percent mixture) | 2 | 8 |
| E4 arm (pretrained init) | 2 | 8 |
| Battery scoring for all students | 13 | 4 |
| INT8 evaluation | 1 | 1 |
| Margin for reclaims and calibration | | 6 |
| **Total** | | **62 of 50 to 80** |

Every approval request restates the running total against this plan.
