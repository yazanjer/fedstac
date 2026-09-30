# FedStaP v0.5.0 — client-count scaling study: technical report

Status of this document: Sections 1–3 are the pre-registration. They were committed to branch `v0.5.0`
before any run of the pilot or of the full grid was executed. Later sections are appended as results
arrive; every number in them cites the file it was re-derived from.

## 0. Disclosure of runs executed before pre-registration

Four development runs were executed on the session's CPU before this commit, solely to measure run time and
to exercise the pipeline: (i) SCAFFOLD, CICIoT2023, Arm A, K = 10, ρ = 0.5, seed 0, lr 0.02 (a re-execution
of v0.4.0 run `ce26e0f9df6d`; test macro-F1 0.6714 on CPU against 0.6642 on GPU); (ii) SCAFFOLD,
CICIoT2023, Arm B, K = 200, cohort 10, seed 0, lr 0.02, 50 rounds; (iii) two 5-round smoke runs. A 3-round
synthetic check of the tuning scheduler used learning rates outside the pre-registered grid. No FedStaP,
FedAvg or StatAvg result at the pre-registered configuration and 50 rounds was produced, and no gap
between methods was computed. Runs (i)–(iii) are not part of any analysis below.

## 1. Question and claim under test

The v0.4.0 sensitivity analysis (three seeds, Arm A only) showed SCAFFOLD ahead of FedStaP at K ∈ {10, 20}
and FedStaP ahead at K ∈ {50, 100} on both datasets. In Arm A, increasing K simultaneously (a) increases the
number of clients, (b) decreases the number of samples per client and hence the number of local steps per
round (CICIoT2023: about 275, 137, 55, 27, 14 steps for K = 10, 20, 50, 100, 200; Edge-IIoTset: about 171,
85, 34, 17, 9), and (c) under a fixed cohort size, increases the interval between successive
participations of a client. The study separates these factors.

Claim under test (brief §2): in the cross-device regime a stateless single-upload method (FedStaP)
overtakes stateful drift correction (SCAFFOLD) and the advantage grows with K; the effect is not an
artefact of client size; it is explained by the degradation of control-variate estimates.

Competing explanations, each with a distinct predicted signature:

| Hypothesis | Mechanism | Arm A | Arm B (steps fixed) | Cohort vs ρ in Arm B | Arm C (more epochs at K = 200) |
|---|---|---|---|---|---|
| H-size | client dataset size | slope > 0 | slope ≈ 0 | no difference | — |
| H-steps | few local steps → noisy variates | slope > 0 | no trend (gap level, SCAFFOLD weak at all K) | no difference | SCAFFOLD recovers as E grows |
| H-stale | rare participation → stale variates | slope > 0 | slope ≈ 0 under ρ = 0.5 | slope larger under cohort | no recovery required |
| H-count | number of clients per se | slope > 0 | slope > 0 under ρ = 0.5 | — | — |

These signatures are not mutually exclusive; the analysis reports each test and does not select a single
hypothesis unless the pattern is unambiguous.

## 2. Design

- Datasets: CICIoT2023 subset mirror (1,005,585 rows, 46 features, 34 classes) and Edge-IIoTset DNN subset
  (624,105 rows, 95 features, 15 classes); prepared files verified by SHA-256 against `meta.json` before
  every execution. Fine labels. Dirichlet α = 0.1.
- Methods: FedAvg, StatAvg, SCAFFOLD (option II, plain SGD), FedStaP (SFS + PCL, τ = 1.0). Model, local
  epochs (1), batch size (256), rounds (50), evaluation cadence and model selection (best validation
  macro-F1 round) as in v0.4.0.
- K ∈ {10, 20, 50, 100, 200}.
- Arm A (fixed total data): the full pool is partitioned over K clients (v0.4.0 partitioner, minimum 50
  samples per client).
- Arm B (fixed samples per client): n_k = ⌊N / 200⌋ (CICIoT2023 5,027; Edge-IIoTset 3,120) samples per
  client (train + validation + test) for every K. The K · n_k samples are a class-stratified subsample
  of the pool, nested across K for a given seed; the partition draws the same Dir(α) class proportions
  as Arm A and rescales them by iterative proportional fitting to equal client sizes. Local steps per round
  are therefore constant across K (about 14 and 9). The achieved label skew (mean total-variation distance
  of client label distributions to the pooled one) is recorded for every run; for seed 0 it is 0.69–0.77
  in both arms and both datasets.
- Participation: ρ = 0.5 (fraction), and a fixed cohort of 10 clients per round. Configurations in which
  the two coincide (K = 20) are executed once.
- Seeds 0–9 in every cell that carries a claim (main block and Arm C). A two-sided exact Wilcoxon
  signed-rank test on 5 pairs cannot reach p < 0.0625, which is why the brief's five seeds were raised to
  ten with the principal's approval (30 September 2026).
- Tuning: learning rate per (dataset, arm, K, method) on seed 100, ρ = 0.5, grid {0.01, 0.02, 0.05, 0.1,
  0.2}, extended by {0.005, 0.002, 0.001} or {0.5, 1.0} one value at a time while an edge value wins;
  selection by best validation macro-F1. Cohort-mode, Arm C, α = 0.5 and 150-round cells inherit the lr of
  the corresponding ρ = 0.5, α = 0.1, E = 1 cell. SCAFFOLD + SFS + PCL inherits SCAFFOLD's lr. Any cell
  whose selection remains at an edge after extension is flagged in `index/scaling_tuned.json` and reported.
- Additional blocks: Arm C — local epochs E ∈ {2, 5} at K ∈ {20, 200}, Arm B, ρ = 0.5, SCAFFOLD and
  FedStaP, seeds 0–9. Fairness control — SCAFFOLD + SFS + PCL, Arms A and B, ρ = 0.5, seeds 0–4.
  α = 0.5 — SCAFFOLD and FedStaP, Arm B, ρ = 0.5, seeds 0–4. Round budget — 150 rounds at K = 200, cohort
  10, both arms, SCAFFOLD and FedStaP, seeds 0–4. Five-seed blocks are reported as directions only.
- Instrumentation (read-only; training verified bit-identical with it on and off): per round, mean local
  steps of the sampled clients (all methods); every 5 rounds, gradient dissimilarity at the global model
  and, for SCAFFOLD, ‖c‖, mean and max ‖c_k‖, mean ‖c − c_k‖, the relative error ‖(c − c_k) − (g − g_k)‖ /
  ‖g − g_k‖ of the correction applied to each sampled client against the ideal correction estimated on a
  fixed 1,024-sample probe per client, its cosine, a probe-noise floor from two probe halves, the
  staleness (rounds since the client's variate was last refreshed) and the fraction of sampled clients
  whose variate was never refreshed.
- Execution: CPU, one thread per run, PyTorch deterministic mode. v0.4.0 runs were executed on GPU; those
  configurations are re-executed on CPU and the GPU values serve only as a reproduction check (gate:
  |Δ test macro-F1| ≤ 2 pp on two runs).
- Pilot (checkpoint B): tuning for K ∈ {10, 200} and seed 0 of the main block at K ∈ {10, 200}. The
  pilot is a feasibility and instrumentation check; its gaps are not tested and the full grid is run
  regardless of their sign unless the principal decides otherwise.

## 3. Pre-registered analysis

Primary outcome: test macro-F1 (fine labels, %) at the best-validation round. Gap
Δ(K, s) = F1_FedStaP − F1_SCAFFOLD for seed s, paired within (dataset, arm, mode, K, s): both methods see
the same partition, which is verified from the stored client sizes.

A family is one (dataset, arm, participation mode): 2 × 2 × 2 = 8 families.

1. Scaling slope (primary). For each seed, OLS of Δ(K, s) on log₂ K over the five K values; b_s is the
   per-seed slope (pp per doubling of K). Report the mean of b_s with a 95 % percentile bootstrap CI over
   seeds (10,000 resamples, generator seed 20260930) and a two-sided exact Wilcoxon signed-rank test of
   b_s against 0. Holm correction across the 8 families. A family supports "the gap grows with K" when the
   CI lies above 0 and the Holm-adjusted p < 0.05.
2. Per-K comparison. Two-sided exact Wilcoxon signed-rank test of Δ(K, s) over the 10 seeds at each K,
   Holm correction across the five K within a family. Mean Δ and its bootstrap CI reported for every K.
3. Crossover. For each seed, the smallest K at which Δ changes sign from negative to non-negative, located
   by linear interpolation in log₂ K between adjacent grid points; seeds with Δ ≥ 0 at K = 10 are recorded
   as "≤ 10", seeds with Δ < 0 at K = 200 as "> 200". Reported: median and range over seeds and the count
   in each censored category.
4. Confounder test. Per seed, b_s(Arm A) − b_s(Arm B) within (dataset, mode); Wilcoxon over seeds, Holm
   across the four (dataset, mode) pairs. H-size is supported when the Arm A slope is positive (test 1)
   and the Arm B slope is not, and this difference is significant.
5. Staleness test. Per seed, b_s(cohort) − b_s(ρ = 0.5) within (dataset, Arm B); Wilcoxon, Holm across
   the two datasets.
6. Local-steps test (Arm C). At K = 200 and K = 20 separately, per seed, OLS of Δ(E, s) on log₂ E for
   E ∈ {1, 2, 5} (E = 1 from the main block); mean slope, bootstrap CI and Wilcoxon, Holm across the four
   (dataset, K) cells. H-steps predicts a negative slope at K = 200 (SCAFFOLD recovers with more steps).
7. Mechanism association. For each SCAFFOLD run of the main block, the run-level mean over rounds ≥ 10 of
   the relative correction error, of staleness and of local steps. Spearman correlations (i) error vs
   local steps, (ii) error vs staleness, (iii) Δ of the paired seed vs error; computed within dataset,
   with a bootstrap CI that resamples seeds. The probe-noise floor is reported alongside the error so
   that estimation noise is not read as variate degradation.

Decision rules fixed in advance:
- The scaling claim of brief §2 is reported as established only if test 1 supports it in Arm B for at
  least one participation mode on both datasets. If it holds only in Arm A, the paper reports that the
  v0.4.0 crossover is a client-size effect (brief §10 fallback).
- If slopes are positive but not significant at ten seeds, seeds are added in the crossover region (brief
  §10) before the claim is weakened; the added seeds and the reason are logged in Section 4.
- The mechanism sentence of the paper names the hypothesis whose signature matches; if tests 4–6 point to
  different mechanisms on the two datasets, both are reported.
- All eight families, both arms and both modes are reported, including unfavourable ones.

Secondary, descriptive: FedAvg and StatAvg in every cell; FedStaP vs StatAvg and FedAvg gaps; SCAFFOLD +
SFS + PCL; α = 0.5; 150 rounds; worst-client F1; tail recall; upload per round.

Independent re-derivation: `evidence/rederive_scaling.py` recomputes every reported number directly from
the per-run `results.json` files without importing `scripts/analyze_scaling.py` or any `fedstac` module.

## 4. Deviations from the pre-registration

None so far.
