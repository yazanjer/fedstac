# FedStaP — method and experimental protocol

FedStaP (Shared feature **Sta**tistics and **P**rior-calibrated training) produces one global model. The
codebase keeps the working package name `fedstac`; configuration name of the proposed method: `fedstap`.

## Notation
K clients; client k holds a training set D_k with n_k samples and class counts n_{k,c}, c = 1..C.
The model is f_θ = h_ψ ∘ g_φ: an MLP body g_φ (BatchNorm, ReLU, dropout) and a linear head
h_ψ with class rows (w_c, b_c). Local prior π_{k,c} = (n_{k,c} + 1) / (n_k + C).

## Component 1 — Shared feature statistics (SFS)
Before round 1, each client uploads (n_k, Σ_i x_i, Σ_i x_i²) computed on its training split only.
The server forms the pooled mean μ and standard deviation σ (floor 1e-6), broadcasts them once, and every
client standardises all its splits with (μ, σ). The pooled statistics are identical to those a
centralised scaler would fit on the union of training splits. Without SFS, each client standardises
with statistics fitted on its own training split (the realistic default when nothing is shared).
SFS with FedAvg aggregation and cross-entropy is StatAvg (Bouzinis et al., IEEE TNSM 2025).
Upload cost: 2d + 1 floats per client, once.

## Evaluated alternative — Class-aware aggregation of the head (CAA; not part of FedStaP)
CAA was designed as a third component and is retained in the code and the ablation. Its main effect on
test macro-F1 is negative (−2.16 pp over 80 paired comparisons, Wilcoxon p = 3.6e-3), so it is excluded
from the proposed method and reported as a negative result.
Body parameters φ (and BatchNorm buffers) are aggregated with FedAvg weights n_k / Σ_j n_j.
Head row c is aggregated with class-specific weights
  ω_{k,c} = (n_{k,c} + β n_k / C) / Σ_{j∈S_t} (n_{j,c} + β n_j / C),
  w_c ← Σ_{k∈S_t} ω_{k,c} w_{k,c},   b_c ← Σ_{k∈S_t} ω_{k,c} b_{k,c},
where S_t is the set of clients sampled in round t. β = 0 gives pure class-count weighting;
β → ∞ recovers FedAvg for the head. A client without class c contributes to row c only through the
β term. Upload cost: C integers per client, once (the same label-histogram disclosure assumed by
FedLC and FedRS).

## Component 2 — Prior-calibrated local loss (PCL)
Local training minimises the logit-adjusted cross-entropy
  ℓ_k(x, y) = −log softmax(z(x) + τ log π_k)_y,
which removes the local label prior from the learned logits, so the aggregated model approximates a
prior-free (balanced) classifier. Inference uses the raw logits z(x). No extra communication.

## FedStaP = SFS + PCL, aggregated with FedAvg weights.
Ablation: all 2³ on/off combinations of SFS, CAA and PCL (FedAvg = none; StatAvg = SFS only;
FedStaP = SFS + PCL). FedStaP and the ablation cell `abl_s1a0p1` are the same configuration.

## Scope of the claim
The comparison is restricted to methods that deliver a single global model. FedBN keeps BatchNorm
statistics and affine parameters on each client and is evaluated with them; it therefore yields no single
global model. FedBN and FedBN + SFS are reported as a personalised reference and are excluded from the
Friedman ranking and the Holm-corrected tests; their gap to FedStaP is reported separately.

## Baselines
Centralised (pooled data, same model and epoch budget; upper reference), FedAvg, FedProx (μ),
SCAFFOLD (option II control variates), FedBN (BatchNorm kept local), FedLC (τ n_{k,c}^{-1/4} calibration),
FedRS (restricted softmax, α on absent classes), StatAvg, and each of FedProx, SCAFFOLD, FedLC and FedRS
combined with SFS (a fairness control, so that shared normalisation is not credited to FedStaP alone).
Personalised reference: FedBN and FedBN + SFS. A protocol-matched comparison with FedMPSQ on its frozen
10-client CICIoT2023 partition is planned and has not yet been run.

## Data protocol
Datasets: CICIoT2023 (34 fine classes / 8 families) and Edge-IIoTset DNN subset (15 classes / 6 families).
Fit-free preprocessing only before partitioning: drop identifiers/payload text, hash-encode the few
categorical fields, replace ±inf/NaN by 0, signed log1p. Exact duplicate rows removed. Per-class cap
(uniform bottom-k sampling with a fixed hash seed) bounds each class; rarer classes are kept whole.
Partition: Dirichlet label skew Dir(α) over fine labels with a minimum client size; the grouped task
reuses the same partition. Within each client, per-class 70/10/20 train/validation/test split.
All statistics (scalers, priors, class counts) come from training splits only.
Model selection: best round by pooled validation macro-F1; the test splits are evaluated once,
with the selected model. Test predictions of each client use that client's normaliser (and, for FedBN,
its local BatchNorm).

## Metrics
Macro-F1 (primary), accuracy, balanced accuracy, weighted F1, tail-class recall (mean recall over the
quartile of classes with fewest training samples), worst-client macro-F1 (10th percentile across
clients), per-class recall; communication (uploaded MB), wall-clock, peak memory, parameters, FLOPs,
inference latency.

## Statistics
Five seeds per configuration (seed controls partition, split, initialisation, client sampling).
FedStaP vs each global-model baseline: two-sided Wilcoxon signed-rank test on paired (setting, seed) results, Holm
correction across baselines, rank-biserial effect size, and bootstrap 95% CI of the mean paired
difference. Omnibus: Friedman test over the global-model methods with a Nemenyi critical-difference diagram.

## Hyperparameter selection
Every tunable hyperparameter (learning rate, μ, FedLC τ, FedRS α, FedStaP τ, and CAA β for the ablation) is selected on
pooled validation macro-F1 with a held-out tuning seed (100) that is never used in reported results.

## Sensitivity
Dirichlet α ∈ {0.05, 0.1, 0.5, 1.0}; clients K ∈ {10, 20, 50, 100}; PCL τ ∈ {0.25, 0.5, 1, 2};
participation ∈ {0.2, 0.5, 1.0}; three seeds each, fine labels.
