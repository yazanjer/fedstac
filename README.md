# FedSTAC

Code for **"FedSTAC: Statistics- and Class-Aware Aggregation for Non-IID Federated Intrusion Detection in IoT Networks"**.

FedSTAC combines three components, each switchable from configuration:
**SFS**, which shares feature statistics so all clients use one global normaliser (StatAvg-style);
**CAA**, class-aware aggregation of the classifier head, with head rows weighted by each client's class counts and smoothed by β;
and **PCL**, a local loss calibrated to each client's class prior (logit adjustment with temperature τ).
Equations and the full protocol are in [`docs/METHOD.md`](docs/METHOD.md).

## Layout
```
configs/        Hydra: main.yaml, dataset/, method/ (baselines + 8 ablation variants), suites/, tuned.yaml
src/fedstac/    data (fit-free prep, Dirichlet partition, scalers), models (MLP), fl (simulator), evaluation
scripts/        download_data, prepare_data, run_suite (parallel, resumable), select_hparams, analyze
notebooks/      fedstac_colab_runner.ipynb (end-to-end on Colab)
tests/          determinism, resume equivalence, aggregation limits, leakage checks
```

## Reproduce
```bash
pip install -r requirements.txt            # torch/numpy/pandas/sklearn/scipy/matplotlib from the base image
export PYTHONPATH=src
python scripts/download_data.py --dataset edgeiiot   --raw data/raw/edgeiiot     # needs Kaggle credentials
python scripts/download_data.py --dataset ciciot2023 --raw data/raw/ciciot2023
python scripts/prepare_data.py  --dataset edgeiiot   --raw data/raw/edgeiiot
python scripts/prepare_data.py  --dataset ciciot2023 --raw data/raw/ciciot2023
python tests/test_core.py
python scripts/run_suite.py --suite tune_lr      && python scripts/select_hparams.py --suite tune_lr
python scripts/run_suite.py --suite tune_methods && python scripts/select_hparams.py --suite tune_methods
python scripts/run_suite.py --suite main
python scripts/run_suite.py --suite ablation
python scripts/run_suite.py --suite sensitivity
python scripts/analyze.py                         # -> outputs/paper/{tables,figures}
```
Single run: `python -m fedstac.run dataset=edgeiiot method=fedstac seed=0 task.labels=fine`.

## Reproducibility contract
- Seeds: every source of randomness is derived from (seed, round, client) at each use. Resuming a run from its round checkpoint gives bit-identical results, and a test checks this.
- No leakage: preprocessing before partitioning is fit-free. Scalers, priors and class counts are computed from training splits only. Model selection uses pooled validation macro-F1, and the test splits are evaluated once.
- Hyperparameters are selected on a tuning seed (100) that is never reported (`configs/tuned.yaml`, with the selection log in `outputs/index/*_selection.json`).
- Each `results.json` records the git commit, hardware, prepared-data SHA-256 and resolved config. `run_id` is a hash of everything that changes the numbers.
- Statistics: 5 seeds; Wilcoxon signed-rank with Holm correction, rank-biserial effect size, bootstrap CIs, and Friedman test with a Nemenyi CD diagram.

## Data
- CICIoT2023 (Neto et al., 2023): Kaggle mirror `himadri07/ciciot2023`. 34 classes, or 8 families using the FedMPSQ mapping.
- Edge-IIoTset (Ferrag et al., 2022): `DNN-EdgeIIoT-dataset.csv`. 15 classes, or 6 threat families.

The data is not redistributed. See each dataset's licence.
