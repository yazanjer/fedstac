"""One federated (or centralised) experiment. Thin Hydra entry point around `execute`."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from fedstac.utils.seed import configure_env

configure_env()  # CUBLAS workspace before torch touches CUDA

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from fedstac.data.normalize import global_scaler, local_scalers
from fedstac.data.partition import build_federation
from fedstac.data.prepare import load_prepared
from fedstac.evaluation.metrics import compute_metrics
from fedstac.evaluation.profile import latency_ms, peak_memory_mb
from fedstac.fl.simulator import ClientData, FLCfg, MethodCfg, Simulator
from fedstac.models.mlp import MLP, count_params, mlp_flops
from fedstac.utils.io import atomic_torch_save, atomic_write_json, git_commit, hardware_info, stable_hash
from fedstac.utils.seed import set_seed

REPO = Path(__file__).resolve().parents[2]


def run_identity(cfg: DictConfig) -> dict:
    """Everything that changes the numbers — and nothing else (not names, paths or logging)."""
    method = OmegaConf.to_container(cfg.method, resolve=True)
    method.pop("name", None)
    meta_path = Path(cfg.paths.data_root) / cfg.dataset.name / "meta.json"
    prepared = json.loads(meta_path.read_text()).get("prepared_sha256") if meta_path.exists() else "unprepared"
    ident = {
        "dataset": cfg.dataset.name, "prepared_sha256": prepared, "timing_run": bool(cfg.get("timing_run", False)),
        "labels": cfg.task.labels, "seed": int(cfg.seed),
        "federation": OmegaConf.to_container(cfg.federation, resolve=True),
        "fl": OmegaConf.to_container(cfg.fl, resolve=True),
        "method": method,
    }
    if method.get("pooled"):
        ident["fl"] = dict(ident["fl"], rounds=int(cfg.centralised_rounds), participation=1.0)
    ident["fl"].pop("ckpt_every", None)
    return ident


def build_clients(cfg, X, y_fine, y_task, n_classes, device):
    fed = cfg.federation
    splits = build_federation(y_fine, int(fed.n_clients), float(fed.alpha), int(cfg.seed),
                              int(fed.min_client_size), float(fed.val_frac), float(fed.test_frac))
    m = cfg.method
    if m.pooled:  # centralised reference: identical samples, one client
        splits = [{s: np.sort(np.concatenate([sp[s] for sp in splits])) for s in ("train", "val", "test")}]
    tr = [X[s["train"]] for s in splits]
    if m.norm == "global":
        mu, sd = global_scaler(tr)
        scalers = [(mu, sd)] * len(splits)
    else:
        scalers = local_scalers(tr)
    clients = []
    to = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=device)
    for s, (mu, sd) in zip(splits, scalers):
        norm = lambda idx: (X[idx] - mu) / sd
        counts = np.bincount(y_task[s["train"]], minlength=n_classes).astype(np.float32)
        clients.append(ClientData(
            to(norm(s["train"])), to(y_task[s["train"]], torch.long),
            to(norm(s["val"])), to(y_task[s["val"]], torch.long),
            to(norm(s["test"])), to(y_task[s["test"]], torch.long), to(counts)))
    return clients, splits


def execute(cfg: DictConfig) -> dict:
    device = cfg.device if cfg.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(int(cfg.seed), cfg.cpu_threads)
    ident = run_identity(cfg)
    run_id = stable_hash(ident)
    run_dir = Path(cfg.paths.out_root) / "runs" / run_id
    res_path = run_dir / "results.json"
    if res_path.exists():
        print(f"[skip] {run_id} already complete")
        return json.loads(res_path.read_text())
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))

    X, y_fine, meta = load_prepared(Path(cfg.paths.data_root) / cfg.dataset.name)
    if cfg.task.labels == "grouped":
        y_task = np.asarray(meta["group_of_class"], dtype=np.int64)[y_fine]
        class_names = meta["groups"]
    else:
        y_task, class_names = y_fine, meta["classes"]
    C = len(class_names)
    clients, splits = build_clients(cfg, X, y_fine, y_task, C, device)
    train_counts = np.bincount(np.concatenate([y_task[s["train"]] for s in splits]), minlength=C)

    flc = OmegaConf.to_container(cfg.fl, resolve=True)
    flc["hidden"] = tuple(flc["hidden"])
    if cfg.method.pooled:
        flc.update(rounds=int(cfg.centralised_rounds), participation=1.0)
    fl = FLCfg(**flc)
    method = MethodCfg(**OmegaConf.to_container(cfg.method, resolve=True))

    wb = None
    if cfg.wandb.mode != "disabled":
        import wandb
        wb = wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, mode=cfg.wandb.mode,
                        group=cfg.suite, name=f"{cfg.dataset.name}-{cfg.task.labels}-{cfg.method.name}-s{cfg.seed}-{run_id}",
                        config={**ident, "run_id": run_id, "method_name": cfg.method.name, "git": git_commit(REPO)},
                        dir=os.environ.get("WANDB_DIR", str(run_dir)), reinit=True)

    def log(rec):
        print(f"[{run_id}] r{rec['round']:>3} val_mF1={rec['val_macro_f1']:.4f} acc={rec['val_acc']:.4f}", flush=True)
        if wb:
            wb.log(rec, step=rec["round"])

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    sim = Simulator(clients, C, X.shape[1], method, fl, int(cfg.seed), device, run_dir, log)
    ckpt = run_dir / "ckpt.pt"
    if ckpt.exists():
        sim.load_state(torch.load(ckpt, map_location=device, weights_only=False))
        print(f"[resume] {run_id} from round {sim.round}")
    wall0 = time.time()
    best = sim.run(ckpt)
    y, p, cl = sim.predict("test", best["sd"], best["bn_local"])
    metrics = compute_metrics(y, p, cl, C, train_counts)

    model = MLP(X.shape[1], C, fl.hidden, fl.dropout, fl.model_norm).to(device)
    model.load_state_dict(best["sd"])
    prof = {
        "params": count_params(model),
        "flops_per_sample": mlp_flops(X.shape[1], C, fl.hidden),
        "latency_ms_b1": latency_ms(model, X.shape[1], device, 1),
        "latency_ms_b1024": latency_ms(model, X.shape[1], device, 1024),
        "train_time_s": sim.train_time,
        "wall_time_s": time.time() - wall0,
        "comm_up_MB": sim.comm_up_bytes / 1e6,
        **peak_memory_mb(device),
    }
    atomic_torch_save({"sd": best["sd"], "bn_local": best["bn_local"], "round": best["round"]}, run_dir / "best_model.pt")
    result = {
        "run_id": run_id, "identity": ident, "method_name": cfg.method.name, "suite": cfg.suite,
        "dataset": cfg.dataset.name, "labels": cfg.task.labels, "seed": int(cfg.seed),
        "class_names": class_names, "train_class_counts": train_counts.tolist(),
        "client_train_sizes": [int(len(s["train"])) for s in splits],
        "best_round": best["round"], "best_val_macro_f1": best["val_macro_f1"],
        "test": metrics, "profile": prof, "curve": sim.curve,
        "provenance": {"git": git_commit(REPO), "hardware": hardware_info(), "device": device,
                       "prepared_sha256": meta.get("prepared_sha256"), "argv": sys.argv,
                       "cpu_threads": cfg.cpu_threads, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")},
    }
    atomic_write_json(res_path, result)
    if ckpt.exists():
        ckpt.unlink()  # results + best model are the durable record; round checkpoints are transient
    if wb:
        wb.summary.update({f"test/{k}": v for k, v in metrics.items() if isinstance(v, float)})
        wb.summary.update({f"prof/{k}": v for k, v in prof.items()})
        wb.finish()
    print(f"[done] {run_id} {cfg.method.name} test mF1={metrics['macro_f1']:.4f} acc={metrics['accuracy']:.4f} "
          f"tail={metrics['tail_recall']:.4f} ({prof['wall_time_s']:.0f}s)")
    return result


@hydra.main(config_path=str(REPO / "configs"), config_name="main", version_base="1.3")
def main(cfg: DictConfig) -> None:
    execute(cfg)


if __name__ == "__main__":
    main()
