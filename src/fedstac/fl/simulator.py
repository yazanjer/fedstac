"""Single-process federated simulator (exact FedAvg semantics, GPU-resident client data).

Randomness is derived from (seed, round, client) at each use, so a run resumed from a
round checkpoint reproduces an uninterrupted run exactly and no RNG state has to be saved.
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from fedstac.models.mlp import MLP, is_bn_key
from fedstac.utils.io import atomic_torch_save


@dataclass
class ClientData:
    xtr: torch.Tensor
    ytr: torch.Tensor
    xva: torch.Tensor
    yva: torch.Tensor
    xte: torch.Tensor
    yte: torch.Tensor
    counts: torch.Tensor  # training class counts, task label space (float)

    @property
    def n(self) -> int:
        return int(self.ytr.numel())


@dataclass
class MethodCfg:
    name: str
    norm: str = "local"          # local | global (SFS)
    agg: str = "fedavg"          # fedavg | class_aware (CAA)
    beta: float = 0.1            # CAA smoothing
    loss: str = "ce"             # ce | la (PCL) | fedlc | fedrs
    tau: float = 1.0             # PCL / FedLC temperature
    fedrs_alpha: float = 0.5
    prox_mu: float = 0.0
    scaffold: bool = False
    fedbn: bool = False
    pooled: bool = False         # centralised reference
    client_momentum: float | None = None  # None -> fl.momentum; SCAFFOLD uses plain SGD (0.0)


@dataclass
class FLCfg:
    rounds: int = 50
    participation: float = 0.5
    local_epochs: int = 1
    batch_size: int = 256
    lr: float = 0.05
    momentum: float = 0.9
    weight_decay: float = 1e-4
    eval_every: int = 5
    ckpt_every: int = 10
    hidden: tuple = (256, 128, 64)
    dropout: float = 0.1
    model_norm: str = "batch"
    cohort_size: int | None = None   # v0.5.0: fixed number of sampled clients per round (overrides participation)


@dataclass
class DiagCfg:
    enabled: bool = False    # read-only instrumentation; never changes training (tested bit-identical)
    every: int = 5           # diagnostic rounds: 1 and every `every`-th round
    probe: int = 1024        # per-client probe batch (fixed samples), evaluated as two halves for a noise floor


def _gen(*key: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(int(np.random.SeedSequence(list(key)).generate_state(1)[0]))
    return g


def local_loss(logits, y, m: MethodCfg, counts: torch.Tensor):
    if m.loss == "ce":
        return F.cross_entropy(logits, y)
    if m.loss == "la":
        prior = (counts + 1.0) / (counts.sum() + counts.numel())
        return F.cross_entropy(logits + m.tau * torch.log(prior), y)
    if m.loss == "fedlc":
        cal = m.tau * torch.pow(counts.clamp_min(1e-8), -0.25)  # FL-bench reference: absent classes ~removed
        return F.cross_entropy(logits - cal, y)
    if m.loss == "fedrs":
        scale = torch.where(counts > 0, torch.ones_like(counts), torch.full_like(counts, m.fedrs_alpha))
        return F.cross_entropy(logits * scale, y)
    raise ValueError(m.loss)


class Simulator:
    def __init__(self, clients: list[ClientData], n_classes: int, in_dim: int, method: MethodCfg,
                 fl: FLCfg, seed: int, device: str, run_dir: Path, logger=None, diag: DiagCfg | None = None):
        self.c, self.C, self.m, self.fl, self.seed = clients, n_classes, method, fl, seed
        self.device, self.run_dir, self.log = device, Path(run_dir), logger
        torch.manual_seed(seed)
        self.model = MLP(in_dim, n_classes, fl.hidden, fl.dropout, fl.model_norm).to(device)
        self.keys = list(self.model.state_dict().keys())
        self.bn_keys = {k for k in self.keys if is_bn_key(k, self.model)}
        self.param_keys = [n for n, _ in self.model.named_parameters()]
        self.global_sd = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        self.bn_local: dict[int, dict] = {}
        self.c_global = {k: torch.zeros_like(self.global_sd[k]) for k in self.param_keys} if method.scaffold else None
        self.c_local: dict[int, dict] = {}
        self.round = 0
        self.diag_cfg = diag or DiagCfg()
        self.diagnostics: list[dict] = []
        self.steps_log: list[float] = []          # mean local steps per sampled client, every round
        self.last_refresh: dict[int, int] = {}    # client -> last round its control variate was refreshed
        self._diag_model = None
        self.best = {"val_macro_f1": -1.0, "round": 0, "sd": None, "bn_local": None}
        self.curve: list[dict] = []
        self.comm_up_bytes = 0.0
        self.train_time = 0.0
        self.K = len(clients)
        self.n_params = sum(v.numel() for k, v in self.global_sd.items() if k in self.param_keys)
        self.n_bn_params = sum(self.global_sd[k].numel() for k in self.param_keys if k in self.bn_keys)
        self._one_time_comm()

    # ---------------- communication accounting ----------------
    def _one_time_comm(self):
        d = self.c[0].xtr.shape[1]
        if self.m.norm == "global" and not self.m.pooled:
            self.comm_up_bytes += self.K * (2 * d + 1) * 4
        if self.m.agg == "class_aware" and not self.m.pooled:
            self.comm_up_bytes += self.K * self.C * 4

    def _round_up_bytes(self, n_sampled: int) -> float:
        if self.m.pooled:
            return 0.0
        p = self.n_params - (self.n_bn_params if self.m.fedbn else 0)
        return n_sampled * p * 4 * (2 if self.m.scaffold else 1)

    # ---------------- client update ----------------
    def _client_update(self, k: int, t: int):
        m, fl, cd = self.m, self.fl, self.c[k]
        model = self.model
        sd = dict(self.global_sd)
        if m.fedbn and k in self.bn_local:
            sd.update(self.bn_local[k])
        model.load_state_dict(sd)
        model.train()
        torch.manual_seed(int(np.random.SeedSequence([self.seed, t, k, 7]).generate_state(1)[0]))
        mom = fl.momentum if m.client_momentum is None else m.client_momentum
        opt = torch.optim.SGD(model.parameters(), lr=fl.lr, momentum=mom, weight_decay=fl.weight_decay)
        gref = [self.global_sd[n] for n in self.param_keys] if m.prox_mu > 0 else None
        if m.scaffold:
            ck = self.c_local.get(k) or {n: torch.zeros_like(v) for n, v in self.c_global.items()}
            corr = {n: self.c_global[n] - ck[n] for n in self.param_keys}
        g = _gen(self.seed, t, k)
        steps = 0
        counts = cd.counts
        for _ in range(fl.local_epochs):
            perm = torch.randperm(cd.n, generator=g).to(self.device)
            for i in range(0, cd.n, fl.batch_size):
                b = perm[i:i + fl.batch_size]
                if b.numel() < 2 and self.fl.model_norm == "batch":
                    continue
                logits = model(cd.xtr[b])
                loss = local_loss(logits, cd.ytr[b], m, counts)
                if gref is not None:
                    prox = sum(((p - q) ** 2).sum() for p, q in zip(model.parameters(), gref))
                    loss = loss + 0.5 * m.prox_mu * prox
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if m.scaffold:
                    for n, p in model.named_parameters():
                        p.grad.add_(corr[n])
                opt.step()
                steps += 1
        local = {k2: v.detach().clone() for k2, v in model.state_dict().items()}
        dc = None
        if m.scaffold and steps > 0:
            new_ck = {n: ck[n] - self.c_global[n] + (self.global_sd[n] - local[n]) / (steps * fl.lr) for n in self.param_keys}
            dc = {n: new_ck[n] - ck[n] for n in self.param_keys}
            self.c_local[k] = new_ck
            self.last_refresh[k] = t
        if m.fedbn:
            self.bn_local[k] = {k2: local[k2] for k2 in self.bn_keys}
        self._last_steps = steps
        return local, dc

    # ---------------- aggregation ----------------
    def _aggregate(self, sampled: list[int], locals_: list[dict], dcs: list):
        n = torch.tensor([float(self.c[k].n) for k in sampled], device=self.device)
        w = n / n.sum()
        new = {}
        for key in self.keys:
            if self.m.fedbn and key in self.bn_keys:
                new[key] = self.global_sd[key]
                continue
            ref = self.global_sd[key]
            if not torch.is_floating_point(ref):
                new[key] = torch.stack([l[key] for l in locals_]).max(0).values
                continue
            stack = torch.stack([l[key] for l in locals_])
            new[key] = torch.einsum("k,k...->...", w, stack)
        if self.m.agg == "class_aware":
            cnt = torch.stack([self.c[k].counts for k in sampled])          # S x C
            om = cnt + self.m.beta * n[:, None] / self.C
            den = om.sum(0, keepdim=True)
            om = torch.where(den > 0, om / den.clamp_min(1e-12), w[:, None].expand_as(om))
            Wst = torch.stack([l["head.weight"] for l in locals_])          # S x C x d
            bst = torch.stack([l["head.bias"] for l in locals_])            # S x C
            new["head.weight"] = torch.einsum("sc,scd->cd", om, Wst)
            new["head.bias"] = (om * bst).sum(0)
        if self.m.scaffold:
            for nme in self.param_keys:
                valid = [d[nme] for d in dcs if d is not None]
                if valid:
                    self.c_global[nme] = self.c_global[nme] + torch.stack(valid).sum(0) / self.K
        self.global_sd = new

    # ---------------- evaluation ----------------
    @torch.no_grad()
    def predict(self, split: str, sd: dict, bn_local: dict | None):
        model = self.model
        ys, ps, cs = [], [], []
        for k, cd in enumerate(self.c):
            x, y = (cd.xva, cd.yva) if split == "val" else (cd.xte, cd.yte)
            if y.numel() == 0:
                continue
            s = dict(sd)
            if self.m.fedbn and bn_local and k in bn_local:
                s.update(bn_local[k])
            model.load_state_dict(s)
            model.eval()
            out = torch.cat([model(x[i:i + 65536]).argmax(1) for i in range(0, len(y), 65536)])
            ys.append(y.cpu()); ps.append(out.cpu()); cs.append(torch.full((len(y),), k))
        return torch.cat(ys).numpy(), torch.cat(ps).numpy(), torch.cat(cs).numpy()

    def _eval_val(self, t: int):
        from sklearn.metrics import f1_score, accuracy_score
        y, p, _ = self.predict("val", self.global_sd, self.bn_local)
        f1 = f1_score(y, p, average="macro", labels=np.unique(y), zero_division=0)
        acc = accuracy_score(y, p)
        rec = {"round": t, "val_macro_f1": float(f1), "val_acc": float(acc),
               "comm_up_MB": self.comm_up_bytes / 1e6, "train_time_s": self.train_time}
        self.curve.append(rec)
        if f1 > self.best["val_macro_f1"]:
            self.best = {"val_macro_f1": float(f1), "round": t,
                         "sd": {k: v.clone() for k, v in self.global_sd.items()},
                         "bn_local": copy.deepcopy(self.bn_local) if self.m.fedbn else None}
        if self.log:
            self.log(rec)
        return rec

    # ---------------- checkpointing ----------------
    def state(self) -> dict:
        return {"round": self.round, "global_sd": self.global_sd, "bn_local": self.bn_local,
                "c_global": self.c_global, "c_local": self.c_local, "best": self.best,
                "curve": self.curve, "comm_up_bytes": self.comm_up_bytes, "train_time": self.train_time,
                "diagnostics": self.diagnostics, "steps_log": self.steps_log, "last_refresh": self.last_refresh}

    def load_state(self, s: dict):
        self.round, self.global_sd, self.bn_local = s["round"], s["global_sd"], s["bn_local"]
        self.c_global, self.c_local, self.best = s["c_global"], s["c_local"], s["best"]
        self.curve, self.comm_up_bytes, self.train_time = s["curve"], s["comm_up_bytes"], s["train_time"]
        self.diagnostics = s.get("diagnostics", [])
        self.steps_log = s.get("steps_log", [])
        self.last_refresh = s.get("last_refresh", {})

    def cohort(self) -> int:
        if self.fl.cohort_size:
            return max(1, min(self.K, int(self.fl.cohort_size)))
        return max(1, int(round(self.fl.participation * self.K)))

    # ---------------- read-only diagnostics (v0.5.0) ----------------
    def _probe_grads(self) -> list[torch.Tensor]:
        """Flattened gradient of each client's local loss at the current global model, on a fixed probe
        batch. Uses a private model copy with dropout disabled and BatchNorm in batch-statistics mode,
        so neither the training model, its buffers nor any RNG stream is touched."""
        if self._diag_model is None:
            self._diag_model = copy.deepcopy(self.model)
            self._probe_idx = []
            for k, cd in enumerate(self.c):
                g = torch.Generator(); g.manual_seed(int(np.random.SeedSequence([self.seed, k, 31337]).generate_state(1)[0]))
                self._probe_idx.append(torch.randperm(cd.n, generator=g)[: self.diag_cfg.probe].to(self.device))
        dm = self._diag_model
        dm.load_state_dict(self.global_sd)
        dm.train()
        for mod in dm.modules():
            if isinstance(mod, torch.nn.Dropout):
                mod.eval()
        params = [p for _, p in dm.named_parameters()]
        out = []
        for k, cd in enumerate(self.c):
            b = self._probe_idx[k]
            if b.numel() < 4:
                out.append(None); continue
            halves = []
            for h in (b[0::2], b[1::2]):
                loss = local_loss(dm(cd.xtr[h]), cd.ytr[h], self.m, cd.counts)
                gr = torch.autograd.grad(loss, params)
                halves.append(torch.cat([x.reshape(-1) for x in gr]).detach())
            out.append(halves)
        return out

    def _diagnose(self, sampled: list[int], t: int):
        rng_state = torch.get_rng_state()
        with torch.enable_grad():
            gk = self._probe_grads()
        torch.set_rng_state(rng_state)
        valid = [k for k in range(self.K) if gk[k] is not None]
        n = np.array([float(self.c[k].n) for k in valid])
        w = torch.as_tensor(n / n.sum(), dtype=gk[valid[0]][0].dtype, device=gk[valid[0]][0].device)[:, None]
        GA = torch.stack([gk[k][0] for k in valid]); GB = torch.stack([gk[k][1] for k in valid])
        gA, gB = (w * GA).sum(0), (w * GB).sum(0)
        G = 0.5 * (GA + GB)
        g = 0.5 * (gA + gB)
        pos = {k: i for i, k in enumerate(valid)}
        gn2 = float((g ** 2).sum())
        dissim = float(((G - g[None, :]) ** 2).sum(1).mean() / max(gn2, 1e-30))
        rec = {"round": t, "n_sampled": len(sampled), "grad_norm_global": gn2 ** 0.5, "grad_dissimilarity": dissim}
        if self.m.scaffold:
            flat = lambda d: torch.cat([d[nme].reshape(-1) for nme in self.param_keys])
            c = flat(self.c_global)
            zero = torch.zeros_like(c)
            errs, coss, cks, gaps, stale, noise = [], [], [], [], [], []
            for k in sampled:
                if gk[k] is None:
                    continue
                i = pos[k]
                ck = flat(self.c_local[k]) if k in self.c_local else zero
                corr = c - ck                       # correction SCAFFOLD applies to client k in round t
                ideal = g - G[i]                    # ideal drift correction at the current global model
                inorm = float(ideal.norm())
                # probe noise floor: standard error of the ideal estimated from the two probe halves
                noise.append(0.5 * float(((gA - GA[i]) - (gB - GB[i])).norm()) / max(inorm, 1e-30))
                errs.append(float((corr - ideal).norm()) / max(inorm, 1e-30))
                coss.append(float(F.cosine_similarity(corr, ideal, dim=0)) if float(corr.norm()) > 0 else 0.0)
                cks.append(float(ck.norm())); gaps.append(float(corr.norm()))
                stale.append(t - self.last_refresh.get(k, 0))
            rec.update({"c_norm": float(c.norm()), "ck_norm_mean": float(np.mean(cks)), "ck_norm_max": float(np.max(cks)),
                        "c_minus_ck_mean": float(np.mean(gaps)), "corr_rel_err_mean": float(np.mean(errs)),
                        "corr_rel_err_median": float(np.median(errs)), "corr_cos_mean": float(np.mean(coss)),
                        "staleness_mean": float(np.mean(stale)), "ideal_noise_rel_mean": float(np.mean(noise)),
                        "frac_never_refreshed": float(np.mean([k not in self.c_local for k in sampled]))})
        self.diagnostics.append(rec)

    def run(self, ckpt_path: Path | None = None):
        fl = self.fl
        m_sel = self.cohort()
        while self.round < fl.rounds:
            t = self.round + 1
            rng = np.random.default_rng([self.seed, t, 99])
            sampled = sorted(rng.choice(self.K, size=m_sel, replace=False).tolist())
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            if self.diag_cfg.enabled and (t == 1 or t % self.diag_cfg.every == 0):
                self._diagnose(sampled, t)
            t0 = time.perf_counter()
            locals_, dcs, steps = [], [], []
            for k in sampled:
                l, dc = self._client_update(k, t)
                locals_.append(l); dcs.append(dc); steps.append(self._last_steps)
            self.steps_log.append(float(np.mean(steps)))
            self._aggregate(sampled, locals_, dcs)
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            self.train_time += time.perf_counter() - t0
            self.comm_up_bytes += self._round_up_bytes(len(sampled))
            self.round = t
            if t % fl.eval_every == 0 or t == fl.rounds:
                self._eval_val(t)
            if ckpt_path is not None and (t % fl.ckpt_every == 0 or t == fl.rounds):
                atomic_torch_save(self.state(), ckpt_path)
        return self.best
