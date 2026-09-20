"""Determinism, resume-equivalence, aggregation and leakage checks (CPU, synthetic)."""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fedstac.utils.seed import set_seed  # noqa: E402
from fedstac.fl.simulator import ClientData, FLCfg, MethodCfg, Simulator  # noqa: E402
from fedstac.data.partition import build_federation  # noqa: E402
from fedstac.data.normalize import global_scaler  # noqa: E402


def make_clients(K=6, C=5, d=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(3000, d)).astype(np.float32)
    y = rng.integers(0, C, 3000)
    X += y[:, None] * 0.5
    splits = build_federation(y, K, 0.3, seed, min_size=20)
    out = []
    for s in splits:
        t = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt)
        out.append(ClientData(t(X[s["train"]]), t(y[s["train"]], torch.long), t(X[s["val"]]), t(y[s["val"]], torch.long),
                              t(X[s["test"]]), t(y[s["test"]], torch.long),
                              t(np.bincount(y[s["train"]], minlength=C).astype(np.float32))))
    return out, splits, X, y


def run(method, rounds=6, ckpt=None, resume_from=None, stop_at=None):
    set_seed(0, 1)
    clients, *_ = make_clients()
    fl = FLCfg(rounds=rounds, eval_every=2, ckpt_every=2, hidden=(32, 16))
    sim = Simulator(clients, 5, 8, method, fl, 0, "cpu", Path("/tmp"))
    if resume_from:
        sim.load_state(torch.load(resume_from, weights_only=False))
    if stop_at:
        sim.fl = replace(fl, rounds=stop_at)
        sim.run(ckpt)
        return sim
    sim.run(ckpt)
    return sim


def flat(sd):
    return torch.cat([v.float().flatten() for v in sd.values()])


def test_determinism_and_resume(tmp="/tmp/claude-0/ck.pt"):
    for m in [MethodCfg("fedstac", norm="global", agg="class_aware", loss="la"),
              MethodCfg("scaffold", scaffold=True, client_momentum=0.0), MethodCfg("fedbn", fedbn=True)]:
        a = run(m)
        b = run(m)
        assert torch.equal(flat(a.global_sd), flat(b.global_sd)), f"{m.name} not deterministic"
        run(m, stop_at=4, ckpt=Path(tmp))
        c = run(m, resume_from=tmp)
        assert torch.equal(flat(a.global_sd), flat(c.global_sd)), f"{m.name} resume differs"
    print("determinism + resume: OK")


def test_class_aware_limits():
    set_seed(0, 1)
    clients, *_ = make_clients()
    fl = FLCfg(rounds=1, hidden=(16,))
    big = Simulator(clients, 5, 8, MethodCfg("x", agg="class_aware", beta=1e9), fl, 0, "cpu", Path("/tmp"))
    ref = Simulator(clients, 5, 8, MethodCfg("y"), fl, 0, "cpu", Path("/tmp"))
    big.run(); ref.run()
    assert torch.allclose(big.global_sd["head.weight"], ref.global_sd["head.weight"], atol=1e-5), "beta->inf must equal FedAvg"
    # beta=0: a row is a count-weighted mean over holders of that class
    s = Simulator(clients, 5, 8, MethodCfg("z", agg="class_aware", beta=0.0), fl, 0, "cpu", Path("/tmp"))
    locs = [{"head.weight": torch.full((5, 16), float(i)), "head.bias": torch.zeros(5), **{k: v for k, v in s.global_sd.items() if not k.startswith("head")}} for i in range(2)]
    s.c[0].counts = torch.tensor([10., 0, 0, 0, 0]); s.c[1].counts = torch.tensor([0., 10, 0, 0, 0])
    s._aggregate([0, 1], locs, [None, None])
    assert torch.allclose(s.global_sd["head.weight"][0], torch.zeros(16)) and torch.allclose(s.global_sd["head.weight"][1], torch.ones(16))
    print("class-aware aggregation limits: OK")


def test_splits_disjoint_and_sfs_equals_pooled():
    _, splits, X, y = make_clients()
    allidx = np.concatenate([np.concatenate([s["train"], s["val"], s["test"]]) for s in splits])
    assert len(allidx) == len(np.unique(allidx)), "overlap between clients/splits"
    tr = [X[s["train"]] for s in splits]
    mu, sd = global_scaler(tr)
    pooled = np.concatenate(tr)
    assert np.allclose(mu, pooled.mean(0), atol=1e-5) and np.allclose(sd, pooled.std(0), atol=1e-4)
    print("disjoint splits + SFS == pooled train scaler: OK")


if __name__ == "__main__":
    test_splits_disjoint_and_sfs_equals_pooled()
    test_class_aware_limits()
    test_determinism_and_resume()
