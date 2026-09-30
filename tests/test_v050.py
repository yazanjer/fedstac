"""v0.5.0 checks: Arm B partitioner, fixed cohort, read-only diagnostics (CPU, synthetic)."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fedstac.data.partition import (build_federation_armb, equal_size_dirichlet_partition,  # noqa: E402
                                    label_skew_stats, nested_stratified_subsample)
from fedstac.fl.simulator import DiagCfg, FLCfg, MethodCfg, Simulator  # noqa: E402
from fedstac.utils.seed import set_seed  # noqa: E402
from test_core import flat, make_clients  # noqa: E402


def _y(n=20000, C=12, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(np.full(C, 0.7))
    return rng.choice(C, size=n, p=p)


def test_nested_subsample():
    y = _y()
    for seed in (0, 1):
        prev = None
        for K in (10, 20, 50, 100, 200):
            s = nested_stratified_subsample(y, K * 90, seed)
            assert len(s) == K * 90 and len(np.unique(s)) == len(s)
            if prev is not None:
                assert np.isin(prev, s).all(), "subsample not nested in K"
            # stratification: class shares within one sample of the pool shares
            share = np.bincount(y[s], minlength=12) / len(s)
            assert np.abs(share - np.bincount(y, minlength=12) / len(y)).max() < 1.5 / len(s) * 12
            prev = s
    print("nested stratified subsample: OK")


def test_equal_size_partition():
    y = _y()
    for K, a in ((10, 0.1), (200, 0.1), (50, 0.5)):
        pool = nested_stratified_subsample(y, K * 90, 3)
        parts = equal_size_dirichlet_partition(y, pool, K, 90, a, 3)
        assert all(len(p) == 90 for p in parts)
        allp = np.concatenate(parts)
        assert len(np.unique(allp)) == len(allp) == len(pool) and np.isin(allp, pool).all()
    # skew is retained: alpha = 0.1 is far more heterogeneous than alpha = 100
    sk = {}
    for a in (0.1, 100.0):
        spl = build_federation_armb(y, 50, 90, a, 1)
        sk[a] = label_skew_stats(spl, y, 12)["mean_tv_to_global"]
    assert sk[0.1] > 3 * sk[100.0], sk
    # deterministic
    a1 = build_federation_armb(y, 20, 90, 0.1, 7); a2 = build_federation_armb(y, 20, 90, 0.1, 7)
    assert all(np.array_equal(p["train"], q["train"]) for p, q in zip(a1, a2))
    print("equal-size Dirichlet partition (sizes, disjointness, skew, determinism): OK", sk)


def _sim(method, diag, rounds=6, cohort=None):
    set_seed(0, 1)
    clients, *_ = make_clients()
    fl = FLCfg(rounds=rounds, eval_every=2, ckpt_every=2, hidden=(32, 16), cohort_size=cohort)
    sim = Simulator(clients, 5, 8, method, fl, 0, "cpu", Path("/tmp"), diag=DiagCfg(enabled=diag, every=2, probe=64))
    sim.run()
    return sim


def test_diagnostics_read_only():
    for m in (MethodCfg("scaffold", scaffold=True, client_momentum=0.0),
              MethodCfg("fedstap", norm="global", loss="la")):
        a, b = _sim(m, False), _sim(m, True)
        assert torch.equal(flat(a.global_sd), flat(b.global_sd)), f"{m.name}: diagnostics changed training"
        strip = lambda cv: [{k: v for k, v in r.items() if k != "train_time_s"} for r in cv]
        assert strip(a.curve) == strip(b.curve)
        assert len(b.diagnostics) == 4 and not a.diagnostics
        if m.scaffold:
            d = b.diagnostics[-1]
            assert d["corr_rel_err_mean"] > 0 and 0 <= d["frac_never_refreshed"] <= 1 and d["staleness_mean"] >= 1
            # round 1: all variates zero -> correction is zero -> relative error exactly 1
            assert abs(b.diagnostics[0]["corr_rel_err_mean"] - 1.0) < 1e-6
    print("diagnostics are read-only (bit-identical training): OK")


def test_cohort():
    s = _sim(MethodCfg("fedavg"), False, rounds=2, cohort=2)
    assert s.cohort() == 2
    s = _sim(MethodCfg("fedavg"), False, rounds=2, cohort=100)
    assert s.cohort() == s.K
    assert len(s.steps_log) == 2 and s.steps_log[0] > 0
    print("fixed cohort: OK")


if __name__ == "__main__":
    test_nested_subsample()
    test_equal_size_partition()
    test_diagnostics_read_only()
    test_cohort()
