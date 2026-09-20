"""Synthetic CSV fixture with the same shape of problems as the real data (imbalance, dups, inf)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
sizes = [6000, 5000, 4000, 3000, 2000, 1500, 800, 400, 200, 80]
rows = []
for c, n in enumerate(sizes):
    mu = rng.normal(0, 2, 20)
    x = rng.normal(mu, 1.0 + 0.2 * c, size=(n, 20)) * (10 ** rng.integers(0, 4, 20))
    df = pd.DataFrame(x, columns=[f"f{i}" for i in range(20)])
    df["proto"] = rng.choice(["tcp", "udp", "icmp"], n)
    df["src_ip"] = [f"10.0.0.{i%250}" for i in range(n)]
    df["label"] = f"c{c}"
    rows.append(df)
df = pd.concat(rows).sample(frac=1, random_state=0)
df.iloc[:50, 0] = np.inf
df = pd.concat([df, df.iloc[:300]])  # exact duplicates
for i, part in enumerate([df.iloc[j::3] for j in range(3)]):
    part.to_csv(out / f"part-{i}.csv", index=False)
print(len(df))
