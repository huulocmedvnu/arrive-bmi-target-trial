"""Amendment i summary: rate ratios B versus A per BMI class and model-based standardised risk differences."""
import os, json, glob
import numpy as np, pandas as pd
from numpyro.diagnostics import split_gelman_rubin as R
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W, OUT = os.path.join(ROOT, "work", "bayes", "tte"), os.path.join(ROOT, "outputs")
CLS = ["under", "normal", "over", "ob30", "ob35", "ob40", "ob50"]
res = {}
for name in ["tte__primary", "tte__no_sb39"]:
    m = json.load(open(os.path.join(W, f"{name}.json"))); p = m["p"]
    D = np.stack([np.loadtxt(f, delimiter=",") for f in sorted(glob.glob(os.path.join(W, "draws", f"{name}_chain*.csv")))])
    st = np.load(os.path.join(W, f"{name}_strata.npz")); flat = D.reshape(-1, D.shape[-1])
    a, b, g = flat[:, 0], flat[:, 1:1 + p], flat[:, 1 + p:1 + p + 7]
    eta = a[None, :] + st["X"] @ b.T                       # strata x draws, rate per pregnancy under A
    out = {}
    for j, k in enumerate(CLS):
        sel = st["cls"] == j; w = st["w"][sel] / st["w"][sel].sum()
        rA = (w[:, None] * np.exp(eta[sel])).sum(0); rB = (w[:, None] * np.exp(eta[sel] + g[None, :, j])).sum(0)
        rd = (rB - rA) * 1e4; rr = np.exp(g[:, j])
        out[k] = dict(rr=round(float(np.median(rr)), 3), rr_lo=round(float(np.quantile(rr, .025)), 3), rr_hi=round(float(np.quantile(rr, .975)), 3),
                      rd=round(float(np.median(rd)), 3), rd_lo=round(float(np.quantile(rd, .025)), 3), rd_hi=round(float(np.quantile(rd, .975)), 3),
                      p_B_higher=round(float((rr > 1).mean()), 4), rhat=round(float(R(D[..., 1 + p + j])), 4))
    res[name] = out
json.dump(res, open(os.path.join(OUT, "tte.json"), "w"), indent=1)
for n, o in res.items():
    print(n); print(pd.DataFrame(o).T.to_string())
