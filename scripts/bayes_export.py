"""Exports the Bayesian models of amendment b as JSON inputs for the Rust NUTS sampler (rust_bayes/).
Each run: design X (cells x p), interaction block I (cells x K), y, offset, per-parameter Normal priors, mode."""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W = os.path.join(ROOT, "work", "bayes"); os.makedirs(W, exist_ok=True)
SD = 0.114


def design(c, classes):
    c = c.copy(); c["post"] = (c.period == "post").astype(float)
    B = pd.get_dummies(pd.Categorical(c.bmic.astype(str), categories=classes)).astype(float).drop(columns="normal")
    inter = B.mul(c.post.values, axis=0); inter.columns = [f"{k}:post" for k in B.columns]
    cov = pd.get_dummies(c[["agegrp", "nullip", "race"]].astype(str), drop_first=True).astype(float)
    cov["pdiab"] = c.pdiab.astype(float); cov["phyp"] = c.phyp.astype(float)
    X = pd.concat([B.reset_index(drop=True), c[["post"]].reset_index(drop=True), cov.reset_index(drop=True)], axis=1)
    return X, inter.reset_index(drop=True), c.deaths.values.astype(float), np.log(c.risk.values.astype(float))


def write(name, X, I, y, off, mode, prior_g=(0.0, 1.0), score=None):
    base = float(np.log(y.sum() / np.exp(off).sum()))
    js = dict(name=name, mode=mode, n=int(len(y)), p=int(X.shape[1]), k=int(I.shape[1]), X=X.values.ravel().tolist(),
              I=I.values.ravel().tolist(), y=y.tolist(), off=off.tolist(), a_mu=-8.0, a_sd=5.0, b_sd=2.5,
              g_mu=prior_g[0], g_sd=prior_g[1], slope_sd=0.5, tau_sd=0.2, score=(score or [0.0] * I.shape[1]),
              init_a=base, xnames=list(X.columns), inames=list(I.columns))
    json.dump(js, open(os.path.join(W, f"{name}.json"), "w"))
    print(name, len(y), X.shape[1], I.shape[1])


if __name__ == "__main__":
    from analysis import load, cells, CLS  # only the standalone models of the earlier analysis need it
    d = load(); c = cells(d, lambda x: x.estofd == "N")
    c1 = c.copy(); c1["bmic"] = c1.bmic.astype(str).replace({"ob50": "ob40"})
    c1 = c1.groupby(["period", "bmic", "agegrp", "nullip", "race", "pdiab", "phyp"], observed=True)[["risk", "deaths"]].sum().reset_index()
    X, I, y, off = design(c1, [k for k in CLS if k != "ob50"])
    tgt = I[["ob40:post"]]; Xo = pd.concat([X, I.drop(columns="ob40:post")], axis=1)
    for nm, pr in [("B1_sceptical", (0.0, SD)), ("B1_weak", (0.0, 1.0)), ("B1_optimistic", (float(np.log(0.8)), SD))]:
        write(nm, Xo, tgt, y, off, "b1", pr)
    X, I, y, off = design(c, CLS)
    score = [float(CLS.index(k.split(":")[0]) - CLS.index("normal")) for k in I.columns]
    write("B2_hier", X, I, y, off, "b2", score=score)
