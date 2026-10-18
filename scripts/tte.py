"""Amendment i: matched target-trial emulation, induction in week 39 (A) versus expectant management (B), low-risk
non-anomalous pregnancies ongoing at 39+0, 2019-2024. Builds rust_bayes inputs (work/bayes/tte) and the exactly
stratified conjugate estimates for caesarean, NICU and morbidity. Input work/counts_tte.tsv."""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bayes_export as be

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W, OUT = os.path.join(ROOT, "work"), os.path.join(ROOT, "outputs")
be.W = os.path.join(W, "bayes", "tte"); os.makedirs(be.W, exist_ok=True)
CLS = ["under", "normal", "over", "ob30", "ob35", "ob40", "ob50"]
COLS = ["kind", "year", "month", "ga", "bmic", "agegrp", "nullip", "race", "low", "anom", "educ", "induct", "x3", "estofd",
        "n", "nicu", "cesar", "morb", "nd", "v5", "v6", "v7"]
STR = ["year", "bmic", "agegrp", "nullip", "race", "educ"]
rng = np.random.default_rng(20261017); ND = 100_000; S = 1e4

d = pd.read_csv(os.path.join(W, "counts_tte.tsv"), sep="\t", names=COLS, dtype={"estofd": str})
d = d[(d.year >= 2019) & (d.year <= 2024) & (d.low == 1) & (d.anom == 0) & (d.ga >= 39)].copy()
d["bmic"] = d.bmic.map(dict(enumerate(CLS)))
d["arm"] = np.where((d.ga == 39) & (d.induct == 1) & (d.kind != "fet"), "A", "B")
pop = d[d.kind.isin(["nat", "fet"])]; inf = d[d.kind == "inf"]
js = {"n_A": int(pop[pop.arm == "A"].n.sum()), "n_B": int(pop[pop.arm == "B"].n.sum())}


def cells(drop_sb39=False):
    risk = pop.groupby(STR + ["arm"]).n.sum().rename("risk")
    sb = pop[(pop.kind == "fet") & (pop.estofd == "N") & ~((pop.ga == 39) & drop_sb39)].groupby(STR + ["arm"]).n.sum().rename("sb")
    nd = inf.groupby(STR + ["arm"]).nd.sum().rename("nd")
    c = pd.concat([risk, sb, nd], axis=1).fillna(0).reset_index(); c = c[c.risk > 0]
    c["deaths"] = c.sb + c.nd
    return c.reset_index(drop=True)


def write(c, name):
    X = pd.get_dummies(c[STR].astype(str), drop_first=True).astype(float)
    I = pd.DataFrame({f"B:{k}": ((c.arm == "B") & (c.bmic == k)).astype(float) for k in CLS})
    be.write(name, X, I, c.deaths.values.astype(float), np.log(c.risk.values.astype(float)), "multi", (0.0, 1.0))
    # strata design for standardisation (arm-free), weights = pregnancies in the stratum
    st = c.groupby(STR).risk.sum().reset_index()
    Xs = pd.get_dummies(st[STR].astype(str), drop_first=True).astype(float).reindex(columns=X.columns, fill_value=0.0)
    np.savez(os.path.join(be.W, f"{name}_strata.npz"), X=Xs.values, w=st.risk.values.astype(float),
             cls=np.array([CLS.index(k) for k in st.bmic]))
    c.to_csv(os.path.join(be.W, f"{name}_cells.tsv"), sep="\t", index=False)


write(cells(), "tte__primary"); write(cells(True), "tte__no_sb39")
c = cells(); js["deaths_A"] = int(c[c.arm == "A"].deaths.sum()); js["deaths_B"] = int(c[c.arm == "B"].deaths.sum())
js["crude_per10000"] = {k: {a: round(float(1e4 * c[(c.bmic == k) & (c.arm == a)].deaths.sum() / c[(c.bmic == k) & (c.arm == a)].risk.sum()), 3)
                            for a in "AB"} for k in CLS}

# secondary outcomes among live births: exact-stratified standardised risk difference (B minus A), Beta posteriors
nat = pop[pop.kind == "nat"]
sec = {}
for o in ["cesar", "nicu", "morb"]:
    g = nat.groupby(STR + ["arm"])[["n", o]].sum().unstack("arm").fillna(0)
    g = g[(g[("n", "A")] > 0) & (g[("n", "B")] > 0)]
    out = {}
    for k in CLS:
        gk = g[g.index.get_level_values("bmic") == k]
        if len(gk) == 0: continue
        w = (gk[("n", "A")] + gk[("n", "B")]).values; w = w / w.sum()
        pa = rng.beta(1 + gk[(o, "A")].values[:, None], 1 + (gk[("n", "A")] - gk[(o, "A")]).values[:, None], (len(gk), 4000))
        pb = rng.beta(1 + gk[(o, "B")].values[:, None], 1 + (gk[("n", "B")] - gk[(o, "B")]).values[:, None], (len(gk), 4000))
        rd = (w[:, None] * (pb - pa)).sum(0) * S
        out[k] = dict(median=round(float(np.median(rd)), 1), lo=round(float(np.quantile(rd, .025)), 1), hi=round(float(np.quantile(rd, .975)), 1),
                      p_B_higher=round(float((rd > 0).mean()), 4), strata=int(len(gk)))
    sec[o] = out
js["secondary_B_minus_A_per10000"] = sec
json.dump(js, open(os.path.join(OUT, "tte_conjugate.json"), "w"), indent=1)
print(json.dumps(js, indent=1))
