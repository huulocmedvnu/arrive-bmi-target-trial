"""Amendment j: inverse probability of treatment weighting for the amendment i target trial (induction in week 39, A,
versus expectant management, B), low-risk non-anomalous pregnancies ongoing at 39+0, 2019-2024, plus E-values.
Propensity: binomial GLM on covariate-pattern counts within each BMI class. Uncertainty: Bayesian bootstrap over
individuals (Gamma weights for events and non-events in each pattern and arm), propensity held fixed.
Input work/counts_ps.tsv (rust vitals ps mode). Outputs outputs/iptw.json, outputs/iptw_balance.tsv."""
import os, json
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd, statsmodels.api as sm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W, OUT = os.path.join(ROOT, "work"), os.path.join(ROOT, "outputs")
CLS = ["under", "normal", "over", "ob30", "ob35", "ob40", "ob50"]
COLS = ["kind", "year", "ga", "bmic", "bi", "age6", "nullip", "race", "educ", "precare", "wic", "smoke", "height", "inftr",
        "induct", "payer", "married", "trial", "sga", "est", "n", "nicu", "cesar", "morb", "nd",
        "apg7", "aven1", "aven6", "surf", "anti", "seiz", "mtr", "plac", "rupt", "uhyst", "aicu", "smm"]
CAT = ["year", "age6", "nullip", "race", "educ", "precare", "wic", "smoke", "height", "inftr"]
SENS = ["payer", "married"]
OUTC = {"death": ("deaths", "risk"), "cesar": ("cesar", "live"), "nicu": ("nicu", "live"), "morb": ("morb", "live")}
NDRAW, S = 400, 1e4

d = pd.read_csv(os.path.join(W, "counts_ps.tsv"), sep="\t", names=COLS, dtype={"est": str})
d = d[(d.year >= 2019) & (d.year <= 2024) & (d.ga >= 39)].copy()
d["bmic"] = d.bmic.map(dict(enumerate(CLS)))
d["arm"] = np.where((d.ga == 39) & (d.induct == 1) & (d.kind != "fet"), 1, 0)  # 1 = A induction, 0 = B expectant
pop, nat, inf = d[d.kind != "inf"], d[d.kind == "nat"], d[d.kind == "inf"]


def build(keys):
    g = lambda x, col: x.groupby(keys + ["bmic", "arm"])[col].sum()
    c = pd.concat([g(pop, "n").rename("risk"), g(pop[(pop.kind == "fet") & (pop.est == "N")], "n").rename("sb"),
                   g(inf, "nd").rename("nd"), g(nat, "n").rename("live"), g(nat, "cesar"), g(nat, "nicu"), g(nat, "morb")],
                  axis=1).fillna(0).reset_index()
    lost = int(c[c.risk == 0].nd.sum()); c = c[c.risk > 0].reset_index(drop=True)
    c["deaths"] = c.sb + c.nd
    return c, lost


def design(p, keys):
    X = pd.get_dummies(p[[k for k in keys if k != "bi"]].astype(str), drop_first=True).astype(float)
    b = (p.bi - p.bi.mean()) / max(p.bi.std(), 1e-9)
    X["bi"], X["bi2"] = b, b ** 2
    return sm.add_constant(X, has_constant="add")


def wquant(x, w, q):
    o = np.argsort(x); cw = np.cumsum(w[o]) / w.sum()
    return x[o][np.searchsorted(cw, q)]


def weights(c, keys, denom="risk"):
    """Stabilised ATE weights per (pattern, arm) cell, fitted within each BMI class; truncated at 1st/99th centile."""
    c = c.copy(); c["w"] = np.nan; info = {}
    for k in CLS:
        ck = c[c.bmic == k]
        if ck.empty: continue
        p = ck.pivot_table(index=keys, columns="arm", values=denom, aggfunc="sum", fill_value=0).reset_index()
        for a in (0, 1):
            if a not in p: p[a] = 0
        p = p[(p[0] + p[1]) > 0].reset_index(drop=True)
        X = design(p, keys)
        fit = sm.GLM(np.column_stack([p[1], p[0]]), X, family=sm.families.Binomial()).fit()
        p["e"] = np.clip(fit.predict(X), 1e-6, 1 - 1e-6)
        pa = p[1].sum() / (p[0].sum() + p[1].sum())
        m = ck.merge(p[keys + ["e"]], on=keys, how="left")
        w = np.where(m.arm == 1, pa / m.e, (1 - pa) / (1 - m.e))
        lo, hi = wquant(w, m[denom].values, 0.01), wquant(w, m[denom].values, 0.99)
        w = np.clip(w, lo, hi); c.loc[ck.index, "w"] = w
        ess = {a: float((w[m.arm == a] * m[denom][m.arm == a]).sum() ** 2 / (w[m.arm == a] ** 2 * m[denom][m.arm == a]).sum())
               for a in (0, 1)}
        info[k] = dict(patterns=int(len(p)), n_A=int(p[1].sum()), n_B=int(p[0].sum()), trunc=[round(float(lo), 3), round(float(hi), 3)],
                       ess_A=round(ess[1]), ess_B=round(ess[0]), converged=bool(fit.converged))
    return c, info


def balance(c, keys, denom="risk"):
    rows = []
    for k in CLS:
        ck = c[c.bmic == k]
        if ck.empty: continue
        D = pd.get_dummies(ck[[x for x in keys if x != "bi"]].astype(str)).astype(float); D["bmi"] = ck.bi.values
        for v in D.columns:
            x = D[v].values; r = {"bmic": k, "var": v}
            sd = None
            for lab, wt in (("raw", ck[denom].values), ("weighted", ck[denom].values * ck.w.values)):
                mA = np.average(x[ck.arm == 1], weights=wt[ck.arm == 1]); mB = np.average(x[ck.arm == 0], weights=wt[ck.arm == 0])
                if sd is None:
                    vA = np.average((x[ck.arm == 1] - mA) ** 2, weights=wt[ck.arm == 1]); vB = np.average((x[ck.arm == 0] - mB) ** 2, weights=wt[ck.arm == 0])
                    sd = np.sqrt((vA + vB) / 2) or 1.0
                r[f"smd_{lab}"] = (mA - mB) / sd
            rows.append(r)
    return pd.DataFrame(rows)


def risks(c, wE, wN, outc):
    """Weighted risk per class and arm; wE/wN multiply events/non-events (1 for the point estimate)."""
    res = {}
    for o in outc:
        ev, den = OUTC[o]
        e = c[ev].values * wE[o] if wE is not None else c[ev].values
        ne = (c[den].values - c[ev].values) * wN[o] if wN is not None else c[den].values - c[ev].values
        t = pd.DataFrame({"bmic": c.bmic, "arm": c.arm, "e": c.w * e, "t": c.w * (e + ne)}).groupby(["bmic", "arm"]).sum()
        r = (t.e / t.t).unstack("arm")
        res[o] = {k: (float(r.loc[k, 1]), float(r.loc[k, 0])) for k in r.index}
    return res


G = {}


def one(seed):
    c, outc = G["c"], G["outc"]; rng = np.random.default_rng(seed)
    wE, wN = {}, {}
    for o in outc:
        ev, den = OUTC[o]; k1 = c[ev].values; k0 = c[den].values - k1
        wE[o] = np.where(k1 > 0, rng.gamma(np.maximum(k1, 1e-9)) / np.maximum(k1, 1), 0)
        wN[o] = np.where(k0 > 0, rng.gamma(np.maximum(k0, 1e-9)) / np.maximum(k0, 1), 0)
    return risks(c, wE, wN, outc)


def summarise(c, outc, ndraw, seed0):
    pt = risks(c, None, None, outc); G["c"], G["outc"] = c, outc
    with ProcessPoolExecutor(48) as ex: dr = list(ex.map(one, range(seed0, seed0 + ndraw)))
    out = {}
    for o in outc:
        out[o] = {}
        for k in pt[o]:
            a = np.array([x[o][k] for x in dr]); rd = (a[:, 1] - a[:, 0]) * S; rr = a[:, 1] / np.where(a[:, 0] > 0, a[:, 0], np.nan)
            q = lambda v, p: round(float(np.nanquantile(v, p)), 3)
            out[o][k] = dict(risk_A=round(pt[o][k][0] * S, 3), risk_B=round(pt[o][k][1] * S, 3),
                             rd=round((pt[o][k][1] - pt[o][k][0]) * S, 3), rd_lo=q(rd, .025), rd_hi=q(rd, .975),
                             rr=round(pt[o][k][1] / pt[o][k][0], 3) if pt[o][k][0] > 0 else None, rr_lo=q(rr, .025), rr_hi=q(rr, .975),
                             p_B_higher=round(float((rd > 0).mean()), 4))
    return out


def evalue(rr):
    if rr is None or not np.isfinite(rr): return None
    rr = rr if rr >= 1 else 1 / rr
    return round(float(rr + np.sqrt(rr * (rr - 1))), 2)


if __name__ == "__main__":
    js = {}
    keys = ["bi"] + CAT
    c, lost = build(keys); js["neonatal_deaths_unmatched"] = lost
    c, js["ps"] = weights(c, keys)
    bal = balance(c, keys); bal["analysis"] = "primary"
    js["balance_max_abs_smd"] = {k: dict(raw=round(float(bal[bal.bmic == k].smd_raw.abs().max()), 3),
                                         weighted=round(float(bal[bal.bmic == k].smd_weighted.abs().max()), 3)) for k in CLS if k in set(bal.bmic)}
    js["primary"] = summarise(c, list(OUTC), NDRAW, 1000)
    print(json.dumps(js["primary"]["death"], indent=1), flush=True)
    # sensitivity among live births with payer and marital status (not recorded on fetal death certificates)
    k2 = keys + SENS
    c2, _ = build(k2); c2 = c2[c2.live > 0].reset_index(drop=True); c2["deaths_live"] = c2.nd
    OUTC["neonatal_death"] = ("deaths_live", "live")
    c2, js["ps_sens"] = weights(c2, k2, "live")
    b2 = balance(c2, k2, "live"); b2["analysis"] = "live_births_payer_marital"
    js["sens_balance_max_abs_smd"] = {k: round(float(b2[b2.bmic == k].smd_weighted.abs().max()), 3) for k in CLS if k in set(b2.bmic)}
    js["sens_live_births"] = summarise(c2, ["neonatal_death", "cesar"], 200, 5000)
    # E-values for perinatal death, B versus A (amendment i model-based and amendment j weighted)
    tte = json.load(open(os.path.join(OUT, "tte.json")))
    js["evalue"] = {}
    for src, blk in (("tte_primary", tte["tte__primary"]), ("tte_no_sb39", tte["tte__no_sb39"]), ("iptw", js["primary"]["death"])):
        js["evalue"][src] = {k: dict(rr=v["rr"], e_point=evalue(v["rr"]), e_lower=evalue(v["rr_lo"]) if v["rr_lo"] and v["rr_lo"] > 1 else 1.0)
                             for k, v in blk.items()}
    pd.concat([bal, b2]).round(4).to_csv(os.path.join(OUT, "iptw_balance.tsv"), sep="\t", index=False)
    json.dump(js, open(os.path.join(OUT, "iptw.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in js.items() if k not in ("primary",)}, indent=1))
