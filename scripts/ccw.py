"""Amendments k and n: clone-censor-weight emulation of delivery in week 39 (A) versus expectant management in week 39
(B), week 39 as grace period, low-risk non-anomalous pregnancies ongoing at 39+0, 2019-2024. Courses: I induced and
born in week 39, S other end of pregnancy in week 39, O ongoing at 40+0. A: S weight 1, I weight 1 + P(O|X)/P(I|X),
O censored. B: I censored, S and O weight 1/(1 - P(I|X)). Bayesian bootstrap with both treatment models refitted in
every draw. Also balance, sensitivity analyses, bias analysis and population impact. Output outputs/ccw.json."""
import os, json, warnings
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd, statsmodels.api as sm
from iptw import COLS, CAT, CLS, design, wquant, evalue
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W, OUT = os.path.join(ROOT, "work"), os.path.join(ROOT, "outputs")
OUTC = {"death": ("deaths", "risk"), "cesar": ("cesar", "live"), "nicu": ("nicu", "live"), "morb": ("morb", "live"),
        "stillbirth": ("sb", "risk"), "sb_ante": ("sb_ante", "risk"), "neonatal": ("nd", "risk"), "death_ante": ("deaths_ante", "risk"),
        "sbS": ("sbS", "risk")}
SAFETY = ["apg7", "aven1", "aven6", "surf", "anti", "seiz", "mtr", "plac", "rupt", "uhyst", "aicu", "smm"]
for _v in SAFETY: OUTC[_v] = (_v, "live")
KEYS = ["bi"] + CAT
S = 1e4
OB = ["ob30", "ob35", "ob40", "ob50"]
NPROC = 12


def load():
    d = pd.read_csv(os.path.join(W, "counts_ps.tsv"), sep="\t", names=COLS, dtype={"est": str})
    d = d[(d.year >= 2019) & (d.year <= 2024) & (d.ga >= 39)].copy()
    d["bmic"] = d.bmic.map(dict(enumerate(CLS)))
    d["course"] = np.select([(d.ga == 39) & (d.induct == 1) & (d.kind != "fet"), d.ga == 39], [0, 1], 2)
    return d


def build(d):
    pop, nat, inf = d[d.kind != "inf"], d[d.kind == "nat"], d[d.kind == "inf"]
    g = lambda x, col: x.groupby(KEYS + ["bmic", "course"])[col].sum()
    fet = pop[pop.kind == "fet"]
    c = pd.concat([g(pop, "n").rename("risk"), g(fet, "n").rename("sb"), g(fet[fet.est == "N"], "n").rename("sb_ante"),
                   g(inf, "nd").rename("nd"), g(nat, "n").rename("live"), g(nat, "cesar"), g(nat, "nicu"), g(nat, "morb")]
                  + [g(nat, v) for v in SAFETY],
                  axis=1).fillna(0).reset_index()
    lost = int(c[c.risk == 0].nd.sum()); c = c[c.risk > 0].reset_index(drop=True)
    c["live"] = np.minimum(c.live, c.risk)
    c["deaths"] = np.minimum(c.sb + c.nd, c.risk); c["deaths_ante"] = np.minimum(c.sb_ante + c.nd, c.risk)
    c["sbS"] = np.where(c.course == 1, c.sb, 0)
    return c, lost


def prep(c, cls):
    P = {}
    for k in cls:
        idx = np.where(c.bmic.values == k)[0]
        if len(idx) == 0: continue
        ck = c.iloc[idx]
        pid = ck.groupby(KEYS, sort=False).ngroup().values
        first = pd.Series(np.arange(len(ck))).groupby(pid).first().values
        pat = ck.iloc[first][KEYS].reset_index(drop=True)
        P[k] = dict(idx=idx, pid=pid, course=ck.course.values, X=design(pat, KEYS).values, npat=len(pat))
    return P


def fit_weights(P, T, start=None):
    n_all = len(T); wA = np.zeros(n_all); wB = np.zeros(n_all); st = {}; info = {}
    for k, p in P.items():
        t = T[p["idx"]]; cr = p["course"]; X = p["X"]
        n = [np.bincount(p["pid"][cr == a], weights=t[cr == a], minlength=p["npat"]) for a in (0, 1, 2)]
        tot = n[0] + n[1] + n[2]; m = tot > 0; non = n[1] + n[2]; m2 = non > 0
        s0 = start[k] if start else (None, None)
        f1 = sm.GLM(n[0][m] / tot[m], X[m], family=sm.families.Binomial(), var_weights=tot[m]).fit(start_params=s0[0], maxiter=100)
        f2 = sm.GLM(n[2][m2] / non[m2], X[m2], family=sm.families.Binomial(), var_weights=non[m2]).fit(start_params=s0[1], maxiter=100)
        pI = np.clip(f1.predict(X), 1e-4, 1 - 1e-4); pO = (1 - pI) * np.clip(f2.predict(X), 1e-6, 1 - 1e-6)
        pi, po = pI[p["pid"]], pO[p["pid"]]
        a = np.select([cr == 0, cr == 1], [1 + po / pi, 1.0], 0.0); b = np.where(cr == 0, 0.0, 1 / (1 - pi))
        hiA = wquant(a[cr == 0], t[cr == 0], 0.99); hiB = wquant(b[cr > 0], t[cr > 0], 0.99)
        wA[p["idx"]] = np.minimum(a, hiA); wB[p["idx"]] = np.minimum(b, hiB)
        st[k] = (f1.params, f2.params)
        info[k] = dict(n_I=float(n[0].sum()), n_S=float(n[1].sum()), n_O=float(n[2].sum()), trunc_A=round(float(hiA), 2), trunc_B=round(float(hiB), 3),
                       converged=bool(f1.converged and f2.converged))
    return wA, wB, st, info


def risks(c, wA, wB, ev):
    """ev: dict outcome -> (events array, denominator array). Grouped sums by BMI class with bincount."""
    if "_code" not in G or G.get("_n") != len(c):
        cat = pd.Categorical(c.bmic.values); G["_code"] = cat.codes; G["_cats"] = list(cat.categories); G["_n"] = len(c)
    code, cats = G["_code"], G["_cats"]; K = len(cats); res = {}
    for o, (e, den) in ev.items():
        eA = np.bincount(code, wA * e, K); tA = np.bincount(code, wA * den, K); eB = np.bincount(code, wB * e, K); tB = np.bincount(code, wB * den, K)
        res[o] = {k: (float(eA[i] / tA[i]) if tA[i] > 0 else np.nan, float(eB[i] / tB[i]) if tB[i] > 0 else np.nan) for i, k in enumerate(cats)}
    return res


def point_events(c):
    return {o: (c[e].values.astype(float), c[den].values.astype(float)) for o, (e, den) in OUTC.items()}


def draw_events(c, rng):
    risk = c.risk.values.astype(float)
    T = rng.gamma(risk)                       # cell totals, Dirichlet weights over individuals
    frac_live = np.where(risk > 0, c.live.values / risk, 0)
    ev = {}
    for o, (e, den) in OUTC.items():
        k1 = c[e].values.astype(float); nd = c[den].values.astype(float)
        D = T if den == "risk" else T * frac_live
        p = np.where(k1 <= 0, 0.0, np.where(k1 >= nd, 1.0, rng.beta(np.maximum(k1, 1e-9), np.maximum(nd - k1, 1e-9))))
        ev[o] = (D * p, D)
    return T, ev


G = {}


def one(seed):
    c, P, st = G["c"], G["P"], G["start"]; rng = np.random.default_rng(seed)
    T, ev = draw_events(c, rng)
    wA, wB, _, _ = fit_weights(P, T, st) if G["refit"] else (G["wA"], G["wB"], None, None)
    r = risks(c, wA, wB, ev); del ev, T
    return r


def analyze(d, cls, ndraw, seed0, refit=True):
    c, lost = build(d); P = prep(c, cls); G.pop("_code", None)
    wA, wB, st, info = fit_weights(P, c.risk.values.astype(float))
    pt = risks(c, wA, wB, point_events(c))
    G.update(c=c, P=P, start=st, refit=refit, wA=wA, wB=wB)
    os.environ["OMP_NUM_THREADS"] = "1"
    with ProcessPoolExecutor(NPROC) as ex: dr = list(ex.map(one, range(seed0, seed0 + ndraw), chunksize=2))
    draws = {o: {k: np.array([x[o].get(k, (np.nan, np.nan)) for x in dr]) for k in pt[o]} for o in pt}
    return dict(c=c, P=P, wA=wA, wB=wB, info=info, lost=lost, pt=pt, draws=draws)


def export(c, P, cls, outdir):
    """Write the cell table and per-class design matrices for the rust bootstrap (rust_ccw)."""
    os.makedirs(outdir, exist_ok=True)
    ks = [k for k in cls if k in P]; kidx = {k: i for i, k in enumerate(ks)}
    pid = np.zeros(len(c)); clsv = np.zeros(len(c))
    for k in ks: pid[P[k]["idx"]] = P[k]["pid"]; clsv[P[k]["idx"]] = kidx[k]
    cols = [clsv, pid, c.course.values.astype(float), c.risk.values.astype(float), c.live.values.astype(float)] + [c[e].values.astype(float) for e, _ in OUTC.values()]
    np.concatenate(cols).astype("<f8").tofile(os.path.join(outdir, "cells.bin"))
    for k in ks: np.ascontiguousarray(P[k]["X"], dtype="<f8").tofile(os.path.join(outdir, f"X_{kidx[k]}.bin"))
    json.dump(dict(ncells=len(c), outcomes=[dict(name=o, den=den) for o, (_, den) in OUTC.items()],
                   classes=[dict(name=k, npat=P[k]["npat"], p=P[k]["X"].shape[1]) for k in ks]), open(os.path.join(outdir, "meta.json"), "w"))
    return ks


RUST = os.path.join(ROOT, "rust_ccw", "target", "release", "ccw")


def analyze_rust(d, cls, ndraw, seed, tag, threads=48, cells=None):
    """Same as analyze(), with point fit, point weights and refitted bootstrap done by rust_ccw. Pass cells (an
    already built cell table, filtered or relabelled) to skip rebuilding from records."""
    import subprocess
    if cells is None: c, lost = build(d)
    else: c, lost = cells.reset_index(drop=True), 0
    P = prep(c, cls); G.pop("_code", None)
    outdir = os.path.join(W, "rust_ccw", tag); ks = export(c, P, cls, outdir)
    subprocess.run([RUST, "point", outdir], check=True)
    pj = json.load(open(os.path.join(outdir, "point.json")))
    subprocess.run([RUST, "draws", outdir, str(ndraw), str(seed), str(threads)], check=True)
    arr = np.fromfile(os.path.join(outdir, "draws.bin"), dtype="<f8").reshape(ndraw, len(OUTC), len(ks), 2)
    outs = list(OUTC)
    pt = {o: {k: tuple(pj["risks"][oi][ki]) for ki, k in enumerate(ks)} for oi, o in enumerate(outs)}
    draws = {o: {k: arr[:, oi, ki, :] for ki, k in enumerate(ks)} for oi, o in enumerate(outs)}
    info = {}
    for ki, k in enumerate(ks):
        cr = P[k]["course"]; r = c.risk.values[P[k]["idx"]]
        info[k] = dict(n_I=float(r[cr == 0].sum()), n_S=float(r[cr == 1].sum()), n_O=float(r[cr == 2].sum()),
                       trunc_A=round(pj["trunc"][ki][0], 2), trunc_B=round(pj["trunc"][ki][1], 3), converged=pj["converged"][ki])
    w = np.fromfile(os.path.join(outdir, "point_weights.bin"), dtype="<f8").reshape(2, -1); wA, wB = w[0], w[1]   # rust point weights
    return dict(c=c, P=P, wA=wA, wB=wB, info=info, lost=lost, pt=pt, draws=draws, rust_point=pj)


q = lambda v, p: round(float(np.nanquantile(v, p)), 3)


def summarise(R, outcomes=None):
    out = {}
    for o in (outcomes or R["pt"]):
        out[o] = {}
        for k, (a, b) in R["pt"][o].items():
            dd = R["draws"][o][k]; rd = (dd[:, 1] - dd[:, 0]) * S; rr = dd[:, 1] / np.where(dd[:, 0] > 0, dd[:, 0], np.nan)
            out[o][k] = dict(risk_A=round(a * S, 3), risk_B=round(b * S, 3), rd=round((b - a) * S, 3), rd_lo=q(rd, .025), rd_hi=q(rd, .975),
                             rr=round(b / a, 3) if a > 0 else None, rr_lo=q(rr, .025), rr_hi=q(rr, .975), p_B_higher=round(float(np.mean(rd > 0)), 4))
    return out


def balance(R, cls):
    c, wA, wB = R["c"], R["wA"], R["wB"]; rows = []
    for k in cls:
        ck = c[c.bmic == k]; i = ck.index.values
        D = pd.get_dummies(ck[[x for x in KEYS if x != "bi"]].astype(str)).astype(float); D["bmi"] = ck.bi.values
        w0 = ck.risk.values
        for v in D.columns:
            x = D[v].values; m0 = np.average(x, weights=w0); sd = np.sqrt(np.average((x - m0) ** 2, weights=w0)) or 1.0
            cr = ck.course.values
            mI = np.average(x[cr == 0], weights=w0[cr == 0]); mO = np.average(x[cr == 2], weights=w0[cr == 2])
            mA = np.average(x, weights=w0 * wA[i]); mB = np.average(x, weights=w0 * wB[i])
            rows.append(dict(bmic=k, var=v, raw_I_vs_O=(mI - mO) / sd, A_vs_population=(mA - m0) / sd, B_vs_population=(mB - m0) / sd))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    js = {}
    d = load()
    R = analyze_rust(d, CLS, 1000, 7000, "primary")
    js["neonatal_deaths_unmatched"] = R["lost"]
    m = {}
    for k, v in R["info"].items():
        ck = R["c"][R["c"].bmic == k]
        dth = ck.groupby("course")[["sb", "nd"]].sum()
        m[k] = dict(n_I=int(v["n_I"]), n_S=int(v["n_S"]), n_O=int(v["n_O"]), trunc_A=v["trunc_A"], trunc_B=v["trunc_B"], converged=v["converged"],
                    stillbirths_week39=int(dth.sb.get(1, 0)), deaths_I=int(dth.sum(1).get(0, 0)), deaths_S=int(dth.sum(1).get(1, 0)), deaths_O=int(dth.sum(1).get(2, 0)))
    js["models"] = m
    js["primary"] = summarise(R)
    rdd = {(o, k): (R["draws"][o][k][:, 1] - R["draws"][o][k][:, 0]) * S for o in R["draws"] for k in R["draws"][o]}
    js["gradient_rd_minus_normal"] = {o: {k: dict(diff=round(float(np.median(rdd[(o, k)] - rdd[(o, "normal")])), 3), lo=q(rdd[(o, k)] - rdd[(o, "normal")], .025),
                                                  hi=q(rdd[(o, k)] - rdd[(o, "normal")], .975), p_larger=round(float((rdd[(o, k)] > rdd[(o, "normal")]).mean()), 4)) for k in OB}
                                      for o in ["death", "cesar", "nicu", "morb", "death_ante"]}
    js["evalue_death"] = {k: dict(rr=v["rr"], e_point=evalue(v["rr"]), e_lower=evalue(v["rr_lo"]) if v["rr_lo"] > 1 else 1.0) for k, v in js["primary"]["death"].items()}
    # quantitative bias analysis: fraction x of week-39 stillbirths preventable under A
    js["qba"] = {}
    for x in (0.25, 0.5):
        js["qba"][str(x)] = {}
        for k in CLS:
            dA = R["draws"]["death"][k][:, 0] - x * R["draws"]["sbS"][k][:, 0]; dB = R["draws"]["death"][k][:, 1]
            a = R["pt"]["death"][k][0] - x * R["pt"]["sbS"][k][0]; b = R["pt"]["death"][k][1]; rd = (dB - dA) * S; rr = dB / dA
            js["qba"][str(x)][k] = dict(rd=round((b - a) * S, 3), rd_lo=q(rd, .025), rd_hi=q(rd, .975), rr=round(b / a, 3), rr_lo=q(rr, .025), rr_hi=q(rr, .975))
    # balance
    bal = balance(R, CLS); bal.round(4).to_csv(os.path.join(OUT, "ccw_balance.tsv"), sep="\t", index=False)
    js["balance_max_abs"] = {k: {c_: round(float(bal[bal.bmic == k][c_].abs().max()), 3) for c_ in ["raw_I_vs_O", "A_vs_population", "B_vs_population"]} for k in CLS}
    # population impact (observed minus risk under A), per year
    js["impact"] = {}; YEARS = 6
    c = R["c"]
    for o, (ev, den) in {"death": ("deaths", "risk"), "cesar": ("cesar", "live")}.items():
        js["impact"][o] = {}; DR = {}
        for k in CLS:
            ck = c[c.bmic == k]; obs = ck[ev].sum() / ck[den].sum(); n = ck[den].sum()
            A = R["draws"][o][k][:, 0]; DR[k] = (obs - A) * n / YEARS
            js["impact"][o][k] = dict(observed=obs * S, under_A=R["pt"][o][k][0] * S, per_year=(obs - R["pt"][o][k][0]) * n / YEARS,
                                      lo=q(DR[k], .025), hi=q(DR[k], .975), n_per_year=n / YEARS)
        for grp, ks in (("obese", OB), ("all", CLS)):
            tot = sum(DR[k] for k in ks)
            js["impact"][o][grp] = dict(per_year=sum(js["impact"][o][k]["per_year"] for k in ks), lo=q(tot, .025), hi=q(tot, .975),
                                        n_per_year=sum(js["impact"][o][k]["n_per_year"] for k in ks))
    json.dump(js, open(os.path.join(OUT, "ccw.json"), "w"), indent=1)
    print("primary death", {k: (v["rd"], v["rd_lo"], v["rd_hi"]) for k, v in js["primary"]["death"].items()}, flush=True)
    # sensitivity analyses (400 draws each)
    sens = {}
    d1 = d.copy(); d1.loc[(d1.ga == 40) & (d1.induct == 1) & (d1.kind != "fet"), "course"] = 0
    d2 = d[~((d.kind != "fet") & (d.ga == 39) & (d.induct != 1) & (d.trial == 0))].copy()
    d3 = d[d.sga != 1].copy()
    for name, dd in (("late_induction_to_I", d1), ("no_prelabor_cesarean_wk39", d2), ("exclude_sga", d3)):
        Rs = analyze_rust(dd, CLS, 400, 20000, name)
        sens[name] = summarise(Rs, ["death", "cesar"]); sens[name]["n"] = int(Rs["c"].risk.sum())
        print(name, {k: (v["rd"], v["rd_lo"], v["rd_hi"]) for k, v in sens[name]["death"].items()}, flush=True)
    js["sensitivity"] = sens
    js["n_sga"] = int(d[(d.sga == 1) & (d.kind != "inf")].n.sum()); js["n_prelabor_cs_wk39"] = int(d[(d.kind == "nat") & (d.ga == 39) & (d.induct != 1) & (d.trial == 0)].n.sum())
    js["n_induced_wk40"] = int(d[(d.kind == "nat") & (d.ga == 40) & (d.induct == 1)].n.sum())
    json.dump(js, open(os.path.join(OUT, "ccw.json"), "w"), indent=1)
    print(json.dumps({k: js[k] for k in ["neonatal_deaths_unmatched", "balance_max_abs", "n_sga", "n_prelabor_cs_wk39", "n_induced_wk40"]}, indent=1))
