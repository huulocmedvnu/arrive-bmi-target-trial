"""Amendments m and n: exploratory clone-censor-weight subgroups (weights refitted in every draw) and the BMI curve in
2-unit bins with a natural cubic spline (4 df) fitted within each draw. Output outputs/ccw_sub.json."""
import os, json
import numpy as np, pandas as pd
from patsy import dmatrix, build_design_matrices
import ccw

OUT = ccw.OUT; S = 1e4; ND = 400
SUB = {"parity": ("nullip", {"nulliparous": [1], "parous": [0]}),
       "age": ("age6", {"under35": [0, 1, 2, 3], "35plus": [4, 5]}),
       "race": ("race", {"white": [1], "black": [2], "hispanic": [7], "asian": [4]}),
       "period": ("year", {"2019_2021": [2019, 2020, 2021], "2022_2024": [2022, 2023, 2024]})}
d0 = ccw.load()
c0, lost0 = ccw.build(d0)            # aggregate once; subgroup columns are part of the cell keys
js = {"subgroups": {}}
for name, (col, levels) in SUB.items():
    js["subgroups"][name] = {}
    for lev, vals in levels.items():
        R = ccw.analyze_rust(None, ccw.CLS, ND, 30000, f"sub_{name}_{lev}", cells=c0[c0[col].isin(vals)])
        js["subgroups"][name][lev] = ccw.summarise(R, ["death", "cesar"])
        print(name, lev, {k: v["rd"] for k, v in js["subgroups"][name][lev]["death"].items()}, flush=True)
lab = lambda b: "<18" if b < 18 else ("50+" if b >= 50 else f"{2 * (b // 2)}-{2 * (b // 2) + 1}.9")
c1 = c0.copy(); c1["bmic"] = c1.bi.map(lab)
bins = ["<18"] + [f"{e}-{e + 1}.9" for e in range(18, 50, 2)] + ["50+"]
R = ccw.analyze_rust(None, bins, ND, 40000, "curve", cells=c1)
js["curve"] = ccw.summarise(R, ["death"])["death"]
mid = np.array([17.0] + [e + 1.0 for e in range(18, 50, 2)] + [51.0])
grid = np.arange(17, 51.01, 0.5)
Bd = dmatrix("cr(x, df=4)", {"x": mid}, return_type="dataframe"); B = np.asarray(Bd)
Bg = np.asarray(build_design_matrices([Bd.design_info], {"x": grid})[0])  # same knots as the fit
knots = [float(k) for k in list(Bd.design_info.factor_infos.values())[0].state["transforms"]["_patsy_stobj0__cr__"]._all_knots]
dr = np.stack([(R["draws"]["death"][b][:, 1] - R["draws"]["death"][b][:, 0]) * S for b in bins], 1)
wt = 1 / np.maximum(np.nanvar(dr, 0), 1e-6)
fits = []
for i in range(dr.shape[0]):
    y = dr[i]; ok = np.isfinite(y)
    beta = np.linalg.lstsq(B[ok] * np.sqrt(wt[ok])[:, None], y[ok] * np.sqrt(wt[ok]), rcond=None)[0]; fits.append(Bg @ beta)
fits = np.array(fits)
y0 = np.array([js["curve"][b]["rd"] for b in bins]); beta0 = np.linalg.lstsq(B * np.sqrt(wt)[:, None], y0 * np.sqrt(wt), rcond=None)[0]
js["spline"] = dict(knots=knots, bmi=grid.tolist(), rd=(Bg @ beta0).tolist(), lo=np.quantile(fits, .025, 0).tolist(), hi=np.quantile(fits, .975, 0).tolist(),
                    nnd=(S / (Bg @ beta0)).tolist(), nnd_lo=(S / np.quantile(fits, .975, 0)).tolist(), nnd_hi=(S / np.maximum(np.quantile(fits, .025, 0), 1e-3)).tolist())
json.dump(js, open(os.path.join(OUT, "ccw_sub.json"), "w"), indent=1)
print("spline at 22, 30, 40:", [round(float(np.interp(x, grid, Bg @ beta0)), 2) for x in (22, 30, 40)])
