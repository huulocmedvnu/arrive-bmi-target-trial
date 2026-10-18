"""Exact check of one rust bootstrap draw on a small subset (2019 only): python refits both treatment models with
statsmodels on the same drawn totals and computes risks from the same drawn events; both must agree."""
import os, json, subprocess, time
import numpy as np
import ccw
d = ccw.load(); d = d[d.year == 2019].copy()
c, lost = ccw.build(d); P = ccw.prep(c, ccw.CLS)
_, _, st, _ = ccw.fit_weights(P, c.risk.values.astype(float))
out = os.path.join(ccw.W, "rust_ccw", "check2019"); ks = ccw.export(c, P, ccw.CLS, out)
rep = {"cells": len(c)}
for j in (0, 1, 2):
    subprocess.run([ccw.RUST, "dump", out, "7", str(j)], check=True)
    raw = np.fromfile(os.path.join(out, "dump.bin"), dtype="<f8").reshape(1 + len(ccw.OUTC), len(c)); T = raw[0]
    rj = json.load(open(os.path.join(out, "dump.json")))
    t0 = time.time(); wA, wB, stj, _ = ccw.fit_weights(P, T, st); tpy = time.time() - t0
    ev = {o: (raw[1 + i], (T if den == "risk" else T * c.live.values / c.risk.values)) for i, (o, (_, den)) in enumerate(ccw.OUTC.items())}
    rp = ccw.risks(c, wA, wB, ev)
    coef = max(max(np.abs(np.array(rj["params"][ki][0]) - stj[k][0]).max(), np.abs(np.array(rj["params"][ki][1]) - stj[k][1]).max()) for ki, k in enumerate(ks))
    rel = max(abs(rj["risks"][oi][ki][s] - rp[o][k][s]) / abs(rp[o][k][s]) for oi, o in enumerate(ccw.OUTC) for ki, k in enumerate(ks) for s in (0, 1) if rp[o][k][s] and np.isfinite(rp[o][k][s]))
    rep[f"draw{j}"] = dict(coef_max_abs_diff=float(coef), risk_max_rel_diff=float(rel), python_refit_seconds=round(tpy, 2))
    print(j, rep[f"draw{j}"], flush=True)
p = os.path.join(ccw.OUT, "rust_validation.json"); r0 = json.load(open(p)); r0["exact_draw_check_2019"] = rep; json.dump(r0, open(p, "w"), indent=1)
