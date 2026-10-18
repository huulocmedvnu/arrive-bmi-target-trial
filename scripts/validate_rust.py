"""Check rust_ccw against statsmodels: GLM coefficients, truncation points and point risks for every class and
outcome, then the bootstrap distribution (rust 1,000 draws vs python 48 refitted draws). Writes outputs/rust_validation.json."""
import os, json, time
import numpy as np
import ccw

d = ccw.load()
t0 = time.time(); c, lost = ccw.build(d); P = ccw.prep(c, ccw.CLS)
wA, wB, st, info = ccw.fit_weights(P, c.risk.values.astype(float)); ptp = ccw.risks(c, wA, wB, ccw.point_events(c)); t_py = time.time() - t0
outdir = os.path.join(ccw.W, "rust_ccw", "validate"); ks = ccw.export(c, P, ccw.CLS, outdir)
import subprocess
t0 = time.time(); subprocess.run([ccw.RUST, "point", outdir], check=True); t_rs_point = time.time() - t0
pj = json.load(open(os.path.join(outdir, "point.json")))
rep = {"coef_max_abs_diff": {}, "coef_max_rel_diff": {}, "trunc_diff": {}, "risk_max_rel_diff": {}}
for ki, k in enumerate(ks):
    b1, b2 = np.array(pj["params"][ki][0]), np.array(pj["params"][ki][1])
    rep["coef_max_abs_diff"][k] = float(max(np.abs(b1 - st[k][0]).max(), np.abs(b2 - st[k][1]).max()))
    rep["coef_max_rel_diff"][k] = float(max((np.abs(b1 - st[k][0]) / (np.abs(st[k][0]) + 1e-3)).max(), (np.abs(b2 - st[k][1]) / (np.abs(st[k][1]) + 1e-3)).max()))
    rep["trunc_diff"][k] = [abs(pj["trunc"][ki][0] - info[k]["trunc_A"]), abs(pj["trunc"][ki][1] - info[k]["trunc_B"])]
for oi, o in enumerate(ccw.OUTC):
    rel = []
    for ki, k in enumerate(ks):
        for j in (0, 1):
            a, b = pj["risks"][oi][ki][j], ptp[o][k][j]
            if np.isfinite(a) and np.isfinite(b) and b != 0: rel.append(abs(a - b) / abs(b))
    rep["risk_max_rel_diff"][o] = float(max(rel)) if rel else None
print(json.dumps(rep, indent=1), flush=True)
# bootstrap: rust 1000 draws, python 48 refitted draws
t0 = time.time(); subprocess.run([ccw.RUST, "draws", outdir, "1000", "7", "48"], check=True); t_rs = time.time() - t0
arr = np.fromfile(os.path.join(outdir, "draws.bin"), dtype="<f8").reshape(1000, len(ccw.OUTC), len(ks), 2)
ccw.NPROC = 8
t0 = time.time(); R = ccw.analyze(d, ccw.CLS, 48, 9000); t_pyb = time.time() - t0
cmp = {}
for oi, o in enumerate(["death", "cesar"]):
    oj = list(ccw.OUTC).index(o); cmp[o] = {}
    for ki, k in enumerate(ks):
        rr = (arr[:, oj, ki, 1] - arr[:, oj, ki, 0]) * 1e4; pp = (R["draws"][o][k][:, 1] - R["draws"][o][k][:, 0]) * 1e4
        cmp[o][k] = dict(rust_mean=round(float(rr.mean()), 3), py_mean=round(float(pp.mean()), 3), rust_sd=round(float(rr.std()), 3), py_sd=round(float(pp.std()), 3))
rep["bootstrap_compare"] = cmp
rep["timing_s"] = dict(python_point=round(t_py, 1), rust_point=round(t_rs_point, 1), rust_1000_draws=round(t_rs, 1), python_48_draws_8proc=round(t_pyb, 1))
json.dump(rep, open(os.path.join(ccw.OUT, "rust_validation.json"), "w"), indent=1)
print(json.dumps({k: rep[k] for k in ["bootstrap_compare", "timing_s"]}, indent=1))
