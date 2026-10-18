"""Bootstrap distribution check: rust 1,000 refitted draws (work/rust_ccw/validate/draws.bin) vs python statsmodels
16 refitted draws (2 processes). Appends to outputs/rust_validation.json."""
import os, json
import numpy as np
import ccw
d = ccw.load(); ccw.NPROC = 2
R = ccw.analyze(d, ccw.CLS, 16, 9000)
meta = json.load(open(os.path.join(ccw.W, "rust_ccw", "validate", "meta.json"))); ks = [c["name"] for c in meta["classes"]]
arr = np.fromfile(os.path.join(ccw.W, "rust_ccw", "validate", "draws.bin"), dtype="<f8").reshape(-1, len(ccw.OUTC), len(ks), 2)
cmp = {}
for o in ["death", "cesar"]:
    oj = list(ccw.OUTC).index(o); cmp[o] = {}
    for ki, k in enumerate(ks):
        rr = (arr[:, oj, ki, 1] - arr[:, oj, ki, 0]) * 1e4; pp = (R["draws"][o][k][:, 1] - R["draws"][o][k][:, 0]) * 1e4
        cmp[o][k] = dict(rust_mean=round(float(rr.mean()), 2), py_mean=round(float(pp.mean()), 2), rust_sd=round(float(rr.std()), 3), py_sd=round(float(pp.std(ddof=1)), 3),
                         py_mean_z=round(float((pp.mean() - rr.mean()) / (rr.std() / np.sqrt(len(pp)))), 2))
p = os.path.join(ccw.OUT, "rust_validation.json"); rep = json.load(open(p)) if os.path.exists(p) else {}
rep["bootstrap_compare"] = cmp; json.dump(rep, open(p, "w"), indent=1)
print(json.dumps(cmp, indent=1))
