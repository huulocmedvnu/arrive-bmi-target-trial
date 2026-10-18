"""Diagnostic for neonatal deaths whose covariate pattern has no pregnancy at risk (amendment p). Linked records carry the
year of death, so each unmatched death is re-matched with its year set back and, as a control, set forward by one.
Output outputs/nd_unmatched.json."""
import json, os
from ccw import load, KEYS, OUT

d = load(); K = KEYS + ["bmic", "course"]
have = set(map(tuple, d[d.kind != "inf"][K].drop_duplicates().values))
inf = d[(d.kind == "inf") & (d.nd > 0)]
lost = inf[[tuple(r) not in have for r in inf[K].values]]
def shifted(dy):
    s = lost.copy(); s["year"] = s.year + dy
    return int(sum(n for r, n in zip(s[K].values, s.nd) if tuple(r) in have))
js = dict(unmatched=int(lost.nd.sum()), match_year_minus1=shifted(-1), match_year_plus1=shifted(1))
json.dump(js, open(os.path.join(OUT, "nd_unmatched.json"), "w"), indent=1); print(js)
