"""Numbers and tables for the manuscript. Reads outputs/ (ccw.json, iptw.json, tte.json, flow_counts.tsv) and
work/counts_ps.tsv (Table 1). Writes outputs/manuscript_numbers.json (flat dict of formatted strings, tables as
markdown)."""
import os, json
import numpy as np, pandas as pd
from iptw import COLS, CLS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W, OUT = os.path.join(ROOT, "work"), os.path.join(ROOT, "outputs")
LAB = {"under": "<18.5", "normal": "18.5–24.9", "over": "25.0–29.9", "ob30": "30.0–34.9", "ob35": "35.0–39.9",
       "ob40": "40.0–49.9", "ob50": "≥50.0"}
ccw = json.load(open(os.path.join(OUT, "ccw.json"))); ip = json.load(open(os.path.join(OUT, "iptw.json")))
tte = json.load(open(os.path.join(OUT, "tte.json")))
N = {}
c = lambda x: f"{int(round(x)):,}"
f1 = lambda x: f"{x:.1f}"; f2 = lambda x: f"{x:.2f}"
mn = lambda s: s.replace("-", "−")
ci1 = lambda v: mn(f"{v['rd']:.1f} ({v['rd_lo']:.1f} to {v['rd_hi']:.1f})")
cir = lambda v: f"{v['rr']:.2f} ({v['rr_lo']:.2f} to {v['rr_hi']:.2f})"
pp = lambda p: ">0.99" if p > 0.995 else f"{p:.2f}"

# flow
fl = pd.read_csv(os.path.join(OUT, "flow_counts.tsv"), sep="\t", header=None, usecols=range(5), names=["tag", "kind", "year", "step", "n"])
g = fl.groupby(["kind", "step"]).n.sum().unstack("kind").fillna(0)
N["flow_all"] = c(g.sum().sum()); N["flow_all_nat"] = c(g["nat"].sum()); N["flow_all_fet"] = c(g["fet"].sum())
names = {1: "multiple", 2: "ga", 3: "bmi", 4: "risk", 5: "anom", 6: "final"}
rem = g.sum(1)
left = g.sum().sum()
for s in range(1, 7):
    N[f"flow_ex_{names[s]}"] = c(rem[s]); N[f"flow_ex_{names[s]}_nat"] = c(g.loc[s, "nat"]); N[f"flow_ex_{names[s]}_fet"] = c(g.loc[s, "fet"])
    if s < 6: left -= rem[s]; N[f"flow_left_after_{names[s]}"] = c(left)

# cohort and deaths
m = ccw["models"]
tot = lambda key: sum(v[key] for v in m.values())
nI, nS, nO = tot("n_I"), tot("n_S"), tot("n_O"); n = nI + nS + nO
N.update(n_cohort=c(n), n_I=c(nI), n_S=c(nS), n_O=c(nO), pct_I=f1(100 * nI / n), pct_S=f1(100 * nS / n), pct_O=f1(100 * nO / n),
         sb_wk39=c(tot("stillbirths_week39")), deaths_total=c(tot("deaths_I") + tot("deaths_S") + tot("deaths_O")))
d = pd.read_csv(os.path.join(W, "counts_ps.tsv"), sep="\t", names=COLS, dtype={"est": str})
d = d[(d.year >= 2019) & (d.year <= 2024) & (d.ga >= 39)].copy()
d["course"] = np.select([(d.ga == 39) & (d.induct == 1) & (d.kind != "fet"), d.ga == 39], ["I", "S"], "O")
fet = d[d.kind == "fet"]; inf = d[d.kind == "inf"]
N.update(stillbirths=c(fet.n.sum()), sb_ante=c(fet[fet.est == "N"].n.sum()), neonatal=c(inf.nd.sum()),
         live=c(d[d.kind == "nat"].n.sum()), n_obese=c(d[(d.kind != "inf") & (d.bmic >= 3)].n.sum()),
         pct_obese=f1(100 * d[(d.kind != "inf") & (d.bmic >= 3)].n.sum() / d[d.kind != "inf"].n.sum()))
N["pct_I_by_bmi"] = {LAB[k]: f1(100 * v["n_I"] / (v["n_I"] + v["n_S"] + v["n_O"])) for k, v in m.items()}
for k, v in m.items(): N[f"pctI_{k}"] = f1(100 * v["n_I"] / (v["n_I"] + v["n_S"] + v["n_O"]))

# Table 1: characteristics by course
pop = d[d.kind != "inf"].copy(); pop["bmic"] = pop.bmic.map(dict(enumerate(CLS))).map(LAB)
VARS = [("Maternal age, y", "age6", {0: "<20", 1: "20–24", 2: "25–29", 3: "30–34", 4: "35–39", 5: "≥40"}),
        ("Prepregnancy BMI, kg/m²", "bmic", {LAB[k]: LAB[k] for k in CLS}),
        ("Parity", "nullip", {1: "Nulliparous", 0: "Parous", 9: "Unknown"}),
        ("Race and Hispanic origin", "race", {1: "Non-Hispanic White", 2: "Non-Hispanic Black", 4: "Non-Hispanic Asian", 7: "Hispanic",
                                              3: "Other or more than one race", 8: "Unknown", 9: "Unknown"}),
        ("Education", "educ", {1: "High school or less", 2: "Some college or associate degree", 3: "Bachelor's degree or higher", 9: "Unknown"}),
        ("Prenatal care began", "precare", {1: "First trimester", 2: "Second trimester", 3: "Third trimester", 4: "No prenatal care", 9: "Unknown"}),
        ("WIC during pregnancy", "wic", {1: "Yes", 0: "No", 9: "Unknown"}),
        ("Smoking during pregnancy", "smoke", {1: "Yes", 0: "No", 9: "Unknown"}),
        ("Maternal height, in", "height", {0: "<60", 1: "60–63", 2: "64–67", 3: "≥68", 9: "Unknown"}),
        ("Infertility treatment", "inftr", {1: "Yes"}),
        ("Year of delivery", "year", {y: str(y) for y in range(2019, 2025)})]
cols = [("I", "Induced and born in week 39"), ("S", "Other end of pregnancy in week 39"), ("O", "Ongoing at 40 0/7 weeks"), ("all", "All")]
tot_c = {k: pop[pop.course == k].n.sum() for k in "ISO"}; tot_c["all"] = pop.n.sum()
rows = ["| Characteristic | " + " | ".join(f"{lab} (n={tot_c[k]:,})" for k, lab in cols) + " |", "|" + "---|" * 5]
for title, var, mp in VARS:
    rows.append(f"| **{title}** | | | | |")
    lv = pop.assign(lv=pop[var].map(mp)).dropna(subset=["lv"])
    order = list(dict.fromkeys(mp.values()))
    t = lv.groupby(["lv", "course"]).n.sum().unstack("course").reindex(order).fillna(0); t["all"] = t.sum(1)
    for lvl in order:
        if t.loc[lvl, "all"] == 0: continue
        rows.append(f"| {lvl} | " + " | ".join(f"{int(t.loc[lvl, k]):,} ({100 * t.loc[lvl, k] / tot_c[k]:.1f})" for k, _ in cols) + " |")
N["table1"] = "\n".join(rows)

# Table 2: perinatal death by BMI class, primary analysis
P = ccw["primary"]; G = ccw["gradient_rd_minus_normal"]; E = ccw["evalue_death"]
rows = ["| BMI, kg/m² | Pregnancies | Deaths | Risk, delivery | Risk, expectant | RD (95% CrI) | RR (95% CrI) | Difference in RD (95% CrI) | NND | E-value |",
        "|" + "|".join("-" * w for w in [7, 9, 6, 7, 7, 12, 13, 12, 6, 9]) + "|"]
for k in CLS:
    v = P["death"][k]; mk = m[k]; gr = G["death"].get(k)
    rows.append(f"| {LAB[k]} | {mk['n_I'] + mk['n_S'] + mk['n_O']:,} | {mk['deaths_I'] + mk['deaths_S'] + mk['deaths_O']:,} | {f1(v['risk_A'])} | {f1(v['risk_B'])} | {ci1(v)} | {cir(v)} | "
                + (mn(f"{gr['diff']:.1f} ({gr['lo']:.1f} to {gr['hi']:.1f})") if gr else ("Reference" if k == "normal" else "")) + f" | {round(1e4 / v['rd'], -1):,.0f} | {E[k]['e_point']:.2f} ({E[k]['e_lower']:.2f}) |")
N["table2"] = "\n".join(rows)

# Table 3: secondary outcomes (risk difference per 10,000 live births, expectant minus delivery)
def long_table(outs):
    r = ["| Outcome | BMI, kg/m² | Risk, delivery | Risk, expectant | RD (95% CrI) | RR (95% CrI) |", "|" + "---|" * 6]
    for o, name in outs:
        for i, k in enumerate(CLS):
            rr = cir(P[o][k]) if P[o][k]["rr"] else "Not estimable"
            r.append(f"| {name if i == 0 else ''} | {LAB[k]} | {f1(P[o][k]['risk_A'])} | {f1(P[o][k]['risk_B'])} | {ci1(P[o][k])} | {rr} |")
    return "\n".join(r)
rows = long_table([("cesar", "Cesarean delivery"), ("nicu", "NICU admission"), ("morb", "Composite neonatal morbidity")]).splitlines()
N["table3"] = "\n".join(rows)

# Supplementary Table S1: components; S2: sensitivity analyses
rows = long_table([("stillbirth", "Stillbirth, any timing"), ("sb_ante", "Antepartum stillbirth"), ("neonatal", "Neonatal death"),
                   ("death_ante", "Antepartum stillbirth or neonatal death")]).splitlines()
N["tableS1"] = "\n".join(rows)
rows = ["| BMI, kg/m² | Clone-censor-weight, primary (RR) | Propensity weighting, antepartum stillbirth or neonatal death (RR) | Bayesian stratified model, antepartum stillbirth or neonatal death (RR) | Bayesian stratified model without week-39 stillbirths (RR) | Live births only, neonatal death, propensity weighting with payer and marital status (RR) |",
        "|" + "---|" * 6]
for k in CLS:
    vals = [P["death"][k], ip["primary"]["death"][k], tte["tte__primary"][k], tte["tte__no_sb39"][k], ip["sens_live_births"]["neonatal_death"][k]]
    rows.append(f"| {LAB[k]} | " + " | ".join(cir(v) for v in vals) + " |")
N["tableS2"] = "\n".join(rows)
bal = ip["balance_max_abs_smd"]
N["smd_raw_range"] = f"{min(v['raw'] for v in bal.values()):.2f} to {max(v['raw'] for v in bal.values()):.2f}"
N["smd_w_range"] = f"{min(v['weighted'] for v in bal.values()):.2f} to {max(v['weighted'] for v in bal.values()):.2f}"
N["smd_sens_max"] = f"{max(ip['sens_balance_max_abs_smd'].values()):.2f}"
N["truncA_range"] = f"{min(v['trunc_A'] for v in m.values()):.1f} to {max(v['trunc_A'] for v in m.values()):.1f}"
N["truncB_range"] = f"{min(v['trunc_B'] for v in m.values()):.2f} to {max(v['trunc_B'] for v in m.values()):.2f}"

# flat per-class keys for prose
for o in P:
    for k in CLS:
        v = P[o][k]
        N[f"{o}_{k}_A"] = f1(v["risk_A"]); N[f"{o}_{k}_B"] = f1(v["risk_B"]); N[f"{o}_{k}_rd"] = ci1(v); N[f"{o}_{k}_rr"] = cir(v)
        N[f"{o}_{k}_rdp"] = f1(v["rd"]); N[f"{o}_{k}_rrp"] = f2(v["rr"]); N[f"{o}_{k}_p"] = pp(v["p_B_higher"])
for o in G:
    for k, v in G[o].items(): N[f"grad_{o}_{k}"] = mn(f"{v['diff']:.1f} ({v['lo']:.1f} to {v['hi']:.1f})"); N[f"gradp_{o}_{k}"] = pp(v["p_larger"])
for k in CLS: N[f"ev_{k}_1"] = f1(E[k]["e_point"]); N[f"ev_{k}"] = f2(E[k]["e_point"]); N[f"evlo_{k}"] = f2(E[k]["e_lower"])
rrs = [P["death"][k]["rr"] for k in CLS]; N["rr_range"] = f"{min(rrs):.2f} to {max(rrs):.2f}"
evs = [E[k]["e_point"] for k in CLS]; N["ev_range"] = f"{min(evs):.1f} to {max(evs):.1f}"
evl = [E[k]["e_lower"] for k in CLS]; N["evlo_range"] = f"{min(evl):.1f} to {max(evl):.1f}"
for src, blk in (("iptw", ip["primary"]["death"]), ("tte", tte["tte__primary"]), ("nosb", tte["tte__no_sb39"]), ("live", ip["sens_live_births"]["neonatal_death"])):
    r = [blk[k]["rr"] for k in CLS]; N[f"{src}_rr_range"] = f"{min(r):.2f} to {max(r):.2f}"
    for k in CLS: N[f"{src}_{k}_rd"] = ci1(blk[k]); N[f"{src}_{k}_rr"] = cir(blk[k])
json.dump(N, open(os.path.join(OUT, "manuscript_numbers.json"), "w"), indent=1, ensure_ascii=False)
print(len(N), "keys"); print(N["table2"]); print({k: N[k] for k in list(N)[:40]})

# derived prose values and software versions
import subprocess, sys, statsmodels
N["n_cohort_million"] = f"{n / 1e6:.1f}"
for k in ["ob30", "ob35", "ob40"]: N[f"ratio_{k}"] = f"{P['death'][k]['rd'] / P['death']['normal']['rd']:.1f}"
for k in CLS: N[f"nnt_{k}"] = f"{round(1e4 / P['death'][k]['rd'], -1):,.0f}"
RUSTC = os.path.expanduser("~/.cargo/bin/rustc") if os.path.exists(os.path.expanduser("~/.cargo/bin/rustc")) else "rustc"
N["rust_version"] = subprocess.run([RUSTC, "--version"], capture_output=True, text=True).stdout.split()[1]
N["python_version"] = sys.version.split()[0]; N["statsmodels_version"] = statsmodels.__version__
json.dump(N, open(os.path.join(OUT, "manuscript_numbers.json"), "w"), indent=1, ensure_ascii=False)

# amendment m: subgroups, BMI curve, population impact
sub = json.load(open(os.path.join(OUT, "ccw_sub.json")))
SG = sub["subgroups"]
ciS = lambda v: mn(f"{v['rd']:.1f} ({v['rd_lo']:.1f} to {v['rd_hi']:.1f})")
for g, levs in SG.items():
    for l, r in levs.items():
        for o in r:
            for k, v in r[o].items(): N[f"sg_{g}_{l}_{o}_{k}"] = ciS(v); N[f"sgp_{g}_{l}_{o}_{k}"] = f"{v['rd']:.0f}" if o == "cesar" else f1(v["rd"])
GL = {"parity": ("Parity", {"nulliparous": "Nulliparous", "parous": "Parous"}), "age": ("Maternal age", {"under35": "Under 35 y", "35plus": "35 y or older"}),
      "race": ("Race and Hispanic origin", {"white": "Non-Hispanic White", "black": "Non-Hispanic Black", "hispanic": "Hispanic", "asian": "Non-Hispanic Asian"}),
      "period": ("Period", {"2019_2021": "2019–2021", "2022_2024": "2022–2024"})}
SC = ["normal", "over", "ob30", "ob35", "ob40", "ob50"]
def sg_table(o):
    r = ["| Subgroup | " + " | ".join(LAB[k] for k in SC) + " |", "|" + "---|" * (len(SC) + 1)]
    for g, (gname, levs) in GL.items():
        r.append(f"| **{gname}** |" + " |" * len(SC))
        for l, lname in levs.items():
            cells = []
            for k in SC:
                v = SG[g][l][o].get(k)
                ok = v is not None and v["rr"] is not None and np.isfinite(v["rr"]) and v["rd_hi"] > v["rd_lo"]
                cells.append(ciS(v) if ok else "Too few events")
            r.append(f"| {lname} | " + " | ".join(cells) + " |")
    return "\n".join(r)
N["tableS4"] = sg_table("death"); N["tableS5"] = sg_table("cesar")
cv = sub["curve"]; N["curve_low"] = f1(min(cv[k]["rd"] for k in ["18-19.9", "20-21.9"])); N["curve_40"] = f1(cv["40-41.9"]["rd"])
N["curve_18"] = ciS(cv["18-19.9"]); N["curve_20"] = ciS(cv["20-21.9"]); N["curve_30"] = ciS(cv["30-31.9"]); N["curve_40ci"] = ciS(cv["40-41.9"])
IM = json.load(open(os.path.join(OUT, "ccw.json")))["impact"]
for o in ["death", "cesar"]:
    for g in list(CLS) + ["obese", "all"]:
        v = IM[o][g]; N[f"imp_{o}_{g}"] = f"{v['per_year']:,.0f} ({v['lo']:,.0f} to {v['hi']:,.0f})".replace("-", "−")
        if o == "death": N[f"impn_{g}"] = f"{v['n_per_year']:,.0f}"
rows = ["| BMI, kg/m² | Pregnancies at risk per year | Observed perinatal death | Perinatal death with delivery in week 39 | Perinatal deaths avoided per year (95% CrI) | Cesarean deliveries avoided per year (95% CrI) |",
        "|" + "---|" * 6]
for k in CLS:
    v = IM["death"][k]
    rows.append(f"| {LAB[k]} | {v['n_per_year']:,.0f} | {f1(v['observed'])} | {f1(v['under_A'])} | {N[f'imp_death_{k}']} | {N[f'imp_cesar_{k}']} |")
def pooled(ks, f):  # risk per 10,000 pooled over classes, weighted by pregnancies at risk
    return sum(IM["death"][k][f] * IM["death"][k]["n_per_year"] for k in ks) / sum(IM["death"][k]["n_per_year"] for k in ks)
for g, ks, lab in (("obese", ["ob30", "ob35", "ob40", "ob50"], "30.0 or higher"), ("all", list(CLS), "All")):
    rows.append(f"| {lab} | {IM['death'][g]['n_per_year']:,.0f} | {f1(pooled(ks, 'observed'))} | {f1(pooled(ks, 'under_A'))} | {N[f'imp_death_{g}']} | {N[f'imp_cesar_{g}']} |")
N["table4"] = "\n".join(rows)
json.dump(N, open(os.path.join(OUT, "manuscript_numbers.json"), "w"), indent=1, ensure_ascii=False)

# Table 1 percentages for prose, share of avoided events
for title, var, mp in VARS:
    lv = pop.assign(lv=pop[var].map(mp)).dropna(subset=["lv"])
    t = lv.groupby(["lv", "course"]).n.sum().unstack("course").fillna(0)
    for lvl in t.index:
        for k in "ISO": N[f"t1_{var}_{str(lvl).replace(' ', '_')}_{k}"] = f1(100 * t.loc[lvl, k] / tot_c[k])
N["share_obese_deaths"] = f1(100 * IM["death"]["obese"]["per_year"] / IM["death"]["all"]["per_year"])
N["share_obese_preg"] = f1(100 * IM["death"]["obese"]["n_per_year"] / IM["death"]["all"]["n_per_year"])
N["share_obese_cesar"] = f1(100 * IM["cesar"]["obese"]["per_year"] / IM["cesar"]["all"]["per_year"])
json.dump(N, open(os.path.join(OUT, "manuscript_numbers.json"), "w"), indent=1, ensure_ascii=False)
N["t1_race_NHW_I"] = N["t1_race_Non-Hispanic_White_I"]; N["t1_race_NHW_O"] = N["t1_race_Non-Hispanic_White_O"]
json.dump(N, open(os.path.join(OUT, "manuscript_numbers.json"), "w"), indent=1, ensure_ascii=False)

# amendment n: sensitivity, bias analysis, balance, spline, matched deaths
CJ = json.load(open(os.path.join(OUT, "ccw.json")))
N["nd_unmatched"] = str(CJ["neonatal_deaths_unmatched"])
N["neonatal_matched"] = f"{tot('deaths_I') + tot('deaths_S') + tot('deaths_O') - fet.n.sum():,}"
SE = CJ["sensitivity"]; QB = CJ["qba"]
for nm, key in (("late_induction_to_I", "late"), ("no_prelabor_cesarean_wk39", "pcs"), ("exclude_sga", "sga")):
    for k in CLS:
        N[f"sens_{key}_{k}"] = ci1(SE[nm]["death"][k]); N[f"sensc_{key}_{k}"] = ci1(SE[nm]["cesar"][k])
    r = [SE[nm]["death"][k]["rr"] for k in CLS]; N[f"sens_{key}_rr_range"] = f"{min(r):.2f} to {max(r):.2f}"
    g = SE[nm]["death"]["ob40"]["rd"] - SE[nm]["death"]["normal"]["rd"]; N[f"sens_{key}_grad40"] = f1(g)
for x, key in (("0.25", "q25"), ("0.5", "q50")):
    for k in CLS: N[f"{key}_{k}"] = ci1(QB[x][k])
N["n_sga"] = f"{CJ['n_sga']:,}"; N["n_pcs"] = f"{CJ['n_prelabor_cs_wk39']:,}"; N["n_ind40"] = f"{CJ['n_induced_wk40']:,}"
BA = CJ["balance_max_abs"]
N["bal_raw_range"] = f"{min(v['raw_I_vs_O'] for v in BA.values()):.2f} to {max(v['raw_I_vs_O'] for v in BA.values()):.2f}"
N["bal_w_max"] = f"{max(max(v['A_vs_population'], v['B_vs_population']) for v in BA.values()):.2f}"
SP = sub["spline"]
N["spl_knots"] = ", ".join(f"{k:.1f}" for k in SP["knots"][:-1]) + f" and {SP['knots'][-1]:.1f}"
NU = json.load(open(os.path.join(OUT, "nd_unmatched.json")))
assert NU["unmatched"] == CJ["neonatal_deaths_unmatched"]
N["nd_unm_prev"] = str(NU["match_year_minus1"]); N["nd_unm_next"] = str(NU["match_year_plus1"])
for x in (22, 30, 35, 40, 45):
    i = SP["bmi"].index(float(x)); N[f"spl_rd_{x}"] = mn(f"{SP['rd'][i]:.1f} ({SP['lo'][i]:.1f} to {SP['hi'][i]:.1f})"); N[f"spl_nnd_{x}"] = f"{round(SP['nnd'][i], -1):,.0f}"
rows = ["| BMI, kg/m² | Primary | All induced births at 40 weeks assigned to course I (upper bound) | Prelabor cesarean births in week 39 removed | SGA removed | 25% of week-39 stillbirths preventable | 50% of week-39 stillbirths preventable |", "|" + "---|" * 7]
for k in CLS:
    rows.append(f"| {LAB[k]} | {ci1(P['death'][k])} | {N[f'sens_late_{k}']} | {N[f'sens_pcs_{k}']} | {N[f'sens_sga_{k}']} | {N[f'q25_{k}']} | {N[f'q50_{k}']} |")
N["tableS6"] = "\n".join(rows)
rows = ["| BMI, kg/m² | Course I vs course O, unweighted | Delivery strategy vs class population | Expectant strategy vs class population |", "|" + "---|" * 4]
for k in CLS: rows.append(f"| {LAB[k]} | {BA[k]['raw_I_vs_O']:.2f} | {BA[k]['A_vs_population']:.2f} | {BA[k]['B_vs_population']:.2f} |")
N["tableS7"] = "\n".join(rows)
json.dump(N, open(os.path.join(OUT, "manuscript_numbers.json"), "w"), indent=1, ensure_ascii=False)

# amendment o: safety outcomes (Supplemental Table 8)
SAF = [("nicu", "NICU admission"), ("apg7", "5-minute Apgar score below 7"), ("aven1", "Assisted ventilation immediately after delivery"),
       ("aven6", "Assisted ventilation for more than 6 hours"), ("surf", "Surfactant"), ("anti", "Antibiotics for suspected neonatal sepsis"),
       ("seiz", "Neonatal seizures"), ("morb", "Composite neonatal morbidity"), ("mtr", "Maternal transfusion"),
       ("plac", "Third or fourth degree perineal laceration"), ("rupt", "Ruptured uterus"), ("uhyst", "Unplanned hysterectomy"),
       ("aicu", "Maternal intensive care admission"), ("smm", "Severe maternal morbidity composite")]
rows = ["| Outcome | " + " | ".join(LAB[k] for k in CLS) + " |", "|" + "---|" * (len(CLS) + 1)]
for o, name in SAF:
    if o not in P: continue
    rows.append(f"| {name} | " + " | ".join(ci1(P[o][k]) for k in CLS) + " |")
    for k in CLS:
        N[f"saf_{o}_{k}"] = ci1(P[o][k]); N[f"safA_{o}_{k}"] = f1(P[o][k]["risk_A"]); N[f"safB_{o}_{k}"] = f1(P[o][k]["risk_B"])
N["tableS8"] = "\n".join(rows)
# count of safety outcome-class cells where delivery in week 39 was worse with credible interval excluding zero
worse = [(o, k) for o, _ in SAF if o in P for k in CLS if P[o][k]["rd_hi"] < 0]
better = [(o, k) for o, _ in SAF if o in P for k in CLS if P[o][k]["rd_lo"] > 0]
N["saf_n_worse"] = str(len(worse)); N["saf_n_better"] = str(len(better)); N["saf_n_total"] = str(sum(1 for o, _ in SAF if o in P) * len(CLS))
N["saf_worse_list"] = "; ".join(f"{o} {k}" for o, k in worse)
json.dump(N, open(os.path.join(OUT, "manuscript_numbers.json"), "w"), indent=1, ensure_ascii=False)
NEO = ["nicu", "apg7", "aven1", "aven6", "surf", "anti", "seiz", "morb"]
N["neo_worse_with_delivery"] = str(sum(1 for o in NEO if o in P for k in CLS if P[o][k]["rd_hi"] < 0))
N["smm_ob35_abs"] = f1(-P["smm"]["ob35"]["rd"]) if "smm" in P else ""
N["smm_range_low"] = mn(f"{P['smm']['normal']['rd']:.1f}") if "smm" in P else ""
json.dump(N, open(os.path.join(OUT, "manuscript_numbers.json"), "w"), indent=1, ensure_ascii=False)
neg = lambda v: mn(f"{-v['rd']:.1f} ({-v['rd_hi']:.1f} to {-v['rd_lo']:.1f})")
for k in CLS: N[f"smmabs_{k}"] = neg(P["smm"][k]); N[f"mtrabs_{k}"] = neg(P["mtr"][k])
for o in ["cesar", "nicu"]:
    for k in CLS: N[f"{o}_{k}_int"] = f"{P[o][k]['rd']:.0f}"
json.dump(N, open(os.path.join(OUT, "manuscript_numbers.json"), "w"), indent=1, ensure_ascii=False)
