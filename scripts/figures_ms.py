"""Manuscript figures. Fig 1 study flow and clone-censor-weight design, Fig 2 perinatal death by BMI class,
Fig 3 BMI curve and parity subgroups, Fig 4 secondary outcomes. Inputs outputs/manuscript_numbers.json and outputs/ccw.json."""
import os, json
import plotly.graph_objects as go, plotly.express as px
from plotly.subplots import make_subplots
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT, FIG = os.path.join(ROOT, "outputs"), os.path.join(ROOT, "figures", "manuscript"); os.makedirs(FIG, exist_ok=True)
N = json.load(open(os.path.join(OUT, "manuscript_numbers.json"))); P = json.load(open(os.path.join(OUT, "ccw.json")))["primary"]
CLS = ["under", "normal", "over", "ob30", "ob35", "ob40", "ob50"]
LAB = ["<18.5", "18.5–24.9", "25.0–29.9", "30.0–34.9", "35.0–39.9", "40.0–49.9", "≥50.0"]
COL = px.colors.qualitative.Plotly


def save(f, name, w, h):
    f.update_layout(template="plotly_white", width=w, height=h, font=dict(size=15, family="Arial"))
    f.write_html(os.path.join(FIG, f"{name}.html"), include_plotlyjs="cdn"); f.write_image(os.path.join(FIG, f"{name}.pdf"))
    png = os.path.join(FIG, f"{name}.png"); f.write_image(png, scale=3)
    im = Image.open(png).convert("RGB"); dpi = max(300, round(im.width / (17 / 2.54)))
    im.save(os.path.join(FIG, f"{name}.tif"), compression="tiff_lzw", dpi=(dpi, dpi))


# Figure 1: flow (left) and design (right), drawn in paper coordinates
f = go.Figure(); f.update_xaxes(visible=False, range=[0, 1]); f.update_yaxes(visible=False, range=[0, 1])
def box(x0, y0, x1, y1, text, fill="white", size=13):
    f.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1, line=dict(color="#444", width=1.2), fillcolor=fill)
    f.add_annotation(x=(x0 + x1) / 2, y=(y0 + y1) / 2, text=text, showarrow=False, font=dict(size=size), align="center")
def arrow(x0, y0, x1, y1):
    f.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2, arrowwidth=1.3, arrowcolor="#444", text="")
steps = [("Live births and fetal deaths at 20 weeks or later,<br>US residents, 2019–2024<br>n = " + N["flow_all"], None),
         (None, "Multiple gestation: " + N["flow_ex_multiple"]),
         (None, "Gestation below 39 weeks or unknown: " + N["flow_ex_ga"]),
         (None, "Prepregnancy BMI unknown: " + N["flow_ex_bmi"]),
         (None, "Diabetes, hypertension, eclampsia,<br>previous cesarean, or noncephalic<br>or unknown presentation: " + N["flow_ex_risk"]),
         (None, "Congenital anomaly: " + N["flow_ex_anom"])]
ys = [0.90, 0.76, 0.62, 0.48, 0.34, 0.20]
box(0.02, 0.85, 0.36, 0.97, steps[0][0])
for i in range(1, 6):
    y = ys[i]
    box(0.23, y - 0.05, 0.47, y + 0.05, steps[i][1], fill="#f4f4f4", size=11); arrow(0.19, y, 0.23, y)
f.add_shape(type="line", x0=0.19, y0=0.85, x1=0.19, y1=0.13, line=dict(color="#444", width=1.3)); arrow(0.19, 0.135, 0.19, 0.12)
box(0.02, 0.01, 0.36, 0.12, "Low-risk, nonanomalous singleton pregnancies<br>ongoing at 39 0/7 weeks<br>n = " + N["n_cohort"] + " (" + N["live"] + " live births,<br>" + N["stillbirths"] + " stillbirths)", fill="#eef1ff")
# design panel
f.add_annotation(x=0.76, y=0.985, text="<b>Courses observed in week 39 and their use in each emulated strategy</b>", showarrow=False, font=dict(size=13))
cy = [0.78, 0.50, 0.22]
crs = [("I: induced and born in week 39<br>n = " + N["n_I"], COL[0]), ("S: stillbirth, spontaneous or<br>noninduced birth in week 39<br>n = " + N["n_S"], COL[2]),
       ("O: ongoing at 40 0/7 weeks<br>n = " + N["n_O"], COL[1])]
for (t, cc), y in zip(crs, cy): box(0.52, y - 0.08, 0.70, y + 0.08, t, fill="white", size=12); f.add_shape(type="rect", x0=0.52, y0=y - 0.08, x1=0.53, y1=y + 0.08, fillcolor=cc, line=dict(width=0))
f.add_annotation(x=0.835, y=0.92, text="<b>Delivery in week 39</b>", showarrow=False, font=dict(size=13))
f.add_annotation(x=0.94, y=0.92, text="<b>Expectant<br>management</b>", showarrow=False, font=dict(size=13))
cells = [("Counts, weight<br>1 + P(O)/P(I)", "Censored at<br>induction"), ("Counts, weight 1<br>(includes all week-39<br>stillbirths)", "Counts, weight<br>1/(1 − P(I))<br>(includes all<br>week-39 stillbirths)"),
         ("Censored at<br>40 0/7 weeks", "Counts, weight<br>1/(1 − P(I))")]
for (a, b), y in zip(cells, cy):
    box(0.765, y - 0.08, 0.905, y + 0.08, a, fill="#eef1ff" if "Counts" in a else "#f4f4f4", size=11)
    box(0.915, y - 0.08, 0.995, y + 0.08, b, fill="#eef1ff" if "Counts" in b else "#f4f4f4", size=11)
f.add_annotation(x=0.76, y=0.06, text="P(I) and P(O): probabilities given covariates, estimated within each BMI class", showarrow=False, font=dict(size=11, color="#333"))
f.add_annotation(x=0.0, y=1.0, text="<b>A</b>", showarrow=False, font=dict(size=18), xanchor="left")
f.add_annotation(x=0.50, y=1.0, text="<b>B</b>", showarrow=False, font=dict(size=18), xanchor="left")
f.update_layout(margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
save(f, "Figure_1", 1500, 760)

# Figure 2: perinatal death
f = make_subplots(rows=1, cols=2, horizontal_spacing=0.1, subplot_titles=("Risk of perinatal death", "Risk difference, expectant management minus delivery"))
for j, (arm, nm) in enumerate([("risk_A", "Delivery in week 39"), ("risk_B", "Expectant management")]):
    f.add_trace(go.Scatter(x=LAB, y=[P["death"][k][arm] for k in CLS], mode="lines+markers", name=nm, marker=dict(size=10, color=COL[j]), line=dict(color=COL[j])), 1, 1)
rd = [P["death"][k] for k in CLS]
f.add_trace(go.Scatter(x=LAB, y=[v["rd"] for v in rd], mode="markers", name="Risk difference (95% credible interval)", marker=dict(size=10, color=COL[2]),
                       error_y=dict(type="data", array=[v["rd_hi"] - v["rd"] for v in rd], arrayminus=[v["rd"] - v["rd_lo"] for v in rd], thickness=1.5)), 1, 2)
f.add_hline(y=0, line_dash="dash", line_color="grey", row=1, col=2)
f.update_yaxes(title_text="Per 10,000 pregnancies", rangemode="tozero", row=1, col=1); f.update_yaxes(title_text="Per 10,000 pregnancies", row=1, col=2)
f.update_xaxes(title_text="Prepregnancy BMI, kg/m²")
f.update_layout(legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center"), margin=dict(l=80, r=30, t=60, b=140))
save(f, "Figure_2", 1400, 600)

# Figure 4: secondary outcomes, risk difference
f = make_subplots(rows=1, cols=3, horizontal_spacing=0.08, subplot_titles=("Cesarean delivery", "NICU admission", "Composite neonatal morbidity"))
for j, o in enumerate(["cesar", "nicu", "morb"]):
    v = [P[o][k] for k in CLS]
    f.add_trace(go.Scatter(x=LAB, y=[x["rd"] for x in v], mode="markers", marker=dict(size=9, color=COL[j]), showlegend=False,
                           error_y=dict(type="data", array=[x["rd_hi"] - x["rd"] for x in v], arrayminus=[x["rd"] - x["rd_lo"] for x in v], thickness=1.5)), 1, j + 1)
    f.add_hline(y=0, line_dash="dash", line_color="grey", row=1, col=j + 1)
f.update_yaxes(title_text="Risk difference per 10,000 live births<br>(expectant management minus delivery)", row=1, col=1)
f.update_xaxes(title_text="Prepregnancy BMI, kg/m²", tickangle=-45)
f.update_layout(margin=dict(l=100, r=30, t=60, b=120))
save(f, "Figure_4", 1500, 560)
print("ok")

# Figure 3: BMI curve (A) and parity subgroups (B perinatal death, C cesarean)
sub = json.load(open(os.path.join(OUT, "ccw_sub.json"))); cv = sub["curve"]
keys = list(cv.keys()); mid = {"<18": 17, "50+": 51}
xs = [mid[k] if k in mid else float(k.split("-")[0]) + 1 for k in keys]
f = make_subplots(rows=1, cols=3, horizontal_spacing=0.08, subplot_titles=("Perinatal death, 2-unit BMI bins and spline", "Perinatal death by parity", "Cesarean delivery by parity"))
f.add_trace(go.Scatter(x=xs, y=[cv[k]["rd"] for k in keys], mode="markers", marker=dict(size=8, color=COL[2]), name="2-unit bins",
                       error_y=dict(type="data", array=[cv[k]["rd_hi"] - cv[k]["rd"] for k in keys], arrayminus=[cv[k]["rd"] - cv[k]["rd_lo"] for k in keys], thickness=1.2)), 1, 1)
for j, (lev, nm) in enumerate([("nulliparous", "Nulliparous"), ("parous", "Parous")]):
    for col, o in ((2, "death"), (3, "cesar")):
        v = [sub["subgroups"]["parity"][lev][o][k] for k in CLS]
        f.add_trace(go.Scatter(x=[i + (j - 0.5) * 0.18 for i in range(len(CLS))], y=[x["rd"] for x in v], mode="markers+lines", name=nm, legendgroup=nm, showlegend=(col == 2),
                               marker=dict(size=8, color=COL[j]), line=dict(color=COL[j]),
                               error_y=dict(type="data", array=[x["rd_hi"] - x["rd"] for x in v], arrayminus=[x["rd"] - x["rd_lo"] for x in v], thickness=1.2)), 1, col)
sp = sub["spline"]
f.add_trace(go.Scatter(x=sp["bmi"] + sp["bmi"][::-1], y=sp["hi"] + sp["lo"][::-1], fill="toself", fillcolor="rgba(0,204,150,0.15)", line=dict(width=0), showlegend=False, hoverinfo="skip"), 1, 1)
f.add_trace(go.Scatter(x=sp["bmi"], y=sp["rd"], mode="lines", line=dict(color="black", width=2), name="Spline (95% credible band)"), 1, 1)
for col in (1, 2, 3): f.add_hline(y=0, line_dash="dash", line_color="grey", row=1, col=col)
f.update_xaxes(title_text="Prepregnancy BMI, kg/m²", row=1, col=1)
for col in (2, 3): f.update_xaxes(tickvals=list(range(len(CLS))), ticktext=LAB, tickangle=-45, title_text="Prepregnancy BMI, kg/m²", row=1, col=col)
f.update_yaxes(title_text="Risk difference per 10,000 pregnancies", row=1, col=1); f.update_yaxes(title_text="Per 10,000 pregnancies", row=1, col=2)
f.update_yaxes(title_text="Per 10,000 live births", row=1, col=3)
f.update_layout(legend=dict(orientation="h", y=-0.38, x=0.67, xanchor="center"), margin=dict(l=80, r=30, t=60, b=150))
save(f, "Figure_3", 1600, 600)
print("fig4 ok")
