import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from core import (FEATURES, LIMITATIONS, TARGET, EXAMPLE, load_data,
                  relationship_label, report_text, train)

st.set_page_config(page_title="Business Predictor | Sales Intelligence",
                   page_icon="📊", layout="wide", initial_sidebar_state="expanded")

# -------------------- THEME --------------------
INK, MUTED, GREEN, DARK, LIME, TEAL, SLATE, GRID, PANEL = (
    "#102522", "#435650", "#087E73", "#172B2A", "#B8D94A", "#55B8A9", "#687F78", "#E1E7E2", "#FFFFFF")
LABELS = {"Ad_Spend": "Advertising spend", "Price": "Product price",
          "Economy": "Economic indicator", "Competition": "Competition index"}
STEPS = {"Ad_Spend": 1.0, "Price": 0.1, "Economy": 1.0, "Competition": 1.0}
name = lambda f: f.replace("_", " ")

CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Playfair+Display:wght@500;600;700&display=swap');
:root{--ink:#102522;--mute:#435650;--g:#087E73;--dk:#172B2A;--line:#C9D8CF}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;color:var(--ink)}
.stApp{background:radial-gradient(ellipse at 90% 0%,rgba(8,126,115,.09),transparent 32%),linear-gradient(180deg,#F6F5F0,#F3F2EC)}
.block-container{max-width:1500px;padding:1.5rem 1rem 3rem}
h1,h2,h3,h4,h5,h6{color:var(--ink)!important;font-weight:800!important;letter-spacing:-.025em}
h1,.serif{font-family:'Playfair Display',Georgia,serif!important}
p,li{color:#263B35}
[data-testid=stSidebar]{background:linear-gradient(180deg,#172B2A,#203735);border-right:1px solid rgba(184,217,74,.35)}
[data-testid=stSidebar] *{color:#F3F2EC}
[data-testid=stSidebar] label{font-weight:600!important}
.brand{display:flex;align-items:center;gap:12px;margin-bottom:1.2rem}
.brand-mark{width:48px;height:48px;border-radius:14px;display:flex;align-items:center;justify-content:center;
  background:var(--dk);color:#B8D94A;border:1px solid #B8D94A;font-size:23px;font-weight:800}
.brand-name{font-family:'Playfair Display',serif;font-size:1.35rem;font-weight:800}
.caps{text-transform:uppercase;letter-spacing:.16em;font-size:.68rem;color:var(--mute)}
.hero{background:linear-gradient(120deg,#fff,#EAF4EF);padding:2.2rem;border-radius:24px;margin:.4rem 0 1.25rem;
  box-shadow:0 15px 40px rgba(16,37,34,.10);border:1px solid var(--line)}
.eyebrow{text-transform:uppercase;letter-spacing:.19em;font-size:.75rem;font-weight:800;color:var(--g);margin-bottom:.65rem}
.hero h1{font-size:clamp(2rem,4vw,3.3rem);line-height:1.1;margin:.2rem 0 .8rem}
.hero p{color:var(--mute);max-width:780px;line-height:1.7}
.tag{display:inline-block;margin-top:.8rem;padding:8px 12px;border:1px solid #B7D9CF;border-radius:999px;
  color:var(--g);font-size:.75rem;font-weight:800;background:#F3FAF6}
.section-head{display:flex;align-items:center;gap:11px;margin:2rem 0 .35rem}
.section-num{min-width:31px;height:31px;border-radius:9px;display:inline-flex;align-items:center;
  justify-content:center;background:#DDECE6;color:#075E55;font-weight:800;font-size:.8rem}
.sub{color:#465852;font-size:.9rem;margin-bottom:1rem}
.note{color:var(--mute);font-size:.8rem;line-height:1.6}
.card{background:#fff;border:1px solid var(--line);border-radius:17px;padding:19px;min-height:125px;
  box-shadow:0 5px 18px rgba(16,37,34,.05)}
.card .lbl{color:var(--mute);font-size:.76rem;text-transform:uppercase;letter-spacing:.1em;font-weight:800;margin-bottom:11px}
.card .val{font-family:'Playfair Display',serif;font-size:2rem;font-weight:800;line-height:1.1}
.card .note{margin-top:9px}
.insight{background:#fff;border:1px solid #D3DFD8;border-left:5px solid var(--g);border-radius:13px;padding:15px;margin:.45rem 0}
.insight b.t{display:block;margin-bottom:5px;font-weight:800}
.insight .body{color:var(--mute);font-size:.88rem;line-height:1.6}
.pred{background:linear-gradient(135deg,#F8FAF6,#E8F3EE);border:1px solid #BFDCD1;border-radius:20px;padding:25px}
.pred .lbl{color:#075E55;text-transform:uppercase;letter-spacing:.13em;font-size:.75rem;font-weight:800}
.pred .val{font-family:'Playfair Display',serif;color:#075E55;font-size:clamp(2.5rem,5vw,4rem);font-weight:800;margin:.5rem 0}
div[data-testid=stPlotlyChart]{background:#fff;border:1px solid var(--line);border-radius:16px;padding:8px;
  box-shadow:0 5px 18px rgba(16,37,34,.05)}
div[data-testid=stDataFrame]{border:1px solid #C2D0C8;border-radius:12px;overflow:hidden}
.stButton>button,.stDownloadButton>button{border-radius:11px;border:1px solid var(--g);background:var(--g);
  color:#fff;font-weight:800;padding:.65rem 1rem}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:var(--dk);background:var(--dk);color:#fff}
footer,#MainMenu{visibility:hidden}
"""
html = lambda s: st.markdown(s, unsafe_allow_html=True)
html(f"<style>{CSS}</style>")


def section(num, title, sub):
    html(f'<div class="section-head"><span class="section-num">{num}</span>'
         f'<h2 style="margin:0">{title}</h2></div><div class="sub">{sub}</div>')


def insight(title, body, accent=GREEN, right=""):
    html(f'<div class="insight" style="border-left-color:{accent}"><div style="display:flex;'
         f'justify-content:space-between"><b class="t">{title}</b>'
         f'<b style="color:{accent}">{right}</b></div><div class="body">{body}</div></div>')


def style_plot(fig, height=380):
    axis = dict(showgrid=True, gridcolor=GRID, zeroline=False, linecolor="#AEBDB5",
                title_font=dict(color=INK, size=13), tickfont=dict(color="#263B35", size=11))
    fig.update_layout(
        template="plotly_white", height=height, paper_bgcolor=PANEL, plot_bgcolor=PANEL,
        font=dict(family="DM Sans, sans-serif", color=INK, size=12),
        title_font=dict(family="Playfair Display, Georgia, serif", size=20, color=INK),
        margin=dict(l=35, r=30, t=65, b=45),
        hoverlabel=dict(bgcolor=DARK, font_color="#FFFFFF", bordercolor=LIME),
        legend=dict(bgcolor="rgba(255,255,255,0.85)", borderwidth=0))
    fig.update_xaxes(**axis).update_yaxes(**axis)
    return st.plotly_chart(fig, use_container_width=True)


def scatter(x, y=TARGET, ols=False, **kw):
    fig = px.scatter(data, x=x, y=y, trendline="ols" if ols else None, **kw)
    return fig


def df_view(df, **kw):
    st.dataframe(df, hide_index=True, use_container_width=True, **kw)


# -------------------- DATA + MODEL --------------------
@st.cache_data
def get_data():
    return load_data()


try:
    data = get_data()
except Exception as exc:
    st.error(f"Unable to load data.csv: {exc}")
    st.info("Place data.csv next to app.py. Required columns: " + ", ".join(FEATURES + [TARGET]))
    st.stop()

r = train(data)
corr = data.corr(numeric_only=True)
mae, r2 = r["mae"], r["r2"]

# -------------------- SIDEBAR --------------------
with st.sidebar:
    html('<div style="padding:12px 0 18px"><div style="font-size:1.45rem;font-weight:800">◈ Business Predictor</div>'
         '<div class="caps" style="color:#D8EE8B">Sales Intelligence</div></div>')
    st.markdown("---\n### Prediction studio")
    st.write("Adjust the business factors below, then generate a sales estimate.")
    inputs = {}
    for f in FEATURES:
        lo, hi, mid = (float(data[f].min()), float(data[f].max()), float(data[f].median()))
        inputs[f] = st.slider(LABELS[f], lo, hi, mid, step=STEPS[f],
                              help=f"Observed data range: {lo:g} to {hi:g}")
    st.session_state.setdefault("last_prediction", None)
    if st.button("✦ Generate prediction", use_container_width=True):
        try:
            scenario = pd.DataFrame([inputs], columns=FEATURES)
            st.session_state.last_prediction = dict(
                values=dict(inputs), prediction=float(r["model"].predict(scenario)[0]))
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
    if st.session_state.last_prediction:
        st.metric("Predicted sales", f"{st.session_state.last_prediction['prediction']:,.2f} units")
    st.markdown("---\n### Model notes")
    for line in ["Linear Regression", "Standardised input features", "80/20 train-test split", "Random seed: 42"]:
        st.write(line)
    st.markdown("---")
    st.caption(f"Educational demo · {len(data)} simulated records. Not real business data.")

# -------------------- HERO --------------------
html("""
<div class="brand"><div class="brand-mark">BP</div><div>
  <div class="brand-name">Business Predictor</div><div class="caps">Sales Intelligence Dashboard</div></div></div>
<div class="hero"><div class="eyebrow">Decision intelligence · Edition 01</div>
  <h1>Find the signal<br>behind your sales.</h1>
  <p>Explore how advertising spend, pricing, economic conditions and competition relate to sales.
  Investigate the data, evaluate a machine-learning model and test your own scenario.</p>
  <span class="tag">LINEAR REGRESSION · INTERACTIVE ANALYTICS</span></div>
<div class="note">Educational research demo. All records are simulated.
Model outputs are estimates, not guarantees or causal claims.</div>""")

# -------------------- 01 AT A GLANCE --------------------
section("01", "At a glance", "A quick summary of the dataset and model performance.")
kpis = [("DATA RECORDS", f"{len(data):,}", "Simulated observations"),
        ("TEST R²", f"{r2:.3f}", "Held-out test performance"),
        ("TEST MAE", f"{mae:.2f}", "Average error in sales units"),
        ("MODEL", "Linear", "Standardised regression")]
for col, (lbl, val, note) in zip(st.columns(4), kpis):
    col.markdown(f'<div class="card"><div class="lbl">{lbl}</div><div class="val">{val}</div>'
                 f'<div class="note">{note}</div></div>', unsafe_allow_html=True)

# -------------------- 02 PREDICTION STUDIO --------------------
section("02", "Prediction studio", "Choose values in the sidebar and generate a sales estimate.")
saved = st.session_state.last_prediction
left, right = st.columns([1.2, 1])
with left:
    html(f'<div class="pred"><div class="lbl">Scenario estimate</div>'
         f'<div class="val">{"—" if not saved else format(saved["prediction"], ",.2f")}</div>'
         f'<div class="note">{"Adjust the inputs in the sidebar, then select Generate prediction." if not saved else "Estimated sales for the most recently submitted scenario. This is a model estimate, not a guarantee."}</div></div>')
with right:
    st.markdown("#### Current scenario")
    df_view(pd.DataFrame({"Business factor": [name(f) for f in inputs], "Selected value": list(inputs.values())}), height=205)
    if saved:
        lines = ["BUSINESS PREDICTOR | SALES PREDICTION REPORT", "=" * 45, "",
                 f"Predicted sales: {saved['prediction']:.2f} units", "", "Scenario inputs:",
                 *[f"- {k}: {v}" for k, v in saved["values"].items()], "",
                 "Model: Standardised Linear Regression", f"Dataset size: {len(data)} simulated records",
                 f"Test MAE: {mae:.3f}", f"Test R²: {r2:.3f}", "",
                 "Limitations: Simulated data; exploratory estimates only."]
        st.download_button("↓ Download scenario report", "\n".join(lines),
                           "Business_Prediction_Report.txt", "text/plain", use_container_width=True)

# -------------------- 03 CORRELATION --------------------
section("03", "What moves with sales?", "Correlation measures linear association, not causation.")
heat, notes = st.columns([1.15, 1])
with heat:
    fig = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, aspect="auto", title="Correlation matrix",
                    labels={"color": "Pearson r"},
                    color_continuous_scale=[[0, "#A83E50"], [.25, "#E9B8A9"], [.5, "#F3F2EC"],
                                            [.75, "#9AD5C0"], [1, GREEN]])
    style_plot(fig, 450)
with notes:
    st.markdown("#### Relationship highlights")
    for f in FEATURES:
        v = float(corr.loc[f, TARGET])
        insight(name(f), f"{relationship_label(v)} with sales in this dataset.",
                GREEN if v > .3 else "#A83E50" if v < -.3 else SLATE, f"{v:+.2f}")
    html('<div class="note">Relationships can reflect how the simulated data was created '
         'and may be influenced by relationships among input features.</div>')

# -------------------- 04 EXPLORE --------------------
section("04", "Explore the data", "Choose a factor to explore its relationship with sales.")
a, b = st.columns([1, 2.1])
with a:
    chosen = st.selectbox("Choose a business factor", FEATURES, format_func=name)
    insight("How to read this chart", "Each point represents a simulated observation. "
            "The trend line summarises a linear pattern, not causation.")
with b:
    fig = scatter(chosen, ols=True, color="Competition", hover_data=FEATURES,
                  color_continuous_scale=["#D8EE8B", TEAL, DARK], title=f"{name(chosen)} vs Sales",
                  labels={chosen: name(chosen), TARGET: "Sales units", "Competition": "Competition index"})
    fig.update_traces(marker=dict(size=10, opacity=.82, line=dict(width=1, color=PANEL)))
    style_plot(fig, 430)

# -------------------- 05 VISUALISATIONS --------------------
section("05", "Business data visualisations",
        "All four factor-versus-sales graphs, plus sales distribution and model performance.")
st.markdown("### Individual business factor graphs")
for col, f in zip(st.columns(2) * 2, FEATURES):
    with col:
        fig = scatter(f, ols=True, hover_data=FEATURES, title=f"{name(f)} vs Sales",
                      labels={f: name(f), TARGET: "Sales units"})
        fig.update_traces(marker=dict(size=9, color=GREEN, opacity=.78, line=dict(width=1, color=PANEL)),
                          selector=dict(mode="markers"))
        style_plot(fig)

st.markdown("### Sales distribution")
fig = px.histogram(data, x=TARGET, nbins=15, title="Distribution of Sales", color_discrete_sequence=[GREEN],
                   labels={TARGET: "Sales units", "count": "Number of records"})
fig.update_traces(marker_line_color="#FFFFFF", marker_line_width=1)
style_plot(fig)

st.markdown("### Actual vs predicted sales")
perf = pd.DataFrame({"Actual Sales": r["y_test"].to_numpy(), "Predicted Sales": r["pred"]})
fig = px.scatter(perf, x="Actual Sales", y="Predicted Sales", color_discrete_sequence=[GREEN],
                 title="Actual vs Predicted Sales (Test Set)",
                 labels={"Actual Sales": "Actual sales units", "Predicted Sales": "Predicted sales units"})
lo, hi = perf.min().min(), perf.max().max()
fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(color=GREEN, dash="dash", width=2))
style_plot(fig, 420)

st.markdown("### Prediction errors")
perf["Residual"] = perf["Actual Sales"] - perf["Predicted Sales"]
fig = px.scatter(perf, x="Predicted Sales", y="Residual", color_discrete_sequence=[SLATE],
                 title="Prediction Residuals",
                 labels={"Predicted Sales": "Predicted sales units", "Residual": "Actual minus predicted sales"})
fig.add_hline(y=0, line_dash="dash", line_color=GREEN)
style_plot(fig)
insight("Test-set performance",
        f"Mean Absolute Error: <b>{mae:.2f}</b> sales units. R²: <b>{r2:.3f}</b>. These metrics describe "
        "the held-out test set and do not guarantee future prediction accuracy.")

# -------------------- 06 MODEL COEFFICIENTS --------------------
section("06", "Inside the model",
        "Compare the direction and relative magnitude of the standardised regression coefficients.")
coef = pd.DataFrame({"Business factor": [name(f) for f in FEATURES], "Coefficient": r["std_coef"]})
coef["Direction"] = np.where(coef["Coefficient"] >= 0, "Positive", "Negative")
fig = px.bar(coef.sort_values("Coefficient"), x="Coefficient", y="Business factor", orientation="h",
             color="Direction", color_discrete_map={"Positive": GREEN, "Negative": "#A83E50"},
             text=coef.sort_values("Coefficient")["Coefficient"].map("{:+.2f}".format),
             title="Standardised Regression Coefficients")
fig.update_traces(textposition="outside", cliponaxis=False)
fig.add_vline(x=0, line_width=1, line_color=LIME)
style_plot(fig, 390)

coef_table = pd.DataFrame({"Business factor": FEATURES, "Standardised coefficient": r["std_coef"],
                           "Original-unit coefficient": r["raw_coef"]})
with st.expander("View coefficients in original units"):
    df_view(coef_table.drop(columns="Standardised coefficient").style.format({"Original-unit coefficient": "{:.3f}"}))
    st.caption("From a separate LinearRegression trained on the same records without StandardScaler.")
with st.expander("How to interpret the model and its limitations"):
    st.markdown("""
- **Linear Regression:** estimates a linear relationship between the four inputs and sales.
- **Standardisation:** puts input features on comparable scales.
- **R²:** share of sales variation explained on this particular held-out test split.
- **MAE:** average absolute prediction error in sales units.
- **Small simulated dataset:** results may change with other data or different splits.
- **No causal proof:** coefficients and correlations do not show that changing a factor causes sales to change.
- **Range limitation:** predictions can remain uncertain even when inputs are inside the observed range.""")

# -------------------- 07 FULL REPORT --------------------
section("07", "Business analysis report",
        "Dataset, correlation matrix, model evaluation, insights, coefficients, example prediction and limitations.")
example = pd.DataFrame([EXAMPLE], columns=FEATURES)
with st.expander("01 · Business dataset", expanded=True):
    df_view(data)
    c = st.columns(3)
    c[0].metric("Dataset rows", len(data))
    c[1].metric("Dataset columns", data.shape[1])
    c[2].metric("Source", "Simulated")
with st.expander("02 · Correlation matrix", expanded=True):
    st.dataframe(corr.style.format("{:.6f}"), use_container_width=True)
    st.caption("Positive = positive linear association; negative = negative. Not cause and effect.")
with st.expander("03 · Model evaluation", expanded=True):
    c = st.columns(4)
    c[0].metric("Training records", len(r["X_train"]))
    c[1].metric("Testing records", len(r["X_test"]))
    c[2].metric("MAE", f"{mae:.3f}")
    c[3].metric("R²", f"{r2:.3f}")
    st.write("**MAE:** average absolute difference between actual and predicted sales (lower is better). "
             "**R²:** how well the model explains variation in sales on this test set.")
with st.expander("04 · Correlation-based business insights", expanded=True):
    for f in FEATURES:
        v = float(corr.loc[f, TARGET])
        st.markdown(f"**{name(f)}** → {relationship_label(v)} (corr = {v:.2f})")
with st.expander("05 · Model coefficients", expanded=True):
    df_view(coef_table.style.format({c: "{:.3f}" for c in coef_table.columns[1:]}))
    st.caption("Fitted associations while other inputs are held constant, not proof of cause and effect. "
               "Standardised and original-unit coefficients use different scales.")
with st.expander("06 · Example sales prediction", expanded=True):
    df_view(example)
    st.metric("Predicted sales", f"{float(r['model'].predict(example)[0]):,.2f} units")
    st.caption("Uses the example scenario from program.py. Simulated data; not a dependable real-world forecast.")
with st.expander("07 · Limitations", expanded=True):
    st.markdown("\n".join(f"- {x}" for x in LIMITATIONS))

st.download_button("↓ Download complete business analysis report", report_text(data, corr, r),
                   "Business_Analysis_Report.txt", "text/plain", use_container_width=True)

# -------------------- 08 SOURCE DATA --------------------
section("08", "Inspect the source dataset", "Review the underlying records or download the CSV.")
with st.expander("Open complete source dataset"):
    df_view(data)
    st.download_button("↓ Download dataset CSV", data.to_csv(index=False).encode("utf-8"),
                       "Business_Predictor_Simulated_Data.csv", "text/csv")

# -------------------- FOOTER --------------------
st.markdown("---")
html(f'<div style="display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;color:{MUTED};'
     f'font-size:.78rem;padding:.25rem 0 1rem"><span><b style="color:{GREEN}">BUSINESS PREDICTOR</b>'
     '&nbsp;·&nbsp; Sales Intelligence Dashboard</span>'
     '<span>Educational prototype · Simulated data · Linear Regression</span></div>')
