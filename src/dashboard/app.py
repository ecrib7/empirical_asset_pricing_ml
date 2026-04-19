"""
dashboard/app.py
----------------
Streamlit dashboard for the GKX (2019) replication.

Sections:
  1. OOS R² comparison table (Table 1 replica)
  2. Diebold-Mariano test matrix (Table 3 replica)
  3. Cumulative portfolio returns (Figure 9 replica)
  4. Decile portfolio performance (Table 7 replica)
  5. Variable importance (Figure 4/5 replica)
  6. Sharpe ratio analysis
  7. Transaction cost sensitivity

Run:  streamlit run src/dashboard/app.py
"""

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GKX (2019) ML Asset Pricing",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parents[2]
OUT  = ROOT / "outputs"


# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data
def load_metrics() -> dict:
    p = OUT / "metrics.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


@st.cache_data
def load_r2_table() -> pd.Series:
    p = OUT / "oos_r2.csv"
    if p.exists():
        return pd.read_csv(p, index_col=0).squeeze("columns")
    return pd.Series(dtype=float)


@st.cache_data
def load_dm_table() -> pd.DataFrame:
    p = OUT / "dm_table.csv"
    if p.exists():
        return pd.read_csv(p, index_col=0)
    return pd.DataFrame()


@st.cache_data
def load_portfolio_bundle() -> dict:
    """Load net / gross / turnover portfolio dicts (see ``reporting.portfolio_io``)."""
    from src.reporting.portfolio_io import unpack_portfolio_bundle

    p = OUT / "portfolio_returns.pkl"
    if not p.exists():
        return {"net": {}, "gross": None, "turnover": None, "meta": {}}
    with open(p, "rb") as f:
        raw = pickle.load(f)
    net, gross, turnover, meta = unpack_portfolio_bundle(raw)
    return {"net": net, "gross": gross, "turnover": turnover, "meta": meta}


def _model_metrics(metrics: dict) -> dict:
    return {k: v for k, v in metrics.items() if not str(k).startswith("_") and isinstance(v, dict)}


# ── Colour palette matching GKX figures ───────────────────────────────────────
MODEL_COLORS = {
    "OLS-3":  "#000000",
    "ENet+H": "#2166ac",
    "PCR":    "#4dac26",
    "PLS":    "#d01c8b",
    "GLM+H":  "#f1a340",
    "RF":     "#0571b0",
    "GBRT+H": "#ca0020",
    "NN1":    "#5e3c99",
    "NN2":    "#b2abd2",
    "NN3":    "#e66101",   # highlight – best model
    "NN4":    "#fdb863",
    "NN5":    "#a6611a",
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("GKX (2019) Replication")
st.sidebar.markdown("**IEOR 4733 — Algorithmic Trading**")
st.sidebar.markdown("---")
section = st.sidebar.radio(
    "Navigate",
    ["Overview", "OOS R²", "DM Tests", "Portfolio Returns",
     "Sharpe Ratios", "Transaction Costs", "Run Pipeline"],
)

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("📊 Empirical Asset Pricing via Machine Learning")
st.caption("Replication of Gu, Kelly & Xiu (2019) — NBER WP 25398")

# =============================================================================
#  OVERVIEW
# =============================================================================
if section == "Overview":
    col1, col2, col3, col4 = st.columns(4)
    metrics = load_metrics()
    mm = _model_metrics(metrics)
    best_r2 = max((v["oos_r2_pct"] for v in mm.values()), default=np.nan)
    best_sr = max((v["hl_sharpe"] for v in mm.values() if not np.isnan(v.get("hl_sharpe", np.nan))), default=np.nan)
    best_model = max(mm, key=lambda k: mm[k].get("oos_r2_pct", -np.inf), default="—")

    col1.metric("Best OOS R² (%)", f"{best_r2:.3f}" if not np.isnan(best_r2) else "—")
    col2.metric("Best H-L Sharpe", f"{best_sr:.2f}" if not np.isnan(best_sr) else "—")
    col3.metric("Best Model", best_model)
    col4.metric("Models Evaluated", len(mm))
    rep = metrics.get("_reporting", {})
    if rep.get("hl_returns_are_net_of_engine_tc"):
        st.caption(
            "H-L Sharpe above is **net** of engine TC (see metrics). "
            "Gross H-L Sharpe is in the Sharpe tab when available."
        )

    st.markdown("---")
    st.subheader("Paper Summary")
    st.markdown("""
    **Gu, Kelly & Xiu (2019)** perform a comprehensive comparison of machine learning methods
    for **measuring equity risk premia** across ~30,000 US stocks from 1957–2016.

    | Component | Details |
    |-----------|---------|
    | Universe | NYSE, AMEX, NASDAQ stocks |
    | Sample | March 1957 – December 2016 (60 years) |
    | Features | 94 firm characteristics × 9 macro interactions + 74 industry dummies = **920 signals** |
    | Training | 1957–1974 (recursive, expands 1 yr/yr) |
    | Validation | 1975–1986 (rolling 12-month window) |
    | Test | **1987–2016** (30-year OOS) |

    **Key findings:**
    - Neural networks (NN3) achieve OOS R² of **0.40%/month** vs. 0.16% for OLS-3
    - Shallow learning > deep learning in asset pricing (data scarcity + low SNR)
    - NN3 S&P 500 timing Sharpe ratio: **0.77** vs. 0.51 buy-and-hold
    - Long-short NN3 Sharpe ratio: **1.35** (value-weighted)
    - Top predictors: **momentum > liquidity > volatility**
    """)

    with st.expander("📐 Model Taxonomy"):
        st.markdown("""
        | Model | Type | Key feature |
        |-------|------|------------|
        | OLS-3 | Linear | Size, B/M, Momentum only |
        | ENet+H | Penalized linear | L1+L2 regularisation + Huber loss |
        | PCR | Dim. reduction | PCA then OLS |
        | PLS | Dim. reduction | Target-aware dimension reduction |
        | GLM+H | Semi-parametric | Splines + Group Lasso |
        | RF | Tree ensemble | Random forest (Breiman 2001) |
        | GBRT+H | Tree ensemble | Gradient boosted trees + Huber |
        | NN1–NN5 | Neural network | 1–5 hidden layers, ReLU, BatchNorm |
        """)


# =============================================================================
#  OOS R² TABLE
# =============================================================================
elif section == "OOS R²":
    st.subheader("Out-of-Sample R² (%) — GKX Table 1 Replica")
    r2 = load_r2_table()

    if r2.empty:
        st.warning("No results yet. Run the pipeline first (see 'Run Pipeline' tab).")
    else:
        # Colour-coded bar chart
        try:
            import plotly.express as px
            df_plot = r2.reset_index()
            df_plot.columns = ["Model", "OOS R² (%)"]
            colors = [MODEL_COLORS.get(m, "#888") for m in df_plot["Model"]]
            fig = px.bar(
                df_plot, x="Model", y="OOS R² (%)",
                title="Monthly Stock-Level OOS R²",
                color="Model",
                color_discrete_sequence=colors,
            )
            fig.add_hline(y=0, line_dash="dash", line_color="black")
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.bar_chart(r2)

        st.dataframe(r2.to_frame("OOS R² (%)").style.format("{:.3f}").background_gradient(
            cmap="RdYlGn", axis=0
        ))

        st.info("""
        **Interpretation:** OOS R² is benchmarked against a zero forecast (not historical mean).
        Positive values indicate the model predicts better than a naive zero.
        NN3 achieves ~0.40% in the original paper.
        """)


# =============================================================================
#  DIEBOLD-MARIANO TESTS
# =============================================================================
elif section == "DM Tests":
    st.subheader("Diebold-Mariano Test Statistics — GKX Table 3 Replica")
    dm = load_dm_table()

    if dm.empty:
        st.warning("No DM results yet.")
    else:
        try:
            import plotly.figure_factory as ff
            import plotly.graph_objects as go

            fig = go.Figure(data=go.Heatmap(
                z=dm.values,
                x=dm.columns.tolist(),
                y=dm.index.tolist(),
                colorscale="RdBu",
                zmid=0,
                text=np.round(dm.values, 2),
                texttemplate="%{text}",
                colorbar=dict(title="DM Stat"),
            ))
            fig.update_layout(
                title="DM Test: Positive = column model outperforms row model",
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.dataframe(dm.style.format("{:.2f}").background_gradient(cmap="RdBu", axis=None))

        st.info("Bold values in original paper exceed |1.96| (5% significance). "
                "Asterisks indicate significance after Bonferroni correction (threshold ~2.64).")


# =============================================================================
#  PORTFOLIO RETURNS
# =============================================================================
elif section == "Portfolio Returns":
    st.subheader("Long-Short Decile Portfolio Returns — GKX Figure 9 Replica")
    bundle = load_portfolio_bundle()
    port_rets = bundle["net"]
    gross_bundle = bundle.get("gross")

    if not port_rets:
        st.warning("No portfolio results yet.")
    else:
        rep = load_metrics().get("_reporting", {})
        if rep.get("hl_returns_are_net_of_engine_tc"):
            st.caption(
                "Plotted H-L paths are **net** of transaction costs applied in the backtest engine. "
                "Gross H-L series are available in the pickle bundle when present."
            )
        models_avail = list(port_rets.keys())
        show_gross = bool(
            gross_bundle
            and st.checkbox("Overlay gross H-L (pre-engine TC)", value=False)
        )
        selected = st.multiselect(
            "Select models to plot",
            models_avail,
            default=[m for m in ["NN3", "RF", "OLS-3"] if m in models_avail],
        )

        if selected:
            try:
                import plotly.graph_objects as px2
                fig = px2.Figure()
                for model in selected:
                    hl = port_rets[model].get("H-L", pd.Series(dtype=float))
                    if len(hl) == 0:
                        continue
                    hl = hl.sort_index().dropna()
                    cum = (1 + hl).cumprod()
                    fig.add_trace(px2.Scatter(
                        x=cum.index, y=cum.values,
                        name=f"{model} (net)",
                        line=dict(color=MODEL_COLORS.get(model, "#888"), width=2),
                    ))
                    if show_gross and gross_bundle.get(model):
                        ghl = gross_bundle[model].get("H-L", pd.Series(dtype=float))
                        ghl = ghl.reindex(hl.index).dropna()
                        if len(ghl) > 0:
                            gc = (1 + ghl).cumprod()
                            fig.add_trace(px2.Scatter(
                                x=gc.index, y=gc.values,
                                name=f"{model} (gross)",
                                line=dict(
                                    color=MODEL_COLORS.get(model, "#888"),
                                    width=1,
                                    dash="dash",
                                ),
                            ))
                fig.update_layout(
                    title="Cumulative return: long-short decile spread",
                    yaxis_title="Cumulative return",
                    xaxis_title="Date",
                    height=500,
                    legend=dict(x=0.02, y=0.98),
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                # Fallback to line chart
                hl_df = pd.DataFrame({
                    m: (1 + port_rets[m].get("H-L", pd.Series(dtype=float))).cumprod()
                    for m in selected
                })
                st.line_chart(hl_df)

        # Decile performance table
        st.subheader("Decile Performance (GKX Table 7)")
        if rep.get("hl_returns_are_net_of_engine_tc"):
            st.caption("Table uses **net** decile returns (engine transaction costs applied).")
        model_sel = st.selectbox("Model", models_avail)
        if model_sel:
            from src.evaluation.metrics import sharpe_ratio
            rows = []
            decile_rets = port_rets[model_sel]
            for d in list(range(1, 11)) + ["H-L"]:
                key = str(d)
                if key not in decile_rets:
                    continue
                r = decile_rets[key].dropna()
                rows.append({
                    "Decile":     "Low" if d == 1 else "High" if d == 10 else "H-L" if key == "H-L" else str(d),
                    "Avg Ret (% /mo)": f"{r.mean()*100:.2f}",
                    "Std (% /mo)":     f"{r.std()*100:.2f}",
                    "Ann. Sharpe":     f"{sharpe_ratio(r):.2f}",
                })
            st.dataframe(pd.DataFrame(rows))


# =============================================================================
#  SHARPE RATIOS
# =============================================================================
elif section == "Sharpe Ratios":
    st.subheader("H-L Sharpe Ratios & Campbell-Thompson SR Improvement")
    metrics = load_metrics()
    mm = _model_metrics(metrics)

    if not mm:
        st.warning("No results yet.")
    else:
        df = pd.DataFrame([
            {"Model": k,
             "H-L Sharpe (net)": v.get("hl_sharpe", np.nan),
             "H-L Sharpe (gross)": v.get("hl_sharpe_gross", np.nan),
             "Mean TO (1-way)": v.get("hl_mean_turnover_one_way", np.nan),
             "OOS R² (%)": v.get("oos_r2_pct", np.nan)}
            for k, v in mm.items()
        ])

        if not df.empty:
            try:
                import plotly.express as px
                fig = px.scatter(
                    df, x="OOS R² (%)", y="H-L Sharpe (net)",
                    text="Model", title="OOS R² vs H-L Sharpe (net of engine TC)",
                    color="Model",
                )
                fig.update_traces(textposition="top center")
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.dataframe(df)

            st.dataframe(df.set_index("Model").style.format("{:.3f}").background_gradient(
                cmap="RdYlGn"
            ))


# =============================================================================
#  TRANSACTION COSTS
# =============================================================================
elif section == "Transaction Costs":
    st.subheader("Transaction Cost Sensitivity (incremental on top of engine)")
    bundle = load_portfolio_bundle()
    port_net = bundle["net"]
    turnover_b = bundle.get("turnover")
    metrics = load_metrics()
    rep = metrics.get("_reporting", {})

    if not port_net:
        st.warning("Run the pipeline first.")
    else:
        from src.reporting.portfolio_io import hl_additional_tc_sharpe

        if rep.get("hl_returns_are_net_of_engine_tc"):
            st.info(
                f"Primary H-L series are **already net** of engine TC "
                f"({rep.get('hl_engine_tc_bps_default', '?')} bps one-way). "
                "The chart below applies **additional** hypothetical one-way costs "
                "using stored month-over-month turnover (no double-count of the engine fee)."
            )
        else:
            st.caption(
                "Engine TC was zero; H-L series are gross. "
                "Additional bps below are hypothetical incremental costs."
            )

        extra_tc_range = np.arange(0, 51, 5)
        models_to_plot = [m for m in ["NN3", "RF", "GBRT+H", "ENet+H", "OLS-3"]
                          if m in port_net]

        rows = []
        for extra_bps in extra_tc_range:
            row = {"Additional TC (bps, one-way)": extra_bps}
            for model in models_to_plot:
                hl = port_net[model].get("H-L", pd.Series(dtype=float)).dropna()
                if len(hl) == 0:
                    row[model] = np.nan
                    continue
                to = None
                if turnover_b and model in turnover_b:
                    to = turnover_b[model].get("H-L")
                row[model] = hl_additional_tc_sharpe(hl, to, float(extra_bps))
            rows.append(row)

        df_tc = pd.DataFrame(rows).set_index("Additional TC (bps, one-way)")

        try:
            import plotly.express as px
            fig = px.line(
                df_tc.reset_index().melt(
                    id_vars="Additional TC (bps, one-way)",
                    var_name="Model",
                    value_name="H-L Sharpe",
                ),
                x="Additional TC (bps, one-way)", y="H-L Sharpe", color="Model",
                title="H-L Sharpe vs additional one-way TC (engine TC not re-applied)",
                color_discrete_map=MODEL_COLORS,
            )
            fig.add_hline(y=0, line_dash="dash", line_color="black")
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.line_chart(df_tc)

        st.dataframe(df_tc.style.format("{:.3f}").background_gradient(cmap="RdYlGn", axis=None))
        st.warning(
            "If turnover was not stored (legacy pickle), additional TC Sharpe is blank "
            "for extra bps > 0 — re-run the pipeline to populate turnover."
        )


# =============================================================================
#  RUN PIPELINE
# =============================================================================
elif section == "Run Pipeline":
    st.subheader("Run the Backtest Pipeline")

    mode = st.radio(
        "Mode",
        ["Test (synthetic data)", "Cache (use saved features)", "Full (requires WRDS)"],
    )
    tc_bps = st.slider("Transaction cost (bps, one-way)", 0, 50, 10)
    models_to_run = st.multiselect(
        "Models to run",
        ["OLS-3", "ENet+H", "PCR", "PLS", "GLM+H", "RF", "GBRT+H",
         "NN1", "NN2", "NN3", "NN4", "NN5"],
        default=["OLS-3", "ENet+H", "RF", "NN3"],
    )

    wrds_user = ""
    if "Full" in mode:
        wrds_user = st.text_input("WRDS Username")

    if st.button("▶ Run Pipeline", type="primary"):
        import subprocess, sys
        args = ["python", "main.py",
                "--mode", "test" if "Test" in mode else "cache" if "Cache" in mode else "full",
                "--tc-bps", str(tc_bps)]
        if models_to_run:
            args += ["--models"] + models_to_run
        if wrds_user:
            args += ["--wrds-username", wrds_user]

        with st.spinner("Running pipeline… (may take several minutes for full run)"):
            proc = subprocess.run(
                args,
                capture_output=True, text=True,
                cwd=str(ROOT),
            )
        if proc.returncode == 0:
            st.success("Pipeline completed! Refresh the other tabs to see results.")
            st.code(proc.stdout[-3000:], language="text")
        else:
            st.error("Pipeline failed.")
            st.code(proc.stderr[-3000:], language="text")

    st.markdown("---")
    st.markdown("### How to set up WRDS access")
    st.code("""
# Install wrds package
pip install wrds

# Store credentials (one-time setup)
python -c "import wrds; db = wrds.Connection()"
# Follow prompts to save ~/.pgpass

# Or set environment variable
export WRDS_USERNAME=your_username
    """, language="bash")

    st.markdown("### Download Goyal & Welch macro data")
    st.markdown(
        "Download `PredictorData2023.xlsx` from "
        "[Amit Goyal's website](https://sites.google.com/view/agoyal145) "
        "and pass it via `--goyal-csv path/to/file.xlsx`."
    )
