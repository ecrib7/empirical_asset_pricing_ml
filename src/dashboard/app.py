"""
dashboard/app.py
----------------
Streamlit dashboard for the GKX (2019) replication.

Two pipeline variants are supported and exposed in the sidebar:

  * ``paper``    — strict GKX (2019) reproduction (1957–2016, TC=0).
  * ``improved`` — extended sample (1957 → ~2024) + transaction costs.

Sections:
  1. Overview                — KPIs + sample / settings summary
  2. Comprehensive Metrics   — Sharpe net/gross, SR*, MaxDD, Skew, Kurt, OOS R², Alpha, t(α)
  3. OOS R²                  — Table 1 replica
  4. DM Tests                — Diebold-Mariano stat & p-value heatmaps
  5. Portfolio Returns       — Cumulative H-L paths + decile performance
  6. Sharpe Ratios           — Net vs gross + scatter
  7. Transaction Costs       — Sensitivity to additional bps
  8. Variable Importance     — Per-model importance + heatmap (NEW)
  9. Paper vs Improved       — Side-by-side comparison (NEW)
 10. Run Pipeline            — Trigger backtests from the UI

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


def variant_dir(variant: str) -> Path:
    """Resolve the output directory for a variant. Falls back to legacy
    outputs/ if the variant-specific dir is missing (handles older runs)."""
    p = ROOT / "outputs" / variant
    if p.exists():
        return p
    legacy = ROOT / "outputs"
    return legacy if legacy.exists() else p


# ── Data loaders (variant-scoped) ────────────────────────────────────────────

@st.cache_data
def load_metrics(variant: str) -> dict:
    p = variant_dir(variant) / "metrics.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


@st.cache_data
def load_r2_table(variant: str) -> pd.Series:
    p = variant_dir(variant) / "oos_r2.csv"
    if p.exists():
        return pd.read_csv(p, index_col=0).squeeze("columns")
    return pd.Series(dtype=float)


@st.cache_data
def load_dm_table(variant: str) -> pd.DataFrame:
    p = variant_dir(variant) / "dm_table.csv"
    if p.exists():
        return pd.read_csv(p, index_col=0)
    return pd.DataFrame()


@st.cache_data
def load_dm_pvalues(variant: str) -> pd.DataFrame:
    p = variant_dir(variant) / "dm_pvalues.csv"
    if p.exists():
        return pd.read_csv(p, index_col=0)
    return pd.DataFrame()


@st.cache_data
def load_comprehensive(variant: str) -> pd.DataFrame:
    p = variant_dir(variant) / "comprehensive.csv"
    if p.exists():
        return pd.read_csv(p, index_col=0)
    return pd.DataFrame()


@st.cache_data
def load_var_importance(variant: str) -> pd.DataFrame:
    p = variant_dir(variant) / "var_importance.csv"
    if p.exists():
        return pd.read_csv(p, index_col=0)
    return pd.DataFrame()


@st.cache_data
def load_regimes(variant: str) -> pd.DataFrame:
    p = variant_dir(variant) / "regimes.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


@st.cache_data
def load_portfolio_bundle(variant: str) -> dict:
    """Load net / gross / turnover portfolio dicts (see ``reporting.portfolio_io``)."""
    sys.path.insert(0, str(ROOT))
    from src.reporting.portfolio_io import unpack_portfolio_bundle

    p = variant_dir(variant) / "portfolio_returns.pkl"
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
    "NN3":    "#e66101",
    "NN4":    "#fdb863",
    "NN5":    "#a6611a",
    "ENS-AVG": "#117733",   # ensembles get distinct colours
    "ENS-MSE": "#882255",
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("GKX (2019) Replication")
st.sidebar.markdown("**IEOR 4733 — Algorithmic Trading**")
st.sidebar.markdown("---")

# Variant selector — drives every loader on the page.
# Discover any outputs/<sub>/ directory that has a metrics.json; ensure
# the canonical real variants ('paper', 'improved') and the synthetic
# scoring / future variants are listed even if not yet generated, so the
# Run Pipeline tab still surfaces them.
_KNOWN_VARIANTS = (
    "paper", "improved", "extended_2024", "extended_ciz_2026",
    "post2016_ciz",
    "future2026_base", "future2026_trending", "future2026_mean_reversion",
    "future2026_rotating_leaders", "future2026_choppy",
    "future2026_crisis", "future2026_factor_rotation",
)

discovered = []
outputs_root = ROOT / "outputs"
if outputs_root.exists():
    for sub in sorted(outputs_root.iterdir()):
        if sub.is_dir() and (sub / "metrics.json").exists():
            discovered.append(sub.name)

# Stable ordering: known variants first (in canonical order), then any
# extras that appeared on disk.
available_variants = [v for v in _KNOWN_VARIANTS if v in discovered]
available_variants += [v for v in discovered if v not in _KNOWN_VARIANTS]

if not available_variants and (ROOT / "outputs" / "metrics.json").exists():
    available_variants = ["paper"]

if not available_variants:
    st.sidebar.warning(
        "No results found yet — run the pipeline first "
        "(see the 'Run Pipeline' tab)."
    )
    available_variants = ["paper", "improved"]

variant = st.sidebar.selectbox(
    "Pipeline variant",
    available_variants,
    help=("'paper' = strict GKX 1957–2016 reproduction (TC=0). "
          "'improved' = extended sample to 2024 + transaction costs modelled. "
          "'post2016_ciz' = synthetic CIZ-window scoring (2017–2026). "
          "'future2026_*' = forward synthetic post-WRDS scenarios "
          "(2026-04..2036-03)."),
)
st.sidebar.markdown(f"📁 `outputs/{variant}/`")
st.sidebar.markdown("---")

section = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Comprehensive Metrics",
        "OOS R²",
        "DM Tests",
        "Portfolio Returns",
        "Sharpe Ratios",
        "Transaction Costs",
        "Forecast Combination",
        "Regimes",
        "Variable Importance",
        "Paper vs Improved",
        "Run Pipeline",
    ],
)

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("📊 Empirical Asset Pricing via Machine Learning")
st.caption("Replication of Gu, Kelly & Xiu (2019) — NBER WP 25398")

# Universal banner showing active variant settings
metrics_active = load_metrics(variant)
rep_active = metrics_active.get("_reporting", {})
if rep_active:
    cols_banner = st.columns(5)
    cols_banner[0].markdown(f"**Variant:** `{rep_active.get('variant', variant)}`")
    cols_banner[1].markdown(
        f"**Sample:** {rep_active.get('data_start','?')} → {rep_active.get('data_end','?')}"
    )
    cols_banner[2].markdown(
        f"**Test:** {rep_active.get('test_start','?')} → {rep_active.get('test_end','?')}"
    )
    cols_banner[3].markdown(f"**TC:** {rep_active.get('tc_bps', 'n/a')} bps")
    cols_banner[4].markdown(
        f"**Macro × char:** {'✅' if rep_active.get('use_macro_interactions') else '—'}"
    )
    st.markdown("---")


# =============================================================================
#  OVERVIEW
# =============================================================================
if section == "Overview":
    col1, col2, col3, col4 = st.columns(4)
    metrics = metrics_active
    mm = _model_metrics(metrics)
    best_r2 = max((v["oos_r2_pct"] for v in mm.values()), default=np.nan) if mm else np.nan
    best_sr = (
        max((v["hl_sharpe"] for v in mm.values()
             if not np.isnan(v.get("hl_sharpe", np.nan))), default=np.nan)
        if mm else np.nan
    )
    best_model = (
        max(mm, key=lambda k: mm[k].get("oos_r2_pct", -np.inf), default="—")
        if mm else "—"
    )

    col1.metric("Best OOS R² (%)", f"{best_r2:.3f}" if not np.isnan(best_r2) else "—")
    col2.metric("Best H-L Sharpe", f"{best_sr:.2f}" if not np.isnan(best_sr) else "—")
    col3.metric("Best Model", best_model)
    col4.metric("Models Evaluated", len(mm))

    rep = metrics.get("_reporting", {})
    if rep.get("hl_returns_are_net_of_engine_tc"):
        st.caption(
            f"H-L Sharpe above is **net** of {rep.get('tc_bps','?')}bps engine TC. "
            "Gross H-L Sharpe is in the Sharpe tab."
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

    **Key findings (paper):**
    - Neural networks (NN3) achieve OOS R² of **0.40%/month** vs. 0.16% for OLS-3
    - Shallow learning > deep learning in asset pricing (data scarcity + low SNR)
    - Long-short NN3 Sharpe ratio: **1.35** (value-weighted, gross)
    - Top predictors: **momentum > liquidity > volatility**
    """)

    with st.expander("📐 Model Taxonomy"):
        st.markdown("""
        | Model | Type | Key feature |
        |-------|------|-------------|
        | OLS-3 | Linear | Size, B/M, Momentum only |
        | ENet+H | Penalized linear | L1+L2 + Huber loss |
        | PCR | Dim. reduction | PCA then OLS |
        | PLS | Dim. reduction | Target-aware dimension reduction |
        | GLM+H | Semi-parametric | Splines + Group Lasso |
        | RF | Tree ensemble | Random forest |
        | GBRT+H | Tree ensemble | Gradient boosted trees + Huber |
        | NN1–NN5 | Neural network | 1–5 hidden layers, ReLU, BatchNorm |
        """)


# =============================================================================
#  COMPREHENSIVE METRICS  (NEW)
# =============================================================================
elif section == "Comprehensive Metrics":
    st.subheader("Comprehensive Model Performance")
    st.caption(
        "Sharpe (net & gross), SR\\* (Campbell-Thompson), Max Drawdown, "
        "Skew, Kurtosis, OOS R², Mean Turnover, Alpha (% / yr) and t(α)."
    )
    df = load_comprehensive(variant)
    if df.empty:
        st.warning(
            "No comprehensive metrics yet. Run "
            f"`python main.py --mode evaluate --variant {variant}`."
        )
    else:
        st.dataframe(
            df.style
              .format({
                  "Sharpe (net)":   "{:.3f}",
                  "Sharpe (gross)": "{:.3f}",
                  "SR*":            "{:.3f}",
                  "Max DD (%)":     "{:.2f}",
                  "Skew":           "{:.3f}",
                  "Kurt":           "{:.3f}",
                  "OOS R² (%)":     "{:.3f}",
                  "Mean TO (1-way)":"{:.3f}",
                  "Alpha (% / yr)": "{:.2f}",
                  "t(alpha)":       "{:.2f}",
              })
              .background_gradient(
                  subset=["Sharpe (net)", "OOS R² (%)", "Alpha (% / yr)"],
                  cmap="RdYlGn"
              )
              .background_gradient(subset=["Max DD (%)"], cmap="RdYlGn_r"),
            use_container_width=True,
        )

        st.markdown("##### Quick reads")
        if not df.empty:
            best_net   = df["Sharpe (net)"].idxmax() if df["Sharpe (net)"].notna().any() else None
            best_gross = df["Sharpe (gross)"].idxmax() if df["Sharpe (gross)"].notna().any() else None
            best_r2    = df["OOS R² (%)"].idxmax() if df["OOS R² (%)"].notna().any() else None
            cleanest   = df["Skew"].abs().idxmin() if df["Skew"].notna().any() else None
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Best Sharpe (net)",   best_net or "—")
            c2.metric("Best Sharpe (gross)", best_gross or "—")
            c3.metric("Best OOS R²",         best_r2 or "—")
            c4.metric("Cleanest distrib.",   cleanest or "—")

        st.info(
            "**SR\\*** is Campbell-Thompson (2008): "
            "SR\\* = √(SR² + R²/(1−R²)) where R² is in decimal form. "
            "Alpha is regressed against the **equal-weighted cross-sectional "
            "mean of realised returns** (a proxy market factor); replace with "
            "FF3 / FF5 by feeding `market_factor` to `comprehensive_table`."
        )


# =============================================================================
#  OOS R² TABLE
# =============================================================================
elif section == "OOS R²":
    st.subheader("Out-of-Sample R² (%) — GKX Table 1 Replica")
    r2 = load_r2_table(variant)

    if r2.empty:
        st.warning("No results yet. Run the pipeline first (see 'Run Pipeline' tab).")
    else:
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

        st.dataframe(
            r2.to_frame("OOS R² (%)").style.format("{:.3f}").background_gradient(
                cmap="RdYlGn", axis=0
            )
        )

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
    dm = load_dm_table(variant)
    pv = load_dm_pvalues(variant)

    if dm.empty:
        st.warning(
            "No DM results yet. Run "
            f"`python main.py --mode evaluate --variant {variant}`."
        )
    else:
        try:
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

        st.markdown("##### Two-sided p-values")
        if not pv.empty:
            try:
                import plotly.graph_objects as go
                pv_disp = pv.copy().astype(float)
                fig2 = go.Figure(data=go.Heatmap(
                    z=pv_disp.values,
                    x=pv_disp.columns.tolist(),
                    y=pv_disp.index.tolist(),
                    colorscale="Greens_r",
                    zmin=0, zmax=1,
                    text=np.where(
                        np.isnan(pv_disp.values), "",
                        np.round(pv_disp.values, 3).astype(str)
                    ),
                    texttemplate="%{text}",
                    colorbar=dict(title="p-value"),
                ))
                fig2.update_layout(title="DM p-values (two-sided)", height=500)
                st.plotly_chart(fig2, use_container_width=True)
            except ImportError:
                st.dataframe(pv.style.format("{:.3f}"))
        st.info(
            "Bold values in original paper exceed |1.96| (5% significance). "
            "After Bonferroni correction at α=5% for n²−n pairwise tests, "
            "the threshold is ≈ 2.64 for n=12 models, |DM|>2.64 ≈ p<0.0083."
        )


# =============================================================================
#  PORTFOLIO RETURNS
# =============================================================================
elif section == "Portfolio Returns":
    st.subheader("Long-Short Decile Portfolio Returns — GKX Figure 9 Replica")
    bundle = load_portfolio_bundle(variant)
    port_rets = bundle["net"]
    gross_bundle = bundle.get("gross")

    if not port_rets:
        st.warning(
            "No portfolio results yet. Run "
            f"`python main.py --mode evaluate --variant {variant}`."
        )
    else:
        rep = load_metrics(variant).get("_reporting", {})
        if rep.get("hl_returns_are_net_of_engine_tc"):
            st.caption(
                f"Plotted H-L paths are **net** of {rep.get('tc_bps','?')}bps "
                "engine TC. Toggle below to overlay gross."
            )
        models_avail = list(port_rets.keys())
        show_gross = bool(
            gross_bundle
            and st.checkbox("Overlay gross H-L (pre-engine TC)", value=False)
        )
        selected = st.multiselect(
            "Select models to plot",
            models_avail,
            default=[m for m in ["NN3", "RF", "OLS-3", "PCR", "PLS"] if m in models_avail],
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
                    if show_gross and gross_bundle and gross_bundle.get(model):
                        ghl = gross_bundle[model].get("H-L", pd.Series(dtype=float))
                        ghl = ghl.reindex(hl.index).dropna()
                        if len(ghl) > 0:
                            gc = (1 + ghl).cumprod()
                            fig.add_trace(px2.Scatter(
                                x=gc.index, y=gc.values,
                                name=f"{model} (gross)",
                                line=dict(
                                    color=MODEL_COLORS.get(model, "#888"),
                                    width=1, dash="dash",
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
            except Exception:
                hl_df = pd.DataFrame({
                    m: (1 + port_rets[m].get("H-L", pd.Series(dtype=float))).cumprod()
                    for m in selected
                })
                st.line_chart(hl_df)

        st.subheader("Decile Performance (GKX Table 7)")
        model_sel = st.selectbox("Model", models_avail)
        if model_sel:
            sys.path.insert(0, str(ROOT))
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
    metrics = load_metrics(variant)
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

            st.dataframe(
                df.set_index("Model").style.format("{:.3f}").background_gradient(
                    cmap="RdYlGn"
                )
            )


# =============================================================================
#  TRANSACTION COSTS
# =============================================================================
elif section == "Transaction Costs":
    st.subheader("Transaction Cost Sensitivity (incremental on top of engine)")
    bundle = load_portfolio_bundle(variant)
    port_net = bundle["net"]
    turnover_b = bundle.get("turnover")
    metrics = load_metrics(variant)
    rep = metrics.get("_reporting", {})

    if not port_net:
        st.warning("Run the pipeline first.")
    else:
        sys.path.insert(0, str(ROOT))
        from src.reporting.portfolio_io import hl_additional_tc_sharpe

        if rep.get("hl_returns_are_net_of_engine_tc"):
            st.info(
                f"Primary H-L series are **already net** of engine TC "
                f"({rep.get('tc_bps','?')} bps one-way). "
                "The chart below applies **additional** hypothetical one-way costs."
            )
        else:
            st.caption("Engine TC was zero; H-L series are gross. "
                       "Bps below are hypothetical incremental costs.")

        extra_tc_range = np.arange(0, 51, 5)
        models_to_plot = [m for m in ["NN3", "RF", "GBRT+H", "ENet+H", "OLS-3", "PCR", "PLS"]
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
                title="H-L Sharpe vs additional one-way TC",
                color_discrete_map=MODEL_COLORS,
            )
            fig.add_hline(y=0, line_dash="dash", line_color="black")
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.line_chart(df_tc)

        st.dataframe(df_tc.style.format("{:.3f}").background_gradient(cmap="RdYlGn", axis=None))


# =============================================================================
#  FORECAST COMBINATION  (NEW)
# =============================================================================
elif section == "Forecast Combination":
    st.subheader("Forecast Combination — ENS-AVG and ENS-MSE")
    st.caption(
        "Two ensembles built from the per-model OOS predictions: "
        "**ENS-AVG** (equal-weighted average) and **ENS-MSE** (weighted by "
        "1/MSE on the earliest 10% of the test sample). These are added "
        "to every other tab as if they were separate models."
    )

    metrics = load_metrics(variant)
    mm = _model_metrics(metrics)
    comp = load_comprehensive(variant)
    bundle = load_portfolio_bundle(variant)
    ens_meta = metrics.get("_ensembles", {})

    if not ens_meta:
        st.warning(
            "No ensembles found in metrics.json. Re-run "
            f"`python main.py --mode evaluate --variant {variant}` "
            "(omit --no-ensembles)."
        )
    else:
        # Ensemble vs constituents — Sharpe (net) bar chart
        ens_names = list(ens_meta.keys())
        constituents = ens_meta.get("ENS-AVG", {}).get("constituents", [])

        st.markdown("### Ensembles vs constituents — Sharpe (net)")
        rows = []
        for m in constituents + ens_names:
            if m in mm:
                rows.append({
                    "Model": m,
                    "Sharpe (net)": mm[m].get("hl_sharpe", float("nan")),
                    "Sharpe (gross)": mm[m].get("hl_sharpe_gross", float("nan")),
                    "OOS R² (%)": mm[m].get("oos_r2_pct", float("nan")),
                    "Is ensemble": m in ens_names,
                })
        df_ens = pd.DataFrame(rows)

        if not df_ens.empty:
            try:
                import plotly.express as px
                fig = px.bar(
                    df_ens, x="Model", y="Sharpe (net)",
                    color="Is ensemble",
                    color_discrete_map={True: "#117733", False: "#888"},
                    text="Sharpe (net)",
                    title="H-L Sharpe (net): constituents (grey) vs ensembles (green)",
                )
                fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
                fig.update_layout(height=420, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.bar_chart(df_ens.set_index("Model")["Sharpe (net)"])

            st.dataframe(
                df_ens.set_index("Model").style.format(
                    {"Sharpe (net)": "{:.3f}", "Sharpe (gross)": "{:.3f}",
                     "OOS R² (%)": "{:.3f}", "Is ensemble": "{}"}
                ).background_gradient(
                    subset=["Sharpe (net)", "OOS R² (%)"], cmap="RdYlGn"
                ),
                use_container_width=True,
            )

        # ENS-MSE weights
        st.markdown("### ENS-MSE inverse-validation-MSE weights")
        mse_meta = ens_meta.get("ENS-MSE", {})
        weights = mse_meta.get("weights", {})
        if weights:
            wdf = (pd.Series(weights, name="weight")
                     .sort_values(ascending=False)
                     .to_frame())
            try:
                import plotly.express as px
                fig = px.bar(
                    wdf.reset_index().rename(columns={"index": "Model"}),
                    x="Model", y="weight",
                    color="Model", color_discrete_map=MODEL_COLORS,
                    title=f"ENS-MSE weights (val_frac = {mse_meta.get('val_frac','?')})",
                )
                fig.update_layout(height=380, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.bar_chart(wdf)
            st.dataframe(wdf.style.format("{:.3f}"))
        else:
            st.info("ENS-MSE weights not stored in this run.")

        # Cumulative paths
        st.markdown("### Cumulative H-L return — ensembles vs best single models")
        port_net = bundle["net"]
        if port_net:
            best_single = (
                df_ens[~df_ens["Is ensemble"]]
                .sort_values("Sharpe (net)", ascending=False)
                .head(2)["Model"].tolist()
            )
            to_plot = [m for m in (ens_names + best_single) if m in port_net]
            try:
                import plotly.graph_objects as go
                fig = go.Figure()
                for m in to_plot:
                    hl = port_net[m].get("H-L", pd.Series(dtype=float)).dropna()
                    if len(hl) == 0:
                        continue
                    cum = (1 + hl).cumprod()
                    is_ens = m in ens_names
                    fig.add_trace(go.Scatter(
                        x=cum.index, y=cum.values, name=m,
                        line=dict(
                            color=MODEL_COLORS.get(m, "#888"),
                            width=3 if is_ens else 1.5,
                            dash="solid" if is_ens else "dot",
                        ),
                    ))
                fig.update_layout(
                    title="Ensembles (solid, thick) vs best single models (dotted)",
                    yaxis_title="Cumulative return", xaxis_title="Date", height=440,
                )
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                pass


# =============================================================================
#  REGIMES  (NEW)
# =============================================================================
elif section == "Regimes":
    st.subheader("Regime-conditional evaluation")
    st.caption(
        "Performance sliced by NBER recession status, VIX terciles, and decade. "
        "Run `python main.py --mode regimes --variant <v>` to populate."
    )

    df = load_regimes(variant)
    if df.empty:
        st.warning(
            "No regimes data yet. Run "
            f"`python main.py --mode regimes --variant {variant}` "
            f"(after `--mode evaluate`)."
        )
    else:
        kind = st.selectbox(
            "Regime kind",
            ["nber", "vix", "decade", "full"],
            index=0,
        )
        sub = df[df["regime_kind"] == kind].copy()
        if sub.empty:
            st.info(f"No data for regime kind '{kind}'.")
        else:
            metric = st.selectbox(
                "Metric",
                ["sharpe_net", "sharpe_gross", "mean_ret_pct_pm",
                 "max_dd_pct", "skew", "kurt", "n_months"],
                index=0,
            )
            wide = sub.pivot_table(
                index="model", columns="regime", values=metric,
            )
            # Order columns sensibly
            order_pref = {
                "nber":  ["recession", "expansion"],
                "vix":   ["low_vix", "mid_vix", "high_vix"],
                "decade": sorted(sub["regime"].unique()),
                "full":  ["all"],
            }
            cols = [c for c in order_pref.get(kind, list(wide.columns))
                    if c in wide.columns]
            wide = wide[cols]

            try:
                import plotly.graph_objects as go
                fig = go.Figure()
                for c in wide.columns:
                    fig.add_trace(go.Bar(
                        name=c, x=wide.index.tolist(), y=wide[c].tolist(),
                    ))
                fig.update_layout(
                    barmode="group",
                    title=f"{metric} by {kind} regime",
                    height=480,
                )
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.bar_chart(wide)

            st.dataframe(
                wide.style.format("{:.3f}")
                    .background_gradient(cmap="RdYlGn", axis=None),
                use_container_width=True,
            )

            with st.expander("Show full regimes table"):
                st.dataframe(sub, use_container_width=True)

            if kind == "nber":
                st.info(
                    "**Read:** strategies whose `recession` Sharpe is far "
                    "below their `expansion` Sharpe rely on a benign macro "
                    "regime; persistent strategies post similar numbers in "
                    "both columns. The gap (expansion − recession) is the "
                    "implicit cyclical exposure of the alpha."
                )
            elif kind == "vix":
                st.info(
                    "**Read:** alpha that survives across VIX terciles is "
                    "more likely to be a true characteristic premium and "
                    "less likely to be compensation for tail risk."
                )


# =============================================================================
#  VARIABLE IMPORTANCE  (NEW)
# =============================================================================
elif section == "Variable Importance":
    st.subheader("What drives the predictions? — Variable Importance")
    st.caption(
        "Importance = drop in OOS R² when each base characteristic is "
        "set to its cross-sectional median (= 0 in normalised form). "
        "Aggregated from the 920 Kronecker features back to the 94 base chars."
    )

    imp = load_var_importance(variant)
    if imp.empty:
        st.warning(
            "No variable importance results yet. Run "
            f"`python main.py --mode importance --variant {variant} "
            "--models <model-list>`."
        )
    else:
        avail_models = list(imp.columns)
        sel_model = st.selectbox("Model", avail_models, key="vi_model")
        top_n = st.slider("Top-N characteristics", 5, 50, 20, key="vi_topn")

        s = imp[sel_model].dropna().sort_values(ascending=False).head(top_n)
        try:
            import plotly.express as px
            fig = px.bar(
                s.reset_index().rename(
                    columns={"index": "characteristic", sel_model: "importance"}
                ),
                x="importance", y="characteristic", orientation="h",
                title=f"Top {top_n} characteristics — {sel_model}",
                color="importance",
                color_continuous_scale="Viridis",
            )
            fig.update_layout(yaxis=dict(autorange="reversed"), height=600)
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.bar_chart(s)

        st.markdown("##### Cross-model heatmap (top characteristics)")
        top_feats = (
            imp.abs()
              .max(axis=1)
              .sort_values(ascending=False)
              .head(top_n)
              .index
        )
        heat = imp.loc[top_feats]
        heat_norm = heat.div(heat.abs().max(axis=0).replace(0, np.nan), axis=1)
        try:
            import plotly.graph_objects as go
            fig2 = go.Figure(data=go.Heatmap(
                z=heat_norm.values,
                x=heat_norm.columns.tolist(),
                y=heat_norm.index.tolist(),
                colorscale="Viridis",
                colorbar=dict(title="rel. importance"),
            ))
            fig2.update_layout(
                title="Variable importance — characteristics × models (max-normalised)",
                height=20 * len(top_feats) + 100,
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig2, use_container_width=True)
        except ImportError:
            st.dataframe(heat.style.format("{:.4f}").background_gradient(cmap="Viridis", axis=0))

        with st.expander("Categories to watch (per GKX 2019)"):
            st.markdown("""
            - **Price trends:** `mom1m`, `mom12m`, `chmom`, `indmom`, `maxret`
            - **Liquidity:** `mvel1`, `dolvol`, `turn`, `ill`, `baspread`
            - **Risk measures:** `retvol`, `idiovol`, `beta`, `betasq`
            - **Valuation:** `ep`, `sp`, `agr`, `nincr`
            """)


# =============================================================================
#  PAPER vs IMPROVED  (NEW)
# =============================================================================
elif section == "Paper vs Improved":
    st.subheader("Side-by-side: paper reproduction vs improved pipeline")
    st.caption(
        "Compares the strict GKX 1957–2016 reproduction with the extended "
        "1957–2024 + transaction-cost variant. Run **both** variants for the "
        "comparison to populate."
    )

    paper_metrics    = load_metrics("paper")
    improved_metrics = load_metrics("improved")
    paper_comp       = load_comprehensive("paper")
    improved_comp    = load_comprehensive("improved")

    if not paper_metrics and not improved_metrics:
        st.warning("Neither variant has results yet.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### `paper` settings")
            rep = paper_metrics.get("_reporting", {})
            if rep:
                st.json({k: v for k, v in rep.items() if not k.startswith("hl_")})
            else:
                st.info("paper results missing — run with `--variant paper`.")
        with col2:
            st.markdown("### `improved` settings")
            rep = improved_metrics.get("_reporting", {})
            if rep:
                st.json({k: v for k, v in rep.items() if not k.startswith("hl_")})
            else:
                st.info("improved results missing — run with `--variant improved`.")

        st.markdown("---")
        st.markdown("### Comprehensive metrics by variant")
        metric_cols = ["Sharpe (net)", "Sharpe (gross)", "SR*", "Max DD (%)",
                       "Skew", "Kurt", "OOS R² (%)", "Alpha (% / yr)"]
        if not paper_comp.empty and not improved_comp.empty:
            common_models = paper_comp.index.intersection(improved_comp.index)
            if len(common_models):
                metric_pick = st.selectbox("Metric to compare", metric_cols)
                df_cmp = pd.DataFrame({
                    "paper":    paper_comp.loc[common_models, metric_pick],
                    "improved": improved_comp.loc[common_models, metric_pick],
                })
                df_cmp["delta (improved − paper)"] = df_cmp["improved"] - df_cmp["paper"]
                try:
                    import plotly.graph_objects as go
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        name="paper",    x=df_cmp.index, y=df_cmp["paper"],
                        marker_color="#666"))
                    fig.add_trace(go.Bar(
                        name="improved", x=df_cmp.index, y=df_cmp["improved"],
                        marker_color="#1f77b4"))
                    fig.update_layout(
                        barmode="group", height=420,
                        title=f"{metric_pick}: paper vs improved",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except ImportError:
                    st.bar_chart(df_cmp[["paper", "improved"]])
                st.dataframe(df_cmp.style.format("{:.3f}"))
            else:
                st.warning("No models in common between the two variants yet.")
        else:
            st.info(
                "Run **both** variants to enable the comparison. "
                "Specifically: `python main.py --mode evaluate --variant paper` "
                "and `--variant improved`."
            )


# =============================================================================
#  RUN PIPELINE
# =============================================================================
elif section == "Run Pipeline":
    st.subheader("Run the Backtest Pipeline")

    chosen_variant = st.radio(
        "Variant", ["paper", "improved"],
        index=["paper", "improved"].index(variant) if variant in ["paper", "improved"] else 0,
    )
    mode = st.radio(
        "Mode",
        ["Test (synthetic data)", "Cache (use saved features)", "Full (requires WRDS)"],
    )
    tc_bps = st.slider(
        "Transaction cost override (bps, one-way)",
        0, 50, 0 if chosen_variant == "paper" else 10,
    )
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
        import subprocess
        args = ["python", "main.py",
                "--variant", chosen_variant,
                "--mode", "test" if "Test" in mode else "cache" if "Cache" in mode else "full",
                "--tc-bps", str(tc_bps)]
        if models_to_run:
            args += ["--models"] + models_to_run
        if wrds_user:
            args += ["--wrds-username", wrds_user]

        with st.spinner("Running pipeline… (may take several minutes for full run)"):
            proc = subprocess.run(
                args, capture_output=True, text=True, cwd=str(ROOT),
            )
        if proc.returncode == 0:
            st.success("Pipeline completed! Refresh the other tabs to see results.")
            st.code(proc.stdout[-3000:], language="text")
        else:
            st.error("Pipeline failed.")
            st.code(proc.stderr[-3000:], language="text")

    st.markdown("---")
    st.markdown("### Run regime evaluation")
    st.caption(
        "Requires `--mode evaluate` to have already been run for this "
        "variant. Slices the saved H-L returns by NBER/VIX/decade."
    )
    if st.button("▶ Run --mode regimes"):
        import subprocess
        with st.spinner("Computing regime metrics…"):
            proc = subprocess.run(
                ["python", "main.py", "--mode", "regimes", "--variant", chosen_variant],
                capture_output=True, text=True, cwd=str(ROOT),
            )
        if proc.returncode == 0:
            st.success("Regimes complete — see the 'Regimes' tab.")
            st.code(proc.stdout[-2000:], language="text")
        else:
            st.error("Failed.")
            st.code(proc.stderr[-2000:], language="text")

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
