"""
Regression: CharacteristicsBuilder.build() vectorised path matches explicit per-permno loop.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import FREQ_MONTH_END
from src.data.characteristics import (
    AccrualsBuilder,
    CharacteristicsBuilder,
    FundamentalsBuilder,
    IndustryBuilder,
    LiquidityBuilder,
    MomentumBuilder,
    RiskBuilder,
    _cs_rank,
)


def _build_reference_loop(panel: pd.DataFrame, mkt_ret: pd.Series) -> pd.DataFrame:
    """Explicit per-permno loop (same logic as pre-vectorisation build, current maxret/baspread)."""
    df = panel.copy()
    df["me_lag1"] = df.groupby("permno")["me"].shift(1)
    df = df.merge(mkt_ret.rename("mkt_ret").reset_index(), on="date", how="left")
    df = df.sort_values(["permno", "date"], kind="mergesort").reset_index(drop=True)

    _has_baspread = "bid" in df.columns and "ask" in df.columns
    results = []
    for _permno, g in df.groupby("permno", sort=False):
        g = g.sort_values("date").copy()

        g["mom1m"] = g["ret"]
        g["mom6m"] = MomentumBuilder.mom6m(g["ret"])
        g["mom12m"] = MomentumBuilder.mom12m(g["ret"])
        g["mom36m"] = MomentumBuilder.mom36m(g["ret"])
        g["chmom"] = MomentumBuilder.chmom(g["ret"])
        g["maxret"] = np.nan

        g["mvel1"] = LiquidityBuilder.mvel1(g["prc"], g["shrout"])
        g["dolvol"] = LiquidityBuilder.dolvol(g["prc"], g["vol"])
        g["turn"] = LiquidityBuilder.turn(g["vol"], g["shrout"])
        g["std_turn"] = LiquidityBuilder.std_turn(g["vol"], g["shrout"])
        g["ill"] = LiquidityBuilder.ill(g["ret"], g["dolvol"])
        g["zerotrade"] = LiquidityBuilder.zerotrade(g["vol"])
        g["baspread"] = (
            LiquidityBuilder.baspread(g["bid"], g["ask"], g["prc"])
            if _has_baspread
            else np.nan
        )
        g["std_dolvol"] = LiquidityBuilder.std_dolvol(g["prc"], g["vol"])

        g["beta"] = RiskBuilder.beta(g["ret"], g["mkt_ret"])
        g["betasq"] = RiskBuilder.betasq(g["ret"], g["mkt_ret"])
        g["retvol"] = RiskBuilder.retvol(g["ret"])
        g["idiovol"] = RiskBuilder.idiovol(g["ret"], g["mkt_ret"])

        if "at" in g.columns:
            g["agr"] = FundamentalsBuilder.agr(g)
            g["invest"] = FundamentalsBuilder.invest(g)
            g["lev"] = FundamentalsBuilder.lev(g)
            g["bm"] = FundamentalsBuilder.bm(g)
            g["ep"] = FundamentalsBuilder.ep(g)
            g["sp"] = FundamentalsBuilder.sp(g)
            g["cfp"] = FundamentalsBuilder.cfp(g)
            g["dy"] = FundamentalsBuilder.dy(g)
            g["operprof"] = FundamentalsBuilder.operprof(g)
            g["gma"] = FundamentalsBuilder.gma(g)
            g["acc"] = AccrualsBuilder.acc(g)
            g["pctacc"] = AccrualsBuilder.pctacc(g)
            g["absacc"] = AccrualsBuilder.absacc(g)
            g["chcsho"] = FundamentalsBuilder.chcsho(g)
            g["nincr"] = FundamentalsBuilder.nincr(g)
            g["rd_mve"] = FundamentalsBuilder.rd_mve(g)
            g["cashdebt"] = FundamentalsBuilder.cashdebt(g)
            g["chinv"] = FundamentalsBuilder.chinv(g)
            g["lgr"] = FundamentalsBuilder.lgr(g)
            g["egr"] = FundamentalsBuilder.egr(g)
            g["sgr"] = FundamentalsBuilder.sgr(g)
            g["depr"] = FundamentalsBuilder.depr(g)
            g["cashpr"] = FundamentalsBuilder.cashpr(g)
            g["convind"] = FundamentalsBuilder.convind(g)
            g["securedind"] = FundamentalsBuilder.securedind(g)
            g["roeq"] = FundamentalsBuilder.roeq(g)
            g["roaq"] = FundamentalsBuilder.roaq(g)
            g["orgcap"] = FundamentalsBuilder.orgcap(g)
            g["rd_sale"] = (
                g.get("xrd", pd.Series(0.0, index=g.index)).fillna(0)
                / g.get("sale", pd.Series(np.nan, index=g.index)).replace(0, np.nan)
            )
            if "datadate" in g.columns:
                first_year = g["datadate"].min().year if "datadate" in g.columns else None
                if first_year:
                    g["age"] = g["date"].dt.year - first_year
                else:
                    g["age"] = np.nan
            else:
                g["age"] = np.nan

        results.append(g)

    out = pd.concat(results, ignore_index=True)
    out["indmom"] = IndustryBuilder.indmom_panel(out)
    sic_dummies = IndustryBuilder.sic2_dummies(out.get("siccd", pd.Series(["00"] * len(out))))
    out = pd.concat([out, sic_dummies], axis=1)

    cb = CharacteristicsBuilder(panel, mkt_ret)
    char_cols = cb._get_char_cols(out)
    for col in char_cols:
        out[col] = out.groupby("date")[col].transform(_cs_rank)
    for col in char_cols:
        out[col] = out.groupby("date")[col].transform(lambda x: x.fillna(x.median()))

    return out


def _make_panel(n_perm: int = 5, n_months: int = 60, seed: int = 42, with_bid_ask: bool = False):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2000-01", periods=n_months, freq=FREQ_MONTH_END)
    rows = []
    for p in range(1, n_perm + 1):
        for t in dates:
            prc = float(rng.uniform(10, 100))
            shr = float(rng.uniform(1e5, 5e5))
            vol = float(rng.uniform(100, 5000))
            ret = float(rng.normal(0.01, 0.05))
            me = prc * shr
            at = float(rng.uniform(50, 500))
            row = {
                "permno": p,
                "date": t,
                "ret": ret,
                "prc": prc,
                "shrout": shr,
                "vol": vol,
                "me": me,
                "at": at,
                "lt": at * 0.5,
                "act": at * 0.2,
                "lct": at * 0.15,
                "ib": at * 0.02,
                "ibq": at * 0.005,
                "sale": at * 1.5,
                "seq": at * 0.4,
                "depr_a": at * 0.02,
                "che": at * 0.05,
                "dlc": at * 0.08,
                "revt": at * 1.2,
                "cogs": at * 0.5,
                "xsga": at * 0.03,
                "xint": at * 0.01,
                "ppent": at * 0.35,
                "dvc": 0.0,
                "capx": at * 0.02,
                "invt": at * 0.1,
                "dltt": at * 0.2,
                "dcvt": 0.0,
                "dm": 0.0,
                "xrd": 0.0,
                "ceq": at * 0.3,
                "pstk": 0.0,
                "atq": at,
                "csho": shr / 1e3,
                # At least 6 months after datadate (CharacteristicsBuilder.__init__ guard)
                "datadate": t - pd.DateOffset(months=8),
                "siccd": 100 * p + 10,
            }
            if with_bid_ask:
                row["bid"] = prc * 0.99
                row["ask"] = prc * 1.01
            rows.append(row)
    panel = pd.DataFrame(rows)
    mkt = pd.Series(rng.normal(0.008, 0.04, len(dates)), index=dates)
    mkt.index.name = "date"
    return panel, mkt


@pytest.mark.parametrize("with_bid_ask", [False, True])
def test_vectorized_build_matches_permnos_loop(with_bid_ask: bool):
    panel, mkt = _make_panel(with_bid_ask=with_bid_ask)
    vec = CharacteristicsBuilder(panel, mkt).build()
    ref = _build_reference_loop(panel, mkt)

    vec = vec.sort_values(["permno", "date"], kind="mergesort").reset_index(drop=True)
    ref = ref.sort_values(["permno", "date"], kind="mergesort").reset_index(drop=True)

    assert vec.columns.tolist() == ref.columns.tolist()
    for c in vec.columns:
        if vec[c].dtype != ref[c].dtype:
            if vec[c].dtype.kind in "fiu" and ref[c].dtype.kind in "fiu":
                vec[c] = vec[c].astype("float64")
                ref[c] = ref[c].astype("float64")
            else:
                assert vec[c].dtype == ref[c].dtype, c

    num_cols = [c for c in vec.columns if pd.api.types.is_numeric_dtype(vec[c])]
    for c in num_cols:
        v = vec[c].astype(float).to_numpy()
        r = ref[c].astype(float).to_numpy()
        np.testing.assert_allclose(v, r, rtol=1e-12, atol=1e-12, equal_nan=True, err_msg=c)
