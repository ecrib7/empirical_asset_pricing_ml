"""
data/characteristics.py
-----------------------
Constructs the 94 firm-level characteristics from Green et al. (2017)
used in Gu, Kelly & Xiu (2019).

Each function takes a merged CRSP+Compustat panel and returns a series
or column to be added to that panel.

Naming follows the GKX (2019) Appendix Table A.6 exactly.

Organisation
------------
  MomentumBuilder   – price-trend signals (CRSP only)
  LiquidityBuilder  – market microstructure signals (CRSP only)
  RiskBuilder       – beta / volatility signals (CRSP only)
  AccrualsBuilder   – accrual-based signals (Compustat)
  FundamentalsBuilder – valuation & profitability (Compustat + CRSP)
  IndustryBuilder   – industry dummies (SIC)
  CharacteristicsBuilder – orchestrates all of the above
"""

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

# Suppress benign pandas warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


# ════════════════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════════════════

def _cs_rank(s: pd.Series) -> pd.Series:
    """Cross-sectional rank, normalised to [-1, 1]."""
    r = s.rank(method="average", na_option="keep")
    n = r.notna().sum()
    if n <= 1:
        return pd.Series(np.nan, index=s.index)
    return 2 * (r - 1) / (n - 1) - 1


def _groupby_permno_apply(df: pd.DataFrame, func):
    """Apply ``func(g)`` per ``permno``; use ``include_groups=False`` when available (pandas ≥2.2)."""
    gb = df.groupby("permno", sort=False, group_keys=False)
    try:
        return gb.apply(func, include_groups=False)
    except TypeError:
        return gb.apply(func)


def _winsorise(s: pd.Series, p: float = 0.01) -> pd.Series:
    lo, hi = s.quantile(p), s.quantile(1 - p)
    return s.clip(lower=lo, upper=hi)


def _rolling_beta(ret: pd.Series, mkt: pd.Series, window: int = 60,
                  min_periods: int = 24) -> pd.Series:
    """OLS beta from rolling window."""
    cov  = ret.rolling(window, min_periods=min_periods).cov(mkt)
    var  = mkt.rolling(window, min_periods=min_periods).var()
    return cov / var


# ════════════════════════════════════════════════════════════════════
#  Momentum signals  (CRSP monthly returns only)
# ════════════════════════════════════════════════════════════════════

class MomentumBuilder:
    @staticmethod
    def mom1m(ret: pd.Series) -> pd.Series:
        """1-month return (short-term reversal)."""
        return ret

    @staticmethod
    def mom6m(ret: pd.Series) -> pd.Series:
        """Cumulative return months t-7 to t-2."""
        return (1 + ret).rolling(6, min_periods=4).apply(np.prod, raw=True) - 1

    @staticmethod
    def mom12m(ret: pd.Series) -> pd.Series:
        """Cumulative return months t-13 to t-2 (skip 1 month)."""
        comp11 = (1 + ret).rolling(11, min_periods=8).apply(np.prod, raw=True) - 1
        return comp11.shift(1)  # shift to skip month t-1

    @staticmethod
    def mom36m(ret: pd.Series) -> pd.Series:
        """Cumulative return months t-37 to t-13."""
        comp24 = (1 + ret).rolling(24, min_periods=16).apply(np.prod, raw=True) - 1
        return comp24.shift(12)

    @staticmethod
    def chmom(ret: pd.Series) -> pd.Series:
        """Change in 6-month momentum (mom6m[t] – mom6m[t-6])."""
        m6 = MomentumBuilder.mom6m(ret)
        return m6 - m6.shift(6)

    @staticmethod
    def maxret(daily_ret: Optional[pd.Series] = None) -> pd.Series:
        """
        Maximum daily return in the past month (GKX 2019).
        Requires a daily return series aggregated to monthly frequency
        by taking the max within each calendar month.

        If only monthly data is available, set this to NaN and exclude
        from the feature set — do NOT use monthly ret as a proxy.
        """
        if daily_ret is None:
            # ``daily_ret.index`` is undefined here; placeholder until daily CRSP is wired.
            return pd.Series(np.nan, dtype=float)
        return daily_ret  # caller must pass monthly-max-of-daily already aggregated

    @staticmethod
    def indmom(ret: pd.Series, sic: pd.Series) -> pd.Series:
        """
        Industry momentum: value-weighted average past-year return of the
        2-digit SIC industry, lagged 1 month.
        Computed cross-sectionally each month.
        """
        warnings.warn(
            "MomentumBuilder.indmom() is a placeholder. "
            "Industry momentum must be computed at panel level using "
            "IndustryBuilder.indmom_panel(). This column will be all NaN.",
            stacklevel=2,
        )
        return pd.Series(np.nan, index=ret.index)


# ════════════════════════════════════════════════════════════════════
#  Liquidity / market microstructure signals
# ════════════════════════════════════════════════════════════════════

class LiquidityBuilder:
    @staticmethod
    def mvel1(prc: pd.Series, shrout: pd.Series) -> pd.Series:
        """Log market equity."""
        me = (prc * shrout).clip(lower=1e-6)
        return np.log(me)

    @staticmethod
    def dolvol(prc: pd.Series, vol: pd.Series) -> pd.Series:
        """Log average daily dollar volume in past month."""
        dv = (prc * vol * 1000).clip(lower=1e-6)   # vol in hundreds of shares
        return np.log(dv.rolling(12, min_periods=8).mean())

    @staticmethod
    def turn(vol: pd.Series, shrout: pd.Series) -> pd.Series:
        """Average monthly turnover (vol/shrout) past 12 months."""
        t = vol / shrout.replace(0, np.nan)
        return t.rolling(12, min_periods=8).mean()

    @staticmethod
    def std_turn(vol: pd.Series, shrout: pd.Series) -> pd.Series:
        """Std dev of monthly turnover past 12 months."""
        t = vol / shrout.replace(0, np.nan)
        return t.rolling(12, min_periods=8).std()

    @staticmethod
    def ill(ret: pd.Series, dolvol: pd.Series) -> pd.Series:
        """
        Amihud (2002) illiquidity = |ret| / dollar_volume.

        Parameters
        ----------
        ret    : monthly return series
        dolvol : log dollar volume series (i.e., output of LiquidityBuilder.dolvol).
                 This function internally calls np.exp(dolvol) to recover the level.
                 Do NOT pass raw dollar volume — pass the log.
        """
        dv = np.exp(dolvol).replace(0, np.nan)
        return (ret.abs() / dv).rolling(12, min_periods=8).mean() * 1e6

    @staticmethod
    def zerotrade(vol: pd.Series) -> pd.Series:
        """Number of zero-trading-day months in past 12 months."""
        return (vol == 0).rolling(12, min_periods=8).sum()

    @staticmethod
    def baspread(bid: pd.Series, ask: pd.Series, prc: pd.Series) -> pd.Series:
        """Bid-ask spread as % of price."""
        spread = (ask - bid) / prc.replace(0, np.nan)
        return spread.rolling(12, min_periods=8).mean()

    @staticmethod
    def std_dolvol(prc: pd.Series, vol: pd.Series) -> pd.Series:
        """Std dev of log dollar volume past 12 months."""
        ldv = np.log((prc * vol * 1000).clip(lower=1e-6))
        return ldv.rolling(12, min_periods=8).std()

    @staticmethod
    def pricedelay(ret: pd.Series, mkt_ret: pd.Series) -> pd.Series:
        """
        Hou & Moskowitz (2005) price delay.
        Ratio of R² improvement when lagged market returns are added.
        Approximated as 1 - R²(restricted) / R²(full) from rolling OLS.
        Full implementation requires per-stock regression.
        """
        warnings.warn(
            "LiquidityBuilder.pricedelay() is a placeholder. "
            "Price delay requires per-stock rolling OLS regression on lagged "
            "market returns and is not yet implemented. This column will be all NaN.",
            stacklevel=2,
        )
        return pd.Series(np.nan, index=ret.index)


# ════════════════════════════════════════════════════════════════════
#  Risk signals
# ════════════════════════════════════════════════════════════════════

class RiskBuilder:
    @staticmethod
    def beta(ret: pd.Series, mkt_ret: pd.Series) -> pd.Series:
        """Market beta (Fama-MacBeth style, 60-month rolling)."""
        return _rolling_beta(ret, mkt_ret, window=60, min_periods=24)

    @staticmethod
    def betasq(ret: pd.Series, mkt_ret: pd.Series) -> pd.Series:
        """Beta squared."""
        b = RiskBuilder.beta(ret, mkt_ret)
        return b ** 2

    @staticmethod
    def retvol(ret: pd.Series) -> pd.Series:
        """Total return volatility (std of past 36 monthly returns)."""
        return ret.rolling(36, min_periods=12).std()

    @staticmethod
    def idiovol(ret: pd.Series, mkt_ret: pd.Series) -> pd.Series:
        """
        Idiosyncratic volatility: std of residuals from market model.
        Computed as rolling 36-month residual volatility.
        """
        b    = _rolling_beta(ret, mkt_ret, window=36, min_periods=12)
        resid = ret - b * mkt_ret
        return resid.rolling(36, min_periods=12).std()


# ════════════════════════════════════════════════════════════════════
#  Accounting signals (Compustat – annual unless noted)
# ════════════════════════════════════════════════════════════════════

class AccrualsBuilder:
    @staticmethod
    def acc(df: pd.DataFrame) -> pd.Series:
        """
        Working capital accruals (Sloan 1996).
        acc = (ΔCA - ΔCash - ΔCL + ΔDebt_ST - Dep) / avg_Assets
        """
        cash = df.get("cheq", df.get("che", pd.Series(np.nan, index=df.index)))
        d_st = df.get("dlcq", df.get("dlc", pd.Series(np.nan, index=df.index)))
        dact = df["act"].diff() - cash.diff()
        dlct = df["lct"].diff() - d_st.diff()
        dep = df["depr_a"]
        avg_at = (df["at"] + df["at"].shift(1)) / 2
        return (dact - dlct - dep) / avg_at.replace(0, np.nan)

    @staticmethod
    def pctacc(df: pd.DataFrame) -> pd.Series:
        """Percent accruals (Hafzalla, Lundholm & Van Winkle 2011)."""
        ni = df["ib"].abs().replace(0, np.nan)
        return AccrualsBuilder.acc(df) / ni

    @staticmethod
    def absacc(df: pd.DataFrame) -> pd.Series:
        return AccrualsBuilder.acc(df).abs()

    @staticmethod
    def stdacc(df: pd.DataFrame) -> pd.Series:
        """Accrual volatility (past 4 quarters)."""
        # Uses quarterly data
        return df["acc_q"].rolling(4, min_periods=4).std() if "acc_q" in df.columns \
               else pd.Series(np.nan, index=df.index)


# ════════════════════════════════════════════════════════════════════
#  Valuation & Profitability signals
# ════════════════════════════════════════════════════════════════════

class FundamentalsBuilder:
    @staticmethod
    def _book_equity(df: pd.DataFrame) -> pd.Series:
        """
        Book equity = Stockholders' equity + deferred taxes – preferred stock.
        Following Fama & French (1993, 2015).
        """
        # Stockholders' equity (preferred order: seq, then ceq+pstk, then at-lt)
        se = df.get("seq", pd.Series(np.nan, index=df.index)).fillna(
             df.get("ceq", pd.Series(np.nan, index=df.index))
             + df.get("pstk", pd.Series(np.nan, index=df.index)))
        se = se.fillna(df["at"] - df["lt"])

        # Deferred taxes
        txditc = df.get("txditc", pd.Series(np.nan, index=df.index))

        # Preferred stock (use redemption value first, then liquidation, then carrying)
        ps = df.get("pstkrv", pd.Series(np.nan, index=df.index))
        ps = ps.fillna(df.get("pstkl", pd.Series(np.nan, index=df.index)))
        ps = ps.fillna(df.get("pstk", pd.Series(np.nan, index=df.index)))

        return se + txditc - ps

    @staticmethod
    def bm(df: pd.DataFrame) -> pd.Series:
        """Book-to-market ratio."""
        be = FundamentalsBuilder._book_equity(df)
        me = df.get("me", pd.Series(np.nan, index=df.index)).replace(0, np.nan)
        return be / me

    @staticmethod
    def ep(df: pd.DataFrame) -> pd.Series:
        """Earnings-to-price = ibq / me (quarterly)."""
        ib = df.get("ibq", df.get("ib", pd.Series(np.nan, index=df.index)))
        me = df.get("me", pd.Series(np.nan, index=df.index)).replace(0, np.nan)
        return ib / me

    @staticmethod
    def sp(df: pd.DataFrame) -> pd.Series:
        """Sales-to-price."""
        sale = df.get("saleq", df.get("sale", pd.Series(np.nan, index=df.index)))
        me   = df.get("me", pd.Series(np.nan, index=df.index)).replace(0, np.nan)
        return sale / me

    @staticmethod
    def cfp(df: pd.DataFrame) -> pd.Series:
        """Cash flow to price."""
        cf = df.get("ibq", df.get("ib", pd.Series(np.nan, index=df.index))) \
           + df.get("dp", df.get("depr_a", pd.Series(np.nan, index=df.index)))
        me = df.get("me", pd.Series(np.nan, index=df.index)).replace(0, np.nan)
        return cf / me

    @staticmethod
    def dy(df: pd.DataFrame) -> pd.Series:
        """Dividend yield."""
        # Missing common dividends (dvc) are treated as zero cash payout.
        div = df.get("dvc", pd.Series(np.nan, index=df.index)).fillna(0)
        me  = df.get("me", pd.Series(np.nan, index=df.index)).replace(0, np.nan)
        return div / me

    @staticmethod
    def agr(df: pd.DataFrame) -> pd.Series:
        """Asset growth = (at[t] - at[t-1]) / at[t-1]."""
        return df["at"].pct_change(1, fill_method=None)

    @staticmethod
    def invest(df: pd.DataFrame) -> pd.Series:
        """Capital expenditures and inventory growth."""
        capx = df.get("capx", pd.Series(np.nan, index=df.index))
        dinv = df.get("invt", pd.Series(np.nan, index=df.index)).diff()
        at_l = df["at"].shift(1).replace(0, np.nan)
        return (capx + dinv) / at_l

    @staticmethod
    def lev(df: pd.DataFrame) -> pd.Series:
        """Leverage = long-term debt / market equity."""
        dltt = df.get("dltt", pd.Series(np.nan, index=df.index))
        me   = df.get("me", pd.Series(np.nan, index=df.index)).replace(0, np.nan)
        return dltt / me

    @staticmethod
    def operprof(df: pd.DataFrame) -> pd.Series:
        """Operating profitability (Fama & French 2015)."""
        revt = df.get("revt", df.get("sale", pd.Series(np.nan, index=df.index)))
        cogs = df.get("cogs", pd.Series(0.0, index=df.index)).fillna(0)
        xsga = df.get("xsga", pd.Series(np.nan, index=df.index))
        xint = df.get("xint", pd.Series(np.nan, index=df.index))
        be   = FundamentalsBuilder._book_equity(df).replace(0, np.nan)
        return (revt - cogs - xsga - xint) / be

    @staticmethod
    def gma(df: pd.DataFrame) -> pd.Series:
        """Gross profitability (Novy-Marx 2013)."""
        gp = df.get("revt", df.get("sale", pd.Series(np.nan, index=df.index))) \
           - df.get("cogs", pd.Series(0.0, index=df.index)).fillna(0)
        at = df["at"].replace(0, np.nan)
        return gp / at

    @staticmethod
    def chcsho(df: pd.DataFrame) -> pd.Series:
        """% change in shares outstanding."""
        return df.get("csho", pd.Series(np.nan, index=df.index)).pct_change(1, fill_method=None)

    @staticmethod
    def nincr(df: pd.DataFrame) -> pd.Series:
        """
        Number of consecutive quarters of earnings increases (Barth et al. 1999).
        Approximated as number of YoY quarterly earnings increases in past 8 quarters.
        """
        ibq = df.get("ibq", pd.Series(np.nan, index=df.index))
        yoy_increase = (ibq > ibq.shift(4)).astype(float)
        return yoy_increase.rolling(8, min_periods=4).sum()

    @staticmethod
    def rd_mve(df: pd.DataFrame) -> pd.Series:
        """R&D to market capitalisation."""
        xrd = df.get("xrd", pd.Series(np.nan, index=df.index))
        me  = df.get("me", pd.Series(np.nan, index=df.index)).replace(0, np.nan)
        return xrd / me

    @staticmethod
    def cashdebt(df: pd.DataFrame) -> pd.Series:
        """Cash flow to debt."""
        cf  = df.get("ibq", df.get("ib", pd.Series(np.nan, index=df.index))) \
            + df.get("dp", df.get("depr_a", pd.Series(np.nan, index=df.index)))
        dltt = df.get("dltt", pd.Series(np.nan, index=df.index))
        dlc  = df.get("dlc", pd.Series(np.nan, index=df.index))
        debt = (dltt + dlc).replace(0, np.nan)
        return cf / debt

    @staticmethod
    def chinv(df: pd.DataFrame) -> pd.Series:
        """Change in inventory scaled by sales."""
        dinv = df.get("invt", pd.Series(np.nan, index=df.index)).diff()
        sale = df.get("sale", pd.Series(np.nan, index=df.index)).replace(0, np.nan)
        return dinv / sale

    @staticmethod
    def lgr(df: pd.DataFrame) -> pd.Series:
        """Growth in long-term debt."""
        return df.get("dltt", pd.Series(np.nan, index=df.index)).pct_change(1, fill_method=None)

    @staticmethod
    def egr(df: pd.DataFrame) -> pd.Series:
        """Growth in common shareholder equity."""
        return FundamentalsBuilder._book_equity(df).pct_change(1, fill_method=None)

    @staticmethod
    def sgr(df: pd.DataFrame) -> pd.Series:
        """Sales growth."""
        return df.get("sale", pd.Series(np.nan, index=df.index)).pct_change(1, fill_method=None)

    @staticmethod
    def depr(df: pd.DataFrame) -> pd.Series:
        """Depreciation / PP&E."""
        dp  = df.get("depr_a", pd.Series(np.nan, index=df.index))
        ppe = df.get("ppent", pd.Series(np.nan, index=df.index)).replace(0, np.nan)
        return dp / ppe

    @staticmethod
    def age(df: pd.DataFrame) -> pd.Series:
        """Number of years since first Compustat coverage."""
        return df.get("age_years", pd.Series(np.nan, index=df.index))

    @staticmethod
    def cashpr(df: pd.DataFrame) -> pd.Series:
        """Cash productivity: (me + dltt - at) / cheq."""
        me   = df.get("me", pd.Series(np.nan, index=df.index))
        dltt = df.get("dltt", pd.Series(np.nan, index=df.index))
        at   = df["at"]
        che  = df.get("cheq", df.get("che", pd.Series(np.nan, index=df.index))).replace(0, np.nan)
        return (me + dltt - at) / che

    @staticmethod
    def convind(df: pd.DataFrame) -> pd.Series:
        """Convertible debt indicator."""
        return (df.get("dcvt", pd.Series(np.nan, index=df.index)) > 0).astype(float)

    @staticmethod
    def securedind(df: pd.DataFrame) -> pd.Series:
        """Secured debt indicator."""
        return (df.get("dm", df.get("secured", pd.Series(np.nan, index=df.index))) > 0).astype(float)

    @staticmethod
    def roeq(df: pd.DataFrame) -> pd.Series:
        """Return on equity (quarterly)."""
        ibq = df.get("ibq", pd.Series(np.nan, index=df.index))
        beq = FundamentalsBuilder._book_equity(df).shift(1).replace(0, np.nan)
        return ibq / beq

    @staticmethod
    def roaq(df: pd.DataFrame) -> pd.Series:
        """Return on assets (quarterly)."""
        ibq = df.get("ibq", pd.Series(np.nan, index=df.index))
        atq = df.get("atq", df["at"]).shift(1).replace(0, np.nan)
        return ibq / atq

    @staticmethod
    def orgcap(df: pd.DataFrame) -> pd.Series:
        """Organizational capital (Eisfeldt & Papanikolaou 2013)."""
        xsga = df.get("xsga", pd.Series(np.nan, index=df.index))
        at   = df["at"].replace(0, np.nan)
        return xsga / at * 5   # simplified: 5× SG&A / assets


# ════════════════════════════════════════════════════════════════════
#  Industry signals
# ════════════════════════════════════════════════════════════════════

class IndustryBuilder:
    @staticmethod
    def sic2_dummies(sic: pd.Series) -> pd.DataFrame:
        """74 industry dummies based on first 2 digits of SIC code."""
        sic2 = sic.astype(str).str[:2].str.zfill(2)
        dummies = pd.get_dummies(sic2, prefix="sic2", dtype=float)
        # Ensure we have the right number by padding missing industries
        return dummies

    @staticmethod
    def indmom_panel(panel: pd.DataFrame) -> pd.Series:
        """
        Industry momentum (Moskowitz & Grinblatt 1999):
        equal-weighted average of stocks in the same 2-digit SIC industry,
        using past 12-month cumulative returns, lagged 1 month.
        """
        panel = panel.copy()
        panel["sic2"] = panel["siccd"].astype(str).str[:2]
        panel["mom12m_indmom"] = panel.groupby(["date", "sic2"])["ret"] \
                                      .transform(lambda x: x.shift(1).mean())
        return panel["mom12m_indmom"]


# ════════════════════════════════════════════════════════════════════
#  Master builder
# ════════════════════════════════════════════════════════════════════

class CharacteristicsBuilder:
    """
    Orchestrates the construction of all GKX (2019) characteristics
    from the merged CRSP + Compustat panel.

    Parameters
    ----------
    panel : pd.DataFrame
        Wide panel with columns from CRSP (monthly, sorted by permno+date)
        and Compustat (lagged appropriately before merging).
    mkt_ret : pd.Series
        Value-weighted market excess return (same date index as panel).
    """

    def __init__(self, panel: pd.DataFrame, mkt_ret: pd.Series):
        self.panel   = panel.copy().sort_values(["permno", "date"])
        self.mkt_ret = mkt_ret

    def build(self) -> pd.DataFrame:
        # Vectorised implementation — avoids explicit Python loop for performance on large panels
        df = self.panel.copy()

        # ── Market equity at previous month-end (used by many characteristics)
        df["me_lag1"] = df.groupby("permno", sort=False)["me"].shift(1)

        # ── Merge market return
        df = df.merge(self.mkt_ret.rename("mkt_ret").reset_index(), on="date", how="left")
        df = df.sort_values(["permno", "date"], kind="mergesort").reset_index(drop=True)

        _has_baspread = "bid" in df.columns and "ask" in df.columns
        if not _has_baspread:
            import logging

            logging.getLogger(__name__).warning(
                "baspread: 'bid' and 'ask' columns not found in panel. "
                "baspread will be NaN for all observations."
            )

        _gp = df.groupby("permno", sort=False)

        # ─ Momentum (per-stock rolling via transform) ─
        df["mom1m"] = df["ret"]
        df["mom6m"] = _gp["ret"].transform(lambda s: MomentumBuilder.mom6m(s))
        df["mom12m"] = _gp["ret"].transform(lambda s: MomentumBuilder.mom12m(s))
        df["mom36m"] = _gp["ret"].transform(lambda s: MomentumBuilder.mom36m(s))
        df["chmom"] = _gp["ret"].transform(lambda s: MomentumBuilder.chmom(s))
        # TODO: pass pre-aggregated daily max returns from daily CRSP pull
        df["maxret"] = np.nan  # requires daily CRSP — see MomentumBuilder.maxret

        # ─ Liquidity (vectorised rolling within permno; mvel1 is cross-sectional) ─
        df["mvel1"] = LiquidityBuilder.mvel1(df["prc"], df["shrout"])
        _dv = (df["prc"] * df["vol"] * 1000).clip(lower=1e-6)
        df["dolvol"] = _dv.groupby(df["permno"], sort=False).transform(
            lambda s: np.log(s.rolling(12, min_periods=8).mean())
        )
        _tvr = df["vol"] / df["shrout"].replace(0, np.nan)
        df["turn"] = _tvr.groupby(df["permno"], sort=False).transform(
            lambda s: s.rolling(12, min_periods=8).mean()
        )
        df["std_turn"] = _tvr.groupby(df["permno"], sort=False).transform(
            lambda s: s.rolling(12, min_periods=8).std()
        )
        _dv_level = np.exp(df["dolvol"]).replace(0, np.nan)
        df["ill"] = (
            (df["ret"].abs() / _dv_level)
            .groupby(df["permno"], sort=False)
            .transform(lambda s: s.rolling(12, min_periods=8).mean())
            * 1e6
        )
        df["zerotrade"] = (
            (df["vol"] == 0)
            .astype(float)
            .groupby(df["permno"], sort=False)
            .transform(lambda s: s.rolling(12, min_periods=8).sum())
        )
        if _has_baspread:
            _spread = (df["ask"] - df["bid"]) / df["prc"].replace(0, np.nan)
            df["baspread"] = _spread.groupby(df["permno"], sort=False).transform(
                lambda s: s.rolling(12, min_periods=8).mean()
            )
        else:
            df["baspread"] = np.nan
        _ldv = np.log((df["prc"] * df["vol"] * 1000).clip(lower=1e-6))
        df["std_dolvol"] = _ldv.groupby(df["permno"], sort=False).transform(
            lambda s: s.rolling(12, min_periods=8).std()
        )

        # ─ Risk (beta / idiovol need ret + mkt_ret per permno) ─
        df["beta"] = _groupby_permno_apply(df, lambda g: RiskBuilder.beta(g["ret"], g["mkt_ret"]))
        df["betasq"] = _groupby_permno_apply(df, lambda g: RiskBuilder.betasq(g["ret"], g["mkt_ret"]))
        df["retvol"] = _gp["ret"].transform(lambda s: RiskBuilder.retvol(s))
        df["idiovol"] = _groupby_permno_apply(df, lambda g: RiskBuilder.idiovol(g["ret"], g["mkt_ret"]))

        # ─ Accounting (if Compustat data merged in) ─
        if "at" in df.columns:
            df["agr"] = _groupby_permno_apply(df, lambda g: FundamentalsBuilder.agr(g))
            df["invest"] = _groupby_permno_apply(df, lambda g: FundamentalsBuilder.invest(g))
            df["lev"] = _groupby_permno_apply(df, lambda g: FundamentalsBuilder.lev(g))
            df["bm"] = _groupby_permno_apply(df, lambda g: FundamentalsBuilder.bm(g))
            df["ep"] = _groupby_permno_apply(df, lambda g: FundamentalsBuilder.ep(g))
            df["sp"] = _groupby_permno_apply(df, lambda g: FundamentalsBuilder.sp(g))
            df["cfp"] = _groupby_permno_apply(df, lambda g: FundamentalsBuilder.cfp(g))
            df["dy"] = _groupby_permno_apply(df, lambda g: FundamentalsBuilder.dy(g))
            df["operprof"] = _groupby_permno_apply(df, lambda g: FundamentalsBuilder.operprof(g))
            df["gma"] = _groupby_permno_apply(df, lambda g: FundamentalsBuilder.gma(g))
            df["acc"] = _groupby_permno_apply(df, lambda g: AccrualsBuilder.acc(g))
            df["pctacc"] = _groupby_permno_apply(df, lambda g: AccrualsBuilder.pctacc(g))
            df["absacc"] = _groupby_permno_apply(df, lambda g: AccrualsBuilder.absacc(g))
            df["chcsho"] = _groupby_permno_apply(df, lambda g: FundamentalsBuilder.chcsho(g))
            df["nincr"] = _groupby_permno_apply(df, lambda g: FundamentalsBuilder.nincr(g))
            df["rd_mve"] = _groupby_permno_apply(df, lambda g: FundamentalsBuilder.rd_mve(g))
            df["cashdebt"] = _groupby_permno_apply(df, lambda g: FundamentalsBuilder.cashdebt(g))
            df["chinv"] = _groupby_permno_apply(df, lambda g: FundamentalsBuilder.chinv(g))
            df["lgr"] = _groupby_permno_apply(df, lambda g: FundamentalsBuilder.lgr(g))
            df["egr"] = _groupby_permno_apply(df, lambda g: FundamentalsBuilder.egr(g))
            df["sgr"] = _groupby_permno_apply(df, lambda g: FundamentalsBuilder.sgr(g))
            df["depr"] = _groupby_permno_apply(df, lambda g: FundamentalsBuilder.depr(g))
            df["cashpr"] = _groupby_permno_apply(df, lambda g: FundamentalsBuilder.cashpr(g))
            df["convind"] = _groupby_permno_apply(df, lambda g: FundamentalsBuilder.convind(g))
            df["securedind"] = _groupby_permno_apply(df, lambda g: FundamentalsBuilder.securedind(g))
            df["roeq"] = _groupby_permno_apply(df, lambda g: FundamentalsBuilder.roeq(g))
            df["roaq"] = _groupby_permno_apply(df, lambda g: FundamentalsBuilder.roaq(g))
            df["orgcap"] = _groupby_permno_apply(df, lambda g: FundamentalsBuilder.orgcap(g))
            df["rd_sale"] = (
                df.get("xrd", pd.Series(0.0, index=df.index)).fillna(0)
                / df.get("sale", pd.Series(np.nan, index=df.index)).replace(0, np.nan)
            )
            if "datadate" in df.columns:
                _min_dd = df.groupby("permno", sort=False)["datadate"].transform("min")
                df["age"] = df["date"].dt.year - _min_dd.dt.year
                df.loc[_min_dd.isna(), "age"] = np.nan
            else:
                df["age"] = np.nan

        # ── Industry momentum (requires cross-section, computed here) ──
        df["indmom"] = IndustryBuilder.indmom_panel(df)

        # ── SIC2 dummies (added as separate columns) ──
        sic_dummies = IndustryBuilder.sic2_dummies(df.get("siccd", pd.Series(["00"] * len(df))))
        df = pd.concat([df, sic_dummies], axis=1)

        # ── Cross-sectional rank normalisation to [-1, 1] ──
        char_cols = self._get_char_cols(df)
        for col in char_cols:
            df[col] = df.groupby("date")[col].transform(_cs_rank)

        # ── Fill remaining NaN with cross-sectional median ──
        for col in char_cols:
            df[col] = df.groupby("date")[col].transform(
                lambda x: x.fillna(x.median())
            )

        return df

    def _get_char_cols(self, df: pd.DataFrame):
        known_chars = [
            "mom1m", "mom6m", "mom12m", "mom36m", "chmom", "indmom", "maxret",
            "mvel1", "dolvol", "turn", "std_turn", "ill", "zerotrade", "baspread",
            "std_dolvol", "pricedelay",
            "beta", "betasq", "retvol", "idiovol",
            "agr", "invest", "lev", "bm", "ep", "sp", "cfp", "dy",
            "operprof", "gma", "acc", "pctacc", "absacc", "chcsho", "nincr",
            "rd_mve", "cashdebt", "chinv", "lgr", "egr", "sgr", "depr",
            "cashpr", "convind", "securedind", "roeq", "roaq", "orgcap",
            "rd_sale", "age",
        ]
        return [c for c in known_chars if c in df.columns]


# ════════════════════════════════════════════════════════════════════
#  Feature matrix builder
# ════════════════════════════════════════════════════════════════════

def build_feature_matrix(
    panel: pd.DataFrame,
    macro: pd.DataFrame,
    char_cols: list,
    macro_cols: list = None,
) -> pd.DataFrame:
    """
    Constructs the GKX (2019) feature matrix z_{i,t} = x_t ⊗ c_{i,t}
    where x_t = (1, macro_1, …, macro_8) and c_{i,t} = firm characteristics.

    Adds 74 industry dummies as additional features.

    Returns full feature matrix appended to panel (permno, date, ret + features).
    """
    if macro_cols is None:
        macro_cols = ["dp", "ep", "bm", "ntis", "tbl", "tms", "dfy", "svar"]

    # Merge macro
    macro_sub = macro[["date"] + macro_cols].copy()
    macro_rename = {m: f"macro_{m}" for m in macro_cols}
    macro_sub = macro_sub.rename(columns=macro_rename)
    macro_prefixed = [f"macro_{m}" for m in macro_cols]
    panel = panel.merge(macro_sub, on="date", how="left")
    macro_filled = panel[macro_prefixed].fillna(0.0)

    # Kronecker product: for each macro variable (+ constant), multiply all chars
    feature_dfs = []

    # Constant × chars
    for c in char_cols:
        if c in panel.columns:
            feature_dfs.append(panel[[c]].rename(columns={c: f"{c}_const"}))

    # Macro × chars
    for m in macro_cols:
        m_col = f"macro_{m}"
        for c in char_cols:
            if c in panel.columns:
                feat_name = f"{c}_{m}"
                feature_dfs.append(
                    (panel[c] * macro_filled[m_col]).rename(feat_name).to_frame()
                )

    features = pd.concat(feature_dfs, axis=1)

    # Add SIC2 dummies
    sic_cols = [c for c in panel.columns if c.startswith("sic2_")]
    if sic_cols:
        features = pd.concat([features, panel[sic_cols]], axis=1)

    # Final feature matrix
    result = pd.concat([panel[["permno", "date", "ret", "me"]], features], axis=1)
    return result.reset_index(drop=True)
