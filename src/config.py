"""
config.py
---------
Central configuration for the GKX (2019) replication.
All tunable parameters live here; nothing is hard-coded in model files.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import pandas as _pd

# ─────────────────────────────────────────────────────────────────────────────
#  Pandas frequency alias compatibility
#  pandas <  2.2  : "M"  = month-end,  "AS" = year-start
#  pandas >= 2.2  : "ME" = month-end,  "YS" = year-start
# ─────────────────────────────────────────────────────────────────────────────
_pd_ver = tuple(int(x) for x in _pd.__version__.split(".")[:2])
FREQ_MONTH_END  = "ME" if _pd_ver >= (2, 2) else "M"
FREQ_YEAR_START = "YS" if _pd_ver >= (2, 2) else "AS"


# ─────────────────────────────────────────
#  Sample-split dates  (paper: Table 1)
# ─────────────────────────────────────────
TRAIN_START   = "1957-03-01"
TRAIN_END     = "1974-12-31"   # initial training window end
VAL_START     = "1975-01-01"
VAL_END       = "1986-12-31"   # fixed 12-yr validation
TEST_START    = "1987-01-01"
TEST_END      = "2016-12-31"   # 30-yr out-of-sample test


# ─────────────────────────────────────────
#  Real-vs-synthetic data boundary
#  Verified against WRDS on 2026-05-10. See
#  scripts/check_wrds_coverage.py and the manifest at
#  outputs/data_coverage/coverage_latest.json.
#
#  Subscription coverage on 2026-05-10:
#    * Legacy CRSP monthly stock file (crsp.msf) stops at 2024-12-31.
#    * CIZ/v2 monthly tables extend further:
#        - crsp.msf_v2 / crsp.stkmthsecuritydata     -> 2025-12-31
#        - crsp_q_stock.msf_v2 / .stkmthsecuritydata -> 2026-03-31
#    * Compustat funda/fundq -> 2026-04-30; CCM linkenddt -> 2026-01-30.
#
#  REAL_DATA_END is the most extended real-data endpoint usable by the
#  CIZ-aware pipeline (2026-03-31 from crsp_q_stock). Anything strictly
#  after it must be flagged as synthetic by downstream code.
#
#  Legacy callers that must remain compatible with the legacy crsp.msf
#  endpoint should reference ``LEGACY_REAL_DATA_END`` instead. The
#  ``paper`` variant (1957–2016) is unaffected by either constant; the
#  ``improved`` variant remains legacy-compatible (data_end = 2024-12-31)
#  while the new ``extended_ciz_2026`` variant uses the CIZ endpoint.
# ─────────────────────────────────────────
LEGACY_REAL_DATA_END   = "2024-12-31"   # crsp.msf max(date) on user's WRDS
LEGACY_SYNTHETIC_START = "2025-01-31"   # first month-end after legacy endpoint
REAL_DATA_END          = "2026-03-31"   # CIZ-aware: crsp_q_stock max(mthcaldt)
SYNTHETIC_START        = "2026-04-30"   # first month-end strictly after REAL_DATA_END


# ─────────────────────────────────────────
#  Two-pipeline configuration
#  ----------------------------------------
#  "paper"    : strict GKX (2019) reproduction
#  "improved" : extended sample, macro × char interactions ON,
#               industry dummies ON, transaction costs modelled
# ─────────────────────────────────────────
VARIANT_DEFAULTS = {
    "paper": {
        "data_start":   "1957-01-01",
        "data_end":     "2016-12-31",
        "train_start":  "1957-03-01",
        "val_start":    "1975-01-01",
        "val_end":      "1986-12-31",
        "test_start":   "1987-01-01",
        "test_end":     "2016-12-31",
        "use_macro_interactions": True,   # GKX uses 920 = 94 × (1+8) + 74 dummies
        "use_industry_dummies":   True,
        "tc_bps":                 0.0,    # paper headline numbers are gross
        "tc_model":               "flat",
        "output_dir":             "outputs/paper",
        "model_dir":              "outputs/paper/models",
        "feature_cache":          "data/cache/feature_matrix_paper.parquet",
        "checkpoint_subdir":      "backtest_checkpoint_paper",
    },
    "improved": {
        "data_start":   "1957-01-01",
        "data_end":     "2024-12-31",     # extends paper sample by ~8 years
        "train_start":  "1957-03-01",
        "val_start":    "1975-01-01",
        "val_end":      "1986-12-31",
        "test_start":   "1987-01-01",
        "test_end":     "2024-12-31",
        "use_macro_interactions": True,
        "use_industry_dummies":   True,
        "tc_bps":                 10.0,   # fallback for stocks with missing metadata
        "tc_model":               "stock_level",  # FIM + per-stock vol scaling
        "tc_vol_spread_bps":      8.0,    # +bps half-spread per +1 σ_ret z-score
        "tc_vol_impact_scale":    0.4,    # impact multiplier slope on vol
        "tc_nav_billions":        1.0,    # strategy AUM ($B) — scales sqrt-impact
        "output_dir":             "outputs/improved",
        "model_dir":              "outputs/improved/models",
        "feature_cache":          "data/cache/feature_matrix_improved.parquet",
        "checkpoint_subdir":      "backtest_checkpoint_improved",
    },
    # ─────────────────────────────────────────────────────────────
    # extended_2024 — explicit "real-only, post-paper extension"
    # variant. Same pipeline knobs as 'improved' but the dates are
    # documented to align with the verified WRDS coverage cutoff
    # (REAL_DATA_END = 2024-12-31). It carries metadata about the
    # real/synthetic boundary so downstream code can distinguish
    # real-data backtests from synthetic stress tests without
    # mutating the existing 'paper' / 'improved' variants.
    # ─────────────────────────────────────────────────────────────
    "extended_2024": {
        "data_start":   "1957-01-01",
        "data_end":     "2024-12-31",
        "train_start":  "1957-03-01",
        "val_start":    "1975-01-01",
        "val_end":      "1986-12-31",
        "test_start":   "2017-01-01",   # post-paper out-of-sample
        "test_end":     "2024-12-31",
        "use_macro_interactions": True,
        "use_industry_dummies":   True,
        "tc_bps":                 10.0,
        "tc_model":               "stock_level",
        "tc_vol_spread_bps":      8.0,
        "tc_vol_impact_scale":    0.4,
        "tc_nav_billions":        1.0,
        "output_dir":             "outputs/extended_2024",
        "model_dir":              "outputs/extended_2024/models",
        "feature_cache":          "data/cache/feature_matrix_extended_2024.parquet",
        "checkpoint_subdir":      "backtest_checkpoint_extended_2024",
        # Real/synthetic metadata (read by scripts/check_wrds_coverage.py
        # and src/synthetic/regimes.py — does not affect existing variants).
        # This variant is pinned to the LEGACY crsp.msf endpoint
        # (2024-12-31) to remain reproducible against the user's legacy
        # subscription. CIZ-aware extension lives in extended_ciz_2026.
        "real_data_end":          LEGACY_REAL_DATA_END,
        "synthetic_start":        LEGACY_SYNTHETIC_START,
        "synthetic_enabled":      False,
    },
    # ─────────────────────────────────────────────────────────────
    # extended_ciz_2026 — CIZ/v2-aware extension. Same pipeline knobs
    # as 'extended_2024' but the data_end / test_end follow the
    # furthest CIZ monthly endpoint observed on the user's WRDS
    # subscription (crsp_q_stock.* -> 2026-03-31). Use this variant
    # for CIZ-aware backtests; keep 'extended_2024' for legacy-msf
    # reproducibility.
    # ─────────────────────────────────────────────────────────────
    "extended_ciz_2026": {
        "data_start":   "1957-01-01",
        "data_end":     "2026-03-31",
        "train_start":  "1957-03-01",
        "val_start":    "1975-01-01",
        "val_end":      "1986-12-31",
        "test_start":   "2017-01-01",   # post-paper out-of-sample
        "test_end":     "2026-03-31",
        "use_macro_interactions": True,
        "use_industry_dummies":   True,
        "tc_bps":                 10.0,
        "tc_model":               "stock_level",
        "tc_vol_spread_bps":      8.0,
        "tc_vol_impact_scale":    0.4,
        "tc_nav_billions":        1.0,
        "output_dir":             "outputs/extended_ciz_2026",
        "model_dir":              "outputs/extended_ciz_2026/models",
        "feature_cache":          "data/cache/feature_matrix_extended_ciz_2026.parquet",
        "checkpoint_subdir":      "backtest_checkpoint_extended_ciz_2026",
        # CIZ-aware boundary: REAL_DATA_END/SYNTHETIC_START are now the
        # CIZ endpoints (2026-03-31 / 2026-04-30).
        "real_data_end":          REAL_DATA_END,
        "synthetic_start":        SYNTHETIC_START,
        "synthetic_enabled":      False,
    },
    # ─────────────────────────────────────────────────────────────
    # post2016_ciz — *extension/scoring* variant intended to reuse
    # existing trained-model pickles (paper or improved) over the
    # post-2016 out-of-sample window without re-running the full
    # 1957→2026 backtest. The data_start is shifted to 2015-01-01
    # so the 12-month / rolling characteristics have one full year
    # of warmup before test_start. It is NOT a full retrain — see
    # ``run_predict`` in main.py.
    # ─────────────────────────────────────────────────────────────
    "post2016_ciz": {
        "data_start":   "2015-01-01",  # 12m warmup before test_start
        "data_end":     "2026-03-31",
        # train_start / val_* kept for compatibility with code that
        # expects them, but training is NOT run for this variant.
        "train_start":  "2015-01-01",
        "val_start":    "2016-01-01",
        "val_end":      "2016-12-31",
        "test_start":   "2017-01-01",
        "test_end":     "2026-03-31",
        "use_macro_interactions": True,
        "use_industry_dummies":   True,
        "tc_bps":                 10.0,
        "tc_model":               "stock_level",
        "tc_vol_spread_bps":      8.0,
        "tc_vol_impact_scale":    0.4,
        "tc_nav_billions":        1.0,
        "output_dir":             "outputs/post2016_ciz",
        "model_dir":              "outputs/post2016_ciz/models",
        "feature_cache":          "data/cache/feature_matrix_post2016_ciz.parquet",
        "checkpoint_subdir":      "backtest_checkpoint_post2016_ciz",
        "real_data_end":          REAL_DATA_END,
        "synthetic_start":        SYNTHETIC_START,
        "synthetic_enabled":      False,
        # Marker: this variant is intended for `--mode predict`,
        # not `--mode train`. main.py reads this flag for guard logic.
        "is_scoring_variant":     True,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
#  future2026 — fully synthetic post-WRDS scenarios (2026-04-30..2036-03-31).
#  These variants do not consume WRDS data. They are produced by
#  ``generate_synthetic_results.py`` and emit the standard output artifacts
#  (models/*.pkl, portfolio_returns.pkl, metrics.json, comprehensive.csv,
#  oos_r2.csv, sharpe_table.csv, dm_table.csv, dm_pvalues.csv, regimes.csv,
#  var_importance.csv) so the existing dashboard reads them unchanged.
#  Scenarios are inspired by the anticor-trader regime taxonomy.
# ─────────────────────────────────────────────────────────────────────────────
FUTURE2026_START = "2026-04-30"
FUTURE2026_END   = "2036-03-31"
FUTURE2026_SCENARIOS = (
    "future2026_base",
    "future2026_trending",
    "future2026_mean_reversion",
    "future2026_rotating_leaders",
    "future2026_choppy",
    "future2026_crisis",
    "future2026_factor_rotation",
)


def _future2026_defaults(name: str) -> dict:
    return {
        # Synthetic months only: no warmup, no real data.
        "data_start":   FUTURE2026_START,
        "data_end":     FUTURE2026_END,
        "train_start":  FUTURE2026_START,
        "val_start":    FUTURE2026_START,
        "val_end":      FUTURE2026_START,
        "test_start":   FUTURE2026_START,
        "test_end":     FUTURE2026_END,
        "use_macro_interactions": True,
        "use_industry_dummies":   True,
        "tc_bps":                 10.0,
        "tc_model":               "stock_level",
        "tc_vol_spread_bps":      8.0,
        "tc_vol_impact_scale":    0.4,
        "tc_nav_billions":        1.0,
        "output_dir":             f"outputs/{name}",
        "model_dir":              f"outputs/{name}/models",
        "feature_cache":          f"data/cache/feature_matrix_{name}.parquet",
        "checkpoint_subdir":      f"backtest_checkpoint_{name}",
        # Stock-level synthetic panel that backs this variant. See
        # ``src/synthetic/panels.py`` and ``generate_synthetic_results.py``.
        # The panel is a 120-month × 800-permno parquet used as the source
        # of truth for decile portfolio construction.
        "synthetic_panel_path":   f"data/cache/synthetic_panels/{name}.parquet",
        # No-WRDS / synthetic semantics: every month in the panel is
        # synthetic. No real-data lookup is permitted.
        "real_data_end":          REAL_DATA_END,
        "synthetic_start":        FUTURE2026_START,
        "synthetic_enabled":      True,
        "is_scoring_variant":     False,
        "is_synthetic_only":      True,
        "scenario":               name.replace("future2026_", ""),
    }


for _scn in FUTURE2026_SCENARIOS:
    VARIANT_DEFAULTS[_scn] = _future2026_defaults(_scn)


def get_variant_config(name: str) -> dict:
    """Return a dict of defaults for the named variant ('paper' or 'improved')."""
    if name not in VARIANT_DEFAULTS:
        raise ValueError(
            f"Unknown variant {name!r}. Use one of: {list(VARIANT_DEFAULTS)}"
        )
    return dict(VARIANT_DEFAULTS[name])


# ─────────────────────────────────────────
#  Macro predictors  (Welch & Goyal 2008)
# ─────────────────────────────────────────
MACRO_VARS = ["dp", "ep", "bm", "ntis", "tbl", "tms", "dfy", "svar"]


# ─────────────────────────────────────────
#  Characteristic groups (Green et al. 2017)
# ─────────────────────────────────────────
MOMENTUM_CHARS   = ["mom1m", "mom6m", "mom12m", "mom36m", "chmom", "indmom"]
LIQUIDITY_CHARS  = ["mvel1", "dolvol", "turn", "std_turn", "ill", "zerotrade", "baspread",
                    "std_dolvol"]
RISK_CHARS       = ["beta", "betasq", "idiovol", "retvol"]
VALUATION_CHARS  = ["bm", "ep", "sp", "cfp", "dy", "rd_mve", "cashpr"]
QUALITY_CHARS    = ["agr", "invest", "chcsho", "nincr", "operprof", "gma",
                    "roeq", "roaq", "acc", "lev", "egr", "sgr", "lgr"]
ACCRUAL_CHARS    = ["acc", "pctacc", "absacc", "stdacc", "cashdebt"]
OTHER_CHARS      = ["age", "rd_sale", "depr", "convind", "securedind", "chinv",
                    "chmom", "chpmia", "chatoia", "orgcap"]

ALL_CHARACTERISTICS = list(dict.fromkeys(
    MOMENTUM_CHARS + LIQUIDITY_CHARS + RISK_CHARS + VALUATION_CHARS +
    QUALITY_CHARS + ACCRUAL_CHARS + OTHER_CHARS
))

# Number of macro interactions (8 macros + 1 constant = 9)
N_MACRO = len(MACRO_VARS) + 1   # "+1" for constant
N_INDUSTRY_DUMMIES = 74


# ─────────────────────────────────────────
#  Model hyper-parameter grids
# ─────────────────────────────────────────
@dataclass
class ElasticNetConfig:
    alpha_grid: List[float] = field(default_factory=lambda: [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1])
    l1_ratio: float = 0.5        # rho=0.5 (paper default)
    huber_epsilon: float = 1.35  # Huber loss parameter
    max_iter: int = 2000


@dataclass
class PCRConfig:
    n_components_grid: List[int] = field(default_factory=lambda: [1, 3, 5, 10, 20, 30, 40, 50])


@dataclass
class PLSConfig:
    n_components_grid: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 8, 10])


@dataclass
class RandomForestConfig:
    n_estimators: int = 300
    max_depth_grid: List[Optional[int]] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])
    max_features_grid: List[str] = field(default_factory=lambda: ["sqrt", "log2"])
    n_jobs: int = -1
    random_state: int = 42


@dataclass
class GBRTConfig:
    n_estimators_grid: List[int] = field(default_factory=lambda: [100, 300, 500, 1000])
    max_depth_grid: List[int] = field(default_factory=lambda: [1, 2])
    learning_rate_grid: List[float] = field(default_factory=lambda: [0.01, 0.1])
    subsample: float = 0.5
    random_state: int = 42


@dataclass
class NeuralNetConfig:
    # NN1: [32], NN2: [32,16], NN3: [32,16,8], NN4: [32,16,8,4], NN5: [32,16,8,4,2]
    architectures: List[List[int]] = field(default_factory=lambda: [
        [32],
        [32, 16],
        [32, 16, 8],
        [32, 16, 8, 4],
        [32, 16, 8, 4, 2],
    ])
    l1_lambda_grid: List[float] = field(default_factory=lambda: [1e-5, 1e-4, 1e-3])
    learning_rate_grid: List[float] = field(default_factory=lambda: [0.001, 0.01])
    batch_size: int = 10000
    max_epochs: int = 100
    patience: int = 5       # early stopping
    n_ensemble: int = 10    # ensemble seeds
    dropout_rate: float = 0.0
    random_seed: int = 42


# ─────────────────────────────────────────
#  Portfolio construction
# ─────────────────────────────────────────
@dataclass
class PortfolioConfig:
    n_deciles: int = 10
    weighting: str = "value"           # "value" or "equal"
    long_decile: int = 10              # top decile → long
    short_decile: int = 1             # bottom decile → short
    transaction_cost_bps: float = 10.0 # one-way cost in basis points
    max_leverage: float = 1.5          # Campbell-Thompson market timing cap


# ─────────────────────────────────────────
#  Paths
# ─────────────────────────────────────────
DATA_DIR    = "data/"
OUTPUT_DIR  = "outputs/"
LOG_DIR     = "logs/"
MODEL_DIR   = "outputs/models/"
CACHE_DIR   = "data/cache/"