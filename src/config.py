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
