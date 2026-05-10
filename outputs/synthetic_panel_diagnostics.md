# Synthetic panel diagnostics

Diagnostics derived from `data/cache/synthetic_panels/*.parquet`. Values are computed from realised panel data only — not from scenario target parameters.

## future2026_base

- rows: 96000, months: 120, permnos: 800
- date range: 2026-04-30 → 2036-03-31
- market: mean(monthly)=0.0085, vol(monthly)=0.0410, ann.vol=0.1420
- market drawdown: max_dd=-0.2017, worst_month=-0.0871, best_month=0.1029
- cross-sectional dispersion (avg std of ret per date): 0.0809
- factor proxy |corr(stock, mkt)| avg: 0.4462
- rank AR(1) (latent signal): 0.2737
- momentum spread (1m fwd): 0.0235, reversal spread (1m fwd): -0.0235
- factor corrs: value~ret=0.0482, momentum~ret=0.0416
- warnings: none

## future2026_choppy

- rows: 96000, months: 120, permnos: 800
- date range: 2026-04-30 → 2036-03-31
- market: mean(monthly)=0.0013, vol(monthly)=0.0693, ann.vol=0.2400
- market drawdown: max_dd=-0.5003, worst_month=-0.1669, best_month=0.1974
- cross-sectional dispersion (avg std of ret per date): 0.1271
- factor proxy |corr(stock, mkt)| avg: 0.4848
- rank AR(1) (latent signal): 0.0347
- momentum spread (1m fwd): 0.0059, reversal spread (1m fwd): -0.0059
- factor corrs: value~ret=0.0447, momentum~ret=-0.0022
- notes: choppy/base market vol ratio=1.69 (threshold>=1.15)
- warnings: none

## future2026_crisis

- rows: 96000, months: 120, permnos: 800
- date range: 2026-04-30 → 2036-03-31
- market: mean(monthly)=-0.0009, vol(monthly)=0.0553, ann.vol=0.1917
- market drawdown: max_dd=-0.4805, worst_month=-0.2263, best_month=0.1182
- cross-sectional dispersion (avg std of ret per date): 0.0897
- factor proxy |corr(stock, mkt)| avg: 0.5208
- rank AR(1) (latent signal): 0.1840
- momentum spread (1m fwd): 0.0194, reversal spread (1m fwd): -0.0194
- factor corrs: value~ret=0.0491, momentum~ret=0.0705
- notes: crisis check: worst_month=-0.2263, max_dd=-0.4805
- warnings: none

## future2026_factor_rotation

- rows: 96000, months: 120, permnos: 800
- date range: 2026-04-30 → 2036-03-31
- market: mean(monthly)=0.0031, vol(monthly)=0.0346, ann.vol=0.1198
- market drawdown: max_dd=-0.2925, worst_month=-0.0812, best_month=0.1041
- cross-sectional dispersion (avg std of ret per date): 0.0801
- factor proxy |corr(stock, mkt)| avg: 0.3967
- rank AR(1) (latent signal): 0.2523
- momentum spread (1m fwd): 0.0205, reversal spread (1m fwd): -0.0205
- factor corrs: value~ret=0.0269, momentum~ret=0.0082
- notes: factor_rotation: value_sign_changes=56, momentum_sign_changes=41
- warnings: none

## future2026_mean_reversion

- rows: 96000, months: 120, permnos: 800
- date range: 2026-04-30 → 2036-03-31
- market: mean(monthly)=0.0090, vol(monthly)=0.0357, ann.vol=0.1236
- market drawdown: max_dd=-0.1460, worst_month=-0.0741, best_month=0.1231
- cross-sectional dispersion (avg std of ret per date): 0.0849
- factor proxy |corr(stock, mkt)| avg: 0.3843
- rank AR(1) (latent signal): -0.5056
- momentum spread (1m fwd): -0.0334, reversal spread (1m fwd): 0.0334
- factor corrs: value~ret=0.0197, momentum~ret=0.0320
- warnings: none

## future2026_rotating_leaders

- rows: 96000, months: 120, permnos: 800
- date range: 2026-04-30 → 2036-03-31
- market: mean(monthly)=0.0050, vol(monthly)=0.0377, ann.vol=0.1305
- market drawdown: max_dd=-0.2272, worst_month=-0.1174, best_month=0.1115
- cross-sectional dispersion (avg std of ret per date): 0.0799
- factor proxy |corr(stock, mkt)| avg: 0.4181
- rank AR(1) (latent signal): 0.0831
- momentum spread (1m fwd): 0.0032, reversal spread (1m fwd): -0.0032
- factor corrs: value~ret=0.0329, momentum~ret=0.0008
- notes: rotating_leaders: rank churn=0.878
- warnings: none

## future2026_trending

- rows: 96000, months: 120, permnos: 800
- date range: 2026-04-30 → 2036-03-31
- market: mean(monthly)=0.0009, vol(monthly)=0.0305, ann.vol=0.1056
- market drawdown: max_dd=-0.2306, worst_month=-0.0903, best_month=0.1026
- cross-sectional dispersion (avg std of ret per date): 0.0698
- factor proxy |corr(stock, mkt)| avg: 0.3884
- rank AR(1) (latent signal): 0.6135
- momentum spread (1m fwd): 0.0400, reversal spread (1m fwd): -0.0400
- factor corrs: value~ret=-0.0398, momentum~ret=0.0699
- warnings: none
