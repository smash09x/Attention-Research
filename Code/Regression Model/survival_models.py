"""
survival_models.py

Builds Table A8: right-censored survival models for the age-proxy dataset.

All 198 keyword-era rows are retained. Censored rows are peaks that never fell
back below the 20 percent threshold inside their era window, so their recorded
peak_width_days is a lower bound rather than a missing value.

Produces Kaplan-Meier medians, log-rank tests overall and within each era,
Cox proportional hazards with and without era terms, the three parametric AFT
fits, the likelihood-ratio test on all era terms, and restricted mean
attention duration.

Requires lifelines.

    python survival_models.py
"""

import argparse

from scipy import stats

from table_utils import ERA_LABELS, add_common_args, load_raw


def table_a8(age_raw):
    try:
        from lifelines import (CoxPHFitter, KaplanMeierFitter, LogNormalAFTFitter,
                               LogLogisticAFTFitter, WeibullAFTFitter)
        from lifelines.statistics import logrank_test
        from lifelines.utils import restricted_mean_survival_time
    except ImportError:
        raise SystemExit("Table A8 needs lifelines: pip install lifelines")

    df = age_raw.copy()
    df['platform_driven'] = df['platform_driven'].astype(int)
    df['has_persistence_reason'] = df['has_persistence_reason'].astype(int)
    df = df[['keyword', 'peak_width_days', 'event', 'era_ordinal',
             'platform_driven', 'has_persistence_reason']].dropna()

    print("\nTable A8")
    print("Right-Censored Survival Models for the Age-Proxy Dataset")
    print(f"n = {len(df)}, events = {int(df['event'].sum())}, "
          f"censored = {int((1 - df['event']).sum())}, clusters = {df['keyword'].nunique()}")

    print("\n--- Kaplan-Meier medians ---")
    km = KaplanMeierFitter()
    groups = (("all", df['event'].notna()),
              ("platform = 0", df['platform_driven'] == 0),
              ("platform = 1", df['platform_driven'] == 1),
              ("persistence = 0", df['has_persistence_reason'] == 0),
              ("persistence = 1", df['has_persistence_reason'] == 1))
    for name, mask in groups:
        grp = df[mask]
        km.fit(grp['peak_width_days'], grp['event'])
        print(f"{name:<18}n = {len(grp):<5}median = {km.median_survival_time_}")

    print("\n--- Log-rank, platform-driven ---")
    for era in (None, 0, 1, 2):
        grp = df if era is None else df[df['era_ordinal'] == era]
        a, b = grp[grp['platform_driven'] == 0], grp[grp['platform_driven'] == 1]
        res = logrank_test(a['peak_width_days'], b['peak_width_days'], a['event'], b['event'])
        tag = "overall" if era is None else ERA_LABELS[era]
        print(f"{tag:<12}chi2 = {res.test_statistic:.3f}, p = {res.p_value:.4f}")

    full_formula = ("era_ordinal + platform_driven + has_persistence_reason "
                    "+ era_ordinal:platform_driven + era_ordinal:has_persistence_reason")
    noera_formula = "platform_driven + has_persistence_reason"

    print("\n--- Cox, era ordinal (cluster-robust by keyword) ---")
    cox_full = CoxPHFitter()
    cox_full.fit(df, duration_col='peak_width_days', event_col='event',
                 cluster_col='keyword', formula=full_formula)
    cox_full.print_summary(decimals=3)

    print("\n--- Cox, no era terms ---")
    cox_noera = CoxPHFitter()
    cox_noera.fit(df, duration_col='peak_width_days', event_col='event',
                  cluster_col='keyword', formula=noera_formula)
    cox_noera.print_summary(decimals=3)

    "AFT fitters do not take a cluster column, so the keyword identifier is dropped."
    aft_df = df.drop(columns=['keyword'])

    print("\n--- Parametric AFT, era-ordinal specification (model fit) ---")
    weibull_full = None
    for name, fitter in (("weibull", WeibullAFTFitter()),
                         ("lognormal", LogNormalAFTFitter()),
                         ("loglogistic", LogLogisticAFTFitter())):
        fitter.fit(aft_df, duration_col='peak_width_days', event_col='event',
                   formula=full_formula)
        if name == "weibull":
            weibull_full = fitter
        key = [k for k in fitter.params_.index
               if 'era_ordinal' in str(k) and 'platform_driven' in str(k)]
        b = fitter.params_[key[0]]
        p = fitter.summary['p'][key[0]]
        print(f"{name:<14}AIC = {fitter.AIC_:.1f}   era x platform b = {b:.3f}, p = {p:.3f}")

    print("\n--- Lognormal AFT, no era terms (reported specification) ---")
    aft_noera = LogNormalAFTFitter()
    aft_noera.fit(aft_df, duration_col='peak_width_days', event_col='event',
                  formula=noera_formula)
    aft_noera.print_summary(decimals=3)
    print(f"AIC = {aft_noera.AIC_:.1f}")

    print("\n--- Likelihood-ratio test, all era terms (Weibull AFT) ---")
    weibull_noera = WeibullAFTFitter()
    weibull_noera.fit(aft_df, duration_col='peak_width_days', event_col='event',
                      formula=noera_formula)
    lr = 2 * (weibull_full.log_likelihood_ - weibull_noera.log_likelihood_)
    print(f"chi2(3) = {lr:.2f}, p = {stats.chi2.sf(lr, 3):.3f}")

    print("\n--- Restricted mean attention duration ---")
    for horizon in (90, 365):
        means = {}
        for flag in (0, 1):
            grp = df[df['platform_driven'] == flag]
            km.fit(grp['peak_width_days'], grp['event'])
            means[flag] = restricted_mean_survival_time(km, t=horizon)
        print(f"{horizon:>4}d   non-platform = {means[0]:.1f}   "
              f"platform = {means[1]:.1f}   ratio = {means[1] / means[0]:.3f}")


if __name__ == '__main__':
    parser = add_common_args(argparse.ArgumentParser(description="Build Table A8."))
    args = parser.parse_args()
    table_a8(load_raw(args.trends_age, args.strat_age, args.youtube_age))
