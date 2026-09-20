"Table A8: right-censored survival models, age-proxy data. Needs lifelines. Run: python survival_models.py"

import argparse

import numpy as np
from scipy import stats

from table_utils import ERA_LABELS, add_common_args, load_raw


def fit_clustered_aft(fitter_cls, df, formula):
    "AFT fit with keyword-clustered sandwich SEs, G/(G-1) corrected. lifelines has no cluster option."
    from autograd import grad
    from lifelines import utils

    groups = df['keyword'].values
    data = df.drop(columns='keyword').reset_index(drop=True)

    class Clustered(fitter_cls):
        def _compute_sandwich_errors(self, Ts, E, weights, entries, Xs):
            with np.errstate(all='ignore'):
                score = grad(self._neg_likelihood_with_penalty_function)
                params = self.params_.values
                sums = {}
                for i, (ts, e, w, s, (_, xs)) in enumerate(zip(
                        utils.safe_zip(*Ts), E, weights, entries, Xs.iterrows())):
                    xs = utils.DataframeSlicer(xs.to_frame().T)
                    sums[groups[i]] = sums.get(groups[i], 0) + score(params, ts, e, w, s, xs)
                meat = sum(np.outer(v, v) for v in sums.values())
                bread = self.variance_matrix_.values
                return bread @ meat @ bread * len(sums) / (len(sums) - 1)

    fitter = Clustered()
    fitter.fit(data, duration_col='peak_width_days', event_col='event',
               formula=formula, robust=True)
    return fitter


def table_a8(age_raw):
    "Censored rows are lower bounds on peak width, so all rows are kept."
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

    "AFT fitters take no cluster column, so drop keyword."
    aft_df = df.drop(columns=['keyword'])

    "Paper AIC is on the log-time scale: lifelines AIC minus 2 x sum of log(t) over events."
    log_t = float(np.log(df.loc[df['event'] == 1, 'peak_width_days']).sum())

    print("\n--- Parametric AFT, AIC by distribution (lifelines / log-time scale) ---")
    fits = {}
    for spec, formula in (("era ordinal", full_formula), ("no era", noera_formula)):
        for name, cls in (("weibull", WeibullAFTFitter),
                          ("lognormal", LogNormalAFTFitter),
                          ("loglogistic", LogLogisticAFTFitter)):
            fitter = cls()
            fitter.fit(aft_df, duration_col='peak_width_days', event_col='event',
                       formula=formula)
            fits[(spec, name)] = fitter
            line = (f"{spec:<12}{name:<13}AIC = {fitter.AIC_:.1f}   "
                    f"log-time AIC = {fitter.AIC_ - 2 * log_t:.1f}")
            print(line)

    "Era x platform p-values quoted in the paper are keyword-clustered, like every other SE here."
    print("\n--- Era x platform interaction, keyword-clustered SEs ---")
    for name, cls in (("weibull", WeibullAFTFitter),
                      ("lognormal", LogNormalAFTFitter),
                      ("loglogistic", LogLogisticAFTFitter)):
        clustered = fit_clustered_aft(cls, df, full_formula)
        key = [k for k in clustered.params_.index
               if 'era_ordinal' in str(k) and 'platform_driven' in str(k)][0]
        row = clustered.summary.loc[key]
        print(f"{name:<13}b = {clustered.params_[key]:.3f}, SE = {row['se(coef)']:.3f}, "
              f"p = {row['p']:.3f}")

    print("\n--- Lognormal AFT, no era terms (reported specification), keyword-clustered ---")
    aft_noera = fit_clustered_aft(LogNormalAFTFitter, df, noera_formula)
    table = aft_noera.summary.loc['mu_'].loc[['platform_driven', 'has_persistence_reason']]
    for name, row in table.iterrows():
        print(f"{name:<24}b = {row['coef']:.3f}, SE = {row['se(coef)']:.3f}, "
              f"p = {row['p']:.4f}, time ratio = {row['exp(coef)']:.3f} "
              f"[{row['exp(coef) lower 95%']:.3f}, {row['exp(coef) upper 95%']:.3f}]")
    print(f"AIC = {aft_noera.AIC_:.1f}   log-time AIC = {aft_noera.AIC_ - 2 * log_t:.1f}")

    print("\n--- Likelihood-ratio test, all era terms (Weibull AFT) ---")
    weibull_full = fits[("era ordinal", "weibull")]
    weibull_noera = fits[("no era", "weibull")]
    lr = 2 * (weibull_full.log_likelihood_ - weibull_noera.log_likelihood_)
    print(f"chi2(3) = {lr:.2f}, p = {stats.chi2.sf(lr, 3):.3f}")

    print("\n--- Restricted mean attention duration, platform / non-platform ---")
    subsets = [("all", df)] + [(ERA_LABELS[e], df[df['era_ordinal'] == e]) for e in (0, 1, 2)]
    for horizon in (90, 365):
        for label, sub in subsets:
            means = {}
            for flag in (0, 1):
                grp = sub[sub['platform_driven'] == flag]
                km.fit(grp['peak_width_days'], grp['event'])
                means[flag] = restricted_mean_survival_time(km, t=horizon)
            print(f"{horizon:>4}d {label:<9}non-platform = {means[0]:.1f}   "
                  f"platform = {means[1]:.1f}   ratio = {means[1] / means[0]:.3f}")


if __name__ == '__main__':
    parser = add_common_args(argparse.ArgumentParser(description="Build Table A8."))
    args = parser.parse_args()
    table_a8(load_raw(args.trends_age, args.strat_age, args.youtube_age))
