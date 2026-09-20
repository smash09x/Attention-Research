"Robustness figures quoted in the prose, not in tables. Run: python robustness_checks.py"

import argparse

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from table_utils import (BASE_TERMS, BOOTSTRAP_REPS, BOOTSTRAP_SEED, add_common_args,
                         analytic, fit_ols, load_raw, main_effect_term, platform_term)


def primary_checks(primary):
    print("\n--- Robustness checks, primary dataset ---")

    model = fit_ols(primary, f"log_decay_speed ~ {BASE_TERMS}")
    inter = platform_term(model)
    print(f"HC3:               b = {model.params[inter]:.3f}, "
          f"SE = {model.bse[inter]:.3f}, p = {model.pvalues[inter]:.4f}")

    clustered = smf.ols(f"log_decay_speed ~ {BASE_TERMS}", data=primary).fit(
        cov_type='cluster', cov_kwds={'groups': primary['keyword']})
    print(f"Cluster-robust:    b = {clustered.params[inter]:.3f}, "
          f"SE = {clustered.bse[inter]:.3f}, p = {clustered.pvalues[inter]:.4f}")

    "Bootstrap resamples keywords, not rows (7 keywords have 2 rows)."
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    keywords = primary['keyword'].unique()
    by_keyword = {k: g for k, g in primary.groupby('keyword')}
    draws = []
    for _ in range(BOOTSTRAP_REPS):
        picked = rng.choice(keywords, size=len(keywords), replace=True)
        sample = pd.concat([by_keyword[k] for k in picked], ignore_index=True)
        try:
            draws.append(smf.ols(f"log_decay_speed ~ {BASE_TERMS}",
                                 data=sample).fit().params[inter])
        except Exception:
            continue
    draws = np.array(draws)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    print(f"Cluster bootstrap: 95% CI [{lo:.3f}, {hi:.3f}] from {len(draws)} reps, "
          f"{100 * (draws <= 0).mean():.1f}% at or below zero (seed {BOOTSTRAP_SEED})")

    plain = fit_ols(primary, f"log_decay_speed ~ {BASE_TERMS}", robust=False)
    cooks = plain.get_influence().cooks_distance[0]
    threshold = 4 / len(primary)
    refit = fit_ols(primary[cooks <= threshold], f"log_decay_speed ~ {BASE_TERMS}")
    print(f"Cook's 4/n:        {int((cooks > threshold).sum())} flagged, "
          f"refit n = {int(refit.nobs)}, b = {refit.params[inter]:.3f}, "
          f"p = {refit.pvalues[inter]:.4f}")

    return model, inter


def shorts_checks(primary):
    "The four Shorts-share p-values quoted in the Results, in the order they are quoted."
    print("\n--- Shorts share, H3 specifications ---")
    noera = "C(platform_driven) + C(has_persistence_reason) + shorts_share"
    post2020 = primary[primary['era_ordinal'] >= 1]
    specs = (("full sample, era terms", primary, f"{BASE_TERMS} + shorts_share"),
             ("2020-26 only, era terms", post2020, f"{BASE_TERMS} + shorts_share"),
             ("2020-26 only, no era", post2020, noera),
             ("full sample, no era", primary, noera))
    for label, data, formula in specs:
        sub = data.dropna(subset=['shorts_share'])
        model = fit_ols(sub, f"log_decay_speed ~ {formula}")
        term = main_effect_term(model, 'shorts_share')
        print(f"{label:<26}n = {int(model.nobs):<5}b = {model.params[term]:>7.3f}, "
              f"SE = {model.bse[term]:.3f}, p = {model.pvalues[term]:.4f}")

    "Minimum detectable effect at 80% power, two-sided alpha = .05, from the full-sample SE."
    full = fit_ols(primary.dropna(subset=['shorts_share']),
                   f"log_decay_speed ~ {BASE_TERMS} + shorts_share")
    term = main_effect_term(full, 'shorts_share')
    se = full.bse[term]
    lo, hi = full.conf_int().loc[term]
    mde = 2.802 * se
    print(f"\n95% CI [{lo:.2f}, {hi:.2f}] -> duration ratio {np.exp(-hi):.2f} to {np.exp(-lo):.2f}")
    print(f"Minimum detectable |b| at 80% power = {mde:.2f} "
          f"-> duration ratio {np.exp(mde):.1f}x")


def window_correlations(data, label):
    "Spearman of collection window length against each YouTube covariate."
    print(f"\n--- Window circularity, {label} ---")
    for name, column in (("sample size", 'sample_size_all'),
                         ("log volume", 'log_volume'),
                         ("shorts share", 'shorts_share')):
        clean = data.dropna(subset=[column])
        rho, p = stats.spearmanr(clean['window_days'], clean[column])
        print(f"spearman(window, {name}) = {rho:.3f}, p = {p:.4f}  (n = {len(clean)})")


def h4_and_volume(model, inter, age_raw):
    age_cc = analytic(age_raw)
    age_model = fit_ols(age_cc, f"log_decay_speed ~ {BASE_TERMS}")
    age_inter = platform_term(age_model)

    print("\n--- H4 two-sample z test, complete case ---")
    b1, se1 = model.params[inter], model.bse[inter]
    b2, se2 = age_model.params[age_inter], age_model.bse[age_inter]
    z = (b2 - b1) / np.sqrt(se1 ** 2 + se2 ** 2)
    print(f"primary b = {b1:.3f} (SE {se1:.3f}), age-proxy b = {b2:.3f} (SE {se2:.3f})")
    print(f"z = {z:.3f}, p = {2 * stats.norm.sf(abs(z)):.4f}")
    print("The censored-retained comparison is in paper_extras.py.")

    print("\n--- Volume circularity, age-proxy Model B ---")
    vol = age_cc.dropna(subset=['log_volume']).copy()
    published = fit_ols(vol, f"log_decay_speed ~ {BASE_TERMS} + log_volume")
    vol_term = main_effect_term(published, 'log_volume')
    print(f"as collected:      b = {published.params[vol_term]:.3f}, "
          f"p = {published.pvalues[vol_term]:.6f}")

    "Residualise log volume on log window length (both logs) to remove the DV baked in."
    vol['log_volume_resid'] = fit_ols(vol, "log_volume ~ np.log(window_days)",
                                      robust=False).resid
    orth = fit_ols(vol, f"log_decay_speed ~ {BASE_TERMS} + log_volume_resid")
    orth_term = main_effect_term(orth, 'log_volume_resid')
    print(f"orthogonalised:    b = {orth.params[orth_term]:.3f}, "
          f"p = {orth.pvalues[orth_term]:.4f}")

    return age_cc


if __name__ == '__main__':
    parser = add_common_args(argparse.ArgumentParser(
        description="Reproduce the robustness figures quoted in the prose."))
    args = parser.parse_args()

    primary = analytic(load_raw(args.trends, args.strat, args.youtube))
    model, inter = primary_checks(primary)
    shorts_checks(primary)
    age_cc = h4_and_volume(model, inter,
                           load_raw(args.trends_age, args.strat_age, args.youtube_age))
    window_correlations(primary, "primary, complete case")
    window_correlations(age_cc, "age proxy, complete case")
