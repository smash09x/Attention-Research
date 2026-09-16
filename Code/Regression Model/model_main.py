import argparse

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices
from scipy import stats

from data_utils_final import load_sheets, build_main, cell_counts, DECAY_OUTCOMES

BASE_TERMS = ("era_ordinal + C(platform_driven) + C(has_persistence_reason) "
              "+ era_ordinal:C(platform_driven) + era_ordinal:C(has_persistence_reason)")

# Each spec adds at most one YouTube predictor. 'predictors' is the RHS only;
# the DV is filled in per outcome from DECAY_OUTCOMES.
MODEL_SPECS = {
    'A': {'predictors': BASE_TERMS,
          'extra_vars': (), 'needs_youtube': False},
    'B': {'predictors': f"{BASE_TERMS} + log_volume",
          'extra_vars': ('log_volume',), 'needs_youtube': True},
    'C': {'predictors': f"{BASE_TERMS} + shorts_share",
          'extra_vars': ('shorts_share',), 'needs_youtube': True},
}


def run_ols(df, formula):
    model = smf.ols(formula, data=df).fit(cov_type='HC3')
    print(model.summary())

    y, X = dmatrices(formula, data=df, return_type='dataframe')
    vif = pd.DataFrame({'term': X.columns,
                         'VIF': [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]})
    print("\n--- VIF ---")
    print(vif)

    stat, p = stats.shapiro(model.resid)
    print(f"\nShapiro-Wilk residual normality: stat={stat:.4f}, p={p:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run model A (trends only), B (trends + log_volume), "
                     "or C (trends + shorts_share).")
    parser.add_argument('model', choices=['A', 'B', 'C'])
    parser.add_argument('--trends', default='trends.csv')
    parser.add_argument('--strat', default='keywords_stratified.csv')
    parser.add_argument('--youtube', default='youtube_volume.csv',
                         help="Only read for models B/C; ignored for model A.")
    args = parser.parse_args()

    spec = MODEL_SPECS[args.model]
    youtube_path = args.youtube if spec['needs_youtube'] else None
    trends, youtube, strat = load_sheets(args.trends, args.strat, youtube_path)

    df = build_main(trends, strat, youtube, spec['extra_vars'])

    # Run twice: peak_width_days (rise+decay), then post_peak_width_days (decay only)
    for outcome in DECAY_OUTCOMES:
        print(f"\n{'#' * 80}\n# Outcome: {outcome}\n{'#' * 80}")

        if outcome not in df.columns:
            print(f"[skipped] '{outcome}' is not available in this dataset "
                  f"(post_peak_width_days wasn't in the input trends file).")
            continue

        df_outcome = df.dropna(subset=[outcome])
        if df_outcome.empty:
            print(f"[skipped] '{outcome}' has no valid (non-NaN) rows in this dataset.")
            continue

        cell_counts(df_outcome, label=outcome)
        run_ols(df_outcome, f"{outcome} ~ {spec['predictors']}")
