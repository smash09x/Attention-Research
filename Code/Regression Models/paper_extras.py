"Paper figures no other script prints: primary Cox, censored H4 z test, flag correlations. Needs lifelines. Run: python paper_extras.py"

import argparse

import numpy as np
from scipy import stats

from table_utils import add_common_args, load_raw

COX_FORMULA = ("era_ordinal + platform_driven + has_persistence_reason "
               "+ era_ordinal:platform_driven + era_ordinal:has_persistence_reason")


def cox_interaction(raw):
    "Cox era x platform term, keyword-clustered SEs, censored rows kept."
    try:
        from lifelines import CoxPHFitter
    except ImportError:
        raise SystemExit("paper_extras.py needs lifelines: pip install lifelines")

    df = raw[['keyword', 'peak_width_days', 'event', 'era_ordinal',
              'platform_driven', 'has_persistence_reason']].dropna().copy()
    df['platform_driven'] = df['platform_driven'].astype(int)
    df['has_persistence_reason'] = df['has_persistence_reason'].astype(int)
    fit = CoxPHFitter().fit(df, duration_col='peak_width_days', event_col='event',
                            cluster_col='keyword', formula=COX_FORMULA)
    row = fit.summary.loc['era_ordinal:platform_driven']
    return row, len(df), int(df['event'].sum())


def main(args):
    primary_raw = load_raw(args.trends, args.strat, args.youtube)
    age_raw = load_raw(args.trends_age, args.strat_age, args.youtube_age)

    print("\n--- Cox era x platform, all rows kept, keyword-clustered ---")
    rows = {}
    for label, raw in (("primary", primary_raw), ("age proxy", age_raw)):
        row, n, events = cox_interaction(raw)
        rows[label] = row
        print(f"{label:<10}n = {n}, events = {events}, b = {row['coef']:.3f}, "
              f"SE = {row['se(coef)']:.3f}, HR = {row['exp(coef)']:.3f}, p = {row['p']:.4f}")

    print("\n--- H4 two-sample z test, censored rows retained ---")
    b1, se1 = rows['primary']['coef'], rows['primary']['se(coef)']
    b2, se2 = rows['age proxy']['coef'], rows['age proxy']['se(coef)']
    z = (b2 - b1) / np.sqrt(se1 ** 2 + se2 ** 2)
    print(f"z = {z:.3f}, p = {2 * stats.norm.sf(abs(z)):.4f}")

    print("\n--- Correlation of platform_driven and has_persistence_reason, all rows ---")
    for label, raw in (("primary", primary_raw), ("age proxy", age_raw)):
        r = np.corrcoef(raw['platform_driven'].astype(float),
                        raw['has_persistence_reason'].astype(float))[0, 1]
        print(f"{label:<10}n = {len(raw)}, r = {r:.3f}")


if __name__ == '__main__':
    parser = add_common_args(argparse.ArgumentParser(
        description="Primary Cox, censored H4 z test, flag correlations."))
    main(parser.parse_args())
