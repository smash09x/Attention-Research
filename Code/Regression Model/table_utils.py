"""
table_utils.py

Shared loading and helper functions for the table and robustness scripts.
Not run directly. Sits next to data_utils_final.py in Code/Regression Model/.
"""

import numpy as np
import statsmodels.formula.api as smf

from data_utils_final import load_sheets, ERA_MAP

BASE_TERMS = ("era_ordinal + C(platform_driven) + C(has_persistence_reason) "
              "+ era_ordinal:C(platform_driven) + era_ordinal:C(has_persistence_reason)")

FACTOR_TERMS = ("C(era_ordinal) + C(platform_driven) + C(has_persistence_reason) "
                "+ C(era_ordinal):C(platform_driven) + C(era_ordinal):C(has_persistence_reason)")

ERA_LABELS = {0: "2015-19", 1: "2020-22", 2: "2023-26"}

"Seed stated in the paper's Robustness paragraph. Do not change it."
BOOTSTRAP_SEED = 20260917
BOOTSTRAP_REPS = 5000

"The YouTube collection window is the measured peak window plus three days either side."
WINDOW_PAD_DAYS = 6


def load_raw(trends_path, strat_path, youtube_path=None):
    """
    Same merge as build_main, but censored rows are KEPT and flagged.
    Table A7 and the survival models need the rows that build_main drops.
    """
    trends, youtube, strat = load_sheets(trends_path, strat_path, youtube_path)

    df = trends.merge(
        strat[['keyword', 'platform_driven', 'has_persistence_reason']],
        on='keyword', how='left'
    )
    df = df[df['platform_driven'].notna() & df['has_persistence_reason'].notna()]
    df = df[df['peak_width_days'] > 0]

    unknown = set(df['era'].unique()) - set(ERA_MAP)
    if unknown:
        raise ValueError(f"Unrecognized era label(s): {unknown}")

    df['era_ordinal'] = df['era'].map(ERA_MAP)
    df['censored'] = df['never_decays_in_scope'].astype(bool)
    df['event'] = (~df['censored']).astype(int)
    df['log_decay_speed'] = np.log(1 / df['peak_width_days'])

    if 'post_peak_width_days' in df.columns:
        valid_post = df['post_peak_width_days'] > 0
        df['log_decay_speed_post'] = np.log(
            np.where(valid_post, 1 / df['post_peak_width_days'], np.nan))

    if youtube is not None:
        yt = youtube[['keyword', 'era', 'total_views_all', 'total_views_shorts']].copy()
        df = df.merge(yt, on=['keyword', 'era'], how='left')
        df['log_volume'] = np.log(df['total_views_all'].replace(0, np.nan))
        df['shorts_share'] = df['total_views_shorts'] / df['total_views_all']

    "Collection window length is a deterministic function of the DV. Used by the circularity check."
    df['window_days'] = df['peak_width_days'] + WINDOW_PAD_DAYS

    return df


def analytic(df):
    """Complete-case sample: the rows the OLS tables in the paper actually use."""
    return df[~df['censored']].copy()


def add_category(df):
    """
    In the primary set the two flags are a one-to-one recoding of the four
    categories, so the label is derived rather than read from a column.
    Exact by construction, and cannot drift from the stratification file.
    """
    labels = {(0, 0): 'control', (0, 1): 'movie',
              (1, 0): 'platform_event', (1, 1): 'viral'}
    df = df.copy()
    df['category'] = [labels[(int(p), int(r))]
                      for p, r in zip(df['platform_driven'], df['has_persistence_reason'])]
    return df


def fit_ols(df, formula, robust=True):
    if robust:
        return smf.ols(formula, data=df).fit(cov_type='HC3')
    return smf.ols(formula, data=df).fit()


def platform_term(model):
    """Name of the era x platform-driven interaction term in a patsy-built model."""
    for name in model.params.index:
        if 'era_ordinal' in name and ':' in name and 'platform_driven' in name:
            return name
    raise KeyError("era x platform-driven interaction term not found")


def main_effect_term(model, var):
    for name in model.params.index:
        if var in name and ':' not in name:
            return name
    raise KeyError(f"main effect for {var} not found")


def add_common_args(parser):
    parser.add_argument('--trends', default='trends.csv')
    parser.add_argument('--strat', default='keywords_stratified.csv')
    parser.add_argument('--youtube', default='youtube_volume.csv')
    parser.add_argument('--trends-age', default='trends_age.csv')
    parser.add_argument('--strat-age', default='keywords_stratified_age.csv')
    parser.add_argument('--youtube-age', default='youtube_volume_age.csv')
    return parser
