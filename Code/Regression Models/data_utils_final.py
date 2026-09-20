"Loaders and variable builders for the OLS models. Imported by model_main.py and table_utils.py."

import numpy as np
import pandas as pd

ERA_MAP = {'baseline': 0, 'reels_covid': 1, 'ai_gen': 2}

"DVs: full peak width, and post-peak width only."
DECAY_OUTCOMES = ('log_decay_speed', 'log_decay_speed_post')


def load_sheets(trends_path, strat_path, youtube_path=None):
    "YouTube CSV is optional (models B/C only)."
    trends = pd.read_csv(trends_path, encoding='utf-8-sig')
    strat = pd.read_csv(strat_path, encoding='utf-8-sig')
    youtube = pd.read_csv(youtube_path, encoding='utf-8-sig') if youtube_path else None
    return trends, youtube, strat


def build_main(trends, strat, youtube=None, extra_vars=()):
    df = trends.merge(
        strat[['keyword', 'platform_driven', 'has_persistence_reason']],
        on='keyword', how='left'
    )
    df = df[df['platform_driven'].notna() & df['has_persistence_reason'].notna()]
    "Drop censored rows: never decayed inside the window."
    df = df[~df['never_decays_in_scope'].astype(bool)]
    df = df[df['peak_width_days'] > 0]

    unknown = set(df['era'].unique()) - set(ERA_MAP)
    if unknown:
        raise ValueError(f"Unrecognized era label(s): {unknown}")
    df['era_ordinal'] = df['era'].map(ERA_MAP)
    df['decay_speed'] = 1 / df['peak_width_days']
    df['log_decay_speed'] = np.log(df['decay_speed'])

    "post_peak_width_days is not in every trends file. Compute only if present."
    if 'post_peak_width_days' in df.columns:
        valid_post = df['post_peak_width_days'] > 0
        df['decay_speed_post'] = np.where(valid_post, 1 / df['post_peak_width_days'], np.nan)
        df['log_decay_speed_post'] = np.log(df['decay_speed_post'])

    if extra_vars:
        if youtube is None:
            raise ValueError(f"extra_vars={list(extra_vars)} requested but no youtube data was loaded")
        yt = youtube[['keyword', 'era', 'total_views_all', 'total_views_shorts']].copy()
        "Inner merge: keep only rows with YouTube data."
        df = df.merge(yt, on=['keyword', 'era'], how='inner')

        if 'log_volume' in extra_vars:
            df['log_volume'] = np.log(df['total_views_all'].replace(0, np.nan))
        if 'shorts_share' in extra_vars:
            df['shorts_share'] = df['total_views_shorts'] / df['total_views_all']

        df = df.dropna(subset=list(extra_vars))

    return df


def cell_counts(df, label=None):
    header = "=== Cell counts (platform_driven x has_persistence_reason) ==="
    if label:
        header = f"=== Cell counts for {label} (platform_driven x has_persistence_reason) ==="
    print(f"\n{header}")
    print(df.groupby(['platform_driven', 'has_persistence_reason']).size())
    print(f"Total n: {len(df)}")
