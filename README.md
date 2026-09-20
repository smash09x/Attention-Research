# Attention: The Most Valuable Resource of the Twenty-First Century

Data and code for a study of how fast public attention decays, and whether that
speed changed across three eras: 2015–2019, 2020–2022 and 2023–2026.

Author: Hassan Tariq Malik · Pioneer Academics "Data Detectives"
Research mentor: Prof. Shafik Islam, Tufts University

## Quick start

```bash
pip install -r Code/requirements.txt
cd "Code/Regression Model"
python run_all.py
```

`run_all.py` runs every analysis script in order on the CSVs in `Data/CSVs/` and
saves each script's printed output, plus `figure1.png`, to `Code/Regression Model/results/`.
Pass `--data <folder>` if the CSVs are somewhere else.

## What is measured

Search interest for each keyword comes from Google Trends. A keyword's **peak** is
the interval over which its interest stays above 20 percent of its own local
maximum.

- **Outcome:** `log_decay_speed = log(1 / peak_width_days)`. Larger values mean
  faster decay. A coefficient `b` converts to a duration ratio as `exp(-b)`.
- **Second outcome:** `post_peak_width_days = peak_width_days - time_to_peak_days`,
  which measures the falling side only.
- **Predictors:** each keyword is coded on two dummies, `platform_driven` and
  `has_persistence_reason`, and each row belongs to one of three eras
  (`era_ordinal` 0, 1, 2).
- **YouTube covariates:** `log_volume` (log of summed lifetime views) and
  `shorts_share` (Shorts views divided by all views), collected inside each peak
  window plus three days on either side.

Two keyword sets are used:

| Set | Keywords | Keyword-era rows | Rows used by OLS | Right-censored rows |
|---|---|---|---|---|
| Primary | 108 | 116 | 112 | 4 |
| Age proxy (adolescent slang) | 114 | 198 | 108 | 90 |

In the primary set the two dummies are a one-to-one recoding of four categories
(control 0/0, movie 0/1, platform event 1/0, viral 1/1). In the age-proxy set they
vary independently, but are strongly correlated (r = −0.712).

## Repository layout

```
Data/
  CSVs/                    the six input files (see Data dictionary)
  Overall_Results.xlsx     the same data as one workbook
Code/
  requirements.txt         pinned dependencies
  Data Collection/
    trends.py              two-stage Google Trends collection (pytrends)
    youtube_videos.py      YouTube Data API v3 volume and Shorts share
  Regression Model/
    run_all.py             runs everything below, writes results/
    data_utils_final.py    loading, merging and variable construction
    table_utils.py         shared helpers (not run directly)
    model_main.py          Tables A1-A4
    regression_tables.py   Table 1, A5, A6, A7
    survival_models.py     Table A8
    robustness_checks.py   robustness figures quoted in the prose
    paper_extras.py        primary Cox, censored H4 test, flag correlations
    figures.py             Figure 1
```

## Reproducing each output

Every script takes the same path flags: `--trends`, `--strat`, `--youtube` for the
primary set and `--trends-age`, `--strat-age`, `--youtube-age` for the age-proxy
set. The defaults are file names in the current folder, so either copy the CSVs
next to the scripts or pass the paths. `run_all.py` does this for you.

| Output | Command |
|---|---|
| Tables A1, A2 (primary) | `python model_main.py A`, then `B`, then `C` |
| Tables A3, A4 (age proxy) | same, passing `--trends`, `--strat`, `--youtube` the `_age` files |
| Table 1, A5, A6, A7 | `python regression_tables.py all` (or one of `1`, `A5`, `A6`, `A7`) |
| Table A8 | `python survival_models.py` |
| Robustness figures in the prose | `python robustness_checks.py` |
| Primary Cox model, censored H4 z test, flag correlations | `python paper_extras.py` |
| Figure 1 | `python figures.py --out figure1.png` |

`model_main.py` takes a positional `A`, `B` or `C`. A is the base model, B adds
`log_volume`, C adds `shorts_share`. All models use OLS with HC3
heteroskedasticity-consistent standard errors, and both outcomes are run for each.

## Expected headline output

Running `run_all.py` on the CSVs in this repository should give these values.

| Result | Value |
|---|---|
| Model A, era × platform-driven (full peak width) | b = 0.742, SE = 0.254, p = .003, n = 112 |
| Model A, era × platform-driven (post-peak width) | b = 0.823, SE = 0.287, p = .004, n = 112 |
| Has persistence reason (Model A) | b = −1.279, p < .001 |
| Table 1, platform-driven duration ratio | 1.61 (2015–19), 0.77 (2020–22), 0.37 (2023–26) |
| Cluster-robust SE / p | SE = 0.237, p = .002 |
| Cluster bootstrap 95% CI (seed 20260917, 5000 reps) | [0.265, 1.245] |
| Cook's distance refit (4/n) | 7 flagged, n = 105, b = 0.956 |
| Nested F test, ordinal vs factor era | F(3, 103) = 1.91, p = .132 |
| H4 z test, complete case | z = −2.646, p = .008 |
| H4 z test, censored rows retained | z = −1.569, p = .117 |
| Age-proxy Cox, era × platform | b = 0.084, HR = 1.088, p = .823 |
| Age-proxy Cox, platform only | HR = 3.343 |

## Data dictionary

**`keywords_stratified.csv`, `keywords_stratified_age.csv`** (one row per keyword)

| Column | Meaning |
|---|---|
| `keyword` | Search term |
| `category` | `control`, `movie`, `viral`, `platform_event` (primary) or `age_proxy` |
| `platform_driven` | 1 if attention is driven by platform or algorithmic spread |
| `has_persistence_reason` | 1 if there is an intrinsic reason to recur (annual event, franchise, sequel) |
| `is_age_proxy`, `is_control`, `is_movie`, `is_viral`, `is_platform_event` | One-hot category flags |

**`trends.csv`, `trends_age.csv`** (one row per keyword-era)

| Column | Meaning |
|---|---|
| `era` | `baseline` (2015–2019), `reels_covid` (2020–2022), `ai_gen` (2023–2026) |
| `geo` | Region of the query (`global` for all rows used in the paper) |
| `rough_peak_date`, `rel_to_anchor` | Stage-1 candidate peak date and its height relative to the keyword's overall maximum (kept only if above 0.50) |
| `data_mode` | Plain-text query or Google Topic entity (assigned automatically) |
| `event_start_date`, `event_end_date` | Measured peak window; the YouTube window is this plus three days each side |
| `peak_value` | Always 100 (Trends normalises the stage-2 window to its own maximum) |
| `peak_width_days`, `time_to_peak_days`, `post_peak_width_days` | Full width above the 20 percent threshold, rise time, and fall time |
| `tight_pad_days` | Padding used in the stage-2 query: 45, 90, 180 or 360 days |
| `never_decays_in_scope` | 1 if the series never fell back below the threshold inside the era window (censored) |
| `width_truncated` | Identical to `never_decays_in_scope` (same code path); only the latter is used |
| `possible_weekly_resolution` | 1 if the stage-2 window was long enough for Trends to return weekly data |

**`youtube_volume.csv`, `youtube_volume_age.csv`** (one row per keyword-era)

Per-bucket columns for `all`, `shorts` and `videos` (Shorts are 183 seconds or
shorter): `sample_size_*`, `above_threshold_*`, `below_threshold_*`,
`pct_above_threshold_*`, `median_views_*`, `mean_views_*`, `min_views_seen_*`,
`max_views_seen_*` and `total_views_*`. `error` is empty for successful rows.
The search returns at most 500 videos per keyword-era, ranked by relevance.

## Collecting new data

The two collectors feed each other. Google Trends comes first because YouTube
uses its measured windows.

```bash
python trends.py --file keywords_stratified.csv          # writes results_keywords_stratified_global.csv
python youtube_videos.py --file results_keywords_stratified_global.csv --api-key YOUR_KEY
```

`trends.py` accepts `--retry-failed <previous results>` to re-run only failed
keywords and `--cache` to resume a partial run. `youtube_videos.py` saves after
every keyword, so an interrupted run resumes where it stopped. The YouTube API
allows 10,000 units per key per day, and one 500-video keyword-era costs close
to 1,000.

The CSVs in `Data/CSVs/` are the audited final files. A fresh run will not
reproduce them exactly: Google Trends is resampled on every query, and the data
were collected between September 1 and 17, 2026. Every row of both datasets was
checked by hand against the real event date before modelling.

## Notes

- **Volume is circular by construction.** The YouTube window is the peak window
  plus six days, so a longer peak collects more videos regardless of any real
  relationship. `robustness_checks.py` shows this directly: the age-proxy volume
  coefficient falls from −0.210 to 0.013 once log volume is residualised on log
  window length, and window length correlates with volume (ρ = 0.329, primary).
  The Shorts share is a ratio and is unaffected (ρ = −0.023).
- **Censoring.** A row is censored when the series never falls back below 20
  percent inside its era window. OLS drops these rows. `survival_models.py` and
  `paper_extras.py` keep them as lower bounds.
- **Survival-model errors.** lifelines accelerated failure time models have no
  cluster option, so `survival_models.py` computes keyword-clustered sandwich
  errors itself (with a G/(G−1) correction). AIC is reported on the log-time
  scale, that is lifelines' AIC minus 2 × the sum of log durations over events.
- **Limitations.** Keyword sets were built by hand during the study, with no
  preregistration and no validated slang corpus. Each observation is a single
  pull, so there is no test-retest reliability. The paper discusses these in
  full.

## License

Code released for review and replication of the accompanying paper.
