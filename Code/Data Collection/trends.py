"Build a Google Trends attention pipeline that finds qualifying peaks across predefined eras and then measures each peak more precisely inside its own era window. Results are returned as rows containing peak timing, width, resolution, and decay information."

import argparse
import json
import os
import time
import random
import warnings
from datetime import timedelta
import numpy as np
import pandas as pd
from pytrends.request import TrendReq

"pytrends' own request.py calls df.fillna(False) internally on an object-dtype"
"frame -- that's their bug, not ours, and it's unfixed upstream. Silencing it"
"here since there's nothing in this file to change to avoid it."
warnings.filterwarnings(
    "ignore",
    message=".*Downcasting object dtype arrays on .fillna.*",
    category=FutureWarning,
)


ERA_WINDOWS = {
    "baseline":    ("2015-01-01", "2019-12-31"),
    "reels_covid": ("2020-01-01", "2022-12-31"),
    "ai_gen":      ("2023-01-01", "2026-08-30"),
}
WIDE_START, WIDE_END = "2015-01-01", "2026-08-30"

TIGHT_PAD_DAYS = 45
THRESHOLD_FRAC = 0.2
ANCHOR_REL_THRESH = 0.5
DAILY_RESOLUTION_LIMIT_DAYS = 270
"~9mo -- Trends auto-drops to weekly past this span"
MIN_DELAY_SEC, MAX_DELAY_SEC = 8, 15
MAX_RETRIES = 4
GEO_CODES = {"global": "", "india_banned": "IN", "indonesia_nonbanned": "ID", "pakistan_mixed": "PK"}
ERA_BOUNDS = {
    era: (pd.Timestamp(start), pd.Timestamp(end))
    for era, (start, end) in ERA_WINDOWS.items()
}


def find_topic_mid(pytrends, keyword, prefer_types=("Film", "Movie"), cache=None):
    "Ask Google Trends for suggested entities related to a keyword and prefer matching entity types such as Film or Movie. Cache the result when a cache dictionary is supplied."
    if cache is not None and keyword in cache:
        return cache[keyword]
    result = None
    for attempt in range(2):
        try:
            suggestions = pytrends.suggestions(keyword=keyword)
            time.sleep(random.uniform(1, 2))
            for s in suggestions:
                if any(t.lower() == s.get('type', '').lower() for t in prefer_types):
                    result = (s['mid'], s['title'], s['type'])
                    break
            break
        except Exception as e:
            print(f"    entity lookup for '{keyword}' failed ({e}); retrying...")
            time.sleep(2 + attempt * 2)
    if cache is not None:
        cache[keyword] = result
    return result


def load_entity_cache(path="entity_cache.json"):
    "Load the saved keyword-to-entity mapping from JSON. Return an empty dictionary when the cache file does not yet exist."
    try:
        with open(path) as f:
            raw = json.load(f)
        return {k: (tuple(v) if v else None) for k, v in raw.items()}
    except FileNotFoundError:
        return {}


def save_entity_cache(cache, path="entity_cache.json"):
    "Persist the current entity cache as readable JSON so later runs can reuse successful entity matches."
    with open(path, "w") as f:
        json.dump(cache, f, indent=2)


def fetch_raw(pytrends, query_term, timeframe, geo=""):
    "Fetch one Google Trends time series for a single query term and time range, remove the partial-data marker when present, and return the first data column."
    pytrends.build_payload([query_term], timeframe=timeframe, geo=geo)
    df = pytrends.interest_over_time()
    if df.empty:
        return None
    df = df.drop(columns=['isPartial'], errors='ignore')
    return df.iloc[:, 0]


def safe_fetch_raw(pytrends, query_term, timeframe, geo=""):
    "Wrap a single Google Trends fetch with retries, backoff, and a randomized delay after successful requests to reduce the impact of temporary empty responses or rate limits."
    for attempt in range(MAX_RETRIES):
        try:
            series = fetch_raw(pytrends, query_term, timeframe, geo)
            if series is not None and len(series) > 0:
                time.sleep(random.uniform(MIN_DELAY_SEC, MAX_DELAY_SEC))
                return series
            wait = (2 ** attempt) * 2 + random.uniform(0, 1)
            print(f"    empty response (attempt {attempt + 1}); "
                  f"likely soft rate-limit -- retrying in {wait:.1f}s")
            time.sleep(wait)
        except Exception as e:
            wait = (2 ** attempt) * 2 + random.uniform(0, 1)
            print(f"    attempt {attempt + 1} failed ({e}); retrying in {wait:.1f}s")
            time.sleep(wait)
    return None


def fetch_raw_multi(pytrends, query_terms, timeframe, geo=""):
    "Fetch a Google Trends time series for several query terms at once and return the resulting DataFrame after removing the partial-data marker."
    pytrends.build_payload(query_terms, timeframe=timeframe, geo=geo)
    df = pytrends.interest_over_time()
    if df.empty:
        return None
    return df.drop(columns=['isPartial'], errors='ignore')


def safe_fetch_raw_multi(pytrends, query_terms, timeframe, geo=""):
    "Retry a multi-term Google Trends request with backoff and randomized delay until a non-empty DataFrame is obtained or all retries are exhausted."
    for attempt in range(MAX_RETRIES):
        try:
            df = fetch_raw_multi(pytrends, query_terms, timeframe, geo)
            if df is not None and not df.empty:
                time.sleep(random.uniform(MIN_DELAY_SEC, MAX_DELAY_SEC))
                return df
            wait = (2 ** attempt) * 2 + random.uniform(0, 1)
            print(f"    empty batch response (attempt {attempt + 1}); retrying in {wait:.1f}s")
            time.sleep(wait)
        except Exception as e:
            wait = (2 ** attempt) * 2 + random.uniform(0, 1)
            print(f"    batch attempt {attempt + 1} failed ({e}); retrying in {wait:.1f}s")
            time.sleep(wait)
    return None


def find_era_peaks(wide_series, era_bounds, anchor_rel_thresh=ANCHOR_REL_THRESH):

    "Find the strongest point inside each predefined era and keep that era as a candidate only when its local peak reaches the required fraction of the series-wide anchor peak."
    values = wide_series.values.astype(float)
    dates = wide_series.index

    if len(values) == 0:
        return []

    anchor_idx = int(np.argmax(values))
    anchor_val = values[anchor_idx]
    if anchor_val <= 0:
        return []

    candidates = []
    for era, (era_start, era_end) in era_bounds.items():
        mask = (dates >= era_start) & (dates <= era_end)
        if not mask.any():
            continue
        era_values = values[mask]
        era_dates = dates[mask]
        local_idx = int(np.argmax(era_values))
        local_val = era_values[local_idx]
        rel = local_val / anchor_val
        if rel >= anchor_rel_thresh:
            candidates.append({
                "date": era_dates[local_idx],
                "wide_value": float(local_val),
                "rel_to_anchor": float(rel),
                "era": era,
            })

    candidates.sort(key=lambda c: c["date"])
    return candidates


def peak_metrics(values, dates, threshold_frac=THRESHOLD_FRAC):
    "Measure a peak using a fraction-of-peak threshold. The first point below the threshold before the peak defines the rise boundary, and the first point below it after the peak defines the fall boundary."
    peak_idx = int(np.argmax(values))
    peak_val = values[peak_idx]
    threshold = threshold_frac * peak_val

    rise_start_idx = 0
    for i in range(peak_idx, -1, -1):
        if values[i] < threshold:
            rise_start_idx = i
            break

    fall_end_idx = len(values) - 1
    for i in range(peak_idx, len(values)):
        if values[i] < threshold:
            fall_end_idx = i
            break

    return {
        "peak_value": peak_val,
        "peak_width_days": dates[fall_end_idx] - dates[rise_start_idx],
        "time_to_peak_days": dates[peak_idx] - dates[rise_start_idx],
        "_rise_start_idx": rise_start_idx,
        "_fall_end_idx": fall_end_idx,
    }


def fetch_with_fallback(pytrends, keyword, mid_info, timeframe, geo=""):
    "Try an identified Google Trends topic entity first when available, then fall back to the literal keyword if the entity produces no usable series."
    if mid_info:
        mid, matched_title, matched_type = mid_info
        print(f"    trying TOPIC entity '{matched_title}' (type={matched_type}, mid={mid})...")
        series = safe_fetch_raw(pytrends, mid, timeframe, geo)
        if series is not None and not series.empty:
            return series, f"TOPIC entity ('{matched_title}', type={matched_type})"
        print(f"    entity mid returned no data after retries -- falling back to plain text '{keyword}'")

    print(f"    trying plain text '{keyword}'...")
    series = safe_fetch_raw(pytrends, keyword, timeframe, geo)
    if series is not None and not series.empty:
        mode = "plain text (fallback after empty entity match)" if mid_info else "plain text (no entity match found)"
        return series, mode

    return None, "no data from EITHER entity match OR plain text"


def stage1_batch_wide(pytrends, batch, geo=""):
    "Run the broad first-stage Trends pull for a small batch of topics and use an individual fallback fetch for any topic whose batched column is missing or empty."
    query_terms = [(mid_info[0] if mid_info else keyword) for keyword, mid_info in batch]
    wide_timeframe = f"{WIDE_START} {WIDE_END}"
    print(f"  Stage 1 (BATCHED, {len(batch)} topics): pulling full range {wide_timeframe}...")
    df = safe_fetch_raw_multi(pytrends, query_terms, wide_timeframe, geo)

    results = {}
    for (keyword, mid_info), term in zip(batch, query_terms):
        col = df[term] if (df is not None and term in df.columns) else None
        if col is not None and col.sum() > 0:
            mode = f"TOPIC entity (batched, type={mid_info[2]})" if mid_info else "plain text (batched)"
            results[keyword] = (col, mode)
        else:
            print(f"    '{keyword}' came back empty inside the batch -- retrying individually...")
            series, mode = fetch_with_fallback(pytrends, keyword, mid_info, wide_timeframe, geo)
            results[keyword] = (series, mode)

    return results


def locate_and_measure_stage2(pytrends, keyword, mid_info, mode, peak_date, era_start, era_end, geo=""):
    "Re-fetch a candidate peak in a tighter window, measure its threshold-based width and time to peak, and expand the window when an edge is hit and the relevant era still has room to grow."
    stage2_mid_info = mid_info if mid_info and "entity" in mode else None
    today_ts = pd.Timestamp.now().normalize()
    "real \"today\" -- can't pull future data"
    floor_ts = era_start
    "NEVER look before this era's own window"
    ceiling_ts = min(era_end, today_ts)
    "NEVER look past this era's own window (or past real today)"
    pad_days = TIGHT_PAD_DAYS
    last_result = None

    while True:
        pad_td = timedelta(days=pad_days)
        tight_start_ts = max(peak_date - pad_td, floor_ts)
        tight_end_ts = min(peak_date + pad_td, ceiling_ts)
        tight_start = tight_start_ts.strftime('%Y-%m-%d')
        tight_end = tight_end_ts.strftime('%Y-%m-%d')
        tight_timeframe = f"{tight_start} {tight_end}"
        window_span_days = (tight_end_ts - tight_start_ts).days
        possible_weekly = window_span_days > DAILY_RESOLUTION_LIMIT_DAYS
        if possible_weekly:
            print(f"    NOTE: window span {window_span_days}d exceeds "
                  f"{DAILY_RESOLUTION_LIMIT_DAYS}d -- Trends may silently return "
                  f"weekly (not daily) resolution for this pull")

        print(f"    Stage 2 (tight, pad={pad_days}d): re-pulling {tight_timeframe} for daily resolution...")
        tight_series, mode2 = fetch_with_fallback(pytrends, keyword, stage2_mid_info, tight_timeframe, geo)
        if tight_series is None:
            return {"error": "no data in tight pull (tried entity AND plain text)"}

        values = tight_series.values.astype(float)
        dates = tight_series.index.to_pydatetime()
        day_offsets = np.array([(d - dates[0]).days for d in dates])
        metrics = peak_metrics(values, day_offsets, THRESHOLD_FRAC)

        rise_start_idx = metrics["_rise_start_idx"]
        fall_end_idx = metrics["_fall_end_idx"]
        hit_rise_edge = rise_start_idx == 0
        hit_fall_edge = fall_end_idx == len(values) - 1
        rise_start_date = dates[rise_start_idx].date().isoformat()
        fall_end_date = dates[fall_end_idx].date().isoformat()

        "Is there still real room to expand on the side(s) that hit an"
        "edge, or are we already pinned against this era's own boundary"
        "(or today)? Doubling pad further can't move a side that's"
        "already clamped -- so if a clamped side still hasn't found a"
        "real crossing, more doubling is pointless, not just expensive."
        can_grow_rise = tight_start_ts > floor_ts
        can_grow_fall = tight_end_ts < ceiling_ts
        stuck_rise = hit_rise_edge and not can_grow_rise
        stuck_fall = hit_fall_edge and not can_grow_fall

        never_decays = stuck_rise or stuck_fall
        note = ""
        if never_decays:
            if stuck_fall and ceiling_ts == today_ts:
                note = (f"never drops below {int(THRESHOLD_FRAC*100)}% threshold as of today "
                        f"({today_ts.date()}) -- still ongoing / hasn't decayed yet")
            else:
                note = (f"never drops below {int(THRESHOLD_FRAC*100)}% threshold within this era's "
                        f"own window ({era_start.date()} to {era_end.date()}) -- true start/end may "
                        f"lie outside this era by design (each era is measured in isolation)")

        last_result = {
            "data_mode": mode2, "event_start_date": rise_start_date,
            "event_end_date": fall_end_date,
            "peak_value": metrics["peak_value"],
            "peak_width_days": metrics["peak_width_days"],
            "time_to_peak_days": metrics["time_to_peak_days"],
            "tight_pad_days": pad_days,
            "width_truncated": int(hit_rise_edge or hit_fall_edge),
            "possible_weekly_resolution": int(possible_weekly),
            "never_decays_in_scope": int(never_decays),
            "note": note,
        }

        if never_decays:
            print(f"      STOPPED at era boundary -- {note}")
            break

        if not (hit_rise_edge or hit_fall_edge):
            break
            "true width found strictly inside the window -- done"

        print(f"      window edge hit (rise={hit_rise_edge}, fall={hit_fall_edge}), room to grow "
              f"(rise={can_grow_rise}, fall={can_grow_fall}) -- doubling pad {pad_days}d -> {pad_days * 2}d")
        pad_days *= 2

    return last_result


def run_pipeline(csv_path, geo_key="global", cache_path=None):
    "Run the complete Trends pipeline: update the current era boundary, load topics and cached work, resolve entities, process topics in batches, find era-specific peaks, and return all result rows as a DataFrame."
    global WIDE_END
    "ai_gen is the \"current, ongoing\" era by design -- its upper bound"
    "should track real today, not the stale hardcoded date it started as."
    "baseline/reels_covid are closed historical eras and keep fixed bounds."
    today_str = pd.Timestamp.now().strftime('%Y-%m-%d')
    WIDE_END = today_str
    era_bounds = dict(ERA_BOUNDS)
    era_bounds["ai_gen"] = (ERA_BOUNDS["ai_gen"][0], pd.Timestamp(today_str))

    pytrends = TrendReq(hl='en-US', tz=0, timeout=(10, 30))
    topics = pd.read_csv(csv_path)
    geo = GEO_CODES.get(geo_key, "")

    already_done = set()
    if cache_path:
        try:
            prev = pd.read_csv(cache_path)
            if "error" in prev.columns:
                already_done = set(prev[prev["error"].isna()]["keyword"])
                print(f"Skipping {len(already_done)} topics already successful in {cache_path}")
        except FileNotFoundError:
            pass

    topics = topics[~topics["keyword"].isin(already_done)]
    if topics.empty:
        print("Nothing left to fetch -- all topics already cached.")
        return pd.DataFrame()

    entity_cache = load_entity_cache()
    entries = []
    for kw in topics["keyword"]:
        entries.append((kw, find_topic_mid(pytrends, kw, cache=entity_cache)))
        save_entity_cache(entity_cache)
    cat_by_kw = dict(zip(topics["keyword"], topics["category"]))

    rows = []
    for i in range(0, len(entries), 5):
        chunk = entries[i:i + 5]
        print(f"\n=== Batch of {len(chunk)}: {[k for k, _ in chunk]} ===")
        stage1_results = stage1_batch_wide(pytrends, chunk, geo)

        for keyword, mid_info in chunk:
            wide_series, mode = stage1_results[keyword]
            category = cat_by_kw[keyword]

            if wide_series is None:
                rows.append({"keyword": keyword, "category": category, "geo": geo_key,
                             "data_mode": mode, "error": "no data in wide pull (entity + plain text both failed)"})
                print(f"  SKIPPED '{keyword}' -- no data in wide pull")
                continue

            candidates = find_era_peaks(wide_series, era_bounds)
            if not candidates:
                rows.append({"keyword": keyword, "category": category, "geo": geo_key,
                             "data_mode": mode, "error": "no qualifying peak found in wide pull"})
                print(f"  SKIPPED '{keyword}' -- no qualifying peak")
                continue

            print(f"  '{keyword}': {len(candidates)} era surge(s) found")

            for surge_num, cand in enumerate(candidates, start=1):
                era = cand["era"]
                print(f"    Surge {surge_num}: {cand['date'].date()} "
                      f"(rel_to_anchor={cand['rel_to_anchor']:.2f}, era={era})")

                era_start, era_end = era_bounds[era]
                result = locate_and_measure_stage2(pytrends, keyword, mid_info, mode, cand["date"],
                                                    era_start, era_end, geo)
                row = {
                    "keyword": keyword, "category": category, "era": era, "geo": geo_key,
                    "surge_number": surge_num, "rough_peak_date": cand["date"].date().isoformat(),
                    "rel_to_anchor": round(cand["rel_to_anchor"], 4),
                    **result,
                }
                rows.append(row)
                if "error" in result:
                    print(f"      SKIPPED -- {result['error']}")
                else:
                    print(f"      OK -- era={era}, peak_width={result['peak_width_days']}d")

    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True,
                         help="CSV with columns: keyword, category (era is no longer a column -- "
                              "the algorithm finds it automatically).")
    parser.add_argument("--geo", default="global", choices=list(GEO_CODES.keys()))
    parser.add_argument("--retry-failed", default=None,
                         help="Path to a previous results CSV. Re-runs ONLY the keywords whose rows "
                              "had an error, instead of the whole keyword file.")
    parser.add_argument("--cache", action="store_true",
                         help="Skip topics already successful in this run's output file "
                              "(useful if a previous run partially completed).")
    args = parser.parse_args()

    out_name = f"results_{args.file.replace('.csv', '')}_{args.geo}.csv"

    if args.retry_failed:
        prev = pd.read_csv(args.retry_failed)
        failed_kws = set(prev[prev["error"].notna()]["keyword"]) if "error" in prev.columns else set(prev["keyword"])
        orig = pd.read_csv(args.file)
        failed = orig[orig["keyword"].isin(failed_kws)][["keyword", "category"]]
        print(f"Retrying {len(failed)} previously-failed topics only...")
        failed.to_csv("_retry_temp.csv", index=False)
        retry_df = run_pipeline("_retry_temp.csv", geo_key=args.geo)
        "MERGE, never overwrite: keep every prior row whose keyword wasn't"
        "in the retried set, then append the fresh retry results. This is"
        "the fix for a bug that used to replace the entire output file with"
        "only the retried subset, silently discarding all prior good rows."
        untouched = prev[~prev["keyword"].isin(failed_kws)]
        results_df = pd.concat([untouched, retry_df], ignore_index=True)
    else:
        cache = out_name if args.cache else None
        new_df = run_pipeline(args.file, geo_key=args.geo, cache_path=cache)
        if args.cache and os.path.exists(out_name):
            "SAME merge fix as retry-failed: run_pipeline only returns the"
            "freshly-processed subset when caching, so the previously-saved"
            "successful rows must be merged back in, not discarded."
            prev = pd.read_csv(out_name)
            already_done = set(prev[prev["error"].isna()]["keyword"]) if "error" in prev.columns else set()
            untouched = prev[prev["keyword"].isin(already_done)]
            results_df = pd.concat([untouched, new_df], ignore_index=True)
        else:
            results_df = new_df

    results_df.to_csv(out_name, index=False)
    n_ok = results_df['error'].isna().sum() if 'error' in results_df.columns else len(results_df)
    print(f"\nSaved {out_name} -- {len(results_df)} rows, {n_ok} successful.")
