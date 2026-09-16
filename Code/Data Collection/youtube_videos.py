"""Collects YouTube video results for keywords, measures view-count summaries, and separates Shorts from longer videos. The program can restrict searches to each keyword's event window and can resume from previously saved CSV results."""

import argparse
import re
import time

import pandas as pd
import requests

SHORTS_MAX_DURATION_SEC = 183
PAGE_SIZE = 50
MAX_ATTEMPTS_PER_CALL = 3


class QuotaExceededError(Exception):
    """Custom exception used to stop the run when YouTube reports that the daily API quota has been exhausted."""
    pass


def parse_iso8601_duration(duration_str):

    """Convert a YouTube ISO 8601 duration such as PT1H2M3S into a duration in seconds. Return None when the input is empty or does not match the expected pattern."""
    if not duration_str:
        return None
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", duration_str)
    if not m:
        return None
    h, mnt, s = (int(x) if x else 0 for x in m.groups())
    return h * 3600 + mnt * 60 + s


def _get(url, params):

    """Send a GET request with a small retry loop. Network/request failures and non-success HTTP responses are retried, while an explicit YouTube quota error is raised immediately so the main process can stop cleanly."""
    for attempt in range(MAX_ATTEMPTS_PER_CALL):
        try:
            resp = requests.get(url, params=params, timeout=15)
        except requests.exceptions.RequestException as e:
            if attempt == MAX_ATTEMPTS_PER_CALL - 1:
                raise
            time.sleep(2 * (attempt + 1))
            continue

        if resp.status_code == 200:
            return resp.json()

        body = resp.text
        if resp.status_code == 403 and "quotaExceeded" in body:
            raise QuotaExceededError("YouTube daily search quota is used up.")

        if attempt == MAX_ATTEMPTS_PER_CALL - 1:
            raise RuntimeError(f"YouTube API error {resp.status_code}: {body[:200]}")
        time.sleep(2 * (attempt + 1))

    raise RuntimeError("unreachable")


def search_video_ids(api_key, keyword, max_results, published_after=None, published_before=None):

    """Search YouTube for videos matching one keyword and collect up to max_results video IDs, optionally restricted by publication dates. Pagination is followed until the requested number of IDs is reached or YouTube has no next page."""
    video_ids = []
    page_token = None
    while len(video_ids) < max_results:
        params = {
            "part": "id", "q": keyword, "type": "video",
            "maxResults": min(PAGE_SIZE, max_results - len(video_ids)),
            "order": "relevance", "key": api_key,
        }
        if published_after:
            params["publishedAfter"] = published_after
        if published_before:
            params["publishedBefore"] = published_before
        if page_token:
            params["pageToken"] = page_token
        data = _get("https://www.googleapis.com/youtube/v3/search", params)
        video_ids.extend(item["id"]["videoId"] for item in data.get("items", []))
        page_token = data.get("nextPageToken")
        if not page_token:
            break
    return video_ids[:max_results]


def get_video_stats(api_key, video_ids):

    """Fetch duration and view-count statistics for the supplied video IDs in batches of PAGE_SIZE and return them as simple dictionaries."""
    results = []
    for i in range(0, len(video_ids), PAGE_SIZE):
        batch = video_ids[i:i + PAGE_SIZE]
        params = {"part": "contentDetails,statistics", "id": ",".join(batch), "key": api_key}
        data = _get("https://www.googleapis.com/youtube/v3/videos", params)
        for item in data.get("items", []):
            results.append({
                "video_id": item["id"],
                "duration_sec": parse_iso8601_duration(item["contentDetails"].get("duration")),
                "view_count": int(item["statistics"].get("viewCount", 0)),
            })
    return results


def summarize_bucket(stats_list, min_views, label):
    """Calculate descriptive view-count statistics for one group of videos and record how many meet the labeling threshold. The threshold labels results only; it does not filter the videos."""
    if not stats_list:
        return {"division": label, "sample_size": 0}
    views = pd.Series([v["view_count"] for v in stats_list])
    above = int((views >= min_views).sum())
    return {
        "division": label, "sample_size": len(views),
        "above_threshold": above, "below_threshold": len(views) - above,
        "pct_above_threshold": round(100 * above / len(views), 1),
        "total_views": int(views.sum()),
        "median_views": int(views.median()), "mean_views": int(views.mean()),
        "min_views_seen": int(views.min()), "max_views_seen": int(views.max()),
    }


def analyze_keyword(api_key, keyword, category, min_views, max_results,
                     published_after=None, published_before=None, era=None):
    """Returns a single flat dict: one row per keyword/era, with each
    summary stat suffixed by division (_all / _shorts / _videos)."""

    video_ids = search_video_ids(api_key, keyword, max_results, published_after, published_before)
    if not video_ids:
        """FIX: "no videos found" is a legitimate, completed result (zero matches"""
        """in the search window), not a failed API call. It previously shared the"""
        """generic "error" field with real exceptions, which made the resume logic"""
        """below (line ~201) treat it as incomplete and re-fetch it -- and burn"""
        """quota on -- every single resume run. error stays None so it's marked"""
        """done; zero_results records the outcome for downstream code that wants"""
        """to distinguish "confirmed nothing found" from "not yet attempted"."""
        return {"keyword": keyword, "category": category, "era": era,
                "error": None, "zero_results": True}

    stats = [v for v in get_video_stats(api_key, video_ids) if v["duration_sec"] is not None]
    shorts = [v for v in stats if v["duration_sec"] <= SHORTS_MAX_DURATION_SEC]
    longform = [v for v in stats if v["duration_sec"] > SHORTS_MAX_DURATION_SEC]

    row = {"keyword": keyword, "category": category, "era": era, "error": None, "zero_results": False}
    for group, label in [(stats, "all"), (shorts, "shorts"), (longform, "videos")]:
        summary = summarize_bucket(group, min_views, label)
        summary.pop("division", None)
        row.update({f"{stat}_{label}": val for stat, val in summary.items()})
    return row


def main():
    """Parse command-line arguments, prepare the input and output data, resume previously completed work, fetch remaining keywords, and save after every keyword so partial progress survives interruptions."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", required=True, help="CSV with columns: keyword, category")
    p.add_argument("--api-key", required=True)
    p.add_argument("--max-results", type=int, default=500)
    p.add_argument("--min-views", type=int, default=0, help="Just a labeling threshold for "
                    "above/below in the output -- never filters or excludes videos.")
    p.add_argument("--pad-days", type=int, default=3, help="Days to pad on each side of "
                    "event_start_date/event_end_date when filtering by publish date, to catch "
                    "reaction/anticipation videos just outside the measured attention window. "
                    "Default 3. Set to 0 for an exact window, or use --no-date-filter to disable "
                    "date filtering entirely (searches all-time, old behavior).")
    p.add_argument("--no-date-filter", action="store_true", help="Disable publish-date "
                    "filtering even if event_start_date/event_end_date columns are present.")
    args = p.parse_args()

    stem = args.file.rsplit(".", 1)[0]
    for prefix in ("youtube_results_", "results_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
    out_name = f"youtube_results_{stem}_top{args.max_results}.csv"

    topics = pd.read_csv(args.file)
    has_era = "era" in topics.columns

    def make_key(kw, era_val):
        """Some keywords (e.g. recurring slang, or events that spiked in more"""
        """than one era) appear more than once with the SAME keyword text but"""
        """a DIFFERENT date window. Keying on keyword text alone would let a"""
        """later era's fetch silently delete an earlier era's already-saved"""
        """results (same keyword text = "the same row" to naive dedup). When"""
        """an era column is present we key on (keyword, era) instead so each"""
        """occurrence is tracked independently; falls back to keyword-only"""
        """for simple files with no era column (e.g. a flat new-keywords CSV)."""
        return (kw, era_val) if has_era else kw

    if has_era:
        print("NOTE: input has an 'era' column -- tracking progress per (keyword, era) "
              "so keywords that recur across multiple eras don't overwrite each other.\n")


    has_dates = ("event_start_date" in topics.columns and "event_end_date" in topics.columns
                 and not args.no_date_filter)
    if has_dates:
        topics["pub_after_iso"] = pd.to_datetime(topics["event_start_date"], errors="coerce")
        topics["pub_before_iso"] = pd.to_datetime(topics["event_end_date"], errors="coerce")
        n_missing = (topics["pub_after_iso"].isna() | topics["pub_before_iso"].isna()).sum()
        if n_missing:
            print(f"NOTE: {n_missing} row(s) missing a usable event_start_date/event_end_date "
                  f"-- those keywords will be searched with NO date filter (all-time), "
                  f"same as before this fix.")
        pad = pd.Timedelta(days=args.pad_days)
        topics["pub_after_iso"] = (topics["pub_after_iso"] - pad).dt.strftime("%Y-%m-%dT00:00:00Z")
        topics["pub_before_iso"] = (topics["pub_before_iso"] + pad).dt.strftime("%Y-%m-%dT00:00:00Z")
        print(f"Filtering each keyword's video search to its own event_start_date/"
              f"event_end_date window (+/- {args.pad_days} day pad) so YouTube volume reflects "
              f"content from that topic's actual attention window, not all-time uploads.\n")
    else:
        if args.no_date_filter:
            print("Date filtering disabled (--no-date-filter) -- searching all-time uploads.\n")
        else:
            print("NOTE: no event_start_date/event_end_date columns found in --file -- "
                  "searching all-time uploads (old behavior). Pass a CSV with those columns "
                  "(e.g. the Trends pipeline output) to scope each search to that topic's real "
                  "attention window.\n")

    all_rows = []
    already_done = set()
    try:
        prev = pd.read_csv(out_name)
        all_rows = prev.to_dict("records")
        if "error" in prev.columns:
            ok = prev[prev["error"].isna()]
            if has_era and "era" in ok.columns:
                already_done = set(zip(ok["keyword"], ok["era"]))
            else:
                already_done = set(ok["keyword"])
        print(f"Resuming: {out_name} already has {len(already_done)} completed keyword(s), skipping those.")
    except FileNotFoundError:
        pass

    if has_era:
        todo_mask = ~topics.apply(lambda r: make_key(r["keyword"], r["era"]), axis=1).isin(already_done)
    else:
        todo_mask = ~topics["keyword"].isin(already_done)
    todo = topics[todo_mask]
    if todo.empty:
        print("Nothing left to fetch -- every keyword already completed.")
        return

    est_cost_per_kw = 100 * -(-args.max_results // PAGE_SIZE)
    """ceil division"""
    print(f"{len(todo)} keyword(s) to fetch, ~{est_cost_per_kw} quota units each "
          f"-> ~{est_cost_per_kw * len(todo)} units total this run.\n")

    for n, row in enumerate(todo.itertuples(index=False), start=1):
        keyword, category = row.keyword, row.category
        era = getattr(row, "era", None) if has_era else None
        pub_after = getattr(row, "pub_after_iso", None) if has_dates else None
        pub_before = getattr(row, "pub_before_iso", None) if has_dates else None
        """pandas NaT.strftime already produced "NaT" string above for missing dates --"""
        """treat that as "no filter for this row" rather than sending it to the API"""
        if pub_after == "NaT" or pub_before == "NaT":
            pub_after = pub_before = None
        window_note = f" [{pub_after} to {pub_before}]" if pub_after else " [no date filter]"
        era_note = f" ({era})" if era else ""
        print(f"[{n}/{len(todo)}] '{keyword}'{era_note}{window_note}...")
        """this (keyword, era) is being (re)attempted -- drop any stale row from a"""
        """prior failed attempt so retries don't pile up duplicate error rows."""
        """Matching on era (when present) so a DIFFERENT era's already-saved"""
        """rows for the same keyword text are never touched."""
        if has_era:
            all_rows = [r for r in all_rows if not (r.get("keyword") == keyword and r.get("era") == era)]
        else:
            all_rows = [r for r in all_rows if r.get("keyword") != keyword]
        try:
            new_row = analyze_keyword(args.api_key, keyword, category, args.min_views,
                                       args.max_results, pub_after, pub_before, era)
        except QuotaExceededError as e:
            print(f"\nSTOPPING: {e}\n"
                  f"  Completed {n - 1}/{len(todo)} keyword(s) this run before running out.\n"
                  f"  Everything fetched so far is already saved in {out_name}.\n"
                  f"  Re-run this exact same command once quota resets (usually midnight Pacific) "
                  f"to pick up right where this left off.")
            break
        except Exception as e:
            print(f"  ERROR on '{keyword}': {e} -- recorded and moving on.")
            new_row = {"keyword": keyword, "category": category, "era": era, "error": str(e)}

        all_rows.append(new_row)
        pd.DataFrame(all_rows).to_csv(out_name, index=False)
        """saved after EVERY keyword"""

    df = pd.DataFrame(all_rows)
    n_ok = df["error"].isna().sum() if "error" in df.columns else len(df)
    print(f"\nSaved {out_name} -- {len(df)} rows, {n_ok} successful.")


if __name__ == "__main__":
    main()
