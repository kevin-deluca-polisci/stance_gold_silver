"""
Utilities for downloading and filtering American Stories data.

The American Stories dataset (Dell Research Harvard) provides article-level
text from digitized historical U.S. newspapers. We download parquet files
directly via huggingface_hub (bypassing the `datasets` library to avoid
Python 3.14 pickle compatibility issues) and filter for articles related
to the gold/silver monetary standard debate.
"""

import os
import re
from typing import Optional

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Tier 1 keywords: high-confidence monetary debate terms
# ---------------------------------------------------------------------------
TIER1_KEYWORDS = [
    # Core debate terms
    "free silver",
    "free coinage",
    "gold standard",
    "bimetallism",
    "bimetallic",
    "silver standard",
    "sound money",
    "honest money",
    # Ratio / slogans
    "sixteen to one",
    "16 to 1",
    # Historical references
    "crime of 73",
    "crime of 1873",
    "cross of gold",
    # Factions
    "goldbugs",
    "gold bugs",
    "gold bug",
    "silverites",
    "silverite",
    # Legislation
    "sherman silver",
    "bland-allison",
    "bland allison",
    # Coinage terms
    "silver coinage",
    "gold coinage",
    "coinage of silver",
    "coinage of gold",
    # Broader monetary
    "monetary standard",
    "currency question",
]

# Pre-compile a single regex pattern for efficiency
TIER1_PATTERN = re.compile(
    "|".join(re.escape(kw) for kw in TIER1_KEYWORDS),
    re.IGNORECASE,
)


# Pattern to extract LCCN from article_id
# e.g. "12_1890-04-23_p2_sn84027718_00271762707_1890042301_0085" -> "sn84027718"
_LCCN_PATTERN = re.compile(r"(sn\d+)")


def extract_lccn(article_id: str) -> str:
    """Extract the LCCN (e.g. 'sn84027718') from an article_id string."""
    if not article_id:
        return ""
    match = _LCCN_PATTERN.search(str(article_id))
    return match.group(1) if match else ""


def article_matches_keywords(
    text: str,
    pattern: re.Pattern = TIER1_PATTERN,
) -> bool:
    """Check whether an article's text matches any Tier 1 keyword."""
    if not text:
        return False
    return bool(pattern.search(text))


def get_matched_keywords(
    text: str,
    pattern: re.Pattern = TIER1_PATTERN,
) -> list[str]:
    """Return all Tier 1 keywords found in the text (deduplicated, lowered)."""
    if not text:
        return []
    matches = pattern.findall(text)
    return list(set(m.lower() for m in matches))


# ---------------------------------------------------------------------------
# Column name mapping — the parquet files on HuggingFace may use different
# column names than what the `datasets` loading script exposes. We normalise
# to a consistent set of names used throughout the project.
# ---------------------------------------------------------------------------
COLUMN_MAP = {
    # Possible parquet column -> our standard name
    "article_text": "article",
    "article": "article",
    "full_text": "article",
    "text": "article",
    "newspaper_name": "newspaper_name",
    "newspaper": "newspaper_name",
    "date": "date",
    "edition": "edition",
    "page": "page",
    "page_number": "page",
    "headline": "headline",
    "byline": "byline",
    "article_id": "article_id",
    "lccn": "lccn",
}

REQUIRED_OUTPUT_COLS = [
    "article_id", "newspaper_name", "edition", "date",
    "page", "headline", "byline", "article", "lccn",
]


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to our standard names, filling missing ones."""
    rename = {}
    for col in df.columns:
        key = col.lower().strip()
        if key in COLUMN_MAP:
            rename[col] = COLUMN_MAP[key]
    df = df.rename(columns=rename)
    for col in REQUIRED_OUTPUT_COLS:
        if col not in df.columns:
            df[col] = ""
    return df


# We use the community parquet conversion of American Stories, which has
# pre-converted parquet files organized by year under data/.
PARQUET_REPO_ID = "davanstrien/AmericanStories-parquet"


def _list_parquet_files_for_year(year: str) -> list[tuple[str, str]]:
    """
    Find parquet file paths for a given year in the community parquet repo.

    Returns a list of (filename, revision) tuples.
    Files follow the pattern: data/YEAR-NNNNN-of-NNNNN-hash.parquet
    """
    api = HfApi()

    try:
        all_files = api.list_repo_files(
            repo_id=PARQUET_REPO_ID,
            repo_type="dataset",
            revision="main",
        )
        # Match files like data/1893-00000-of-00002-abc123.parquet
        # Use startswith to avoid e.g. "1893" matching inside "18930"
        year_files = [
            f for f in all_files
            if f.endswith(".parquet") and f.startswith(f"data/{year}-")
        ]
        if year_files:
            return [(f, "main") for f in sorted(year_files)]
    except Exception as e:
        print(f"  Warning: could not list files for {year}: {e}")

    return []


def download_and_filter_year(
    year: str,
    max_articles: Optional[int] = None,
    min_article_length: int = 100,
    cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Download parquet files for one year of American Stories data and return
    a DataFrame containing only articles that match Tier 1 keywords.

    Parameters
    ----------
    year : str
        Four-digit year string, e.g. "1893".
    max_articles : int, optional
        If set, only process this many articles (useful for testing).
    min_article_length : int
        Minimum character length for an article to be considered.
    cache_dir : str, optional
        Local directory to cache downloaded parquet files.

    Returns
    -------
    pd.DataFrame
        Filtered articles with standardised columns plus 'matched_keywords'.
    """
    print(f"Finding parquet files for year {year}...")
    file_info = _list_parquet_files_for_year(year)

    if not file_info:
        print(f"  No parquet files found for year {year}.")
        return pd.DataFrame()

    print(f"  Found {len(file_info)} parquet file(s) for {year}.")

    # Download and read all parquet files for this year
    dfs = []
    for filepath, revision in file_info:
        print(f"  Downloading {filepath}...")
        local_path = hf_hub_download(
            repo_id=PARQUET_REPO_ID,
            filename=filepath,
            repo_type="dataset",
            revision=revision,
            cache_dir=cache_dir,
        )
        df_part = pd.read_parquet(local_path)
        dfs.append(df_part)

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    print(f"  Loaded {len(df):,} total articles for {year}.")
    print(f"  Columns in parquet: {list(df.columns)}")

    # Normalise column names
    df = _normalise_columns(df)

    # Apply max_articles limit if set
    if max_articles and len(df) > max_articles:
        df = df.head(max_articles)
        print(f"  Truncated to first {max_articles:,} articles for testing.")

    # Filter by article length
    df["article"] = df["article"].fillna("")
    df = df[df["article"].str.len() >= min_article_length].copy()

    # Filter by keywords
    mask = df["article"].apply(article_matches_keywords)
    df_filtered = df[mask].copy()

    # Add matched keywords column
    df_filtered["matched_keywords"] = df_filtered["article"].apply(
        get_matched_keywords
    )
    df_filtered["year"] = year

    # Extract LCCN from article_id if the lccn column is empty
    if df_filtered["lccn"].eq("").all() or df_filtered["lccn"].isna().all():
        df_filtered["lccn"] = df_filtered["article_id"].apply(extract_lccn)
        n_found = df_filtered["lccn"].ne("").sum()
        print(f"  Extracted LCCN from article_id for {n_found:,} articles.")

    # Keep only the columns we need
    output_cols = REQUIRED_OUTPUT_COLS + ["year", "matched_keywords"]
    df_filtered = df_filtered[[c for c in output_cols if c in df_filtered.columns]]

    print(f"  Year {year}: {len(df_filtered):,} articles match keywords.")

    return df_filtered


def download_and_filter_years(
    years: list[str],
    max_articles_per_year: Optional[int] = None,
    min_article_length: int = 100,
    save_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Download and filter multiple years, concatenating the results.

    Parameters
    ----------
    years : list[str]
        List of year strings, e.g. ["1890", "1891", ...].
    max_articles_per_year : int, optional
        Cap per year (for testing).
    min_article_length : int
        Minimum article length in characters.
    save_path : str, optional
        If provided, save the combined DataFrame to this parquet path.
    cache_dir : str, optional
        Local directory to cache downloaded parquet files.

    Returns
    -------
    pd.DataFrame
        All filtered articles across the requested years.
    """
    all_dfs = []

    for year in years:
        df_year = download_and_filter_year(
            year=year,
            max_articles=max_articles_per_year,
            min_article_length=min_article_length,
            cache_dir=cache_dir,
        )
        all_dfs.append(df_year)

    df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal filtered articles across {len(years)} years: {len(df):,}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df_save = df.copy()
        df_save["matched_keywords"] = df_save["matched_keywords"].apply(
            lambda x: "|".join(x) if isinstance(x, list) else x
        )
        df_save.to_parquet(save_path, index=False)
        print(f"Saved to {save_path}")

    return df


# ---------------------------------------------------------------------------
# Keep old function names as aliases for backward compatibility
# ---------------------------------------------------------------------------
stream_and_filter_year = download_and_filter_year
stream_and_filter_years = download_and_filter_years
