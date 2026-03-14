"""
Utilities for looking up geographic information for newspapers via LCCN.

Two-stage approach:
  1. Extract state/city from newspaper names (fast, no API needed)
  2. Fall back to loc.gov API for any that can't be resolved from names

This avoids rate-limit issues with the LoC APIs while still getting
good geographic coverage.
"""

import re
import time
import json
from pathlib import Path
from typing import Optional

import requests
import pandas as pd
from tqdm import tqdm


# -------------------------------------------------------------------------
# Stage 1: Extract geography from newspaper names
# -------------------------------------------------------------------------

# US states and territories (1890s era)
US_STATES = {
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
    "Connecticut", "Delaware", "District of Columbia", "Florida", "Georgia",
    "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky",
    "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan",
    "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada",
    "New Hampshire", "New Jersey", "New Mexico", "New York",
    "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon",
    "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota",
    "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington",
    "West Virginia", "Wisconsin", "Wyoming",
}

# Major US cities -> state mapping (focused on 1890s newspaper cities)
CITY_TO_STATE = {
    # Major cities
    "new york": "New York", "new-york": "New York", "brooklyn": "New York",
    "los angeles": "California", "san francisco": "California",
    "san jose": "California", "sacramento": "California",
    "oakland": "California",
    "chicago": "Illinois",
    "philadelphia": "Pennsylvania", "pittsburgh": "Pennsylvania",
    "pittsburg": "Pennsylvania",
    "boston": "Massachusetts",
    "baltimore": "Maryland",
    "st. louis": "Missouri", "st louis": "Missouri",
    "kansas city": "Missouri",
    "new orleans": "Louisiana",
    "cleveland": "Ohio", "cincinnati": "Ohio", "columbus": "Ohio",
    "detroit": "Michigan",
    "milwaukee": "Wisconsin",
    "minneapolis": "Minnesota", "st. paul": "Minnesota",
    "indianapolis": "Indiana",
    "denver": "Colorado",
    "atlanta": "Georgia", "savannah": "Georgia",
    "memphis": "Tennessee", "nashville": "Tennessee",
    "birmingham": "Alabama", "mobile": "Alabama",
    "richmond": "Virginia", "norfolk": "Virginia",
    "charleston": "South Carolina",
    "charlotte": "North Carolina", "raleigh": "North Carolina",
    "dallas": "Texas", "houston": "Texas", "san antonio": "Texas",
    "galveston": "Texas", "fort worth": "Texas",
    "omaha": "Nebraska", "lincoln": "Nebraska",
    "portland": "Oregon",
    "seattle": "Washington", "tacoma": "Washington",
    "salt lake": "Utah", "salt lake city": "Utah",
    "prescott": "Arizona", "phoenix": "Arizona", "tucson": "Arizona",
    "tombstone": "Arizona",
    "boise": "Idaho",
    "helena": "Montana", "butte": "Montana",
    "cheyenne": "Wyoming",
    "bismarck": "North Dakota",
    "pierre": "South Dakota", "deadwood": "South Dakota",
    "topeka": "Kansas", "wichita": "Kansas",
    "little rock": "Arkansas",
    "des moines": "Iowa",
    "jackson": "Mississippi",
    "baton rouge": "Louisiana",
    "spokane": "Washington",
    "hartford": "Connecticut",
    "providence": "Rhode Island",
    "trenton": "New Jersey",
    "wilmington": "Delaware",
    "honolulu": "Hawaii",
    "santa fe": "New Mexico", "albuquerque": "New Mexico",
    "reno": "Nevada", "carson city": "Nevada", "virginia city": "Nevada",
    "burlington": "Vermont",
    "concord": "New Hampshire",
    "augusta": "Maine", "bangor": "Maine",
    "lexington": "Kentucky", "louisville": "Kentucky",
    "wheeling": "West Virginia",
    "columbia": "South Carolina",
    "montgomery": "Alabama",
    "tallahassee": "Florida", "jacksonville": "Florida",
    "macon": "Georgia",
    "washington": "District of Columbia",
    "abbeville": "South Carolina",
    "olympia": "Washington",
    "sacramento": "California",
    "lansing": "Michigan",
    "springfield": "Illinois",
    "jefferson city": "Missouri",
    "harrisburg": "Pennsylvania",
    "albany": "New York",
    "dover": "Delaware",
    "annapolis": "Maryland",
    "montpelier": "Vermont",
    "frankfort": "Kentucky",
    "madison": "Wisconsin",
    "st. paul": "Minnesota",
    "carson": "Nevada",
    "salem": "Oregon",
    "baton rouge": "Louisiana",
}

# State abbreviations that might appear in newspaper names
STATE_ABBREVS = {
    "ala": "Alabama", "ariz": "Arizona", "ark": "Arkansas",
    "cal": "California", "calif": "California",
    "colo": "Colorado", "conn": "Connecticut", "del": "Delaware",
    "fla": "Florida", "ga": "Georgia",
    "ill": "Illinois", "ind": "Indiana",
    "kan": "Kansas", "kans": "Kansas",
    "ky": "Kentucky", "la": "Louisiana",
    "md": "Maryland", "mass": "Massachusetts",
    "mich": "Michigan", "minn": "Minnesota",
    "miss": "Mississippi", "mo": "Missouri",
    "mont": "Montana", "neb": "Nebraska", "nebr": "Nebraska",
    "nev": "Nevada", "n.h": "New Hampshire",
    "n.j": "New Jersey", "n.m": "New Mexico",
    "n.y": "New York", "n.c": "North Carolina",
    "n.d": "North Dakota", "o": "Ohio",
    "okla": "Oklahoma", "or": "Oregon", "ore": "Oregon",
    "pa": "Pennsylvania", "r.i": "Rhode Island",
    "s.c": "South Carolina", "s.d": "South Dakota",
    "tenn": "Tennessee", "tex": "Texas",
    "vt": "Vermont", "va": "Virginia",
    "wash": "Washington", "w.va": "West Virginia",
    "wis": "Wisconsin", "wyo": "Wyoming",
    "d.c": "District of Columbia",
}


def extract_state_from_name(newspaper_name: str) -> tuple[str, str]:
    """
    Try to extract state (and optionally city) from a newspaper name.

    Returns (state, city) tuple. Either or both may be empty strings.

    Examples:
        "Arizona weekly journal-miner." -> ("Arizona", "")
        "Los Angeles herald."           -> ("California", "Los Angeles")
        "New-York tribune."             -> ("New York", "New York")
        "The American."                 -> ("", "")
    """
    if not newspaper_name:
        return ("", "")

    name = newspaper_name.strip().rstrip(".")
    name_lower = name.lower()

    # 1. Check for full state names in the newspaper name
    for state in sorted(US_STATES, key=len, reverse=True):
        if state.lower() in name_lower:
            return (state, "")

    # 2. Check for city names in the newspaper name
    for city, state in sorted(CITY_TO_STATE.items(), key=lambda x: len(x[0]), reverse=True):
        if city in name_lower:
            return (state, city.title())

    # 3. Check for state abbreviations (often in parentheses)
    # e.g., "Daily gazette (Wilmington, Del.)"
    paren_match = re.search(r"\(([^)]+)\)", name)
    if paren_match:
        paren_content = paren_match.group(1).lower()
        for abbr, state in STATE_ABBREVS.items():
            if abbr in paren_content:
                return (state, "")

    return ("", "")


# -------------------------------------------------------------------------
# Stage 2: loc.gov API fallback
# -------------------------------------------------------------------------

LOC_API_BASE = "https://www.loc.gov/item"


def lookup_lccn(lccn: str) -> Optional[dict]:
    """
    Look up a single LCCN via the loc.gov API with retry on rate limiting.
    """
    url = f"{LOC_API_BASE}/{lccn}/?fo=json"
    max_retries = 5
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 429:
                wait = 2 ** (attempt + 1)
                print(f"  Rate limited on {lccn}, waiting {wait}s "
                      f"(attempt {attempt+1}/{max_retries})...")
                time.sleep(wait)
                continue
            if resp.status_code == 200:
                data = resp.json()
                item = data.get("item", {})

                state_list = item.get("location_state", [])
                city_list = item.get("location_city", [])
                state = state_list[0] if state_list else ""
                city = city_list[0] if city_list else ""

                title = item.get("title", "")

                date_str = item.get("date", "")
                start_year = ""
                end_year = ""
                if date_str:
                    parts = date_str.replace("?", "").split("-")
                    if len(parts) >= 1:
                        start_year = parts[0].strip()
                    if len(parts) >= 2:
                        end_year = parts[1].strip()

                place = item.get("location_str", "")
                if isinstance(place, list):
                    place = place[0] if place else ""

                publisher = ""
                contributors = item.get("contributor_names", [])
                if contributors:
                    publisher = (contributors[0]
                                 if isinstance(contributors[0], str) else "")

                return {
                    "lccn": lccn,
                    "title": title,
                    "place_of_publication": place,
                    "city": city,
                    "state": state,
                    "start_year": start_year,
                    "end_year": end_year,
                    "publisher": publisher,
                }
            else:
                print(f"  Warning: status {resp.status_code} for LCCN {lccn}")
                return None
        except requests.RequestException as e:
            print(f"  Warning: request failed for LCCN {lccn}: {e}")
            return None
    print(f"  Failed after {max_retries} retries for LCCN {lccn}")
    return None


def normalize_state(state_str: str) -> str:
    """Normalize a state string to title case."""
    if not state_str:
        return ""
    return state_str.strip().title()


# -------------------------------------------------------------------------
# Combined crosswalk builder
# -------------------------------------------------------------------------

def build_lccn_crosswalk(
    lccns: list[str],
    newspaper_names: Optional[dict[str, str]] = None,
    cache_path: Optional[str] = None,
    rate_limit_seconds: float = 8.0,
    use_api: bool = True,
) -> pd.DataFrame:
    """
    Build a crosswalk DataFrame mapping LCCNs to geographic info.

    Three-stage process:
      1. Load any cached API results
      2. Try to extract state from newspaper names (instant, no API)
      3. Fall back to loc.gov API for remaining unknowns

    Parameters
    ----------
    lccns : list[str]
        Unique LCCNs to look up.
    newspaper_names : dict[str, str], optional
        Mapping of LCCN -> newspaper_name for name-based extraction.
    cache_path : str, optional
        Path to a JSON file for caching results.
    rate_limit_seconds : float
        Seconds to wait between API calls.
    use_api : bool
        Whether to query the loc.gov API for unresolved LCCNs.

    Returns
    -------
    pd.DataFrame
        Crosswalk with columns: lccn, title, place_of_publication,
        city, state, state_full, start_year, end_year, publisher, source.
    """
    # --- Stage 1: Load cache ---
    cache = {}
    if cache_path and Path(cache_path).exists():
        with open(cache_path, "r") as f:
            cached_data = json.load(f)
        for k, v in cached_data.items():
            if v.get("state") or v.get("city") or v.get("title"):
                cache[k] = v
        print(f"Loaded {len(cache)} cached LCCN lookups.")

    # --- Stage 2: Name-based extraction ---
    name_resolved = 0
    if newspaper_names:
        for lccn in lccns:
            if lccn in cache:
                continue
            name = newspaper_names.get(lccn, "")
            state, city = extract_state_from_name(name)
            if state:
                cache[lccn] = {
                    "lccn": lccn,
                    "title": name,
                    "place_of_publication": f"{city}, {state}" if city else state,
                    "city": city,
                    "state": state.lower(),
                    "start_year": "",
                    "end_year": "",
                    "publisher": "",
                    "source": "name_extraction",
                }
                name_resolved += 1

    print(f"Resolved {name_resolved} LCCNs from newspaper names.")

    # --- Stage 3: API fallback ---
    still_missing = [lccn for lccn in lccns if lccn not in cache and lccn]

    if still_missing and use_api:
        print(f"Looking up {len(still_missing)} remaining LCCNs via loc.gov API "
              f"(~{len(still_missing) * rate_limit_seconds:.0f}s)...")

        for lccn in tqdm(still_missing, desc="API lookups"):
            info = lookup_lccn(lccn)
            if info:
                info["source"] = "loc_gov_api"
                cache[lccn] = info
            else:
                cache[lccn] = {
                    "lccn": lccn, "title": "", "city": "", "state": "",
                    "place_of_publication": "", "start_year": "",
                    "end_year": "", "publisher": "", "source": "failed",
                }
            time.sleep(rate_limit_seconds)
    elif still_missing:
        print(f"Skipping API lookup for {len(still_missing)} unresolved LCCNs "
              f"(use_api=False).")
        for lccn in still_missing:
            if lccn not in cache:
                cache[lccn] = {
                    "lccn": lccn, "title": "", "city": "", "state": "",
                    "place_of_publication": "", "start_year": "",
                    "end_year": "", "publisher": "", "source": "unresolved",
                }

    # --- Save cache ---
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(cache, f, indent=2)
        print(f"Saved {len(cache)} LCCN lookups to cache.")

    # --- Build output DataFrame ---
    results = [cache[lccn] for lccn in lccns if lccn in cache]
    df = pd.DataFrame(results)
    if "state" in df.columns:
        df["state_full"] = df["state"].apply(normalize_state)

    # Summary
    n_with_state = df["state_full"].ne("").sum() if "state_full" in df.columns else 0
    print(f"\nGeographic coverage: {n_with_state}/{len(df)} LCCNs have state info.")

    return df
