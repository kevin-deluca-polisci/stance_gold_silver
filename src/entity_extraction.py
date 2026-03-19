"""
Entity-level stance detection for historical newspaper articles.

This module handles:
1. Sentence splitting (via nltk)
2. Named entity detection via regex matching against a known-figures table,
   plus a general capitalized-name heuristic for unknown persons
3. Politician disambiguation against a lookup table of ~18 key 1890s figures
4. Separating person-attributed sentences from residual article text
5. Entity-level and residual stance detection using Political DEBATE

Requires: nltk (no spaCy dependency -- avoids Python 3.14 Pydantic v1 issues)
"""

import re
from typing import Optional
from collections import defaultdict

import pandas as pd
import nltk
from tqdm import tqdm

# Ensure punkt tokenizer is available
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


# ---------------------------------------------------------------------------
# Known politicians of the 1890s monetary debate
# ---------------------------------------------------------------------------
# Each entry maps a canonical name to known name variants, their stance on
# the gold/silver question, party, state, and role. This enables
# disambiguation and validation against the historical record.

KNOWN_POLITICIANS = {
    "William Jennings Bryan": {
        "variants": [
            "bryan", "w.j. bryan", "william bryan", "wm. bryan",
            "william j. bryan", "wm j bryan", "mr. bryan", "mr bryan",
            "w. j. bryan", "jennings bryan",
        ],
        "known_stance": "pro-silver",
        "party": "Democrat",
        "state": "NE",
        "role": "Representative (NE); 1896 presidential nominee",
    },
    "William McKinley": {
        "variants": [
            "mckinley", "william mckinley", "wm. mckinley", "mr. mckinley",
            "wm mckinley", "president mckinley", "gov. mckinley",
            "governor mckinley", "maj. mckinley",
        ],
        "known_stance": "pro-gold",
        "party": "Republican",
        "state": "OH",
        "role": "Governor (OH); 1896 presidential nominee; President 1897-1901",
    },
    "Grover Cleveland": {
        "variants": [
            "cleveland", "grover cleveland", "president cleveland",
            "mr. cleveland", "mr cleveland",
        ],
        "known_stance": "pro-gold",
        "party": "Democrat",
        "state": "NY",
        "role": "President 1893-1897",
    },
    "John Sherman": {
        "variants": [
            "sherman", "john sherman", "senator sherman", "mr. sherman",
        ],
        "known_stance": "pro-gold",
        "party": "Republican",
        "state": "OH",
        "role": "Senator (OH); author of Sherman Silver Purchase Act",
    },
    "Richard P. Bland": {
        "variants": [
            "bland", "richard bland", "r.p. bland", "mr. bland",
            "silver dick", "silver dick bland", "congressman bland",
        ],
        "known_stance": "pro-silver",
        "party": "Democrat",
        "state": "MO",
        "role": "Representative (MO); co-author of Bland-Allison Act",
    },
    "Henry M. Teller": {
        "variants": [
            "teller", "henry teller", "senator teller", "mr. teller",
        ],
        "known_stance": "pro-silver",
        "party": "Republican (Silver Republican)",
        "state": "CO",
        "role": "Senator (CO); led Silver Republican bolt in 1896",
    },
    "William M. Stewart": {
        "variants": [
            "stewart", "william stewart", "senator stewart",
        ],
        "known_stance": "pro-silver",
        "party": "Republican (Silver)",
        "state": "NV",
        "role": "Senator (NV)",
    },
    "Benjamin Harrison": {
        "variants": [
            "harrison", "benjamin harrison", "president harrison",
            "mr. harrison",
        ],
        "known_stance": "pro-gold",
        "party": "Republican",
        "state": "IN",
        "role": "President 1889-1893",
    },
    "Nelson Aldrich": {
        "variants": [
            "aldrich", "nelson aldrich", "senator aldrich",
        ],
        "known_stance": "pro-gold",
        "party": "Republican",
        "state": "RI",
        "role": "Senator (RI); finance committee chair",
    },
    "Adlai Stevenson I": {
        "variants": [
            "stevenson", "adlai stevenson", "mr. stevenson",
        ],
        "known_stance": "pro-silver",
        "party": "Democrat",
        "state": "IL",
        "role": "Vice President 1893-1897",
    },
    "William H. Harvey": {
        "variants": [
            "harvey", "william harvey", "coin harvey", "w.h. harvey",
        ],
        "known_stance": "pro-silver",
        "party": "N/A",
        "state": "IL",
        "role": "Author of 'Coin's Financial School'; silver propagandist",
    },
    "David B. Hill": {
        "variants": [
            "hill", "david hill", "senator hill", "david b. hill",
        ],
        "known_stance": "pro-gold",
        "party": "Democrat",
        "state": "NY",
        "role": "Senator (NY); Gold Democrat leader",
    },
    "Arthur Sewall": {
        "variants": [
            "sewall", "arthur sewall",
        ],
        "known_stance": "pro-silver",
        "party": "Democrat",
        "state": "ME",
        "role": "1896 VP nominee (Democratic ticket)",
    },
    "Thomas B. Reed": {
        "variants": [
            "reed", "thomas reed", "tom reed", "speaker reed",
        ],
        "known_stance": "pro-gold",
        "party": "Republican",
        "state": "ME",
        "role": "Speaker of the House",
    },
    "Marcus Hanna": {
        "variants": [
            "hanna", "mark hanna", "marcus hanna", "mr. hanna",
        ],
        "known_stance": "pro-gold",
        "party": "Republican",
        "state": "OH",
        "role": "RNC Chair; McKinley campaign manager",
    },
    "James B. Weaver": {
        "variants": [
            "weaver", "james weaver", "general weaver",
        ],
        "known_stance": "pro-silver",
        "party": "Populist",
        "state": "IA",
        "role": "1892 Populist presidential nominee",
    },
    "John Peter Altgeld": {
        "variants": [
            "altgeld", "john altgeld", "governor altgeld", "gov. altgeld",
        ],
        "known_stance": "pro-silver",
        "party": "Democrat",
        "state": "IL",
        "role": "Governor (IL)",
    },
    "William Allen": {
        "variants": [
            "william allen", "senator allen",
        ],
        "known_stance": "pro-silver",
        "party": "Populist",
        "state": "NE",
        "role": "Senator (NE)",
    },
}

# ---------------------------------------------------------------------------
# Build fast lookup structures
# ---------------------------------------------------------------------------

# Lowercased variant -> canonical name
_VARIANT_TO_CANONICAL = {}
for canonical, info in KNOWN_POLITICIANS.items():
    for variant in info["variants"]:
        _VARIANT_TO_CANONICAL[variant.lower()] = canonical

# Build a single compiled regex that matches any known-politician variant
# Sort by length (longest first) so "william jennings bryan" matches before "bryan"
_all_variants = []
for canonical, info in KNOWN_POLITICIANS.items():
    for variant in info["variants"]:
        _all_variants.append(variant)
_all_variants.sort(key=len, reverse=True)

_KNOWN_POLITICIAN_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(v) for v in _all_variants) + r")\b",
    re.IGNORECASE,
)

# General person-name heuristic: sequences of 2-4 capitalized words
# (catches names not in our lookup table)
# Excludes common non-name capitalized words that appear in 1890s newspaper text
_COMMON_NON_NAMES = {
    "the", "united", "states", "congress", "senate", "house",
    "representatives", "republican", "democrat", "democratic", "party",
    "national", "silver", "gold", "standard", "free", "coinage",
    "american", "new", "york", "city", "state", "county",
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "north", "south", "east", "west", "act", "bill", "law",
}

_PERSON_NAME_PATTERN = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?(?:\s+[A-Z][a-z]+){1,3})\b"
)


def disambiguate_entity(name: str) -> Optional[str]:
    """
    Try to match a raw name string to a known politician.

    Checks against the variant lookup table. Returns the canonical name
    if found, None otherwise.
    """
    name_lower = name.lower().strip()

    # Direct match against variants
    if name_lower in _VARIANT_TO_CANONICAL:
        return _VARIANT_TO_CANONICAL[name_lower]

    # Try matching last name only (common in newspaper text)
    parts = name_lower.split()
    if parts:
        last = parts[-1]
        if last in _VARIANT_TO_CANONICAL:
            return _VARIANT_TO_CANONICAL[last]

    return None


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using nltk."""
    if not text or not text.strip():
        return []
    try:
        return nltk.sent_tokenize(text)
    except Exception:
        # Fallback: split on period-space patterns
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _find_entities_in_sentence(sentence: str) -> tuple[list[str], list[str]]:
    """
    Find person entities in a sentence using regex matching.

    Returns (known_matches, unknown_matches) where:
    - known_matches: list of canonical politician names found
    - unknown_matches: list of raw name strings from the general heuristic
    """
    known_matches = []
    unknown_matches = []

    # 1. Check for known politician variants
    for match in _KNOWN_POLITICIAN_PATTERN.finditer(sentence):
        matched_text = match.group(0)
        canonical = disambiguate_entity(matched_text)
        if canonical and canonical not in known_matches:
            known_matches.append(canonical)

    # 2. General person-name heuristic for unknown persons
    for match in _PERSON_NAME_PATTERN.finditer(sentence):
        name = match.group(0).strip()
        # Skip if it's already matched as a known politician
        if disambiguate_entity(name) is not None:
            continue
        # Skip if all words are common non-name words
        words = name.lower().split()
        if all(w.rstrip(".") in _COMMON_NON_NAMES for w in words):
            continue
        # Skip very short matches (likely false positives)
        if len(name) < 5:
            continue
        if name not in unknown_matches:
            unknown_matches.append(name)

    return known_matches, unknown_matches


def extract_entities_from_article(
    text: str,
) -> list[dict]:
    """
    Split article into sentences and find person entities in each.

    Returns a list of dicts, one per sentence, with:
        - 'sentence_idx': int
        - 'sentence_text': str
        - 'known_entities': list of canonical politician names
        - 'unknown_entities': list of raw name strings (general heuristic)
        - 'has_person': bool
    """
    sentences = _split_sentences(text)
    results = []

    for i, sent in enumerate(sentences):
        known, unknown = _find_entities_in_sentence(sent)
        results.append({
            "sentence_idx": i,
            "sentence_text": sent,
            "known_entities": known,
            "unknown_entities": unknown,
            "has_person": len(known) > 0 or len(unknown) > 0,
        })

    return results


def separate_article_text(
    sentence_records: list[dict],
) -> dict:
    """
    Separate an article's sentences into person-attributed and residual buckets.

    Parameters
    ----------
    sentence_records : list[dict]
        Output of extract_entities_from_article.

    Returns
    -------
    dict with keys:
        - 'person_sentences': dict mapping name -> list of sentence texts
        - 'residual_sentences': list of sentence texts with no person mentions
        - 'person_text': dict mapping name -> joined text
        - 'residual_text': str, joined residual sentences
        - 'all_known_entities': set of canonical names found
        - 'n_person_sentences': int
        - 'n_residual_sentences': int
    """
    person_sentences = defaultdict(list)
    residual_sentences = []
    all_known = set()

    for rec in sentence_records:
        if rec["known_entities"]:
            for name in rec["known_entities"]:
                person_sentences[name].append(rec["sentence_text"])
                all_known.add(name)
        elif rec["unknown_entities"]:
            for raw_name in rec["unknown_entities"]:
                person_sentences[raw_name].append(rec["sentence_text"])
        else:
            residual_sentences.append(rec["sentence_text"])

    person_text = {
        name: " ".join(sents) for name, sents in person_sentences.items()
    }
    residual_text = " ".join(residual_sentences)

    return {
        "person_sentences": dict(person_sentences),
        "residual_sentences": residual_sentences,
        "person_text": person_text,
        "residual_text": residual_text,
        "all_known_entities": all_known,
        "n_person_sentences": sum(len(s) for s in person_sentences.values()),
        "n_residual_sentences": len(residual_sentences),
    }


def process_articles_for_entities(
    df: pd.DataFrame,
    text_column: str = "article",
) -> pd.DataFrame:
    """
    Run entity extraction and sentence splitting on all articles in a DataFrame.

    Returns a new DataFrame with one row per (article, entity) pair,
    plus metadata about residual text.

    Parameters
    ----------
    df : pd.DataFrame
        Must have an 'article_id' column and a text column.
    text_column : str
        Column containing article text.

    Returns
    -------
    pd.DataFrame with columns:
        - article_id
        - entity_name: canonical name if known politician, raw name otherwise
        - is_known_politician: bool
        - entity_text: joined text of sentences mentioning this entity
        - residual_text: joined text of sentences not mentioning any person
        - n_entity_sentences: number of sentences attributed to this entity
        - n_residual_sentences: number of residual sentences
        - n_total_sentences: total sentences in article
    """
    records = []
    texts = df[text_column].fillna("").tolist()
    article_ids = df["article_id"].tolist()

    print(f"Processing {len(texts)} articles for entity extraction...")

    for i in tqdm(range(len(texts)), desc="Entity extraction"):
        article_id = article_ids[i]
        text = texts[i]

        sentence_records = extract_entities_from_article(text)
        separation = separate_article_text(sentence_records)
        n_total = len(sentence_records)

        if not separation["person_text"]:
            # No entities found — still record the article for residual analysis
            records.append({
                "article_id": article_id,
                "entity_name": None,
                "is_known_politician": False,
                "entity_text": "",
                "residual_text": separation["residual_text"],
                "n_entity_sentences": 0,
                "n_residual_sentences": separation["n_residual_sentences"],
                "n_total_sentences": n_total,
            })
        else:
            for entity_name, entity_text in separation["person_text"].items():
                is_known = entity_name in KNOWN_POLITICIANS
                n_entity_sents = len(separation["person_sentences"].get(
                    entity_name, []
                ))
                records.append({
                    "article_id": article_id,
                    "entity_name": entity_name,
                    "is_known_politician": is_known,
                    "entity_text": entity_text,
                    "residual_text": separation["residual_text"],
                    "n_entity_sentences": n_entity_sents,
                    "n_residual_sentences": separation["n_residual_sentences"],
                    "n_total_sentences": n_total,
                })

    result_df = pd.DataFrame(records)
    print(f"  Generated {len(result_df)} entity records from {len(df)} articles.")

    return result_df


def build_entity_hypotheses(entity_name: str) -> dict:
    """
    Build stance detection hypotheses for a specific entity.

    Parameters
    ----------
    entity_name : str
        The person's name (canonical or raw).

    Returns
    -------
    dict with keys 'pro_gold' and 'pro_silver' mapping to hypothesis strings.
    """
    return {
        "pro_gold": f"According to this text, {entity_name} supports the gold standard.",
        "pro_silver": f"According to this text, {entity_name} supports the free coinage of silver.",
    }


RESIDUAL_HYPOTHESES = {
    "pro_gold": "The author of this text supports the gold standard.",
    "pro_silver": "The author of this text supports the free coinage of silver.",
}
