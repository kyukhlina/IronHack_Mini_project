"""Cleaning functions for the GSAF Shark Attack dataset.

Practice project (Ironhack Data Analytics bootcamp, Week 2 Quest). Each
function tackles one data cleaning technique; `clean_shark_data` chains them
into a single pipeline.
"""

import re

import pandas as pd

UNUSED_COLUMNS = [
    "Unnamed: 21",
    "Unnamed: 22",
    "href formula",
    "href",
    "pdf",
    "Case Number.1",
    "original order",
]

COLUMN_RENAMES = {
    "Case Number": "case_number",
    "Fatal Y/N": "fatal",
}

ORDINAL_SUFFIX_RE = re.compile(r"(\d{1,2})(st|nd|rd|th)\b", flags=re.IGNORECASE)

COUNTRY_FIXES = {
    "GALAPOGOS ISLANDS": "GALAPAGOS ISLANDS",
}

AGE_NUMBER_RE = re.compile(r"(\d{1,3})")


def drop_unused_columns(df):
    """Drop columns that are empty or purely administrative (source links,
    a near-duplicate of `Case Number`, and the original spreadsheet row
    index) — none of them are useful for analysis."""
    return df.drop(columns=UNUSED_COLUMNS, errors="ignore")


def strip_whitespace(df):
    """Strip leading/trailing whitespace from every text column.

    Several categories (e.g. Activity `"Swimming"` vs `"Swimming "`) are
    silently duplicated in the raw data only because of stray whitespace.
    """
    df = df.copy()
    text_cols = df.select_dtypes(include="object").columns
    for col in text_cols:
        df[col] = df[col].apply(lambda v: v.strip() if isinstance(v, str) else v)
    return df


def standardize_column_names(df):
    """Rename columns to clean, consistent snake_case identifiers."""
    df = df.rename(columns=COLUMN_RENAMES)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def _normalize_type(value):
    if not isinstance(value, str) or not value.strip():
        return "Unknown"
    v = value.strip().lower()
    known = {
        "unprovoked": "Unprovoked",
        "provoked": "Provoked",
        "questionable": "Questionable",
        "watercraft": "Watercraft",
        "sea disaster": "Sea Disaster",
        "unconfirmed": "Unconfirmed",
        "unverified": "Unverified",
        "invalid": "Invalid",
        "under investigation": "Under Investigation",
        "boat": "Boat",
    }
    return known.get(v, "Unknown")


def _normalize_sex(value):
    if not isinstance(value, str):
        return "Unknown"
    v = value.strip().upper()
    if v == "M":
        return "M"
    if v == "F":
        return "F"
    return "Unknown"


def _normalize_fatal(value):
    if not isinstance(value, str):
        return "Unknown"
    v = value.strip().upper()
    if v.startswith("Y"):
        return "Y"
    if v.startswith("N"):
        return "N"
    return "Unknown"


def clean_categorical_columns(df):
    """Collapse inconsistent spellings/casing/stray values in `type`, `sex`
    and `fatal` into a fixed, small set of categories (plus `"Unknown"` for
    anything that doesn't map cleanly, e.g. `2017`, `"lli"`, `"Y x 2"`)."""
    df = df.copy()
    df["type"] = df["type"].apply(_normalize_type)
    df["sex"] = df["sex"].apply(_normalize_sex)
    df["fatal"] = df["fatal"].apply(_normalize_fatal)
    return df


def clean_country(df):
    """Uppercase country names and fix known typos."""
    df = df.copy()
    df["country"] = df["country"].apply(
        lambda v: v.strip().upper() if isinstance(v, str) else v
    )
    df["country"] = df["country"].replace(COUNTRY_FIXES)
    return df


def extract_numeric_age(df):
    """Extract a numeric age from the messy `age` column using regex.

    Keeps the original text as `age_raw` and adds a new numeric `age` column.
    Handles simple digits (`"20"`), decades (`"20's"`, `"20s"`) and ranges/
    multi-victim entries (`"21, 34,24 & 35"`) by taking the first number
    found. Purely qualitative descriptions (`"teen"`, `"adult"`, `"elderly"`)
    and garbage entries (`"?"`, `"MAKE LINE GREEN"`) become `NaN` — recovering
    a precise age for those isn't possible from the text alone.
    """
    df = df.copy()
    df["age_raw"] = df["age"]

    def parse_age(value):
        if not isinstance(value, str):
            return None
        match = AGE_NUMBER_RE.search(value)
        if not match:
            return None
        return int(match.group(1))

    df["age"] = df["age_raw"].apply(parse_age)
    return df


def _strip_ordinal(text):
    return ORDINAL_SUFFIX_RE.sub(r"\1", text)


def parse_attack_date(df):
    """Parse the `date` column into a real `datetime`.

    The raw data mixes several formats: some rows already look like ISO
    timestamps or `"27-Feb-25"`, others only have a day + month with no year
    (`"23rd June"`, which needs the ordinal suffix stripped and the `year`
    column merged in), and some are just free text (`"Before 1906"`,
    `"1900-1905"`) that can't be parsed into a single date at all.

    Keeps the original text as `date_raw`; unparseable rows become `NaT`
    (not silently dropped) and a `month` column is derived where possible.
    """
    df = df.copy()
    df["date_raw"] = df["date"]

    def parse_one(raw, year):
        if not isinstance(raw, str):
            return pd.NaT
        text = raw.strip()
        if not text:
            return pd.NaT

        parsed = pd.to_datetime(text, errors="coerce")
        if pd.notna(parsed):
            return parsed

        no_ordinal = _strip_ordinal(text)
        if pd.notna(year) and year:
            candidate = f"{no_ordinal} {int(year)}"
            parsed = pd.to_datetime(candidate, errors="coerce")
            if pd.notna(parsed):
                return parsed

        return pd.NaT

    df["date_parsed"] = [
        parse_one(raw, year) for raw, year in zip(df["date_raw"], df["year"])
    ]
    df["month"] = df["date_parsed"].dt.month_name()
    return df


def remove_duplicate_rows(df):
    """Drop exact duplicate rows (checked after cleaning, since cleaning can
    collapse rows that only differed by whitespace/casing)."""
    return df.drop_duplicates()


def fill_missing_categoricals(df):
    """Fill missing text fields with `"Unknown"` instead of leaving `NaN`,
    so groupby/value_counts during EDA don't silently drop those rows."""
    df = df.copy()
    categorical_cols = ["activity", "species", "location", "state", "name", "country"]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")
    return df


def clean_shark_data(df):
    """Run the full cleaning pipeline in order and return the cleaned df."""
    df = drop_unused_columns(df)
    df = strip_whitespace(df)
    df = standardize_column_names(df)
    df = clean_categorical_columns(df)
    df = clean_country(df)
    df = extract_numeric_age(df)
    df = parse_attack_date(df)
    df = fill_missing_categoricals(df)
    df = remove_duplicate_rows(df)
    return df
