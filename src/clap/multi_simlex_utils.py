import pandas as pd
from typing import List

from clap.const import MULTI_SIMLEX_PATH
from clap.lang_utils import simlex_id_to_iso_code


def load_multi_simlex_raw() -> pd.DataFrame:
    return pd.read_csv(MULTI_SIMLEX_PATH)


def map_and_clean_columns(columns: List[str]) -> List[str]:
    new_columns = []
    for col in columns:
        base, _ = col.rsplit(" ", 1)
        lang_code = simlex_id_to_iso_code.get(base, base)
        if lang_code == "scores":
            lang_code = "word_original"
        new_columns.append(lang_code)
    return new_columns


def get_suffix_columns(columns: List[str], suffix: str) -> List[str]:
    return [
        col
        for col in columns
        if col.endswith(suffix) and not col.startswith(("translation"))
    ]


def merge_df(df: pd.DataFrame) -> pd.DataFrame:
    suffix_1_cols = get_suffix_columns(df.columns, " 1")
    suffix_2_cols = get_suffix_columns(df.columns, " 2")
    common_cols = ["ID", "PoS"]

    df1 = df[common_cols + suffix_1_cols].copy()
    df1.columns = common_cols + map_and_clean_columns(suffix_1_cols)

    df2 = df[common_cols + suffix_2_cols].copy()
    df2.columns = common_cols + map_and_clean_columns(suffix_2_cols)

    df = pd.concat([df1, df2], ignore_index=True)
    return df


def remove_conflicting_pos(df: pd.DataFrame) -> pd.DataFrame:
    pos_conflicts = (
        df.groupby("word_original")["PoS"].nunique().reset_index().query("PoS > 1")
    )
    conflict_words = pos_conflicts["word_original"].tolist()
    before_len = len(df)
    removed_words = set()
    conflicting_rows = df[df["word_original"].isin(conflict_words)]
    for col in simlex_id_to_iso_code.values():
        for cell in conflicting_rows[col]:
            if isinstance(cell, str):
                cell = [cell]
            for word in cell:
                removed_words.add(word)
    df = df[~df["word_original"].isin(conflict_words)]
    print(
        f"Removed {before_len - len(df)} rows with conflicting PoS tags "
        f"({len(removed_words)} unique words across all langs): {conflict_words}"
    )
    df.reset_index(inplace=True, drop=True)
    return df


def remove_empty_translations(df: pd.DataFrame) -> pd.DataFrame:
    def is_effectively_empty(val):
        if isinstance(val, str):
            if len(val) == 0:
                return True
        if pd.isna(val):
            return True
        return False

    language_columns = get_language_columns(df.columns)
    non_empty_mask = (
        df[language_columns].map(lambda x: not is_effectively_empty(x)).all(axis=1)
    )
    before_len = len(df)
    df = df[non_empty_mask]
    print(f"Removed {before_len - len(df)} empty translations from the dataset")
    df.reset_index(inplace=True, drop=True)
    return df


def merge_same_pos(df):
    language_columns = get_language_columns(df.columns)
    duplicated_words = df["word_original"].value_counts()
    num_merged = sum(duplicated_words > 1)
    df = df.groupby("word_original", as_index=False).agg(
        {
            **{col: merge_lists for col in language_columns},
            "ID": "first",
            "PoS": "first",
        }
    )
    print(f"{num_merged} unique words were merged.")

    def has_merged_list(row):
        return any(
            isinstance(row[col], list) and len(row[col]) > 1 for col in language_columns
        )

    merged_rows = df[df.apply(has_merged_list, axis=1)]
    print(
        f"{len(merged_rows)} words were merged with multiple translations for at least"
        " one language."
    )
    return df


def multi_simlex_to_df(df: pd.DataFrame) -> pd.DataFrame:
    df = merge_df(df)

    # Remove empty translations
    df = remove_empty_translations(df)

    # Remove rows with conflicting PoS tags
    df = remove_conflicting_pos(df)

    # Merge translations for words with the same PoS
    df = merge_same_pos(df)

    return df


def merge_lists(series):
    if isinstance(series, str):
        series = [series]
        return series
    seen = set()
    merged = []
    for lst in series:
        if isinstance(lst, str):
            lst = [lst]
        if not isinstance(lst, list):
            raise ValueError(f"Expected a list, got {type(lst)}: {lst}")
        for item in lst:
            if item not in seen:
                seen.add(item)
                merged.append(item)
    return merged


def get_language_columns(columns: List[str]) -> List[str]:
    return [
        col for col in columns if col not in ["ID", "PoS", "scores", "word_original"]
    ]
