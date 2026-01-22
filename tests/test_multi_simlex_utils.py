import pandas as pd
import pandas.testing as pdt

from clap.multi_simlex_utils import (
    load_multi_simlex_raw,
    get_language_columns,
    merge_df,
    multi_simlex_to_df,
    remove_conflicting_pos,
    simlex_id_to_iso_code,
    merge_lists,
    map_and_clean_columns,
    get_suffix_columns,
)


def test_load_multi_simlex_raw():
    df = load_multi_simlex_raw()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "ID" in df.columns
    assert "PoS" in df.columns
    assert "scores 1" in df.columns
    assert "translation 1" in df.columns
    assert "scores 2" in df.columns
    assert "translation 2" in df.columns
    # 13 languages + 2 translations + 2 scores + ID + PoS
    assert len(df.columns) == 13 * 2 + 2 + 2 + 1 + 1
    assert not df.duplicated(subset=["ID"]).any()


def test_get_language_columns():
    simlex_processed_columns = {c for c in simlex_id_to_iso_code.values()}
    simlex_processed_columns.update({"ID", "PoS", "scores", "word_original"})
    language_columns = get_language_columns(simlex_processed_columns)
    assert len(language_columns) == 13


def test_merge_lists():
    list_of_lists = [["a", "b"], ["b", "c"], ["a"], ["d", "c"]]
    merged = merge_lists(list_of_lists)
    assert merged == ["a", "b", "c", "d"]


def test_map_and_clean_columns():
    uncleaned_columns = [
        "ID",
        "ENG 1",
        "ENG 2",
        "PoS",
        "ARA 1",
        "ARA 2",
        "CMN 1",
        "CMN 2",
        "CYM 1",
        "CYM 2",
        "EST 1",
        "EST 2",
        "FIN 1",
        "FIN 2",
        "FRA 1",
        "FRA 2",
        "HEB 1",
        "HEB 2",
        "POL 1",
        "POL 2",
        "RUS 1",
        "RUS 2",
        "SPA 1",
        "SPA 2",
        "SWA 1",
        "SWA 2",
        "YUE 1",
        "YUE 2",
        "scores 1",
        "scores 2",
        "translation 1",
        "translation 2",
    ]
    unclean_1 = get_suffix_columns(uncleaned_columns, " 1")
    for col in unclean_1:
        assert col.endswith(" 1") or col in ["ID", "PoS"]
    unclean_2 = get_suffix_columns(uncleaned_columns, " 2")
    for col in unclean_2:
        assert col.endswith(" 2") or col in ["ID", "PoS"]
    cleaned_columns_1 = map_and_clean_columns(unclean_1)
    cleaned_columns_2 = map_and_clean_columns(unclean_2)
    assert len(unclean_1) == len(unclean_2)
    assert len(cleaned_columns_1) == 13 + 1
    assert len(cleaned_columns_2) == 13 + 1
    assert all(
        c in simlex_id_to_iso_code.values() or c == "word_original"
        for c in cleaned_columns_1
    )
    assert all(
        c in simlex_id_to_iso_code.values() or c == "word_original"
        for c in cleaned_columns_2
    )


def test_merge_df():
    df = load_multi_simlex_raw().head(20)
    merged_df = merge_df(df)
    assert isinstance(merged_df, pd.DataFrame)
    assert (
        len(merged_df.columns) == 1 + 1 + 1 + 13
    )  # ID, PoS, word_original, 13 languages
    assert "ID" in merged_df.columns
    assert "PoS" in merged_df.columns
    assert all(
        col in simlex_id_to_iso_code.values()
        for col in merged_df.columns
        if col not in ["ID", "PoS", "word_original"]
    )
    assert len(merged_df) == len(df) * 2


def test_remove_conflicting_pos():
    df = pd.DataFrame(
        {
            "ID": ["1", "2", "3", "4", "5"],
            "PoS": ["nouns", "verbs", "nouns", "adjectives", "adjectives"],
            "word_original": ["plant", "plant", "horse", "pretty", "pretty"],
            "en": ["plant", "plant", "horse", "pretty", "pretty"],
            "es": ["planta", "crecer", "caballo", "bonito", "hermoso"],
            "ar": ["نبات", "ينمو", "حصان", "جميل", "جميلة"],
            "zh": ["植物", "生长", "马", "漂亮", "美丽"],
            "yue": ["植物", "生长", "马", "漂亮", "美丽"],
            "cy": ["planh", "tyfu", "ceffyl", "prydferth", "prydferth"],
            "fi": ["kasvi", "kasvaa", "hevonen", "kaunis", "kaunis"],
            "et": ["taim", "kasvama", "hobune", "ilus", "ilus"],
            "fr": ["plante", "grandir", "cheval", "joli", "beau"],
            "pl": ["roślina", "rosnąć", "koń", "ładny", "piękny"],
            "ru": ["растение", "расти", "лошадь", "красивый", "красивый"],
            "sw": ["mimea", "kukua", "farasi", "mzuri", "mzuri"],
            "he": ["צמח", "לגדול", "סוס", "יפה", "יפה"],
        }
    )
    df = remove_conflicting_pos(df)
    assert len(df) == 3
    assert "plant" not in df["word_original"].values
    assert df["word_original"].tolist() == ["horse", "pretty", "pretty"]
    expected_df = pd.DataFrame(
        {
            "ID": ["3", "4", "5"],
            "PoS": ["nouns", "adjectives", "adjectives"],
            "word_original": ["horse", "pretty", "pretty"],
            "en": ["horse", "pretty", "pretty"],
            "es": ["caballo", "bonito", "hermoso"],
            "ar": ["حصان", "جميل", "جميلة"],
            "zh": ["马", "漂亮", "美丽"],
            "yue": ["马", "漂亮", "美丽"],
            "cy": ["ceffyl", "prydferth", "prydferth"],
            "fi": ["hevonen", "kaunis", "kaunis"],
            "et": ["hobune", "ilus", "ilus"],
            "fr": ["cheval", "joli", "beau"],
            "pl": ["koń", "ładny", "piękny"],
            "ru": ["лошадь", "красивый", "красивый"],
            "sw": ["farasi", "mzuri", "mzuri"],
            "he": ["סוס", "יפה", "יפה"],
        }
    ).reset_index(drop=True)
    pdt.assert_frame_equal(df, expected_df)


def test_multi_simlex_to_df():
    # multi_simlex is ordered by word class, these will all be nouns
    df = load_multi_simlex_raw().head(10)
    df = multi_simlex_to_df(df)
    assert isinstance(df, pd.DataFrame)
    assert len(df.columns) == 1 + 1 + 1 + 13  # ID, PoS, word_original, 13 languages
    assert "ID" in df.columns
    assert "PoS" in df.columns
    assert "word_original" in df.columns
    assert len(df) == 10 * 2
    for col in df.columns:
        if col not in ["ID", "PoS", "word_original"]:
            assert col in simlex_id_to_iso_code.values()
            assert all(isinstance(x, list) for x in df[col]), (
                f"Not all entries in {col} are lists"
            )


def combine_rows(row1, row2, df, new_id=None):
    new_row = pd.Series(index=df.columns, dtype="object")
    for col in df.columns:
        if col.endswith(" 1"):
            new_row[col] = row1[col]
        elif col.endswith(" 2"):
            new_row[col] = row2[col]
        else:
            if col == "ID":
                new_row[col] = new_id if new_id is not None else 9999
            elif col == "PoS":
                new_row[col] = row1[col]
            else:
                new_row[col] = ""
    return new_row


def test_multi_simlex_to_df_duplicate_pos():
    # multi_simlex is ordered by word class, these will all be nouns
    df = load_multi_simlex_raw().head(10)
    new_row = combine_rows(df.iloc[0].copy(), df.iloc[2].copy(), df, new_id="9999")
    for col in new_row.index:
        if col not in [
            "ID",
            "PoS",
            "word_original",
            "translation 1",
            "translation 2",
            "scores 1",
            "scores 2",
        ]:
            # pretend we have synonyms
            new_row[col] = f"{new_row[col]}_synonym"
    df = pd.concat([df, new_row.to_frame().T], ignore_index=True)
    df = multi_simlex_to_df(df)
    assert isinstance(df, pd.DataFrame)
    assert len(df.columns) == 1 + 1 + 1 + 13  # ID, PoS, word_original, 13 languages
    assert "ID" in df.columns
    assert "PoS" in df.columns
    assert "word_original" in df.columns
    assert len(df) == 10 * 2  # original 10 rows, no duplicates
    for col in df.columns:
        if col not in ["ID", "PoS", "word_original"]:
            assert col in simlex_id_to_iso_code.values()
            assert all(isinstance(x, list) for x in df[col]), (
                f"Not all entries in {col} are lists"
            )

    w1 = new_row["translation 1"]
    w2 = new_row["translation 2"]

    subset = df[df["word_original"].isin([w1, w2])]
    for _, row in subset.iterrows():
        for lang in simlex_id_to_iso_code.values():
            assert isinstance(row[lang], list), f"{lang} entry is not a list"
            assert len(row[lang]) == 2, (
                f"{lang} entry does not contain 2 items: {row[lang]}"
            )


def test_multi_simlex_to_df_conflicting_pos():
    # multi_simlex is ordered by word class, these will all be nouns
    df = load_multi_simlex_raw().head(10)
    new_row = combine_rows(df.iloc[0].copy(), df.iloc[2].copy(), df, new_id="9999")
    new_row["PoS"] = "adjectives"  # Force a conflict in PoS
    df = pd.concat([df, new_row.to_frame().T], ignore_index=True)
    df = multi_simlex_to_df(df)
    assert isinstance(df, pd.DataFrame)
    assert len(df.columns) == 1 + 1 + 1 + 13  # ID, PoS, word_original, 13 languages
    assert "ID" in df.columns
    assert "PoS" in df.columns
    assert "word_original" in df.columns
    assert len(df) == 9 * 2  # original 10 rows, no duplicates
    for col in df.columns:
        if col not in ["ID", "PoS", "word_original"]:
            assert col in simlex_id_to_iso_code.values()
            assert all(isinstance(x, list) for x in df[col]), (
                f"Not all entries in {col} are lists"
            )
