from collections import Counter
from unittest.mock import patch
import pytest
import pandas as pd
from clap.paired_prompts_utils import prepare_tokens_map, select_concept_pairings
from clap.prompt_tools import Prompt, build_prompt_str, update_target_prompt


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        [
            {
                "word_original": "house",
                "en": "house",
                "fr": "maison",
                "es": "casa",
                "ja": "家",
            },
            {
                "word_original": "cat",
                "en": "cat",
                "fr": "chat",
                "es": "gato",
                "ja": "猫",
            },
            {
                "word_original": "dog",
                "en": "dog",
                "fr": "chien",
                "es": "perro",
                "ja": "犬",
            },
            {
                "word_original": "car",
                "en": "car",
                "fr": "voiture",
                "es": "coche",
                "ja": "車",
            },
            {
                "word_original": "tree",
                "en": "tree",
                "fr": "arbre",
                "es": "árbol",
                "ja": "木",
            },
            {
                "word_original": "book",
                "en": "book",
                "fr": "livre",
                "es": "libro",
                "ja": "本",
            },
            {
                "word_original": "computer",
                "en": "computer",
                "fr": "ordinateur",
                "es": "ordenador",
                "ja": "コンピュータ",
            },
            {
                "word_original": "phone",
                "en": "phone",
                "fr": "téléphone",
                "es": "teléfono",
                "ja": "電話",
            },
            {
                "word_original": "table",
                "en": "table",
                "fr": "table",
                "es": "mesa",
                "ja": "テーブル",
            },
            {
                "word_original": "chair",
                "en": "chair",
                "fr": "chaise",
                "es": "silla",
                "ja": "椅子",
            },
            {
                "word_original": "window",
                "en": "window",
                "fr": "fenêtre",
                "es": "ventana",
                "ja": "窓",
            },
            {
                "word_original": "door",
                "en": "door",
                "fr": "porte",
                "es": "puerta",
                "ja": "ドア",
            },
            {
                "word_original": "pen",
                "en": "pen",
                "fr": "stylo",
                "es": "pluma",
                "ja": "ペン",
            },
            {
                "word_original": "pencil",
                "en": "pencil",
                "fr": "crayon",
                "es": "lápiz",
                "ja": "鉛筆",
            },
            {
                "word_original": "notebook",
                "en": "notebook",
                "fr": "carnet",
                "es": "cuaderno",
                "ja": "ノート",
            },
        ]
    )


def test_build_prompt_str(sample_df):
    input_lang = "es"
    target_lang = "fr"
    fs_words = ["cat", "dog"]
    word = "house"

    prompt = build_prompt_str(
        df=sample_df,
        word=word,
        fs_words=fs_words,
        input_lang=input_lang,
        target_lang=target_lang,
    )
    print(prompt)
    expected_prompt_str = (
        'Español: "gato" - Français: "chat"\n'
        'Español: "perro" - Français: "chien"\n'
        'Español: "casa" - Français: "'
    )
    assert prompt == expected_prompt_str


@pytest.fixture
def mock_df():
    return pd.DataFrame(
        [
            {
                "word_original": "house",
                "en": "house",
                "es": "casa",
                "fr": "maison",
                "scores": 0.8,
                "PoS": "noun",
                "ID": 1,
            },
            {
                "word_original": "car",
                "en": "car",
                "es": "coche",
                "fr": "voiture",
                "scores": 0.9,
                "PoS": "noun",
                "ID": 2,
            },
        ]
    )


def tok_side_effect(target_strings, latent_strings, tokenizer):
    if latent_strings.get("en") == "house":
        tokens_map = {"en": [101, 102], "fr": [201], "es": [301, 302]}
        filtered_toks = {
            lang: tokens
            for lang, tokens in tokens_map.items()
            if lang in latent_strings
        }
        return [], filtered_toks
    elif latent_strings.get("en") == "car":
        tokens_map = {"en": [103], "fr": [202, 203], "es": [303]}
        filtered_toks = {
            lang: tokens
            for lang, tokens in tokens_map.items()
            if lang in latent_strings
        }
        return [], filtered_toks
    else:
        return [], {}


@patch("clap.prompt_tools.Prompt.get_target_latent_tokens")
def test_prepare_tokens_map_basic(mock_get_tokens, mock_df):
    mock_get_tokens.side_effect = tok_side_effect
    result = prepare_tokens_map(mock_df, tokenizer=None)

    assert "house" in result
    assert "car" in result
    assert result["house"] == {101, 102, 201, 301, 302}
    assert result["car"] == {103, 202, 203, 303}
    assert mock_get_tokens.call_count == 2


@patch("clap.prompt_tools.Prompt.get_target_latent_tokens")
def test_prepare_tokens_map_skip(mock_get_tokens, mock_df):
    mock_get_tokens.side_effect = tok_side_effect
    result = prepare_tokens_map(
        mock_df,
        tokenizer=None,
        skip_langs={"fr"},
    )

    assert "house" in result
    assert "car" in result
    assert result["house"] == {101, 102, 301, 302}
    assert result["car"] == {103, 303}


@pytest.mark.parametrize("strictly_balanced", [True, False])
def test_directional_usage_balancing(strictly_balanced):
    concept_map = {
        "A": ["B", "C", "D"],
        "B": ["A", "C", "D"],
        "C": ["A", "B"],
        "D": ["A", "B"],
    }
    num_pairs = 8

    pairs = select_concept_pairings(
        concept_map, num_pairs, strictly_balanced=strictly_balanced
    )

    usage_left = Counter()
    usage_right = Counter()
    for pair in pairs:
        left, right = pair.split("_")
        usage_left[left] += 1
        usage_right[right] += 1

    concepts = set(concept_map.keys())
    expected = num_pairs / len(concepts)

    if strictly_balanced:
        assert len(pairs) <= num_pairs, "Too many pairs returned"
    else:
        assert len(pairs) == num_pairs, "Incorrect number of pairs returned"

    for concept in concepts:
        left = usage_left[concept]
        right = usage_right[concept]

        if strictly_balanced and (concept in usage_left or concept in usage_right):
            assert abs(left - expected) <= 1, (
                f"Left usage imbalance for {concept}: {left} vs {expected}"
            )
            assert abs(right - expected) <= 1, (
                f"Right usage imbalance for {concept}: {right} vs {expected}"
            )
        else:
            assert left <= expected + 1, f"Left overused: {concept} used {left} times"
            assert right <= expected + 1, (
                f"Right overused: {concept} used {right} times"
            )


def test_update_target_prompt():
    src_p = [
        Prompt(
            prompt="dummy src",
            word_original="house",
            target_tokens=[101, 102],
            target_strings=["maison"],
            latent_tokens={"fr": [303], "es": [301, 302]},
            latent_strings={"fr": ["maison"], "es": ["casa"]},
        )
    ]
    targ_p = Prompt(
        prompt="dummy tgt",
        word_original="car",
        target_strings=["voiture"],
        target_tokens=[201],
        latent_tokens={"fr": [401], "es": [501, 502]},
        latent_strings={"fr": ["voiture"], "es": ["coche"]},
    )
    target_lang = "fr"
    updated_prompt = update_target_prompt(src_p, targ_p, target_lang)
    assert updated_prompt.word_original == "car"
    assert updated_prompt.target_tokens == [201]
    assert updated_prompt.latent_tokens == {
        "src_es": [301, 302],
        "src_fr": [303],
        "tgt_es": [501, 502],
    }
    assert updated_prompt.latent_strings == {
        "src_es": ["casa"],
        "src_fr": ["maison"],
        "tgt_es": ["coche"],
    }
