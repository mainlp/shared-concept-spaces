import json
import math
from unittest import mock
from unittest.mock import MagicMock, patch

import pytest

from clap.exp_tools import load_model, set_seed
from clap.lang_utils import (
    get_ellm_langs,
    get_multi_simlex_langs,
    get_permutations,
)
from clap.obj_patching_utils import (
    already_run,
    get_settings,
    get_settings_to_prompts,
    get_sources_targets,
)
from clap.prompt_tools import process_tokens_with_tokenization
from clap.utils import ulist


def expected_permutation_cnt(n: int, r: int) -> int:
    if n == 0:
        return 0
    if n < r:
        return 0
    return math.factorial(n) // math.factorial(n - r)


@pytest.mark.parametrize(
    "target_input, target_output, selected_pairs",
    [
        ("sw", "cy", "unseen_unseen"),  # both unseen
        ("fr", "es", "seen_seen"),  # both seen
        ("ru", "cy", "seen_unseen"),  # both seen
        ("cy", "pl", "unseen_seen"),  # both seen
        ("zh", "en", "seen_seen"),  # both seen
    ],
)
def test_get_settings(target_input, target_output, selected_pairs):
    skip_langs = {"ar", "he"}
    lang_pairs = get_permutations(skip_langs=skip_langs)
    multi_simlex_lang_cnt = 13 - len(skip_langs)
    multi_simlex_langs = get_multi_simlex_langs()
    assert len(multi_simlex_langs) == multi_simlex_lang_cnt
    ellm_lang_set = get_ellm_langs()
    multi_simlex_lang_set = set(multi_simlex_langs)
    seen_langs = ellm_lang_set.intersection(multi_simlex_lang_set)

    keys_to_settings = get_settings(target_input, target_output, lang_pairs)
    print(f"{selected_pairs.upper()} -- {target_input}_{target_output}")

    for key, setting in keys_to_settings.items():
        if key not in {"tgt", "tgt_tgt", "self"}:
            for pair in setting:
                if key == "en_en" and target_input == "en":
                    continue
                if key == "en_en" and target_output == "en":
                    continue
                assert target_input not in pair and target_output not in pair

        if key == "en_en":
            assert len(setting) == 1, setting
            pair_tuple = tuple(setting[0].split("_"))
            assert pair_tuple == ("en", "en"), setting
        elif key == "self":
            assert len(setting) == 1, setting
            pair_tuple = tuple(setting[0].split("_"))
            assert pair_tuple == (target_output, target_output), setting
        elif key in {"tgt", "tgt_tgt"}:
            assert len(setting) == 1, setting
            pair_tuple = tuple(setting[0].split("_"))
            assert pair_tuple == (target_input, target_output), setting
        elif key == "seen":
            for pair in setting:
                pair_tuple = tuple(pair.split("_"))
                assert pair_tuple[0] in seen_langs
                assert pair_tuple[1] in seen_langs
                assert pair_tuple[0] != "en"
                assert pair_tuple[1] != "en"
            expected_cnt = expected_permutation_cnt(
                len(seen_langs.difference({target_input, target_output, "en"})), 2
            )
            assert len(setting) == expected_cnt
        else:
            raise AssertionError("Unknown key")
        print(
            f"Setting {key} has {len(setting)} language pairs for "
            f"{target_input} and {target_output}"
        )


@pytest.fixture
def setup_experiment(tmp_path):
    # Set up expected structure:
    # experiments/
    # └── run1/
    #     ├── exp_args.json
    #     └── run1.json
    parent = tmp_path / "experiments"
    run_dir = parent / "run1"
    run_dir.mkdir(parents=True)

    args_path = run_dir / "exp_args.json"
    result_path = run_dir / "run1.json"
    args_path.write_text("{}")
    result_path.write_text('{"status": "done"}')

    save_path = parent / "dummy_save_file.txt"

    return {
        "save_path": str(save_path),
        "args_path": args_path,
        "result_path": result_path,
        "run_dir": run_dir,
    }


def test_already_run_true(setup_experiment):
    source_lang_pairs = [("en", "fr"), ("de", "es")]
    target_input = "it"
    target_output = "pt"

    mock_config = MagicMock()
    mock_config.obj_patching_langs = [source_lang_pairs, target_input, target_output]

    with (
        patch(
            "clap.obj_patching_utils.is_empty_json_file",
            return_value=False,
        ),
        patch(
            "clap.obj_patching_utils.load_config_from_file",
            return_value=mock_config,
        ),
    ):
        result = already_run(
            save_path=setup_experiment["save_path"],
            source_lang_pairs=source_lang_pairs,
            target_input=target_input,
            target_output=target_output,
        )
        assert result is True


def test_already_run_false_due_to_mismatch(setup_experiment):
    source_lang_pairs = [("en", "fr")]
    target_input = "it"
    target_output = "pt"

    # Different config
    mock_config = MagicMock()
    mock_config.obj_patching_langs = [[("en", "de")], "it", "pt"]

    with (
        patch(
            "clap.obj_patching_utils.is_empty_json_file",
            return_value=False,
        ),
        patch(
            "clap.obj_patching_utils.load_config_from_file",
            return_value=mock_config,
        ),
    ):
        result = already_run(
            save_path=setup_experiment["save_path"],
            source_lang_pairs=source_lang_pairs,
            target_input=target_input,
            target_output=target_output,
        )
        assert result is False


@pytest.fixture
def mock_config_unpatched():
    mock_config = mock.MagicMock()
    mock_config.obj_patching_langs = "unpatched"
    mock_config.unpatched_langs = {"en_fr": ["source", "target"]}
    return mock_config


@pytest.fixture
def setup_unpatched_files(tmp_path, mock_config_unpatched):
    # experiments/
    # └── exp1/
    #     ├── exp_args.json
    #     └── exp1.json
    parent = tmp_path / "experiments"
    exp_dir = parent / "exp1"
    exp_dir.mkdir(parents=True)

    args_path = exp_dir / "exp_args.json"
    result_path = exp_dir / "exp1.json"
    args_path.write_text("dummy config")
    result_path.write_text('{"result": "not empty"}')

    save_path = parent / "any_save_file.txt"

    with (
        mock.patch(
            "clap.obj_patching_utils.load_config_from_file",
            return_value=mock_config_unpatched,
        ),
        mock.patch(
            "clap.obj_patching_utils.is_empty_json_file",
            return_value=False,
        ),
    ):
        yield save_path, "en", "fr", None, {"en_fr": ["source", "target"]}


def test_already_run_unpatched_true(setup_unpatched_files):
    save_path, target_input, target_output, source_lang_pairs, prompts = (
        setup_unpatched_files
    )
    result = already_run(
        save_path=save_path,
        target_input=target_input,
        target_output=target_output,
        source_lang_pairs=source_lang_pairs,
        prompts=prompts,
    )
    assert result is True


def test_already_run_unpatched_false_prompt_mismatch(
    setup_unpatched_files, mock_config_unpatched
):
    tmp_path, target_input, target_output, source_lang_pairs, _ = setup_unpatched_files
    save_path = tmp_path / "some" / "nested" / "exp"
    mock_config_unpatched.unpatched_langs = {"en_fr": ["target"]}  # missing "source"

    with mock.patch(
        "clap.obj_patching_utils.load_config_from_file",
        return_value=mock_config_unpatched,
    ):
        result = already_run(
            save_path,
            target_input,
            target_output,
            source_lang_pairs,
            ["source", "target"],
        )
        assert result is False


def test_already_run_unpatched_false_lang_pair_mismatch(
    setup_unpatched_files, mock_config_unpatched
):
    tmp_path, _, _, source_lang_pairs, prompts = setup_unpatched_files
    save_path = tmp_path / "some" / "nested" / "exp"
    mock_config_unpatched.unpatched_langs = {"de_fr": ["source", "target"]}

    with mock.patch(
        "clap.obj_patching_utils.load_config_from_file",
        return_value=mock_config_unpatched,
    ):
        result = already_run(save_path, "en", "fr", source_lang_pairs, prompts)
        assert result is False


@pytest.fixture
def unpatched_prompts():
    return {"tgt_unpatched": ["source_prompt", "target_prompt"]}


@pytest.fixture
def nn_model():
    return load_model("TinyLlama/TinyLlama_v1.1")


@pytest.fixture
def setup_tgt_tgt_case(tmp_path):
    # Create directory for experiment
    exp_dir = tmp_path / "run1"
    exp_dir.mkdir()

    # Dummy args
    args_path = exp_dir / "exp_args.json"
    args_path.write_text("{}")  # Content mocked

    # Dummy result with tgt_tgt key
    result_path = exp_dir / "run1.json"
    result_content = {"some_key": 123, "tgt_tgt source_fr": [0.1, 0.2]}
    result_path.write_text(json.dumps(result_content))

    save_path = tmp_path / "dummy_save.json"

    return {
        "save_path": save_path,
        "args_path": args_path,
        "result_path": result_path,
        "source_lang_pairs": [("en", "de")],
        "target_input": "en",
        "target_output": "fr",
    }


def test_already_run_tgt_tgt_match(setup_tgt_tgt_case):
    case = setup_tgt_tgt_case

    # Mock config returned from exp_args.json
    mock_config = mock.MagicMock()
    mock_config.obj_patching_langs = (
        case["source_lang_pairs"],
        case["target_input"],
        case["target_output"],
    )

    with (
        mock.patch(
            "clap.obj_patching_utils.load_config_from_file",
            return_value=mock_config,
        ),
        mock.patch(
            "clap.obj_patching_utils.is_empty_json_file",
            return_value=False,
        ),
    ):
        result = already_run(
            save_path=case["save_path"],
            target_input=case["target_input"],
            target_output=case["target_output"],
            source_lang_pairs=case["source_lang_pairs"],
            tgt_tgt=True,
        )
        assert result is True


@pytest.fixture
def ellm_nn_model():
    return load_model("utter-project/EuroLLM-1.7B")


def test_get_settings_self_vs_en():
    skip_langs = {"ar", "he"}
    lang_pairs = get_permutations(skip_langs=skip_langs)
    s1 = get_settings("fr", "de", lang_pairs)
    assert "self" in s1
    assert s1["self"] == ["de_de"]
    assert s1["en_en"] == ["en_en"]

    s2 = get_settings("es", "en", lang_pairs)
    assert "self" not in s2
    assert s2["en_en"] == ["en_en"]


def test_assumptions(ellm_nn_model):
    set_seed(42)
    skip_langs = {"ar", "he"}
    sources, targets = get_sources_targets(
        "data/prompts_cache/EuroLLM-1.7B_42_5_16_prompts.pkl"
    )
    lang_pairs = get_permutations(skip_langs=skip_langs)
    target_input, target_output = "es", "en"
    settings_to_keys, settings_to_prompts = get_settings_to_prompts(
        target_input, target_output, lang_pairs, sources, targets
    )
    assert settings_to_keys.keys() == settings_to_prompts.keys()
    assert set(settings_to_keys.keys()) == {
        "tgt",
        "tgt_tgt",
        "seen",
        "en_en",
    }

    source_words = [pp for p in sources["es_en"] for pp in p.target_strings]
    tokenizer = ellm_nn_model.tokenizer
    source_tokens = {
        w: process_tokens_with_tokenization(w, tokenizer) for w in source_words
    }
    source_tokens = {}
    for p in sources["es_en"]:
        source_tokens[p.target_strings[0]] = ulist(
            [
                t
                for w in p.target_strings
                for t in process_tokens_with_tokenization(w, tokenizer)
            ]
        )

    for source_tok, source_prompt, target_prompt in zip(
        source_tokens.values(), sources["es_en"], targets["es_en"], strict=True
    ):
        assert source_tok == source_prompt.target_tokens
        assert source_tok == target_prompt.latent_tokens["src_en"]

    assert len(target_prompt.latent_tokens) == 21
    assert len(target_prompt.latent_strings) == 21
