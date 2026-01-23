import json
import os
from pathlib import Path
import sys
import numpy as np
from tqdm import tqdm
from clap.config import load_config_from_file
from clap.exp_tools import run_prompts, set_up_exp
from clap.utils import is_empty_json_file
from clap.interventions import object_lens
from clap.lang_utils import get_ellm_langs
from clap.nnsight_utils import collect_activations_batched, get_num_layers
from clap.paired_prompts_utils import load_prompts
from clap.prompt_tools import get_obj_id
from time import time

EXP_NAME = "obj_patch_translation"

OBJ_PATCH_KEY_SET = {
    "seen",
    "en_en",
    "tgt",
    "tgt_tgt",
    "unpatched",
    "self",
}


def get_target_pair(target_langs=None):
    delim = "_" if "_" in target_langs else "-"
    target_input, target_output = target_langs.split(delim)
    return target_input, target_output


def object_patching(nn_model, prompt_batch, idx, source_prompts_str):
    offset = object_patching.offset
    batch_size = len(prompt_batch)
    source_prompt_batch = source_prompts_str[offset : offset + batch_size]
    hiddens = collect_activations_batched(
        nn_model, source_prompt_batch.flatten(), batch_size=batch_size
    )
    hiddens = hiddens.transpose(0, 1)  # (all_prompts, layer, hidden_size)
    hiddens = hiddens.reshape(
        batch_size, source_prompt_batch.shape[1], get_num_layers(nn_model), -1
    ).mean(dim=1)  # (batch_size, num_layers, hidden_size)
    hiddens = hiddens.transpose(0, 1)  # (num_layers, batch_size, hidden_size)
    return object_lens(
        nn_model,
        prompt_batch,
        idx,
        hiddens=hiddens,
    )


def run_object_patching(
    output_lang,
    source_prompts,
    target_prompts,
    nn_model,
    save_path,
    exp_id,
    batch_size,
):
    """
    Collect activation at the last token of the prompt and generate a mean
    latent representation for each layer

    For each layer `j`: Run the target prompts which are translations from
    the input_lang to the target_lang. During the forward pass, patch at the
    last token of the concept to be translated with the mean latent
    representation of the source prompts from `j` to the last layer.
    """
    source_prompts_str = np.array(
        [['"'.join(p.prompt.split('"')[:-2]) for p in ps] for ps in source_prompts]
    )

    idx = get_obj_id(target_prompts[0].prompt, nn_model.tokenizer)
    object_patching.offset = 0
    target_probs, latent_probs, generated_answers = run_prompts(
        nn_model,
        target_prompts,
        batch_size=batch_size,
        get_probs=object_patching,
        get_probs_kwargs=dict(
            idx=idx,
            source_prompts_str=source_prompts_str,
        ),
        tqdm=tqdm,
    )

    json_dic = {output_lang: target_probs.tolist()}
    for label, probs in latent_probs.items():
        json_dic[label] = probs.tolist()

    json_dic["generated_answers"] = generated_answers

    output_path = Path(save_path)
    json_file = output_path / f"{exp_id}.json"
    with open(json_file, "w") as f:
        json.dump(json_dic, f, indent=4)


def get_settings(target_input: str, target_output: str, lang_pairs: list[tuple]):
    source_lang_pairs_tgt = [f"{target_input}_{target_output}"]
    source_lang_pairs_seen = []
    source_lang_pairs_en_en = ["en_en"]
    source_lang_pairs_tgt_tgt = [f"{target_output}_{target_output}"]
    ellm_langs_extended = get_ellm_langs(extended=True)
    for p in lang_pairs:
        pair_str = f"{'_'.join(p)}"
        # Check if pair should be included at all
        if target_input in p or target_output in p:
            continue

        if "en" in p:
            continue

        # Filter for pairs seen in ellm_langs
        if p[0] in ellm_langs_extended and p[1] in ellm_langs_extended:
            source_lang_pairs_seen.append(pair_str)
    settings = {
        "tgt": source_lang_pairs_tgt,
        "tgt_tgt": source_lang_pairs_tgt,
        "seen": source_lang_pairs_seen,
        "en_en": source_lang_pairs_en_en,
        "self": source_lang_pairs_tgt_tgt,
    }
    if target_output == "en":
        # en_en and self are the same in this case
        settings.pop("self")
    return settings


def get_settings_to_prompts(
    target_input: str,
    target_output: str,
    lang_pairs: list[tuple[str]],
    sources: dict[str, list],
    targets: dict[str, list],
):
    keys_to_settings = get_settings(target_input, target_output, lang_pairs)
    settings_to_prompts = {}
    for key, setting in keys_to_settings.items():
        if len(setting) == 0:
            print(f"Skipping {key} as it has no language pairs")
            continue
        print(f"Setting {key} has {len(setting)} language pairs")
        prompts = np.array(
            [source for lang_key, source in sources.items() if lang_key in setting]
        )
        if len(prompts) == 0:
            print(f"Skipping {key} as it has no source prompts")
            sys.exit(
                f"No source prompts for {key}, cannot proceed (most likely "
                f"for setting `self` and you need to run `prepare_prompts` "
                f"with the --extend_lang flag first)."
            )
        settings_to_prompts[key] = prompts.transpose(1, 0)

        if key == "tgt_tgt":
            # tgt tgt is a special case where we want to use the target prompts
            # in the target language
            settings_to_prompts[key] = np.array(
                [target for lang_key, target in targets.items() if lang_key in setting]
            ).transpose(1, 0)
    # expected shape: (num_prompts, len(setting))
    return keys_to_settings, settings_to_prompts


def get_sources_targets(prompts_cache):
    if os.path.exists(prompts_cache):
        sources, targets = load_prompts(prompts_cache)
    else:
        raise ValueError(
            f"Prompts cache not found at {prompts_cache}, run `prepare_prompts` first."
        )
    return sources, targets


def get_prompts_for_unpatched(target_input: str, target_output: str, sources, targets):
    # Source prompts in target pair language, measure Source Concept Prob.
    # Target prompts in target pair language, measure Source Concept Prob.
    lang_key = f"{target_input}_{target_output}"
    # expected shape: (num_prompts, 1)
    return {
        "src_unpatched": np.array(sources[lang_key]).reshape(-1, 1),
        "tgt_unpatched": np.array(targets[lang_key]).reshape(-1, 1),
    }


def run_all_settings(
    config,
    target_input,
    target_output,
    settings_to_keys,
    settings_to_prompts,
    targets,
    nn_model,
    dry_run,
):
    for setting, source_prompts in settings_to_prompts.items():
        keys = settings_to_keys[setting]
        source_lang_pairs = tuple(tuple(p.split("_")) for p in keys)

        config.obj_patching_langs = source_lang_pairs, target_input, target_output
        config.exp_id = f"{int(time())}_{setting}"
        config.save_path = None
        config.set_save_path()
        save_path = config.save_path

        if dry_run:
            print(f"[DRY RUN] {config.model}, setting {setting}")
            print(f"[DRY RUN] Save path: {save_path}")
            continue

        config = set_up_exp(config)
        print(f"Running patched: {setting} -> {target_input}-{target_output}")

        target_prompts = targets[f"{target_input}_{target_output}"]
        assert len(source_prompts) > 0
        assert len(target_prompts) > 0

        run_object_patching(
            target_output,
            source_prompts,
            target_prompts,
            nn_model,
            config.save_path,
            config.exp_id,
            config.batch_size,
        )


def run_unpatched(
    config, target_input, target_output, unpatched_prompts, nn_model, dry_run
):
    config.obj_patching_langs = "unpatched"
    config.unpatched_langs = {(f"{target_input}_{target_output}"): ["source", "target"]}
    config.exp_id = None
    config.save_path = None
    config.set_save_path()
    save_path = config.save_path

    if dry_run:
        print(f"[DRY RUN] Unpatched run for {config.model}")
        print(f"[DRY RUN] Save path: {save_path}")
        return

    config = set_up_exp(config)
    print(f"Running unpatched: {target_input}-{target_output}")

    json_dic = {}
    for key, prompts in unpatched_prompts.items():
        prompts = np.array(prompts)
        prompts = prompts.flatten()
        prompt_probs, latent_probs, generated_answers = run_prompts(
            nn_model,
            prompts,
            batch_size=config.batch_size,
            get_probs=object_lens,
            get_probs_kwargs=dict(
                idx=-1,
                patch=False,
            ),
            tqdm=tqdm,
        )
        # maybe unnecessary because we don't expect multiple source pairs
        prompt_probs = prompt_probs.squeeze().reshape(len(prompts), -1)
        json_dic[f"{key} {target_output}"] = prompt_probs.tolist()
        for label, probs in latent_probs.items():
            json_dic[f"{key} {label}"] = probs.squeeze().tolist()
        json_dic[f"{key} generated_answers"] = generated_answers
    output_path = Path(save_path)
    json_file = output_path / f"{config.exp_id}.json"

    with open(json_file, "w") as f:
        json.dump(json_dic, f, indent=4)


def already_run(
    save_path,
    target_input,
    target_output,
    source_lang_pairs=None,
    prompts: dict[str, list[str]] | None = None,
    tgt_tgt=False,
    self_setting=False,
):
    parent_path = Path(save_path).parent
    if not parent_path.exists():
        return False

    for subdir in parent_path.iterdir():
        if not subdir.is_dir():
            continue

        args_path = subdir / "exp_args.json"
        result_path = subdir / f"{subdir.name}.json"

        if not (args_path.exists() and result_path.exists()):
            continue

        if is_empty_json_file(result_path):
            continue

        seen_config = load_config_from_file(args_path)
        if source_lang_pairs is None:
            if seen_config.obj_patching_langs != "unpatched":
                continue

            if seen_config.unpatched_langs is None:
                continue

            if f"{target_input}_{target_output}" not in seen_config.unpatched_langs:
                continue

            if all(
                p in seen_config.unpatched_langs[f"{target_input}_{target_output}"]
                for p in prompts[f"{target_input}_{target_output}"]
            ):
                return True
        else:
            if seen_config.obj_patching_langs == "unpatched":
                continue
            seen_sources, seen_target_input, seen_target_output = (
                seen_config.obj_patching_langs
            )

            if (
                len(seen_sources) == len(source_lang_pairs)
                and set(seen_sources) == set(source_lang_pairs)
                and seen_target_input == target_input
                and seen_target_output == target_output
            ):
                if tgt_tgt and "tgt_tgt" not in str(result_path):
                    continue
                if self_setting and "self" not in str(result_path):
                    continue
                return True
    return False
