from argparse import ArgumentParser
import os
import torch
import gc
from time import time

import clap.load_env  # noqa: F401
from clap.config import load_config_from_file
from clap.exp_tools import load_model, set_seed
from clap.lang_utils import get_permutations
from clap.obj_patching_utils import (
    get_prompts_for_unpatched,
    get_settings_to_prompts,
    get_sources_targets,
    get_target_pair,
    run_all_settings,
    run_unpatched,
    already_run,
)
from clap.hf_utils import (
    register_revision_user,
    unregister_revision_user,
    delete_hf_revision_cache,
    get_sorted_steps,
    get_policy_for_model,
)


def main(
    config_path: str,
    slug: str,
    cuda_device: int | str | None = None,
    target_langs: str | None = None,
    dry_run: bool = False,
) -> None:
    config = load_config_from_file(config_path)
    set_seed(config.seed)

    if cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

    skip_langs = {"ar", "he"}
    sources, targets = get_sources_targets(config.prompts_cache)
    lang_pairs = get_permutations(skip_langs=skip_langs)
    target_input, target_output = get_target_pair(target_langs)
    slug = f"{slug}/{target_input}_{target_output}"

    settings_to_keys, settings_to_prompts = get_settings_to_prompts(
        target_input, target_output, lang_pairs, sources, targets
    )
    unpatched_prompts = get_prompts_for_unpatched(
        target_input, target_output, sources, targets
    )
    models = []
    sorted_step_keys, step_to_path = get_sorted_steps(config.model)
    should_run = get_policy_for_model(config.model)
    if config.model == "utter-project/EuroLLM-1.7B":
        models = [
            step_to_path[step_num]
            for i, step_num in enumerate(sorted_step_keys)
            if should_run(i, len(sorted_step_keys))
        ]
        revisions = [None] * len(models)
    else:
        revisions = [
            step_to_path[step_num]
            for i, step_num in enumerate(sorted_step_keys)
            if should_run(i, len(sorted_step_keys))
        ]
        models = [config.model] * len(revisions)

    for model, revision in zip(models, revisions, strict=True):
        lock_dir = token = None
        try:
            lock_dir, token = register_revision_user(model, revision or "none")

            config.model = model
            config.revision = revision
            config.slug = slug

            to_run_settings = []
            for setting, source_prompts in settings_to_prompts.items():
                keys = settings_to_keys[setting]
                source_lang_pairs = tuple(tuple(p.split("_")) for p in keys)

                config.obj_patching_langs = (
                    source_lang_pairs,
                    target_input,
                    target_output,
                )
                config.exp_id = f"{int(time())}_{setting}"
                config.save_path = None
                config.set_save_path()
                save_path = config.save_path

                if already_run(
                    save_path,
                    target_input,
                    target_output,
                    source_lang_pairs,
                    tgt_tgt=setting == "tgt_tgt",
                ):
                    print(
                        f"Already run: ({setting}) {source_lang_pairs} -> {target_input}-{target_output}"
                    )
                    continue

                to_run_settings.append((setting, source_prompts, save_path))

            config.obj_patching_langs = "unpatched"
            config.unpatched_langs = {
                (f"{target_input}_{target_output}"): ["source", "target"]
            }
            config.exp_id = None
            config.save_path = None
            config.set_save_path()
            unpatched_save_path = config.save_path

            run_unpatched_flag = not already_run(
                unpatched_save_path,
                target_input,
                target_output,
                source_lang_pairs=None,
                prompts=config.unpatched_langs,
            )

            if dry_run:
                for setting, _, save_path in to_run_settings:
                    print(f"[DRY RUN] {config.model}@{revision}, setting {setting}")
                    print(f"[DRY RUN] Save path: {save_path}")
                if run_unpatched_flag:
                    print(f"[DRY RUN] Unpatched run for {config.model}@{revision}")
                    print(f"[DRY RUN] Save path: {unpatched_save_path}")
                if not to_run_settings and not run_unpatched_flag:
                    print(f"[DRY RUN] Nothing left for {model}@{revision}")
                continue  # skip model load entirely

            if not to_run_settings and not run_unpatched_flag:
                print(f"All runs already done for {model}@{revision}")
                continue

            nn_model = load_model(
                config.model,
                revision=config.revision,
                tokenizer=config.tokenizer,
                device_map="auto",
            )
            if cuda_device is not None and "cuda" not in str(
                next(nn_model.parameters()).device
            ):
                raise RuntimeError(f"Model {config.model} not on CUDA device.")

            for setting, source_prompts, _ in to_run_settings:
                run_all_settings(
                    config,
                    target_input,
                    target_output,
                    {setting: settings_to_keys[setting]},
                    {setting: source_prompts},
                    targets,
                    nn_model,
                    dry_run=False,
                )

            if run_unpatched_flag:
                run_unpatched(
                    config,
                    target_input,
                    target_output,
                    unpatched_prompts,
                    nn_model,
                    dry_run=False,
                )

        finally:
            try:
                del nn_model
                torch.cuda.empty_cache()
                gc.collect()
            except Exception:
                pass

            if lock_dir and token:
                is_last = unregister_revision_user(lock_dir, token)
                if is_last and model != "utter-project/EuroLLM-1.7B":
                    delete_hf_revision_cache(model, revision)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to the JSON configuration file.",
    )
    parser.add_argument(
        "--dry-run",
        "-d",
        action="store_true",
        help="See which experiments would be run without running them.",
    )
    parser.add_argument(
        "--target",
        "-t",
        type=str,
        required=True,
        help="Target input/output language pair xx_yy",
    )
    parser.add_argument("--cuda-device", "-g", help="CUDA device to use.")
    parser.add_argument(
        "--slug", "-s", required=True, type=str, help="Experiment slug."
    )
    args = parser.parse_args()
    print(
        f"Running with config: {args.config}, slug: {args.slug}, target: {args.target}, cuda_device: {args.cuda_device}, dry_run: {args.dry_run}"
    )
    main(
        config_path=args.config,
        slug=args.slug,
        target_langs=args.target,
        cuda_device=args.cuda_device,
        dry_run=args.dry_run,
    )
