from collections import Counter, defaultdict
import heapq
import logging
import math
import os
import pickle
from random import choice, random, shuffle
from tqdm import tqdm

import clap.load_env  # noqa: F401
from clap.config import LangGroup
from clap.const import DATA_DIR, PROMPTS_CACHE
from clap.exp_tools import set_seed, load_model
from clap.lang_utils import get_permutations
from clap.load_dataset import multi_simlex_from_csv
from clap.prompt_tools import (
    Prompt,
    build_prompt_str,
    translation_prompt,
    update_target_prompt,
)


def prepare_tokens_map(
    df,
    tokenizer,
    skip_langs: set[str] | None = None,
):
    word_to_tokens = {}

    def skip(key):
        if skip_langs is not None and key in skip_langs:
            return True
        if key in {"word_original", "scores", "PoS", "ID"}:
            return True
        return False

    for _, row in df.iterrows():
        word = row["word_original"]
        target_strings = []
        latent_strings = {k: v for k, v in row.items() if not skip(k)}
        _, latent_tokens = Prompt.get_target_latent_tokens(
            target_strings, latent_strings, tokenizer
        )
        if len(latent_tokens) > 0:
            word_to_tokens[word] = {
                tok for tokens in latent_tokens.values() for tok in tokens
            }

    return word_to_tokens


def get_compatible_concept_map(
    df,
    tokenizer,
    skip_langs: set[str] | None = None,
):
    word_to_tokens = prepare_tokens_map(df, tokenizer, skip_langs)
    words = list(word_to_tokens.keys())
    result = defaultdict(list)

    n = len(words)
    for i in range(n - 1):
        word = words[i]
        word_tokens = word_to_tokens[word]
        for j in range(i + 1, n):
            other_word = words[j]
            other_tokens = word_to_tokens[other_word]
            if other_tokens.intersection(word_tokens) == set():
                result[word].append(other_word)
                result[other_word].append(word)
    return result


def _select_pairings(left_usage, right_usage, pairings, num_pairs, max_usage):
    selected_pairs = []

    def pair_score(pair):
        c1, c2 = pair.split("_")
        return -(left_usage[c1] + right_usage[c2])

    heap = []
    for pair in pairings:
        tie_breaker = random()
        heapq.heappush(heap, (pair_score(pair), tie_breaker, pair))

    while heap and len(selected_pairs) < num_pairs:
        _, _, pair = heapq.heappop(heap)
        c1, c2 = pair.split("_")

        if left_usage[c1] < max_usage and right_usage[c2] < max_usage:
            selected_pairs.append(pair)
            left_usage[c1] += 1
            right_usage[c2] += 1
            new_heap = []
            for _, _, p in heap:
                tie_breaker = random()
                new_heap.append((pair_score(p), tie_breaker, p))
            heap = new_heap
            heapq.heapify(heap)
    return selected_pairs


def select_concept_pairings(concept_map, num_pairs, strictly_balanced=False):
    pairings = []
    concept_set = set()
    for concept, compatible_concepts in concept_map.items():
        for compatible in compatible_concepts:
            pairings.append(f"{concept}_{compatible}")
            concept_set.update([concept, compatible])

    shuffle(pairings)

    total_concepts = len(concept_set)
    if total_concepts == 0:
        return []

    if num_pairs < 0:
        return pairings

    max_usage = math.ceil(num_pairs / total_concepts)
    left_usage = Counter()
    right_usage = Counter()

    selected_pairs = []
    selected_pairs = _select_pairings(
        left_usage, right_usage, pairings, num_pairs, max_usage
    )

    if not strictly_balanced:
        while len(selected_pairs) < num_pairs:
            max_usage += 1
            print(
                f"Increasing max usage to {max_usage} to fill up to {num_pairs} pairs."
            )
            additional_pairs = _select_pairings(
                left_usage,
                right_usage,
                pairings,
                num_pairs - len(selected_pairs),
                max_usage,
            )
            selected_pairs.extend(additional_pairs)

    print(
        f"Selected {len(selected_pairs)} pairs strictly_balanced={strictly_balanced}."
    )
    print(f"Left usage: {left_usage}")
    print(f"Right usage: {right_usage}")
    left_usage.update(right_usage)
    print(f"Total concepts used {len(left_usage)}")
    return selected_pairs


def get_multi_way_obj_prompts(
    nn_model,
    lang_pairs: list[LangGroup],
    df,
    num_few_shot=5,
    num_pairs: int | None = None,
    strictly_balanced=False,
):
    all_langs = {lang for pair in lang_pairs for lang in pair}
    skip_langs = {lang for lang in df.columns if lang not in all_langs}
    concept_map = get_compatible_concept_map(
        df, nn_model.tokenizer, skip_langs=skip_langs
    )
    pairings = select_concept_pairings(concept_map, num_pairs, strictly_balanced)
    source_prompts = defaultdict(list)
    target_prompts = defaultdict(list)
    concepts = list(concept_map.keys())
    for pair in tqdm(pairings, desc="Generating object prompts"):
        concept, compatible_concept = pair.split("_")
        # TODO, do we care that the few-shot examples are not compatible?
        few_shot = []
        cnt = 0
        while cnt < 2 * num_few_shot:
            fs_candidate = choice(concepts)
            if (
                fs_candidate not in few_shot
                and fs_candidate != concept
                and fs_candidate != compatible_concept
            ):
                few_shot.append(fs_candidate)
                cnt += 1

        target_fs = few_shot[:num_few_shot]
        source_fs = few_shot[num_few_shot:]
        source_prompts_for_concept = []
        for input_lang, output_lang in lang_pairs:
            key = f"{input_lang}_{output_lang}"
            source_prompt_str = build_prompt_str(
                df,
                concept,
                few_shot[num_few_shot:],
                input_lang,
                output_lang,
                cut_at_obj=False,
            )
            source_prompt = translation_prompt(
                df.loc[df["word_original"] == concept].iloc[0],
                source_prompt_str,
                nn_model.tokenizer,
                input_lang,
                output_lang,
                latent_langs=all_langs,
                fs_examples=source_fs,
            )
            source_prompts[key].append(source_prompt)
            source_prompts_for_concept.append(source_prompt)

        for input_lang, output_lang in lang_pairs:
            key = f"{input_lang}_{output_lang}"
            target_prompt_str = build_prompt_str(
                df,
                compatible_concept,
                target_fs,
                input_lang,
                output_lang,
                cut_at_obj=False,
            )
            target_prompt = translation_prompt(
                df.loc[df["word_original"] == compatible_concept].iloc[0],
                target_prompt_str,
                nn_model.tokenizer,
                input_lang,
                output_lang,
                latent_langs=all_langs,
                fs_examples=target_fs,
            )
            target_prompt = update_target_prompt(
                source_prompts_for_concept, target_prompt, output_lang
            )
            target_prompts[key].append(target_prompt)
    return source_prompts, target_prompts


def construct_source_target_prompts(
    nn_model, df, skip_langs, num_few_shot, num_pairs, strictly_balanced=False
):
    permutations = get_permutations(skip_langs=skip_langs)
    sources, targets = get_multi_way_obj_prompts(
        nn_model,
        permutations,
        df,
        num_few_shot=num_few_shot,
        num_pairs=num_pairs,
        strictly_balanced=strictly_balanced,
    )
    return sources, targets


def prepare_prompts(
    seed: int,
    model: str,
    num_few_shot: int,
    num_pairs: int,
    cuda_device: int | str | None = None,
    strictly_balanced=False,
) -> None:
    set_seed(seed)
    if cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    skip_langs = {"ar", "he"}
    nn_model = load_model(model, device_map="auto")
    # TODO make general for other languages and PoS
    path = f"{DATA_DIR}/multi_simlex_processed.csv"
    df = multi_simlex_from_csv(path)
    sources, targets = construct_source_target_prompts(
        nn_model,
        df,
        skip_langs=skip_langs,
        num_few_shot=num_few_shot,
        num_pairs=num_pairs,
        strictly_balanced=strictly_balanced,
    )
    prompts_cache = get_prompts_cache_path(
        seed, model, num_few_shot, len(sources["en_en"])
    )
    if os.path.exists(prompts_cache):
        logging.warning(
            f"Prompts cache {prompts_cache} already exists, skipping saving."
        )
        return sources, targets
    save_prompts(sources, targets, prompts_cache)
    return sources, targets


def get_prompts_cache_path(
    seed: int, model: str, num_few_shot: int, num_pairs: int
) -> str:
    model_name = os.path.basename(model)
    os.makedirs(PROMPTS_CACHE, exist_ok=True)
    return os.path.join(
        PROMPTS_CACHE, f"{model_name}_{seed}_{num_few_shot}_{num_pairs}_prompts.pkl"
    )


def load_prompts(prompts_cache=None):
    with open(prompts_cache, "rb") as f:
        print(f"Loading prompts from {prompts_cache}")
        data = pickle.load(f)
        if "sources" not in data or "targets" not in data:
            print(f"Prompts cache {prompts_cache} does not contain sources and targets")
            return data, None
        sources = data["sources"]
        targets = data["targets"]
    return sources, targets


def save_prompts(sources, targets, prompts_cache):
    with open(prompts_cache, "wb") as f:
        logging.info(f"Saving prompts to {prompts_cache}")
        pickle.dump({"sources": sources, "targets": targets}, f)


def extend_prompts(sources, lang, df):
    lang_pair = f"{lang}_{lang}"
    sources[lang_pair] = []
    for input_prompt, output_prompt in zip(
        sources[f"{lang}_en"], sources[f"en_{lang}"], strict=True
    ):
        concept = input_prompt.word_original
        prompt_str = build_prompt_str(
            df,
            concept,
            input_prompt.fs_examples,
            lang,
            lang,
            cut_at_obj=False,
        )
        new_prompt = Prompt(
            prompt=prompt_str,
            target_tokens=output_prompt.target_tokens,
            latent_tokens=output_prompt.latent_tokens,
            target_strings=output_prompt.target_strings,
            latent_strings=output_prompt.latent_strings,
            input_string=input_prompt.input_string,
            word_original=output_prompt.word_original,
            fs_examples=input_prompt.fs_examples,
        )
        sources[lang_pair].append(new_prompt)
    return sources


def retokenize_prompts(old_sources, old_targets, model):
    nn_model = load_model(model, device_map="auto")
    new_sources = defaultdict(list)
    new_targets = defaultdict(list)
    for key, prompts in old_sources.items():
        for prompt in prompts:
            prompt.retokenize(nn_model.tokenizer)
            new_sources[key].append(prompt)
    for key, prompts in old_targets.items():
        for prompt in prompts:
            prompt.retokenize(nn_model.tokenizer)
            new_targets[key].append(prompt)
    return new_sources, new_targets
