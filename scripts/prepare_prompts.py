from argparse import ArgumentParser
import re
from clap.paired_prompts_utils import (
    prepare_prompts,
    multi_simlex_from_csv,
    load_prompts,
    save_prompts,
    extend_prompts,
    retokenize_prompts,
)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="utter-project/EuroLLM-1.7B",
        help="Model to use for prompts.",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=42,
        help="Seed",
    )
    parser.add_argument(
        "--num-pairs",
        "-n",
        type=int,
        default=256,
        help="Number of prompts to construct, set to -1 for all possible pairs (not recommended).",
    )
    parser.add_argument(
        "--few-shot",
        "-f",
        type=int,
        default=5,
        help="Number of examples for few shot.",
    )
    parser.add_argument(
        "--cuda-device", "-g", type=int, default=None, help="CUDA device to use."
    )
    parser.add_argument(
        "--strictly-balanced",
        "-b",
        action="store_true",
        help="Whether to strictly balance the concepts (may mean fewer resulting prompts).",
    )
    parser.add_argument(
        "--extend-lang",
        type=str,
        help="Extend source prompts with copying task for lang.",
    )
    parser.add_argument(
        "--prompts-cache",
        type=str,
        help="Path to the prompts cache file.",
    )
    parser.add_argument(
        "--retokenize",
        action="store_true",
        help="Retokenize prompts in the cache with a different tokenizer.",
    )
    args = parser.parse_args()
    if args.extend_lang:
        if args.prompts_cache is None:
            raise ValueError("Please provide a prompts cache file to extend prompts.")
        df = multi_simlex_from_csv()
        sources, targets = load_prompts(args.prompts_cache)
        new_sources = extend_prompts(sources, args.extend_lang, df)
        new_prompts_cache = re.sub(
            ".pkl", f"_extended_{args.extend_lang}.pkl", args.prompts_cache
        )
        save_prompts(new_sources, targets, new_prompts_cache)
    elif args.retokenize:
        if args.prompts_cache is None:
            raise ValueError(
                "Please provide a prompts cache file to retokenize prompts."
            )
        model_short = args.model.split("/")[-1]
        if model_short in args.prompts_cache:
            raise ValueError("Please provide a different model for retokenization.")
        old_sources, old_targets = load_prompts(args.prompts_cache)
        new_sources, new_targets = retokenize_prompts(
            old_sources, old_targets, args.model
        )
        old_model_short = args.prompts_cache.split("/")[-1].split("_")[0]
        new_prompts_cache = args.prompts_cache.replace(old_model_short, model_short)
        save_prompts(new_sources, new_targets, new_prompts_cache)
    else:
        prepare_prompts(
            args.seed,
            args.model,
            args.few_shot,
            args.num_pairs,
            args.cuda_device,
            args.strictly_balanced,
        )
