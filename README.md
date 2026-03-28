### Setup
1. Install `requirements.txt` (this code was developed using python 3.11.10)
2. Install this package (in editable mode): `pip install -e .`
3. Copy `.env_template` to `.env` and fill in the values (`HF_HUB_CACHE` is optional to override the default cache location of Huggingface)


### Prepare the data
1. Get the raw dataset from [multisimlex.com](multisimlex.com), save it under `data/multi_simlex.csv`
2. Run `python src/process_multi_simlex.py`


### Prepare the prompts
Prompts will be pickled under `prompts_cache/`.<br>

See `python src/prepare_prompts.py --help` for information on flags. The script has three paths:
1. `python src/prepare_prompts.py`: prepare the initial prompts. This is the main path for experiments in the paper, and will produce prompts in `data/prompts_cache/<model-name>_<seed>_<num-fewshot>_<num-prompts>.pkl`

2. `python src/prepare_prompts.py --extend-lang <new-lang> --prompts-cache <prompts-to-extend-path>`: this can be used to add copying task for a particular language, based on the prompts under `prompts-to-extend-path`. Note that the copying en-en prompts are produced by default. 
If you pass it a cache path like: `data/prompts_cache/<model-name>_<seed>_<num-fewshot>_<num-prompts>.pkl`, it will produce `data/prompts_cache/<model-name>_<seed>_<num-fewshot>_<num-prompts>_extended_<new-lang>.pkl`.

3. `python src/prepare_prompts.py --model <new-model> --retokenize --prompts-cache <prompts-to-retokenize-path>`: this can be used to retokenize existing prompts for new models, for example to produce the supplementary experiments in the appendix with Apertus. If you pass it a cache path like: `data/prompts_cache/<model-name>_<seed>_<num-fewshot>_<num-prompts>.pkl`, it will produce `data/prompts_cache/<new-model-name>_<seed>_<num-fewshot>_<num-prompts>.pkl`.

### To run Crosslingual Concept Patching (CLAP)
First, define a configuration. See `configs/obj_patch_olmo_config.json` for an example. You must link to the prompts you created in the previous step.

You can call the script like:
`python scripts/run_clap_over_time.py --config <path/to.config.json> --target <xx-yy> --slug <example-slug>`

where `xx-yy` is the target language pair, the config file defines source prompts and experimental settings and `slug` is the alias under which results are solved. `--dry-run` will run through the script without actually executing the experiments, use this to check for basic errors in setup.

NOTE: Due to historical reasons, *if* you have target output lang *other than en*, the script expects a `self` prompt, i.e. the copy task for that language. For example, if you have target pair `en-fr`, you *must* first run `python src/prepare_prompts.py --extend-lang fr`, and link to the new, extended set of prompts in your config. This will produce prompts for `fr-fr`.

### Models to Test
1. EuroLLM: as of publishing, we only have local copies of EuroLLM, shared by the developers. Set `ELLM_PATH` in .env to the directory.
2. Apertus 8B and OLMo2-7B: these should work out of the box by downloading revisions from HuggingFace.
3. Other models: to test other models, you must adapt `clap.hf_utils.get_sorted_steps`, `clap.hf_utils.get_policy_for_model`, as well as define a policy in POLICIES (defines which subset of checkpoints will be tested). You may also need to adapt `clap.hf_utils.get_all_refs`.

Please open an issue if you get stuck, we will try to help you!
