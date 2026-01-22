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

2. `python src/prepare_prompts.py --extend-lang <new-lang> --prompts-cache <prompts-to-extend-path>`: this can be used to add copying task for a particular language, based on the prompts under `prompts-to-extend-path`. Note that the copying en-en prompts are produced by default. This path was ultimately not used for experiments in the paper. If you pass it a cache path like: `data/prompts_cache/<model-name>_<seed>_<num-fewshot>_<num-prompts>.pkl`, it will produce `data/prompts_cache/<model-name>_<seed>_<num-fewshot>_<num-prompts>_extended_<new-lang>.pkl`.

3. `python src/prepare_prompts.py --model <new-model> --retokenize --prompts-cache <prompts-to-retokenize-path>`: this can be used to retokenize existing prompts for new models, for example to produce the supplementary experiments in the appendix with Apertus. If you pass it a cache path like: `data/prompts_cache/<model-name>_<seed>_<num-fewshot>_<num-prompts>.pkl`, it will produce `data/prompts_cache/<new-model-name>_<seed>_<num-fewshot>_<num-prompts>.pkl`.

