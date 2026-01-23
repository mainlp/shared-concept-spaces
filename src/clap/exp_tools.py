import os
from typing import Callable, Optional
import torch as th
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer
from nnsight import LanguageModel
from torch.utils.data import DataLoader

from clap.config import BaseExperimentConfig
from clap.prompt_tools import Prompt

GetProbFunction = Callable[[LanguageModel, str | list[str], bool], th.Tensor]


def load_model(
    model_name: str,
    trust_remote_code: bool = False,
    revision: Optional[bool] = None,
    **kwargs_,
):
    """
    Load a model into nnsight.
    Default device is "auto" and default dtype is th.float16.
    """
    # Seems we have to force fp32 for Apertus because otherwise there is a
    # clash with nnsight..
    dtype = th.float32 if "Apertus" in model_name else th.float16
    kwargs = dict(dtype=dtype, trust_remote_code=trust_remote_code, device_map="auto")

    tokenizer_kwargs = kwargs_.pop("tokenizer_kwargs", {})
    tokenizer_kwargs.update(
        dict(add_prefix_space=False, trust_remote_code=trust_remote_code)
    )
    kwargs |= kwargs_
    if kwargs_.get("tokenizer", None) is not None:
        # loading a custom/local model
        config = AutoConfig.from_pretrained(kwargs["tokenizer"], **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(
            kwargs["tokenizer"],
            config=config,
            padding_side="left",
            **tokenizer_kwargs,
        )
        tokenizer.pad_token = tokenizer.eos_token
        kwargs["tokenizer"] = tokenizer
    nn_model = LanguageModel(
        model_name,
        revision=revision,
        dispatch=True,
        tokenizer_kwargs=tokenizer_kwargs,
        **kwargs,
    )
    nn_model.eval()
    return nn_model


def set_seed(seed: int):
    if seed is None or seed < 0:
        return
    import random
    import numpy as np

    print(f"Setting seed to {seed}")

    # Set seed for PyTorch
    th.manual_seed(seed)

    # Set seed for CUDA (if using GPUs)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Set seed for Python's random module
    random.seed(seed)

    # Set seed for NumPy
    np.random.seed(seed)

    # Ensure deterministic behavior for PyTorch operations
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False


def set_up_exp(config: BaseExperimentConfig) -> BaseExperimentConfig:
    if config.exp_id is None:
        config.set_exp_id()
    if config.save_path is None:
        config.set_save_path()
    os.makedirs(config.save_path, exist_ok=True)
    config.save()
    print(f"Running with {config}")
    set_seed(config.seed)
    import torch as th

    _ = th.set_grad_enabled(False)
    return config


def next_token_probs(
    nn_model: LanguageModel, prompt: str | list[str], remote=False
) -> th.Tensor:
    out = nn_model.trace(prompt, trace=False, remote=remote).logits
    return out[:, -1].softmax(-1).cpu()


def next_token_probs_unsqueeze(
    nn_model: LanguageModel,
    prompt: str | list[str],
    remote=False,
) -> th.Tensor:
    probs = next_token_probs(nn_model, prompt, remote=remote)
    return probs.unsqueeze(1)  # Add a fake layer dimension


@th.no_grad
def run_prompts(
    nn_model: LanguageModel,
    prompts: list[Prompt],
    batch_size: int = 32,
    get_probs: GetProbFunction | None = None,
    get_probs_kwargs: dict | None = None,
    tqdm=tqdm,
):
    """
    Run a list of prompts through the model and return the probabilities of the next token for both the target and latent languages.

    Args:
        nn_model: The NNSight model
        prompts: A list of prompts
        batch_size: The batch size to use
        get_probs: The function to get the probabilities of the next token, default to next token prediction
        get_probs_kwargs: The kwargs to pass to the get_probs function
        tqdm: The tqdm function to use, default to tqdm.auto.tqdm.

    Returns:
        Two tensors target_probs and latent_probs of shape (num_prompts, num_layers)
    """
    str_prompts = [prompt.prompt for prompt in prompts]
    dataloader = DataLoader(str_prompts, batch_size=batch_size)
    probs = []
    generated_answers = []
    if get_probs is None:
        get_probs = next_token_probs_unsqueeze
    if get_probs_kwargs is None:
        get_probs_kwargs = {}
    for prompt_batch in tqdm(dataloader, total=len(dataloader), desc="Running prompts"):
        p = get_probs(nn_model, prompt_batch, **get_probs_kwargs)
        if len(p) == 2:
            # If the result is a tuple, we assume it's (probs, generated_answers)
            p, g = p
            generated_answers.append(g)
        probs.append(p)
    probs = th.cat(probs)
    target_probs = []
    latent_probs = {lang: [] for lang in prompts[0].latent_tokens.keys()}
    for i, prompt in enumerate(prompts):
        target_probs.append(probs[i, :, prompt.target_tokens].sum(dim=1))
        for lang, tokens in prompt.latent_tokens.items():
            latent_probs[lang].append(probs[i, :, tokens].sum(dim=1))
    target_probs = th.stack(target_probs).cpu()
    latent_probs = {lang: th.stack(probs).cpu() for lang, probs in latent_probs.items()}
    return target_probs, latent_probs, generated_answers
