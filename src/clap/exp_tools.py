from typing import Optional
import torch as th
from nnsight import LanguageModel
from transformers import AutoConfig, AutoTokenizer


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
