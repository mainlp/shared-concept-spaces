from nnsight.intervention.envoy import Envoy
from nnsight import LanguageModel
import torch as th
from torch.utils.data import DataLoader


def get_layer(nn_model: LanguageModel, layer: int) -> Envoy:
    """
    Get the layer of the model
    Args:
        nn_model: The NNSight model
        layer: The layer to get
    Returns:
        The Envoy for the layer
    """
    return nn_model.model.layers[layer]


def get_layer_output(nn_model: LanguageModel, layer: int):
    """
    Get the output of a layer
    Args:
        nn_model: The NNSight model
        layer: The layer to get the output of
    Returns:
        The Proxy for the output of the layer
    """
    output = get_layer(nn_model, layer).output
    return output


def get_logits(nn_model: LanguageModel):
    """
    Get the logits of the model
    Args:
        nn_model: The NNSight model
    Returns:
        The Proxy for the logits of the model
    """
    return nn_model.output.logits


def get_next_token_probs(nn_model: LanguageModel):
    """
    Get the probabilities of the model
    Args:
        nn_model: The NNSight model
    Returns:
        The Proxy for the probabilities of the model
    """
    return get_logits(nn_model)[:, -1, :].softmax(-1)


def get_num_layers(nn_model: LanguageModel):
    """
    Get the number of layers in the model
    Args:
        nn_model: The NNSight model
    Returns:
        The number of layers in the model
    """
    return len(nn_model.model.layers)


@th.no_grad()
def collect_activations(
    nn_model: LanguageModel,
    prompts,
    layers=None,
    get_activations=None,
    remote=False,
    idx=None,
):
    """
    Collect the hidden states of the last token of each prompt at each layer

    Args:
        nn_model: The NNSight model
        prompts: The prompts to collect activations for
        layers: The layers to collect activations for, default to all layers
        get_activations: The function to get the activations, default to layer output
        remote: Whether to run the model on the remote device
        idx: The index of the token to collect activations for
        open_context: Whether to open a trace context to collect activations. Set to false if you want to
            use this function in a context that already has a trace context open

    Returns:
        The hidden states of the last token of each prompt at each layer, moved to cpu. If open_context is False, returns a list of
        Proxies. Dimensions are (num_layers, num_prompts, hidden_size)
    """
    if get_activations is None:
        get_activations = get_layer_output
    tok_prompts = nn_model.tokenizer(prompts, return_tensors="pt", padding=True)
    # Todo?: This is a hacky way to get the last token index but it works for both left and right padding
    last_token_index = tok_prompts.attention_mask.flip(1).cumsum(1).bool().int().sum(1)
    if idx is None:
        idx = last_token_index.sub(1)  # Default to the last token
    elif idx < 0:
        idx = last_token_index + idx
    else:
        raise ValueError(
            "positive index is currently not supported due to left padding"
        )
    if layers is None:
        layers = range(get_num_layers(nn_model))

    try:
        model_device = next(nn_model.model.parameters()).device
    except Exception:
        model_device = th.device("cpu")

    batch = len(tok_prompts.input_ids)
    batch_idx = th.arange(batch, dtype=th.long, device=model_device)
    idx = idx.to(dtype=th.long, device=model_device)

    acts = []
    with nn_model.trace(prompts, remote=remote):
        for layer in layers:
            h = get_activations(nn_model, layer)
            v = h[batch_idx, idx]
            v = v.to("cpu").save()
            acts.append(v)

    return th.stack(acts, dim=0)


def collect_activations_batched(
    nn_model: LanguageModel,
    prompts,
    batch_size,
    layers=None,
    get_activations=None,
    remote=False,
    idx=None,
    tqdm=None,
):
    """
    Collect the hidden states of the last token of each prompt at each layer in batches

    Args:
        nn_model: The NNSight model
        prompts: The prompts to collect activations for
        batch_size: The batch size to use
        layers: The layers to collect activations for, default to all layers
        get_activations: The function to get the activations, default to layer output
        remote: Whether to run the model on the remote device
        idx: The index of the token to collect activations for

    Returns:
        The hidden states of the last token of each prompt at each layer, moved to cpu. Dimensions are (num_layers, num_prompts, hidden_size)
    """
    dataloader = DataLoader(prompts, batch_size=batch_size)
    if tqdm is not None:
        dataloader = tqdm(dataloader)
    acts = []
    for batch in dataloader:
        acts_batch = collect_activations(
            nn_model, batch, layers, get_activations, remote, idx
        )
        acts.append(acts_batch)
    return th.cat(acts, dim=1)
