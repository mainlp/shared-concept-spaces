import torch as th
from clap.nnsight_utils import (
    collect_activations,
    get_layer_output,
    get_next_token_probs,
    get_num_layers,
)


@th.no_grad()
def object_lens(
    nn_model,
    target_prompts,
    idx,  # object index
    source_prompts=None,
    hiddens=None,
    steering_vectors=None,
    num_patches=-1,
    scan=True,
    remote=False,
    generate=True,
    patch=True,
):
    if isinstance(target_prompts, str):
        target_prompts = [target_prompts]
    num_layers = get_num_layers(nn_model)
    if num_patches == -1:
        num_patches = num_layers
    if steering_vectors is not None:
        if hiddens is None:
            if source_prompts is None:
                raise ValueError("Either source_prompts or hiddens must be provided")
            hiddens = collect_activations(
                nn_model,
                source_prompts,
                remote=remote,
            )
        for i, (h, s) in enumerate(zip(hiddens, steering_vectors, strict=True)):
            hiddens[i] = h + s
    probs_l = []
    generated_answers_l = []
    for layer in range(num_layers):
        with nn_model.trace(target_prompts, remote=remote):
            if patch:
                for target_layer in range(layer, min(layer + num_patches, num_layers)):
                    get_layer_output(nn_model, target_layer)[:, idx] = hiddens[
                        target_layer
                    ]
            probs = get_next_token_probs(nn_model).cpu().save()
            probs_l.append(probs)

        if generate:
            max_tokens = 10
            tokenized = nn_model.tokenizer(
                target_prompts, return_tensors="pt", padding=True
            )
            with nn_model.generate(
                tokenized,
                max_new_tokens=max_tokens,
                remote=remote,
            ):
                if patch:
                    for target_layer in range(
                        layer, min(layer + num_patches, num_layers)
                    ):
                        get_layer_output(nn_model, target_layer)[:, idx] = hiddens[
                            target_layer
                        ]

                out = nn_model.generator.output.save()
            generated_answers_l.append(
                nn_model.tokenizer.batch_decode(
                    [o[-max_tokens:] for o in out], skip_special_tokens=True
                )
            )

    probs_l = (
        th.cat(probs_l, dim=0)
        .reshape(num_layers, len(target_prompts), -1)
        .transpose(0, 1)
    )
    return probs_l, generated_answers_l
