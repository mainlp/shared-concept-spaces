import functools
import os

ellm_phase_mapping = {
    "train_annealing_megatronlm": "phase_1",
    "second_phase_annealing_more_multilingual_fw": "phase_2",
}


def get_ellm_path():
    ellm_path = os.getenv("ELLM_PATH")
    if ellm_path is None:
        raise ValueError("ELLM_PATH environment variable is not set.")
    return ellm_path


def compare_checkpoints(
    item1: str | tuple[str, str] | list[str],
    item2: str | tuple[str, str] | list[str],
) -> int:
    # Custom comparison function to sort checkpoints
    if isinstance(item1, (tuple, list)):
        item1 = item1[1]
        item2 = item2[1]
    if not isinstance(item1, str) or not isinstance(item2, str):
        raise ValueError("Both items must be strings.")
    if not item1.startswith("phase_") or not item2.startswith("phase_"):
        raise ValueError("Both items must start with 'phase_'.")
    phase1 = int(item1.split("_")[1])
    phase2 = int(item2.split("_")[1])
    step_num1 = int(item1.split("_")[-1])
    step_num2 = int(item2.split("_")[-1])

    if phase1 != phase2:
        return phase1 - phase2
    return step_num1 - step_num2


def get_switch_step_idx(checkpoints, labels):
    if not isinstance(labels[0], str) or "phase" not in labels[0]:
        # most likely not EuroLLM...
        return None
    phase_switch_step = None
    sorted_checkpoints = sorted(checkpoints)
    sorted_labels = sorted(labels, key=functools.cmp_to_key(compare_checkpoints))
    for checkpoint, label in zip(sorted_checkpoints, sorted_labels, strict=True):
        if "phase_2" in label:
            phase_switch_step = checkpoint
            break
    return phase_switch_step
