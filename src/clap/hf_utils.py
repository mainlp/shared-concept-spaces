import os
import shutil
import uuid
import re
import glob

from huggingface_hub import snapshot_download, HfApi
from pathlib import Path

from clap.ellm_utils import get_ellm_path, ellm_phase_mapping


def get_sorted_steps(model):
    if model == "utter-project/EuroLLM-1.7B":
        ellm_path = get_ellm_path()
        model_paths = glob.glob(f"{ellm_path}/*/*")
        max_p_1_step_num = -1
        step_to_path = {}

        for model in model_paths:
            step_num = get_step_number(model)
            if step_num.split("_")[1] == "1":
                max_p_1_step_num = max(max_p_1_step_num, int(step_num.split("_")[2]))
            step_to_path[step_num] = model

        def sort_key(s):
            match = re.match(r"phase_(\d+)_(\d+)", s)
            if match:
                phase, number = match.groups()
                phase = int(phase)
                number = int(number)
                if phase == 2:
                    number += max_p_1_step_num  # offset phase 2
                return number
            else:
                return 0  # fallback

        sorted_step_keys = sorted(step_to_path.keys(), key=sort_key)
        return sorted_step_keys, step_to_path
    else:
        refs = [
            r for r in get_all_refs(model) if "stage2" not in r and "longctx" not in r
        ]
        if "Apertus" in model:
            pat = re.compile(r"step(?P<step>\d+)-tokens(?P<tokens>\d+)B")
        elif "bloom" in model:
            pat = re.compile(r"global_step(?P<step>\d+)")
        else:
            # assume OLMo-like naming
            pat = re.compile(
                r"stage(?P<stage>\d+)-step(?P<step>\d+)-tokens(?P<tokens>\d+)B"
            )
        parsed = []

        for r in refs:
            m = pat.match(r)
            if not m:
                continue
            step = int(m.group("step"))
            parsed.append((step, r))

        parsed.sort(key=lambda x: x[0])
        sorted_step_keys = [step for step, _ in parsed]
        step_to_path = {step: r for step, r in parsed}
        return sorted_step_keys, step_to_path


def _sanitize(*parts: str) -> str:
    s = "__".join(parts)
    return s.replace("/", "__").replace(":", "_")


def register_revision_user(repo: str, revision: str | None) -> tuple[Path, Path]:
    """
    Create a token under a repo+revision-specific dir.
    """
    base = os.getenv("HF_HOME") or os.getenv("HF_HUB_CACHE") or "/tmp"
    locks_root = Path(base) / "locks"
    lock_dir = locks_root / _sanitize(repo, revision or "none")
    lock_dir.mkdir(parents=True, exist_ok=True)
    token = lock_dir / f"{uuid.uuid4()}.lock"
    fd = os.open(str(token), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(fd)
    return lock_dir, token


def unregister_revision_user(lock_dir: Path, token: Path) -> bool:
    try:
        token.unlink(missing_ok=True)  # py>=3.8
    except TypeError:
        try:
            token.unlink()
        except FileNotFoundError:
            pass
    try:
        return next(lock_dir.iterdir(), None) is None
    except FileNotFoundError:
        return True


def delete_hf_revision_cache(repo_id: str, revision: str | None):
    """
    Remove just the local cache folder for a given repo@revision.
    """
    try:
        # This resolves to the local snapshot dir even with local_files_only
        snap = snapshot_download(
            repo_id,
            revision=revision,
            local_files_only=True,
            allow_patterns="*",  # no download
        )
    except Exception:
        return  # nothing cached locally, or can't resolve → skip
    try:
        shutil.rmtree(snap, ignore_errors=True)
    except Exception:
        pass


def delete_hf_repo_cache(model_id: str):
    cache_dir = os.getenv("HF_HUB_CACHE")
    repo_dir = os.path.join(cache_dir, f"models--{model_id.replace('/', '--')}")
    shutil.rmtree(repo_dir, ignore_errors=True)
    print(f"Deleted HF cache directory: {repo_dir}")


def get_all_refs(model_name: str):
    refs = HfApi().list_repo_refs(model_name, repo_type="model")
    ref_names = []
    for branch in refs.branches:
        if "Apertus" in model_name and "longctx" in branch.name:
            continue
        ref_names.append(branch.name)
    # fallback to tag names
    if len(ref_names) < 2:
        for tag in refs.tags:
            if "Apertus" in model_name and "longctx" in tag.name:
                continue
            ref_names.append(tag.name)
    return ref_names


def make_should_run(schedule):
    """
    Build a should_run_on_checkpoint function from a schedule.

    Args:
        schedule (list of (max_index, stride)):
            - max_index: upper bound on i (exclusive), or None for "rest"
            - stride: keep every stride-th element in that range (1 = all)

    Example schedule:
        [
            (10, 1),   # keep all up to i=10
            (20, 2),   # keep every 2nd until i=20
            (None, 5), # keep every 5th for the rest
        ]
    """

    def should_run(i, total_cnt):
        for max_i, stride in schedule:
            if max_i is None or i < max_i:
                if i == total_cnt - 1:
                    return True
                return i % stride == 0
        return False

    return should_run


POLICIES = {
    "eurollm": make_should_run(
        [
            (10, 1),
            (20, 2),
            (None, 5),
        ]
    ),
    "olmo": make_should_run(
        [
            (10, 1),
            (20, 2),
            (30, 10),
            (None, 20),
        ]
    ),
    "apertus": make_should_run(
        [
            (10, 1),
            (20, 2),
            (30, 5),
            (None, 10),
        ]
    ),
    "bloom": make_should_run(
        [
            (None, 1),
        ]
    ),
}


def get_policy_for_model(model_name: str):
    lname = model_name.lower()
    if "olmo" in lname:
        return POLICIES["olmo"]
    elif "apertus" in lname:
        return POLICIES["apertus"]
    elif "bloom" in lname:
        return POLICIES["bloom"]
    else:
        return POLICIES["eurollm"]


def get_step_number(path):
    if re.search(r"step(\d+)", path):
        return int(re.search(r"step(\d+)", path).group(1))
    elif re.search(r"iter_(\d+)", path):
        match = re.search(r"(.*/)([^/]*)(/iter_\d+)", path)
        if match:
            phase = match.group(2)
            phase = ellm_phase_mapping[phase]
        iteration = re.search(r"iter_(\d+)", path).group(1)
        return f"{phase}_{iteration}" if match else iteration
    else:
        return int("inf")
