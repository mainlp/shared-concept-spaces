from __future__ import annotations

from enum import Enum
import json
import os
from pathlib import Path
from time import time
from typing import Optional, Type
from coolname import generate_slug
from pydantic import BaseModel

from clap.const import RESULTS_DIR


class ExperimentType(str, Enum):
    OBJ_PATCHING = "obj_patching"
    CONCEPT_TRANSLATION = "concept_translation"
    SHIFTED_TRANSLATION = "shifted_translation"


LangPair = tuple[str, str]
LangGroup = tuple[tuple[LangPair, ...], str, str]


class BaseExperimentConfig(BaseModel):
    experiment_type: ExperimentType
    model: str
    tokenizer: Optional[str] = None
    revision: Optional[str] = None
    batch_size: int = 32
    seed: int = 42
    exp_id: Optional[str] = None
    save_path: Optional[str] = None
    slug: Optional[str] = None
    prompts_cache: str

    def model_post_init(self, __context) -> None:
        self.set_exp_id()
        if self.slug is None:
            self.slug = generate_slug(2)
        self.set_save_path()

    def re_init(self) -> None:
        # use this to re-init a config, e.g. after changing a var
        self.save_path = None
        self.exp_id = None
        self.set_exp_id()
        self.set_save_path()

    def set_exp_id(self) -> None:
        if self.exp_id is not None:
            return
        self.exp_id = f"{int(time())}_" + generate_slug(2)

    def get_model_name(self) -> None:
        return Path(self.tokenizer or self.model).name

    def get_revision(self) -> None:
        if self.revision is not None:
            # we are loading a huggingface revision
            return self.revision
        if self.tokenizer is None or self.tokenizer == self.model:
            # we are loading the huggingface main model
            return None
        # we are loading a local model
        return "/".join(Path(self.model).parts[-2:])

    def set_save_path(self) -> None:
        if self.save_path is not None:
            return

        if self.exp_id is None:
            self.set_exp_id()

        model_name = self.get_model_name()
        rev = self.get_revision()
        midfix = ""
        if self.slug is not None:
            midfix = self.slug
        if rev is None:
            rev = "main"
        midfix += f"/{rev}"
        save_path = f"{RESULTS_DIR}/{self.experiment_type.value}/{model_name}/{midfix}/{self.exp_id}"
        self.save_path = save_path

    def save(self) -> None:
        file_name = f"{self.save_path}/exp_args.json"
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, "w") as f:
            json.dump(self.model_dump(), f, indent=2)


class ObjPatchingConfig(BaseExperimentConfig):
    experiment_type: ExperimentType = ExperimentType.OBJ_PATCHING
    obj_patching_langs: Optional[LangGroup | str] = None
    unpatched_langs: Optional[dict[str, list[str]]] = None

    def model_post_init(self, __context):
        if isinstance(self.obj_patching_langs, str):
            assert self.obj_patching_langs in {"unpatched"}, (
                "obj_patching_langs must be a list of tuples or 'unpatched'"
            )
            if self.obj_patching_langs == "unpatched":
                assert self.unpatched_langs is not None, (
                    "unpatched_langs must be set if obj_patching_langs is 'unpatched'"
                )
                if isinstance(self.unpatched_langs, dict):
                    for prompts in self.unpatched_langs.values():
                        assert 0 < len(prompts) < 3
                        for p in prompts:
                            assert p in {"source", "target"}
        super().model_post_init(__context)

    def get_target_input_lang(self) -> str:
        if (
            isinstance(self.obj_patching_langs, str)
            and self.obj_patching_langs == "unpatched"
        ):
            return list(self.unpatched_langs.keys())[0].split("_")[0]
        return self.obj_patching_langs[1]

    def get_target_output_lang(self) -> str:
        if (
            isinstance(self.obj_patching_langs, str)
            and self.obj_patching_langs == "unpatched"
        ):
            # kind of hacky... but probably okay since we only support one unpatched
            # language for now
            return list(self.unpatched_langs.keys())[0].split("_")[1]
        return self.obj_patching_langs[2]

    def get_source_lang_pairs(self) -> str | tuple[LangPair, ...]:
        if isinstance(self.obj_patching_langs, str):
            return self.obj_patching_langs
        return self.obj_patching_langs[0]


class ConceptTranslationConfig(BaseExperimentConfig):
    experiment_type: ExperimentType = ExperimentType.CONCEPT_TRANSLATION


EXPERIMENT_TYPE_TO_CLASS: dict[ExperimentType, Type[BaseExperimentConfig]] = {
    ExperimentType.OBJ_PATCHING: ObjPatchingConfig,
    ExperimentType.CONCEPT_TRANSLATION: ConceptTranslationConfig,
}


def load_config_from_file(
    path: Path,
) -> BaseExperimentConfig:
    with open(path) as f:
        data = json.load(f)
    config_cls = EXPERIMENT_TYPE_TO_CLASS[data["experiment_type"]]
    return config_cls(**data)
