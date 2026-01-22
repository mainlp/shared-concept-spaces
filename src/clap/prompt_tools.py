from dataclasses import dataclass
from typing import Optional

from clap.utils import ulist
from clap.lang_utils import lang2name


def process_tokens_with_tokenization(
    words: str | list[str], tokenizer, i_am_hacky=False
):
    if isinstance(words, str):
        words = [words]
    final_tokens = []
    for word in words:
        # If you get the value error even with add_prefix_space=False,
        # you can use the following hacky code to get the token without the prefix
        if i_am_hacky:
            hacky_token = tokenizer("🍐", add_special_tokens=False).input_ids
            length = len(hacky_token)
            tokens = tokenizer("🍐" + word, add_special_tokens=False).input_ids
            if tokens[:length] != hacky_token:
                raise ValueError(
                    "I didn't expect this to happen, please check this code"
                )
            if len(tokens) > length:
                final_tokens.append(tokens[length])
        else:
            # Assuming the tokenizer was initialized with add_prefix_space=False
            token = tokenizer(word, add_special_tokens=False).input_ids[0]
            token_with_start_of_word = tokenizer(
                " " + word, add_special_tokens=False
            ).input_ids[0]
            if token == token_with_start_of_word:
                raise ValueError(
                    "Seems like you're using a tokenizer that wasn't initialized with add_prefix_space=False."
                )
            final_tokens.append(token)
            if (
                token_with_start_of_word
                != tokenizer(" ", add_special_tokens=False).input_ids[0]
            ):
                final_tokens.append(token_with_start_of_word)
    return ulist(final_tokens)


@dataclass
class Prompt:
    prompt: str
    target_tokens: list[int]
    latent_tokens: dict[str, list[int]]
    target_strings: str
    latent_strings: dict[str, str | list[str]]
    input_string: Optional[str | list[str]] = None
    word_original: Optional[str] = None
    fs_examples: Optional[list[str]] = None

    @classmethod
    def from_strings(
        cls, prompt, target_strings, latent_strings, tokenizer, augment_token=False
    ):
        target_tokens, latent_tokens = cls.get_target_latent_tokens(
            target_strings, latent_strings, tokenizer, augment_token
        )
        return cls(
            target_tokens=target_tokens,
            latent_tokens=latent_tokens,
            target_strings=target_strings,
            latent_strings=latent_strings,
            prompt=prompt,
        )

    @staticmethod
    def get_target_latent_tokens(target_strings, latent_strings, tokenizer):
        target_tokens = process_tokens_with_tokenization(target_strings, tokenizer)
        latent_tokens = {
            lang: process_tokens_with_tokenization(words, tokenizer)
            for lang, words in latent_strings.items()
        }
        return target_tokens, latent_tokens

    def get_target_probs(self, probs):
        target_probs = probs[:, :, self.target_tokens].sum(dim=2)
        return target_probs.cpu()

    def get_latent_probs(self, probs, layer=None):
        latent_probs = {
            lang: probs[:, :, tokens].sum(dim=2).cpu()
            for lang, tokens in self.latent_tokens.items()
        }
        if layer is not None:
            latent_probs = {
                lang: probs_[:, layer] for lang, probs_ in latent_probs.items()
            }
        return latent_probs

    def retokenize(self, new_tokenizer):
        self.target_tokens = process_tokens_with_tokenization(
            self.target_strings, new_tokenizer
        )
        self.latent_tokens = {
            lang: process_tokens_with_tokenization(words, new_tokenizer)
            for lang, words in self.latent_strings.items()
        }


def build_prompt_str(
    df,
    word,
    fs_words,
    input_lang,
    target_lang,
    cut_at_obj=False,
):
    pref_input = f"{lang2name[input_lang]}: "
    pref_target = f"{lang2name[target_lang]}: "
    prompt = ""
    for fs_word in fs_words:
        fs_row = df.loc[df["word_original"] == fs_word].squeeze()
        in_word = fs_row[input_lang]
        target_word = fs_row[target_lang]
        if isinstance(in_word, list):
            in_word = in_word[0]
        if isinstance(target_word, list):
            target_word = target_word[0]
        prompt += f'{pref_input}"{in_word}" - {pref_target}"{target_word}"\n'

    input_word = df.loc[df["word_original"] == word].squeeze()[input_lang]
    if isinstance(input_word, list):
        input_word = input_word[0]
    prompt += f'{pref_input}"{input_word}'
    if not cut_at_obj:
        prompt += f'" - {pref_target}"'
    return prompt


def translation_prompt(
    row,
    prompt_str,
    tokenizer,
    input_lang,
    target_lang,
    latent_langs=None,
    fs_examples=None,
):
    target_words = row[target_lang]
    target_tokens = process_tokens_with_tokenization(target_words, tokenizer)
    latent_tokens = {}
    latent_words = {}
    for lang in latent_langs:
        l_words = row[lang]
        latent_words[lang] = l_words
        latent_tokens[lang] = process_tokens_with_tokenization(l_words, tokenizer)
    if len(target_tokens) > 0 and all(
        len(latent_tokens_) > 0 for latent_tokens_ in latent_tokens.values()
    ):
        return Prompt(
            prompt_str,
            target_tokens,
            latent_tokens,
            target_words,
            latent_words,
            row[input_lang],
            word_original=row["word_original"],
            fs_examples=fs_examples,
        )


def update_target_prompt(src_p, targ_p, target_lang):
    source_concept = set(s_p.word_original for s_p in src_p)
    assert len(source_concept) == 1
    # src_p is a list of prompts, targ_p is a single prompt,
    # src_p[n].latent_tokens should be the same for all n

    updated_latent_tokens = dict()
    for lang, tokens in targ_p.latent_tokens.items():
        if lang == target_lang:
            continue
        updated_latent_tokens[f"tgt_{lang}"] = tokens

    for lang, tokens in src_p[0].latent_tokens.items():
        updated_latent_tokens[f"src_{lang}"] = tokens

    if f"src_{target_lang}" not in updated_latent_tokens:
        for s_p in src_p:
            if target_lang in s_p.latent_tokens:
                updated_latent_tokens[f"src_{target_lang}"] = s_p.latent_tokens[
                    target_lang
                ]
                break

    updated_latent_strings = dict()
    for lang, strs in targ_p.latent_strings.items():
        if lang == target_lang:
            continue
        updated_latent_strings[f"tgt_{lang}"] = strs

    for lang, strs in src_p[0].latent_strings.items():
        updated_latent_strings[f"src_{lang}"] = strs

    if f"src_{target_lang}" not in updated_latent_strings:
        for s_p in src_p:
            if target_lang in s_p.latent_tokens:
                updated_latent_tokens[f"src_{target_lang}"] = s_p.latent_strings[
                    target_lang
                ]
                break
    targ_p.latent_tokens = updated_latent_tokens
    targ_p.latent_strings = updated_latent_strings

    return targ_p


def get_obj_id(sample_prompt, tokenizer):
    """
    For a prompt with the format '..."object" - X: "', return the index of the last token of the object.
    """
    split = sample_prompt.split('"')
    start = '"'.join(split[:-2])
    end = '"' + '"'.join(split[-2:])
    tok_start = tokenizer.encode(start, add_special_tokens=False)
    tok_end = tokenizer.encode(end, add_special_tokens=False)
    full = tokenizer.encode(sample_prompt, add_special_tokens=False)
    if tok_start + tok_end != full:
        raise ValueError("This is weird, check code")
    return -len(tok_end) - 1
