from itertools import permutations

lang2name = {
    "fr": "Français",
    "de": "Deutsch",
    "ru": "Русский",
    "en": "English",
    "zh": "中文",
    "es": "Español",
    "ko": "한국어",
    "ja": "日本語",
    "it": "Italiano",
    "nl": "Nederlands",
    "et": "Eesti",
    "fi": "Suomi",
    "hi": "हिन्दी",
    "ar": "العربية",
    "cy": "Cymraeg",
    "he": "עברית",
    "pl": "Polski",
    "sw": "Kiswahili",
    "yue": "粵語",
}

simlex_id_to_iso_code = {
    "ENG": "en",
    "ARA": "ar",
    "CMN": "zh",
    "CYM": "cy",
    "EST": "et",
    "FIN": "fi",
    "FRA": "fr",
    "HEB": "he",
    "POL": "pl",
    "RUS": "ru",
    "SPA": "es",
    "SWA": "sw",
    "YUE": "yue",
}

SKIP_LANGS = {"ar", "he"}


def get_multi_simlex_langs(skip_langs: set[str] | None = None) -> set[str]:
    if skip_langs is None:
        skip_langs = SKIP_LANGS
    return {lang for lang in simlex_id_to_iso_code.values() if lang not in skip_langs}


def get_permutations(skip_langs: set[str] | None = None) -> list[tuple[str, str]]:
    if skip_langs is None:
        skip_langs = set()
    langs = get_multi_simlex_langs(skip_langs=skip_langs)
    lang_pairs = list(permutations(langs, 2))
    lang_pairs.append(("en", "en"))
    return lang_pairs
