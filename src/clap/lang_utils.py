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

ellm_langs = {
    "bg",  # Bulgarian
    "hr",  # Croatian
    "cs",  # Czech
    "da",  # Danish
    "nl",  # Dutch
    "en",  # English
    "et",  # Estonian
    "fi",  # Finnish
    "fr",  # French
    "de",  # German
    "el",  # Greek
    "hu",  # Hungarian
    "ga",  # Irish
    "it",  # Italian
    "lv",  # Latvian
    "lt",  # Lithuanian
    "mt",  # Maltese
    "pl",  # Polish
    "pt",  # Portuguese
    "ro",  # Romanian
    "sk",  # Slovak
    "sl",  # Slovenian
    "es",  # Spanish
    "sv",  # Swedish
    "ar",  # Arabic
    "ca",  # Catalan
    "zh",  # Chinese
    "gl",  # Galician
    "hi",  # Hindi
    "ja",  # Japanese
    "ko",  # Korean
    "no",  # Norwegian
    "ru",  # Russian
    "tr",  # Turkish
    "uk",  # Ukrainian
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


def get_ellm_langs(extended: bool = True) -> set[str]:
    ellm_langs_extended = ellm_langs.copy()
    if extended:
        ellm_langs_extended.add("yue")
    return ellm_langs_extended
