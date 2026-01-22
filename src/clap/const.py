import os

ROOT_DIR = os.getenv(
    "ROOT_DIR",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
DATA_DIR = f"{ROOT_DIR}/data"
MULTI_SIMLEX_PATH = f"{DATA_DIR}/multi_simlex.csv"
MULTI_SIMLEX_PROCESSED_PATH = f"{DATA_DIR}/multi_simlex_processed.csv"
PROMPTS_CACHE = f"{DATA_DIR}/prompts_cache"
