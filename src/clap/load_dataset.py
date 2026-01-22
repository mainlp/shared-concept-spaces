import pandas as pd

from clap.utils import safe_eval
from clap.const import MULTI_SIMLEX_PROCESSED_PATH


def multi_simlex_from_csv(path=None):
    if path is None:
        path = MULTI_SIMLEX_PROCESSED_PATH
    df = pd.read_csv(path)
    df = df.map(safe_eval)
    return df
