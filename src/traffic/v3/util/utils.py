import os

import pandas as pd


def save_or_extend_dataframe(dataframe, path: str):
    if os.path.exists(path):
        results = [pd.read_pickle(path), dataframe]
        pd.concat(results).reset_index(drop=True).to_pickle(path)
    else:
        dataframe.to_pickle(path)
