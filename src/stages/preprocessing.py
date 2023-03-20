import pandas as pd
from typing import Text


def prepr(df: pd.DataFrame, session_key: Text, item_key: Text):
    df = df[[session_key, item_key]]
    df[item_key] = df[item_key].astype('str')

    df[session_key] = pd.to_numeric(df[session_key], errors='coerce')
    df.dropna(axis=0, how='all', inplace=True)
    df[session_key] = df[session_key].astype('str')
    df = df.groupby(session_key).agg(lambda x: x.tolist())

    return df
