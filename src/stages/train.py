import pandas as pd
import sys
import os
from typing import Text

sys.path.append(os.path.abspath('../SBRS'))
from src.algoritms import association_rules as ar
from src.algoritms import eclat


def train_apriori(df: pd.DataFrame, pruning: int,
                  session_key: Text, item_key: Text):
    AR = ar.AssociationRules(
        pruning=pruning,
        session_key=session_key,
        item_key=item_key
    )
    AR.fit(df)

    return AR


def train_eclat(df: pd.DataFrame, min_support: float, max_items: int, min_items: int,
                session_key: Text, item_key: Text):

    ECLAT = eclat.Eclat(min_support=min_support,
                        max_items=max_items,
                        min_items=min_items
    )
    ECLAT.fit(df)
    return ECLAT.transform()

