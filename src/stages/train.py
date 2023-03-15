import pandas as pd
from src.algoritms import association_rules as ar
from typing import Text


def train(df: pd.DataFrame, pruning: int,
          session_key: Text, item_key: Text, algoritm: Text):
    if algoritm == 'apriori':
        AR = ar.AssociationRules(
            pruning=pruning,
            session_key=session_key,
            item_key=item_key
        )
        AR.fit(df)

        return AR

    elif algoritm == 'eclat':
        return 0

    else:
        return 0
