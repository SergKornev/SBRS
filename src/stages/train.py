import pandas as pd
import sys
import os
from typing import Text

sys.path.append(os.path.abspath('../SBRS'))
from src.algoritms import association_rules as ar


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
