import pandas as pd
from src.algoritms import association_rules as ar
from typing import Dict, Text

class UnsupportedClassifier(Exception):

    def __init__(self, estimator_name):

        self.msg = f'Unsupported estimator {estimator_name}'
        super().__init__(self.msg)


def get_supported_estimator() -> Dict:
    """
    Returns:
        Dict: supported classifiers
    """

    return {
        'apriori': ar
    }

def train(df: pd.DataFrame, target_column: Text,
          estimator_name: Text, param_grid: Dict,  cv: int):

    return clf