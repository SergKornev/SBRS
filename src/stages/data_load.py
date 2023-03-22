import yaml
import pandas as pd
import argparse
from typing import Text


def data_load(config_path: Text) -> None:

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    raw_data_path = config['data_load']['dataset_xlsx']

    data = pd.read_excel(raw_data_path)
    data = data.drop(['Description', 'Quantity', 'UnitPrice', 'userid', 'Country'], axis=1)
    data.rename(columns={'transactionid': 'session_key', 'itemid': 'item_key', 'datetime': 'time_key'}, inplace=True)

    data.to_csv(config['data_load']['dataset_xlsx'], index=False)


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config_path', dest='config', required=True)
    args = args_parser.parse_args()

    data_load(config_path=args.config)
