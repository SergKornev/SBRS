import json
import pandas as pd
from typing import Text
import yaml
import random
import numpy as np
from train import train
from pathlib import Path
from src.report.visualize import session_hist, item_hist
import argparse


def generating_test_session(df: pd.DataFrame):
    sessions = df.session_key.unique()

    random_session = random.choice(sessions)
    items_arr = np.array(df[df.session_key == random_session].item_key)
    return items_arr, random_session


def rang_elements(predictions, item):
    for i, tuple_ in enumerate(predictions):
        if tuple_[0] == str(item):
            return i + 1
    else:
        return 0


def predict(df, session, model, items):
    if len(session) == 1:
        predictions = model.predict_next('1', session[0], items)
    else:
        for i in range(1, len(session)):
            session_name = str(i)
            model.predict_next(session_name, session[:i + 1], items, skip=True)
        predictions = model.predict_next(session_name, session[-1], items)

    predictions = predictions[predictions > 0.655]
    name = predictions.index.to_list()

    for i in range(len(name)):
        name[i] = str(name[i])

    value = predictions.to_list()
    prd = [*zip(name, value)]
    sort_prd = sorted(prd, key=lambda x: x[1], reverse=True)

    return sort_prd


def evaluate_model_MRR(config_path: Text) -> float:
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    data = pd.read_csv(config['data_load']['dataset_xlsx'])
    items = data.item_key.unique()

    AR = train(
        df=data,
        pruning=config['train']['apriori']['pruning'],
        session_key=config['train']['apriori']['session_key'],
        item_key=config['train']['apriori']['item_key'],
        algoritm='apriori'
    )

    items_arr, rand_session = generating_test_session(data)

    sum = 0
    len_items_arr = len(items_arr) - 1
    if len_items_arr == 0:
        return 0
    else:
        for i in range(0, len_items_arr):
            pred = predict(data, items_arr[:i + 1], AR, items)
            try:
                if pred == []:
                    return 0
                else:
                    sum += 1 / rang_elements(pred, items_arr[i + 1])
            except ZeroDivisionError:
                sum += 0
    MRR = 1 / (round(sum / len_items_arr, 4))

    reports_folder = Path(config['evaluate']['reports_dir'])
    metrics_path = reports_folder / config['evaluate']['metrics_file']

    json.dump(
        obj={
            f'apriori_mrr_{rand_session}': MRR},
        fp=open(metrics_path, 'w')
    )

    plot_png_path_first = reports_folder / config['evaluate']['item_hist']
    plot_png_path_second = reports_folder / config['evaluate']['session_hist']

    plt_first = item_hist(data)
    plt_first.savefig(plot_png_path_first)
    plt_second = session_hist(data)
    plt_second.savefig(plot_png_path_second)


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config_path', dest='config', required=True)
    args = args_parser.parse_args()

    evaluate_model_MRR(config_path=args.config)


