import json
import pandas as pd
from typing import Text
import yaml
import random
import numpy as np
from pathlib import Path
import argparse
from abc import ABC, abstractmethod
import sys
import os

sys.path.append(os.path.abspath('../SBRS'))
from src.stages.train import train_apriori, train_eclat
from src.stages import preprocessing
from src.report.visualize import session_hist, item_hist


class Association_Rules(ABC):
    """
    Астрактный класс для алгоритмов Apriori and Eclat
    """

    @abstractmethod
    def get_test_session(self, df: pd.DataFrame):
        sessions = df.session_key.unique()

        random_session = random.choice(sessions)
        items_arr = np.array(df[df.session_key == random_session].item_key)
        return items_arr, random_session

    @abstractmethod
    def get_rank(self, predictions, item):
        for i, tuple_ in enumerate(predictions):
            if tuple_[0] == str(item):
                return i + 1
        else:
            return 0


class Apriori(Association_Rules, ABC):
    def __init__(self, df: pd.DataFrame, items, model):
        self.data = df
        self.model = model
        self.items = items

    def get_test_session(self):
        return super().get_test_session(self.data)

    def get_rank(self, predictions, item):
        return super().get_rank(predictions, item)

    def get_predict(self, session):
        if len(session) == 1:
            predictions = self.model.predict_next('1', session[0], self.items)
        else:
            for i in range(1, len(session)):
                session_name = str(i)
                self.model.predict_next(session_name, session[:i + 1], self.items, skip=True)
            predictions = self.model.predict_next(session_name, session[-1], self.items)

        predictions = predictions[predictions > 0.655]
        name = predictions.index.to_list()

        for i in range(len(name)):
            name[i] = str(name[i])

        value = predictions.to_list()
        prd = [*zip(name, value)]
        sort_prd = sorted(prd, key=lambda x: x[1], reverse=True)

        return sort_prd

    def get_metric(self, test_session):
        sum_ = 0
        len_items_arr = len(test_session) - 1
        if len_items_arr == 0:
            return 0
        else:
            for i in range(0, len_items_arr):
                pred = self.get_predict(test_session[:i + 1])
                try:
                    if pred == []:
                        return 0
                    else:
                        sum_ += 1 / self.get_rank(pred, test_session[i + 1])
                except ZeroDivisionError:
                    sum_ += 0
        if sum_ > 0:
            return 1 / round(sum_ / len_items_arr, 4)
        else:
            return 0


class Eclat(Association_Rules, ABC):

    def __init__(self, df: pd.DataFrame, items, model):
        self.data = df
        self.model = model
        self.items = items

    def get_test_session(self):
        return super().get_test_session(self.data)

    def get_rank(self, predictions, item):
        temp_dict = {}
        for elem in predictions:
            if elem[1] in temp_dict.keys():
                temp_dict[elem[1]].append(elem[0])
            else:
                temp_dict[elem[1]] = [elem[0]]
        for i, key in enumerate(temp_dict.keys()):
            if item in temp_dict[key]:
                return i + 1
        else:
            return 0

    def key_check(self, items_str, count_element):
        return [(key, value) for key, value in self.model.items() if
                key.startswith(items_str) and key.count('|') == count_element]

    def get_metric(self, test_session):
        full_test_session_len = len(test_session) - 1
        sum_ = 0
        if len(test_session) == 1:
            return 0
        else:
            i = 0
            items_str = test_session[i]
            next_elem = ''
            while i < len(test_session) - 1:
                # print('SUM = ', sum_)
                # print(items_str, i, len(test_session))
                next_elem = items_str + '|' + test_session[i + 1]
                try:
                    sum_ += 1 / self.get_rank(self.key_check(items_str, i + 1), next_elem)
                    i += 1
                    # print('try', items_str, next_elem, sum_)
                    items_str, next_elem = next_elem, ''
                except ZeroDivisionError:
                    # print('except', items_str, next_elem, sum_)
                    if i > 0:
                        new_items_str = test_session[i]
                        new_next_elem = new_items_str + '|' + test_session[i + 1]
                        new_sum = self.get_rank(self.key_check(new_items_str, i - 1), new_next_elem)
                        if new_sum == 0:
                            items_str = test_session[i + 1]
                            test_session = test_session[i + 1:]
                            i = 0
                        else:
                            sum_ += 1 / new_sum
                            items_str = new_next_elem
                        # print('except_in', new_items_str, new_next_elem)
                    else:
                        items_str = test_session[i + 1]
                        test_session = test_session[i + 1:]
                        i = 0
            if sum_ > 0:
                return 1 / round(sum_ / full_test_session_len, 4)
            else:
                return 0


def evaluate_model_MRR(config_path: Text):
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    data_apriori = pd.read_csv(config['data_load']['dataset_xlsx'])
    data_eclat = pd.read_csv(config['data_load']['dataset_xlsx'])
    data_eclat = preprocessing.prepr(data_eclat[:201], session_key='session_key', item_key='item_key')

    items = data_apriori.item_key.unique()

    # модели
    AR = train_apriori(
        df=data_apriori,
        pruning=config['train']['apriori']['pruning'],
        session_key=config['train']['apriori']['session_key'],
        item_key=config['train']['apriori']['item_key'],
    )

    ECLAT = train_eclat(
        data_eclat,
        min_support=config['train']['eclat']['min_support'],
        max_items=config['train']['eclat']['max_items'],
        min_items=config['train']['eclat']['min_items'],
        session_key='session_key',
        item_key='item_key'
    )
    # Создаение объектов класса
    apriori_ = Apriori(df=data_apriori, model=AR, items=items)
    eclat_ = Eclat(df=data_eclat, model=ECLAT, items=items)

    # test_session, rand_session = apriori_.get_test_session()

    test_session, rand_session = ['22487', '20724', '22356', '20723', '20719', '22144', '22141', '22142', '22147',
                                  '22150', '22149', '22501', '22752', '21071', '20725', '82483', '37370'], 'test'

    # test_session, rand_session = [22487, 20724, 22356, 20723, 20719, 22144, 22141, 22142, 22147,
    #                               22150, 22149, 22501, 22752, 21071, 20725, 82483, 37370], 'test'

    mrr_apriori = apriori_.get_metric(test_session)
    print(mrr_apriori)
    mrr_eclat = eclat_.get_metric(test_session)

    reports_folder = Path(config['evaluate']['reports_dir'])
    metrics_path = reports_folder / config['evaluate']['metrics_file']

    json.dump(
        obj={
            f'apriori_mrr_{rand_session}': mrr_apriori,
            f'eclat_mrr_{rand_session}': mrr_eclat},
        fp=open(metrics_path, 'w')
    )

    plot_png_path_first = reports_folder / config['evaluate']['item_hist']
    plot_png_path_second = reports_folder / config['evaluate']['session_hist']

    plt_first = item_hist(data_apriori)
    plt_first.savefig(plot_png_path_first)
    plt_second = session_hist(data_apriori)
    plt_second.savefig(plot_png_path_second)
    print('Complite!')


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config_path', dest='config', required=True)
    args = args_parser.parse_args()

    evaluate_model_MRR(config_path=args.config)
