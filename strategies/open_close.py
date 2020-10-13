# -*- coding: utf-8 -*-
import numpy
import utils
import simulator
import conditions
import random
import subprocess
import pandas
from datetime import datetime
from dateutil.relativedelta import relativedelta
from strategy import CombinationCreator
from loader import Loader

class CombinationStrategy(CombinationCreator):
    def __init__(self, setting):
        setting.position_adjust = False
        setting.simple["closing"] = True
        super().__init__(setting)
        self.weights = setting.weights
        self.conditions_by_seed(setting.seed[0])

    def conditions_index(self):
        return self.selected_condition_index

    def load_portfolio(self, date):
        d = utils.to_format(date)
        try:
            data = pandas.read_csv("portfolio/high_update/%s.csv" % d, header=None)
            data.columns = ["code", "price", "count", "date"]
            data = data[data["price"] <= (self.setting.assets / 250)]
            data = data.iloc[:5]
        except:
            data = None
        return data

    def subject(self, date):
        data = self.load_portfolio(utils.to_datetime(date))
        if data is None:
            codes = []
        else:
            codes = data["code"].values.tolist()
        return codes

    def choice(self, conditions, size, weights):
        conditions_with_index = list(map(lambda x: {"x": x}, list(enumerate(conditions))))
        choiced = numpy.random.choice(conditions_with_index, size, p=weights, replace=False).tolist()
        choiced = list(map(lambda x: x["x"], choiced))
        return list(zip(*choiced))

    def apply_weights(self, method):
        base = numpy.array([200] * len(self.conditions_all))

        if method in self.weights.keys():
            for index, weight in self.weights[method].items():
                base[int(index)] = base[int(index)] + weight

        weights = base / sum(base)
        return weights

    def conditions_by_seed(self, seed):
        random.seed(seed)
        numpy.random.seed(seed)

        targets = ["daily", "nikkei", "dow"]
        self.conditions_all         = conditions.all_with_index(targets)

        new, self.new_conditions               = self.choice(self.conditions_all, self.setting.condition_size, self.apply_weights("new"))
        x2, self.x2_conditions                 = self.choice(self.conditions_all, self.setting.condition_size, self.apply_weights("x2"))
        x4, self.x4_conditions                 = self.choice(self.conditions_all, self.setting.condition_size, self.apply_weights("x4"))
        x8, self.x8_conditions                 = self.choice(self.conditions_all, self.setting.condition_size, self.apply_weights("x8"))

        # 選択された条件のインデックスを覚えておく
        self.selected_condition_index = {
            "new":new, "x2": x2, "x4": x4, "x8": x8
        }

    def common(self, setting):
        default = self.default_common()

        default.closing = [
            lambda d: d.position.get_num() > 0,
        ]

        return default

    def new(self):
        return self.new_conditions

    def x2(self):
        return self.x2_conditions

    def x4(self):
        return self.x4_conditions

    def x8(self):
        return self.x8_conditions
