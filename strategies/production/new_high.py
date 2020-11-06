# -*- coding: utf-8 -*-
import numpy
import utils
import simulator
import random
import subprocess
import pandas
from datetime import datetime
from dateutil.relativedelta import relativedelta
from strategy import CombinationCreator
from loader import Loader

from strategies.production import conditions

class CombinationStrategy(CombinationCreator):
    def __init__(self, setting):
        super().__init__(setting)
        self.weights = setting.weights
        self.conditions_by_seed(setting.seed[0])

    def conditions_index(self):
        return self.selected_condition_index

    def load_portfolio(self, date):
        d = utils.to_format(datetime(date.year, date.month, 1))
        try:
            data = pandas.read_csv("portfolio/new_high/%s.csv" % d, header=None)
            data.columns = ["code", "price"]
#            data = data[data["price"] <= (self.setting.assets / 500)]
            data = data[data["price"] <= 5000]
            data = data.iloc[:10]
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

    def apply_weights(self, method):
        base = numpy.array([1] * len(self.conditions_all))

        if method in self.weights.keys():
            for index, weight in self.weights[method].items():
                base[int(index)] = weight

        weights = base / sum(base)
        return weights

    def conditions_by_seed(self, seed):
        random.seed(seed)
        numpy.random.seed(seed)

        targets = ["daily", "nikkei", "dow"]
        self.conditions_all         = conditions.all_with_index(targets)

        new, self.new_conditions               = self.choice(self.conditions_all, self.setting.condition_size, self.apply_weights("new"))
        taking, self.taking_conditions         = self.choice(self.conditions_all, self.setting.condition_size, self.apply_weights("taking"))
        stop_loss, self.stop_loss_conditions   = self.choice(self.conditions_all, self.setting.condition_size, self.apply_weights("stop_loss"))
        closing, self.closing_conditions       = self.choice(self.conditions_all, self.setting.condition_size, self.apply_weights("closing"))
        x2, self.x2_conditions                 = self.choice(self.conditions_all, self.setting.condition_size, self.apply_weights("x2"))
        x4, self.x4_conditions                 = self.choice(self.conditions_all, self.setting.condition_size, self.apply_weights("x4"))
        x8, self.x8_conditions                 = self.choice(self.conditions_all, self.setting.condition_size, self.apply_weights("x8"))

        # 選択された条件のインデックスを覚えておく
        self.selected_condition_index = {
            "new":new, "taking": taking, "stop_loss": stop_loss, "x2": x2, "x4": x4, "x8": x8
        }

    def break_precondition(self, d):
        conditions = [
            d.data.daily["high_update"][-2:].max() == 0 and (d.position.gain(self.price(d), d.position.get_num()) <= 0 or sum(d.stats.gain()) <= 0) and d.position.get_num() >= 0,
            d.data.daily["high_update"][-10:].sum() <= 5
        ]

        return any(conditions)

    def common(self, setting):
        default = self.default_common()
        default.new = [
            lambda d: d.index.data["new_score"].daily["score"].iloc[-1] > -400,
            lambda d: d.data.daily["stop_low"].iloc[-1] == 0
        ]

        default.closing = [
            lambda d: self.break_precondition(d),
        ]

        for i in range(1, len(setting[1:])):
            default.new         = default.new           + [lambda d: self.apply(utils.combination(setting[i].new, self.new_conditions))]
            default.taking      = default.taking        + [lambda d: self.apply(utils.combination(setting[i].taking, self.taking_conditions))]
            default.stop_loss   = default.stop_loss     + [lambda d: self.apply(utils.combination(setting[i].stop_loss, self.stop_loss_conditions))]
            default.closing     = default.closing       + [lambda d: self.apply(utils.combination(setting[i].closing, self.closing_conditions))]
            default.x2          = default.x2            + [lambda d: self.apply(utils.combination(setting[i].x2, self.x2_conditions))]
            default.x4          = default.x4            + [lambda d: self.apply(utils.combination(setting[i].x4, self.x4_conditions))]
            default.x8          = default.x8            + [lambda d: self.apply(utils.combination(setting[i].x8, self.x8_conditions))]
            self.conditions_by_seed(self.setting.seed[i])

        return default

    def new(self):
        return self.new_conditions

    def taking(self):
        return self.taking_conditions

    def stop_loss(self):
        return self.stop_loss_conditions

    def closing(self):
        return self.closing_conditions

    def x2(self):
        return self.x2_conditions

    def x4(self):
        return self.x4_conditions

    def x8(self):
        return self.x8_conditions
