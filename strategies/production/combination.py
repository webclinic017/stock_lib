# -*- coding: utf-8 -*-
import math
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
        setting.strict = True
        super().__init__(setting)
        self.weights = setting.weights
        self.conditions_by_seed(setting.seed[0])

    def load_portfolio(self, date, length=10):
        d = utils.to_format(date)
        try:
            data = pandas.read_csv("portfolio/high_update/%s.csv" % d, header=None)
            data.columns = ["code", "price", "count", "date"]
            data = data[data["price"] <= (self.setting.assets / 500)]

            if len(data) < 50 or len(data) > 550:
                data = None
            else:
                data = data.iloc[:length]
        except:
            data = None
        return data

    def subject(self, date):
        before = self.load_portfolio(utils.to_datetime(date) - utils.relativeterm(1))

        # 前月のポートフォリオの状況次第で変える
        length = 10
        length = int(length/2) if before is None else length

        data = self.load_portfolio(utils.to_datetime(date), length=length)
        if data is None:
            codes = []
        else:
            codes = data["code"].values.tolist()
        return codes

    def conditions_by_seed(self, seed):
        random.seed(seed)
        numpy.random.seed(seed)

        targets = ["middle", "nikkei", "dow"]
        self.conditions_all         = conditions.all_with_index(targets) + conditions.industry_score_conditions()

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
            d.data.middle["high_update"][-2:].max() == 0 and (d.position.gain(self.price(d), d.position.get_num()) <= 0 or sum(d.stats.gain()) <= 0) and d.position.get_num() >= 0,
            d.data.middle["high_update"][-10:].sum() <= 5
        ]

        return any(conditions)

    def common(self, settings):
        default = self.default_common()
        default.new = [
            lambda d: d.index.data["new_score"].middle["score"].iloc[-1] > -400,
            lambda d: d.data.middle["stop_low"].iloc[-1] == 0,
            lambda d: not self.break_precondition(d),
        ]

        default.closing = [
            lambda d: self.break_precondition(d)
        ]

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
