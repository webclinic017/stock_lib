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
        setting.strict = True
        super().__init__(setting)
        self.weights = setting.weights
        self.conditions_by_seed(setting.seed[0])

    def conditions_index(self):
        return self.selected_condition_index

    def load_portfolio(self, date):
        d = utils.to_format(date)
        try:
            data = pandas.read_csv("portfolio/low_update/%s.csv" % d, header=None)
            data.columns = ["code", "price", "count", "date"]
            data = data[data["price"] <= (self.setting.assets / 500)]
            data = data.iloc[:10]
        except:
            data = None
        return data

    def select_dates(self, start_date, end_date, instant):
        dates = super().select_dates(start_date, end_date, instant)
        if instant:
            return [utils.to_datetime(start_date)]
        else:
            return list(set(map(lambda x: datetime(x.year, x.month, 1), dates)))

    def subject(self, date):
        data = self.load_portfolio(utils.to_datetime(date))
        if data is None:
            codes = []
        else:
            codes = data["code"].values.tolist()
        return codes

    def conditions_by_seed(self, seed):
        random.seed(seed)
        numpy.random.seed(seed)

        targets = ["daily", "nikkei", "dow"]
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

        return any(conditions)

    def common(self, settings):
        default = self.default_common()
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
