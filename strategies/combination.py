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

from portfolio import high_update

class CombinationStrategy(CombinationCreator):
    def __init__(self, setting):
        setting.position_adjust = False
        setting.strict = True
        super().__init__(setting)
        self.weights = setting.weights
        self.conditions_by_seed(setting.seed[0])

    def load_portfolio(self, date, length=10):
        return high_update.load_portfolio(date, self.setting.assets / 500, length)

    def select_dates(self, start_date, end_date, instant):
        dates = super().select_dates(start_date, end_date, instant)
        if instant:
            return [utils.to_datetime(start_date)]
        else:
            return list(set(map(lambda x: datetime(x.year, x.month, 1), dates)))

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
        targets = ["daily", "nikkei", "dow"]
        names = ["all", "industry_score"]
        super().conditions_by_seed(seed, targets, names)

    def break_precondition(self, d):
        conditions = [
            d.data.daily["high_update"][-2:].max() == 0 and (d.position.gain(self.price(d), d.position.get_num()) <= 0 or sum(d.stats.gain()) <= 0) and d.position.get_num() >= 0,
            d.data.daily["high_update"][-10:].sum() <= 5
        ]

        return any(conditions)

    def common(self, settings):
        default = self.default_common()
        default.new = [
            lambda d: d.index.data["new_score"].daily["score"].iloc[-1] > -400,
            lambda d: d.data.daily["stop_low"].iloc[-1] == 0,
            lambda d: not self.break_precondition(d),
        ]

        default.closing = [
            lambda d: self.break_precondition(d)
        ]

        return default

