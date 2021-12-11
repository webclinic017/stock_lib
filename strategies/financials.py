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

from portfolio import financials as portfolio

class CombinationStrategy(CombinationCreator):
    def __init__(self, setting):
        setting.position_adjust = False
        setting.strict = True
        setting.appliable_signal = {"condition_size": ["new"]}
        super().__init__(setting)
        self.weights = setting.weights
        self.conditions_by_seed(setting.seed[0])

    def load_portfolio(self, date, length=10):
        return portfolio.load_portfolio(date, self.setting.assets / 500, length)

    def subject(self, date):
        before = self.load_portfolio(utils.to_datetime(date) - utils.relativeterm(1))

        # 前月のポートフォリオの状況次第で変える
        length = 10 if self.setting.portfolio_size is None else self.setting.portfolio_size
        length = int(length/2) if before is None else length

        data = self.load_portfolio(utils.to_datetime(date), length=length)
        if data is None:
            codes = []
        else:
            codes = data["code"].values.tolist()
        return codes

    def conditions_by_seed(self, seed):
        targets = ["middle", "nikkei", "dow"]
        names = ["industry_score"]
        super().conditions_by_seed(seed, targets, names)

    def common(self, settings):
        default = self.default_common()
        default = portfolio.common(default)
        return default

