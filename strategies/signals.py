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

from portfolio import signals as portfolio
from portfolio import high_update as high_update

class CombinationStrategy(CombinationCreator):
    def __init__(self, setting):
        if setting.portfolio is None:
            raise Exception("need --portfolio")
        setting.position_adjust = False
        setting.strict = True
        super().__init__(setting)
        self.weights = setting.weights
        self.conditions_by_seed(setting.seed[0])

    def load_portfolio(self, date, length=10):
#        limit = min([self.setting.assets / 250, 15000])
        limit = (self.setting.assets / 250) if self.setting.portfolio_limit is None else self.setting.portfolio_limit
        return portfolio.load_portfolio(self.setting.portfolio, date, limit, length, by_day=self.setting.portfolio_by_day)

    def subject(self, date):
        length = 10 if self.setting.portfolio_size is None else self.setting.portfolio_size
        data = self.load_portfolio(utils.to_datetime(date), length=length)
        market = high_update.load_portfolio(utils.to_datetime(date), self.setting.assets, 1000)
        if data is None:# or market is None or len(market) < 15 or len(market) > 550:
            codes = []
        else:
            codes = data["code"].values.tolist()
        return codes

    def common(self, setting):
        default = self.default_common()
        default = portfolio.common(default)
        return default

