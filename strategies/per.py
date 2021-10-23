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

from portfolio import per

class CombinationStrategy(CombinationCreator):
    def __init__(self, setting):
        setting.position_adjust = False
        setting.strict = True
        super().__init__(setting)
        self.weights = setting.weights
        self.conditions_by_seed(setting.seed[0])

    def load_portfolio(self, date, length=10):
        return per.load_portfolio(date, length)

    def subject(self, date):
        length = 10 if self.setting.portfolio_size is None else self.setting.portfolio_size
        data = self.load_portfolio(utils.to_datetime(date), length=length)
        if data is None:
            codes = []
        else:
            codes = data["code"].values.tolist()
        return codes

    def common(self, settings):
        default = self.default_common()
        default = per.common(default)
        return default
