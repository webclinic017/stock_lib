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

from portfolio import futures

class CombinationStrategy(CombinationCreator):
    def __init__(self, setting):
        setting.position_adjust = False
        setting.strict = False
        super().__init__(setting)
        self.weights = setting.weights
        self.conditions_by_seed(setting.seed[0])

    def subject(self, date):
        return futures.load_portfolio(date, self.setting.assets, by_day=self.setting.portfolio_by_day)["code"].values.tolist()

    def common(self, settings):
        default = self.default_common()
        default = futures.common(default)
        return default

