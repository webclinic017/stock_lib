# -*- coding: utf-8 -*-
import numpy
import utils
import simulator
from strategy import CombinationCreator
from loader import Loader

class CombinationStrategy(CombinationCreator):
    def __init__(self, setting):
        setting.simple = True
        super().__init__(setting)

    def subject(self, date):
        stocks = Loader.before_ranking(date, "ma_divergence")
        if stocks is None:
            return []
        stocks = stocks[(stocks["key"] <= -0.3) & (stocks["key"] >= -0.45)]
        codes = stocks["code"].as_matrix().tolist()
        return codes

    def common(self):
        default = self.default_common()
        default.new = [
            lambda d: d.position.num() == 0,
        ]

        default.taking = [
            lambda d: d.position.num() > 0,
        ]

        default.stop_loss = [
#            lambda d: d.position.gain_rate(self.price(d)) < -d.setting.stop_loss_rate, # 損益が-2%
        ]

        return default
