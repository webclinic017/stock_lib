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
        stocks = stocks[(stocks["key"] <= -0.05) & (stocks["key"] >= -0.1)]
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

        return default

    def new(self):
        return [
            lambda d: False,
        ]

    def taking(self):
        return [
            lambda d: False,
        ]

    def stop_loss(self):
        return [
            lambda d: False,
        ]

    def closing(self):
        return [
            lambda d: False,
        ]

