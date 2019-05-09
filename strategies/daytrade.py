# -*- coding: utf-8 -*-
import numpy
import utils
import simulator
import conditions
import random
from strategy import CombinationCreator
from loader import Loader

class CombinationStrategy(CombinationCreator):
    def __init__(self, setting):
        self.conditions_all = conditions.all()
        setting.sorted_conditions = False

        random.seed(setting.seed[0])

        self.new_conditions         = random.sample(self.conditions_all, setting.condition_size)
        self.taking_conditions      = random.sample(self.conditions_all, setting.condition_size)
        self.stop_loss_conditions   = random.sample(self.conditions_all, setting.condition_size)

        super().__init__(setting)

    def subject(self, date):
        stocks = Loader.before_ranking(date, "volume")
        if stocks is None:
            return []

        num = self.setting.monitor_size
        codes = stocks["code"].iloc[:num].as_matrix().tolist()

        return codes

    def common(self):
        default = self.default_common()
        default.new = [
        ]

        default.taking = [
            lambda d: d.position.gain(self.price(d)) > 0,
        ]

        default.stop_loss = [
            lambda d: d.position.gain(self.price(d)) < 0,
        ]

        return default

    def new(self):
        return self.new_conditions

    def taking(self):
        return self.taking_conditions

    def stop_loss(self):
        return self.stop_loss_conditions

    def closing(self):
        return [
            lambda d: False,
        ]

