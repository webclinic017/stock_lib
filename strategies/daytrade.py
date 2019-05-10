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
        self.conditions_all         = conditions.all()
        setting.sorted_conditions = False

        super().__init__(setting)

        self.conditions_by_seed(setting.seed[0])

    def conditions_by_seed(self, seed):
        random.seed(seed)

        self.new_conditions         = random.sample(self.conditions_all, self.setting.condition_size)
        self.taking_conditions      = random.sample(self.conditions_all, self.setting.condition_size)
        self.stop_loss_conditions   = random.sample(self.conditions_all, self.setting.condition_size)

    def subject(self, date):
        stocks = Loader.before_ranking(date, "volume")
        if stocks is None:
            return []

        num = self.setting.monitor_size
        codes = stocks["code"].iloc[:num].as_matrix().tolist()

        return codes

    def common(self, setting):
        default = self.default_common()
        default.new = [
        ]

        default.taking = [
            lambda d: d.position.gain(self.price(d)) > 0,
        ]

        default.stop_loss = [
            lambda d: d.position.gain(self.price(d)) < 0,
        ]

        for i in range(1, len(setting[1:])):
            default.new         = default.new           + [lambda d: self.apply(utils.combination(setting[i].new, self.new_conditions))]
            default.taking      = default.taking        + [lambda d: self.apply(utils.combination(setting[i].taking, self.taking_conditions))]
            default.stop_loss   = default.stop_loss     + [lambda d: self.apply(utils.combination(setting[i].stop_loss, self.stop_loss_conditions))]
            self.conditions_by_seed(self.setting.seed[i])

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

