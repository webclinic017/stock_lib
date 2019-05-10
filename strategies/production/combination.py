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

        self.conditions_by_seed(setting.seed[0])

        super().__init__(setting)

    def conditions_by_seed(self, seed):
        random.seed(seed)

        self.new_conditions         = random.sample(self.conditions_all, setting.condition_size)
        self.taking_conditions      = random.sample(self.conditions_all, setting.condition_size)
        self.stop_loss_conditions   = random.sample(self.conditions_all, setting.condition_size)


    def subject(self, date):
        return ["nikkei"]

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

