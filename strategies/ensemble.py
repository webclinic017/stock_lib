# -*- coding: utf-8 -*-
import numpy
import utils
import simulator
import random
import strategy
from strategy import CombinationCreator
from loader import Loader

class CombinationStrategy(CombinationCreator):
    def __init__(self, setting):
        super().__init__(setting)
        self.conditions_by_seed(setting.seed[0], setting.ensemble)

    def conditions_by_seed(self, seed, files):

        strategies = strategy.create_ensemble_strategies(files)

        new_rules         = list(map(lambda x: x.new_rules[0].callback, strategies))
        taking_rules      = list(map(lambda x: x.taking_rules[0].callback, strategies))
        stop_loss_rules   = list(map(lambda x: x.stop_loss_rules[0].callback, strategies))

        conditions_all = new_rules + taking_rules + stop_loss_rules

        size = len(strategies) if len(strategies) < self.setting.condition_size else self.setting.condition_size

        random.seed(seed)
        self.new_conditions         = random.sample(conditions_all, size)
        self.taking_conditions      = random.sample(conditions_all, size)
        self.stop_loss_conditions   = random.sample(conditions_all, size)

    def subject(self, date):
        return ["nikkei"]

    def common(self, setting):
        default = self.default_common()
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

