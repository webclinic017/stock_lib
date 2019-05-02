# -*- coding: utf-8 -*-
import numpy
import utils
import simulator
import conditions
from strategy import CombinationCreator
from loader import Loader
import random

class CombinationStrategy(CombinationCreator):
    def __init__(self, setting):
        self.conditions_all = numpy.array(conditions.all())
        setting.sorted_conditions = False

        random.seed(setting.seed)

        self.new_conditions         = random.sample(list(range(len(self.conditions_all))), 5)
        self.taking_conditions      = random.sample(list(range(len(self.conditions_all))), 5)
        self.stop_loss_conditions   = random.sample(list(range(len(self.conditions_all))), 5)

#        print("new:", self.new_conditions)
#        print("taking:", self.taking_conditions)
#        print("stop_loss:", self.stop_loss_conditions)

        self.new_conditions         = self.conditions_all[self.new_conditions].tolist()
        self.taking_conditions      = self.conditions_all[self.taking_conditions].tolist()
        self.stop_loss_conditions   = self.conditions_all[self.stop_loss_conditions].tolist()


        super().__init__(setting)

    def subject(self, date):
        return ["nikkei"]

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

