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

        if len(setting.seed) > 1:
            # seed値で既存の設定を利用する範囲を決める
            new, taking, stop_loss = [], [], []
            for i, seed in enumerate(setting.seed[1:]):
                random.seed(seed)
                new         = new       + [random.choice(self.new_conditions)]
                taking      = taking    + [random.choice(self.taking_conditions)]
                stop_loss   = stop_loss + [random.choice(self.stop_loss_conditions)]

            # 足りない条件は全体から追加する
            self.new_conditions         = new       + random.sample(self.conditions_all, setting.condition_size - len(new))
            self.taking_conditions      = taking    + random.sample(self.conditions_all, setting.condition_size - len(taking))
            self.stop_loss_conditions   = stop_loss + random.sample(self.conditions_all, setting.condition_size - len(stop_loss))


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

