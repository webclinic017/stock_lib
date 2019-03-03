# -*- coding: utf-8 -*-
import numpy
import utils
import simulator
import conditions
from strategy import CombinationCreator
from loader import Loader

class CombinationStrategy(CombinationCreator):
    def __init__(self, setting):
        self.conditions_all = conditions.all()
        setting.sorted_conditions = False
        super().__init__(setting)

    def subject(self, date):
        return ["nikkei"]

    def common(self):
        default = self.default_common()
        default.new = [
            #lambda d, s: d.data["weekly"]["rci_trend"].iloc[-1] > 0, # 週足が上昇トレンド
            #lambda d, s: d.data["weekly"]["daily_average_trend"].iloc[-1] > 0, # 週足が上昇トレンド
            lambda d, s: self.risk(d, s) < self.max_risk(d, s), # リスクが2%以内
            lambda d, s: self.risk(d, s) < self.goal(d, s), # リスクより利益のほうが大きい
        ]

        return default

    def new(self):
        return self.conditions_all

    def taking(self):
        return self.conditions_all + [
            lambda d, s: self.risk(d, s) > self.goal(d, s),
            lambda d, s: self.upper(d, s) < d.data["daily"]["high"].iloc[-1],
        ]

    def stop_loss(self):
        return self.conditions_all

    def closing(self):
        return [
            lambda d, s: False,
        ]

