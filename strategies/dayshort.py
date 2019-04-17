# -*- coding: utf-8 -*-
import numpy
import utils
import simulator
from strategy import CombinationCreator
from loader import Loader

class CombinationStrategy(CombinationCreator):
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
            lambda d: self.risk(d) < self.max_risk(d), # リスクが2%以内
            lambda d: self.risk(d) < self.goal(d), # リスクより利益のほうが大きい
        ]
        default.stop_loss = [
            lambda d: self.drawdown(d) > (self.max_risk(d) / 2),
        ]

        return default

    def new(self):
        return [
            lambda d: d.data["daily"]["rci_long_trend"].iloc[-1] == -1, # 週足が上昇トレンド
            lambda d: d.data["daily"]["weekly_average_trend"].iloc[-1] == -1, # 週足が上昇トレンド
            lambda d: d.data["daily"]["stages_average"].iloc[-1] > 0,
            lambda d: d.data["daily"]["weekly_average_trend"].iloc[-1] == -1 and d.data["daily"]["daily_average_trend"].iloc[-1] >= 0,
            lambda d: d.data["daily"]["resistance"].iloc[-1] < d.data["daily"]["high"].iloc[-1], # 下方へのブレイクアウトではない
            lambda d: d.data["daily"]["stages"].iloc[-10:].max() > -2,
            lambda d: d.data["daily"]["macdhist_convert"].iloc[-1] == 1,
            lambda d: d.data["daily"]["rci_trend"].iloc[-1] == 1,
            lambda d: d.data["daily"]["average_cross"].iloc[-5:].max() == 0, # 平均線のデッドクロス直後ではない
            lambda d: d.data["daily"]["high_rounddown"].iloc[-1] == 1,
        ]

    def taking(self):
        return [
            lambda d: d.position.gain(d.data["daily"]["close"].iloc[-1]) > 5000,
            lambda d: d.position.gain(d.data["daily"]["close"].iloc[-1]) > 2500,
            lambda d: self.risk(d) > self.goal(d),
            lambda d: self.lower(d) < d.data["daily"]["low"].iloc[-1],
            lambda d: d.data["daily"]["stages_average"].iloc[-1] > 0,
            lambda d: d.data["daily"]["macdhist_convert"].iloc[-1] == -1,
            lambda d: d.data["daily"]["rci_trend"].iloc[-1] == -1,
            lambda d: d.data["daily"]["high_rounddown"].iloc[-1] == 0,
            lambda d: d.data["daily"]["weekly_average_trend"].iloc[-1] == 1,
            lambda d: d.data["daily"]["rci_long_trend"].iloc[-1] == 1,
            lambda d: d.data["daily"]["macdhist_trend"].iloc[-1] == 1,
        ]

    def stop_loss(self):
        return [
            lambda d: d.data["daily"]["high"].iloc[-1] < self.safety(d, self.term(d)), # セーフゾーンを割った
            lambda d: d.data["daily"]["macdhist_trend"].iloc[-5:].min() == 1 and d.data["daily"]["volume_average_trend"].iloc[-1] == 1, # ダマシでない下方ブレイク
            lambda d: d.data["daily"]["macdhist_trend"].iloc[-1] == 1,
            lambda d: d.data["daily"]["macd_cross"].iloc[-1] == 1,
            lambda d: d.data["daily"]["rci_trend"].iloc[-1] == -1,
            lambda d: d.data["daily"]["high_rounddown"].iloc[-1] == 0,
            lambda d: d.data["daily"]["weekly_average_trend"].iloc[-1] == 1,
            lambda d: d.data["daily"]["rci_trend"].iloc[-1] == 1,
        ]

