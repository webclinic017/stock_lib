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

        num = self.setting.monitor_num
        codes = stocks["code"].iloc[:num].as_matrix().tolist()

        return codes

    def common(self):
        default = self.default_common()
        default.new = [
            lambda d, s: d.data["weekly"]["rci_trend"].iloc[-1] > 0, # 週足が上昇トレンド
            lambda d, s: d.data["weekly"]["daily_average_trend"].iloc[-1] > 0, # 週足が上昇トレンド
            lambda d, s: self.risk(d, s) < self.max_risk(d, s), # リスクが2%以内
            lambda d, s: self.risk(d, s) < self.goal(d, s), # リスクより利益のほうが大きい
        ]

        return default

    def new(self):
        return [
            lambda d, s: d.data["daily"]["stages_average"].iloc[-1] < 0,
            lambda d, s: d.data["daily"]["weekly_average_trend"].iloc[-1] > 0 and d.data["daily"]["daily_average_trend"].iloc[-1] <= 0,
            lambda d, s: d.data["daily"]["support"].iloc[-1] < d.data["daily"]["low"].iloc[-1], # 下方へのブレイクアウトではない
            lambda d, s: d.data["daily"]["stages"].iloc[-10:].max() < 2,
            lambda d, s: d.data["daily"]["macdhist_convert"].iloc[-1] == -1,
            lambda d, s: d.data["daily"]["rci_trend"].iloc[-1] == -1,
            lambda d, s: d.data["daily"]["average_cross"].iloc[-5:].min() == 0, # 平均線のデッドクロス直後ではない
            lambda d, s: d.data["daily"]["low_roundup"].iloc[-1] == 1,
            lambda d, s: d.data["weekly"]["low_roundup"].iloc[-1] == 1,
        ]

    def taking(self):
        return [
            lambda d, s: self.risk(d, s) > self.goal(d, s),
            lambda d, s: self.upper(d, s) < d.data["daily"]["high"].iloc[-1],
            lambda d, s: d.data["daily"]["stages_average"].iloc[-1] > 0,
            lambda d, s: d.data["daily"]["macdhist_convert"].iloc[-1] == 1,
            lambda d, s: d.data["daily"]["rci_trend"].iloc[-1] == 1,
            lambda d, s: d.data["daily"]["low_roundup"].iloc[-1] == 0,
            lambda d, s: d.data["weekly"]["daily_average_trend"].iloc[-1] == -1,
            lambda d, s: d.data["weekly"]["rci_trend"].iloc[-1] == -1,
            lambda d, s: d.data["weekly"]["macdhist_trend"].iloc[-1] == -1,
        ]

    def stop_loss(self):
        return [
            lambda d, s: d.data["daily"]["low"].iloc[-1] < self.safety(d, s, self.term(d, s)), # セーフゾーンを割った
            lambda d, s: d.data["daily"]["macdhist_trend"].iloc[-5:].max() < 1 and d.data["daily"]["volume_average_trend"].iloc[-1] > 1, # ダマシでない下方ブレイク
            lambda d, s: d.data["daily"]["macdhist_trend"].iloc[-1] == -1, # MACDヒストグラムが下降 
            lambda d, s: d.data["daily"]["macd_cross"].iloc[-1] == -1, # MACDデッドクロス
            lambda d, s: d.data["daily"]["rci_trend"].iloc[-1] == 1,
            lambda d, s: d.data["daily"]["low_roundup"].iloc[-1] == 0,
            lambda d, s: d.data["weekly"]["daily_average_trend"].iloc[-1] == -1,
            lambda d, s: d.data["weekly"]["rci_trend"].iloc[-1] == -1,
            lambda d, s: d.data["weekly"]["macdhist_trend"].iloc[-1] == -1,
        ]


