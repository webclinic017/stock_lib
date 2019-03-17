# -*- coding: utf-8 -*-
import numpy
import utils
import simulator
from strategy import CombinationCreator
from loader import Loader

class CombinationStrategy(CombinationCreator):
    def subject(self, date):
        stocks = Loader.steady_trend_stocks(date, date, "fall_stocks.csv")
        codes = stocks["all"]
        return codes

    def common(self):
        default = self.default_common()
        default.new = [
#            lambda d: d.data["weekly"]["rci_trend"].iloc[-1] <= 0, # 週足が下落トレンド
#            lambda d: d.data["weekly"]["daily_average_trend"].iloc[-1] <= 0, # 週足が下落トレンド
            lambda d: self.risk(d) < self.max_risk(d), # リスクが2%以内
            lambda d: self.risk(d) < self.goal(d), # リスクより利益のほうが大きい
        ]

        return default

    def new(self):
        return [
#            lambda d: d.data["daily"]["stages_average"].iloc[-1] > 0,
            lambda d: d.data["daily"]["weekly_average_trend"].iloc[-1] < 0 and d.data["daily"]["daily_average_trend"].iloc[-1] >= 0,
            lambda d: d.data["daily"]["resistance"].iloc[-1] < d.data["daily"]["high"].iloc[-1], # 下方へのブレイクアウトではない
            lambda d: d.index["nikkei"]["trend"].iloc[-10:].max() < 1,
#            lambda d: d.data["daily"]["stages"].iloc[-10:].min() > -2,
            lambda d: d.data["daily"]["macdhist_convert"].iloc[-1] == 1,
            lambda d: d.data["daily"]["rci_trend"].iloc[-1] == -1,
#            lambda d: d.data["daily"]["average_cross"].iloc[-5:].min() == 0, # 平均線のゴールデンクロス直後ではない
#            lambda d: d.data["daily"]["high_rounddown"].iloc[-1] == 1,
#            lambda d: d.data["weekly"]["high_roundup"].iloc[-1] == 1,
        ]

    def taking(self):
        return [
            lambda d: self.risk(d) > self.goal(d),
            lambda d: self.lower(d) < d.data["daily"]["low"].iloc[-1],
            lambda d: d.index["nikkei"]["trend"].iloc[-1] > 0, # 日経平均が上昇トレンドに
            lambda d: d.data["daily"]["stages_average"].iloc[-1] < 0,
            lambda d: d.data["daily"]["macdhist_convert"].iloc[-1] == -1,
            lambda d: d.data["daily"]["rci_trend"].iloc[-1] == -1,
            lambda d: d.data["daily"]["low_roundup"].iloc[-1] == 1,
 #           lambda d: d.data["weekly"]["daily_average_trend"].iloc[-1] == 1,
 #           lambda d: d.data["weekly"]["rci_trend"].iloc[-1] == 1,
 #           lambda d: d.data["weekly"]["macdhist_trend"].iloc[-1] == 1,
        ]

    def stop_loss(self):
        return [
            lambda d: d.data["daily"]["high"].iloc[-1] > self.safety(d, self.term(d)), # セーフゾーンを割った
            lambda d: d.data["daily"]["macdhist_trend"].iloc[-5:].min() > -1 and d.data["daily"]["volume_average_trend"].iloc[-1] < -1, # ダマシでない下方ブレイク
            lambda d: d.index["nikkei"]["trend"].iloc[-1] > 0, # 日経平均が上昇トレンドに
            lambda d: d.data["daily"]["macdhist_trend"].iloc[-1] == 1, # MACDヒストグラムが上昇
            lambda d: d.data["daily"]["macd_cross"].iloc[-1] == 1, # MACDゴールデンクロス
            lambda d: d.data["daily"]["rci_trend"].iloc[-1] == 1,
            lambda d: d.data["daily"]["low_roundup"].iloc[-1] == 1,
 #           lambda d: d.data["weekly"]["daily_average_trend"].iloc[-1] == 1,
 #           lambda d: d.data["weekly"]["rci_trend"].iloc[-1] == 1,
 #           lambda d: d.data["weekly"]["macdhist_trend"].iloc[-1] == 1,
        ]

    def closing(self):
        return [
            lambda d: False,
        ]

