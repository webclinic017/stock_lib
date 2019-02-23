# -*- coding: utf-8 -*-
import numpy
import utils
import pandas
import simulator
from dateutil.relativedelta import relativedelta
from strategy import CombinationCreator
from loader import Loader

class CombinationStrategy(CombinationCreator):
    def subject(self, date):
        num = 2
        high_performance_stocks = Loader.high_performance_stocks()
        high_performance_codes = high_performance_stocks["code"].as_matrix().tolist()

        codes = self.current(date)

        targets = list(set(codes) & set(high_performance_codes))
        return targets if len(targets) < num else targets[:num]

    def last_week(self, date):
        now = utils.to_datetime(date)
        before_begin = now - relativedelta(weeks=1, days=now.weekday(), hour=0, minute=0, second=0)
        before_end = before_begin + relativedelta(days=6)
        codes = {"code":[], "price":[]}
        for d in utils.daterange(before_begin, before_end):
            stocks = Loader.new_score(utils.to_format(d), "new_high.csv")
            if stocks is None:
                continue
            codes["code"] += stocks["code"].as_matrix().tolist()
            codes["price"] += stocks["price"].as_matrix().tolist()
        detail = pandas.DataFrame(codes)
        detail = detail.sort_values("price")
        return detail["code"].as_matrix().tolist()

    def current(self, date):
        stocks = Loader.new_score(date, "new_high.csv")
        if stocks is None:
            return []
        return stocks["code"].as_matrix().tolist()

    def volume_rate(self, d):
        before = d.data["daily"]["volume"].iloc[-2]
        current = d.data["daily"]["volume"].iloc[-1]
        rate = 0.0
        if before > 0:
            rate = current / before
        return rate

    def opened(self, data, term=3):
        columns = ["open", "high", "low", "close"]
        d = data.data["daily"].iloc[-term:]
        l = utils.each(lambda i, x: len(list(set(x[columns].as_matrix().tolist()))) > 1, d)
        return all(l)

    def common(self):
        default = self.default_common()
        default.new = [
            lambda d, s: int(d.data["daily"]["volume"].iloc[-1]) >= 100, # 出来高が100単位以上
            lambda d, s: self.volume_rate(d) >= 2.0, # 出来高が前日の倍以上
            lambda d, s: self.opened(d), # 数日以内に寄らなかった日がない
        ]
        default.stop_loss = [
            lambda d, s: d.stats.drawdown[-1]["drawdown"] > 0.02,
        ]

        return default

    def new(self):
        return [
            lambda d, s: self.volume_rate(d) >= 4.0, # 出来高が前日の倍以上
            lambda d, s: self.volume_rate(d) >= 5.0, # 出来高が前日の倍以上
            lambda d, s: d.index["nikkei"]["close"].iloc[-1] > d.index["nikkei"]["close"].iloc[-2], # 日経も前日より上げている
            lambda d, s: d.index["nikkei"]["trend"].iloc[-10:].min() > -1,
            lambda d, s: d.index["nikkei"]["trend"].iloc[-1].min() > -1,
            lambda d, s: d.data["daily"]["macd_trend"].iloc[-1] > 0,
            lambda d, s: d.data["daily"]["macdhist_trend"].iloc[-1] > 0,
        ]

    def taking(self):
        return [
            lambda d, s: d.data["daily"]["weekly_average"].iloc[-1] > d.data["daily"]["close"].iloc[-1],
            lambda d, s: d.data["daily"]["macd_cross"].iloc[-1] == -1, # MACDデッドクロス
            lambda d, s: d.data["daily"]["macd"].iloc[-1] > 0,
            lambda d, s: d.data["daily"]["macd"].iloc[-1] < 0,
            lambda d, s: d.data["daily"]["macdsignal"].iloc[-1] > 0,
            lambda d, s: d.data["daily"]["macdsignal"].iloc[-1] < 0,
            lambda d, s: d.data["daily"]["macdhist"].iloc[-1] > 0,
            lambda d, s: d.data["daily"]["macdhist"].iloc[-1] < 0,
            lambda d, s: d.data["daily"]["macd_trend"].iloc[-1] < 0,
            lambda d, s: d.data["daily"]["macdhist_trend"].iloc[-1] < 0,
        ]

    def stop_loss(self):
        return [
            lambda d, s: d.data["daily"]["weekly_average"].iloc[-1] > d.data["daily"]["close"].iloc[-1],
            lambda d, s: d.data["daily"]["macd_cross"].iloc[-1] == -1, # MACDデッドクロス
            lambda d, s: d.data["daily"]["macd"].iloc[-1] > 0,
            lambda d, s: d.data["daily"]["macd"].iloc[-1] < 0,
            lambda d, s: d.data["daily"]["macdsignal"].iloc[-1] > 0,
            lambda d, s: d.data["daily"]["macdsignal"].iloc[-1] < 0,
            lambda d, s: d.data["daily"]["macdhist"].iloc[-1] > 0,
            lambda d, s: d.data["daily"]["macdhist"].iloc[-1] < 0,
            lambda d, s: d.data["daily"]["macd_trend"].iloc[-1] < 0,
            lambda d, s: d.data["daily"]["macdhist_trend"].iloc[-1] < 0,
        ]

    def closing(self):
        return [
            lambda d, s: False,
        ]

