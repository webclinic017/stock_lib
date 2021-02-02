# -*- coding: utf-8 -*-
import os
import numpy
import random
import pandas
import utils
import copy
import math
from itertools import groupby

# 売買の状態
class Position:
    def __init__(self, num = 0, value = 0, term = 0, initial = None, system="actual", method="long", min_unit=None):
        assert min_unit is not None, "min_unit is None."
        self.num = int(num)
        self.value = []
        self.initial = initial
        if value != 0:
            self.value = [value]
        self.term = term # 保有期間
        self.system = system
        self.method = method
        self.min_unit = min_unit

    def add_history(self, num, value):
        for _ in range(int(num)):
            self.value.append(value) # 平均取得価格の計算のため
        self.num += num
        if self.get_num() == 0:
          self.term = 0
          self.value = []

    # 現在の評価額
    def eval(self, value, num):
        return value * num * self.min_unit

    def cost(self, value, num):
        price = self.eval(self.get_value(), num)
        price = price + self.gain(value, num)
        return price

    def commission(self, value, num):
        import rakuten
        return rakuten.default_commission(self.cost(value, num), is_credit=False)

    # 新規
    def new(self, num, value):
        self.add_history(num, value)
        price = self.eval(value, num)
        return price

    # 返済
    def repay(self, num, value):
        assert len(self.value) > 0, "do not repay. not hold."
        price = self.cost(value, num)
        self.add_history(-num, value)
        return price

    # 保有株数
    def get_num(self):
        return self.num

    # 平均取得価格
    def get_value(self):
        if len(self.value) == 0:
            return 0
        return sum(self.value) / len(self.value)

    # ポジション取得時の価格
    def get_initial(self):
        if self.initial is None:
            if len(self.value) > 0:
                return self.value[0]
            else:
                return None
        return self.initial

    # 損益
    def gain(self, value, num):
        if self.get_value() is None:
            return 0
        if self.is_short():
            return self.eval(self.get_value() - value, num)
        else:
            return self.eval(value - self.get_value(), num)

    def current_gain(self, value):
        return self.gain(value, self.get_num())

    # 損益レシオ
    def gain_rate(self, value):
        if self.get_value() is None or self.get_value() == 0:
            return 0
        if self.is_short():
            return (self.get_value() - value) / self.get_value()
        else:
            return (value - self.get_value()) / self.get_value()

    # 取得時の価格を設定
    def set_initial(self, initial):
        self.initial = initial

    # ポジション取得からの期間
    def get_term(self):
        return self.term

    # 期間を設定
    def set_term(self, term):
        self.term = term

    # 保有期間を加算
    def increment_term(self):
        self.term += 1

    # 特定口座（現物）
    def is_actual(self):
        return self.system == "actual"

    # 信用口座（建玉）
    def is_credit(self):
        return self.system == "credit"

    # 売建
    def is_short(self):
        return self.is_credit() and self.method == "short"

    # 買建
    def is_long(self):
        return self.is_credit() and self.method == "long"

# 注文
class Order:
    def __init__(self, num, conditions):
        self.num = int(num)
        self.term = 0
        self.price = None
        self.conditions = conditions
        self.is_short = False
        self.is_reverse_limit = False
        self.is_limit = False
        self.on_close = False
        self.valid_term = None
        self.order_type = None
        self.binding = 0


    def is_market(self):
        return all([
            not self.is_limit,
            not self.is_reverse_limit
        ])

    def increment_term(self):
        self.term += 1

    def is_valid(self):
        return True if self.valid_term is None else self.term <= self.valid_term

    def only_on_the_day(self):
        return False if self.valid_term is None else self.valid_term == 0

    def signal(self, price, position):
        data = {
            "price": price,
            "position": position
        }
        return all(list(map(lambda x:x(data), self.conditions)))

class MarketOrder(Order):
    def __init__(self, num, is_short=False, on_close=False):
        conditions = [lambda x: True]
        super().__init__(num, conditions)
        self.is_short = is_short
        self.on_close = on_close
        self.order_type = "market"

class LimitOrder(Order):
    def __init__(self, num, price, is_short=False, is_repay=False):
        conditions = [lambda x: x["price"] < price if is_short else x["price"] > price] if is_repay else [lambda x: x["price"] > price if is_short else x["price"] < price]
        super().__init__(num, conditions)
        self.price = price
        self.is_short = is_short
        self.is_limit = True
        self.order_type = "limit"
        assert price is not None, "require 'price'"

class ReverseLimitOrder(Order):
    def __init__(self, num, price, is_short=False, is_repay=False, valid_term=None):
        conditions = [lambda x: x["price"] >= price if is_short else x["price"] <= price] if is_repay else [lambda x: x["price"] <= price if is_short else x["price"] >= price]
        super().__init__(num, conditions)
        self.price = price
        self.is_short = is_short
        self.is_reverse_limit = True
        self.valid_term = valid_term
        self.order_type = "reverse_limit"
        assert price is not None, "require 'price'"

# ルール適用用データ
class AppliableData:
    def __init__(self, data, index, position, assets, setting, stats):
        self.data = data # データ
        self.index = index # 指標データ
        self.position = position # ポジション
        self.assets = assets # 資産
        self.setting = setting # 設定
        self.stats = stats # 統計データ

    def dates(self, start_date, end_date):
        dates = list(set(self.data.dates(start_date, end_date)) & set(self.index.dates(start_date, end_date)))
        return sorted(dates, key=lambda x: utils.to_datetime_by_term(x))

    def at(self, date):
        return AppliableData(self.data.at(date), self.index.at(date), self.position, self.assets, self.setting, self.stats)

class SimulatorData:
    def __init__(self, code, daily, rule):
        self.code = code
        self.daily = daily
        self.rule = rule

    def split(self, start_date, end_date):
        data = self.split_from(start_date).split_to(end_date)
        return data

    def split_from(self, start_date):
        d = self.daily[self.daily["date"] >= start_date]
        return SimulatorData(self.code, d, self.rule)

    def split_to(self, end_date):
        d = self.daily[self.daily["date"] <= end_date]
        return SimulatorData(self.code, d, self.rule)

    def split_until(self, end_date):
        d = self.daily[self.daily["date"] < end_date]
        return SimulatorData(self.code, d, self.rule)

    def dates(self, start_date, end_date):
        d = self.daily[self.daily["date"] >= start_date]
        d = d[d["date"] <= end_date]
        dates = d["date"].copy().astype(str).values.tolist()
        return dates

    def at(self, date):
        return SimulatorData(self.code, self.daily[self.daily["date"] == date], self.rule)

    def index(self, begin, end):
        d = self.daily.iloc[begin:end]
        return SimulatorData(self.code, d, self.rule)

    def create_empty(self, date):
        data = pandas.DataFrame([[0] * len(self.daily.columns)], columns=self.daily.columns)
        data["date"].iloc[0] = date
        data['date'] = pandas.to_datetime(data['date'], format='%Y-%m-%d')
        return SimulatorData(self.code, pandas.DataFrame([[0] * len(self.daily.columns)], columns=self.daily.columns), self.rule)

class SimulatorIndexData:
    def __init__(self, data):
        self.data = data

    def complement(self, data, date):
        return data if len(data.daily) > 0 else data.create_empty(date)

    def dates(self, start_date, end_date):
        dates = []
        for k, v in self.data.items():
            dates = list(set(dates + v.dates(start_date, end_date)))
        dates = sorted(dates, key=lambda x: utils.to_datetime_by_term(x))
        return dates

    def split_from(self, date):
        term_index = {}
        for k, v in self.data.items():
            term_index[k] = v.split_from(date)
        return SimulatorIndexData(term_index)

    def split_to(self, date):
        term_index = {}
        for k, v in self.data.items():
            if k in ["dow", "nasdaq"]:
                d = v.split_until(date)
            else:
                d = v.split_to(date)
            term_index[k] = self.complement(d, date)
        return SimulatorIndexData(term_index)

    def split(self, start_date, end_date):
        return self.split_from(start_date).split_to(end_date)

    def at(self, date):
        return self.split_from(date).split_to(date)

# シミュレーター設定
class SimulatorSetting:
    def __init__(self):
        self.strategy = None
        self.min_data_length = 30
        self.assets = 0
        self.commission = 150
        self.debug = False
        self.error_rate = 0.00
        self.virtual_trade = True # 仮想取引 Falseにすると注文を処理しない
        self.short_trade = False
        self.long_short_trade = {"long": None, "short": None}
        self.stop_loss_rate = 0.02
        self.taking_rate = 1.0
        self.min_unit = 100
        self.trade_step = 1
        self.use_before_stick = False
        self.auto_stop_loss = None
        self.ignore_volume = False # 指値時の出来高チェックをスキップ

# 統計
class SimulatorStats:
    def __init__(self):
        self.trade_history = []

    def create_trade_data(self):
        trade_data = {
            "date": None,
            "new": None,
            "repay": None,
            "signal": None,
            "new_order": None,
            "repay_order": None,
            "closing_order": None,
            "gain": None,
            "unrealized_gain": None,
            "gain_rate": None,
            "assets": None,
            "min_assets": None,
            "unavailable_assets": None,
            "term": 0,
            "size": 0,
            "canceled": None,
            "closing": False,
            "contract_price": None,
            "order_type": None,
            "commission": None
        }
        return trade_data

    def append(self, trade_data):
        self.trade_history.append(trade_data)

    def apply(self, trade_data):
        self.trade_history = self.trade_history[:-1] + [trade_data]

    def size(self):
        return list(map(lambda x: x["size"], self.trade_history))

    def term(self):
        return list(map(lambda x: x["term"], self.trade_history))

    def max_size(self):
        return max(self.size()) if len(self.size()) > 0 else 0

    def max_term(self):
        return max(self.term()) if len(self.term()) > 0 else 0

    def trade(self):
        return list(filter(lambda x: x["gain"] is not None, self.trade_history))

    def win_trade(self):
        return list(filter(lambda x: x["gain"] > 0, self.trade()))

    def lose_trade(self):
        return list(filter(lambda x: x["gain"] < 0, self.trade()))

    def trade_num(self):
        return len(self.trade())

    def win_trade_num(self):
        return len(self.win_trade())

    def lose_trade_num(self):
        return len(self.lose_trade())

    # 勝率
    def win_rate(self):
        trade_num = self.trade_num()
        if trade_num == 0:
            return 0
        return self.win_trade_num() / float(trade_num)

    def assets(self):
        return list(map(lambda x: x["assets"], self.trade_history))

    def min_assets(self):
        return list(map(lambda x: x["min_assets"], self.trade_history))

    def max_assets(self):
        if len(self.assets()) == 0:
            return 0
        return max(self.assets())

    def unavailable_assets(self):
        return list(map(lambda x: x["unavailable_assets"], self.trade_history))

    def max_unavailable_assets(self):
        if len(self.unavailable_assets()) == 0:
            return 0
        return max(self.unavailable_assets())

    def contract_price(self):
        return list(map(lambda x: x["contract_price"], self.trade_history))

    def sum_contract_price(self):
        return sum(list(filter(lambda x: x is not None, self.contract_price())))

    def drawdown(self):
        return utils.drawdown(self.assets())

    # 最大ドローダウン
    def max_drawdown(self):
        dd = self.drawdown()
        if len(dd) == 0:
            return 0
        return max(dd)

    def gain(self):
        return list(filter(lambda x: x is not None, map(lambda x: x["gain"], self.trade_history)))

    def unrealized_gain(self):
        histories = utils.split_list(self.trade_history, lambda x: x["closing"]==True)
        choice = lambda history: list(filter(lambda x: x is not None, list(map(lambda x: x["unrealized_gain"], history))))
        return list(map(choice, histories))

    def last_unrealized_gain(self):
        history = self.unrealized_gain()
        return [] if len(history) == 0 else history[-1]

    def max_unrealized_gain(self):
        unrealized_gain = self.last_unrealized_gain()
        return 0 if len(unrealized_gain) == 0 else max(unrealized_gain) # TODO 手仕舞いされてたらそれ以降のものを取得する

    def gain_rate(self):
        return list(filter(lambda x: x is not None, map(lambda x: x["gain_rate"], self.trade_history)))

    def commission(self):
        return list(filter(lambda x: x is not None, map(lambda x: x["commission"], self.trade_history)))

    def profits(self):
        return list(filter(lambda x: x > 0, self.gain()))

    def loss(self):
        return list(filter(lambda x: x < 0, self.gain()))

    def profits_rate(self):
        return list(filter(lambda x: x > 0, self.gain_rate()))

    def loss_rate(self):
        return list(filter(lambda x: x < 0, self.gain_rate()))

    def win_streak(self):
        streak = self.streak()
        if len(streak) == 0:
            return 0
        else:
            is_win, count = streak[-1]
            return count if is_win else 0 # 連勝のみ

    def lose_streak(self):
        streak = self.streak()
        if len(streak) == 0:
            return 0
        else:
            is_win, count = streak[-1]
            return count if not is_win else 0 # 連勝のみ

    # 勝敗の連続数
    def streak(self):
        return [ (is_win, len(list(l))) for is_win, l in groupby(self.gain(), key=lambda x: x > 0)]

    # 平均利益率
    def average_profit_rate(self):
        if self.win_trade_num() == 0:
            return 0
        return numpy.average(self.profits_rate()) / self.win_trade_num()

    # 平均損失率
    def average_loss_rate(self):
        if self.lose_trade_num() == 0:
            return 0
        return numpy.average(self.loss_rate()) / self.lose_trade_num()

    def reword_ratio(self):
        return self.average_profit_rate() * self.win_rate()

    def risk_ratio(self):
        return abs(self.average_loss_rate()) * (1 - self.win_rate())

    # リワードリスクレシオ
    def rewordriskratio(self):
        reword = self.reword_ratio()
        risk = self.risk_ratio()
        return reword / risk if risk > 0 else reword

    def canceled(self):
        return self.trade_history[-1]["canceled"] is not None

    def new_canceled(self):
        return self.trade_history[-1]["canceled"] == "new"

    def repay_canceled(self):
        return self.trade_history[-1]["canceled"] == "repay"

    def closing_canceled(self):
        return self.trade_history[-1]["canceled"] == "closing"

    def all_canceled(self):
        return self.trade_history[-1]["canceled"] == "all"

    def new_orders(self):
        order = self.trade_history[-1]["new_order"]
        return [] if order is None else [order]

    def repay_orders(self):
        order = self.trade_history[-1]["repay_order"]
        return [] if order is None else [order]

    def closing_orders(self):
        order = self.trade_history[-1]["closing_order"]
        return [] if order is None else [order]

    def orders(self):
        return list(filter(lambda x: x["signal"] == "new" or x["signal"] == "repay", self.trade_history))

    def executed(self):
        return list(filter(lambda x: x["new"] is not None or x["repay"] is not None, self.trade_history))

    def auto_stop_loss(self):
        return list(filter(lambda x: x["repay"] is not None and x["order_type"] == "reverse_limit", self.trade_history))

    def crash(self, loss=0):
        history = list(filter(lambda x: x["term"] == 1 and x["gain"] is not None and x["gain"] < loss, self.trade_history))
        return list(map(lambda x: x["gain"], history))

# シミュレーター
class Simulator:
    def __init__(self, setting, position = None):
        system = "credit" if setting.short_trade else "actual"
        method = "short" if setting.short_trade else "long"
        self.setting = setting
        self.position = position if position is not None else Position(system=system, method=method, min_unit=self.setting.min_unit)
        self.assets = setting.assets
        self.capacity = setting.assets * 3.33 # 要調整
        self.binding = 0
        self.unbound = 0
        self.stats = SimulatorStats()
        self.logs = []
        self.new_orders = []
        self.repay_orders = []
        self.closing_orders = []
        self.force_stop = False

    def log(self, message):
        if self.setting.debug:
            print(message)
            self.logs.append(message)

    # ルールすべてに当てはまるならTrue
    def apply_all_rules(self, data, index, rules):
        if len(rules) == 0:
            return False
        appliable_data = self.create_appliable_data(data, index)
        results = list(map(lambda x: x.apply(appliable_data), rules))
        results = list(filter(lambda x: x is not None, results))
        return results

    def create_appliable_data(self, data, index):
        return AppliableData(data, index, self.position, self.total_assets(data.daily["close"].iloc[-1].item()), self.setting, self.stats)

    # 総資産
    def total_assets(self, value):
        holdings = self.position.cost(value, self.position.get_num())
        return self.assets + holdings

    def total_capacity(self):
        total = self.capacity - self.total_binding()
        return 0 if total <= 0 else total

    def total_binding(self):
        return self.binding + self.order_binding()

    def order_binding(self):
        return sum(list(map(lambda x: x.binding, self.new_orders)))

    def get_stats(self): 
        stats = dict()
        stats["assets"] = int(self.assets)
        stats["gain"] = stats["assets"] - self.setting.assets
        stats["return"] = float((stats["assets"] - self.setting.assets) / float(self.setting.assets))
        stats["win_rate"] = float(self.stats.win_rate())
        stats["drawdown"] = float(self.stats.max_drawdown())
        stats["max_unavailable_assets"] = self.stats.max_unavailable_assets()
        stats["trade"] = int(self.stats.trade_num())
        stats["win_trade"] = int(self.stats.win_trade_num())
        # 平均利益率
        stats["average_profit_rate"] = self.stats.average_profit_rate()
        # 平均損失率
        stats["average_loss_rate"] = self.stats.average_loss_rate()
        # リワードリスクレシオ
        stats["rewordriskratio"] = self.stats.rewordriskratio()

        # トレード履歴
        stats["trade_history"] = self.stats.trade_history

        if self.setting.debug:
            stats["logs"] = self.logs

        return stats

    # 新規
    def new(self, value, num):
        value = value * (1 + random.uniform(0.0, self.setting.error_rate))
#        if (self.capacity - self.position.eval(value, num)) <= 0:
#            # 資産が足りなければスルー
#            self.log(" - assets_not_enough: num %s, value %s" % (num, value))
#            return False
        if num <= 0:
            return False

        cost = self.position.new(num, value)
        commission = self.position.commission(value, num)
        self.assets -= cost
        self.capacity -= cost

        self.log("[%s] new: %s yen x %s, total %s, ave %s, assets %s, cost %s, commission %s" % (self.position.method, value, num, self.position.get_num(), self.position.get_value(), self.total_assets(value), cost, commission))

        return True

    # 返済
    def repay(self, value, num):
        value = value * (1 - random.uniform(0.0, self.setting.error_rate))
        if (num <= 0 or self.position.get_num() <= 0):
            return False

        gain_rate = self.position.gain_rate(value)
        gain = self.position.gain(value, num)
        commission = self.position.commission(value, num)
        cost = self.position.repay(num, value)
        self.assets += cost
        self.capacity += cost

        self.log("[%s] repay: %s yen x %s, total %s, ave %s, assets %s, cost %s, commission %s, term %s : gain %s" % (self.position.method, value, num, self.position.get_num(), self.position.get_value(), self.total_assets(value), cost, commission, self.position.get_term(), gain))
        return True

    # 強制手仕舞い(以降トレードしない)
    def force_closing(self, date, data):
        low, high, value = data.daily["low"].iloc[-1], data.daily["high"].iloc[-1], data.daily["close"].iloc[-1]
        trade_data = self.create_trade_data(date, low, high, value)
        num = self.position.get_num()
        gain = self.position.gain(value, num)
        gain_rate = self.position.gain_rate(value)
        cost = self.position.cost(num, value)
        if self.repay(value, num):
            self.log(" - force closing: price %s x %s" % (value, num))
            trade_data["repay"] = value
            trade_data["gain"] = gain
            trade_data["gain_rate"] = gain_rate
            trade_data["closing"] = True

        self.new_orders = []
        self.repay_orders = []
        self.stats.append(trade_data)
        self.force_stop = True

    # 手仕舞い
    def closing(self):
        if self.position.get_num() > 0:
            self.log(" - closing_order: num %s, price %s" % (self.position.get_num(), self.position.get_value()))
            self.repay_orders = [MarketOrder(self.position.get_num(), is_short=self.position.is_short())]

            if len(self.stats.trade_history) > 0:
                trade_data = self.stats.trade_history[-1]
                trade_data["closing"] = True
                self.stats.apply(trade_data)


    def tick_price(self, price):
        tick_prices = [
            [3000, 1],
            [5000, 5],
            [30000, 10],
            [50000, 50],
        ]
        tick_price = None
        for t in tick_prices:
            if price < t[0]:
                tick_price = t[1]
                break

        if tick_price is None:
            tick_price = 100

        return tick_price

    # 注文を分割する
    def split_order(self, order, num):
        hit = copy.copy(order)
        remain = copy.copy(order)

        hit.num = num
        remain.num = remain.num - num

        self.log("[split order] hit: %s, remain: %s" % (hit.num, remain.num))

        return [hit, remain]

    def exec_order(self, condition, orders, price=None, volume=None):
        hit_orders = list(filter(lambda x: condition(x) and x.is_valid(), orders)) # 条件を満たした注文

        # 注文の価格を設定
        if price is not None:
            for i in range(len(hit_orders)):
                hit_orders[i].price = price

        remain = list(filter(lambda x: not condition(x) and x.is_valid(), orders)) # 残っている注文

        # 出来の確認をする
        if volume is not None:
            if volume <= 0:
                return [], orders # 条件を満たしていても出来なし
            # 注文の数量より出来が少ない場合は[出来有, 出来無]に分ける
            part = list(map(lambda x: self.split_order(x, volume) if volume < x.num else [x, None], hit_orders))
            hit_part = list(map(lambda x: x[0], part))
            remain_part = list(filter(lambda x: x is not None, map(lambda x: x[1], part)))
            return hit_part, remain + remain_part

        return hit_orders, remain

    # 出来がなければhit_ordersをまたキューに入れなおす
    # 出来高が注文数より少ない場合はhit_ordersのordersを現在の出来高にし、残りをキューに入れ直す
    def new_order(self, price, on_close=False):
        execution = lambda x: x.signal(price, self.position) and x.is_market() and x.on_close == on_close
        hit_orders, self.new_orders = self.exec_order(execution, self.new_orders, price)
        return hit_orders

    def limit_new_order(self, low, high, volume):
        execution = lambda x: x.signal(self.limit(low, high, x), self.position) and x.is_limit
        hit_orders, self.new_orders = self.exec_order(execution, self.new_orders, volume=volume)
        return hit_orders

    def reverse_limit_new_order(self, low, high, volume):
        execution = lambda x: x.signal(self.reverse_limit(low, high, x), self.position) and x.is_reverse_limit
        hit_orders, self.new_orders = self.exec_order(execution, self.new_orders, volume=volume)
        return hit_orders

    def repay_order(self, price, on_close=False):
        execution = lambda x: x.signal(price, self.position) and x.is_market() and x.on_close == on_close
        hit_orders, self.repay_orders = self.exec_order(execution, self.repay_orders, price)
        return hit_orders

    def limit_repay_order(self, low, high, volume):
        execution = lambda x: x.signal(self.reverse_limit(low, high, x), self.position) and x.is_limit
        hit_orders, self.repay_orders = self.exec_order(execution, self.repay_orders, volume=volume)
        return hit_orders

    def reverse_limit_repay_order(self, low, high, volume):
        execution = lambda x: x.signal(self.limit(low, high, x), self.position) and x.is_reverse_limit
        hit_orders, self.repay_orders = self.exec_order(execution, self.repay_orders, volume=volume)
        return hit_orders

    def closing_order(self, price):
        execution = lambda x: x.signal(price, self.position) and x.is_market()
        hit_orders, self.closing_orders = self.exec_order(execution, self.closing_orders, price)
        return hit_orders

    # 指値の条件に使うデータ
    def limit(self, low, high, order):
        return high if order.is_short else low

    def reverse_limit(self, low, high, order):
        return low if order.is_short else high

    # 指値の約定価格
    def new_agreed_price(self, data, order):

        slippage = self.tick_price(order.price) * 2
        slippage = -slippage if order.is_short else slippage

        if order.is_market():
            return order.price

        if order.is_limit:
            worst_price = order.price
        else:
            worst_price = order.price + slippage
        return worst_price


    def repay_agreed_price(self, data, order):

        slippage = self.tick_price(order.price) * 2
        slippage = slippage if order.is_short else -slippage

        if order.is_market():
            return order.price

        if order.is_limit:
            worst_price = order.price
        else:
            worst_price = order.price + slippage
        return worst_price

    # 新規シグナル
    def new_signal(self, strategy, data, index):
        return self.apply_all_rules(data, index, strategy.new_rules)

    def stop_loss_signal(self, strategy, data, index):
        return self.apply_all_rules(data, index, strategy.stop_loss_rules)

    def taking_signal(self, strategy, data, index):
        return self.apply_all_rules(data, index, strategy.taking_rules)

    def closing_signal(self, strategy, data, index):
        return self.apply_all_rules(data, index, strategy.closing_rules)

    def new_signals(self, strategy, data, index):
        signal = None
        for order in self.new_signal(strategy, data, index):
            price = data.daily["close"].iloc[-1] if order.price is None else order.price
            cost = self.position.eval(price, order.num)
            if (self.total_capacity() - cost) <= 0:
                self.log(" - [over capacity] new_order: num %s, price %s, cost %d - %s unbound: %s" % (order.num, order.price, self.total_capacity(), cost, self.unbound))
                return signal
            self.log(" - new_order: num %s, price %s, cost %s - %s" % (order.num, order.price, self.total_capacity(), cost))
            order.binding = cost
            self.new_orders = [order]
            signal = order

        return signal

    def repay_signals(self, strategy, data, index):
        signal = None
        for order in self.taking_signal(strategy, data, index):
            if order.num > 0:
                self.log(" - taking_order: num %s, price %s" % (order.num, order.price))
            self.repay_orders = [order]
            signal = order

        for order in self.stop_loss_signal(strategy, data, index):
            if order.num > 0:
                self.log(" - stop_loss_order: num %s, price %s" % (order.num, order.price))
            self.repay_orders = [order]
            signal = order

        return signal

    def closing_signals(self, strategy, data, index):
        for order in self.closing_signal(strategy, data, index):
            if order.num > 0:
                self.log(" - closing_order: num %s, price %s" % (order.num, order.price))
            self.closing_orders = [order]
            return order
        return None

    def signals(self, strategy, data, index, trade_data):
        strategies = self.setting.long_short_trade
        if None in strategies.values():
            # 新規ルールに当てはまる場合買う
            trade_data["new_order"] = self.new_signals(strategy, data, index)
            # 返済注文
            trade_data["repay_order"] = self.repay_signals(strategy, data, index)
            # 手仕舞い注文
            trade_data["closing_order"] = self.closing_signals(strategy, data, index)
        else:
            self.setting.short_trade = False
            long_new = self.new_signals(strategies["long"], data, index)
            self.setting.short_trade = True
            short_new = self.new_signals(strategies["short"], data, index)

            trade_data["new_order"] = short_new if long_new is None else long_new

            if self.position.get_num() > 0:
                self.setting.short_trade = self.position.is_short()
                if self.position.is_short() and long_new is not None:
                    self.log(" - long active.")
                    trade_data["new_order"] = None
                    trade_data["repay_order"] = MarketOrder(self.position.get_num(), is_short=self.position.is_short())
                    trade_data["closing_order"] = None
                elif not self.position.is_short() and short_new is not None:
                    self.log(" - short active.")
                    trade_data["new_order"] = None
                    trade_data["repay_order"] = MarketOrder(self.position.get_num(), is_short=self.position.is_short())
                    trade_data["closing_order"] = None
                else:
                    trade_data["new_order"] = short_new if self.setting.short_trade else long_new
                    trade_data["repay_order"] = self.repay_signals(strategies["short"], data, index) if self.setting.short_trade else self.repay_signals(strategies["long"], data, index)
                    trade_data["closing_order"] = self.closing_signals(strategies["short"], data, index) if self.setting.short_trade else self.closing_signals(strategies["long"], data, index)
            else:
                self.setting.short_trade = long_new is None
                trade_data["repay_order"] = self.repay_signals(strategies["short"], data, index) if self.setting.short_trade else self.repay_signals(strategies["long"], data, index)
                trade_data["closing_order"] = self.closing_signals(strategies["short"], data, index) if self.setting.short_trade else self.closing_signals(strategies["long"], data, index)


            self.new_orders = [] if trade_data["new_order"] is None else [trade_data["new_order"]]
            self.repay_orders = [] if trade_data["repay_order"] is None else [trade_data["repay_order"]]
            self.closing_orders = [] if trade_data["closing_order"] is None else [trade_data["closing_order"]]

        # 注文の整理
        trade_data = self.order_adjust(trade_data)

        return trade_data

    def auto_stop_loss(self, price, position):
        if self.setting.auto_stop_loss is None:
            return

        if position.get_num() > 0:
            allowable_loss = (self.setting.assets * self.setting.auto_stop_loss) / (position.get_num() * position.min_unit)

            tick_price = self.tick_price(price)
            price_range = tick_price * int(allowable_loss / tick_price)

            hold = tick_price * int(position.get_value() / tick_price)

            if position.is_short():
                limit = hold + price_range
            else:
                limit = hold - price_range

            if limit > 0:
                self.log("[auto_stop_loss][%s] price: %s, stop: %s, %s - %s" % (position.method, position.get_value(), limit, hold, price_range))
                self.repay_orders = self.repay_orders + [ReverseLimitOrder(position.get_num(), limit, is_repay=True, is_short=position.is_short(), valid_term=0)]

    def order_adjust(self, trade_data):

        # 期限切れの注文を消す
        self.new_orders = list(filter(lambda x: not x.only_on_the_day() , self.new_orders))
        self.repay_orders = list(filter(lambda x: not x.only_on_the_day() , self.repay_orders))

        # 手仕舞いの場合全部キャンセル
        if self.position.get_num() > 0 and len(self.closing_orders) > 0 or len(self.stats.auto_stop_loss()) > 0:
            self.log("[cancel] new/repay order. force closed")
            self.new_orders = []
            self.repay_orders = []
            self.force_stop = True
            trade_data["closing"] = True

        # ポジションがなければ返済シグナルは捨てる
        if self.position.get_num() <= 0 and len(self.closing_orders) > 0:
            self.log("[cancel] closing order")
            self.closing_orders = []
            trade_data["canceled"] = "closing"

        # ポジションがなければ返済シグナルは捨てる
        if self.position.get_num() <= 0 and len(self.repay_orders) > 0:
            self.log("[cancel] repay order")
            self.repay_orders = []
            trade_data["canceled"] = "repay"

        # 新規・返済が同時に出ている場合返済を優先
        if len(self.new_orders) > 0 and len(self.repay_orders) > 0:
            self.log("[cancel] new order")
            self.new_orders = []
            trade_data["canceled"] = "new"

        return trade_data

    def simulate(self, dates, data, index={}):
        assert type(data) is SimulatorData, "data is not SimulatorData."

        print(dates)
        for date in dates:
            self.simulate_by_date(date, data, index)

        # 統計取得のために全部手仕舞う
        self.force_closing(dates[-1], data)

        stats = self.get_stats()

        self.log("result assets: %s" % stats["assets"])

        return stats

    # 高速化
    def simulate_by_date(self, date, data, index={}):
        term_data = data.split_to(date)

        term_index = index.split_to(date)

        today = term_data.at(date).daily

        if len(today) == 0:
            self.log("less data: %s" % (date))
            trade_data = self.create_trade_data(date, term_data.daily["low"].iloc[-1], term_data.daily["high"].iloc[-1], term_data.daily["close"].iloc[-1])
            self.stats.append(trade_data)
            return

        today = today.iloc[-1]

        price = today["open"].item() # 約定価格
        volume = None if self.setting.ignore_volume else math.ceil(today["volume"].item() * 10)
        self.log("date: %s, price: %s, volume: %s, hold: %s, capacity: %d, binding: %d" % (date, price, volume, self.position.get_num(), self.capacity, self.total_binding()))

        self.trade(self.setting.strategy, price, volume, term_data, term_index)

    def open_trade(self, volume, data, trade_data):
        price = data.daily["open"].iloc[-1].item()
        # 仮想トレードなら注文をキューから取得

        new_orders = []
        new_orders += self.new_order(price)

        repay_orders = []
        if self.position.get_num() > 0:
            repay_orders += self.repay_order(price)

        return self.virtual_trade(data, new_orders, repay_orders, trade_data)

    def close_trade(self, volume, data, trade_data):
        price = data.daily["close"].iloc[-1].item()
        # 仮想トレードなら注文をキューから取得

        new_orders = []
        new_orders += self.new_order(price, on_close=True)

        repay_orders = []
        if self.position.get_num() > 0:
            repay_orders += self.repay_order(price, on_close=True)
            repay_orders += self.closing_order(price)

        return self.virtual_trade(data, new_orders, repay_orders, trade_data)

    def intraday_trade(self, volume, data, trade_data):
        price = data.daily["open"].iloc[-1].item()
        # 仮想トレードなら注文をキューから取得
        low = data.daily["low"].iloc[-1]
        high = data.daily["high"].iloc[-1]

        new_orders = []
        new_orders += self.limit_new_order(low, high, volume)
        new_orders += self.reverse_limit_new_order(low, high, volume)

        repay_orders = []
        if self.position.get_num() > 0:
            repay_orders += self.limit_repay_order(low, high, volume)
            repay_orders += self.reverse_limit_repay_order(low, high, volume)

        return self.virtual_trade(data, new_orders, repay_orders, trade_data)

    def virtual_trade(self, data, new_orders, repay_orders, trade_data):
        # 新規注文実行
        for order in new_orders:
            agreed_price = self.new_agreed_price(data, order)
            self.position.system = "credit" if order.is_short else "actual"
            self.position.method = "short" if order.is_short else "long"
            if self.new(agreed_price, order.num):
                # 拘束資金の解放
                self.unbound += order.binding
                trade_data["new"] = agreed_price
                trade_data["order_type"] = order.order_type
                trade_data["assets"]              = self.total_assets(agreed_price)
                trade_data["min_assets"]          = self.total_assets(data.daily["high"].iloc[-1] if self.position.is_short() else data.daily["low"].iloc[-1])
                trade_data["unavailable_assets"]  = trade_data["assets"] - self.assets
                trade_data["commission"]          = self.position.commission(agreed_price, order.num)

        # 返済注文実行
        for order in repay_orders:
            if self.position.get_num() <= 0:
                self.repay_orders = [] # ポジションがなくなってたら以降の注文はキャンセル
                break
            agreed_price = self.repay_agreed_price(data, order)
            gain        = self.position.gain(agreed_price, order.num)
            gain_rate   = self.position.gain_rate(agreed_price)
            commission  = self.position.commission(agreed_price, order.num)
            if self.repay(agreed_price, order.num):
                trade_data["repay"] = agreed_price
                trade_data["gain"] = gain
                trade_data["gain_rate"] = gain_rate
                trade_data["order_type"] = order.order_type
                trade_data["assets"]              = self.total_assets(agreed_price)
                trade_data["unavailable_assets"]  = trade_data["assets"] - self.assets
                trade_data["commission"]          = commission

        return trade_data

    def create_trade_data(self, date, low, high, price):
        self.unbound = 0

        trade_data = self.stats.create_trade_data()
        trade_data["date"]                = date
        trade_data["assets"]              = self.total_assets(price)
        trade_data["min_assets"]          = self.total_assets(high if self.position.is_short() else low)
        trade_data["unavailable_assets"]  = trade_data["assets"] - self.assets
        return trade_data

    def increment_term(self):
        # ポジションの保有期間を増やす
        if self.position.get_num() > 0:
            self.position.increment_term()

        # 注文の保持期間を増やす
        for i in range(len(self.new_orders)):
            self.new_orders[i].increment_term()

        for i in range(len(self.repay_orders)):
            self.repay_orders[i].increment_term()

    def apply_signal(self):
        if len(self.stats.trade_history) > 0:
            trade_data = self.stats.trade_history[-1]
            trade_data["signal"] = "new" if len(self.new_orders) > 0 else "repay" if len(self.repay_orders) > 0 else None
            self.stats.apply(trade_data)

    # トレード
    def trade(self, strategy, price, volume, data, index):
        assert type(data) is SimulatorData, "data is not SimulatorData."

        date = data.daily["date"].iloc[-1]

        # トレード履歴に直前のシグナルを反映
        self.apply_signal()

        # stats
        trade_data = self.create_trade_data(date, data.daily["low"].iloc[-1], data.daily["high"].iloc[-1], price)

        # 判断に必要なデータ数がない
        if price == 0 or len(data.daily) < self.setting.min_data_length:
            self.log("less data. skip trade. [%s - %s]. price: %s == 0 or length: %s < %s" % (data.daily["date"].iloc[0], date, price, len(data.daily), self.setting.min_data_length))
            self.stats.append(trade_data)
            return

        # 注文・ポジションの保有日数をインクリメント
        self.increment_term()
        trade_data["size"] = self.position.get_num()
        trade_data["term"] = self.position.get_term()

        # 寄り付き====================================================================
        if self.setting.virtual_trade: # 注文の約定チェック
            trade_data = self.open_trade(volume, data, trade_data)

        # 損切の逆指値
        self.auto_stop_loss(data.daily["close"].iloc[-2], self.position)

        # ザラバ
        if self.setting.virtual_trade:
            trade_data = self.intraday_trade(volume, data, trade_data)

        # トレード履歴に追加
        trade_data["contract_price"] = list(filter(lambda x: x is not None, [trade_data["new"], trade_data["repay"]]))
        trade_data["contract_price"] = None if len(trade_data["contract_price"]) == 0 else trade_data["contract_price"][0] * trade_data["size"] * self.setting.min_unit
        trade_data["unrealized_gain"] = self.position.current_gain(data.daily["close"].iloc[-1])
        self.stats.append(trade_data)

        # 手仕舞い後はもう何もしない
        if self.force_stop:
            self.log("force stopped. [%s - %s]" % (data.daily["date"].iloc[0], date))
            return

        # 注文を出す
        trade_data = self.signals(strategy, data, index, trade_data)

        # 引け
        if self.setting.virtual_trade:
            trade_data = self.close_trade(volume, data, trade_data)

