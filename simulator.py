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
    METHOD_LONG = "long"
    METHOD_SHORT = "short"

    SYSTEM_ACTUAL = "actual"
    SYSTEM_CREDIT = "credit"

    def __init__(self, num = 0, value = 0, term = 0, initial = None, system=SYSTEM_ACTUAL, method=METHOD_LONG, min_unit=None):
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
        self.interest = 0

    def add_history(self, num, value):
        for _ in range(int(num)):
            self.value.append(value) # 平均取得価格の計算のため
        self.num += num
        if self.get_num() == 0:
          self.term = 0
          self.value = []
          self.interest = 0

    # 現在の評価額
    def eval(self, value, num):
        return value * num * self.min_unit

    def cost(self, value, num):
        price = self.eval(self.get_value(), num)
        price = price + self.gain(value, num)
        return price

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

    def apply_split_ratio(self, ratio):
        self.num = int(math.ceil(self.num * ratio))
        self.value = list(map(lambda x: x / ratio, self.value))

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

    # 金利を取得
    def get_interest(self):
        return self.interest

    # 金利を追加
    def add_interest(self, interest):
        self.interest += interest

    # 特定口座（現物）
    def is_actual(self):
        return self.system == self.SYSTEM_ACTUAL

    # 信用口座（建玉）
    def is_credit(self):
        return self.system == self.SYSTEM_CREDIT

    # 売建
    def is_short(self):
        return self.is_credit() and self.method == self.METHOD_SHORT

    # 買建
    def is_long(self):
        return self.is_credit() and self.method == self.METHOD_LONG

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
        self.set_order = None

    def is_market(self):
        return all([
            not self.is_limit,
            not self.is_reverse_limit
        ])

    def increment_term(self):
        self.term += 1
        return self

    def set_price(self, price):
        self.price = price
        return self

    def is_valid(self):
        return True if self.valid_term is None else self.term <= self.valid_term

    def only_on_the_day(self):
        return False if self.valid_term is None else self.valid_term == 0

    def signal(self, position, price, low=None, high=None):
        data = {
            "low": low,
            "high": high,
            "price": price,
            "position": position
        }
        return all(list(map(lambda x:x(data), self.conditions)))

    def add_set_order(self, order):
        self.set_order = order
        return self

class MarketOrder(Order):
    def __init__(self, num, is_short=False, on_close=False):
        conditions = [lambda x: True]
        super().__init__(num, conditions)
        self.is_short = is_short
        self.on_close = on_close
        self.order_type = "market"

class LimitOrder(Order):
    def __init__(self, num, price, is_short=False, is_repay=False):
        conditions = [lambda x: x["price"] < price if is_short else x["price"] > price] if is_repay else [lambda x: x["price"] > price if is_short else x["price"] < price] # TODO
        super().__init__(num, conditions)
        self.price = price
        self.is_short = is_short
        self.is_limit = True
        self.order_type = "limit"
        assert price is not None, "require 'price'"

class ReverseLimitOrder(Order):
    def __init__(self, num, price, is_short=False, is_repay=False, valid_term=None):
#        conditions = [lambda x: x["price"] >= price if is_short else x["price"] <= price] if is_repay else [lambda x: x["price"] <= price if is_short else x["price"] >= price]
        conditions = [lambda x: self.reverse_limit_conditions(x, is_repay)]
        super().__init__(num, conditions)
        self.price = price
        self.is_short = is_short
        self.is_reverse_limit = True
        self.valid_term = valid_term
        self.order_type = "reverse_limit"
        assert price is not None, "require 'price'"

    def reverse_limit_conditions(self, x, is_repay):
        if x["high"] is None or x["low"] is None:
            return False

        if is_repay:
            return all([
                x["price"] >= self.price if self.is_short else x["price"] <= self.price,
                x["high"] >= self.price if self.is_short else x["low"] <= self.price
            ])
        else:
            return all([
                x["price"] <= self.price if self.is_short else x["price"] >= self.price,
                x["low"] <= self.price if self.is_short else x["high"] >= self.price
            ])

        return all(conditions)

# ルール適用用データ
class Appliable:
    def __init__(self, data, index):
        self.data = data # データ
        self.index = index # 指標データ

    def dates(self, start_date, end_date):
        dates = list(set(self.data.dates(start_date, end_date)) & set(self.index.dates(start_date, end_date)))
        return sorted(dates, key=lambda x: utils.to_datetime_by_term(x))

    def at(self, date):
        return Appliable(self.data.at(date), self.index.at(date))

class AppliableData(Appliable):
    def __init__(self, data, index, position, assets, setting, stats):
        self.data = data # データ
        self.index = index # 指標データ
        self.position = position # ポジション
        self.assets = assets # 資産
        self.setting = setting # 設定
        self.stats = stats # 統計データ

    def at(self, date):
        return AppliableData(self.data.at(date), self.index.at(date), self.position, self.assets, self.setting, self.stats)

class SimulatorTermData:
    def __init__(self, data):
        self.data = pandas.DataFrame([[""]], columns=["date"]) if data is None else data

    def split(self, start_date, end_date):
        data = self.split_from(start_date).split_to(end_date)
        return data

    def split_from(self, start_date):
        d = self.data[self.data["date"] >= start_date]
        return SimulatorTermData(d)

    def split_to(self, end_date):
        d = self.data[self.data["date"] <= end_date]
        return SimulatorTermData(d)

    def split_until(self, end_date):
        d = self.data[self.data["date"] < end_date]
        return SimulatorTermData(d)

    def dates(self, start_date, end_date):
        d = self.data[self.data["date"] >= start_date]
        d = d[d["date"] <= end_date]
        dates = d["date"].copy().astype(str).values.tolist()
        return dates

    def at(self, date):
        return SimulatorTermData(self.data[self.data["date"] == date])

    def index(self, begin, end):
        d = self.data.iloc[begin:end]
        return SimulatorTermData(d)

    def create_empty(self, date):
        data = pandas.DataFrame([[0] * len(self.data.columns)], columns=self.data.columns)
        data["date"].iloc[0] = date
        data['date'] = pandas.to_datetime(data['date'], format='%Y-%m-%d')
        return SimulatorTermData(pandas.DataFrame([[0] * len(self.data.columns)], columns=self.data.columns))

class SimulatorData:
    def __init__(self, code, middle, short=None):
        self.code = code
        self.middle = middle
        self.short = short

    def split(self, start_date, end_date):
        return self.split_from(start_date).split_to(end_date)

    def split_from(self, start_date):
        return SimulatorData(self.code,
            SimulatorTermData(self.middle).split_from(start_date).data,
            SimulatorTermData(self.short).split_from(start_date).data
        )

    def split_to(self, end_date):
        return SimulatorData(self.code,
            SimulatorTermData(self.middle).split_to(end_date).data,
            SimulatorTermData(self.short).split_to(end_date).data
        )

    def split_until(self, end_date):
        return SimulatorData(self.code,
            SimulatorTermData(self.middle).split_until(end_date).data,
            SimulatorTermData(self.short).split_until(end_date).data
        )

    def dates(self, start_date, end_date):
        return SimulatorTermData(self.middle).dates(start_date, end_date)

    def at(self, date):
        return SimulatorData(self.code,
            SimulatorTermData(self.middle).at(date).data,
            SimulatorTermData(self.short).at(date).data
        )

    def index(self, begin, end):
        return SimulatorData(self.code,
            SimulatorTermData(self.middle).index(begin, end).data,
            SimulatorTermData(self.short).index(begin, end).data
        )

    def create_empty(self, date):
        return SimulatorData(self.code,
            SimulatorTermData(self.middle).create_empty(date).data,
            SimulatorTermData(self.short).create_empty(date).data
        )

class SimulatorIndexData:
    def __init__(self, data):
        self.data = data

    def complement(self, data, date):
        return data if len(data.middle) > 0 else data.create_empty(date)

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
        self.stop_loss_rate = 0.02
        self.taking_rate = 1.0
        self.min_unit = 100
        self.hard_limit = None
        self.ignore_volume = False # 指値時の出来高チェックをスキップ
        self.use_deposit = False

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
            "max_unrealized_gain": None,
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
            "commission": None,
            "interest": None
        }
        return trade_data

    def append(self, trade_data):
        self.trade_history.append(trade_data)

    def last(self):
        return self.create_trade_data() if len(self.trade_history) == 0 else self.trade_history[-1]

    def apply(self, trade_data):
        self.trade_history = self.trade_history[:-1] + [trade_data]

    def get(self, key, default=None):
        return list(map(lambda x: default if x[key] is None else x[key], self.trade_history))

    def find_by_date(self, date):
        return self.find(lambda x: x["date"] == date)

    def find(self, condition):
        return list(filter(lambda x: condition(x), self.trade_history))

    def dates(self):
        return list(filter(lambda x: x is not None, self.get("date")))

    def size(self):
        return list(map(lambda x: x["size"], self.trade_history))

    def max_size(self):
        return max(self.size()) if len(self.size()) > 0 else 0

    def term(self):
        return list(map(lambda x: x["term"], self.trade_history))

    def max_term(self):
        return max(self.term()) if len(self.term()) > 0 else 0

    def closing_term(self):
        return sum(list(map(lambda x: x["term"], self.closing_trade())))

    def trade(self):
        return list(filter(lambda x: x["gain"] is not None, self.trade_history))

    def win_trade(self):
        return list(filter(lambda x: x["gain"] > 0, self.trade()))

    def lose_trade(self):
        return list(filter(lambda x: x["gain"] < 0, self.trade()))

    def closing_trade(self):
        return self.find(lambda x: x["closing"])

    def trade_num(self):
        return len(self.trade())

    def win_trade_num(self):
        return len(self.win_trade())

    def lose_trade_num(self):
        return len(self.lose_trade())

    def closing_trade_num(self):
        return len(self.closing_trade())

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

    def unrealized_gain(self, split_condition=None):
        default_condition = lambda x: x["closing"] == True
        histories = utils.split_list(self.trade_history, default_condition if split_condition is None else split_condition)
        choice = lambda history: list(filter(lambda x: x is not None, list(map(lambda x: x["unrealized_gain"], history))))
        return list(map(choice, histories))

    def last_unrealized_gain(self, split_condition = None):
        history = self.unrealized_gain(split_condition)
        return [] if len(history) == 0 else history[-1]

    def current_max_unrealized_gain(self, split_condition = None):
        unrealized_gain = self.last_unrealized_gain(split_condition)
        return 0 if len(unrealized_gain) == 0 else max(unrealized_gain)

    def max_unrealized_gain(self, split_condition = None):
        max_unrealized_gain = list(map(lambda x: max(x) if len(x) > 0 else 0, self.unrealized_gain(split_condition)))
        return 0 if len(max_unrealized_gain) == 0 else max(max_unrealized_gain)

    def min_unrealized_gain(self, split_condition = None):
        min_unrealized_gain = list(map(lambda x: min(x) if len(x) > 0 else 0, self.unrealized_gain(split_condition)))
        return 0 if len(min_unrealized_gain) == 0 else min(min_unrealized_gain)

    def closing_gain(self):
        return sum(list(map(lambda x: x["gain"], self.find(lambda x: x["closing"] and x["gain"]))))

    def gain_rate(self):
        return list(filter(lambda x: x is not None, map(lambda x: x["gain_rate"], self.trade_history)))

    def commission(self):
        return list(filter(lambda x: x is not None, map(lambda x: x["commission"], self.trade_history)))

    def interest(self):
        return list(filter(lambda x: x is not None, map(lambda x: x["interest"], self.trade_history)))

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
        return self.last()["canceled"] is not None

    def new_canceled(self):
        return self.last()["canceled"] == "new"

    def repay_canceled(self):
        return self.last()["canceled"] == "repay"

    def closing_canceled(self):
        return self.last()["canceled"] == "closing"

    def all_canceled(self):
        return self.last()["canceled"] == "all"

    def new_orders(self):
        order = self.last()["new_order"]
        return [] if order is None else [order]

    def repay_orders(self):
        order = self.last()["repay_order"]
        return [] if order is None else [order]

    def closing_orders(self):
        order = self.last()["closing_order"]
        return [] if order is None else [order]

    def orders(self):
        return list(filter(lambda x: x["signal"] == "new" or x["signal"] == "repay", self.trade_history))

    def executed(self):
        return list(filter(lambda x: x["new"] is not None or x["repay"] is not None, self.trade_history))

    def auto_stop_loss(self):
        return list(filter(lambda x: x["repay"] is not None and x["order_type"] == "reverse_limit", self.trade_history))

    def win_auto_stop_loss(self):
        return list(filter(lambda x: x["gain"] > 0, self.auto_stop_loss()))

    def lose_auto_stop_loss(self):
        return list(filter(lambda x: x["gain"] <= 0, self.auto_stop_loss()))

    def crash(self, loss=0):
        history = list(filter(lambda x: x["term"] == 1 and x["gain"] is not None and x["gain"] < loss, self.trade_history))
        return list(map(lambda x: x["gain"], history))


class StockOrder:
    def __init__(self):
        self.orders = []

    def get(self):
        return self.orders

    def add(self, order):
        self.orders = self.orders + [order]

    def set(self, orders):
        self.orders = orders

    def clear(self):
        self.orders = []

    def increment_term(self):
        for i in range(len(self.orders)):
            self.orders[i].increment_term()

class SecuritiesCompony:
    def default_commission(self, price, is_credit=False):
        print("use default default_commission. should override default_commission")
        return 0

    def tick_price(self, price):
        print("use default tick_price. should override tick_price")
        return 1

    def price_limit(self, price):
        print("use default price_limit. should override price_limit")
        return (0, 0) # (価格帯, 値幅)

    def interest(self, price, num):
        print("use default interest. should override interest")
        return 0

# 証券会社
class SecuritiesComponySimulator(SecuritiesCompony):
    def __init__(self, setting, position = None):
        system = Position.SYSTEM_CREDIT if setting.short_trade else Position.SYSTEM_ACTUAL
        method = Position.METHOD_SHORT if setting.short_trade else Position.METHOD_LONG
        self.position = position if position is not None else Position(system=system, method=method, min_unit=setting.min_unit)
        self.setting = setting
        self.leverage = 3.33 # 要調整
        self.assets = setting.assets
        self.update_deposit(self.assets)
        self.binding = 0 # 拘束資産（注文中）
        self.unbound = 0 # 解放された資産（約定↓痔）
        self.stats = SimulatorStats()
        self.logs = []
        self.new_orders = StockOrder()
        self.repay_orders = StockOrder()
        self.closing_orders = StockOrder()
        self.force_stop = False

    def log(self, message_callback):
        if self.setting.debug:
            message = message_callback()
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
        return AppliableData(data, index, self.position, self.total_assets(data.middle["close"].iloc[-1].item()), self.setting, self.stats)

    def clear_orders(self):
        self.new_orders.clear()
        self.repay_orders.clear()
        self.closing_orders.clear()

    def update_deposit(self, deposit):
        self.deposit = deposit
        self.capacity = self.deposit * self.leverage

    # 総資産
    def total_assets(self, value):
        return self.assets + self.unavailable_assets(value)

    def unavailable_assets(self, value):
        return self.position.cost(value, self.position.get_num())

    def total_capacity(self):
        total = self.capacity - self.total_binding()
        return 0 if total <= 0 else total

    def total_binding(self):
        return self.binding + self.order_binding()

    def order_binding(self):
        return sum(list(map(lambda x: x.binding, self.new_orders.get())))

    # TODO 利用ないかも
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
        if num <= 0:
            return False

        cost = self.position.new(num, value)
        self.add_interest()
        self.assets -= cost
        if self.setting.use_deposit:
            self.update_deposit(self.deposit - int(cost / self.leverage))
        else:
            self.capacity -= cost

        self.log(lambda: "[%s] new: %s yen x %s, total %s, ave %s, assets %s, cost %s" % (self.position.method, value, num, self.position.get_num(), self.position.get_value(), self.total_assets(value), cost))

        return True

    # 返済
    def repay(self, value, num):
        value = value * (1 - random.uniform(0.0, self.setting.error_rate))
        if (num <= 0 or self.position.get_num() <= 0):
            return False

        gain_rate = self.position.gain_rate(value)
        gain = self.position.gain(value, num)
        term = self.position.get_term()
        cost = self.position.repay(num, value)
        self.assets += cost
        if self.setting.use_deposit:
            # 実際の損益はレバレッジ関係なく直接assetsに反映
            # cost / leverageだと実際の損益の1/3が反映されてしまうのでその分を引く
            self.update_deposit(self.deposit + int(cost / self.leverage) + gain - int(gain / self.leverage))
        else:
            self.capacity += cost

        self.log(lambda: "[%s] repay: %s yen x %s, total %s, ave %s, assets %s, cost %s, term %s : gain %s" % (self.position.method, value, num, self.position.get_num(), self.position.get_value(), self.total_assets(value), cost, term, gain))
        return True

    # 強制手仕舞い(以降トレードしない)
    def force_closing(self, date, data):
        low, high, value = data.middle["low"].iloc[-1], data.middle["high"].iloc[-1], data.middle["close"].iloc[-1]
        trade_data = self.stats.last()
        num = self.position.get_num()
        gain = self.position.gain(value, num)
        gain_rate = self.position.gain_rate(value)
        cost = self.position.cost(num, value)
        if self.repay(value, num):
            self.log(lambda: " - force closing: price %s x %s" % (value, num))
            trade_data["date"] = date
            trade_data["repay"] = value
            trade_data["gain"] = gain
            trade_data["gain_rate"] = gain_rate
            trade_data["contract_price"] = value * num * self.setting.min_unit
            trade_data["closing"] = True

        self.new_orders.clear()
        self.repay_orders.clear()
        self.stats.apply(trade_data)
        self.force_stop = True

    # 手仕舞い
    def closing(self, force_stop=False):
        self.force_stop = force_stop

        if self.force_stop:
            self.log(lambda: "[cancel] new/repay order. force closed")
            self.new_orders.clear()
            self.repay_orders.clear()

        if self.position.get_num() > 0:
            self.log(lambda: " - closing_order: num %s, price %s" % (self.position.get_num(), self.position.get_value()))
            self.repay_orders.set([MarketOrder(self.position.get_num(), is_short=self.position.is_short())])

            if len(self.stats.trade_history) > 0:
                trade_data = self.stats.last()
                trade_data["closing"] = True
                self.stats.apply(trade_data)

    def commission(self, value, num):
        return self.default_commission(self.position.cost(value, num), is_credit=True)

    def add_interest(self):
        interest = self.interest(self.position.get_value(), self.position.get_num() * self.position.min_unit)
        self.position.add_interest(interest)

    def adjust_tick(self, price):
        tick_price = self.tick_price(price)
        return tick_price * int(price / tick_price)

    # 注文を分割する
    def split_order(self, order, num):
        hit = copy.copy(order)
        remain = copy.copy(order)

        hit.num = num
        remain.num = remain.num - num

        self.log(lambda: "[split order] hit: %s, remain: %s" % (hit.num, remain.num))

        return [hit, remain]

    def exec_order(self, condition, orders, volume=None):
        hit_orders = list(filter(lambda x: condition(x) and x.is_valid(), orders)) # 条件を満たした注文

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
        execution = lambda x: x.signal(self.position, price) and x.is_market() and x.on_close == on_close
        hit_orders, self.new_orders.orders = self.exec_order(execution, self.new_orders.get())
        hit_orders = list(map(lambda x: x.set_price(price), hit_orders))
        return hit_orders

    def limit_new_order(self, price, low, high, volume):
        execution = lambda x: x.signal(self.position, self.limit(price, low, high, x), low, high) and x.is_limit
        hit_orders, self.new_orders.orders = self.exec_order(execution, self.new_orders.get(), volume=volume)
        hit_orders = list(map(lambda x: x.set_price(self.limit(price, low, high, x)), hit_orders))
        return hit_orders

    def reverse_limit_new_order(self, price, low, high, volume):
        execution = lambda x: x.signal(self.position, self.reverse_limit(price, low, high, x), low, high) and x.is_reverse_limit
        hit_orders, self.new_orders.orders = self.exec_order(execution, self.new_orders.get(), volume=volume)
        hit_orders = list(map(lambda x: x.set_price(self.reverse_limit(price, low, high, x)), hit_orders))
        return hit_orders

    def repay_order(self, price, on_close=False):
        execution = lambda x: x.signal(self.position, price) and x.is_market() and x.on_close == on_close
        hit_orders, self.repay_orders.orders = self.exec_order(execution, self.repay_orders.get())
        hit_orders = list(map(lambda x: x.set_price(price), hit_orders))
        return hit_orders

    def limit_repay_order(self, price, low, high, volume):
        execution = lambda x: x.signal(self.position, self.reverse_limit(price, low, high, x), low, high) and x.is_limit
        hit_orders, self.repay_orders.orders = self.exec_order(execution, self.repay_orders.get(), volume=volume)
        hit_orders = list(map(lambda x: x.set_price(self.reverse_limit(price, low, high, x)), hit_orders))
        return hit_orders

    def reverse_limit_repay_order(self, price, low, high, volume):
        execution = lambda x: x.signal(self.position, self.limit(price, low, high, x), low, high) and x.is_reverse_limit
        hit_orders, self.repay_orders.orders = self.exec_order(execution, self.repay_orders.get(), volume=volume)
        hit_orders = list(map(lambda x: x.set_price(self.limit(price, low, high, x)), hit_orders))
        return hit_orders

    def closing_order(self, price):
        execution = lambda x: x.signal(self.position, price) and x.is_market()
        hit_orders, self.closing_orders.orders = self.exec_order(execution, self.closing_orders.get())
        hit_orders = list(map(lambda x: x.set_price(price), hit_orders))
        return hit_orders

    # 指値の条件に使うデータ
    def limit(self, price, low, high, order):
        if order.is_short:
            best_price = price if low > order.price else order.price
        else:
            best_price = price if high < order.price else order.price
        return best_price # best

    def reverse_limit(self, price, low, high, order):
        if order.is_short:
            best_price = price if high < order.price else order.price
        else:
            best_price = price if low > order.price else order.price
        return best_price # best

    # 指値の約定価格
    def new_agreed_price(self, data, order):

        slippage = self.tick_price(order.price) * 2
        slippage = -slippage if order.is_short else slippage

        if order.is_market():
            return order.price

        if order.is_limit:
            price = order.price
        else:
            price = order.price + slippage
        return price

    def repay_agreed_price(self, data, order):

        slippage = self.tick_price(order.price) * 2
        slippage = slippage if order.is_short else -slippage

        if order.is_market():
            return order.price

        if order.is_limit:
            price = order.price
        else:
            price = order.price + slippage
        return price

    def signals(self, strategy, data, index, trade_data):
        print("use default signals. should override signals")
        return trade_data

    def auto_stop_loss(self, price, position):
        return

    def open_trade(self, volume, data, trade_data):
        price = data.middle["open"].iloc[-1].item()
        # 仮想トレードなら注文をキューから取得

        new_orders = []
        new_orders += self.new_order(price)

        repay_orders = []
        if self.position.get_num() > 0:
            repay_orders += self.repay_order(price)

        return self.virtual_trade(data, new_orders, repay_orders, trade_data)

    def close_trade(self, volume, data, trade_data):
        price = data.middle["close"].iloc[-1].item()
        # 仮想トレードなら注文をキューから取得

        new_orders = []
        new_orders += self.new_order(price, on_close=True)

        repay_orders = []
        if self.position.get_num() > 0:
            repay_orders += self.repay_order(price, on_close=True)
            repay_orders += self.closing_order(price)

        return self.virtual_trade(data, new_orders, repay_orders, trade_data)

    def intraday_trade(self, volume, data, trade_data):
        price = data.middle["open"].iloc[-1].item()
        # 仮想トレードなら注文をキューから取得
        low = data.middle["low"].iloc[-1]
        high = data.middle["high"].iloc[-1]

        new_orders = []
        new_orders += self.limit_new_order(price, low, high, volume)
        new_orders += self.reverse_limit_new_order(price, low, high, volume)

        repay_orders = []
        if self.position.get_num() > 0:
            repay_orders += self.limit_repay_order(price, low, high, volume)
            repay_orders += self.reverse_limit_repay_order(price, low, high, volume)

        return self.virtual_trade(data, new_orders, repay_orders, trade_data)

    def virtual_trade(self, data, new_orders, repay_orders, trade_data):
        # 新規注文実行
        for order in new_orders:
            agreed_price = self.new_agreed_price(data, order)

            self.position.system = Position.SYSTEM_CREDIT if order.is_short else Position.SYSTEM_ACTUAL
            self.position.method = Position.METHOD_SHORT if order.is_short else Position.METHOD_LONG

            self.position.method = "short" if order.is_short else "long"
            if self.new(agreed_price, order.num):
                # 拘束資金の解放
                self.unbound += order.binding
                trade_data["new"] = agreed_price
                trade_data["order_type"] = order.order_type
                trade_data["assets"]              = self.total_assets(agreed_price)
                trade_data["min_assets"]          = self.total_assets(data.middle["high"].iloc[-1] if self.position.is_short() else data.middle["low"].iloc[-1])
                trade_data["unavailable_assets"]  = self.unavailable_assets(agreed_price)
                trade_data["commission"]          = self.commission(agreed_price, order.num)

        # 返済注文実行
        for order in repay_orders:
            if self.position.get_num() <= 0:
                self.repay_orders.clear() # ポジションがなくなってたら以降の注文はキャンセル
                break
            agreed_price = self.repay_agreed_price(data, order)
            gain         = self.position.gain(agreed_price, order.num)
            gain_rate    = self.position.gain_rate(agreed_price)
            commission   = self.commission(agreed_price, order.num)
            interest     = self.position.get_interest()
            if self.repay(agreed_price, order.num):
                trade_data["repay"] = agreed_price
                trade_data["gain"] = gain
                trade_data["gain_rate"] = gain_rate
                trade_data["order_type"] = order.order_type
                trade_data["assets"]              = self.total_assets(agreed_price)
                trade_data["unavailable_assets"]  = self.unavailable_assets(agreed_price)
                trade_data["commission"]          = commission
                trade_data["interest"]            = interest

        return trade_data

    def create_trade_data(self, date, low, high, price):
        self.unbound = 0

        trade_data = self.stats.create_trade_data()
        trade_data["date"]                = date
        trade_data["assets"]              = self.total_assets(price)
        trade_data["min_assets"]          = self.total_assets(high if self.position.is_short() else low)
        trade_data["unavailable_assets"]  = self.unavailable_assets(price)
        return trade_data

    def apply_signal(self):
        if len(self.stats.trade_history) > 0:
            trade_data = self.stats.last()
            trade_data["signal"] = "new" if len(self.new_orders.get()) > 0 else "repay" if len(self.repay_orders.get()) > 0 else None
            self.stats.apply(trade_data)

    def increment_term(self):
        # ポジションの保有期間を増やす
        if self.position.get_num() > 0:
            self.position.increment_term()
            # 金利を手数料に加算する
            self.add_interest()

        # 注文の保持期間を増やす
        self.new_orders.increment_term()
        self.repay_orders.increment_term()

    def simulate_by_date(self, date, data, index={}):
        term_data = data.split_to(date)

        term_index = index.split_to(date)

        today = term_data.at(date).middle

        if len(today) == 0:
            self.log(lambda: "less data: %s" % (date))
            trade_data = self.create_trade_data(date, term_data.middle["low"].iloc[-1], term_data.middle["high"].iloc[-1], term_data.middle["close"].iloc[-1])
            self.stats.append(trade_data)
            return

        today = today.iloc[-1]

        price = today["open"].item() # 約定価格
        num = self.position.get_num()
        volume = None if self.setting.ignore_volume else math.ceil(today["volume"].item() * 10)
        self.log(lambda: "date: %s, price: %s, volume: %s, ave: %.2f, hold: %s, deposit: %s, assets: %s, capacity: %d, binding: %d, gain : %.2f"
            % (date, price, volume, self.position.get_value(), num, self.deposit, self.assets ,self.capacity, self.total_binding(), self.position.gain(price, num)))

        self.trade(self.setting.strategy, price, volume, term_data, term_index)

    # トレード
    def trade(self, strategy, price, volume, data, index):
        assert type(data) is SimulatorData, "data is not SimulatorData."

        date = data.middle["date"].astype(str).iloc[-1]

        # トレード履歴に直前のシグナルを反映
        self.apply_signal()

        # stats
        trade_data = self.create_trade_data(date, data.middle["low"].iloc[-1], data.middle["high"].iloc[-1], price)

        # 判断に必要なデータ数がない
        if price == 0 or len(data.middle) < self.setting.min_data_length:
            self.log(lambda: "less data. skip trade. [%s - %s]. price: %s == 0 or length: %s < %s" % (data.middle["date"].iloc[0], date, price, len(data.middle), self.setting.min_data_length))
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
        self.auto_stop_loss(data.middle["close"].iloc[-2])

        # ザラバ
        if self.setting.virtual_trade:
            trade_data = self.intraday_trade(volume, data, trade_data)

        # トレード履歴に追加
        trade_data["contract_price"] = list(filter(lambda x: x is not None, [trade_data["new"], trade_data["repay"]]))
        trade_data["contract_price"] = None if len(trade_data["contract_price"]) == 0 else trade_data["contract_price"][0] * trade_data["size"] * self.setting.min_unit
        trade_data["max_unrealized_gain"] = self.position.current_gain(data.middle["low"].iloc[-1] if self.position.is_short() else data.middle["high"].iloc[-1])
        trade_data["unrealized_gain"] = self.position.current_gain(data.middle["close"].iloc[-1])
        self.stats.append(trade_data)

        # 手仕舞い後はもう何もしない
        if self.force_stop:
            self.log(lambda: "force stopped. [%s - %s]" % (data.middle["date"].iloc[0], date))
            return

        # 注文を出す
        trade_data = self.signals(strategy, data, index, trade_data)
        assert type(trade_data) is dict, "trade_data is not dict, return trade_data."

        # 引け
        if self.setting.virtual_trade:
            trade_data = self.close_trade(volume, data, trade_data) 

# シミュレーター
class Simulator(SecuritiesComponySimulator):

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
            price = data.middle["close"].iloc[-1] if order.price is None else order.price
            # 成り行き注文なら値幅の上限を基準にcostを計算する
            price = price + self.price_limit(price)[1] if order.is_market() else price
            cost = self.position.eval(price, order.num)
            if (self.total_capacity() - cost) <= 0:
                self.log(lambda: " - [over capacity] new_order: num %s, price %s, cost %d - %s unbound: %s" % (order.num, order.price, self.total_capacity(), cost, self.unbound))
                return signal
            self.log(lambda: " - new_order: num %s, price %s, cost %s - %s" % (order.num, order.price, self.total_capacity(), cost))
            order.binding = cost
            self.new_orders.set([order])
            signal = order

        return signal

    def repay_signals(self, strategy, data, index):
        signal = None
        for order in self.taking_signal(strategy, data, index):
            if order.num > 0:
                self.log(lambda: " - taking_order: num %s, price %s" % (order.num, order.price))
            self.repay_orders.set([order])
            signal = order

        for order in self.stop_loss_signal(strategy, data, index):
            if order.num > 0:
                self.log(lambda: " - stop_loss_order: num %s, price %s" % (order.num, order.price))
            self.repay_orders.set([order])
            signal = order

        return signal

    def closing_signals(self, strategy, data, index):
        for order in self.closing_signal(strategy, data, index):
            if order.num > 0:
                self.log(lambda: " - closing_order: num %s, price %s" % (order.num, order.price))
            self.closing_orders.set([order])
            return order
        return None

    def signals(self, strategy, data, index, trade_data):
        # 新規ルールに当てはまる場合買う
        trade_data["new_order"] = self.new_signals(strategy, data, index)
        # 返済注文
        trade_data["repay_order"] = self.repay_signals(strategy, data, index)
        # 手仕舞い注文
        trade_data["closing_order"] = self.closing_signals(strategy, data, index)

        # 注文の整理
        trade_data = self.order_adjust(trade_data)

        return trade_data

    def auto_stop_loss(self, price):
        if self.setting.hard_limit is None:
            return

        # ストップまでの価格差
        price_range = self.adjust_tick(self.setting.hard_limit)

        if self.position.is_short():
            limit = self.adjust_tick(price + price_range)
        else:
            limit = self.adjust_tick(price - price_range)

        if limit > 0:
            self.log(lambda: "[auto_stop_loss][%s] price: %s, stop: %s, range: %s" % (self.position.method, self.position.get_value(), limit, self.setting.hard_limit))
            self.repay_orders.add(ReverseLimitOrder(self.position.get_num(), limit, is_repay=True, is_short=self.position.is_short(), valid_term=0))

    def order_adjust(self, trade_data):

        # 期限切れの注文を消す
        self.new_orders.set(list(filter(lambda x: not x.only_on_the_day() , self.new_orders.get())))
        self.repay_orders.set(list(filter(lambda x: not x.only_on_the_day() , self.repay_orders.get())))

        # 手仕舞いの場合全部キャンセル
        if self.position.get_num() > 0 and len(self.closing_orders.get()) > 0:
            self.log(lambda: "[cancel] new/repay order. force closed")
            self.new_orders.clear()
            self.repay_orders.clear()
            self.force_stop = True
            trade_data["closing"] = True

        # ポジションがなければ返済シグナルは捨てる
        if self.position.get_num() <= 0 and len(self.closing_orders.get()) > 0:
            self.log(lambda: "[cancel] closing order")
            self.closing_orders.clear()
            trade_data["canceled"] = "closing"

        # ポジションがなければ返済シグナルは捨てる
        if self.position.get_num() <= 0 and len(self.repay_orders.get()) > 0:
            self.log(lambda: "[cancel] repay order")
            self.repay_orders.clear()
            trade_data["canceled"] = "repay"

        # 新規・返済が同時に出ている場合返済を優先
        if len(self.new_orders.get()) > 0 and len(self.repay_orders.get()) > 0:
            self.log(lambda: "[cancel] new order")
            self.new_orders.clear()
            trade_data["canceled"] = "new"

        return trade_data
