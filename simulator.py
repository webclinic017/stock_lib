# -*- coding: utf-8 -*-
import os
import numpy
import random
import pandas
import utils
import copy
import math

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

    # 新規
    def new(self, num, value):
        self.add_history(num, value)
        price = -self.eval(value, num)
        return price

    # 返済
    def repay(self, num, value):
        assert len(self.value) > 0, "do not repay. not hold."
        price = self.eval(self.get_value(), num)
        price = price + self.gain(value)
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
    def gain(self, value):
        if self.get_value() is None:
            return 0
        if self.is_short():
            return self.eval(self.get_value() - value, self.get_num())
        else:
            return self.eval(value - self.get_value(), self.get_num())

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
    def __init__(self, num, conditions, is_short=False, is_reverse_limit=False, is_limit=False, price=None):
        self.num = int(num)
        self.term = 0
        self.price = price
        self.conditions = conditions
        self.is_short = is_short
        self.is_reverse_limit = is_reverse_limit
        self.is_limit = is_limit

        assert not is_limit or (is_limit and price is not None), "require 'price'"
        assert not is_reverse_limit or (is_reverse_limit and price is not None), "require 'price'"

    def is_market(self):
        return all([
            not self.is_limit,
            not self.is_reverse_limit
        ])

    def increment_term(self):
        self.term += 1

    def signal(self, price, position):
        data = {
            "price": price,
            "position": position
        }
        return all(list(map(lambda x:x(data), self.conditions)))

class MarketOrder(Order):
    def __init__(self, num, is_short=False):
        conditions = [lambda x: True]
        super().__init__(num, conditions)

class LimitOrder(Order):
    def __init__(self, num, price, is_short=False, is_repay=False):
        conditions = [lambda x: x["price"] < price if is_short else x["price"] > price] if is_repay else [lambda x: x["price"] > price if is_short else x["price"] < price]
        super().__init__(num, conditions, is_limit=True, price=price)

class ReverseLimitOrder(Order):
    def __init__(self, num, price, is_short=False, is_repay=False):
        conditions = [lambda x: x["price"] >= price if is_short else x["price"] <= price] if is_repay else [lambda x: x["price"] <= price if is_short else x["price"] >= price]
        super().__init__(num, conditions, is_reverse_limit=True, price=price)

# ルール適用用データ
class AppliableData:
    def __init__(self, data, index, position, assets, setting, stats):
        self.data = data # データ
        self.index = index # 指標データ
        self.position = position # ポジション
        self.assets = assets # 資産
        self.setting = setting # 設定
        self.stats = stats # 統計データ

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

    def dates(self, start_date, end_date):
        d = self.daily[self.daily["date"] >= start_date]
        d = d[d["date"] <= end_date]
        dates = d["date"].copy().astype(str).as_matrix().tolist()
        return dates

    def at(self, date):
        return self.daily[self.daily["date"] == date]

    def index(self, begin, end):
        d = self.daily.iloc[begin:end]
        return SimulatorData(self.code, d, self.rule)

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
        self.taking_rate = 0.005
        self.min_unit = 100
        self.trade_step = 1
        self.use_before_stick = False

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
            "gain_rate": None,
            "assets": None,
            "unavailable_assets": None,
            "term": 0,
            "size": 0,
            "canceled": None,
            "contract_price": None
        }
        return trade_data

    def size(self):
        return list(map(lambda x: x["size"], self.trade_history))

    def term(self):
        return list(map(lambda x: x["term"], self.trade_history))

    def max_size(self):
        return max(self.size())

    def max_term(self):
        return max(self.term())

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
        assets = numpy.array(self.assets())
        max_assets = numpy.maximum.accumulate(assets)
        drawdown = max_assets - assets
        drawdown = drawdown / max_assets
        return drawdown.tolist()

    # 最大ドローダウン
    def max_drawdown(self):
        dd = self.drawdown()
        if len(dd) == 0:
            return 0
        return max(dd)

    def gain(self):
        return list(filter(lambda x: x is not None, map(lambda x: x["gain"], self.trade_history)))

    def gain_rate(self):
        return list(filter(lambda x: x is not None, map(lambda x: x["gain_rate"], self.trade_history)))

    def profits(self):
        return list(filter(lambda x: x > 0, self.gain()))

    def loss(self):
        return list(filter(lambda x: x < 0, self.gain()))

    def profits_rate(self):
        return list(filter(lambda x: x > 0, self.gain_rate()))

    def loss_rate(self):
        return list(filter(lambda x: x < 0, self.gain_rate()))

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

class TradeRecorder:
    def __init__(self, min_unit, output_dir=""):
        self.output_dir = "/tmp/trade_recorder/%s" % output_dir
        self.pattern = {"new": [], "repay": [], "gain": []}
        self.columns = None
        self.min_unit = min_unit

    def set_columns(self, columns):
        self.columns = columns

    def pattern_num(self, num):
        return int(num / self.min_unit)

    def new(self, pattern, num):
        assert self.columns is not None, "columns is None"
        for _ in range(self.pattern_num(num)):
            self.pattern["new"].append(pattern[self.columns].as_matrix().tolist())

    def repay(self, pattern, gain_rate, num):
        assert self.columns is not None, "columns is None"
        for _ in range(self.pattern_num(num)):
            self.pattern["repay"].append(pattern[self.columns].as_matrix().tolist())
            self.pattern["gain"].append(gain_rate)

    def concat(self, recorder):
        self.pattern["new"].extend(recorder.pattern["new"])
        self.pattern["repay"].extend(recorder.pattern["repay"])
        self.pattern["gain"].extend(recorder.pattern["gain"])
        self.columns = recorder.columns

    def output(self, name, append=False):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        mode = "a" if append else "w"
        new = pandas.DataFrame(self.pattern["new"], columns=self.columns)
        new.to_csv("%s/new_%s.csv" % (self.output_dir, name), index=None, header=None, mode=mode)
        repay = pandas.DataFrame(self.pattern["repay"], columns=self.columns)
        repay.to_csv("%s/repay_%s.csv" % (self.output_dir, name), index=None, header=None, mode=mode)
        gain = pandas.DataFrame(self.pattern["gain"], columns=["gain"])
        gain.to_csv("%s/gain_%s.csv" % (self.output_dir, name), index=None, header=None, mode=mode)
        columns = pandas.DataFrame([], columns=self.columns)
        columns.to_csv("%s/columns.csv" % self.output_dir, index=None)

# シミュレーター
class Simulator:
    def __init__(self, setting, position = None):
        system = "credit" if setting.short_trade else "actual"
        method = "short" if setting.short_trade else "long"
        self.setting = setting
        self.position = position if position is not None else Position(system=system, method=method, min_unit=self.setting.min_unit)
        self.assets = setting.assets
        self.stats = SimulatorStats()
        self.logs = []
        self.new_orders = []
        self.repay_orders = []
        self.closing_orders = []

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
        holdings = self.position.eval(value, self.position.get_num())
        return self.assets + holdings

    # 新規
    def new(self, value, num):
        value = value * (1 + random.uniform(0.0, self.setting.error_rate))
        if (self.assets - self.position.eval(value, num)) <= 0:
            # 資産が足りなければスルー
            self.log(" - assets_not_enough: num %s, value %s" % (num, value))
            return False
        if num <= 0:
            return False

        cost = self.position.new(num, value)
        self.assets += cost
        self.commission()

        self.log(" new: %s yen x %s, total %s, ave %s, assets %s, cost %s" % (value, num, self.position.get_num(), self.position.get_value(), self.total_assets(value), cost))

        return True

    # 返済
    def repay(self, value, num):
        value = value * (1 - random.uniform(0.0, self.setting.error_rate))
        if (num <= 0 or self.position.get_num() <= 0):
            return False

        gain_rate = self.position.gain_rate(value)
        gain = self.position.gain(value)
        cost = self.position.repay(num, value)
        self.assets += cost
        self.commission()

        self.log(" repay: %s yen x %s, total %s, ave %s, assets %s, cost %s : gain %s" % (value, num, self.position.get_num(), self.position.get_value(), self.total_assets(value), cost, gain))
        return True

    # 全部売る
    def closing(self, date, value, data=None):
        self.log(" - closing: price %s" % (value))
        trade_data = self.create_trade_data(date, value)
        num = self.position.get_num()
        gain = self.position.gain(value)
        gain_rate = self.position.gain_rate(value)
        if self.repay(value, num):
            trade_data["repay"] = value
            trade_data["gain"] = gain
            trade_data["gain_rate"] = gain_rate

        self.new_orders = []
        self.repay_orders = []
        self.stats.trade_history.append(trade_data)

    # 取引手数料
    # TODO 実際のものに合わせる
    def commission(self):
        self.assets -= self.setting.commission

    # 注文を分割する
    def split_order(self, order, num):
        hit = copy.copy(order)
        remain = copy.copy(order)

        hit.num = num
        remain.num = remain.num - num

        self.log("[split order] hit: %s, remain: %s" % (hit.num, remain.num))

        return [hit, remain]

    def exec_order(self, condition, orders, price=None, volume=None):
        hit_orders = list(filter(lambda x: condition(x), orders)) # 条件を満たした注文

        # 注文の価格を設定
        if price is not None:
            for i in range(len(hit_orders)):
                hit_orders[i].price = price

        remain = list(filter(lambda x: not condition(x), orders)) # 残っている注文

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


    # TODO _order系に出来があったかどうかのチェックを入れる
    # 出来がなければhit_ordersをまたキューに入れなおす
    # 出来高が注文数より少ない場合はhit_ordersのordersを現在の出来高にし、残りをキューに入れ直す
    def new_order(self, price):
        execution = lambda x: x.signal(price, self.position) and x.is_market()
        hit_orders, self.new_orders = self.exec_order(execution, self.new_orders, price)
        return hit_orders

    def limit_new_order(self, price, volume):
        execution = lambda x: x.signal(price, self.position) and x.is_limit
        hit_orders, self.new_orders = self.exec_order(execution, self.new_orders, volume=volume)
        return hit_orders

    def reverse_limit_new_order(self, price, volume):
        execution = lambda x: x.signal(price, self.position) and x.is_reverse_limit
        hit_orders, self.new_orders = self.exec_order(execution, self.new_orders, volume=volume)
        return hit_orders

    def repay_order(self, price):
        execution = lambda x: x.signal(price, self.position) and x.is_market()
        hit_orders, self.repay_orders = self.exec_order(execution, self.repay_orders, price)
        return hit_orders

    def limit_repay_order(self, price, volume):
        execution = lambda x: x.signal(price, self.position) and x.is_limit
        hit_orders, self.repay_orders = self.exec_order(execution, self.repay_orders, volume=volume)
        return hit_orders

    def reverse_limit_repay_order(self, price, volume):
        execution = lambda x: x.signal(price, self.position) and x.is_reverse_limit
        hit_orders, self.repay_orders = self.exec_order(execution, self.repay_orders, volume=volume)
        return hit_orders

    def closing_order(self, price):
        execution = lambda x: x.signal(price, self.position) and x.is_market()
        hit_orders, self.closing_orders = self.exec_order(execution, self.closing_orders, price)
        return hit_orders

    # 指値の条件に使うデータ
    def limit(self):
        return "high" if self.setting.short_trade else "low"

    def reverse_limit(self):
        return "low" if self.setting.short_trade else "high"

    # 指値の約定価格
    def new_agreed_price(self, data, order):
        if order.is_market():
            return order.price

        if order.is_limit:
            worst_price = order.price
        else:
            limit = self.reverse_limit()
            worst_price = data.daily[limit].iloc[-1]
        return worst_price


    def repay_agreed_price(self, data, order):
        if order.is_market():
            return order.price

        if order.is_limit:
            worst_price = order.price
        else:
            limit = self.limit()
            worst_price = data.daily[limit].iloc[-1]
        return worst_price

    # 損切りの逆指値価格
    def reverse_limit_price(self, price, assets=None):
        # 注文の価格を設定
        total = self.total_assets(price) if assets is None else assets
        price_range = (total * 0.02) / self.position.get_num()
        print("reverse_limit_price:", total, self.position.get_num(), price_range)
        if self.setting.short_trade:
            value = self.position.get_value() + price_range
        else:
            value = self.position.get_value() - price_range
        return value

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
            self.log(" - new_order: num %s, price %s" % (order.num, order.price))
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
        # 新規ルールに当てはまる場合買う
        trade_data["new_order"] = self.new_signals(strategy, data, index)
        # 返済注文
        trade_data["repay_order"] = self.repay_signals(strategy, data, index)
        # 手仕舞い注文
        trade_data["closing_order"] = self.closing_signals(strategy, data, index)
        # 注文の整理
        trade_data = self.order_adjust(trade_data)

        return trade_data

    def order_adjust(self, trade_data):
        # ポジションがなければ返済シグナルは捨てる
        if self.position.get_num() <= 0 and len(self.repay_orders) > 0:
            self.log("[cancel] repay order")
            self.repay_orders = []
            trade_data["canceled"] = "repay"

        # 新規・返済が同時に出ている場合何もしない
        if len(self.new_orders) > 0 and len(self.repay_orders) > 0:
            self.log("[cancel] new/repay order")
            self.new_orders = []
            self.repay_orders = []
            trade_data["canceled"] = "all"

        return trade_data

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

    def simulate(self, dates, data, index={}):
        assert type(data) is SimulatorData, "data is not SimulatorData."

        print(dates)
        for date in dates:
            self.simulate_by_date(date, data, index)

        # 統計取得のために全部手仕舞う
        self.closing(dates[-1], data.daily["close"].iloc[-1])

        stats = self.get_stats()

        self.log("result assets: %s" % stats["assets"])

        return stats

    # 高速化
    def simulate_by_date(self, date, data, index={}):
        term_data = data.split_to(date)

        term_index = {}
        for k, v in index.items():
            term_index[k] = v[v["date"] <= date]

        today = term_data.daily.iloc[-1]

        assert len(today) > 0, "not found %s data" % date

        price = today["open"].item() # 約定価格
        volume = math.ceil(today["volume"].item() * 10)
        self.log("date: %s, price: %s, volume: %s" % (date, price, volume))

        self.trade(self.setting.strategy, price, volume, term_data, term_index)

    def open_trade(self, price, volume, data, trade_data):
        # 仮想トレードなら注文をキューから取得

        new_orders = []
        new_orders += self.new_order(price)

        repay_orders = []
        if self.position.get_num() > 0:
            repay_orders += self.repay_order(price)

        return self.virtual_trade(data, new_orders, repay_orders, trade_data)

    def close_trade(self, price, volume, data, trade_data):
        closing_orders = []
        if self.position.get_num() > 0:
            closing_orders += self.closing_order(price)

        return self.virtual_trade(data, [], closing_orders, trade_data)

    def intraday_trade(self, price, volume, data, trade_data):
        # 仮想トレードなら注文をキューから取得
        limit = self.limit()
        reverse_limit = self.reverse_limit()

        limit_price = data.daily[limit].iloc[-1]
        reverse_limit_price = data.daily[reverse_limit].iloc[-1]

        new_orders = []
        new_orders += self.limit_new_order(limit_price, volume)
        new_orders += self.reverse_limit_new_order(reverse_limit_price, volume)

        repay_orders = []
        if self.position.get_num() > 0:
            repay_orders += self.limit_repay_order(reverse_limit_price, volume)
            repay_orders += self.reverse_limit_repay_order(limit_price, volume)

        return self.virtual_trade(data, new_orders, repay_orders, trade_data)

    def virtual_trade(self, data, new_orders, repay_orders, trade_data):
        # 新規注文実行
        for order in new_orders:
            agreed_price = self.new_agreed_price(data, order)
            if self.new(agreed_price, order.num):
                trade_data["new"] = agreed_price

        # 返済注文実行
        for order in repay_orders:
            if self.position.get_num() <= 0:
                self.repay_orders = [] # ポジションがなくなってたら以降の注文はキャンセル
                break
            agreed_price = self.repay_agreed_price(data, order)
            gain        = self.position.gain(agreed_price)
            gain_rate   = self.position.gain_rate(agreed_price)
            if self.repay(agreed_price, order.num):
                trade_data["repay"] = agreed_price
                trade_data["gain"] = gain
                trade_data["gain_rate"] = gain_rate

        return trade_data

    def create_trade_data(self, date, price):
        trade_data = self.stats.create_trade_data()
        trade_data["date"]                = date
        trade_data["assets"]              = self.total_assets(price)
        trade_data["unavailable_assets"]  = trade_data["assets"] - self.assets
        return trade_data

    # トレード
    def trade(self, strategy, price, volume, data, index):
        assert type(data) is SimulatorData, "data is not SimulatorData."

        date = data.daily["date"].iloc[-1]

        # step == 0なら注文
        time = int(date.minute)
        step = time % self.setting.trade_step

        # stats
        trade_data = self.create_trade_data(date, price)

        # 判断に必要なデータ数がない
        if price == 0 or len(data.daily) < self.setting.min_data_length:
            self.log("less data. skip trade. [%s - %s]" % (data.daily["date"].iloc[0], date))
            self.stats.trade_history.append(trade_data)
            return

        # ポジションの保有期間を増やす
        if self.position.get_num() > 0:
            self.position.increment_term()

        # 注文の保持期間を増やす
        for i in range(len(self.new_orders)):
            self.new_orders[i].increment_term()

        for i in range(len(self.repay_orders)):
            self.repay_orders[i].increment_term()

        # 寄り付き====================================================================
        if self.setting.virtual_trade: # 注文の約定チェック
            trade_data = self.open_trade(price, volume, data, trade_data)
            trade_data = self.intraday_trade(price, volume, data, trade_data)

        # 注文を出す
        if step == 0:
            term_data = data.index(0, -1) if self.setting.use_before_stick else data
            self.log("[order stick] %s:%s" % (term_data.daily["date"].iloc[-1], term_data.daily["open"].iloc[-1]))
            trade_data = self.signals(strategy, term_data, index, trade_data)

        # 引け========================================================================
        if self.setting.virtual_trade:
            close = data.daily["close"].iloc[-1].item()
            trade_data = self.close_trade(close, volume, data, trade_data)

        # トレード履歴
        trade_data["signal"] = "new" if len(self.new_orders) > 0 else "repay" if len(self.repay_orders) > 0 else None
        trade_data["size"] = self.position.get_num()
        trade_data["term"] = self.position.get_term()
        trade_data["contract_price"] = list(filter(lambda x: x is not None, [trade_data["new"], trade_data["repay"]]))
        trade_data["contract_price"] = None if len(trade_data["contract_price"]) == 0 else trade_data["contract_price"][0] * trade_data["size"] * self.setting.min_unit
        self.stats.trade_history.append(trade_data)
