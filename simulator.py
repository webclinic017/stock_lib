# -*- coding: utf-8 -*-
import os
import numpy
import random
import utils
import pandas

# 売買の状態
class Position:
    def __init__(self, num = 0, value = 0, term = 0, initial = None, system="actual", method="long"):
        self.num = int(num)
        self.value = []
        self.initial = initial
        if value != 0:
            self.value = [value]
        self.term = term # 保有期間
        self.system = system
        self.method = method

    def add_history(self, num, value):
        for _ in range(int(num/100)):
            self.value.append(value) # 平均取得価格の計算のため
        self.num += num
        if self.get_num() == 0:
          self.term = 0
          self.value = []

    # 新規
    def new(self, num, value):
        self.add_history(num, value)
        price = -num * value
        return price

    # 返済
    def repay(self, num, value):
        assert len(self.value) > 0, "do not repay. not hold."
        price = num * self.get_value()
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
            return (self.get_value() - value) * self.get_num()
        else:
            return (value - self.get_value()) * self.get_num()

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
    def __init__(self, num, conditions, is_short=False, is_reverse_limit=False, is_limit=False):
        self.num = num
        self.term = 0
        self.price = None
        self.conditions = conditions
        self.is_short = is_short
        self.is_reverse_limit = is_reverse_limit
        self.is_limit = is_limit


    def increment_term(self):
        self.term += 1

    def signal(self, value, position):
        data = {
            "value": value,
            "position": position
        }
        return all(list(map(lambda x:x(data), self.conditions)))

# ルール適用用データ
class AppliableData:
    def __init__(self, data, index, position, assets, setting, stats, rate):
        self.data = data # データ
        self.index = index # 指標データ
        self.position = position # ポジション
        self.assets = assets # 資産
        self.setting = setting # 設定
        self.stats = stats # 統計データ
        self.rate = rate # 取引レート

class SimulatorData:
    def __init__(self, code, daily, weekly, rule):
        self.code = code
        self.daily = daily
        self.weekly = weekly
        self.rule = rule

    def split(self, start_date, end_date):
        d = self.daily[self.daily["date"] <= end_date]
        d = d[d["date"] >= start_date]
        w = self.weekly[self.weekly["date"] <= end_date]
        w = w[w["date"] >= start_date]
        return SimulatorData(self.code, d, w, self.rule)

    def dates(self, start_date, end_date):
        d = self.daily[self.daily["date"] >= start_date]
        d = d[d["date"] <= end_date]
        dates = d["date"].copy().astype(str).as_matrix().tolist()
        return dates

    def at(self, date):
        return self.daily[self.daily["date"] == date]

# シミュレーター設定
class SimulatorSetting:
    def __init__(self):
        self.strategy = None
        self.min_data_length = 30
        self.assets = 0
        self.commission = 150
        self.debug = False
        self.error_rate = 0.00
        self.virtual_trade = True # 仮想取引 Falseにすると注文をスタックしていく
        self.short_trade = False
        self.auto_stop_loss = False
        self.stop_loss_rate = 0.02
        self.taking_rate = 0.005

# 統計
class SimulatorStats:
    def __init__(self):
        self.trade_history = []

    def create_trade_data(self):
        trade_data = {
            "date": None,
            "new": None,
            "repay": None,
            "gain": None,
            "gain_rate": None,
            "assets": None,
            "available_assets": None,
            "term": 0,
            "size": 0
        }
        return trade_data

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
        return list(map(lambda x: x["assets"] - x["available_assets"], self.trade_history))

    def max_unavailable_assets(self):
        if len(self.unavailable_assets()) == 0:
            return 0
        return max(self.unavailable_assets())

    def drawdown(self):
        drawdown = []
        assets = self.assets()
        for i in range(len(assets)):
            max_assets = max(assets[:i]) if len(assets[:i]) > 0 else assets[i]
            dd = round((max_assets - assets[i])/float(max_assets), 2)
            drawdown.append(dd)
        return drawdown

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

class TradeRecorder:
    def __init__(self, output_dir=""):
        self.output_dir = "/tmp/trade_recorder/%s" % output_dir
        self.pattern = {"new": [], "repay": [], "gain": []}
        self.columns = None
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def set_columns(self, columns):
        self.columns = columns

    def pattern_num(self, num):
        return int(num / 100)

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
        self.position = position if position is not None else Position(system=system, method=method)
        self.assets = setting.assets
        self.stats = SimulatorStats()
        self.logs = []
        self.new_orders = []
        self.repay_orders = []
        self.trade_recorder = TradeRecorder()

    def log(self, message):
        if self.setting.debug:
            print(message)
            self.logs.append(message)

    # ルールすべてに当てはまるならTrue
    def apply_all_rules(self, data, index, rules, rate=1.0):
        if len(rules) == 0:
            return False
        appliable_data = self.create_appliable_data(data, index, rate=rate)
        results = list(map(lambda x: x.apply(appliable_data), rules))
        results = list(filter(lambda x: x is not None, results))
        return results

    def create_appliable_data(self, data, index, rate=1.0):
        return AppliableData(data, index, self.position, self.total_assets(data.daily["close"].iloc[-1].item()), self.setting, self.stats, rate)

    # 総資産
    def total_assets(self, value):
        num = self.position.get_num()
        holdings = float(value) * int(num)
        return self.assets + holdings

    # 新規
    def new(self, value, num):
        value = value * (1 + random.uniform(0.0, self.setting.error_rate))
        if (self.assets - value * num) <= 0:
            # 資産が足りなければスルー
            self.log(" - assets_not_enough: num %s, value %s" % (num, value))
            return False
        if num <= 0:
            return False

        pos = self.total_assets(self.position.get_value())
        self.assets += self.position.new(num, value)
        self.commission()

        self.log(" new: %s yen x %s, total %s, ave %s, assets %s" % (value, num, self.position.get_num(), self.position.get_value(), self.total_assets(value)))

        if self.setting.auto_stop_loss:
            order = Order(num, [lambda x: x["position"].gain(x["value"]) < - (pos * self.setting.stop_loss_rate)], is_reverse_limit=True) # 損失が総資産の2%以上なら即損切
            self.log(" - auto_stop_loss_order: num %s" % (num))

        return True

    # 返済
    def repay(self, value, num):
        value = value * (1 - random.uniform(0.0, self.setting.error_rate))
        if (num <= 0 or self.position.get_num() <= 0):
            return False

        pos = self.total_assets(self.position.get_value())
        gain_rate = self.position.gain_rate(value)
        gain = self.position.gain(value)
        self.assets += self.position.repay(num, value)
        self.commission()

        self.log(" repay: %s yen x %s, total %s, ave %s, assets %s : gain %s" % (value, num, self.position.get_num(), self.position.get_value(), self.total_assets(value), gain))
        return True

    # 全部売る
    def closing(self, value, data=None):
        num = self.position.get_num()
        gain = self.position.gain_rate(value)
        if self.repay(value, num) and data is not None:
            self.trade_recorder.repay(data.iloc[-1], gain, num)

        self.new_orders = []
        self.repay_orders = []

    # 取引手数料
    # TODO 実際のものに合わせる
    def commission(self):
        self.assets -= self.setting.commission

    def new_order(self, price, agreed_price=None):
        signals = list(filter(lambda x: x.signal(price, self.position), self.new_orders)) # 条件を満たした注文
        self.new_orders = list(filter(lambda x: not x.signal(price, self.position), self.new_orders)) # 残っている注文

        # 注文の価格を設定
        for i in range(len(signals)):
            signals[i].price = price if agreed_price is None else agreed_price
        return signals

    def repay_order(self, price, agreed_price=None):
        signals = list(filter(lambda x: x.signal(price, self.position), self.repay_orders)) # 条件を満たした注文
        self.repay_orders = list(filter(lambda x: not x.signal(price, self.position), self.repay_orders)) # 残っている注文

        for i in range(len(signals)):
            signals[i].price = price if agreed_price is None else agreed_price

        return signals

    def reverse_limit_repay_order(self, price, agreed_price=None):
        signals = list(filter(lambda x: x.signal(price, self.position) and x.is_reverse_limit, self.repay_orders)) # 条件を満たした注文
        self.repay_orders = list(filter(lambda x: not x.signal(price, self.position) or not x.is_reverse_limit, self.repay_orders)) # 残っている注文

        for i in range(len(signals)):
            signals[i].price = price if agreed_price is None else agreed_price

        return signals

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
    def new_signal(self, strategy, data, index, rate=1.0):
        return self.apply_all_rules(data, index, strategy.new_rules, rate=rate)

    def stop_loss_signal(self, strategy, data, index, rate=1.0):
        return self.apply_all_rules(data, index, strategy.stop_loss_rules, rate=rate)

    def taking_signal(self, strategy, data, index, rate=1.0):
        return self.apply_all_rules(data, index, strategy.taking_rules, rate=rate)

    def closing_signal(self, strategy, data, index, rate=1.0):
        return self.apply_all_rules(data, index, strategy.closing_rules, rate=rate)

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
        win = stats["average_profit_rate"] * stats["win_rate"]
        loss = abs(stats["average_loss_rate"]) * (1 - stats["win_rate"])
        stats["reword"] = win
        stats["risk"] = loss
        stats["rewordriskratio"] = win / loss if loss > 0 else win

        # トレード履歴
        stats["trade_history"] = self.stats.trade_history

        if self.setting.debug:
            stats["logs"] = self.logs

        return stats

    def create_term_data(self, date, data):
        assert type(data) is SimulatorData, "data is not SimulatorData."

        weekly = data.weekly[data.weekly["date"] <= date].iloc[:-1] # weeklyは最新の足は確定していないので最新のは除外する

        daily = data.daily[data.daily["date"] <= date]
        term_data = SimulatorData(data.code, daily, weekly, data.rule)

        return term_data

    def simulate(self, dates, data, index):
        assert type(data) is SimulatorData, "data is not SimulatorData."

        print(dates)
        for date in dates:
            total_assets = self.simulate_by_date(date, data, index)

        # 統計取得のために全部手仕舞う
        self.closing(data.daily["close"].iloc[-1])

        stats = self.get_stats()

        self.log("result assets: %s" % stats["assets"])

        return stats

    # 高速化
    def simulate_by_date(self, date, data, index):
        term_data = self.create_term_data(date, data)

        term_index = {}
        for k, v in index.items():
            term_index[k] = v[v["date"] <= date]

        today = data.daily[data.daily["date"] == date]

        assert len(today) > 0, "not found %s data" % date

        price = today["open"].iloc[-1].item()
        self.log("date: %s, price %s" % (date, price))

        self.trade(self.setting.strategy, price, term_data, term_index)
        return self.total_assets(price)

    def repay_signal(self, strategy, data, index):
        for order in self.taking_signal(strategy, data, index):
            if order.num > 0:
                self.log(" - taking_order: num %s" % (order.num))
            self.repay_orders.append(order)

        for order in self.stop_loss_signal(strategy, data, index):
            if order.num > 0:
                self.log(" - stop_loss_order: num %s" % (order.num))
            self.repay_orders.append(order)

        for order in self.closing_signal(strategy, data, index):
            if order.num > 0:
                self.log(" - closing_order: num %s" % (order.num))
            self.repay_orders.append(order)

    def order_adjust(self):
        # ポジションがなければ返済シグナルは捨てる
        if self.position.get_num() <= 0 and len(self.repay_orders) > 0:
            self.log("[cancel] repay order")
            self.repay_orders = []

        # 新規・返済が同時に出ている場合何もしない
        if len(self.new_orders) > 0 and len(self.repay_orders) > 0:
            self.log("[cancel] new/repay order")
            self.new_orders = []
            self.repay_orders = []

    # トレード
    def trade(self, strategy, price, data, index):
        assert type(data) is SimulatorData, "data is not SimulatorData."
        # stats
        date = data.daily["date"].iloc[-1]
        self.trade_recorder.set_columns(data.daily.columns)
        trade_data = self.stats.create_trade_data()
        trade_data["date"]              = date
        trade_data["assets"]            = self.total_assets(price)
        trade_data["available_assets"]  = self.assets

        # 判断に必要なデータ数がない
        if len(data.daily) < self.setting.min_data_length or price == 0:
            self.log("less data. skip trade. [%s - %s]" % (data.daily["date"].iloc[0], data.daily["date"].iloc[-1]))
            self.stats.trade_history.append(trade_data)
            return self.total_assets(0)

        # ポジションの保有期間を増やす
        if self.position.get_num() > 0:
            self.position.increment_term()

        # 寄り付き====================================================================
        if self.setting.virtual_trade:
            # 仮装トレードなら注文をキューから取得
            new_orders = self.new_order(price)
            repay_orders = []
            if self.position.get_num() > 0:
                # 自動損切の注文処理（longなら安値、shortなら高値）
                if self.setting.auto_stop_loss:
                    if self.setting.short_trade:
                        value = self.reverse_limit_price(data.daily["high"].iloc[-1])
                        repay_orders += self.reverse_limit_repay_order(price, value)
                    else:
                        value = self.reverse_limit_price(data.daily["low"].iloc[-1])
                        repay_orders += self.reverse_limit_repay_order(price, value)
                repay_orders += self.repay_order(price)

            # 新規注文実行
            for order in new_orders:
                if self.new(order.price, order.num):
                    self.trade_recorder.new(data.daily.iloc[-1], order.num)
                    trade_data["new"] = order.price

            # 返済注文実行
            for order in repay_orders:
                if self.position.get_num() <= 0:
                    self.repay_orders = [] # ポジションがなくなってたら以降の注文はキャンセル
                    break
                gain        = self.position.gain(order.price)
                gain_rate   = self.position.gain_rate(order.price)
                if self.repay(order.price, order.num):
                    self.trade_recorder.repay(data.daily.iloc[-1], gain_rate, order.num)
                    trade_data["repay"] = order.price
                    trade_data["gain"] = gain
                    trade_data["gain_rate"] = gain_rate

        ## 引け後=====================================================================
        # 新規ルールに当てはまる場合買う
        for order in self.new_signal(strategy, data, index):
            self.log(" - new_order: num %s" % (order.num))
            self.new_orders.append(order)

        # 返済注文
        self.repay_signal(strategy, data, index)
        # 注文の整理
        self.order_adjust()

        # トレード履歴
        trade_data["size"] = self.position.get_num()
        trade_data["term"] = self.position.get_term()
        self.stats.trade_history.append(trade_data)

