# -*- coding: utf-8 -*-
import numpy
import random

# 売買の状態
class Position:
    def __init__(self, num = 0, value = 0, term = 0, initial = None, system="actual", method="long"):
        self._num = num
        self._value = []
        self._initial = initial
        if value != 0:
            self._value = [value]
        self._term = term # 保有期間
        self.system = system
        self.method = method

    def add_history(self, num, value):
        self._value.append(value) # 平均取得価格の計算のため
        self._num += num
        if self.num() == 0:
          self._term = 0
          self._value = []

    # 新規
    def new(self, num, value):
        self.add_history(num, value)
        price = -num * value
        return price

    # 返済
    def repay(self, num, value):
        assert len(self._value) > 0, "do not repay. not hold."
        price = num * self.value()
        price = price + self.gain(value)
        self.add_history(-num, value)
        return price

    # 保有株数
    def num(self):
        return self._num

    # 平均取得価格
    def value(self):
        if len(self._value) == 0:
            return 0
        return sum(self._value) / len(self._value)

    # ポジション取得時の価格
    def initial(self):
        if self._initial is None:
            if len(self._value) > 0:
                return self._value[0]
            else:
                return None
        return self._initial

    # 損益
    def gain(self, value):
        if self.value() is None:
            return 0
        if self.is_short():
            return (self.value() - value) * self.num()
        else:
            return (value - self.value()) * self.num()

    # 損益レシオ
    def gain_rate(self, value):
        if self.value() is None or self.value() == 0:
            return 0
        if self.is_short():
            return (self.value() - value) / self.value()
        else:
            return (value - self.value()) / self.value()

    # 取得時の価格を設定
    def set_initial(self, initial):
        self._initial = initial

    # ポジション取得からの期間
    def term(self):
        return self._term

    # 期間を設定
    def set_term(self, term):
        self._term = term

    # 保有期間を加算
    def increment_term(self):
        self._term += 1

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
        self._term += 1

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
        self.strategy = {"daily": None}
        self.min_data_length = 30
        self.assets = 0
        self.commission = 150
        self.debug = False
        self.sizing = False
        self.error_rate = 0.00
        self.virtual_trade = True # 仮想取引 Falseにすると注文をスタックしていく
        self.short_trade = False
        self.auto_stop_loss = False
        self.stop_loss_rate = 0.02

# 統計
class SimulatorStats:
    def __init__(self, setting):
        self.assets_history = []
        self.max_assets = setting.assets
        self.drawdown = []
        self.trade = []
        self.trade_history = []
        self.profits = []
        self.loss = []

    def trade_num(self):
        return len(list(filter(lambda x: x!=0, self.trade)))

    def win_trade_num(self):
        return len(list(filter(lambda x: x==1, self.trade)))

    def loss_trade_num(self):
        return len(list(filter(lambda x: x==-1, self.trade)))

    # 勝率
    def win_rate(self):
        trade_num = self.trade_num()
        if trade_num == 0:
            return 0
        win = len(list(filter(lambda x: x==1, self.trade)))
        return win / float(trade_num)

    # 資産の履歴を追加する
    def add_assets_history(self, date, assets):
        self.assets_history.append({"date": date, "assets": assets})
        if self.max_assets < assets:
            self.max_assets = assets
            self.drawdown.append({"date": date, "drawdown": 0})
        else:
            diff = self.max_assets - assets
            dd = diff/float(self.max_assets)
            dd = int(dd * 100)
            dd = 0.0 if dd == 0 else dd / 100
            self.drawdown.append({"date": date, "drawdown": dd})

    # 最大ドローダウン
    def max_drawdown(self):
        dd = list(map(lambda x: x["drawdown"], self.drawdown))
        if len(dd) == 0:
            return 0
        return max(dd)

    # 平均利益率
    def average_profit_rate(self):
        return numpy.average(self.profits) / self.win_trade_num()

    # 平均損失率
    def average_loss_rate(self):
        return numpy.average(self.loss) / self.loss_trade_num()

# シミュレーター
class Simulator:
    def __init__(self, setting, position = None):
        system = "credit" if setting.short_trade else "actual"
        method = "short" if setting.short_trade else "long"
        self._setting = setting
        self._position = position if position is not None else Position(system=system, method=method)
        self._assets = setting.assets
        self._stats = SimulatorStats(setting)
        self._logs = []
        self._new_orders = []
        self._repay_orders = []

    def log(self, message):
        if self._setting.debug:
            print(message)
            self._logs.append(message)

    def stats(self):
        return self._stats

    def position(self):
        return self._position

    # ルールすべてに当てはまるならTrue
    def apply_all_rules(self, data, index, rules, rate=1.0):
        if len(rules) == 0:
            return False
        appliable_data = self.create_appliable_data(data, index, rate=rate)
        results = list(map(lambda x: x.apply(appliable_data), rules))
        results = list(filter(lambda x: x is not None, results))
        return results

    def create_appliable_data(self, data, index, rate=1.0):
        return AppliableData(data, index, self._position, self.total_assets(data["daily"]["close"].iloc[-1]), self._setting, self._stats, rate)

    # 総資産
    def total_assets(self, value):
        num = self._position.num()
        holdings = value * num
        return self._assets + holdings

    # 新規
    def new(self, value, num):
        value = value * (1 + random.uniform(0.0, self._setting.error_rate))
        if (self._assets - value * num) <= 0:
            # 資産が足りなければスルー
            self.log(" - assets_not_enough: num %s, value %s" % (num, value))
            return False
        if num <= 0:
            return False

        pos = self.total_assets(self._position.value())
        self._assets += self._position.new(num, value)
        self.commission()

        self.log(" new: %s yen x %s, total %s, ave %s, assets %s" % (value, num, self._position.num(), self._position.value(), self.total_assets(value)))

        if self._setting.auto_stop_loss:
            order = Order(num, [lambda x: x["position"].gain(x["value"]) < - (pos * 0.02)], is_reverse_limit=True) # 損失が総資産の2%以上なら即損切
            self.log(" - auto_stop_loss_order: num %s" % (num))

        return True

    # 返済
    def repay(self, value, num):
        value = value * (1 - random.uniform(0.0, self._setting.error_rate))
        if (num <= 0 or self._position.num() <= 0):
            return False

        pos = self.total_assets(self._position.value())
        gain_rate = self._position.gain_rate(value)
        gain = self._position.gain(value)
        self._assets += self._position.repay(num, value)
        self.commission()

        # stats
        if self.total_assets(value) > pos:
          self._stats.trade.append(1)
          self._stats.profits.append(gain_rate)
        else:
          self._stats.trade.append(-1)
          self._stats.loss.append(gain_rate)

        self.log(" repay: %s yen x %s, total %s, ave %s, assets %s : gain %s" % (value, num, self._position.num(), self._position.value(), self.total_assets(value), gain))
        return True

    # 全部売る
    def closing(self, value):
        num = self._position.num()
        self.repay(value, num)

    # 取引手数料
    # TODO 実際のものに合わせる
    def commission(self):
        self._assets -= self._setting.commission

    def new_order(self, price, agreed_price=None):
        signals = list(filter(lambda x: x.signal(price, self._position), self._new_orders)) # 条件を満たした注文
        self._new_orders = list(filter(lambda x: not x.signal(price, self._position), self._new_orders)) # 残っている注文

        # 注文の価格を設定
        for i in range(len(signals)):
            signals[i].price = price if agreed_price is None else agreed_price
        return signals

    def repay_order(self, price, agreed_price=None):
        signals = list(filter(lambda x: x.signal(price, self._position), self._repay_orders)) # 条件を満たした注文
        self._repay_orders = list(filter(lambda x: not x.signal(price, self._position), self._repay_orders)) # 残っている注文

        for i in range(len(signals)):
            signals[i].price = price if agreed_price is None else agreed_price

        return signals

    def reverse_limit_repay_order(self, price, agreed_price=None):
        signals = list(filter(lambda x: x.signal(price, self._position) and x.is_reverse_limit, self._repay_orders)) # 条件を満たした注文
        self._repay_orders = list(filter(lambda x: not x.signal(price, self._position) or not x.is_reverse_limit, self._repay_orders)) # 残っている注文

        for i in range(len(signals)):
            signals[i].price = price if agreed_price is None else agreed_price

        return signals

    # 損切りの逆指値価格
    def reverse_limit_price(self, price):
        # 注文の価格を設定
        price_range = (self.total_assets(price) * 0.02) / self._position.num()
        if self._setting.short_trade:
            value = self._position.value() + price_range
        else:
            value = self._position.value() - price_range
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
        stats["assets"] = int(self._assets)
        stats["gain"] = stats["assets"] - self._setting.assets
        stats["return"] = float((stats["assets"] - self._setting.assets) / float(self._setting.assets))
        stats["win_rate"] = float(self.stats().win_rate())
        stats["drawdown"] = float(self.stats().max_drawdown())
        stats["trade"] = int(self.stats().trade_num())
        stats["win_trade"] = int(self.stats().win_trade_num())
        # 平均利益率
        stats["average_profit_rate"] = self.stats().average_profit_rate() if len(self.stats().profits) > 0 else 0.0
        # 平均損失率
        stats["average_loss_rate"] = self.stats().average_loss_rate() if len(self.stats().loss) > 0 else 0.0
        # リワードリスクレシオ
        win = stats["average_profit_rate"] * stats["win_rate"]
        loss = abs(stats["average_loss_rate"]) * (1 - stats["win_rate"])
        stats["reword"] = win
        stats["risk"] = loss
        stats["rewordriskratio"] = win / loss if loss > 0 else win

        # トレード履歴
        stats["trade_history"] = self.stats().trade_history

        if self._setting.debug:
            stats["logs"] = self._logs

        return stats

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
        assert type(data) is SimulatorData, "data is not SimulatorData."

        term_data = {
            "daily": data.daily[data.daily["date"] <= date],
            "weekly": data.weekly[data.weekly["date"] <= date]
        }
        term_index = {}
        for k, v in index.items():
            term_index[k] = v[v["date"] <= date]

        today = data.daily[data.daily["date"] == date]

        assert len(today) > 0, "not found %s data" % date

        price = today["close"].iloc[-1]
        self.log("date: %s, price %s" % (date, price))

        self.trade(self._setting.strategy["daily"], price, term_data, term_index)
        return self.total_assets(price)


    # トレード
    def trade(self, strategy, price, data, index):
        assert type(data) is dict, "data is not dict."
        # stats
        date = data["daily"]["date"].iloc[-1]
        self._stats.trade.append(0)
        self._stats.add_assets_history(date, self.total_assets(price))
#        self.log("[drawdown] %s" % self._stats.drawdown[-1])
        trade_data = {"date": date, "new": None, "repay": None, "gain": None, "term": 0, "size": 0}

        # 判断に必要なデータ数がない
        if len(data["daily"]) < self._setting.min_data_length or price == 0:
            self.log("less data. skip trade. [%s - %s]" % (data["daily"]["date"].iloc[0], data["daily"]["date"].iloc[-1]))
            self._stats.trade_history.append(trade_data)
            return self.total_assets(0)

        # 新規ポジションがあり、返済ルールに当てはまる場合売る
        if self._position.num() > 0:
            self._position.increment_term()

        # 寄り付き====================================================================
        # 新規実行
        open_price = data["daily"]["open"].iloc[-1]
        if self._setting.virtual_trade:
            new_orders = self.new_order(open_price)
        else:
            new_orders = []
        for order in new_orders:
            if self.new(order.price, order.num):
                trade_data["new"] = order.price

        ## 引け直前===================================================================
        if self._position.num() > 0:
            # 損切・返済気配のときは新規シグナルが出てても買わない
            for order in self.taking_signal(strategy, data, index):
                self.log(" - taking_order: num %s" % (order.num))
                self._repay_orders.append(order)

            for order in self.stop_loss_signal(strategy, data, index):
                self.log(" - stop_loss_order: num %s" % (order.num))
                self._repay_orders.append(order)

            for order in self.closing_signal(strategy, data, index):
                self.log(" - closing_order: num %s" % (order.num))
                self._repay_orders.append(order)

        # 返済注文実行
        repay_orders = []
        if self._setting.virtual_trade and self._position.num() > 0:
            # 自動損切の注文処理（longなら安値、shortなら高値）
            if self._setting.auto_stop_loss:
                if self._setting.short_trade:
                    value = self.reverse_limit_price(data["daily"]["high"].iloc[-1])
                    repay_orders += self.reverse_limit_repay_order(price, value)
                else:
                    value = self.reverse_limit_price(data["daily"]["low"].iloc[-1])
                    repay_orders += self.reverse_limit_repay_order(price, value)
            repay_orders += self.repay_order(price)

        for order in repay_orders:
            gain = self._position.gain(order.price)
            if self.repay(order.price, order.num):
                trade_data["repay"] = order.price
                trade_data["gain"] = gain

        ## 引け後=====================================================================
        # 新規ルールに当てはまる場合買う
        for order in self.new_signal(strategy, data, index):
            self.log(" - new_order: num %s" % (order.num))
            self._new_orders.append(order)


        # トレード履歴
        trade_data["size"] = self._position.num()
        trade_data["term"] = self._position.term()
        self._stats.trade_history.append(trade_data)

