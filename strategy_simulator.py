import sys
import numpy
import time
from datetime import datetime
from itertools import groupby

sys.path.append("lib")
import checker
import cache
import utils
import strategy
from loader import Loader
from simulator import SimulatorStats

from rakuten import RakutenSimulator as Simulator
from rakuten import Rakuten as SecuritiesCompony

class StrategySimulator:
    def __init__(self, simulator_setting, combination_setting, strategy_settings, verbose=False):
        self.simulator_setting = simulator_setting
        self.combination_setting = combination_setting
        self.strategy_settings = strategy_settings
        self.verbose = verbose
        self.stats = SimulatorStats()

        self.load_manda()

    def select_weekday(self, date, days):
        current = utils.to_datetime(date)
        for i in range(days):
            current = utils.select_weekday(current - utils.relativeterm(1, True))
        return utils.to_format(current)

    def load_manda(self):
        self.stock_split = Loader.stock_split()
        self.reverse_stock_split = Loader.reverse_stock_split()

        # 株式分割・統合が予定されていれば変化率を適用する
        self.stock_split["date"] = list(map(lambda x: self.apply_manda_date(x), self.stock_split["date"].astype(str).values.tolist()))
        self.reverse_stock_split["date"] = list(map(lambda x: self.apply_manda_date(x), self.reverse_stock_split["date"].astype(str).values.tolist()))

    def apply_manda_date(self, date):
        if utils.to_datetime(date) < utils.to_datetime("2019-08-01"):
            # 2019-08-01より前は3日前から株価データは変わっている
            return self.select_weekday(date, 3)
        else:
            # 2019-08-01以降は2日前から株価データは変わっている
            return self.select_weekday(date, 2)

    def strategy_creator(self, args):
        return strategy.load_strategy_creator(args, self.combination_setting)

    def select_codes(self, args, start_date, end_date):
        codes = []

        dates = self.strategy_creator(args).select_dates(start_date, end_date, instant=args.instant)
        for d in dates:
            date = utils.to_format(d)
            targets = self.get_targets(args, codes, date)
            codes = list(set(codes + targets))

        return codes

    def get_targets(self, args, targets, date):
        if args.code is None:
            if args.instant:
                date = utils.to_format(utils.select_weekday(utils.to_datetime(date), to_before=False))
            targets = list(self.strategy_creator(args).subject(date))
        else:
            targets = [args.code]
        return targets

    def log(self, message):
        if self.verbose:
             print(message)

    def create_simulator(self, args, codes, stocks, start_date, end_date, strategy_setting, ignore_manda=True):
        simulators = {}
        simulator_setting = self.simulator_setting
        simulator_setting.debug = self.verbose
        strategy_settings = self.strategy_settings[:-1] + [strategy_setting] # 最後の設定を変えて検証
        strategy_creator = self.strategy_creator(args)
        combination = strategy_creator.create_combination(strategy_settings)
        simulator_setting.strategy = strategy_creator.create(strategy_settings)
        for code in codes:
            if ignore_manda and self.is_manda(start_date, end_date, code):
                continue
            if not code in stocks.keys():
                continue
            simulators[code] = Simulator(simulator_setting)
        return combination, simulators

    def simulate_dates(self, codes, stocks, start_date, end_date):
        dates = []
        dates_dict = {}
        for code in codes:
            if not code in stocks.keys():
                continue
            dates_dict[code] = stocks[code].dates(start_date, end_date)
            dates = list(set(dates + dates_dict[code]))
        self.log("dates: %s" % dates)

        # 日付ごとにシミュレーション
        dates = sorted(dates, key=lambda x: utils.to_datetime(x))
        return dates

    def force_closing(self, dates, stocks, simulators):
        if len(dates) > 0:
            self.log("=== [closing] ===")
            for code in simulators.keys():
                split_data = stocks[code].split(dates[0], dates[-1])
                if len(split_data.daily) == 0:
                    continue
                self.log("[%s] closing: %s" % (code, split_data.daily["date"].iloc[-1]))
                simulators[code].force_closing(dates[-1], split_data)
        return simulators

    def closing(self, stats, simulators):
        auto_stop_loss = sum(list(map(lambda x: len(x.stats.lose_auto_stop_loss()) - len(x.stats.win_auto_stop_loss()), simulators.values())))

        conditions = [
            auto_stop_loss >= 3
        ]

        if all(conditions):
            for code in simulators.keys():
                simulators[code].closing(force_stop=True)
                stats["closing"] = True
        return stats, simulators

    def is_stock(self, code):
        return str(code).isdigit()

    def get_stock_split(self, start_date, end_date, code):
        if not self.is_stock(code):
            return []
        stock_split = self.stock_split[
            (self.stock_split["date"] >= start_date) &
            (self.stock_split["date"] <= end_date) &
            (self.stock_split["code"] == int(code))
        ]

        return stock_split

    def get_reverse_stock_split(self, start_date, end_date, code):
        if not self.is_stock(code):
            return []
        reverse_stock_split = self.reverse_stock_split[
            (self.reverse_stock_split["date"] >= start_date) &
            (self.reverse_stock_split["date"] <= end_date) &
            (self.reverse_stock_split["code"] == int(code))
        ]
        return reverse_stock_split

    def is_manda(self, start_date, end_date, code):
        stock_split = self.get_stock_split(start_date, end_date, code)
        reverse_stock_split = self.get_reverse_stock_split(start_date, end_date, code)

        return len(stock_split) > 0 or len(reverse_stock_split) > 0

    def manda_by_date(self, date, code, simulators):
        stock_split = self.get_stock_split(date, date, code)
        reverse_stock_split = self.get_reverse_stock_split(date, date, code)

        if len(stock_split) > 0:
            simulators[code].position.apply_split_ratio(stock_split["ratio"].iloc[0])
        if len(reverse_stock_split) > 0:
            simulators[code].position.apply_split_ratio(reverse_stock_split["ratio"].iloc[0])

        return simulators

    def simulates(self, strategy_setting, data, start_date, end_date, with_closing=True, ignore_manda=True):
        self.log("simulating %s %s" % (start_date, end_date))

        self.stats = SimulatorStats()
        args = data["args"]
        stocks = data["data"]
        index = data["index"]

        codes = self.get_targets(args, [], start_date)

        # シミュレーター準備
        combination, simulators = self.create_simulator(args, codes, stocks, start_date, end_date, strategy_setting, ignore_manda)

        # 日付のリストを取得
        dates = self.simulate_dates(codes, stocks, start_date, end_date)

        self.log("targets: %s" % list(simulators.keys()))
        capacity = None

        for step, date in enumerate(dates):
            # 休日はスキップ
            if not utils.is_weekday(utils.to_datetime(date)):
                self.log("%s is not weekday" % date)
                continue

            self.log("\n=== [%s] ===" % date)

            stats = self.stats.create_trade_data()
            stats["date"] = date

            binding = sum(list(map(lambda x: x.order_binding(), simulators.values())))

            for code in simulators.keys():
                # 対象日までのデータの整形
                self.log("[%s]" % code)
                # M&A 適用
                simulators = self.manda_by_date(date, code, simulators)

                simulators[code].capacity = simulators[code].capacity if capacity is None else capacity
                simulators[code].binding = binding - simulators[code].order_binding() # 自分の拘束分はsimulator側で加算するので引いておく
                simulators[code].simulate_by_date(date, stocks[code], index)
                capacity = simulators[code].capacity
                binding += simulators[code].order_binding() - simulators[code].unbound

            stats["unrealized_gain"] = self.sum_stats(simulators, lambda x: x.stats.last_unrealized_gain())
            stats["gain"] = self.sum_stats(simulators, lambda x: x.stats.get("gain", 0))
            self.stats.append(stats)

            stats, simulators = self.closing(stats, simulators)
            self.log("gain: %s, %s" % (stats["unrealized_gain"], stats["gain"]))

        # 手仕舞い
        if with_closing:
            simulators = self.force_closing(dates, stocks, simulators)

        return simulators

    def sum_stats(self, simulators, callback):
        return sum(list(map(lambda x: 0 if len(callback(x)) == 0 else callback(x)[-1], simulators.values())))

    def get_stats(self, simulators, start_date, end_date):
        # 統計 ====================================
        stats = {}
        for code in simulators.keys():
            stats[code] = simulators[code].stats

        results = self.create_stats(stats, start_date, end_date)
        results["per_day"] = self.create_stats_per_day(stats, simulators)
        return results


    def agg(self, stats, target, proc=None):
        results = {}
        for s in stats.values():
            for history in s.trade_history:
                date = history["date"]
                if date is None:
                    continue
                d = history[target] if proc is None else proc(history[target])
                if date in results.keys():
                    results[date] += d
                else:
                    results[date] = d
        return results

    def create_stats_per_day(self, stats, simulators):
        per_day = []
        dates = list(set(sum(list(map(lambda x: x.dates(), stats.values())),[])))
        gains = self.agg(stats, "gain", proc=lambda x: 0 if x is None else x)
        unrealized_gains = self.agg(stats, "unrealized_gain", proc=lambda x: 0 if x is None else x)
        for date in dates:
            trade = list(map(lambda x: x.find_by_date(date), stats.values()))
            trade = sum(list(filter(lambda x: len(x) > 0 and x[0]["gain"] is not None, trade)), [])
            per_day = per_day + [{
                "date": str(date),
                "gain": int(gains[date]),
                "unrealized_gain": int(unrealized_gains[date]),
                "trade": len(trade)
            }]

        return per_day

    def create_stats(self, stats, start_date, end_date):
        # 統計 =======================================
        wins = list(filter(lambda x: sum(x[1].gain_rate()) > 0, stats.items()))
        lose = list(filter(lambda x: sum(x[1].gain_rate()) < 0, stats.items()))
        win_codes = list(map(lambda x: x[0], wins))
        lose_codes = list(map(lambda x: x[0], lose))
        codes = win_codes + lose_codes
        commission = list(map(lambda x: sum(x[1].commission()), stats.items()))
        gain = list(map(lambda x: sum(x[1].gain()), stats.items()))
        position_size = self.agg(stats, "size").values()
        position_size = list(filter(lambda x: x != 0, position_size))
        position_term = list(map(lambda x: x[1].term(), stats.items()))
        position_term = list(filter(lambda x: x != 0, sum(position_term, [])))
        unavailable_assets = self.agg(stats, "unavailable_assets").values()
        min_assets = self.agg(stats, "min_assets", proc=lambda x: x - self.simulator_setting.assets).values()
        min_assets = list(map(lambda x: self.simulator_setting.assets + x, min_assets))
        sum_contract_price = list(map(lambda x: x[1].sum_contract_price(), stats.items()))
        agg_contract_price = self.agg(stats, "contract_price", proc=lambda x: 0 if x is None else x).values()
        oneday_commission = list(map(lambda x: SecuritiesCompony().oneday_commission(x), agg_contract_price))
        interest = list(map(lambda x: int(x * 0.028 / 365), unavailable_assets))
        auto_stop_loss = list(map(lambda x: len(x.auto_stop_loss()), stats.values()))
        win_auto_stop_loss = list(map(lambda x: len(x.win_auto_stop_loss()), stats.values()))
        lose_auto_stop_loss = list(map(lambda x: len(x.lose_auto_stop_loss()), stats.values()))
        drawdown = list(map(lambda x: x.drawdown(), stats.values()))
        drawdown = numpy.array(drawdown).T
        max_unrealized_gain = list(map(lambda x: max(x) if len(x) > 0 else 0, self.stats.unrealized_gain()))
        crash = sum(list(map(lambda x: x.crash(), stats.values())), [])

        if self.verbose:
            print(start_date, end_date, "assets:", self.simulator_setting.assets, "gain:", gain, sum(gain))
            for code, s in sorted(stats.items(), key=lambda x: sum(x[1].gain())):
                print("[%s] return: %s, unrealized:%s, drawdown: %s, trade: %s, win: %s, term: %s" % (code, sum(s.gain()), s.max_unrealized_gain(), s.max_drawdown(), s.trade_num(), s.win_trade_num(), s.max_term()))

        s = stats.values()
        results = {
            "start_date": start_date,
            "end_date": end_date,
            "codes": codes,
            "win": win_codes,
            "lose": lose_codes,
            "gain": sum(gain),
            "commission": sum(commission),
            "oneday_commission": sum(oneday_commission),
            "interest": sum(interest),
            "return": round(sum(gain) / self.simulator_setting.assets, 2),
            "init_assets": self.simulator_setting.assets,
            "min_assets": min(min_assets) if len(min_assets) > 0 else 0,
            "drawdown": round(max(list(map(lambda x: max(x), drawdown))) if len(drawdown) > 0 else 0, 2),
            "max_drawdown": round(max(list(map(lambda x: sum(x), drawdown))) if len(drawdown) > 0 else 0, 2),
            "win_trade": sum(list(map(lambda x: x.win_trade_num(), s))) if len(s) > 0 else 0,
            "trade": sum(list(map(lambda x: x.trade_num(), s))) if len(s) > 0 else 0,
            "position_size": round(numpy.average(position_size).item(), -2) if len(position_size) > 0 else 0,
            "max_position_size": max(position_size) if len(position_size) > 0 else 0,
            "position_term": round(numpy.average(position_term).item()) if len(position_term) > 0 else 0,
            "max_position_term": max(position_term) if len(position_term) > 0 else 0,
            "max_unavailable_assets": max(unavailable_assets) if len(s) > 0 and len(unavailable_assets) > 0 else 0,
            "sum_contract_price": sum(sum_contract_price) if len(s) > 0 else 0,
            "auto_stop_loss": sum(auto_stop_loss) if len(s) > 0 else 0,
            "win_auto_stop_loss": sum(win_auto_stop_loss) if len(s) > 0 else 0,
            "lose_auto_stop_loss": sum(lose_auto_stop_loss) if len(s) > 0 else 0,
            "max_unrealized_gain": max(max_unrealized_gain) if len(max_unrealized_gain) > 0 else 0,
            "crash": min(crash) if len(crash) > 0 else 0,
        }

        return results

