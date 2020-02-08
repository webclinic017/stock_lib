import sys
import numpy
import time

sys.path.append("lib")
import checker
import cache
import utils
import strategy
from simulator import Simulator, TradeRecorder


class StrategySimulator:
    def __init__(self, simulator_setting, combination_setting, strategy_settings, verbose=False):
        self.simulator_setting = simulator_setting
        self.combination_setting = combination_setting
        self.strategy_settings = strategy_settings
        self.verbose = verbose

    def strategy_creator(self, args):
        return strategy.load_strategy_creator(args, self.combination_setting)

    def append_daterange(self, codes, date, daterange):
        for code in codes:
            if not code in daterange.keys():
                daterange[code] = []
            daterange[code].append(date)
            daterange[code].sort()
        return daterange

    def select_codes(self, args, start_date, end_date):
        codes = []
        daterange = {}
        start = utils.to_datetime_by_term(start_date, with_time=args.daytrade)
        end = utils.to_datetime_by_term(end_date, with_time=args.daytrade) + utils.relativeterm(1, with_time=True)
        for date in utils.daterange(start, end):
            codes = self.get_targets(args, codes, utils.to_format_by_term(date, args.daytrade))
            daterange = self.append_daterange(codes, date, daterange)
        validate_codes = codes
        return codes, validate_codes, daterange

    def get_targets(self, args, targets, date):
        if args.code is None:
            date = utils.to_format(utils.to_datetime_by_term(date, args.daytrade))
            targets = list(set(targets + self.strategy_creator(args).subject(date)))
        else:
            targets = [args.code]
        return targets

    def log(self, message):
        if self.verbose:
             print(message)

    def get_data_by_date(data, date):
        # filter -> ohlc をすべてoにする-> add_stats
        d = data.daily
        d = d[d["date"] <= date].iloc[-300:].copy()
        for column in ["high", "low", "close"]:
            tmp = d[column].as_matrix().tolist()
            tmp[-1] = d["open"].iloc[-1]
            d[column] = tmp
        d = strategy.add_stats(data.code, d, data.rule)
        return d

    def simulates(self, strategy_setting, data, start_date, end_date):
        self.log("simulating %s %s" % (start_date, end_date))

        args = data["args"]
        stocks = data["data"]
        index = data["index"]
        daytrade = args.daytrade

        # シミュレーター準備
        simulators = {}
        simulator_setting = self.simulator_setting
        simulator_setting.debug = self.verbose
        strategy_settings = self.strategy_settings[:-1] + [strategy_setting] # 最後の設定を変えて検証
        simulator_setting.strategy = self.strategy_creator(args).create(strategy_settings)
        for code in stocks.keys():
            if stocks[code].split(start_date, end_date).daily["manda"].isin([1]).any(): # M&Aがあった銘柄はスキップ
                self.log("skip. M&A. %s" % code)
                continue
            simulators[code] = Simulator(simulator_setting)

        # 日付のリストを取得
        dates = []
        dates_dict = {}
        for code in stocks.keys():
            dates_dict[code] = stocks[code].dates(start_date, end_date)
            dates = list(set(dates + dates_dict[code]))

        # 日付ごとにシミュレーション
        dates = sorted(dates, key=lambda x: utils.to_datetime_by_term(x, daytrade))
        self.log("targets: %s" % simulators.keys())

        for date in dates:
            # 休日はスキップ
            if not utils.is_weekday(utils.to_datetime_by_term(date, daytrade)):
                self.log("%s is not weekday" % date)
                continue

            self.log("=== [%s] ===" % date)

            for code in simulators.keys():
                # 対象日までのデータの整形
                if date in dates_dict[code]:
                    self.log("[%s]" % code)
                    simulators[code].simulate_by_date(date, stocks[code], index)
                else:
                    self.log("[%s] is less data: %s" % (code, date))

        # 手仕舞い
        if len(dates) > 0:
            for code in simulators.keys():
                split_data = stocks[code].split(dates[0], dates[-1])
                if len(split_data.daily) == 0:
                    continue
                simulators[code].closing(dates[-1], split_data.daily["close"].iloc[-1])

        # 統計 ====================================
        stats = {}
        for code in simulators.keys():
            stats[code] = simulators[code].stats

        return self.get_results(stats, start_date, end_date)


    def agg(self, stats, target):
        results = {}
        for s in stats.values():
            for history in s.trade_history:
                date = history["date"]
                if date is None:
                    continue
                if date in results.keys():
                    results[date] += history[target]
                else:
                    results[date] = history[target]
        return results

    def get_results(self, stats, start_date, end_date):
        # 統計 =======================================
        wins = list(filter(lambda x: sum(x[1].gain_rate()) > 0, stats.items()))
        lose = list(filter(lambda x: sum(x[1].gain_rate()) < 0, stats.items()))
        win_codes = list(map(lambda x: x[0], wins))
        lose_codes = list(map(lambda x: x[0], lose))
        codes = win_codes + lose_codes
        gain = list(map(lambda x: sum(x[1].gain()), stats.items()))
        position_size = self.agg(stats, "size").values()
        position_size = list(filter(lambda x: x != 0, position_size,))
        position_term = list(map(lambda x: x[1].term(), stats.items()))
        position_term = list(filter(lambda x: x != 0, sum(position_term, [])))
        max_unavailable_assets = self.agg(stats, "unavailable_assets").values()
        sum_contract_price = list(map(lambda x: x[1].sum_contract_price(), stats.items()))

        if self.verbose:
            print(start_date, end_date, "assets:", self.simulator_setting.assets, "gain:", gain, sum(gain))
            for code, s in sorted(stats.items(), key=lambda x: sum(x[1].gain())):
                print("[%s] return: %s, drawdown: %s, trade: %s, win: %s" % (code, sum(s.gain()), s.max_drawdown(), s.trade_num(), s.win_trade_num()))

        s = stats.values()
        results = {
            "start_date": start_date,
            "end_date": end_date,
            "codes": codes,
            "win": win_codes,
            "lose": lose_codes,
            "gain": sum(gain),
            "return": round(sum(gain) / self.simulator_setting.assets, 2),
            "drawdown": round(numpy.average(list(map(lambda x: x.max_drawdown(), s))).item() if len(s) > 0 else 0, 2),
            "max_drawdown": round(max(list(map(lambda x: x.max_drawdown(), s))) if len(s) > 0 else 0, 2),
            "win_trade": sum(list(map(lambda x: x.win_trade_num(), s))) if len(s) > 0 else 0,
            "trade": sum(list(map(lambda x: x.trade_num(), s))) if len(s) > 0 else 0,
            "position_size": round(numpy.average(position_size).item(), -2) if len(position_size) > 0 else 0,
            "max_position_size": max(position_size) if len(position_size) > 0 else 0,
            "position_term": round(numpy.average(position_term).item()) if len(position_term) > 0 else 0,
            "max_position_term": max(position_term) if len(position_term) > 0 else 0,
            "max_unavailable_assets": max(max_unavailable_assets) if len(s) > 0 and len(max_unavailable_assets) > 0 else 0,
            "sum_contract_price": sum(sum_contract_price) if len(s) > 0 else 0,
        }

        return results


