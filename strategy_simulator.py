import sys
import numpy
import time
from datetime import datetime

sys.path.append("lib")
import checker
import cache
import utils
import strategy
from simulator import Simulator


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
            daterange[code] = list(set(daterange[code]))
            daterange[code].sort()
        return daterange

    def select_codes(self, args, start_date, end_date):
        codes = []
        daterange = {}

        dates = list(utils.daterange(utils.to_datetime(start_date), utils.to_datetime(end_date)))
        for d in dates:
            date = utils.to_format(d)
            targets = self.get_targets(args, codes, date)
            codes = list(set(codes + targets))
            daterange = self.append_daterange(targets, d, daterange)

        return codes, daterange

    def get_targets(self, args, targets, date):
        if args.code is None:
            date = utils.to_format(utils.to_datetime(date))
            targets = list(self.strategy_creator(args).subject(date))
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
            tmp = d[column].values.tolist()
            tmp[-1] = d["open"].iloc[-1]
            d[column] = tmp
        d = strategy.add_stats(data.code, d, data.rule)
        return d

    def simulates(self, strategy_setting, data, start_date, end_date, daterange):
        self.log("simulating %s %s" % (start_date, end_date))

        args = data["args"]
        stocks = data["data"]
        index = data["index"]

        codes = self.get_targets(args, [], start_date)

        # シミュレーター準備
        simulators = {}
        simulator_setting = self.simulator_setting
        simulator_setting.debug = self.verbose
        strategy_settings = self.strategy_settings[:-1] + [strategy_setting] # 最後の設定を変えて検証
        strategy_creator = self.strategy_creator(args)
        simulator_setting = strategy.apply_long_short(args, simulator_setting)
        simulator_setting.strategy = strategy_creator.create(strategy_settings)
        for code in codes:
            if stocks[code].split(start_date, end_date).daily["manda"].isin([1]).any(): # M&Aがあった銘柄はスキップ
                self.log("skip. M&A. %s" % code)
                continue
            simulators[code] = Simulator(simulator_setting)

        # 日付のリストを取得
        dates = []
        dates_dict = {}
        for code in codes:
            dates_dict[code] = stocks[code].dates(start_date, end_date)
            dates = list(set(dates + dates_dict[code]))
        self.log("dates: %s" % dates)

        # 日付ごとにシミュレーション
        dates = sorted(dates, key=lambda x: utils.to_datetime(x))
        self.log("targets: %s" % list(simulators.keys()))
        capacity = None

        for date in dates:
            # 休日はスキップ
            if not utils.is_weekday(utils.to_datetime(date)):
                self.log("%s is not weekday" % date)
                continue

            self.log("=== [%s] ===" % date)

            for code in codes:
                if not code in simulators.keys():
                    continue
                # 対象日までのデータの整形
                if date in dates_dict[code]:
                    self.log("[%s]" % code)
                    simulators[code].capacity = simulators[code].capacity if capacity is None else capacity
                    simulators[code].simulate_by_date(date, stocks[code], index)
                    capacity = simulators[code].capacity
                else:
                    self.log("[%s] is less data: %s" % (code, date))

        # 手仕舞い
        if len(dates) > 0:
            self.log("=== [closing] ===")
            for code in codes:
                if not code in simulators.keys():
                    continue
                split_data = stocks[code].split(dates[0], dates[-1])
                if len(split_data.daily) == 0:
                    continue
                self.log("[%s] closing: %s" % (code, split_data.daily["date"].iloc[-1]))
                simulators[code].closing(dates[-1], split_data.daily["low"].iloc[-1], split_data.daily["high"].iloc[-1], split_data.daily["close"].iloc[-1])

        # 統計 ====================================
        stats = {}
        for code in simulators.keys():
            stats[code] = simulators[code].stats

        return self.get_results(stats, start_date, end_date)


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

    def get_results(self, stats, start_date, end_date):
        # 統計 =======================================
        wins = list(filter(lambda x: sum(x[1].gain_rate()) > 0, stats.items()))
        lose = list(filter(lambda x: sum(x[1].gain_rate()) < 0, stats.items()))
        win_codes = list(map(lambda x: x[0], wins))
        lose_codes = list(map(lambda x: x[0], lose))
        codes = win_codes + lose_codes
        commission = list(map(lambda x: sum(x[1].commission()), stats.items()))
        gain = list(map(lambda x: sum(x[1].gain()), stats.items()))
        position_size = self.agg(stats, "size").values()
        position_size = list(filter(lambda x: x != 0, position_size,))
        position_term = list(map(lambda x: x[1].term(), stats.items()))
        position_term = list(filter(lambda x: x != 0, sum(position_term, [])))
        unavailable_assets = self.agg(stats, "unavailable_assets").values()
        min_assets = self.agg(stats, "min_assets", proc=lambda x: x - self.simulator_setting.assets).values()
        min_assets = list(map(lambda x: self.simulator_setting.assets + x, min_assets))
        sum_contract_price = list(map(lambda x: x[1].sum_contract_price(), stats.items()))
        agg_contract_price = self.agg(stats, "contract_price", proc=lambda x: 0 if x is None else x).values()
        import rakuten
        oneday_commission = list(map(lambda x: rakuten.oneday_commission(x), agg_contract_price))
        interest = list(map(lambda x: int(x * 0.028 / 365), unavailable_assets))
        auto_stop_loss = list(map(lambda x: len(x[1].auto_stop_loss()), stats.items()))

        if self.verbose:
            print(start_date, end_date, "assets:", self.simulator_setting.assets, "gain:", gain, sum(gain))
            for code, s in sorted(stats.items(), key=lambda x: sum(x[1].gain())):
                print("[%s] return: %s, commission: %s, drawdown: %s, trade: %s, win: %s, term: %s" % (code, sum(s.gain()), sum(s.commission()), s.max_drawdown(), s.trade_num(), s.win_trade_num(), s.max_term()))

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
            "min_assets": min(min_assets) if len(min_assets) > 0 else 0,
            "drawdown": round(numpy.average(list(map(lambda x: x.max_drawdown(), s))).item() if len(s) > 0 else 0, 2),
            "max_drawdown": round(max(list(map(lambda x: x.max_drawdown(), s))) if len(s) > 0 else 0, 2),
            "win_trade": sum(list(map(lambda x: x.win_trade_num(), s))) if len(s) > 0 else 0,
            "trade": sum(list(map(lambda x: x.trade_num(), s))) if len(s) > 0 else 0,
            "position_size": round(numpy.average(position_size).item(), -2) if len(position_size) > 0 else 0,
            "max_position_size": max(position_size) if len(position_size) > 0 else 0,
            "position_term": round(numpy.average(position_term).item()) if len(position_term) > 0 else 0,
            "max_position_term": max(position_term) if len(position_term) > 0 else 0,
            "max_unavailable_assets": max(unavailable_assets) if len(s) > 0 and len(unavailable_assets) > 0 else 0,
            "sum_contract_price": sum(sum_contract_price) if len(s) > 0 else 0,
            "auto_stop_loss": sum(auto_stop_loss) if len(s) > 0 else 0,
        }

        return results


