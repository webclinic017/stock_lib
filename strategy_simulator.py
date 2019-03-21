import sys
import numpy

sys.path.append("lib")
import checker
import utils
from simulator import Simulator


class StrategySimulator:
    def __init__(self, simulator_setting, strategy_creator, verbose=False):
        self.simulator_setting = simulator_setting
        self.strategy_creator = strategy_creator
        self.verbose = verbose

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
        start = utils.to_datetime_by_term(start_date, tick=args.tick)
        end = utils.to_datetime_by_term(end_date, tick=args.tick) + utils.relativeterm(1, tick=True)
        for date in utils.daterange(start, end):
            codes = self.get_targets(args, codes, utils.to_format_by_term(date, args.tick))
            daterange = self.append_daterange(codes, date, daterange)
        validate_codes = codes
        return codes, validate_codes, daterange

    def get_targets(self, args, targets, date):
        if args.code is None:
            date = utils.to_format(utils.to_datetime_by_term(date, args.tick))
            targets = list(set(targets + self.strategy_creator.subject(date)))
        else:
            targets = [args.code]
        return targets

    def simulates(self, strategy_setting, data, start_date, end_date, verbose=False):
        if verbose:
            print("simulating %s %s" % (start_date, end_date))

        args = data["args"]
        tick = args.tick

        # この期間での対象銘柄
        stocks, _, _ = self.select_codes(args, start_date, end_date)
        index = data["index"]
        datas = {}
        for code in stocks:
            if code in data["data"].keys():
                datas[code] = data["data"][code]

        # シミュレーター準備
        simulators = {}
        self.simulator_setting.debug = verbose
        for code in datas.keys():
            self.simulator_setting.strategy["daily"] = self.strategy_creator.create(strategy_setting)
            simulators[code] = Simulator(self.simulator_setting)

        # 日付のリストを取得
        dates = []
        for d in datas.values():
            dates = list(set(dates + d.dates(start_date, end_date)))

        # 日付ごとにシミュレーション
        targets = []
        dates = sorted(dates, key=lambda x: utils.to_datetime_by_term(x, tick))
        for date in dates:
            # 休日はスキップ
            if not utils.is_weekday(utils.to_datetime_by_term(date, tick)):
                if verbose:
                    print("%s is not weekday" % date)
                continue

            if verbose:
                print("=== [%s] ===" % date)

            # この日の対象銘柄
            targets = self.get_targets(args, targets, date)

            # 保有銘柄を対象に追加
            for code, simulator in simulators.items() :
                if simulator.position().num() > 0:
                    targets.append(code)

            targets = list(set(targets))

            if verbose:
                print("targets: %s" % targets)

            for code in targets:
                # M&Aのチェックのために期間を区切ってデータを渡す(M&Aチェックが重いから)
                start = utils.to_format_by_term(utils.to_datetime_by_term(date, tick) - utils.relativeterm(args.validate_term, tick), tick)
                split_data = datas[code].split(start, date)
                if checker.manda(split_data.daily):
                    if verbose:
                        print("[%s] is manda" % code)
                    continue

                if len(datas[code].at(date)) > 0:
                    if verbose:
                        print("[%s]" % code)
                    simulators[code].simulate_by_date(date, datas[code], index)
                else:
                    if verbose:
                        print("[%s] is less data: %s" % (code, date))
        # 手仕舞い
        if len(dates) > 0:
            for code in datas.keys():
                split_data = datas[code].split(dates[0], dates[-1])
                if len(split_data.daily) == 0:
                    continue
                simulators[code].closing(split_data.daily["close"].iloc[-1])

        # 統計 ====================================
        stats = {}
        for code in datas.keys():
            s = simulators[code].get_stats()
            keys = ["return", "drawdown", "win_trade", "trade", "assets", "trade_history"]
            result = {}
            for k in keys:
                result[k] = s[k]
            stats[code] = result

        # 統計 =======================================
        wins = list(filter(lambda x: x[1]["return"] > 0, stats.items()))
        lose = list(filter(lambda x: x[1]["return"] < 0, stats.items()))
        win_codes = list(map(lambda x: x[0], wins))
        lose_codes = list(map(lambda x: x[0], lose))
        codes = win_codes + lose_codes
        gain = list(map(lambda x: x[1]["assets"] - self.simulator_setting.assets, stats.items()))
        trade_history = list(map(lambda x: x[1]["trade_history"], stats.items()))
        position_size = list(map(lambda x: list(map(lambda y: y["size"], x)), trade_history))
        position_size = list(filter(lambda x: x != 0, sum(position_size, [])))
        position_term = list(map(lambda x: list(map(lambda y: y["term"], x)), trade_history))
        position_term = list(filter(lambda x: x != 0, sum(position_term, [])))

        if verbose:
            print(start_date, end_date, "assets:", self.simulator_setting.assets, "gain:", gain, sum(gain))
            for code, s in sorted(stats.items(), key=lambda x: x[1]["return"]):
                print("[%s] return: %s, drawdown: %s, trade: %s, win: %s" % (code, s["return"], s["drawdown"], s["trade"], s["win_trade"]))

        s = stats.values()
        results = {
            "codes": codes,
            "win": win_codes,
            "lose": lose_codes,
            "gain": sum(gain),
            "return": sum(gain) / self.simulator_setting.assets,
            "drawdown": numpy.average(list(map(lambda x: x["drawdown"], s))) if len(s) > 0 else 0,
            "max_drawdown": max(list(map(lambda x: x["drawdown"], s))) if len(s) > 0 else 0,
            "win_trade": sum(list(map(lambda x: x["win_trade"], s))) if len(s) > 0 else 0,
            "trade": sum(list(map(lambda x: x["trade"], s))) if len(s) > 0 else 0,
            "position_size": numpy.average(position_size) if len(position_size) > 0 else 0,
            "max_position_size": max(position_size) if len(position_size) > 0 else 0,
            "position_term": numpy.average(position_term) if len(position_term) > 0 else 0,
            "max_position_term": max(position_term) if len(position_term) > 0 else 0,
        }

        return results


