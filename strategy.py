# -*- coding: utf-8 -*-
import re
import math
import numpy
import rules
import inspect
import utils
import simulator
import itertools
from loader import Loader
from collections import namedtuple
from simulator import SimulatorData
from argparse import ArgumentParser
import strategies.production as production
import strategies as develop

def add_options(parser):
    parser.add_argument("--position_sizing", action="store_true", default=False, dest="position_sizing", help="ポジションサイジング")
    parser.add_argument("--production", action="store_true", default=False, dest="production", help="本番向け") # 実行環境の選択
    parser.add_argument("--short", action="store_true", default=False, dest="short", help="空売り戦略")
    parser.add_argument("--tick", action="store_true", default=False, dest="tick", help="ティックデータを使う")
    parser.add_argument("--realtime", action="store_true", default=False, dest="realtime", help="リアルタイムデータを使う")
    parser.add_argument("--daytrade", action="store_true", default=False, dest="daytrade", help="デイトレ")
    parser.add_argument("--falling", action="store_true", default=False, dest="falling", help="下落相場")
    parser.add_argument("--new_high", action="store_true", default=False, dest="new_high", help="新高値")
    return parser

def create_parser():
    parser = ArgumentParser()
    return add_options(parser)

def get_prefix(args):
    prefix = "production_" if args.production else ""

    tick = "tick_" if args.tick else ""

    method = "short_" if args.short else ""

    target = ""
    if args.daytrade:
        target = "daytrade_"
    if args.falling:
        target = "falling_"
    if args.new_high:
        target = "new_high_"

    return "%s%s%s%s" % (prefix, target, tick, method)

def get_filename(args):
    prefix = get_prefix(args)
    filename = "%ssimulate_setting.json" % prefix
    return filename

def load_simulator_data(code, start_date, end_date, args):
    if args.realtime:
        days = (utils.to_datetime(end_date) - utils.to_datetime(start_date)).days
        data = Loader.loads_realtime(code, end_date, days+1)
        rule = "30T"
    elif args.tick:
        data = Loader.load_tick_ohlc(code, start_date, end_date)
        rule = "30T"
    else:
        data = Loader.load_with_realtime(code, start_date, end_date)
        rule = "W"

    if data is None:
        print("%s: %s is None" % (start_date, code))
        return None

    weekly = Loader.resample(data, rule=rule)

    try:
        data = utils.add_stats(data)
        data = utils.add_cs_stats(data)
        weekly = utils.add_stats(weekly)
        weekly = utils.add_cs_stats(weekly)
        return SimulatorData(code, data, weekly, rule)
    except Exception as e:
        print("load_error: %s" % e)
        return None

def load_strategy_setting(args):
    filename = get_filename(args)
    setting_dict = Loader.simulate_setting(filename)
    print(filename, setting_dict)
    if setting_dict is None:
        strategy_setting = StrategySetting()
    else:
        strategy_setting = create_setting_by_dict(setting_dict["setting"])
    return setting_dict, strategy_setting

def load_strategy(args, combination_setting=None):
    _, settings = load_strategy_setting(args)
    return load_strategy_creator(args, combination_setting).create(settings)

def load_strategy_creator(args, combination_setting=None):
    if args.production:
        if args.daytrade:
            from strategies.production.daytrade import CombinationStrategy
            return CombinationStrategy(combination_setting)
        elif args.falling:
            from strategies.production.falling import CombinationStrategy
            return CombinationStrategy(combination_setting)
        else:
            from strategies.production.combination import CombinationStrategy
            return CombinationStrategy(combination_setting)
    else:
        if args.daytrade:
            from strategies.daytrade import CombinationStrategy
            return CombinationStrategy(combination_setting)
        elif args.falling:
            from strategies.falling import CombinationStrategy
            return CombinationStrategy(combination_setting)
        else:
            from strategies.combination import CombinationStrategy
            return CombinationStrategy(combination_setting)

# ========================================================================
class StrategyCreator:
    def __init__(self, new=None, taking=None, stop_loss=None, closing=None):
        self.new = new
        self.taking = taking
        self.stop_loss = stop_loss
        self.closing = closing

    def create_new_rules(self, data, setting):
        return self.new

    def create_taking_rules(self, data, setting):
        return self.taking

    def create_stop_loss_rules(self, data, setting):
        return self.stop_loss

    def create_closing_rules(self, data, setting):
        return self.closing

    def create(self, setting):
        new_rules = [lambda x: self.create_new_rules(x, setting)]
        taking_rules = [lambda x: self.create_taking_rules(x, setting)]
        stop_loss_rules = [lambda x: self.create_stop_loss_rules(x, setting)]
        closing_rules = [lambda x: self.create_closing_rules(x, setting)]
        return Strategy(new_rules, taking_rules, stop_loss_rules, closing_rules)

    def ranges(self):
        return []

# 売買戦略
class Strategy:
    def __init__(self, new_rules, taking_rules, stop_loss_rules, closing_rules):
        self.new_rules = list(map(lambda x: rules.Rule(x), new_rules))
        self.taking_rules = list(map(lambda x: rules.Rule(x), taking_rules))
        self.stop_loss_rules = list(map(lambda x: rules.Rule(x), stop_loss_rules))
        self.closing_rules = list(map(lambda x: rules.Rule(x), closing_rules))

class StrategySetting():
    def __init__(self):
        self.new = 0
        self.taking = 0
        self.stop_loss = 0
        self.closing = 0

    def by_array(self, params):
        self.new = int(params[0])
        self.taking = int(params[1])
        self.stop_loss = int(params[2])
        self.closing = int(params[3])
        return self

class StrategyConditions():
    def __init__(self):
        self.new = []
        self.taking = []
        self.stop_loss = []
        self.closing = []

    def by_array(self, params):
        self.new = params[0]
        self.taking = params[1]
        self.stop_loss = params[2]
        self.closing = params[3]
        return self

def create_setting_by_dict(params):
     return namedtuple("StrategySetting", params.keys())(*params.values())

class StrategyUtil:
    def max_risk(self, data, setting):
        return data.assets * 0.02

    # 利益目標
    def goal(self, data, setting):
        order = data.position.num() # 現在の保有数
        if data.setting.short_trade:
            price = data.data["daily"]["low"].iloc[-1]
            line = data.data["daily"]["support"].iloc[-1]
            goal = (price - line)
        else:
            price = data.data["daily"]["high"].iloc[-1]
            line = data.data["daily"]["resistance"].iloc[-1]
            goal = (line - price)

        goal = 0 if goal < 0 else goal * (order + 100)
        return goal

    # 損失リスク
    def risk(self, data, setting):
        order = data.position.num() # 現在の保有数
        price = data.data["daily"]["close"].iloc[-1]

        if data.setting.short_trade:
            safety = data.data["daily"]["fall_safety"].iloc[-1]
            risk = (safety - price)
        else:
            safety = data.data["daily"]["rising_safety"].iloc[-1]
            risk = (price - safety)

        # riskがマイナスの場合、safetyを抜けているのでリスクが高い
        risk = 0 if risk < 0 else risk * (order + 100)
        return risk

    # 上限
    def upper(self, data, setting, term=1):
        upper = data.data["daily"]["resistance"].iloc[-term]

        return upper

    # 下限
    def lower(self, data, setting, term=1):
        lower= data.data["daily"]["support"].iloc[-term]

        return lower

    # 不安要素
    def falling(self, data, setting, risk):
        conditions = [
            risk == 0, # セーフティーを下回っている
        ]
        return any(conditions)

    # 最大ポジションサイズ
    def max_order(self, max_risk, risk):
        return int(max_risk / risk) * 100

    # ポジションサイズ
    def order(self, data, setting, max_risk, risk):
        current = data.position.num()
        print(max_risk, risk)
        max_order = int((self.max_order(max_risk, risk) - current)) # 保有できる最大まで
        max_order = int(max_order / 2) # 半分ずつ
        max_order = int(math.ceil(max_order / 100) * 100) # 端数を切り上げ
        order = 100 if self.falling(data, setting, risk) else max_order

        if order < 100:
            order = 0

        return order

    def safety(self, data, setting, term):
        return data.data["daily"]["rising_safety"].iloc[-term:].max() * 0.98

    def term(self, data, setting):
        return 1 if data.position.term()  == 0 else data.position.term()

    # 強気のダイバージェンス
    def rising_divergence(self, data, setting):
        conditions = [
            data.data["daily"]["weekly_average_trend"].iloc[-1] < 0,
            data.data["daily"]["daily_average_trend"].iloc[-1] >= 0,
            data.data["daily"]["macd_trend"].iloc[-1] > 0,
            data.data["daily"]["macdhist_trend"].iloc[-1] > 0,
        ]

        return [
            data.data["daily"]["weekly_average_trend"].iloc[-1] > 0 and data.data["daily"]["daily_average_trend"].iloc[-1] <= 0,
            data.data["daily"]["weekly_average_trend"].iloc[-1] >= 0 and data.data["daily"]["daily_average_trend"].iloc[-1] < 0,
            all(conditions),
        ]

## 指定可能な戦略====================================================================================================================

class CombinationSetting:
    simple = False
    position_sizing = False

class Combination(StrategyCreator, StrategyUtil):
    def __init__(self, conditions, common, setting=None):
        self.conditions = conditions
        self.common = common
        self.setting = CombinationSetting() if setting is None else setting

    def apply(self, data, setting, conditions):
        if len(conditions) == 0:
            return False
        a = list(map(lambda x: x(data, setting), conditions[0]))
        b = list(map(lambda x: x(data, setting), conditions[1]))
        return all(a) and any(b)

    def apply_common(self, data, setting, conditions):
        common = list(map(lambda x: x(data, setting), conditions))
        return all(common)

    # 買い
    def create_new_rules(self, data, setting):
        drawdown = list(map(lambda x: x["drawdown"], data.stats.drawdown[-20:]))
        drawdown_gradient = list(filter(lambda x: x > 0.06, numpy.gradient(drawdown))) if len(drawdown) > 1 else []
        drawdown_sum = list(filter(lambda x: x > 0, numpy.gradient(drawdown))) if len(drawdown) > 1 else []
        risk = self.risk(data, setting)
        max_risk = self.max_risk(data, setting)
        conditions = [
            len(drawdown_gradient) == 0, # 6%ルール条件外(-6%を超えて一定期間たった)
            sum(drawdown_sum) < 0.06, # 6%ルール(直近のドローダウン合計が6%以下)
            (data.position.num() < self.max_order(max_risk, risk)) if risk > 0 else False, # 最大ポジションサイズ以下
            self.apply_common(data, setting, self.common.new)
        ]

        order = self.order(data, setting, max_risk, risk) if self.setting.position_sizing else 100

        if self.apply(data, setting, self.conditions.new):
            if all(conditions) or self.setting.simple:
                return simulator.Order(order, [lambda x: True])

        return None

    # 利食い
    def create_taking_rules(self, data, setting):
        conditions = [
            self.apply_common(data, setting, self.common.taking),
            self.apply(data, setting, self.conditions.taking)
        ]
        return simulator.Order(data.position.num(), [lambda x: True]) if all(conditions) else None

    # 損切
    def create_stop_loss_rules(self, data, setting):
        conditions = [
            utils.rate(data.position.value(), data.data["daily"]["close"].iloc[-1]) < -0.02, # 損益が-2%
            self.apply_common(data, setting, self.common.stop_loss) and self.apply(data, setting, self.conditions.stop_loss),
        ]
        if any(conditions):
            return simulator.Order(data.position.num(), [lambda x: True])
        return None

    # 手仕舞い
    def create_closing_rules(self, data, setting):
        conditions = [
            self.apply_common(data, setting, self.common.closing) and self.apply(data, setting, self.conditions.closing),
        ]

        if any(conditions):
            return simulator.Order(data.position.num(), [lambda x: True])

        return None

class CombinationCreator(StrategyCreator, StrategyUtil):
    def __init__(self, setting=None):
        self.new_combinations = utils.combinations(self.new())
        self.taking_combinations = utils.combinations(self.taking())
        self.stop_loss_combinations = utils.combinations(self.stop_loss())
        self.closing_combinations = utils.combinations(self.closing())
        self.setting = setting

    def ranges(self):
        return [
            list(range(len(self.new_combinations))),
            list(range(len(self.taking_combinations))),
            list(range(len(self.stop_loss_combinations))),
            list(range(len(self.closing_combinations))),
        ]

    def create(self, setting):
        condition = [
            self.new_combinations[setting.new],
            self.taking_combinations[setting.taking],
            self.stop_loss_combinations[setting.stop_loss],
            self.closing_combinations[setting.closing],
        ]
        c = StrategyConditions().by_array(condition)
        return Combination(c, self.common(), self.setting).create(setting)

    def subject(self):
        raise Exception("Need override subject.")

    def default_common(self):
        rules = [lambda d, s: True]
        return StrategyCreator(rules, rules, rules, rules)

    # @return StrategyCreator
    def common(self):
        return self.default_common()

    def new(self):
        return [
            lambda d, s: False
        ]

    def taking(self):
        return [
            lambda d, s: False
        ]

    def stop_loss(self):
        return [
            lambda d, s: False
        ]

    def closing(self):
        return [
            lambda d, s: False
        ]

class CombinationChecker:
    def __init__(self, combination_strategy):
        self.new_combinations = utils.combinations(range(len(combination_strategy.new())))
        self.taking_combinations = utils.combinations(range(len(combination_strategy.taking())))
        self.stop_loss_combinations = utils.combinations(range(len(combination_strategy.stop_loss())))
        self.closing_combinations = utils.combinations(range(len(combination_strategy.closing())))

        self.combinations_dict = {
            "new": (self.new_combinations, combination_strategy.new),
            "taking": (self.taking_combinations, combination_strategy.taking),
            "closing": (self.closing_combinations, combination_strategy.closing),
            "stop_loss": (self.stop_loss_combinations, combination_strategy.stop_loss),
        }

    def get_strategy_source_by_index(self, index, method):
        source = inspect.getsource(method)
        pattern = r"d, s: .*[,$\n]"
        source = re.findall(pattern, source)
        return source[index].strip("\n")

    def get_strategy_source(self, combinations, method):
        source = inspect.getsource(method)
        pattern = r"d, s: .*[,$\n]"
        source = re.findall(pattern, source)
        a = numpy.array([source[i].strip("\n") for i in combinations[0]])
        b = numpy.array([source[i].strip("\n") for i in combinations[1]])
        return a, b

    def get_strategy_sources(self, setting):
        results = []
        for k, v in self.combinations_dict.items():
           combinations = v[0][setting["setting"][k]]
           a, b = self.get_strategy_source(combinations, v[1])
           results.append({"key":k, "all": a, "any": b, "combinations": combinations})
        return results


