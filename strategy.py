# -*- coding: utf-8 -*-
import re
import math
import numpy
import inspect
import utils
import simulator
import itertools
import time as t
import glob
from loader import Loader
from loader import Bitcoin
from collections import namedtuple
from simulator import SimulatorData
from simulator import SimulatorSetting
from argparse import ArgumentParser

def add_options(parser):
    parser.add_argument("--code", type=str, action="store", default=None, dest="code", help="code")
    parser.add_argument("--use_limit", action="store_true", default=False, dest="use_limit", help="指値を使う")
    parser.add_argument("--position_sizing", action="store_true", default=False, dest="position_sizing", help="ポジションサイジング")
    parser.add_argument("--max_position_size", action="store", default=None, dest="max_position_size", help="最大ポジションサイズ")
    parser.add_argument("--production", action="store_true", default=False, dest="production", help="本番向け") # 実行環境の選択
    parser.add_argument("--short", action="store_true", default=False, dest="short", help="空売り戦略")
    parser.add_argument("--daytrade", action="store_true", default=False, dest="daytrade", help="ティックデータを使う")
    parser.add_argument("--realtime", action="store_true", default=False, dest="realtime", help="リアルタイムデータを使う")
    parser.add_argument("--stop_loss_rate", action="store", default=None, dest="stop_loss_rate", help="損切レート")
    parser.add_argument("--taking_rate", action="store", default=None, dest="taking_rate", help="利食いレート")
    parser.add_argument("--min_unit", action="store", default=None, dest="min_unit", help="最低単元")
    parser.add_argument("--rule", action="store", default=None, dest="rule", help="足の単位")
    parser.add_argument("--trade_step", type=int, action="store", default=None, dest="trade_step", help="注文の間隔")
    parser.add_argument("--ensemble_dir", action="store", default=None, dest="ensemble_dir", help="アンサンブルディレクトリ")

    # strategy
    parser.add_argument("--ensemble", action="store_true", default=False, dest="ensemble", help="アンサンブル")
    return parser

def create_parser():
    parser = ArgumentParser()
    return add_options(parser)

def get_prefix(args, ignore_code=False):
    code = "" if args.code is None or ignore_code else "%s_" % args.code

    prefix = "production_" if args.production else ""

    daytrade = "daytrade_" if args.daytrade else ""

    method = "short_" if args.short else ""

    target = get_strategy_name(args)
    target = "" if target == "combination" else "%s_" % target

    return "%s%s%s%s%s" % (prefix, code, target, daytrade, method)

def get_filename(args, ignore_code=False):
    prefix = get_prefix(args, ignore_code=ignore_code)
    filename = "%ssimulate_setting.json" % prefix
    return filename

class StrategyType:
    ENSEMBLE="ensemble"
    COMBINATION="combination"

    def list(self):
        return [
            self.ENSEMBLE,
            self.COMBINATION
        ]

def get_strategy_name(args):
    strategy_types = StrategyType()
    if args.ensemble:
        return strategy_types.ENSEMBLE
    else:
        return strategy_types.COMBINATION

# load ================================================

def load_strategy_creator_by_type(strategy_type, is_production, combination_setting, ignore_ensemble=False):
    strategy_types = StrategyType()
    if is_production:
        if strategy_types.ENSEMBLE == strategy_type and not ignore_ensemble:
            from strategies.production.ensemble import CombinationStrategy
            return CombinationStrategy(combination_setting)
        else:
            from strategies.production.combination import CombinationStrategy
            return CombinationStrategy(combination_setting)
    else:
        if strategy_types.ENSEMBLE == strategy_type and not ignore_ensemble:
            from strategies.ensemble import CombinationStrategy
            return CombinationStrategy(combination_setting)
        else:
            from strategies.combination import CombinationStrategy
            return CombinationStrategy(combination_setting)

def load_strategy_creator(args, combination_setting=None):
    combination_setting = CombinationSetting() if combination_setting is None else combination_setting
    strategy_type = get_strategy_name(args)
    return load_strategy_creator_by_type(strategy_type, args.production, combination_setting)

def load_strategy_setting(args):
    filename = get_filename(args)

    setting_dict, strategy_setting = load_strategy_setting_by_filename(filename)

    # 個別銘柄の設定がなければ共通の設定を読む
    if args.code is not None and setting_dict is None:
        filename = get_filename(args, ignore_code=True)
        setting_dict = Loader.simulate_setting(filename)

    return setting_dict, strategy_setting

def load_strategy_creator_by_setting(create_setting, ignore_ensemble=False):
    return load_strategy_creator_by_type(
        create_setting.strategy_type,
        create_setting.is_production,
        create_setting.combination_setting,
        ignore_ensemble)

def load_strategy_setting_by_filename(filename):
    setting_dict = Loader.simulate_setting(filename)

    strategy_setting = create_strategy_setting(setting_dict)

    return setting_dict, strategy_setting

def load_strategy(args, combination_setting=None):
    _, settings = load_strategy_setting(args)
    return load_strategy_creator(args, combination_setting).create(settings)

def load_simulator_data(code, start_date, end_date, args, time=None):
    if args.realtime:
        rule = "5T" if args.rule is None else args.rule
        start = utils.to_datetime(start_date) - utils.relativeterm(3, True)
        days = (utils.to_datetime(end_date) - start).days
        for i in range(5):
            data = Loader.loads_realtime(code, end_date, days+i, time=time, rule=rule)
            if len(data) >= 250: # weekleyのstats生成で必要な分
                break
    elif args.daytrade:
        rule = "5T" if args.rule is None else args.rule
        start = utils.to_format(utils.to_datetime(start_date) - utils.relativeterm(1, True))
        data = Loader.load_tick_ohlc(code, start, end_date, time=time, rule=rule)
    else:
        rule = "D"
        start = utils.to_format(utils.to_datetime(start_date) - utils.relativeterm(6))
        data = Loader.load_with_realtime(code, start, end_date)

    if data is None:
        print("%s: %s is None" % (start_date, code))
        return None

    simulator_data = add_stats(code, data, rule)
    print("loaded:", utils.timestamp(), code, data["date"].iloc[0], data["date"].iloc[-1])
    return simulator_data

def add_stats(code, data, rule):
    try:
        data = utils.add_stats(data)
        data = utils.add_cs_stats(data)
        return SimulatorData(code, data, rule)
    except Exception as e:
        print("load_error: %s" % e)
        return None

# create ================================================

def create_strategy_setting(setting_dict):
    if setting_dict is None:
        strategy_setting = []
    else:
        strategy_setting = list(map(lambda x: create_setting_by_dict(x), setting_dict["setting"]))
    return strategy_setting

def create_ensemble_strategies(files):
    settings = list(map(lambda x: StrategyCreateSetting(x), files))
    ensembles = list(map(lambda x: load_strategy_creator_by_setting(x, ignore_ensemble=True).create(x.strategy_setting), settings))
    return ensembles

def ensemble_files(directory):
    files = glob.glob("%s/*" % directory)
    return files

# args > json > default の優先度
def create_combination_setting(args, use_json=True):
    combination_setting = create_combination_setting_by_json(args) if use_json else CombinationSetting()
    combination_setting.use_limit = args.use_limit if args.use_limit else combination_setting.use_limit
    combination_setting.position_sizing = args.position_sizing if args.position_sizing else combination_setting.position_sizing
    combination_setting.max_position_size = combination_setting.max_position_size if args.max_position_size is None else int(args.max_position_size)
    combination_setting.ensemble = [] if args.ensemble_dir is None else ensemble_files(args.ensemble_dir)
    return combination_setting

def create_combination_setting_by_json(args):
    setting_dict, _ = load_strategy_setting(args)
    return create_combination_setting_by_dict(setting_dict)

def create_combination_setting_by_dict(setting_dict):
    combination_setting = CombinationSetting()
    if setting_dict is None:
        return combination_setting
    combination_setting.use_limit = setting_dict["use_limit"] if "use_limit" in setting_dict.keys() else combination_setting.use_limit
    combination_setting.position_sizing = setting_dict["position_sizing"] if "position_sizing" in setting_dict.keys() else combination_setting.position_sizing
    combination_setting.max_position_size = setting_dict["max_position_size"] if "max_position_size" in setting_dict.keys() else combination_setting.max_position_size
    combination_setting.seed = setting_dict["seed"] if "seed" in setting_dict.keys() else combination_setting.seed
    combination_setting.ensemble = ensemble_files(setting_dict["ensemble"]) if "ensemble" in setting_dict.keys() else combination_setting.ensemble
    return combination_setting

def create_simulator_setting(args, use_json=True):
    simulator_setting = create_simulator_setting_by_json(args) if use_json else SimulatorSetting()
    simulator_setting.stop_loss_rate = simulator_setting.stop_loss_rate if args.stop_loss_rate is None else float(args.stop_loss_rate)
    simulator_setting.taking_rate = simulator_setting.taking_rate if args.taking_rate is None else float(args.taking_rate)
    simulator_setting.min_unit = simulator_setting.min_unit if args.min_unit is None else float(args.min_unit)
    simulator_setting.short_trade = args.short
    simulator_setting.trade_step = simulator_setting.trade_step if args.trade_step is None else args.trade_step
    simulator_setting.use_before_stick = args.daytrade
    return simulator_setting

def create_simulator_setting_by_json(args):
    simulator_setting = SimulatorSetting()
    setting_dict, _ = load_strategy_setting(args)
    if setting_dict is None:
        return simulator_setting
    simulator_setting.stop_loss_rate = setting_dict["stop_loss_rate"] if "stop_loss_rate" in setting_dict.keys() else simulator_setting.stop_loss_rate
    simulator_setting.taking_rate = setting_dict["taking_rate"] if "taking_rate" in setting_dict.keys() else simulator_setting.taking_rate
    simulator_setting.min_unit = setting_dict["min_unit"] if "min_unit" in setting_dict.keys() else simulator_setting.min_unit
    simulator_setting.short_trade = args.short
    return simulator_setting

# ========================================================================
# 売買ルール
class Rule:
    def __init__(self, callback):
        self.callback = callback

    def apply(self, data):
        return self.callback(data)

# 売買戦略
class Strategy:
    def __init__(self, new_rules, taking_rules, stop_loss_rules, closing_rules):
        self.new_rules = list(map(lambda x: Rule(x), new_rules))
        self.taking_rules = list(map(lambda x: Rule(x), taking_rules))
        self.stop_loss_rules = list(map(lambda x: Rule(x), stop_loss_rules))
        self.closing_rules = list(map(lambda x: Rule(x), closing_rules))

def create_setting_by_dict(params):
     return StrategySetting().by_array([
        params["new"],
        params["taking"],
        params["stop_loss"],
        params["closing"]
     ])

def strategy_setting_to_dict(strategy_setting):
    return {"new": strategy_setting.new, "taking": strategy_setting.taking, "stop_loss": strategy_setting.stop_loss, "closing": strategy_setting.closing}

def strategy_setting_to_array(strategy_setting):
        return [
            strategy_setting.new,
            strategy_setting.taking,
            strategy_setting.stop_loss,
            strategy_setting.closing
        ]

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


    def to_dict(self):
        return strategy_setting_to_dict(self)

class StrategyCreateSetting:
    def __init__(self, filename):
        self.strategy_type = self.get_strategy_name(filename)
        self.is_production = "production" in filename
        self.setting_dict = Loader.simulate_setting(filename, "")
        self.strategy_setting = create_strategy_setting(self.setting_dict)
        self.combination_setting = create_combination_setting_by_dict(self.setting_dict)

    def get_strategy_name(self, filename):
        strategy_types = StrategyType()
        for t in strategy_types.list():
            if t in filename:
                return t
        return strategy_types.COMBINATION

class StrategyCreator:
    def __init__(self, new=None, taking=None, stop_loss=None, closing=None):
        self.new = new
        self.taking = taking
        self.stop_loss = stop_loss
        self.closing = closing

    def create_new_rules(self, data):
        return simulator.Order(0, [lambda x: True]) if self.new is None else self.new(data)

    def create_taking_rules(self, data):
        return simulator.Order(0, [lambda x: True]) if self.taking is None else self.taking(data)

    def create_stop_loss_rules(self, data):
        return simulator.Order(0, [lambda x: True]) if self.stop_loss is None else self.stop_loss(data)

    def create_closing_rules(self, data):
        return simulator.Order(0, [lambda x: True]) if self.closing is None else self.closing(data)

    def create(self):
        new_rules = [lambda x: self.create_new_rules(x)]
        taking_rules = [lambda x: self.create_taking_rules(x)]
        stop_loss_rules = [lambda x: self.create_stop_loss_rules(x)]
        closing_rules = [lambda x: self.create_closing_rules(x)]
        return Strategy(new_rules, taking_rules, stop_loss_rules, closing_rules)

    def ranges(self):
        return []

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


class StrategyUtil:
    def apply(self, data, conditions):
        if len(conditions) == 0:
            return False
        a = list(map(lambda x: x(data), conditions[0]))
        b = list(map(lambda x: x(data), conditions[1]))
        return all(a) and any(b)

    def apply_common(self, data, conditions):
        common = list(map(lambda x: x(data), conditions))
        return all(common)

    def price(self, data):
        return data.data.daily["close"].iloc[-1]

    # ドローダウン
    def drawdown(self, data):
        term = self.term(data)
        price = self.price(data)
        gain = data.position.gain(price)
        max_gain = self.max_gain(data)
        return max_gain - gain

    def max_gain(self, data):
        if data.setting.short_trade:
            max_gain = data.position.gain(data.data.daily["low"].min())
        else:
            max_gain = data.position.gain(data.data.daily["high"].max())
        return max_gain

    def take_gain(self, data):
        return data.assets * data.setting.taking_rate

    # 最大許容損失
    def max_risk(self, data):
        return data.assets * data.setting.stop_loss_rate

    # 利益目標
    def goal(self, data):
        order = data.position.get_num() # 現在の保有数
        if data.setting.short_trade:
            price = data.data.daily["low"].iloc[-1]
            line = data.data.daily["support"].iloc[-1]
            goal = (price - line)
        else:
            price = data.data.daily["high"].iloc[-1]
            line = data.data.daily["resistance"].iloc[-1]
            goal = (line - price)

        goal = 0 if goal < 0 else goal * (order + 1)
        return goal

    # 損失リスク
    def risk(self, data):
        order = data.position.get_num() # 現在の保有数
        price = self.price(data)
        safety = self.safety(data, 1)

        if data.setting.short_trade:
            risk = safety - price
        else:
            risk = price - safety

        # riskがマイナスの場合、safetyを抜けているのでリスクが高い
        if risk < 0:
            return 0

        risk = risk * (order + 1)
        return risk

    # 上限
    def upper(self, data, term=1):
        upper = data.data.daily["resistance"].iloc[-term]

        return upper

    # 下限
    def lower(self, data, term=1):
        lower= data.data.daily["support"].iloc[-term]

        return lower

    # 不安要素
    def caution(self, data):
        gain = data.stats.gain()
        conditions = [
            self.risk(data) == 0, # セーフティーを下回っている
            data.position.get_num() == 0, # 初回の仕掛け
            data.position.gain(self.price(data)) < 0, # 損益がマイナス
            gain[-1] < 0 if len(gain) > 0 else False, # 最後のトレードで損失
        ]
        return any(conditions)

    # 注目要素
    def attention(self, data):
        gain = data.stats.gain()
        conditions = [
            gain[-1] > 0 if len(gain) > 0 else False, # 最後のトレードで利益
            data.position.gain(self.price(data)) > 0, # 損益がプラス
        ]
        return any(conditions)

    # 最大ポジションサイズ
    def max_order(self, data, max_risk, risk):
        if risk == 0:
            return 0
        max_order = int((max_risk / risk) / data.setting.min_unit)
        return max_order

    # ポジションサイズ
    def order(self, data, max_risk, risk, max_position_size):
        current = data.position.get_num()
        max_order = self.max_order(data, max_risk, risk)
        max_order = max_order if max_order < max_position_size else max_position_size
        max_order = max_order - current # 保有できる最大まで
        max_order = max_order if self.attention(data) else max_order / 2 # 最後負けトレードなら半分ずつ
        order = 1 if self.caution(data) else int(max_order) # 不安要素があれば、最小単位から

        if order < 1:
            order = 0

        return order

    def safety(self, data, term):
        if data.setting.short_trade:
            return data.data.daily["fall_safety"].iloc[-term]
        else:
            return data.data.daily["rising_safety"].iloc[-term]

    def term(self, data):
        return 1 if data.position.get_term()  == 0 else data.position.get_term()

    # 保有数最大で最大の損切ラインを適用
    def stop_loss_rate(self, data, max_position_size):
        position_rate = data.position.get_num() / max_position_size
        return data.setting.stop_loss_rate * position_rate

## 指定可能な戦略====================================================================================================================

class CombinationSetting:
    simple = False
    use_limit = False
    position_sizing = False
    max_position_size = 5
    condition_size = 5
    seed = [t.time()]
    ensemble = []

class Combination(StrategyCreator, StrategyUtil):
    def __init__(self, conditions, common, setting=None):
        self.conditions = conditions
        self.common = common
        self.setting = CombinationSetting() if setting is None else setting

    def drawdown_allowable(self, data):
        allowable_dd = data.setting.stop_loss_rate * 3
        drawdown = data.stats.drawdown()[-20:]
        drawdown_diff = list(filter(lambda x: x > allowable_dd, numpy.diff(drawdown))) if len(drawdown) > 1 else []
        drawdown_sum = list(filter(lambda x: x > 0, numpy.diff(drawdown))) if len(drawdown) > 1 else []
        drawdown_conditions = [
            len(drawdown_diff) == 0, # 6%ルール条件外(-6%を超えて一定期間たった)
            sum(drawdown_sum) < allowable_dd # 6%ルール(直近のドローダウン合計が6%以下)
        ]

        allow = all(drawdown_conditions)

        if not allow and data.setting.debug:
            print("over drawdown: ", drawdown_conditions, len(drawdown_diff), sum(drawdown_sum))

        return allow

    # 買い
    def create_new_rules(self, data):
        risk = self.risk(data)
        max_risk = self.max_risk(data)

        max_order = self.max_order(data, max_risk, risk)
        max_order = max_order if max_order < self.setting.max_position_size else self.setting.max_position_size

        conditions = [
            self.drawdown_allowable(data), # ドローダウンが問題ない状態
            data.position.get_num() < max_order, # 最大ポジションサイズ以下
            self.apply_common(data, self.common.new)
        ]

        order = self.order(data, max_risk, risk, self.setting.max_position_size) if self.setting.position_sizing else 1

        if not self.setting.simple:
            conditions = conditions + [self.apply(data, self.conditions.new)]

        if all(conditions):
            if self.setting.use_limit:
                return simulator.LimitOrder(order, self.price(data))
            else:
                return simulator.MarketOrder(order)

        return None

    # 利食い
    def create_taking_rules(self, data):
        if self.setting.simple:
            conditions = [self.apply_common(data, self.common.taking)]
        else:
            conditions = [
                self.apply_common(data, self.common.taking),
                self.apply(data, self.conditions.taking)
            ]
        if all(conditions):
            order = data.position.get_num()
            if self.setting.use_limit:
                return simulator.LimitOrder(order, self.price(data), is_repay=True)
            else:
                return simulator.MarketOrder(order)
        return None

    # 損切
    def create_stop_loss_rules(self, data):
        if self.setting.simple:
            conditions = [self.apply_common(data, self.common.stop_loss)]
        else:
            conditions = [
                data.position.gain_rate(data.data.daily["close"].iloc[-1]) < -self.stop_loss_rate(data, self.setting.max_position_size), # 保有数最大で最大の損切ラインを適用
                self.apply_common(data, self.common.stop_loss) and self.apply(data, self.conditions.stop_loss),
            ]
        if any(conditions):
            order = data.position.get_num()
            if self.setting.use_limit:
                return simulator.ReverseLimitOrder(order, self.price(data), is_repay=True)
            else:
                return simulator.MarketOrder(order)
        return None

    # 手仕舞い
    def create_closing_rules(self, data):
        if self.setting.simple:
            conditions = [self.apply_common(data, self.common.closing)]
        else:
            conditions = [
                self.apply_common(data, self.common.closing) and self.apply(data, self.conditions.closing),
            ]
        if any(conditions):
            order = data.position.get_num()
            return simulator.MarketOrder(order)

        return None

class CombinationCreator(StrategyCreator, StrategyUtil):
    def __init__(self, setting=None):
        self.setting = CombinationSetting() if setting is None else setting

    def ranges(self):
        return [
            list(range(utils.combinations_size(self.new()))),
            list(range(utils.combinations_size(self.taking()))),
            list(range(utils.combinations_size(self.stop_loss()))),
            list(range(utils.combinations_size(self.closing()))),
        ]

    def create(self, settings):
        condition = self.conditions(settings[0])
        c = StrategyConditions().by_array(condition)
        return Combination(c, self.common(settings), self.setting).create()

    # インデックスから直接条件を生成
    def conditions(self, setting):
        return [
            utils.combination(setting.new, self.new()),
            utils.combination(setting.taking, self.taking()),
            utils.combination(setting.stop_loss, self.stop_loss()),
            utils.combination(setting.closing, self.closing()),
        ]

    def conditions_by_index(self, conditions, index):
        a = [conditions[i] for i in index[0]]
        b = [conditions[i] for i in index[1]]
        return [a, b]


    def subject(self):
        raise Exception("Need override subject.")

    # 何か追加データが欲しいときはoverrideする
    def add_data(self, data):
        return data

    def default_common(self):
        rules = [lambda d: True]
        return StrategyCreator(new=rules, taking=rules, stop_loss=rules, closing=[lambda d: False])

    # @return StrategyCreator
    def common(self, setting):
        return self.default_common()

    def new(self):
        return [
            lambda d: False
        ]

    def taking(self):
        return [
            lambda d: False
        ]

    def stop_loss(self):
        return [
            lambda d: False
        ]

    def closing(self):
        return [
            lambda d: False
        ]

class CombinationChecker:

    def get_replaced_source(self, condition):
        source = inspect.getsource(condition)
        closure_vars = inspect.getclosurevars(condition)

        for name, value in closure_vars.nonlocals.items():
            source = source.replace(name, "\"%s\"" % str(value))

        return source

    def get_source(self, conditions):
        sources = list(map(lambda x: self.get_replaced_source(x), conditions))
        sources = list(map(lambda x: x.strip("\n"), sources))
        sources = list(map(lambda x: x.strip(","), sources))
        sources = list(map(lambda x: x.strip(), sources))
        return sources

    def get_strategy_sources(self, combination_strategy, setting):
        new, taking, stop_loss = [], [], []
        for i, seed in enumerate(setting["seed"]):
            combination_strategy.conditions_by_seed(seed)
            s = setting["setting"][i]
            new         = new       + [utils.combination(s["new"], combination_strategy.new_conditions)]
            taking      = taking    + [utils.combination(s["taking"], combination_strategy.taking_conditions)]
            stop_loss   = stop_loss + [utils.combination(s["stop_loss"], combination_strategy.stop_loss_conditions)]

        conditions = {
            "new": new,
            "taking": taking,
            "stop_loss": stop_loss
        }

        results = {}

        for name, condition in conditions.items():
            sources = {"all": [], "any": []}
            for a, b in condition:
                sources["all"] = self.get_source(a)
                sources["any"] = self.get_source(b)
            results[name] = sources

        return results

