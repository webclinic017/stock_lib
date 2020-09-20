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
import pandas
from loader import Loader
from loader import Bitcoin
from collections import namedtuple
from simulator import SimulatorData
from simulator import SimulatorIndexData
from simulator import SimulatorSetting
from argparse import ArgumentParser

def add_options(parser):
    parser.add_argument("--code", type=str, action="store", default=None, dest="code", help="code")
    parser.add_argument("--use_limit", action="store_true", default=False, dest="use_limit", help="指値を使う")
    parser.add_argument("--position_sizing", action="store_true", default=False, dest="position_sizing", help="ポジションサイジング")
    parser.add_argument("--max_position_size", action="store", default=None, dest="max_position_size", help="最大ポジションサイズ")
    parser.add_argument("--production", action="store_true", default=False, dest="production", help="本番向け") # 実行環境の選択
    parser.add_argument("--short", action="store_true", default=False, dest="short", help="空売り戦略")
    parser.add_argument("--long_short", action="store_true", default=False, dest="long_short", help="ロングショート戦略")
    parser.add_argument("--auto_stop_loss", type=float, action="store", default=None, dest="auto_stop_loss", help="自動損切")
    parser.add_argument("--stop_loss_rate", action="store", default=None, dest="stop_loss_rate", help="損切レート")
    parser.add_argument("--taking_rate", action="store", default=None, dest="taking_rate", help="利食いレート")
    parser.add_argument("--min_unit", action="store", default=None, dest="min_unit", help="最低単元")
    parser.add_argument("--assets", type=int, action="store", default=None, dest="assets", help="assets")

    # strategy
    parser.add_argument("--ensemble_dir", action="store", default=None, dest="ensemble_dir", help="アンサンブルディレクトリ")
    parser.add_argument("--ensemble", action="store_true", default=False, dest="ensemble", help="アンサンブル")
    parser.add_argument("--open_close", action="store_true", default=False, dest="open_close", help="寄せ引け")
    parser.add_argument("--futures", action="store_true", default=False, dest="futures", help="先物")
    parser.add_argument("--new_high", action="store_true", default=False, dest="new_high", help="新高値")
    return parser

def create_parser():
    parser = ArgumentParser()
    return add_options(parser)

def get_prefix(args, ignore_code=False):
    return create_prefix(args, args.production, args.short, ignore_code)

def create_prefix(args, is_production, is_short, ignore_code=False):
    code = "" if args.code is None or ignore_code else "%s_" % args.code

    prefix = "production_" if is_production else ""

    method = "short_" if is_short else ""

    target = get_strategy_name(args)
    target = "" if target == "combination" else "%s_" % target

    return "%s%s%s%s" % (prefix, code, target, method)

def get_filename(args, ignore_code=False):
    prefix = get_prefix(args, ignore_code=ignore_code)
    filename = create_filename(prefix)
    return filename

def create_filename(prefix):
    return "%ssimulate_setting.json" % prefix

class StrategyType:
    ENSEMBLE="ensemble"
    COMBINATION="combination"
    OPEN_CLOSE="open_close"
    FUTURES="futures"
    NEW_HIGH="new_high"

    def list(self):
        return [
            self.ENSEMBLE,
            self.COMBINATION,
            self.OPEN_CLOSE,
            self.FUTURES,
            self.NEW_HIGH
        ]

def get_strategy_name(args):
    strategy_types = StrategyType()
    if args.ensemble:
        return strategy_types.ENSEMBLE
    elif args.open_close:
        return strategy_types.OPEN_CLOSE
    elif args.futures:
        return strategy_types.FUTURES
    elif args.new_high:
        return strategy_types.NEW_HIGH
    else:
        return strategy_types.COMBINATION

# load ================================================

def load_strategy_creator_by_type(strategy_type, is_production, combination_setting, ignore_ensemble=False):
    strategy_types = StrategyType()
    if is_production:
        if strategy_types.ENSEMBLE == strategy_type and not ignore_ensemble:
            from strategies.production.ensemble import CombinationStrategy
            return CombinationStrategy(combination_setting)
        elif strategy_types.NEW_HIGH == strategy_type:
            from strategies.production.new_high import CombinationStrategy
            return CombinationStrategy(combination_setting)
        elif strategy_types.FUTURES == strategy_type:
            from strategies.production.futures import CombinationStrategy
            return CombinationStrategy(combination_setting)
        else:
            from strategies.production.combination import CombinationStrategy
            return CombinationStrategy(combination_setting)
    else:
        if strategy_types.ENSEMBLE == strategy_type and not ignore_ensemble:
            from strategies.ensemble import CombinationStrategy
            return CombinationStrategy(combination_setting)
        elif strategy_types.OPEN_CLOSE == strategy_type:
            from strategies.open_close import CombinationStrategy
            return CombinationStrategy(combination_setting)
        elif strategy_types.FUTURES == strategy_type:
            from strategies.futures import CombinationStrategy
            return CombinationStrategy(combination_setting)
        elif strategy_types.NEW_HIGH == strategy_type:
            from strategies.new_high import CombinationStrategy
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

def load_strategy_by_option(args, is_short):
    filename = create_filename(create_prefix(args, is_production=args.production, is_short=is_short))
    setting_dict, settings = load_strategy_setting_by_filename(filename)
    combination_setting = create_combination_setting_by_dict(args, setting_dict)
    return load_strategy_creator(args, combination_setting).create(settings)

def load_simulator_data(code, start_date, end_date, args):
    rule = "D"
    start = utils.to_format(utils.to_datetime(start_date) - utils.relativeterm(6))
    data = Loader.load_with_realtime(code, start, end_date)

    if data is None:
        print("%s: %s is None" % (start_date, code))
        return None

    simulator_data = add_stats(code, data, rule)
    print("loaded:", utils.timestamp(), code, data["date"].iloc[0], data["date"].iloc[-1])
    return simulator_data

def load_index(args, start_date, end_date):
    index = {}
    start = utils.to_format(utils.to_datetime(start_date) - utils.relativeterm(6))

    for k in ["nikkei", "dow"]:
        d = Loader.load_index(k, start, end_date, with_filter=True, strict=False)
        d = add_stats(k, d, "D")
#        d = utils.add_stats(d)
#        d = utils.add_cs_stats(d)
        index[k] = d

    index["new_score"] = SimulatorData("new_score", Loader.new_score(), "D")

    return SimulatorIndexData(index)

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
        strategy_setting = list(map(lambda x: StrategySetting().by_dict(x), setting_dict["setting"]))
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
    combination_setting = create_combination_setting_by_json(args) if use_json else apply_assets(args, CombinationSetting())
    combination_setting.use_limit = args.use_limit if args.use_limit else combination_setting.use_limit
    combination_setting.position_sizing = args.position_sizing if args.position_sizing else combination_setting.position_sizing
    combination_setting.max_position_size = combination_setting.max_position_size if args.max_position_size is None else int(args.max_position_size)
    combination_setting.ensemble = [] if args.ensemble_dir is None else ensemble_files(args.ensemble_dir)
    return combination_setting

def create_combination_setting_by_json(args):
    setting_dict, _ = load_strategy_setting(args)
    return create_combination_setting_by_dict(args, setting_dict)

def create_combination_setting_by_dict(args, setting_dict):
    combination_setting = CombinationSetting()
    if setting_dict is None:
        return combination_setting
    combination_setting.use_limit = setting_dict["use_limit"] if "use_limit" in setting_dict.keys() else combination_setting.use_limit
    combination_setting.position_sizing = setting_dict["position_sizing"] if "position_sizing" in setting_dict.keys() else combination_setting.position_sizing
    combination_setting.max_position_size = setting_dict["max_position_size"] if "max_position_size" in setting_dict.keys() else combination_setting.max_position_size
    combination_setting.seed = setting_dict["seed"] if "seed" in setting_dict.keys() else combination_setting.seed
    combination_setting.ensemble = ensemble_files(setting_dict["ensemble"]) if "ensemble" in setting_dict.keys() else combination_setting.ensemble
    combination_setting.weights = setting_dict["weights"] if "weights" in setting_dict.keys() else combination_setting.weights
    combination_setting = apply_assets(args, combination_setting)
    return combination_setting

def create_simulator_setting(args, use_json=True):
    simulator_setting = create_simulator_setting_by_json(args) if use_json else SimulatorSetting()
    simulator_setting.stop_loss_rate = simulator_setting.stop_loss_rate if args.stop_loss_rate is None else float(args.stop_loss_rate)
    simulator_setting.taking_rate = simulator_setting.taking_rate if args.taking_rate is None else float(args.taking_rate)
    simulator_setting.min_unit = simulator_setting.min_unit if args.min_unit is None else int(args.min_unit)
    simulator_setting.short_trade = args.short
    simulator_setting.auto_stop_loss = simulator_setting.auto_stop_loss if args.auto_stop_loss is None else args.auto_stop_loss
    simulator_setting.ignore_volume = args.futures
    simulator_setting = apply_assets(args, simulator_setting)
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
    simulator_setting.auto_stop_loss = setting_dict["auto_stop_loss"] if "auto_stop_loss" in setting_dict.keys() else simulator_setting.auto_stop_loss
    simulator_setting.ignore_volume = args.futures
    simulator_setting = apply_assets(args, simulator_setting)
    return simulator_setting

def apply_assets(args, setting):
    assets = Loader.assets()
    setting.assets = assets["assets"] if args.assets is None else args.assets
    return setting

def apply_long_short(args, setting):
    setting.long_short_trade = {"long": load_strategy_by_option(args, is_short=False), "short": load_strategy_by_option(args, is_short=True)} if args.long_short else setting.long_short_trade
    return setting

# ========================================================================

class StrategyUtil:
    def apply(self, data, conditions, debug=False):
        if len(conditions) == 0:
            return False
        if debug:
            checker = CombinationChecker()
            print(checker.get_source(conditions[0]), checker.get_source(conditions[1]))

        a = list(map(lambda x: x(data), conditions[0]))
        b = list(map(lambda x: x(data), conditions[1]))

        if debug:
            print(a, b)

        return all(a) and (any(b) or len(b) == 0)

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

        risk = risk * (order + 1) * data.setting.min_unit
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
    def max_position(self, data, max_risk, risk):
        if risk == 0:
            return 0
        max_position = int((max_risk / risk))
        return max_position


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

# ========================================================================

# 売買ルール
class Rule:
    def __init__(self, callback):
        self.callback = callback

    def apply(self, data):
        return self.callback(data)

# ========================================================================
# 売買戦略
class Strategy:
    def __init__(self, new_rules, taking_rules, stop_loss_rules, closing_rules):
        self.new_rules = list(map(lambda x: Rule(x), new_rules))
        self.taking_rules = list(map(lambda x: Rule(x), taking_rules))
        self.stop_loss_rules = list(map(lambda x: Rule(x), stop_loss_rules))
        self.closing_rules = list(map(lambda x: Rule(x), closing_rules))

class StrategySetting():
    def __init__(self):
        self.new = 0
        self.taking = 0
        self.stop_loss = 0
        self.closing = 0
        self.x2 = None
        self.x4 = None
        self.x8 = None

    # spaceからの読込用
    def by_array(self, params):
        self.new = int(params[0])
        self.taking = int(params[1])
        self.stop_loss = int(params[2])
        self.closing = int(params[3])
        self.x2 = int(params[4]) if len(params) > 4 and params[4] is not None else None
        self.x4 = int(params[5]) if len(params) > 5 and params[5] is not None else None
        self.x8 = int(params[6]) if len(params) > 6 and params[6] is not None else None
        return self

    # 設定からの読込用
    def by_dict(self, params):
        self.new = int(params["new"])
        self.taking = int(params["taking"])
        self.stop_loss = int(params["stop_loss"])
        self.closing = int(params["closing"])
        self.x2 = int(params["x2"]) if "x2" in params.keys() and params["x2"] is not None else None
        self.x4 = int(params["x4"]) if "x4" in params.keys() and params["x4"] is not None else None
        self.x8 = int(params["x8"]) if "x8" in params.keys() and params["x8"] is not None else None
        return self

    # 設定への書き込み用
    def to_dict(self):
        return {
            "new": self.new,
            "taking": self.taking,
            "stop_loss": self.stop_loss,
            "closing": self.closing,
            "x2": self.x2,
            "x4": self.x4,
            "x8": self.x8
        }

class StrategyConditions():
    def __init__(self):
        self.new = []
        self.taking = []
        self.stop_loss = []
        self.closing = []
        self.x2 = []
        self.x4 = []
        self.x8 = []

    def by_array(self, params):
        self.new = params[0]
        self.taking = params[1]
        self.stop_loss = params[2]
        self.closing = params[3]
        self.x2 = params[4] if len(params) > 4 else []
        self.x4 = params[5] if len(params) > 5 else []
        self.x8 = params[6] if len(params) > 6 else []
        return self

# ========================================================================

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

class CombinationCreator(StrategyCreator, StrategyUtil):
    def __init__(self, setting=None):
        self.setting = CombinationSetting() if setting is None else setting

    def ranges(self):
        return [
            list(range(utils.combinations_size(self.new()))),
            list(range(utils.combinations_size(self.taking()))),
            list(range(utils.combinations_size(self.stop_loss()))),
            list(range(utils.combinations_size(self.closing()))),
            list(range(utils.combinations_size(self.x2()))),
            list(range(utils.combinations_size(self.x4()))),
            list(range(utils.combinations_size(self.x8()))),
        ]

    def create(self, settings):
        strategy_setting = StrategySetting() if len(settings) == 0 else settings[0]
        condition = self.conditions(strategy_setting)
        return Combination(condition, self.common(settings), self.setting).create()

    # インデックスから直接条件を生成
    def conditions(self, setting):
        return StrategyConditions().by_array([
            utils.combination(setting.new, self.new()),
            utils.combination(setting.taking, self.taking()),
            utils.combination(setting.stop_loss, self.stop_loss()),
            utils.combination(setting.closing, self.closing()),
            [] if setting.x2 is None else utils.combination(setting.x2, self.x2()),
            [] if setting.x4 is None else utils.combination(setting.x4, self.x4()),
            [] if setting.x8 is None else utils.combination(setting.x8, self.x8()),
        ])

    def conditions_by_index(self, conditions, index):
        a = [conditions[i] for i in index[0]]
        b = [conditions[i] for i in index[1]]
        return [a, b]


    def subject(self):
        raise Exception("Need override subject.")

    def conditions_index(self):
        raise Exception("Need override conditions_index.")

    # 何か追加データが欲しいときはoverrideする
    def add_data(self, data):
        return data

    def default_common(self):
        rules = [lambda d: True]
        return StrategyCreator(new=rules, taking=rules, stop_loss=rules, closing=[lambda d: False])

    # @return StrategyCreator
    def common(self, setting):
        return self.default_common()

    # 継承したクラスから条件のリストから組み合わせを生成する
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

    def x2(self):
        return [
            lambda d: False
        ]

    def x4(self):
        return [
            lambda d: False
        ]

    def x8(self):
        return [
            lambda d: False
        ]

# ensemble用
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

## 指定可能な戦略====================================================================================================================

class CombinationSetting:
    on_close = {
        "new": False,
        "repay": False
    }
    simple = {
        "new": False,
        "taking": False,
        "stop_loss": False,
        "closing": False,
    }
    use_limit = False
    position_sizing = False
    position_adjust = True
    strict=True
    assets = 0
    max_position_size = 5
    condition_size = 5
    seed = [int(t.time())]
    ensemble = []
    weights = {}

class Combination(StrategyCreator, StrategyUtil):
    def __init__(self, conditions, common, setting=None):
        self.conditions = conditions
        self.common = common
        self.setting = CombinationSetting() if setting is None else setting

    def drawdown_allowable(self, data):
        allowable_dd = data.setting.stop_loss_rate
        drawdown = data.stats.drawdown()[-20:]
        drawdown_diff = list(filter(lambda x: x > allowable_dd, drawdown)) if len(drawdown) > 1 else []
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

        if self.setting.position_adjust:
            risk = self.risk(data)
            max_risk = self.max_risk(data)

            max_position = self.max_position(data, max_risk, risk)
            max_position = max_position if max_position < self.setting.max_position_size else self.setting.max_position_size
        else:
            max_position = self.setting.max_position_size

        additional = [self.apply(data, self.conditions.new)]

        # 数量
        order = 1
        if self.setting.position_sizing:
            # レバレッジ
            order = 2 if self.apply(data, self.conditions.x2) else order
            order = 4 if self.apply(data, self.conditions.x4) else order
            order = 8 if self.apply(data, self.conditions.x8) else order

            # 最大を超える場合は調整
            if order + data.position.get_num() > max_position:
#                if data.setting.debug:
#                    print("order(+position) > max_position: ", order, data.position.get_num(), max_position)
                order = max_position - data.position.get_num()

            # レバレッジシグナルも買いシグナルとする
            if not self.setting.strict:
                additional = additional + [
                    self.apply(data, self.conditions.x2),
                    self.apply(data, self.conditions.x4),
                    self.apply(data, self.conditions.x8)
                ]

        conditions = [
            self.drawdown_allowable(data), # ドローダウンが問題ない状態
            data.position.get_num() < max_position, # 最大ポジションサイズ以下
            order > 0,
            self.apply_common(data, self.common.new)
        ]

        if not self.setting.simple["new"]:
            conditions = conditions + [any(additional)]

        if all(conditions):
            if self.setting.use_limit:
                return simulator.LimitOrder(order, self.price(data), is_short=data.setting.short_trade)
            else:
                return simulator.MarketOrder(order, on_close=self.setting.on_close["new"], is_short=data.setting.short_trade)

        return None

    # 利食い
    def create_taking_rules(self, data):
        if self.setting.simple["taking"]:
            conditions = [self.apply_common(data, self.common.taking)]
        else:
            conditions = [
                self.apply_common(data, self.common.taking),
                self.apply(data, self.conditions.taking)
            ]
        if all(conditions):
            order = data.position.get_num()
            if self.setting.use_limit:
                return simulator.LimitOrder(order, self.price(data), is_repay=True, is_short=data.setting.short_trade)
            else:
                return simulator.MarketOrder(order, on_close=self.setting.on_close["repay"], is_short=data.setting.short_trade)
        return None

    # 損切
    def create_stop_loss_rules(self, data):
        conditions = [
            data.position.gain_rate(data.data.daily["close"].iloc[-1]) < -self.stop_loss_rate(data, self.setting.max_position_size), # 保有数最大で最大の損切ラインを適用
        ]
        if self.setting.simple["stop_loss"]:
            conditions = conditions + [self.apply_common(data, self.common.stop_loss)]
        else:
            conditions = conditions + [
                self.apply_common(data, self.common.stop_loss) and self.apply(data, self.conditions.stop_loss),
            ]
        if any(conditions):
            order = data.position.get_num()
            if self.setting.use_limit:
                return simulator.ReverseLimitOrder(order, self.price(data), is_repay=True, is_short=data.setting.short_trade)
            else:
                return simulator.MarketOrder(order, on_close=self.setting.on_close["repay"], is_short=data.setting.short_trade)
        return None

    # 手仕舞い
    def create_closing_rules(self, data):
        if self.setting.simple["closing"]:
            conditions = [self.apply_common(data, self.common.closing)]
        else:
            conditions = [
                self.apply_common(data, self.common.closing),
                self.apply(data, self.conditions.closing),
            ]
        if all(conditions):
            order = data.position.get_num()
            return simulator.MarketOrder(order, on_close=True, is_short=data.setting.short_trade)

        return None


class CombinationChecker:

    def get_replaced_source(self, condition):
        source = inspect.getsource(condition)
        argspec = inspect.getfullargspec(condition)

        source = re.sub("^ +lambda d.*: ", "", source)

        args = argspec.args
        args.remove("d")

        defaults = [] if argspec.defaults is None else argspec.defaults

        for name, value in zip(args, defaults):
            source = source.replace(name, "\"%s\"" % str(value))

        return source

    def get_source(self, conditions):
        sources = list(map(lambda x: self.get_replaced_source(x), conditions))
        sources = list(map(lambda x: x.strip("\n"), sources))
        sources = list(map(lambda x: x.strip(","), sources))
        sources = list(map(lambda x: x.strip(), sources))
        return sources

    def get_strategy_sources(self, combination_strategy, setting):
        new, taking, stop_loss, closing, x2, x4, x8 = [], [], [], [], [], [], []
        s = setting["setting"][0]
        new         = new       + [utils.combination(s["new"], combination_strategy.new_conditions)] if "new" in s.keys() else []
        taking      = taking    + [utils.combination(s["taking"], combination_strategy.taking_conditions)] if "taking" in s.keys() else []
        stop_loss   = stop_loss + [utils.combination(s["stop_loss"], combination_strategy.stop_loss_conditions)] if "stop_loss" in s.keys() else []
        closing     = closing   + [utils.combination(s["closing"], combination_strategy.closing_conditions)] if "closing" in s.keys() else []
        x2          = x2        + [utils.combination(s["x2"], combination_strategy.x2_conditions)] if "x2" in s.keys() else []
        x4          = x4        + [utils.combination(s["x4"], combination_strategy.x4_conditions)] if "x4" in s.keys() else []
        x8          = x8        + [utils.combination(s["x8"], combination_strategy.x8_conditions)] if "x8" in s.keys() else []

        conditions = {
            "new": new,
            "taking": taking,
            "stop_loss": stop_loss,
            "closing": closing,
            "x2": x2,
            "x4": x4,
            "x8": x8
        }

        results = {}

        for name, condition in conditions.items():
            sources = {"all": {"source":"", "condition":""}, "any": {"source":"", "condition":""}}
            for a, b in condition:
                sources["all"] = {"source": self.get_source(a), "condition": a}
                sources["any"] = {"source": self.get_source(b), "condition": b}
            results[name] = sources

        return results

