# -*- coding: utf-8 -*-
import os
import time
import re
import json
import math
import numpy
import inspect
import utils
import simulator
import itertools
import time as t
import glob
import pandas
import random
import conditions
from datetime import datetime
from loader import Loader
from loader import Bitcoin
from collections import namedtuple
from simulator import SimulatorData
from simulator import SimulatorIndexData
from simulator import SimulatorSetting
from argparse import ArgumentParser

def add_options(parser):
    parser.add_argument("-v", action="store_true", default=False, dest="verbose", help="debug log")
    parser.add_argument("--code", type=str, action="store", default=None, dest="code", help="code")
    parser.add_argument("--setting_dir", type=str, action="store", default=None, dest="setting_dir", help="")
    parser.add_argument("--use_limit", action="store_true", default=False, dest="use_limit", help="指値を使う")
    parser.add_argument("--position_sizing", action="store_true", default=False, dest="position_sizing", help="ポジションサイジング")
    parser.add_argument("--max_position_size", action="store", default=None, dest="max_position_size", help="最大ポジションサイズ")
    parser.add_argument("--production", action="store_true", default=False, dest="production", help="本番向け") # 実行環境の選択
    parser.add_argument("--short", action="store_true", default=False, dest="short", help="空売り戦略")
    parser.add_argument("--soft_limit", type=float, action="store", default=None, dest="soft_limit", help="価格ベース自動損切")
    parser.add_argument("--hard_limit", type=float, action="store", default=None, dest="hard_limit", help="資産ベース自動損切")
    parser.add_argument("--stop_loss_rate", action="store", default=None, dest="stop_loss_rate", help="損切レート")
    parser.add_argument("--taking_rate", action="store", default=None, dest="taking_rate", help="利食いレート")
    parser.add_argument("--min_unit", action="store", default=None, dest="min_unit", help="最低単元")
    parser.add_argument("--assets", type=int, action="store", default=None, dest="assets", help="assets")
    parser.add_argument("--instant", action="store_true", default=False, dest="instant", help="日次トレード")
    parser.add_argument("--max_leverage", type=int, action="store", default=None, dest="max_leverage", help="max_leverage")
    parser.add_argument("--passive_leverage", action="store_true", default=False, dest="passive_leverage", help="passive_leverage")
    parser.add_argument("--condition_size", type=int, action="store", default=None, dest="condition_size", help="条件数")
    parser.add_argument("--conditions", type=str, action="store", default=None, dest="conditions", help="利用する条件リスト名")
    parser.add_argument("--appliable_signal", type=str, action="store", default=None, dest="appliable_signal", help="設定を適用可能なシグナルリスト")
    parser.add_argument("--portfolio", type=str, action="store", default=None, dest="portfolio", help="ポートフォリオ")
    parser.add_argument("--portfolio_size", type=int, action="store", default=None, dest="portfolio_size", help="ポートフォリオの銘柄数")

    # strategy
    parser.add_argument("--ensemble_dir", action="store", default=None, dest="ensemble_dir", help="アンサンブルディレクトリ")
    parser.add_argument("--ensemble", action="store_true", default=False, dest="ensemble", help="アンサンブル")
    parser.add_argument("--open_close", action="store_true", default=False, dest="open_close", help="寄せ引け")
    parser.add_argument("--futures", action="store_true", default=False, dest="futures", help="先物")
    parser.add_argument("--new_high", action="store_true", default=False, dest="new_high", help="新高値")
    parser.add_argument("--high_update", action="store_true", default=False, dest="high_update", help="高値更新")
    parser.add_argument("--low_update", action="store_true", default=False, dest="low_update", help="安値更新")
    parser.add_argument("--per", action="store_true", default=False, dest="per", help="PER")
    parser.add_argument("--financials", action="store_true", default=False, dest="financials", help="決算情報ベース")
    parser.add_argument("--signals", action="store_true", default=False, dest="signals", help="シグナル")
    parser.add_argument("--simple", action="store_true", default=False, dest="simple", help="シンプル")
    return parser

def create_parser():
    parser = ArgumentParser()
    return add_options(parser)

def get_prefix(args):
    return create_prefix(args, args.production, args.short)

def create_prefix(args, is_production, is_short):
    prefix = "production_" if is_production else ""

    method = "short_" if is_short else ""

    instant = "instant_" if args.instant else ""

    target = get_strategy_name(args)
    target = "" if target == "combination" else "%s_" % target

    return "%s%s%s%s" % (prefix, target, method, instant)

def get_filename(args):
    prefix = get_prefix(args)
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
    SIMPLE="simple"
    HIGH_UPDATE="high_update"
    LOW_UPDATE="low_update"
    PER="per"
    FINANCIALS="financials"
    SIGNALS="signals"

    def list(self):
        return [
            self.ENSEMBLE,
            self.COMBINATION,
            self.OPEN_CLOSE,
            self.FUTURES,
            self.NEW_HIGH,
            self.SIMPLE,
            self.HIGH_UPDATE,
            self.LOW_UPDATE,
            self.PER,
            self.FINANCIALS,
            self.SIGNALS
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
    elif args.simple:
        return strategy_types.SIMPLE
    elif args.high_update:
        return strategy_types.HIGH_UPDATE
    elif args.low_update:
        return strategy_types.LOW_UPDATE
    elif args.per:
        return strategy_types.PER
    elif args.financials:
        return strategy_types.FINANCIALS
    elif args.signals:
        return strategy_types.SIGNALS
    else:
        return strategy_types.COMBINATION

# load ================================================

def select_portfolio(portfolio_name):
    strategy_types = StrategyType()
    if "filtered_high_update" == portfolio_name:
        from portfolio import filtered_high_update
        return filtered_high_update
    elif strategy_types.HIGH_UPDATE == portfolio_name:
        from portfolio import high_update
        return high_update
    elif strategy_types.LOW_UPDATE == portfolio_name:
        from portfolio import low_update
        return low_update
    elif strategy_types.NEW_HIGH == portfolio_name:
        from portfolio import new_high
        return new_high
    elif strategy_types.PER == portfolio_name:
        from portfolio import per
        return per
    elif strategy_types.FINANCIALS == portfolio_name:
        from portfolio import financials
        return financials
    elif "%s/" % strategy_types.SIGNALS in portfolio_name:
        from portfolio import signals
        return signals
    elif strategy_types.FUTURES == portfolio_name:
        from portfolio import futures
        return futures
    else:
        raise Exception("unsupported type: %s" % portfolio_name)

def load_portfolio(portfolio_name, date, price, length=10):
    if "%s/" % StrategyType().SIGNALS in portfolio_name: # signals/XXXX
        return select_portfolio(portfolio_name).load_portfolio(portfolio_name, date, price, length)
    else:
        return select_portfolio(portfolio_name).load_portfolio(date, price, length)

def load_strategy_creator_by_type(strategy_type, is_production, combination_setting):
    strategy_types = StrategyType()
    if is_production:
        if strategy_types.ENSEMBLE == strategy_type:
            from strategies.production.ensemble import CombinationStrategy
            return CombinationStrategy(combination_setting)
        elif strategy_types.NEW_HIGH == strategy_type:
            from strategies.production.new_high import CombinationStrategy
            return CombinationStrategy(combination_setting)
        elif strategy_types.FUTURES == strategy_type:
            from strategies.production.futures import CombinationStrategy
            return CombinationStrategy(combination_setting)
        elif strategy_types.HIGH_UPDATE == strategy_type:
            from strategies.production.high_update import CombinationStrategy
            return CombinationStrategy(combination_setting)
        else:
            from strategies.production.combination import CombinationStrategy
            return CombinationStrategy(combination_setting)
    else:
        if strategy_types.ENSEMBLE == strategy_type:
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
        elif strategy_types.HIGH_UPDATE == strategy_type:
            from strategies.high_update import CombinationStrategy
            return CombinationStrategy(combination_setting)
        elif strategy_types.LOW_UPDATE == strategy_type:
            from strategies.low_update import CombinationStrategy
            return CombinationStrategy(combination_setting)
        elif strategy_types.PER == strategy_type:
            from strategies.per import CombinationStrategy
            return CombinationStrategy(combination_setting)
        elif strategy_types.FINANCIALS == strategy_type:
            from strategies.financials import CombinationStrategy
            return CombinationStrategy(combination_setting)
        elif strategy_types.SIGNALS == strategy_type:
            from strategies.signals import CombinationStrategy
            return CombinationStrategy(combination_setting)
        elif strategy_types.SIMPLE == strategy_type:
            from strategies.simple import SimpleStrategy
            return SimpleStrategy()
        else:
            from strategies.combination import CombinationStrategy
            return CombinationStrategy(combination_setting)

def load_strategy_creator(args, combination_setting=None):
    combination_setting = CombinationSetting() if combination_setting is None else combination_setting
    strategy_type = get_strategy_name(args)
    return load_strategy_creator_by_type(strategy_type, args.production, combination_setting)

def load_strategy_setting(args):
    filename = get_filename(args)

    setting_dict, strategy_setting = load_strategy_setting_by_filename(filename, args.setting_dir)

    return setting_dict, strategy_setting

def load_strategy_creator_by_setting(create_setting):
    return load_strategy_creator_by_type(
        create_setting.strategy_type,
        create_setting.is_production,
        create_setting.combination_setting)

def load_strategy_setting_by_filename(filename, path):
    if path is None:
        setting_dict = Loader.simulate_setting(filename)
    else:
        setting_dict = Loader.simulate_setting(filename, path)

    strategy_setting = create_strategy_setting(setting_dict)

    return setting_dict, strategy_setting

def load_strategy(args, combination_setting=None):
    _, settings = load_strategy_setting(args)
    return load_strategy_creator(args, combination_setting).create(settings)

def load_strategy_by_option(args, is_short):
    filename = create_filename(create_prefix(args, is_production=args.production, is_short=is_short))
    setting_dict, settings = load_strategy_setting_by_filename(filename, args.setting_dir)
    combination_setting = create_combination_setting_by_dict(args, setting_dict)
    return load_strategy_creator(args, combination_setting).create(settings)

def load_simulator_data(code, start_date, end_date, args, names=[], technical_setting=None):
    rule = "D"
    start = utils.to_format(utils.to_datetime(start_date) - utils.relativeterm(12))
    data = Loader.load_by_code(code, start, end_date)

    if data is None:
        print("%s: %s is None" % (start_date, code))
        return None

    simulator_data = add_stats(code, data, rule, names, technical_setting)
    if args.verbose:
        print("loaded:", utils.timestamp(), code, data["date"].iloc[0], data["date"].iloc[-1])
    else:
        print(".", end="")
    return simulator_data

def load_index(args, start_date, end_date):
    index = {}
    start = utils.to_format(utils.to_datetime(start_date) - utils.relativeterm(6))

    for k in ["nikkei", "dow"]:
        d = Loader.load_index(k, start, end_date, with_filter=True, strict=False)
        d = add_stats(k, d, "D")
        index[k] = d

    index["new_score"] = SimulatorData("new_score", Loader.new_score(), "D")
    index["industry_score"] = SimulatorData("industry_score", Loader.industry_trend(), "D")
    index["high_update_score"] = SimulatorData("high_update_score", Loader.high_update_score(), "D")

    return SimulatorIndexData(index)

def add_stats(code, data, rule, names=[], technical_setting=None):
    try:
        data = utils.add_stats(data, names=names, technical_setting=technical_setting)
        data = utils.add_cs_stats(data)
        return SimulatorData(code, data, rule)
    except Exception as e:
        print(code, "load_error: %s" % e)
        import traceback
        traceback.print_exc()
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
    ensembles = list(map(lambda x: load_strategy_creator_by_setting(x).create(x.strategy_setting), settings))
    return ensembles

def ensemble_files(directory):
    files = sorted(glob.glob("%s/*" % directory))
    return files

# args > json > default の優先度
def create_combination_setting(args, use_json=True):
    combination_setting = create_combination_setting_by_json(args) if use_json else apply_assets(args, CombinationSetting())
    combination_setting.use_limit = args.use_limit if args.use_limit else combination_setting.use_limit
    combination_setting.position_sizing = args.position_sizing if args.position_sizing else combination_setting.position_sizing
    combination_setting.max_position_size = combination_setting.max_position_size if args.max_position_size is None else int(args.max_position_size)
    combination_setting.max_leverage = combination_setting.max_leverage if args.max_leverage is None else int(args.max_leverage)
    combination_setting.passive_leverage = combination_setting.passive_leverage if args.passive_leverage is None else args.passive_leverage
    combination_setting.condition_size = combination_setting.condition_size if args.condition_size is None else int(args.condition_size)
    combination_setting.ensemble = combination_setting.ensemble if args.ensemble_dir is None else ensemble_files(args.ensemble_dir)
    combination_setting.portfolio = combination_setting.portfolio if args.portfolio is None else args.portfolio
    combination_setting.portfolio_size = combination_setting.portfolio_size if args.portfolio_size is None else args.portfolio_size
    combination_setting.conditions = combination_setting.conditions if args.conditions is None else args.conditions.split(",")
    combination_setting.appliable_signal = combination_setting.appliable_signal if args.appliable_signal is None else json.loads(args.appliable_signal)
    return combination_setting

def create_combination_setting_by_json(args):
    setting_dict, _ = load_strategy_setting(args)
    return create_combination_setting_by_dict(args, setting_dict)

def create_combination_setting_by_dict(args, setting_dict):
    combination_setting = CombinationSetting()
    if setting_dict is None:
        return combination_setting
    combination_setting = apply_combination_setting_by_dict(combination_setting, setting_dict)
    combination_setting = apply_assets(args, combination_setting)
    return combination_setting

def apply_combination_setting_by_dict(combination_setting, setting_dict):
    combination_setting.use_limit = setting_dict["use_limit"] if "use_limit" in setting_dict.keys() else combination_setting.use_limit
    combination_setting.position_sizing = setting_dict["position_sizing"] if "position_sizing" in setting_dict.keys() else combination_setting.position_sizing
    combination_setting.max_position_size = setting_dict["max_position_size"] if "max_position_size" in setting_dict.keys() else combination_setting.max_position_size
    combination_setting.seed = setting_dict["seed"] if "seed" in setting_dict.keys() else combination_setting.seed
    combination_setting.weights = setting_dict["weights"] if "weights" in setting_dict.keys() else combination_setting.weights
    combination_setting.passive_leverage = setting_dict["passive_leverage"] if "passive_leverage" in setting_dict.keys() else combination_setting.passive_leverage
    combination_setting.condition_size = setting_dict["condition_size"] if "condition_size" in setting_dict.keys() else combination_setting.condition_size
    combination_setting.ensemble = ensemble_files(setting_dict["ensemble_dir"]) if "ensemble_dir" in setting_dict.keys() else combination_setting.ensemble
    combination_setting.portfolio = setting_dict["portfolio"] if "portfolio" in setting_dict.keys() else combination_setting.portfolio
    combination_setting.portfolio_size = setting_dict["portfolio_size"] if "portfolio_size" in setting_dict.keys() else combination_setting.portfolio_size
    combination_setting.conditions = setting_dict["conditions"] if "conditions" in setting_dict.keys() else combination_setting.conditions
    combination_setting.appliable_signal = setting_dict["appliable_signal"] if "appliable_signal" in setting_dict.keys() else combination_setting.appliable_signal
    return combination_setting

def create_simulator_setting(args, use_json=True):
    simulator_setting = create_simulator_setting_by_json(args) if use_json else SimulatorSetting()
    simulator_setting.stop_loss_rate = simulator_setting.stop_loss_rate if args.stop_loss_rate is None else float(args.stop_loss_rate)
    simulator_setting.taking_rate = simulator_setting.taking_rate if args.taking_rate is None else float(args.taking_rate)
    simulator_setting.min_unit = simulator_setting.min_unit if args.min_unit is None else int(args.min_unit)
    simulator_setting.short_trade = args.short
    simulator_setting.soft_limit = simulator_setting.soft_limit if args.soft_limit is None else args.soft_limit
    simulator_setting.hard_limit = simulator_setting.hard_limit if args.hard_limit is None else args.hard_limit
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
    simulator_setting.soft_limit = setting_dict["soft_limit"] if "soft_limit" in setting_dict.keys() else simulator_setting.soft_limit
    simulator_setting.hard_limit = setting_dict["hard_limit"] if "hard_limit" in setting_dict.keys() else simulator_setting.hard_limit
    simulator_setting.ignore_volume = args.futures
    simulator_setting = apply_assets(args, simulator_setting)
    return simulator_setting

def apply_assets(args, setting):
    assets = Loader.assets()
    setting.assets = assets["assets"] - assets["itrust"] if args.assets is None else args.assets
    return setting

# ========================================================================

class StrategyUtil:
    def apply(self, data, conditions):
        if len(conditions) == 0:
            return False

        and_cond, or_cond = conditions

        if len(and_cond) == 0 and len(or_cond) == 0:
            return False

        for condition in and_cond:
            if not condition(data):
                return False

        or_cond = list(map(lambda x: x(data), or_cond))

        return any(or_cond) or len(or_cond) == 0

    def apply_common(self, data, conditions):
        common = list(map(lambda x: x(data), conditions))
        return all(common)

    def price(self, data):
        return data.data.middle["close"].iloc[-1]

    def max_gain(self, data):
        if data.setting.short_trade:
            max_gain = data.position.gain(data.data.middle["low"].iloc[-data.position.get_term():].min(), data.position.get_num())
        else:
            max_gain = data.position.gain(data.data.middle["high"].iloc[-data.position.get_term():].max(), data.position.get_num())
        return max_gain

    # 最大許容損失
    def max_risk(self, data):
        return data.assets * data.setting.stop_loss_rate

    # 利益目標
    def goal(self, data):
        order = data.position.get_num() # 現在の保有数
        if data.setting.short_trade:
            price = data.data.middle["low"].iloc[-1]
            line = data.data.middle["support"].iloc[-1]
            goal = (price - line)
        else:
            price = data.data.middle["high"].iloc[-1]
            line = data.data.middle["resistance"].iloc[-1]
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
        upper = data.data.middle["resistance"].iloc[-term]

        return upper

    # 下限
    def lower(self, data, term=1):
        lower= data.data.middle["support"].iloc[-term]

        return lower

    # 不安要素
    def caution(self, data):
        gain = data.stats.gain()
        conditions = [
            self.risk(data) == 0, # セーフティーを下回っている
            data.position.get_num() == 0, # 初回の仕掛け
            data.position.gain(self.price(data), data.position.get_num()) < 0, # 損益がマイナス
            gain[-1] < 0 if len(gain) > 0 else False, # 最後のトレードで損失
        ]
        return any(conditions)

    # 注目要素
    def attention(self, data):
        gain = data.stats.gain()
        conditions = [
            gain[-1] > 0 if len(gain) > 0 else False, # 最後のトレードで利益
            data.position.gain(self.price(data), data.position.get_num()) > 0, # 損益がプラス
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
            return data.data.middle["fall_safety"].iloc[-term]
        else:
            return data.data.middle["rising_safety"].iloc[-term]

    def term(self, data):
        return 1 if data.position.get_term()  == 0 else data.position.get_term()

    # 保有数最大で最大の損切ラインを適用
    def stop_loss_rate(self, data, max_position_size):
        position_rate = data.position.get_num() / max_position_size
        return data.setting.stop_loss_rate * position_rate

    def stop_loss(self, data, max_position_size):
        return data.assets * self.stop_loss_rate(data, max_position_size)

    # 保有数最大で最大の利食ラインを適用
    def taking_rate(self, data, max_position_size):
        position_rate = data.position.get_num() / max_position_size
        return data.setting.taking_rate * position_rate

    def taking_gain(self, data, max_position_size):
        return data.assets * self.taking_rate(data, max_position_size)

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
    def __init__(self, new, taking, stop_loss, closing, x2, x4, x8, x0_5):
        self.new_rules = list(map(lambda x: Rule(x), new))
        self.taking_rules = list(map(lambda x: Rule(x), taking))
        self.stop_loss_rules = list(map(lambda x: Rule(x), stop_loss))
        self.closing_rules = list(map(lambda x: Rule(x), closing))
        self.x2_rules = list(map(lambda x: Rule(x), x2))
        self.x4_rules = list(map(lambda x: Rule(x), x4))
        self.x8_rules = list(map(lambda x: Rule(x), x8))
        self.x0_5_rules = list(map(lambda x: Rule(x), x0_5))

class StrategySetting():
    def __init__(self):
        self.new = 0
        self.taking = 0
        self.stop_loss = 0
        self.closing = 0
        self.x2 = None
        self.x4 = None
        self.x8 = None
        self.x0_5 = None

    def get_optional(self, params, index):
        if isinstance(params, dict):
            return int(params[index]) if index in params.keys() and params[index] is not None else None
        else:
            return int(params[index]) if len(params) > index and params[index] is not None else None

    # spaceからの読込用
    def by_array(self, params):
        self.new = int(params[0])
        self.taking = int(params[1])
        self.stop_loss = int(params[2])
        self.closing = int(params[3])
        self.x2 = self.get_optional(params, 4)
        self.x4 = self.get_optional(params, 5)
        self.x8 = self.get_optional(params, 6)
        self.x0_5 = self.get_optional(params, 7)
        return self

    # 設定からの読込用
    def by_dict(self, params):
        self.new = int(params["new"])
        self.taking = int(params["taking"])
        self.stop_loss = int(params["stop_loss"])
        self.closing = int(params["closing"])
        self.x2 = self.get_optional(params, "x2")
        self.x4 = self.get_optional(params, "x4")
        self.x8 = self.get_optional(params, "x8")
        self.x0_5 = self.get_optional(params, "x0_5")
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
            "x8": self.x8,
            "x0_5": self.x0_5
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
        self.x0_5 = []

    def by_array(self, params):
        self.new = params[0]
        self.taking = params[1]
        self.stop_loss = params[2]
        self.closing = params[3]
        self.x2 = params[4] if len(params) > 4 else []
        self.x4 = params[5] if len(params) > 5 else []
        self.x8 = params[6] if len(params) > 6 else []
        self.x0_5 = params[7] if len(params) > 7 else []
        return self

    def concat(self, left, right):
        return [
            list(left[0]) + list(right[0]), # and
            list(left[1]) + list(right[1])  # or
        ]

    def __add__(self, right):
        return StrategyConditions().by_array([
            self.concat(self.new, right.new),
            self.concat(self.taking, right.taking),
            self.concat(self.stop_loss, right.stop_loss),
            self.concat(self.closing, right.closing),
            self.concat(self.x2, right.x2),
            self.concat(self.x4, right.x4),
            self.concat(self.x8, right.x8),
            self.concat(self.x0_5, right.x0_5)
        ])

# ========================================================================

class StrategyCreator:
    def __init__(self):
        self.new_conditions = []
        self.taking_conditions = []
        self.stop_loss_conditions = []
        self.closing_conditions = []
        self.x2_conditions = []
        self.x4_conditions = []
        self.x8_conditions = []
        self.x0_5_conditions = []
        self.selected_condition_index = {}

    def create_new_orders(self, data):
        raise Exception("Need override create_new_orders.")

    def create_taking_orders(self, data):
        raise Exception("Need override create_taking_orders.")

    def create_stop_loss_orders(self, data):
        raise Exception("Need override create_stop_loss_orders.")

    def create_closing_orders(self, data):
        raise Exception("Need override create_closing_orders.")

    def create_x2(self, data):
        raise Exception("Need override create_x2.")

    def create_x4(self, data):
        raise Exception("Need override create_x4.")

    def create_x8(self, data):
        raise Exception("Need override create_x8.")

    def create_x0_5(self, data):
        raise Exception("Need override create_x0_5.")

    def subject(self, date):
        raise Exception("Need override subject.")

    def conditions_index(self):
        return self.selected_condition_index

    # 何か追加データが欲しいときはoverrideする
    def add_data(self, data, index):
        return data

    # 取引対象とするポートフォリオの日付リスト
#    def select_dates(self, start_date, end_date, instant):
#        return list(utils.daterange(utils.to_datetime(start_date), utils.to_datetime(end_date)))

    def select_dates(self, start_date, end_date, instant):
        dates = list(utils.daterange(utils.to_datetime(start_date), utils.to_datetime(end_date)))
        if instant:
            return [utils.to_datetime(start_date)]
        else:
            return list(set(map(lambda x: datetime(x.year, x.month, 1), dates)))

    def create(self, settings):
        new = [lambda x: self.create_new_orders(x)]
        taking = [lambda x: self.create_taking_orders(x)]
        stop_loss = [lambda x: self.create_stop_loss_orders(x)]
        closing = [lambda x: self.create_closing_orders(x)]
        x2 = [lambda x: self.create_x2(x)]
        x4 = [lambda x: self.create_x4(x)]
        x8 = [lambda x: self.create_x8(x)]
        x0_5 = [lambda x: self.create_x0_5(x)]
        return Strategy(new, taking, stop_loss, closing, x2, x4, x8, x0_5)

    def ranges(self):
        return [[0], [0], [0], [0], [0], [0], [0], [0]]

class CombinationCreator(StrategyCreator, StrategyUtil):
    def __init__(self, setting=None):
        self.setting = CombinationSetting() if setting is None else setting
        self.default_weights = 200
        super().__init__()

    def ranges(self):
        return [
            list(range(utils.combinations_size(self.new()))),
            list(range(utils.combinations_size(self.taking()))),
            list(range(utils.combinations_size(self.stop_loss()))),
            list(range(utils.combinations_size(self.closing()))),
            list(range(utils.combinations_size(self.x2()))),
            list(range(utils.combinations_size(self.x4()))),
            list(range(utils.combinations_size(self.x8()))),
            list(range(utils.combinations_size(self.x0_5()))),
        ]

    def create(self, settings):
        return self.create_combination(settings).create(settings)

    def create_combination(self, settings):
        strategy_settings = [StrategySetting()] if len(settings) == 0 else settings
        conditions = self.conditions(strategy_settings)
        return Combination(conditions, self.common(settings), self.setting)

    # インデックスから直接条件を生成
    def condition(self, setting):
        return StrategyConditions().by_array([
            utils.combination(setting.new, self.new()),
            utils.combination(setting.taking, self.taking()),
            utils.combination(setting.stop_loss, self.stop_loss()),
            utils.combination(setting.closing, self.closing()),
            [] if setting.x2 is None else utils.combination(setting.x2, self.x2()),
            [] if setting.x4 is None else utils.combination(setting.x4, self.x4()),
            [] if setting.x8 is None else utils.combination(setting.x8, self.x8()),
            [] if setting.x0_5 is None else utils.combination(setting.x0_5, self.x0_5()),
        ])

    def conditions(self, settings):
        condition = None
        for i in range(len(settings)):
            self.conditions_by_seed(self.setting.seed[i])
            if condition is None:
                condition = self.condition(settings[i])
            else:
                condition = condition + self.condition(settings[i])
        return condition

    def choice(self, conditions, size, weights):
        conditions_with_index = list(map(lambda x: {"x": x}, list(enumerate(conditions))))
        choiced = numpy.random.choice(conditions_with_index, size, p=weights, replace=False).tolist()
        choiced = list(map(lambda x: x["x"], choiced))
        return list(zip(*choiced))

    def apply_weights(self, method, all_condition_size=None):
        if all_condition_size is None: # ensembleの場合 シグナルごとの条件の内容が違うので個別にsizeが渡される
            all_condition_size = len(self.conditions_all) # TODO conditions_all
        base = numpy.array([self.default_weights] * all_condition_size)

        if method in self.weights.keys():
            for index, weight in self.weights[method].items():
                base[int(index)] = base[int(index)] + weight

        weights = base / sum(base)
        return weights

    # args.condition_size で変更可能なシグナルリスト
    def condition_size_map(self):
        size_map = {}
        default_appliable_signal = CombinationSetting().appliable_signal
        for key in default_appliable_signal["condition_size"]:
            if key in self.setting.appliable_signal["condition_size"]: # 変更可能なシグナルなら
                size_map[key] = self.setting.condition_size
            else:
                size_map[key] = CombinationSetting().condition_size # default
        return size_map

    def conditions_by_seed(self, seed, targets=["middle", "nikkei", "dow"], names=["all"]):
        random.seed(seed)
        numpy.random.seed(seed)

        # override でnames指定/argsでconditions指定のどちらかを適用 なにもしなければallを選択
        selected_names = names if len(self.setting.conditions) == 0 else self.setting.conditions
        selected_names = ["all"] if len(selected_names) == 0 else selected_names

        self.conditions_all         = conditions.by_names(targets=targets, names=selected_names)

        size_map = self.condition_size_map()

        new, self.new_conditions               = self.choice(self.conditions_all, size_map["new"], self.apply_weights("new"))
        taking, self.taking_conditions         = self.choice(self.conditions_all, size_map["taking"], self.apply_weights("taking"))
        stop_loss, self.stop_loss_conditions   = self.choice(self.conditions_all, size_map["stop_loss"], self.apply_weights("stop_loss"))
        closing, self.closing_conditions       = self.choice(self.conditions_all, size_map["closing"], self.apply_weights("closing"))
        x2, self.x2_conditions                 = self.choice(self.conditions_all, size_map["x2"], self.apply_weights("x2"))
        x4, self.x4_conditions                 = self.choice(self.conditions_all, size_map["x4"], self.apply_weights("x4"))
        x8, self.x8_conditions                 = self.choice(self.conditions_all, size_map["x8"], self.apply_weights("x8"))

        # 選択された条件のインデックスを覚えておく
        self.selected_condition_index = {
            "new":new, "taking": taking, "stop_loss": stop_loss, "x2": x2, "x4": x4, "x8": x8
        }


    def default_common(self):
        rules = [lambda d: True]
        return StrategyConditions().by_array([rules, rules, [lambda d: False], [lambda d: False]])

    # @return StrategyConditions
    def common(self, settings):
        return self.default_common()

    # 継承したクラスから条件のリストから組み合わせを生成する
    def new(self):
        return self.new_conditions

    def taking(self):
        return self.taking_conditions

    def stop_loss(self):
        return self.stop_loss_conditions

    def closing(self):
        return self.closing_conditions

    def x2(self):
        return self.x2_conditions

    def x4(self):
        return self.x4_conditions

    def x8(self):
        return self.x8_conditions

    def x0_5(self):
        return [lambda d: False] # defaultでは使わない

# ensemble用
class StrategyCreateSetting:
    def __init__(self, filepath):
        # ディレクトリに判別用の文字列が含まれる場合があるのでファイル名のみにする
        filename = os.path.basename(filepath)
        self.strategy_type = self.get_strategy_name(filename)
        self.is_production = "production" in filename
        self.setting_dict = Loader.simulate_setting(filepath, "") # フルパスの想定
        self.strategy_setting = create_strategy_setting(self.setting_dict)
        self.combination_setting = apply_combination_setting_by_dict(CombinationSetting(), self.setting_dict)

    def get_strategy_name(self, filename):
        strategy_types = StrategyType()
        for t in strategy_types.list():
            if t in filename:
                return t
        return strategy_types.COMBINATION

## 指定可能な戦略====================================================================================================================

class SignalSetting:
    def __init__(self, new=False, taking=False, stop_loss=False, closing=False):
        self.new = new
        self.taking = taking
        self.stop_loss = stop_loss
        self.closing = closing

class CombinationSetting:
    on_close = {
        "new": False,
        "repay": False
    }
    enabled = {
        "signal": SignalSetting(new=True, taking=True, stop_loss=True, closing=False), # 全体のシグナル
        "simple": SignalSetting() # commonのみ
    }
    use_limit = False
    position_sizing = False
    position_adjust = True
    strict = True
    assets = 0
    max_position_size = 4
    max_leverage = None
    passive_leverage = False
    condition_size = 5
    appliable_signal = {
        "condition_size": ["new", "taking", "stop_loss", "closing", "x2", "x4", "x8", "x0_5"]
    }
    seed = [int(t.time())]
    ensemble = []
    weights = {}
    montecarlo = False
    portfolio = None
    portfolio_size = 10
    conditions = []

class Combination(StrategyCreator, StrategyUtil):
    def __init__(self, conditions, common, setting=None):
        self.conditions = conditions
        self.common = common
        self.setting = CombinationSetting() if setting is None else setting

    def drawdown_allowable(self, data, term=20):
        allowable_dd = data.setting.stop_loss_rate
        drawdown = data.stats.drawdown()[-term:]
        drawdown_diff = list(filter(lambda x: x > allowable_dd, drawdown)) if len(drawdown) > 1 else []
        drawdown_sum = list(filter(lambda x: x > 0, numpy.diff(drawdown))) if len(drawdown) > 1 else []
        drawdown_conditions = [
            len(drawdown_diff) == 0, # -n%を超えて一定期間たった
            sum(drawdown_sum) < allowable_dd # 直近のドローダウン合計がn%以下
        ]

        allow = all(drawdown_conditions)

        return allow

    def available_leverage(self, level):
        return self.setting.max_leverage is None or level <= self.setting.max_leverage

    def active_leverage(self, data, order, max_position):
        if self.apply(data, self.conditions.x8) and self.available_leverage(8):
            return 8
        if self.apply(data, self.conditions.x4) and self.available_leverage(4):
            return 4
        if self.apply(data, self.conditions.x2) and self.available_leverage(2):
            return 2
        return order

    def passive_leverage(self, data, order, max_position):
        if self.apply(data, self.conditions.x2) and self.available_leverage(2):
            return 2
        if self.apply(data, self.conditions.x4) and self.available_leverage(4):
            return 4
        if self.apply(data, self.conditions.x8) and self.available_leverage(8):
            return 8
        return order

    def position_sizing(self, data, order, max_position):
        if self.setting.passive_leverage:
            order = self.passive_leverage(data, order, max_position)
        else:
            order = self.active_leverage(data, order, max_position)

        # 最大を超える場合は調整
        if order + data.position.get_num() > max_position:
            order = max_position - data.position.get_num()
        return order

    def additional_new_signals(self, data):
        if self.setting.strict:
            return []
        else:
            return [
                self.apply(data, self.conditions.x2),
                self.apply(data, self.conditions.x4),
                self.apply(data, self.conditions.x8)
            ]

    def position_adjust(self, data):
        risk = self.risk(data)
        max_risk = self.max_risk(data)

        max_position = self.max_position(data, max_risk, risk)
        max_position = max_position if max_position < self.setting.max_position_size else self.setting.max_position_size

        return max_position

    # 買い
    def create_new_orders(self, data):
        if not self.setting.enabled["signal"].new:
            return None

        if self.setting.position_adjust:
            max_position = self.position_adjust(data)
        else:
            max_position = self.setting.max_position_size

        conditions = []
        additional = []

        # 数量
        order = 1
        if self.setting.position_sizing:
            # レバレッジ
            order = self.position_sizing(data, order, max_position)
            # レバレッジシグナルも買いシグナルとする
            additional = additional + self.additional_new_signals(data)

        if not self.drawdown_allowable(data): # ドローダウンが問題ない状態
            return None
        if not order > 0:
            return None
        if not data.position.get_num() < max_position: # 最大ポジションサイズ以下
            return None
        if not self.apply_common(data, self.common.new):
            return None

        additional = additional + [self.apply(data, self.conditions.new)]

        if not self.setting.enabled["simple"].new:
            if self.setting.montecarlo:
                seed = int(time.time())
                numpy.random.seed(seed)
                conditions = conditions + [numpy.random.rand() > 0.5]
            else:
                conditions = conditions + [any(additional)]

        if all(conditions):
            if self.setting.use_limit:
                return simulator.LimitOrder(order, self.price(data), is_short=data.setting.short_trade)
            else:
                return simulator.MarketOrder(order, on_close=self.setting.on_close["new"], is_short=data.setting.short_trade)

        return None

    # 利食い
    def create_taking_orders(self, data):
        if not self.setting.enabled["signal"].taking:
            return None

        order = data.position.get_num()
        gain = data.position.gain(self.price(data), order)

        if not gain > 0:
            return None
        if not self.apply_common(data, self.common.taking):
            return None

        if self.apply(data, self.conditions.x0_5): # x0.5なら半分にする
            order = math.ceil(order / 2)

        if self.setting.enabled["simple"].taking or self.apply(data, self.conditions.taking):
            if self.setting.use_limit:
                return simulator.LimitOrder(order, self.price(data), is_repay=True, is_short=data.setting.short_trade)
            else:
                return simulator.MarketOrder(order, on_close=self.setting.on_close["repay"], is_short=data.setting.short_trade)
        return None

    # 損切
    def create_stop_loss_orders(self, data):
        if not self.setting.enabled["signal"].stop_loss:
            return None

        order = data.position.get_num()
        gain = data.position.gain(self.price(data), order)
        stop_loss = self.stop_loss(data, self.setting.max_position_size)

        if not gain < 0:
            return None

        conditions = [
            gain < -stop_loss
        ]
        if self.setting.enabled["simple"].stop_loss:
            conditions = conditions + [self.apply_common(data, self.common.stop_loss)]
        else:
            conditions = conditions + [
                self.apply_common(data, self.common.stop_loss),
                self.apply(data, self.conditions.stop_loss),
            ]

        if any(conditions):
            if self.setting.use_limit:
                return simulator.ReverseLimitOrder(order, self.price(data), is_repay=True, is_short=data.setting.short_trade)
            else:
                return simulator.MarketOrder(order, on_close=self.setting.on_close["repay"], is_short=data.setting.short_trade)
        return None

    # 手仕舞い
    def create_closing_orders(self, data):
        if not self.setting.enabled["signal"].closing:
            return None

        conditions = [
            self.apply_common(data, self.common.closing)
        ]

        if not self.setting.enabled["simple"].closing:
            conditions = conditions + [self.apply(data, self.conditions.closing)]

        if any(conditions):
            order = data.position.get_num()
            return simulator.MarketOrder(order, on_close=self.setting.on_close["repay"], is_short=data.setting.short_trade)

        return None

    def create_x2(self, data):
        return self.apply(data, self.conditions.x2)

    def create_x4(self, data):
        return self.apply(data, self.conditions.x4)

    def create_x8(self, data):
        return self.apply(data, self.conditions.x8)

    def create_x0_5(self, data):
        return self.apply(data, self.conditions.x0_5)

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
        new, taking, stop_loss, closing, x2, x4, x8, x0_5 = [], [], [], [], [], [], [], []
        s = setting["setting"][0]

        new         = new       + [utils.combination(s["new"], combination_strategy.new_conditions)] if "new" in s.keys() else []
        taking      = taking    + [utils.combination(s["taking"], combination_strategy.taking_conditions)] if "taking" in s.keys() else []
        stop_loss   = stop_loss + [utils.combination(s["stop_loss"], combination_strategy.stop_loss_conditions)] if "stop_loss" in s.keys() else []
        closing     = closing   + [utils.combination(s["closing"], combination_strategy.closing_conditions)] if "closing" in s.keys() else []
        x2          = x2        + [utils.combination(s["x2"], combination_strategy.x2_conditions)] if "x2" in s.keys() else []
        x4          = x4        + [utils.combination(s["x4"], combination_strategy.x4_conditions)] if "x4" in s.keys() else []
        x8          = x8        + [utils.combination(s["x8"], combination_strategy.x8_conditions)] if "x8" in s.keys() else []
        x0_5        = x0_5      + [utils.combination(s["x0_5"], combination_strategy.x0_5_conditions)] if "x0_5" in s.keys() else []

        conditions = {
            "new": new,
            "taking": taking,
            "stop_loss": stop_loss,
            "closing": closing,
            "x2": x2,
            "x4": x4,
            "x8": x8,
            "x0_5": x0_5
        }

        results = {}

        for name, condition in conditions.items():
            sources = {"all": {"source":"", "condition":""}, "any": {"source":"", "condition":""}}
            for a, b in condition:
                sources["all"] = {"source": self.get_source(a), "condition": a}
                sources["any"] = {"source": self.get_source(b), "condition": b}
            results[name] = sources

        return results

