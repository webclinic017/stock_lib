import sys
import numpy
import math

sys.path.append("lib")
import utils
import checker
import strategy
from loader import Loader

class Condition:
    def __init__(self):
        self.condition = {}

    def keys(self):
        return self.condition.keys()

    def strength(self, params=None):
        if params is None:
            return sum(self.condition.values())

        result = []
        keys = sorted(self.condition.keys())
        for i, k in enumerate(keys):
            result.append(self.condition[k] * params[i])
        return sum(result)

class PlusCondition(Condition):
    def __init__(self):
        super().__init__()
        self.condition = {
            "industry_growth": 0,
            "volume_gradient": 0,
            "entity_gradient": 0,
            "long_lower_shadow_hammer": 0,
            "invalid_hammer": 0,
            "rising_hammer": 0,
            "valid_long_entity": 0,
            "volatility": 0,
        }

class MinusCondition(Condition):
    def __init__(self):
        super().__init__()
        self.condition = {
            "volume_gradient": 0,
            "entity_gradient": 0,
            "long_lower_shadow_hammer": 0,
            "fall_meteor": 0,
            "rising_hammer": 0,
            "invalid_meteor": 0,
            "valid_short_entity": 0
        }

def strength(data, industry_index, plus_params=None, minus_params=None):
    data["volume_gradient"]   = numpy.gradient(data["volume"])
    data["volume_gradient_plus"] = list(map(lambda x: 1 if x > 0 else 0, data["volume_gradient"]))
    data["entity_gradient_minus"] = list(map(lambda x: 1 if x < 0 else 0, data["entity_gradient"]))
    data["reversed_gradient"] = data["gradient"] * -1

    ols, resid = checker.ols(data["close"])
    result = { "ols": checker.ols_results_format(ols) }

    plus = plus_condition(result, data, industry_index)
    minus = minus_condition(result, data, industry_index)

    return plus.strength(plus_params) - minus.strength(minus_params)

def plus_condition(ols, data, index):
    # プラスの要素 #######################################################################
    plus = PlusCondition()

    # 株価のトレンド
    trend = int(checker.is_rising_trend(ols)) # 上昇トレンド

    # 業種内での伸び
    index_rate = utils.rate(index["value"].iloc[0], index["value"].iloc[-1]) * 100
    data_rate = utils.rate(data["log"].iloc[0], data["log"].iloc[-1]) * 100

    if index_rate > 0 and data_rate > 0:
        plus.condition["industry_growth"] = data_rate - index_rate
    else:
        plus.condition["industry_growth"] = 0

    ## ろうそく足のパターン
    ### ストッピングボリューム
    # 出来高の前日比がプラスの日が多い
    # 実体の前日比がマイナスの日が多い
    # 下髭の長いハンマーが多い
    plus.condition["volume_gradient"] = sum(data["volume_gradient_plus"]) * trend
    plus.condition["entity_gradient"] = sum(data["entity_gradient_minus"]) * trend
    plus.condition["long_lower_shadow_hammer"] = utils.count_exists(data, ["long_lower_shadow", "hammer"])

    ### 平均以上の出来高のハンマー
    plus.condition["invalid_hammer"] = utils.count_exists(data, ["invalid", "hammer"])

    ### 上昇トレンド中の平均以下の出来高のハンマー
    plus.condition["rising_hammer"] = utils.count_exists(data, ["gradient", "valid", "hammer"])

    ### 平均以上の出来高の長大線
    plus.condition["valid_long_entity"] = utils.count_exists(data, ["valid", "long_entity"])

    ### ボラティリティと制限値幅の割合
    limit = utils.price_limit(data["close"].iloc[-1])
    volatility = numpy.average(data["volatility"].iloc[-3:])
    plus.condition["volatility"] = 0 if limit == 0 else (volatility / limit) * 10

    return plus

def minus_condition(ols, data, index):
    # マイナスの要素 #######################################################################
    minus = MinusCondition()

    # 株価のトレンド
    trend = int(checker.is_fall_trend(ols)) # 下降トレンド

    ## ろうそく足のパターン
    ### トッピングアウトボリューム
    # 出来高の前日比がプラスの日が多い
    # 実体の前日比がマイナスの日が多い
    # 上髭の長い流れ星が多い
    minus.condition["volume_gradient"] = sum(data["volume_gradient_plus"]) * trend
    minus.condition["entity_gradient"] = sum(data["entity_gradient_minus"]) * trend
    minus.condition["long_lower_shadow_hammer"] = utils.count_exists(data, ["long_lower_shadow", "hammer"])

    ### 下降トレンド中の平均以下の出来高の流れ星
    minus.condition["fall_meteor"] = utils.count_exists(data, ["reversed_gradient", "valid", "meteor"])

    ### 上昇トレンド中の平均以上のハンマー
    minus.condition["rising_hammer"] = utils.count_exists(data, ["gradient", "invalid", "hammer"])

    ### 平均以上の出来高の流れ星
    minus.condition["invalid_meteor"] = utils.count_exists(data, ["invalid", "meteor"])

    ### 平均以上の出来高の短小線
    minus.condition["valid_short_entity"] = utils.count_exists(data, ["valid", "short_entity"])

    return minus


