# -*- coding: utf-8 -*-
import numpy
import math
import pandas
import json
import jpholiday
import statsmodels.api as sm
import collections
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from functools import wraps
import subprocess
import talib as ta
import slack
import time

def nan_to_num(data, default=0):
    return data if default is None else numpy.nan_to_num(data)

def add_average_stats(data):
    close = numpy.array(data["close"].values, dtype="f8")
    data["average"]     = ta.SMA(close, timeperiod=5)
    data["long_average"]    = ta.SMA(close, timeperiod=25)
    data["volume_average"]    = ta.SMA(data["volume"].astype(float).values, timeperiod=5)
    data["ma_divergence"]     = (data["close"] - data["long_average"]) / data["long_average"]
    return data

def add_tec_stats(data):
    data["rci"]                 = data["close"].rolling(9).apply(rci, raw=True)
    data["rci_long"]            = data["close"].rolling(27).apply(rci, raw=True)

    close = numpy.array(data["close"].values, dtype="f8")
    macd, macdsignal, macdhist = ta.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    data["macd"]           = macd
    data["macdsignal"]     = macdsignal
    data["macdhist"]       = macdhist
    data["macdhist_convert"] = data["macdhist"].rolling(100).apply(trend_convert, raw=True)

    # average true range
    data["atr"] = ta.ATR(data["high"].astype(float).values, data["low"].astype(float).values, data["close"].astype(float).values, timeperiod=14)
    return data

def add_band_stats(data):
    close = numpy.array(data["close"].values, dtype="f8")
    upper, _, lower = ta.BBANDS(close, timeperiod=25, nbdevup=1, nbdevdn=1, matype=0)
    upper2, _, lower2 = ta.BBANDS(close, timeperiod=25, matype=0)

    data["env12"]          = nan_to_num(upper2)
    data["env11"]          = nan_to_num(upper)
    data["env09"]          = nan_to_num(lower)
    data["env08"]          = nan_to_num(lower2)

    data["env_entity"]     =  each(lambda i, x: x["env12"] - x["env08"], data)
    data["env_entity_average"] = ta.SMA(numpy.array(data["env_entity"].values, dtype="f8"), timeperiod=5)

    return data

def add_safety_stats(data):
    data["low_noize"]         = data["low"].rolling(2).apply(lambda x: x[-2] - x[-1] if x[-2] - x[-1] < 0 else 0, raw=True)
    data["rising_safety"]     = data["low_noize"].rolling(10).apply(rising_safety, raw=True)
    data["rising_safety"]     = data["low"] + data["rising_safety"]
    data["rising_safety"]     = data["rising_safety"].rolling(3).max() # 過去のsafetyより低い場合は高い方に合わせる

    data["high_noize"]        = data["high"].rolling(2).apply(lambda x: x[-2] - x[-1] if x[-2] - x[-1] > 0 else 0, raw=True)
    data["fall_safety"]       = data["high_noize"].rolling(10).apply(fall_safety, raw=True)
    data["fall_safety"]       = data["high"] + data["fall_safety"]
    data["fall_safety"]       = data["fall_safety"].rolling(3).min() # 過去のsafetyより高い場合は低い方に合わせる

    return data

def add_resistance_stats(data):
    data["resistance"] = data["high"].rolling(15).max()

    data["resistance_3d"] = data["high"].rolling(3).max()
    data["resistance_5d"] = data["high"].rolling(5).max()
    data["resistance_10d"] = data["high"].rolling(10).max()
    data["resistance_1m"] = data["high"].rolling(20).max()
    data["resistance_3m"] = data["high"].rolling(60).max()
    data["resistance_6m"] = data["high"].rolling(120).max()
    data["resistance_1y"] = data["high"].rolling(240).max()

    data["resistance_entity"] = (data["resistance_10d"] - data["resistance_3d"]) * (~((data["resistance_3d"] == data["resistance_5d"]) | (data["resistance_5d"] == data["resistance_10d"]) | (data["resistance_3d"] == data["resistance_10d"])) * 1)
    data["resistance_entity"] = data["resistance_entity"] / data["close"]

    return data

def add_support_stats(data):
    data["support"] = data["low"].rolling(15).min()

    data["support_3d"] = data["low"].rolling(3).min()
    data["support_5d"] = data["low"].rolling(5).min()
    data["support_10d"] = data["low"].rolling(10).min()
    data["support_1m"] = data["low"].rolling(20).min()
    data["support_3m"] = data["low"].rolling(60).min()
    data["support_6m"] = data["low"].rolling(120).min()
    data["support_1y"] = data["low"].rolling(240).min()

    data["support_entity"] = (data["support_3d"] - data["support_10d"]) * (~((data["support_3d"] == data["support_5d"]) | (data["support_5d"] == data["support_10d"]) | (data["support_3d"] == data["support_10d"])) * 1)
    data["support_entity"] = data["support_entity"] / data["close"]

    return data


def add_stages_stats(data):
    data["stages_3d"]                   = stages(data, "resistance_3d", "support_3d", "average", with_eq=True)
    data["stages_5d"]                   = stages(data, "resistance_5d", "support_5d", "average", with_eq=True)
    data["stages_10d"]                  = stages(data, "resistance_10d", "support_10d", "average", with_eq=True)
    data["stages_1m"]                   = stages(data, "resistance_1m", "support_1m", "long_average", with_eq=True)
    data["stages_3m"]                   = stages(data, "resistance_3m", "support_3m", "long_average", with_eq=True)
    data["stages_6m"]                   = stages(data, "resistance_6m", "support_6m", "long_average", with_eq=True)
    data["stages_1y"]                   = stages(data, "resistance_1y", "support_1y", "long_average", with_eq=True)

    data["stages"]                      = stages(data)
    data["stages_average"]              = ta.SMA(data["stages"].astype(float).values, timeperiod=10)
    data["stages_sum"]                  = data["stages_3d"] + data["stages_5d"] + data["stages_10d"] + data["stages_1m"] + data["stages_3m"] + data["stages_6m"] + data["stages_1y"]
    data["stages_sum_average"]          = ta.SMA(data["stages_sum"].astype(float).values, timeperiod=5)
    data["stages_sum_average_long"]     = ta.SMA(data["stages_sum"].astype(float).values, timeperiod=20)
    data["macd_stages"]                 = (data["macd"] > 0) * 1
    data["macdhist_stages"]             = (data["macdhist"] > 0) * 1

    return data

def add_cross_stats(data):
    # クロス系
    data["average_cross"] = cross(data["average"], data["long_average"])
    data["macd_cross"] = cross(data["macd"], data["macdsignal"])
    data["rci_cross"] = cross(data["rci"], data["rci_long"])

    data["env12_cross"] = cross(data["high"], data["env12"])
    data["env11_cross"] = cross(data["high"], data["env11"])
    data["env09_cross"] = cross(data["low"], data["env09"])
    data["env08_cross"] = cross(data["low"], data["env08"])

    data["rising_safety_cross"] = cross(data["low"], data["rising_safety"])
    data["fall_safety_cross"] = cross(data["high"], data["fall_safety"])

    return data

def add_trend_stats(data):
    # 気配を出力する
    data["average_gradient"]            = diff(data["average"])
    data["long_average_gradient"]       = diff(data["long_average"])
    data["volume_gradient"]             = diff(data["volume_average"])
    data["macd_gradient"]               = diff(data["macd"])
    data["macdhist_gradient"]           = diff(data["macdhist"])
    data["rci_gradient"]                = diff(data["rci"])
    data["rci_long_gradient"]           = diff(data["rci_long"])
    data["stages_gradient"]             = diff(data["stages"])
    data["stages_average_gradient"]     = diff(data["stages_average"])
    data["stages_sum_gradient"]         = diff(data["stages_sum"])
    data["stages_sum_average_gradient"] = diff(data["stages_sum_average"])
    data["stages_sum_average_long_gradient"] = diff(data["stages_sum_average_long"])
    data["fall_safety_gradient"]        = diff(data["fall_safety"])
    data["rising_safety_gradient"]      = diff(data["rising_safety"])

    data["average_trend"]               = data["average_gradient"].rolling(5).apply(trend, raw=True)
    data["long_average_trend"]          = data["long_average_gradient"].rolling(5).apply(trend, raw=True)
    data["volume_average_trend"]        = data["volume_gradient"].rolling(5).apply(trend, raw=True)
    data["macd_trend"]                  = data["macd_gradient"].rolling(5).apply(trend, raw=True)
    data["macdhist_trend"]              = data["macdhist_gradient"].rolling(1).apply(trend, raw=True)
    data["rci_trend"]                   = data["rci_gradient"].rolling(5).apply(trend, raw=True)
    data["rci_long_trend"]              = data["rci_long_gradient"].rolling(5).apply(trend, raw=True)
    data["stages_trend"]                = data["stages_gradient"].rolling(5).apply(trend, raw=True)
    data["stages_average_trend"]        = data["stages_average_gradient"].rolling(5).apply(trend, raw=True)
    data["stages_sum_trend"]            = data["stages_sum_gradient"].rolling(5).apply(trend, raw=True)
    data["stages_sum_average_trend"]    = data["stages_sum_average_gradient"].rolling(5).apply(trend, raw=True)
    data["stages_sum_average_long_trend"]    = data["stages_sum_average_long_gradient"].rolling(5).apply(trend, raw=True)
    data["rising_safety_trend"]         = data["rising_safety_gradient"].rolling(5).apply(trend, raw=True)
    data["fall_safety_trend"]           = data["fall_safety_gradient"].rolling(5).apply(trend, raw=True)

    return data

def add_manda_stats(data):
    data["pct_change"] = data["close"].pct_change()
    data["stock_split"]             = (data["pct_change"] < -0.45) * 1
    data["reverse_stock_split"]     = (data["pct_change"] > 2.0) * 1
    data["manda"]                   = ((data["stock_split"] == 1) | (data["reverse_stock_split"] == 1)) * 1

    return data

def stop_high(data):
    before = data[-2]
    current = data[-1]
    limit = price_limit(before)
    gradient = current - before
    return limit == gradient

def stop_low(data):
    before = data[-2]
    current = data[-1]
    limit = price_limit(before)
    gradient = current - before
    return -limit == gradient

def add_stats(data, default=0, names=[]):
    is_t = lambda name : len(names) == 0 or name in names

    stats = {
        "average": add_average_stats,
        "resistance": add_resistance_stats,
        "support": add_support_stats,
        "tec": add_tec_stats,
        "band": add_band_stats,
        "safety": add_safety_stats,
        "manda": add_manda_stats,
        "stages": add_stages_stats, # resistance / support
        "cross": add_cross_stats, # average / tec / band / safety
        "trend": add_trend_stats, # average / tec / band / safety / stages
    }

    for name in stats.keys():
        try:
            data = stats[name](data) if is_t(name) else data
        except Exception as e:
            import traceback
            traceback.print_exc()

    if default is None:
        return data

    for column in data.columns.tolist():
        data[column] = nan_to_num(data[column], default)

    return data

def each(callback, data):
    return list(map(lambda x: callback(*x), data.iterrows()))

# ろうそく足のパターン
def add_cs_stats(data):
    toint = lambda x: 1 if x else 0

    # 実体
    data["entity"] = (data["open"] - data["close"]).abs()
    entity = numpy.array(data["entity"].values, dtype="f8")
    entity_average = ta.SMA(entity, timeperiod=5)
    data["entity_average"] = entity_average
    # 上ヒゲ・下ヒゲ
    data["upper_shadow"] = each(lambda i, x: x["high"] - max([x["open"], x["close"]]), data)
    data["lower_shadow"] = each(lambda i, x: min([x["open"], x["close"]]) - x["low"], data)

    ## ここから
    # 長い上ヒゲ・下ヒゲ
    data["long_upper_shadow"] = (data["upper_shadow"] > data["entity"]) * 1
    data["long_lower_shadow"] = (data["lower_shadow"] > data["entity"]) * 1
    # 陽線・陰線
    data["yang"]   = (data["open"] < data["close"]) * 1
    data["yin"]    = (data["open"] > data["close"]) * 1
    data["long_yang"]  = ((data["yang"] == 1) & (data["entity"] > data["entity_average"])) * 1
    data["long_yin"]   = ((data["yin"] == 1) & (data["entity"] > data["entity_average"])) * 1

    # 切り上げ
    data["low_roundup"]  = (data["low"].shift(1) < data["low"]) * 1
    data["high_roundup"] = (data["high"].shift(1) < data["high"]) * 1
    # 切り下げ
    data["low_rounddown"]  = (data["low"].shift(1) > data["low"]) * 1
    data["high_rounddown"] = (data["high"].shift(1) > data["low"]) * 1

    # n日以内の高値安値更新
    data["high_update"] = (data["high"].rolling(5).max() == data["high"]) * 1
    data["low_update"] = (data["low"].rolling(5).min() == data["low"]) * 1

    # ギャップ
    data["yang_gap"] = (data["high"].shift(1) < data["low"]) * 1
    data["yin_gap"]  = (data["low"].shift(1) > data["high"]) * 1
    data["gap"]      = data["yang_gap"] + data["yin_gap"]
    # つつみ線
    data["tsutsumi"]      = (data["entity"].shift(1) < data["entity"]) * 1
    data["yang_tsutsumi"] = ((data["tsutsumi"] == 1) & (data["yin"].shift(1) == 1) & (data["yang"] == 1)) * 1
    data["yin_tsutsumi"]  = ((data["tsutsumi"] == 1) & (data["yang"].shift(1) == 1) & (data["yin"] == 1)) * 1
    # はらみ線
    data["harami"]      = (data["entity"].shift(1) > data["entity"]) * 1
    data["yang_harami"] = ((data["harami"] == 1) & (data["yin"].shift(1) == 1) & (data["yang"] == 1)) * 1
    data["yin_harami"]  = ((data["harami"] == 1) & (data["yang"].shift(1) == 1) & (data["yin"] == 1)) * 1
    # 毛抜き
    data["upper_kenuki"] = (data["upper_shadow"].shift(1) < data["upper_shadow"]) * 1
    data["lower_kenuki"] = (data["lower_shadow"].shift(1) < data["lower_shadow"]) * 1
    # 宵の明星
    data["yoi_mojo"] = ((data["yang_gap"].shift(1) == 1) & (data["yin_gap"] == 1)) * 1
    # 明けの明星
    data["ake_mojo"] = ((data["yin_gap"].shift(1) == 1) & (data["yang_gap"] == 1)) * 1
    # 三空
    data["yang_sanku"] = ((data["yang_gap"].rolling(2).min() == 1) & (data["yang"].rolling(3).min() == 1)) * 1
    data["yin_sanku"]  = ((data["yin_gap"].rolling(2).min() == 1) & (data["yin"].rolling(3).min() == 1)) * 1
    # 三兵
    data["yang_sanpei"] = ((data["yang"].rolling(3).min() == 1) & (data["low_roundup"].rolling(2).min() == 1) & (data["long_upper_shadow"] == 1)) * 1
    data["yin_sanpei"]  = ((data["yin"].rolling(3).min() == 1) & (data["high_rounddown"].rolling(2).min() == 1) & (data["long_lower_shadow"] == 1)) * 1

    # ストップ
    data["stop_high"] = (data["close"].rolling(2).apply(stop_high, raw=True)) * 1
    data["stop_low"] = (data["close"].rolling(2).apply(stop_low, raw=True)) * 1

    # スコア
    data["score"] = each(lambda i, x: score(x), data)

    return data

def score(data):
    plus = ["yang_tsutsumi", "yang_harami", "lower_kenuki", "ake_mojo", "yin_sanku", "yin_sanpei"]
    minus = ["yin_tsutsumi", "yin_harami", "upper_kenuki", "yoi_mojo", "yang_sanku", "yang_sanpei"]

    plus_score = sum(list(map(lambda x: data[x], plus)))
    minus_score = sum(list(map(lambda x: data[x], minus)))

    return plus_score - minus_score

def rising_divergence(data, verbose=False):
    patterns = pattern(data, [
        {"key": "macdhist_convert", "callback": lambda x: x == -1}, # 最安値更新
        {"key": "macd_cross", "callback": lambda x: x == 1}, # ヒストグラムがプラスに
        {"key": "macd_cross", "callback": lambda x: x == -1}, # ヒストグラムがマイナスに
        {"key": "macdhist_trend", "callback": lambda x: x == 1}, # ヒストグラムのトレンドがプラスに 
    ])

    if verbose:
        print(patterns)

    if len(list(filter(lambda x: x is None, patterns))) == 0:
        first = data["macdhist"].iloc[-patterns[0]]
        second = data["macdhist"].iloc[-patterns[2]:-patterns[3]].min()
        return first < second # 二番底の方が高い

    return False

def fall_divergence(data, verbose=False):
    patterns = pattern(data, [
        {"key": "macdhist_convert", "callback": lambda x: x == 1}, # 最高値更新
        {"key": "macd_cross", "callback": lambda x: x == -1}, # ヒストグラムがマイナス
        {"key": "macd_cross", "callback": lambda x: x == 1}, # ヒストグラムがプラスに
        {"key": "macdhist_trend", "callback": lambda x: x == -1}, # ヒストグラムのトレンドがマイナスに
    ])

    if verbose:
        print(patterns)

    if len(list(filter(lambda x: x is None, patterns))) == 0:
        first = data["macdhist"].iloc[-patterns[0]]
        second = data["macdhist"].iloc[-patterns[2]:-patterns[3]].max()
        return second < first # 二番天井の方が低い

    return False

# 指定したパターンが順番に現れるか
def pattern(data, patterns):
    results = []
    index = len(data)
    for pattern in patterns:
        if index is not None:
            index = find(data[pattern["key"]], index, pattern["callback"])
        results.append(index)
    return results

def count(data, callback):
    return len(list(filter(callback, data.iterrows())))

def exists(data, callback):
    d = data.values
    length = len(list(filter(lambda x:callback(x), d)))
    return length > 0

# 末尾から探索して最初に見つかるインデックスを返す
def find(data, term, callback, tail=True):
    if len(data) < term:
        return None

    for i in range(term):
        if tail:
            index = -(i+1)
        else:
            index = i

        if callback(data.iloc[index]):
            return abs(index)
    return None

# トレンドが変わった
def trend_convert(data):
    result = 0
    # 前日が直近の最大値だった
    if max(data) == data[-1]:
        result = 1

    # 前日が直近の最小値だった
    if min(data) == data[-1]:
        result = -1
    return result

def diff(data):
#    return numpy.gradient(data).tolist()
    return [0] + numpy.diff(data).tolist()

def cross(base, target):

    base_pre = base.shift(1)
    target_pre = target.shift(1)

    gc = ((base_pre <= target_pre) & (base >= target)) * 1
    dc = ((base_pre >= target_pre) & (base <= target)) * -1

    return gc + dc

# ローソク足の位置
#   -2: 支持線切り下げ
#   -1: 移動平均と支持線の間
#    0: 移動平均とクロス
#    1: 移動平均と抵抗線の間
#    2: 抵抗線切り上げ
def stages(data, resistance="resistance", support="support", average="average", with_eq=False):
    cross = lambda data, line: (data["low"] < data[line]) & (data[line] < data["high"])

    default = cross(data, average)
    p1 = data[average] < data["low"] # ローソク足が平均線より上
    m1 = data["high"] < data[average] # ローソク足が平均線より下

    stage_p1 = (~(default) & (p1)) * 1
    stage_m1 = (~(default) & (m1) ) * -1

    if with_eq:
        stage_p2 = ((cross(data, resistance)) | (data[resistance] <= data["high"])) * 2
        stage_m2 = ((cross(data, support)) | (data["low"] <= data[support])) * -2

        stage = numpy.maximum(stage_p1, stage_p2) + numpy.minimum(stage_m1, stage_m2)
    else:
        stage_p2 = ((cross(data, resistance)) | (data[resistance] < data["high"])) * 2
        stage_m2 = ((cross(data, support)) | (data["low"] < data[support])) * -2

        stage = stage_p1 + stage_p2 + stage_m1 + stage_m2

    return stage

def resistance(data, term):
    return data.iloc[-term:].max()

def support(data, term):
    return data.iloc[-term:].min()

# 上昇中の底値圏
def rising_safety(data):
    noize = list(filter(lambda x: x < 0, data))
    average = numpy.average(noize) if len(noize) > 0 else 0
    average = average * 3 # 係数
    return average

# 下落中の天井圏
def fall_safety(data):
    noize = list(filter(lambda x: x > 0, data))
    average = numpy.average(noize) if len(noize) > 0 else 0
    average = average * 3 # 係数
    return average

def trend(data):
    high_noize = list(filter(lambda x: x > 0, data))
    low_noize = list(filter(lambda x: x < 0, data))

    high_average = abs(numpy.average(high_noize)) if len(high_noize) > 0 else 0
    low_average = abs(numpy.average(low_noize)) if len(low_noize) > 0 else 0

    diff = abs(low_average - high_average)

    if low_average < high_average and len(low_noize) < len(high_noize) and high_average / 2 < diff:
        return 1

    if low_average > high_average and len(low_noize) > len(high_noize) and low_average / 2 < diff:
        return -1

    return 0

def strict_trend(data, term):
    d = data.iloc[-term:]
    high_noize = list(filter(lambda x: x > 0, d.values))
    low_noize = list(filter(lambda x: x < 0, d.values))

    high_average = abs(numpy.average(high_noize)) if len(high_noize) > 0 else 0
    low_average = abs(numpy.average(low_noize)) if len(low_noize) > 0 else 0

    if low_average < high_average and len(low_noize) < len(high_noize):
        return 1

    if low_average > high_average and len(low_noize) > len(high_noize):
        return -1

    return 0

# 定常過程データへ変換
def to_stationary(data, term=2):
    ts = sm.tsa.seasonal_decompose(data.values, freq=7, filt=None, two_sided=True)
    d = numpy.log(ts.trend + ts.resid)
    d = diff(d) # 前日比
    d = pandas.Series(d)
    return replace_invalid(convolve(d, term, average, padding=True, default=0))

# 騰落率
def rate(base, data):
    if base == 0:
        return base
    return (data - base) / float(base)

def get(data, index=-1):
    return data.values[index]

# NaN inf -inf を置換
def replace_invalid(data, source=0.0):
    data = numpy.array(data)
    data[(numpy.isnan(data)) | (data==float("inf")) | (data==float("-inf"))] = source
    return data

# 畳み込み
def convolve(data, term, callback, padding=True, default=0):
    convolved = [default for _ in range(term-1)] if padding else []
    if len(data) < term:
      return convolved[0:len(data)]
    for i in range(term, len(data)+1):
      d = data[0:i]
      convolved.append(callback(d, term))
    return convolved

# 前日比の割合
def gradient_rate(data):
    d = [1]
    for i in range(1, len(data)):
        r = rate(data["close"].iloc[i-1], data["close"].iloc[i])
        d.append(r)
    return d

# 単純移動平均線
def average(data, term):
    term_data = data.values[-term:]
    return numpy.convolve(term_data, numpy.ones(term)/float(term), 'valid')[0]

# RCI
def rci(data):
    term = len(data)
    term_data = data[::-1].reshape(-1)
    index = numpy.argsort(term_data)[::-1]
    d = 0
    for i, si in enumerate(index):
      d += ((i+1) - (si+1)) ** 2

    return int(( 1.0 - ( (6 * d) / float(term ** 3 - term) ) ) * 100)

# 標準偏差
def sigma(data, term):
    term_data = data.values[-term:]
    return math.sqrt((term * sum(term_data ** 2) - sum(term_data ** 2)) / (term * (term - 1)))

# ストキャスティクス
def stochastics(data, term):
    term_data = data.values[-term:]
    min_data = min(term_data)
    max_data = max(term_data)
    A = term_data[-1] - min_data
    B = max_data - min_data
    if B == 0:
        return 0.0
    s = A / float(B)
    return s * 100

# GCまたはDC
def gdc(d1, d2):
    if golden_cross(d1, d2):
        return 1
    elif dead_cross(d1, d2):
        return -1
    return 0

# ゴールデンクロス
def golden_cross(d1, d2):
    if d1 is None or d2 is None:
        return False
    if not (len(d1) == len(d2)):
        return False
    if not isinstance(d1, list):
        d1 = d1.values
    if not isinstance(d2, list):
        d2 = d2.values
    for i in range(len(d1)-1):
        if d1[i] < d2[i] and d1[i+1] > d2[i+1]:
            return True
    return False

# デッドクロス
def dead_cross(d1, d2):
    if d1 is None or d2 is None:
        return False
    if not (len(d1) == len(d2)):
        return False
    if not isinstance(d1, list):
        d1 = d1.values
    if not isinstance(d2, list):
        d2 = d2.values

    for i in range(len(d1)-1):
        if d1[i] > d2[i] and d1[i+1] < d2[i+1]:
            return True
    return False

# ドローダウン
def drawdown(data):
    base = numpy.array(data, dtype=float)
    maximums = numpy.maximum.accumulate(base)
    drawdown = maximums - base
    drawdown = numpy.divide(drawdown, maximums, out=numpy.zeros_like(drawdown), where=maximums!=0)
    drawdown = drawdown.tolist()
    return drawdown

# =======================================
# 値幅制限
def price_limit(price):
    index, price_range = price_limit_with_index(price)
    return price_range

def price_limit_with_index(price):
    import rakuten
    return rakuten.price_limit(price)

# date util =======================================
def timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def to_datetime(date, input_format="%Y-%m-%d"):
    return datetime.strptime(date, input_format)

def format(date, input_format="%Y-%m-%d", output_format="%Y-%m-%d"):
    return to_datetime(date, input_format).strftime(output_format)

def to_format(date, output_format="%Y-%m-%d"):
    return date.strftime(output_format)

def to_format_by_term(date, with_time=False):
    if with_time:
        return to_format(date, output_format="%Y-%m-%d %H:%M:%S")
    else:
        return to_format(date)

def relativeterm(term, with_time=False):
    if with_time:
        return relativedelta(days=term)
    else:
        return relativedelta(months=term)

def to_datetime_by_term(date, with_time=False):
    if with_time:
        return to_datetime(date, input_format="%Y-%m-%d %H:%M:%S")
    else:
        return to_datetime(date)

def is_weekday(date):
    return date.weekday() < 5 and jpholiday.is_holiday_name(date.date()) is None

def select_weekday(date, to_before=True, target=None):
    select = date
    if target is None:
        target = lambda x: True
    while(not (is_weekday(select) and target(select))):
        if to_before:
            select = select - relativeterm(1, with_time=True)
        else:
            select = select + relativeterm(1, with_time=True)
    return select

# end_dateは含まない
def daterange(start_date, end_date):
    for n in range((end_date - start_date).days):
        yield start_date + timedelta(n)

def daterange_by_month(start_date, end_date):
    delta = relativedelta(end_date, start_date)
    months = delta.years * 12 + delta.months
    for n in range(months):
        yield start_date + relativedelta(months=n)

# combination util =======================================
# 全組み合わせを取得
def combinations(conditions):
    num = combinations_size(conditions)
    cond = [combination(i, conditions) for i in range(num)]
    cond = sorted(cond, key=lambda x: len(x[0]))
    return cond

# 2進数のフラグで取り出すパターンを決める 0 -> a, 1 -> b
def condition(index, conditions):
    a, b = [], []
    for j in range(len(conditions)): # 1ビットずつどちらに割りふるかチェック
        if int(bin(index & pow(2, j)), 0) != 0:
            a.append(conditions[j])
        else:
            b.append(conditions[j])
    return (a, b)

# 生成される組み合わせのサイズ
def combinations_size(conditions):
    return pow(2, len(conditions))*3 - 3

# 指定したindexの組み合わせを取得する
def combination(index, conditions):
    num = combinations_size(conditions)
    exc = {
        0: ([], conditions),
        num-2: (conditions, []),
        num-1: ([], [])
    }

    if index in exc.keys():
        return exc[index]
    r = divmod(index-1, 3)
    cond = condition(r[0]+1, conditions)
    if r[1] == 1:
        return (cond[0], [])
    elif r[1] == 2:
        return ([], cond[1])
    else:
        return cond

# list util =======================================
def split_list(data, condition):
#    if len(data) <= 1:
#        return data
    for i, d in enumerate(data):
        if condition(d):
            return [data[:i+1]] + split_list(data[i+1:], condition)
    return [data]

def chunked(iterable, n):
    return [iterable[x:x + n] for x in range(0, len(iterable), n)]

# dict util =======================================
def to_jsonizable(dic):
    to_numpy = lambda x: (x[0], numpy.array([x[1]]).tolist()[0]) if "numpy" in str(type(x[1])) else x
    jsonizable = lambda x: (x[0], to_jsonizable(x[1])) if type(x[1]) is dict else x
    return dict(list(map(lambda x: jsonizable(to_numpy(x)), list(dic.items()))))

# process =======================================
def proc_call(params, retry=3):
    print(params)
    for _ in range(retry):
        ret = subprocess.call(params, timeout=300)
        if ret == 0:
            return

def stop_watch(func) :
    @wraps(func)
    def wrapper(*args, **kargs):
        start = time.time()
        result = func(*args,**kargs)
        elapsed_time =  time.time() - start
        print("%s: %s sec." % (func.__name__, round(elapsed_time, 6)))
        return result
    return wrapper
