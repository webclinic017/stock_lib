# -*- coding: utf-8 -*-
from statsmodels.tsa import stattools
import statsmodels.api as sm
import csv
import json
import subprocess
import utils
import talib as ta
import numpy
from loader import Loader

def target(data):
    if manda(data) or data["close"].iloc[-1] < 300:
        return False

    volume = numpy.average(data["volume"].as_matrix().tolist())
    if volume <= 500.0: # 平均出来高が10万以上
        return False

    return True

def rising(data):
    if not target(data):
        return False

    result, resid = ols(data["close"])
    result = ols_results_format(result)
    resid_adf = adf(resid) 
    result = {"ols": result, "resid": resid.tolist(), "resid_adf": resid_adf}
    return rising_resid_steady(result)

def fall(data):
    if not target(data):
        return False

    result, resid = ols(data["close"])
    result = ols_results_format(result)
    resid_adf = adf(resid) 
    result = {"ols": result, "resid": resid.tolist(), "resid_adf": resid_adf}
    return fall_resid_steady(result)

def new_high(data):
    today = data["high"].iloc[-1]
    high = max(data["high"].iloc[:-1])
    return high < today

def new_low(data):
    today = data["low"].iloc[-1]
    low = min(data["low"].iloc[:-1])
    return today < low

# M & A
def manda(data):
    return stock_split(data, -0.45) or reverse_stock_split(data, 2.0)

# 株式併合
def reverse_stock_split(data, rate):
    gradient_rate = utils.gradient_rate(data)
    return any(list(map(lambda x: x > rate, gradient_rate))) # 一回でも一定割合以上上昇していたら併合したとみなす

# 株式分割
def stock_split(data, rate):
    gradient_rate = utils.gradient_rate(data)
    return any(list(map(lambda x: x < rate, gradient_rate))) # 一回でも一定割合以上下降していたら分割したとみなす

# ストップ高
def stop_high(data):
    daily = Loader.resample(data, rule="D")
    before = daily["close"].iloc[-2]
    current = data["close"].iloc[-1]
    limit = utils.price_limit(before)
    gradient = current - before
    return limit == gradient

# ストップ安
def stop_low(data):
    daily = Loader.resample(data, rule="D")
    before = daily["close"].iloc[-2]
    current = data["close"].iloc[-1]
    limit = utils.price_limit(before)
    gradient = current - before
    return -limit == gradient

# ストップ
def price_stop(data):
    return stop_high(data) or stop_low(data)

def template(data, output=False):

    close = numpy.array(data["close"].as_matrix(), dtype="f8")

    if len(close) == 0:
        return False

    current = close[-1]
    average50   = ta.SMA(close, timeperiod=50)
    average150  = ta.SMA(close, timeperiod=150)
    average200  = ta.SMA(close, timeperiod=200)
    high260     = data["high"].iloc[-260:].min()
    low260      = data["low"].iloc[-260:].min()
    rsi         = ta.RSI(close, timeperiod=5)

    condition = [
        average150[-1] < current and average200[-1] < current,
        average200[-1] < average150[-1],
        average150[-1] < average50[-1] and average200[-1] < average50[-1],
        average50[-1] < current,
        low260 * 1.3 < current,
        high260 * 0.75 < current and current < high260 * 1.25,
        70 < rsi[-1]
    ]

    if output:
        print(condition)

    if not all(condition):
        return False

    average200_1m  = average200[-20:]
    average200_3m  = average200[-60:]
    average200_5m  = average200[-100:]


    ols_1m = ols_results_format(ols(average200_1m)[0])
    ols_3m = ols_results_format(ols(average200_3m)[0])
    ols_5m = ols_results_format(ols(average200_5m)[0])

    ols_condition = [float(ols_1m["x1"]) > 0 or float(ols_3m["x1"]) > 0 or float(ols_5m["x1"]) > 0]

    return all(ols_condition)

# ランダムウォークであるか
def is_randomwalk(data):
    p_results = float(data["p_value"]["nc"]) > 0.01 # p値が0.01以上
    regs_results = float(data["regs"]["nc"]["reg"]) < 0.0 # 回帰係数がマイナス(どの係数を利用するかはBIC or AICで決定できる)
    return p_results and regs_results

def resid_plus(data):
    return float(data["resid"][-1]) > 0.0

def resid_minus(data):
    return float(data["resid"][-1]) < 0.0

# 上昇トレンド 
def is_rising_trend(data):
    return float(data["ols"]["x1"]) > 0.0

# 下降トレンド
def is_fall_trend(data):
    return float(data["ols"]["x1"]) < 0.0

def rising_resid_steady(data):
    return is_rising_trend(data) and is_randomwalk(data["resid_adf"])# and resid_minus(data)

def fall_resid_steady(data):
    return is_fall_trend(data) and is_randomwalk(data["resid_adf"])# and resid_plus(data)

# データ解析
def analysis(start_date, end_date, code, key="close", use_index=False):
    params = ["python", "scripts/data_checker.py", start_date, end_date, str(code)]
    params = params + ["-i"] if use_index else params
    results = subprocess.check_output(params, timeout=300)
    json_str = results.splitlines()[-1].decode('utf-8')
    print(json_str)
    data = json.loads(json_str)
    return data

# olsの結果整形
def ols_results_format(results):
    summary = results.summary().as_csv()
    params = results.params
    base = params[0]
    scalar = params[1]
    reader = csv.reader(summary.strip().splitlines())
    results = dict()
    for i, row in enumerate(reader):
        row = [ r.strip().replace(":","") for r in row]
        if i in {12,15,16}:
            results[row[0]] = row[1]
        if i in {2, 14,15}:
            results[row[2]] = row[3]
    results["base"] = base
    results["scalar"] = scalar
    return results

#########################################
# 線形回帰
#########################################
def ols(data, output=False, label="close"):
    if output:
        print("#### [OLS] %s ####" % label)
    y = data
    x = range(len(y))
    x = sm.add_constant(x)
    model = sm.OLS(y, x)
    results = model.fit()
    if output:
        print(results.summary())
    return results, results.resid

#########################################
# ADF検定
# ランダムウォークであるかどうか
# regs = ["ctt", "ct", "c", "nc"]
#########################################
def adf(data, regs=["nc"], output=False, label="close"):
    if output:
        print("#### [ADF] %s ####" % label)
    result = dict()
    result["p_value"] = dict()
    for reg in regs:
        res = stattools.adfuller(data, regression=reg)
        # 二つ目の値（p値）が0.01以下とかなら定常過程のデータ
        if output:
            print(reg, res)
        result["p_value"][reg] = res[1]
    result["regs"] = drift(data, regs, output)
    return result

def drift(data, regs, output=False):
    result = dict()
    y = data.diff().dropna()
    x = data.shift(1).dropna()
    model = sm.OLS(y, x)
    results = model.fit()
    if "nc" in regs:
        result["nc"] = {"reg":results.params[0], "aic":results.aic, "bic":results.bic}
    if output:
        print("without drift ", results.params[0]) # [0]:回帰係数
    x = sm.add_constant(x)
    model = sm.OLS(y, x)
    results = model.fit()
    if "c" in regs:
        result["c"] = {"reg":results.params[1], "aic":results.aic, "bic":results.bic}
    if output:
        print("with drift ", results.params[0], results.params[1]) # [1]:回帰係数
    x["t"] = range(len(y))
    model = sm.OLS(y, x)
    results = model.fit()
    if "ct" in regs:
        result["ct"] = {"reg":results.params[1], "aic":results.aic, "bic":results.bic}
    if output:
        print("with drift + time trend ", results.params[0], results.params[1], results.params[2]) # [1]:回帰係数
    return result

#########################################
# 自己相関
#########################################
def acf(data, output=False, label="close"):
    if output:
        print("#### [ACF] %s ####" % label)
    acf,q,pvalue = stattools.acf(data, qstat=True)
    print(acf,q,pvalue)
    return acf

#########################################
# 偏自己相関
#########################################
def pacf(data, output=False, label="close"):
    if output:
        print("#### [PACF] %s ####" % label)
    pacf = stattools.pacf(data)
    print(pacf)
    return pacf


