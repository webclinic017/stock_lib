# -*- coding: utf-8 -*-
import utils
import pandas
from datetime import datetime

def load_portfolio(date, price, length=10, by_day=False):
    if by_day:
        d = utils.to_format(utils.select_weekday(date, to_before=False))
    else:
        d = utils.to_format(datetime(date.year, date.month, 1))
    try:
        data = pandas.read_csv("portfolio/high_update/%s.csv" % d, header=None)
        data.columns = ["code", "price", "count", "date"]
        data = data[data["price"] <= price]
        data = data.iloc[:length]
    except:
        data = None
    return data


def break_precondition(d):
    conditions = [
        d.data.middle["high_update"][-2:].max() == 0 and (d.position.gain(d.data.middle["close"].iloc[-1], d.position.get_num()) <= 0 or sum(d.stats.gain()) <= 0) and d.position.get_num() >= 0,
        d.data.middle["high_update"][-10:].sum() <= 5
    ]

    return any(conditions)

def common(default):
    default.new = [
        lambda d: d.index.data["new_score"].middle["score"].iloc[-1] > -400,
        lambda d: d.data.middle["stop_low"].iloc[-1] == 0,
        lambda d: not break_precondition(d),
    ]

    default.closing = [
        lambda d: break_precondition(d)
    ]

    return default

