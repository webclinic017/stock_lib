# -*- coding: utf-8 -*-
import utils
import pandas
from datetime import datetime
from portfolio import high_update

def load_portfolio(date, length=10, by_day=False):
    d = utils.to_format(date)
    try:
        data = pandas.read_csv("portfolio/per/%s.csv" % d, header=None)
        data.columns = ["code", "per", "close"]
        data = data.iloc[:length]
    except:
        data = None
    return data

def common(default):
    return high_update.common(default)

