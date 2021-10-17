# -*- coding: utf-8 -*-
import utils
import pandas
from datetime import datetime

def load_portfolio(date, price, length=10):
    d = utils.to_format(datetime(date.year, date.month, 1))
    try:
        data = pandas.read_csv("portfolio/financials/%s.csv" % d, header=None)
        data.columns = ["code", "value", "count", "date"]
        data = data.iloc[:length]
    except:
        data = None
    return data

def common(default):
    return default

