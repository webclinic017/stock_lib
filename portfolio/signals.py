# -*- coding: utf-8 -*-
import utils
import pandas
from datetime import datetime

def load_portfolio(signals, date, price, length=10):
#    d = utils.to_format(datetime(date.year, date.month, 1))
    d = utils.to_format(date)
    try:
        data = pandas.read_csv("portfolio/%s/%s.csv" % (signals, d))
        data = data[data["price"] <= price]
        data = data.iloc[:length]
    except:
        data = None
    return data

def common(default):
    return default

