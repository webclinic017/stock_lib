# -*- coding: utf-8 -*-
import utils
import pandas
from datetime import datetime
from portfolio import high_update

def load_portfolio(date, price, length=10, by_day=False):
    if by_day:
        d = utils.to_format(utils.select_weekday(date, to_before=False))
    else:
        d = utils.to_format(datetime(date.year, date.month, 1))
    try:
        data = pandas.read_csv("portfolio/high_update/%s.csv" % d, header=None)
        data.columns = ["code", "price", "count", "date"]
        data = data[data["price"] <= price]

        if len(data) < 50 or len(data) > 550:
            data = None
        else:
            data = data.iloc[:length]
    except:
        data = None
    return data

def common(default):
    return high_update.common(default)

