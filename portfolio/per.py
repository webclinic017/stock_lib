# -*- coding: utf-8 -*-
import utils
import pandas
from datetime import datetime

def load_portfolio(date, length=10):
    d = utils.to_format(date)
    try:
        data = pandas.read_csv("portfolio/per/%s.csv" % d, header=None)
        data.columns = ["code", "per", "close"]
        data = data.iloc[:length]
    except:
        data = None
    return data

