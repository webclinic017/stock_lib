# -*- coding: utf-8 -*-
import utils
import pandas
from datetime import datetime

def load_portfolio(date, price, length=10):
    d = utils.to_datetime(date)
    year = d.year
    month = (int(d.month / 3) + 1) * 3
#    if d.month in [3, 6, 9, 12]:
#        month = month + 3
    if 12 < month:
        year = d.year + 1
        month = 3
    code = "nikkei225mini_%s%02d_daytime" % (year, month)
    return pandas.DataFrame([code], columns=["code"])

