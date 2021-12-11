# -*- coding: utf-8 -*-
import math
from simulator import Simulator, SecuritiesCompony

class Rakuten(SecuritiesCompony):
    # 値幅制限
    def price_limit(self, price):
        prices_low = [
            [100, 30],
            [200, 50],
            [500, 80],
            [700, 100],
            [1000, 150]
        ]
        prices_high = [
            [1000, 150],
            [1500, 300],
            [2000, 400],
            [3000, 500],
            [5000, 700],
            [7000, 1000],
            [10000, 1500]
        ]

        prev_limit = 0
        price_range = None
        is_low = price < 1000
        table = prices_low if is_low else prices_high
        p = price/1000
        if p <= 0.0:
            return 0, prices_low[0][1]

        count = 0 if is_low else int(math.log(int(p), 10))
        default_index = 0 if is_low else len(prices_low) - (count+1)
        index = default_index
        for i, current in enumerate(table):
            next_limit = current[0] * pow(10, count)
            if prev_limit <= price and price < next_limit:
                price_range = current[1] * pow(10, count)
                index = default_index + i + (len(prices_high) * count)
            prev_limit = next_limit

        return index, price_range

    # 呼値(1tickあたりの価格)
    def tick_price(self, price):
        tick_prices = [
            [3000, 1],
            [5000, 5],
            [30000, 10],
            [50000, 50],
        ]
        tick_price = None
        for t in tick_prices:
            if price < t[0]:
                tick_price = t[1]
                break

        if tick_price is None:
            tick_price = 100

        return tick_price

    # price: 一日の約定金額合計
    def oneday_commission(self, price):
        commissions = [
            [500000, 0],
            [1000000, 943],
            [2000000, 2200],
            [3000000, 3300],
        ]

        commission = None
        for c in commissions:
            if price < c[0]:
                commission = c[1]
                break

        # 300万以上の場合
        if commission is None:
            commission = 3300 + (int(price / 1000000) - 2) * 1100

        return commission

    # price: 取引一回の約定代金
    def default_commission(self, price, is_credit):
        actual_commissions = [
            [50000, 50],
            [100000, 90],
            [200000, 105],
            [500000, 250],
            [1000000, 487],
            [1500000, 582],
            [30000000, 921],
        ]

        credit_commissions = [
            [100000, 90],
            [200000, 135],
            [500000, 180],
        ]

        # 最大超え
        over = 350 if is_credit else 973

        commissions = credit_commissions if is_credit else actual_commissions

        commission = None
        for c in commissions:
            if price < c[0]:
                commission = c[1]
                break

        if commission is None:
            commission = over

        return int(commission * 1.1)

class RakutenSimulator(Rakuten, Simulator):
    pass

