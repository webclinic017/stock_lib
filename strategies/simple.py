# -*- coding: utf-8 -*-
import numpy
import utils
import simulator
import conditions
import random
import subprocess
import pandas
from datetime import datetime
from dateutil.relativedelta import relativedelta
from strategy import StrategyCreator, StrategyUtil
from loader import Loader

class SimpleStrategy(StrategyCreator, StrategyUtil):
    def create_new_orders(self, data):
        return simulator.MarketOrder(1, is_short=data.setting.short_trade)

    def create_taking_orders(self, data):
        if data.position.gain(self.price(data), data.position.get_num()) > 0:
            return simulator.MarketOrder(1, on_close=True, is_short=data.setting.short_trade)
        return None

    def create_stop_loss_orders(self, data):
        if data.position.gain(self.price(data), data.position.get_num()) < 0:
            return simulator.MarketOrder(1, on_close=True, is_short=data.setting.short_trade)
        return None

    def create_closing_orders(self, data):
#        if data.position.get_num() > 0:
#            return simulator.MarketOrder(1, is_short=data.setting.short_trade)
        return None

    def subject(self, date):
        return []

    def conditions_index(self):
        return {}
