# -*- coding: utf-8 -*-
import numpy
import utils

# 売買ルール
class Rule:
    def __init__(self, callback):
        self._callback = callback

    def apply(self, data):
        results = self._callback(data)
        return results

