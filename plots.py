# -*- coding: utf-8 -*-
import matplotlib as pltlib
pltlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.finance as mpf
import subprocess
import utils
import slack

class Plotter:
    def __init__(self, code, data, extention="png", axs=None):
        self.code = code
        self.data = data
        self.extention = extention
        self.filename = "results/plot_%s.%s" % (self.code, extention)
        self.axs = axs

    def plot(self, stats=False, alerts=None, resistances=None, supports=None):
        if self.axs is None:
            fig = plt.figure()
            ax1 = fig.add_subplot(1,1,1)
        else:
            ax1 = self.axs

        ax1.grid() # グリッド表示
        mpf.candlestick2_ohlc(ax1, self.data["open"], self.data["high"], self.data["low"], self.data["close"], width=0.7, colorup='r', colordown='b')
        ax1.axhline(y=self.data["close"].iloc[-1], color="orange", alpha=1.0)

#        ax2 = ax1.twinx()
#        if "volume" in self.data.columns:
#            ax2.bar(range(len(self.data["date"])), self.data["volume"], label="volume", alpha=0.5)

        if alerts is not None:
            ax1 = self.alerts(ax1, alerts)
        if resistances is not None:
            ax1 = self.resistance(ax1, resistances)
        if supports is not None:
            ax1 = self.support(ax1, supports)

        if stats:
            self.stats(ax1)

        return ax1

    def stats(self, ax1):
        keys = ["daily_average", "weekly_average", "env12", "env11", "env09", "env08"]
        for key in keys:
            ax1.plot(range(len(self.data["date"])), self.data[key], alpha=0.3)

        ax2 = ax1.twinx()
        keys = ["macd", "macdsignal", "macdhist"]
        for key in keys:
            ax2.plot(range(len(self.data["date"])), self.data[key], alpha=0.8)


        return ax1

    def resistance(self, ax1, resistances):
        for i, dmax in enumerate(sorted(resistances.values(), key=lambda x: x - self.data["close"].iloc[0])):
            ax1.axhline(y=dmax, color="red", alpha=1.0 if i == 0 else 0.3)
        return ax1

    def support(self, ax1, supports):
        objects = []
        for i, dmin in enumerate(sorted(supports.values(), key=lambda x: self.data["close"].iloc[0] - x)):
            ax1.axhline(y=dmin, color="blue", alpha=1.0 if i == 0 else 0.3)
        return ax1

    def alerts(self, ax1, alerts):
        for alert in alerts:
            if alert["alert"] in ["max_near", "min_near"]:
                near = alert["price"] * (alert["near"] / 100)
                high = alert["price"] + near
                low = alert["price"] - near
                ax1.axhline(y=alert["price"], color="red" if alert["alert"] == "max_near" else "blue")
                ax1.axhspan(low, high, color="red" if alert["alert"] == "max_near" else "blue", alpha=0.1)
        return ax1

    def save(self, user=None, channel=None):
        plt.savefig(self.filename)
        if user is None or channel is None:
            slack.file_post(self.extention, self.filename)
        else:
            slack.file_post(self.extention, self.filename, user=user, channel=channel)
