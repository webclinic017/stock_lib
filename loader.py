# -*- coding: utf-8 -*-
import os
import numpy
import pandas
import json
import utils
from datetime import datetime
from dateutil.relativedelta import relativedelta

class Index:
    nikkei = "nikkei"
    topix = "topix"
    jasdaq = "jasdaq"
    usdjpy = "usdjpy"

    code_map = {
        nikkei: "n225",
        topix: "topx",
        jasdaq: "jsdi",
    }

    exchange_map = {
        usdjpy: "usd/jpy"
    }

    def codes(self):
        return self.code_map.values()

    def exchanges(self):
        return self.exchange_map.values()

    def map(self):
        temp = self.code_map
        temp.update(self.exchange_map)
        return temp

    def list(self):
        return list(self.code_map.keys()) + list(self.exchange_map.keys())

class Loader:
    workspace_dir = os.path.expanduser("~/workspace")

    trade_dir = "%s/stocktrade" % workspace_dir
    base_dir = "%s/stock_data" % workspace_dir
    tick_dir = "%s/stock_tick" % workspace_dir
    stock_dir = "%s/stocks" % base_dir
    ranking_dir = "%s/ranking" % base_dir
    settings_dir = "%s/settings" % base_dir
    realtime_dir = "%s/realtime" % base_dir
    realtime_minutes_dir = "%s/realtime/minutes" % base_dir

    foreign_dir = "%s/foreign_stocks" % workspace_dir

    stock_agg = {"open":"first", "high":"max", "low":"min", "close":"last", "volume":"last"}
    index_agg = {"open":"first", "high":"max", "low":"min", "close":"last"}

    @staticmethod
    def resample(data, rule="D", agg=stock_agg):
        data = data.set_index("date")
        data = data.resample(rule)
        data = data.agg(agg)
        data = data.dropna()
        data = data.reset_index()
        return data

    @staticmethod
    def hold_realtime():
        try:
            data = pandas.read_csv("%s/hold_realtime.csv" % Loader.realtime_dir, header=None)
            data.columns = ["code", "high", "low", "price", "volume", "update_time"]
        except:
            data = None
        return data

    @staticmethod
    def format(data, data_type, columns=['date', 'open', 'high', 'low', 'close', "volume"], replace="-", how="any", date_format='%Y-%m-%d'):
        data.columns = columns
        data = data.replace(replace, numpy.nan)
        data = data.dropna(how=how)
        data['date'] = pandas.to_datetime(data['date'], format=date_format)
        for column in columns:
            if column in ["date", "update_time"]:
                continue
            data[column] = data[column].astype(data_type)
        data = data.reset_index(drop=True)
        return data

    @staticmethod
    def load_index(name, start_date=None, end_date=None, with_filter=True, strict=False, data_type=int):
        try:
            data = pandas.read_csv(Loader.base_dir + "/index/" + name + ".csv", header=None)
        except:
            data = None

        if data is None:
          return None

        data = Loader.format(data, data_type, columns=['date', 'open', 'low', 'high', 'close'])
        data["volume"] = 0
        if with_filter:
          filtered = Loader.filter(data, start_date, end_date, strict)
          if len(filtered) == 0:
            return None
          return filtered
        return data

    @staticmethod
    def load_index_all(start_date=None, end_date=None, with_filter=True, strict=False, data_type=int):
        index = Index()
        data = {}
        for key in index.list():
            data[key] = Loader.load_index(key, start_date, end_date, with_filter, strict, data_type)
        return data

    @staticmethod
    def load_tick(code, start_date, end_date, with_filter=True, strict=False, with_stats=False):
        start = utils.to_datetime(start_date)
        end = utils.to_datetime(end_date)
        current = start
        data = None
        while current <= end:
            month = current.strftime("%Y%m")
            current = utils.to_datetime(utils.to_format(current + relativedelta(months=1), output_format="%Y-%m-01"))
            try:
                data_ = pandas.read_csv(Loader.tick_dir + "/" + month + "/" + str(code) + ".csv", header=None)
                if with_stats:
                    stats = pandas.read_csv(Loader.tick_dir + "/" + month + "/" + str(code) + "_stats.csv")
                    stats = stats.drop("date", axis=1)
                    data_ = pandas.concat([data_, stats], axis=1)
            except:
                continue
            data = data_ if (data is None) else pandas.concat([data, data_])

        if data is None:
          return None
        data = Loader.tick_format(data)
        if with_filter:
          filtered = Loader.filter(data, "%s 000000" % start_date, "%s 235959" % end_date, strict)
          if len(filtered) == 0:
            return None
          return filtered
        return data

    @staticmethod
    def load_tick_ohlc(code, start_date, end_date):
        daily = Loader.load(code, start_date, end_date)
        columns = ["code", "date", "time", "price", "volume"]
        columns_dict = {}
        for column in columns:
            columns_dict[column] = []

        for i, row in daily.iterrows():
            for column, d in zip(columns, ["%s0" % code, utils.format(str(row["date"]), input_format="%Y-%m-%d %H:%M:%S", output_format="%Y%m%d"), 90000000000, row["open"], 0]):
                columns_dict[column].append(d)
            for column, d in zip(columns, ["%s0" % code, utils.format(str(row["date"]), input_format="%Y-%m-%d %H:%M:%S", output_format="%Y%m%d"), 150000000000, row["close"], None]):
                columns_dict[column].append(d)

        oc = pandas.DataFrame(columns_dict, columns=columns)
        oc = Loader.tick_format(oc)

        tick = Loader.load_tick(code, start_date, end_date)
        if tick is None:
            tick = oc
        else:
            tick = pandas.concat([tick, oc])

        tick = tick.set_index("date")
        ohlc = pandas.Series(tick["price"]).resample("5T").ohlc()
        volume = pandas.Series(tick["volume"]).resample("5T").sum()
        ohlc["volume"] = volume
        ohlc = ohlc.ffill()
        ohlc = ohlc.reset_index()
        ohlc = Loader.zaraba_filter(ohlc)
        return ohlc

    @staticmethod
    def tick_format(data):
        data.columns = ['code', 'date', 'time', 'price', "volume"]
        data["time"] = data["time"] / 1000
        data["time"] = data["time"].astype("int")
        data['date'] = data["date"].astype("str").str.cat(data["time"].astype("str"), sep=" ")
        data['date'] = pandas.to_datetime(data['date'], format='%Y%m%d %H%M%S%f')
        data = data.reset_index(drop=True)
        return data

    @staticmethod
    def years(start_date, end_date):
        start = 2007
        end = datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y")
        years = range(int(start), int(end)+1)
        return years

    @staticmethod
    def load_stats(code, start_date, end_date, with_filter=True, strict=False, with_stats=False):
        years = Loader.years(start_date, end_date)
        data = None
        for year in years:
          try:
            data_ = pandas.read_csv(Loader.stock_dir + '/' + str(code) + '/' + str(year) +  '_stats.csv')
          except:
            continue
          data = data_ if (data is None) else pandas.concat([data, data_])
        if data is None:
          return None
        if with_filter:
          filtered = Loader.filter(data, start_date, end_date, strict)
          if len(filtered) == 0:
            return None
          return filtered

    @staticmethod
    def load(code, start_date, end_date, with_filter=True, strict=False, with_stats=False):
        years = Loader.years(start_date, end_date)
        data = None
        for year in years:
          try:
            data_ = pandas.read_csv(Loader.stock_dir + '/' + str(code) + '/' + str(year) +  '.csv', header=None)
            data_ = data_.iloc[:,0:6]
            data_ = Loader.format(data_, "int")
            if with_stats:
                stats = pandas.read_csv(Loader.stock_dir + '/' + str(code) + '/' + str(year) +  '_stats.csv')
                stats = stats.drop("date", axis=1)
                data_ = pandas.concat([data_, stats], axis=1)
          except:
            continue
          data = data_ if (data is None) else pandas.concat([data, data_])
        if data is None:
          return None
        if with_filter:
          filtered = Loader.filter(data, start_date, end_date, strict)
          if len(filtered) == 0:
            return None
          return filtered
        return data

    @staticmethod
    def load_all(code):
        stock_brand_dir = "%s/%s" % (Loader.stock_dir, code)
        csvs = sorted(os.listdir(stock_brand_dir))
        data = None
        for csv in csvs:
           try:
               data_ = pandas.read_csv(stock_brand_dir+"/"+csv, header=None)
           except:
               continue
           data = data_ if (data is None) else pandas.concat([data, data_])
        if data is None:
           return None
        data = data.iloc[:,0:6]
        data = Loader.format(data, "int")
        return data

    @staticmethod
    def filter(data, start_date, end_date, strict=False):
        if strict:
            # start_date ~ end_date の間のデータが欠けている場合データ数が少なくなってしまうため除外する
            before = data[data["date"] <= start_date]
            after = data[end_date <= data["date"]]
            if len(before) == 0 or len(after) == 0:
                return []
        data = data[start_date <= data["date"]]
        data = data[data["date"] <= end_date]
        data = data.reset_index(drop=True)
        return data

    @staticmethod
    def load_by_length(code, start_date, end_date, data_length):
        data = Loader.load(code, start_date, end_date, with_filter=False)
        if data is None:
            return None
        filtered = Loader.filter(data, start_date, end_date)
        if len(filtered) == 0:
            return None
        data = data[data["date"] <= end_date]
        length = len(filtered) + data_length
        data = data[-length:]
        data = data.reset_index(drop=True)
        return data

    @staticmethod
    def loads(codes, start_date, end_date, strict=True, with_code=True, with_stats=False):
        data = {} if with_code else []
        for code in codes:
            d = Loader.load(code, start_date, end_date, strict=strict, with_stats=with_stats)
            if d is None:
                continue
            if with_code:
                data[code] = d
            else:
                data.append(d)
        print("load: %s" % len(data))
        return data

    @staticmethod
    def load_with_realtime(code, start_date, end_date, with_stats=False):
        if str(code).isdigit():
            data = Loader.load(code, start_date, end_date, with_stats=with_stats)
        else:
            data = Loader.load_index(code, start_date, end_date)

        d = data[data["date"] == end_date]

        # 当日分の日足があったらすぐ返す
        if len(d) > 0:
            return data

        realtime = Loader.load_realtime(code, end_date)
        if realtime is not None and with_stats == False:
            data = data[data["date"] < end_date]
            open_date = "%s 09:00:00" % end_date
            realtime = realtime[realtime["date"] >= open_date]
            d = {
                "date": pandas.to_datetime([end_date], format='%Y-%m-%d'),
                "open": [realtime["price"].iloc[0]],
                "high": [realtime["price"].max()],
                "low": [realtime["price"].min()],
                "close": [realtime["price"].iloc[-1]],
                "volume": [realtime["volume"].iloc[-1]]
            }
            d = pandas.DataFrame(d)
            data = pandas.concat([data, d])

        return data

    @staticmethod
    def realtime_monitor_codes(date):
        try:
            path = '%s/%s' % (Loader.realtime_dir, date)
            files = os.scandir(path)
            codes = list(map(lambda x: x.name.replace(".csv", ""), files))
            codes = list(filter(lambda x: str(x).isdigit(), codes)) # 指標系・その他は除外
            return codes
        except:
            return []

    @staticmethod
    def load_realtime(code, date, how="any"):
        try:
            if not str(code).isdigit():
                index = Index()
                code = index.map()[code]
                code = code.replace("/", "")
            path = '%s/%s/%s.csv' % (Loader.realtime_dir, date, code)
            data = pandas.read_csv(path, header=None)
            data = Loader.format(data, "float", replace=" ", how=how, columns=['date', 'high', 'low', 'price', 'volume', 'update_time'], date_format="%Y-%m-%d %H:%M:%S")
            data["volume"] = utils.diff(data["volume"].as_matrix() / 1000)
            data["volume"] = list(map(lambda x: 0 if x < 0 else x, data["volume"].as_matrix().tolist()))
        except:
#            import traceback
#            traceback.print_exc()
            data = None
        return data

    @staticmethod
    def load_realtime_ohlc(code, date):
        data = Loader.load_realtime(code, date)
        data = Loader.realtime_to_ohlc(data, with_zaraba_filter=True)
        return data

    @staticmethod
    def load_realtime_minutes(code, date):
        try:
            data = pandas.read_csv("%s/%s/%s.csv" % (Loader.realtime_minutes_dir, date, code))
            data.columns = ["date", "open", "high", "low", "close", "volume", "other"]
            data['date'] = pandas.to_datetime(data['date'], format="%Y%m%d%H%M")
            data = data[data['date'] >= "%s 00:00:00" % date]
            data = data[data['date'] <= "%s 23:59:59" % date]
            data = data.drop("other", axis=1)
            data["volume"] = data["volume"].as_matrix() / 1000
            data = data.dropna()
            data = data.reset_index()
        except:
            data = None
        return data

    @staticmethod
    def loads_realtime(code, date, days, how="any", time=None):
        # 日本の休日を除外
        data = Loader.load_realtime(code, date) # 今日の分
        data = Loader.realtime_to_ohlc(data)
        if time is not None:
            data = data[data["date"] <= "%s %s" % (date, time)]
        length = 1
        current = date
        while length < days:
            current = utils.to_format(utils.to_datetime(current) - relativedelta(days=1))
            if utils.is_weekday(utils.to_datetime(current)):
                d = Loader.load_realtime_minutes(code, current)
                if d is not None: 
                    d = Loader.resample(d, "5T")
                    data = pandas.concat([d, data])
                length = length + 1
        data = Loader.zaraba_filter(data)
        return data


    @staticmethod
    def realtime_to_ohlc(data, rule="5T", with_zaraba_filter=False):
        if with_zaraba_filter:
            data = Loader.zaraba_filter(data)
        data = data.set_index("date")
        volume = pandas.Series(data["volume"]).resample(rule).sum()
        update_time = pandas.Series(data["update_time"]).resample(rule).last()
        data = pandas.Series(data["price"]).resample(rule).ohlc()
        data = data.rename(columns={"price": "close"})
        data["volume"] = volume
        data["update_time"] = update_time
        data = data.dropna()
        data = data.reset_index()
        return data

    @staticmethod
    def zaraba_filter(data):
        data = data[
            (data["date"].dt.hour >= 9) & (data["date"].dt.hour <= 10) |
            (data["date"].dt.hour >= 13) & (data["date"].dt.hour <= 14) |
            (data["date"].dt.hour == 11) & (data["date"].dt.minute <= 30) |
            (data["date"].dt.hour == 12) & (data["date"].dt.minute >= 30) |
            (data["date"].dt.hour == 8) & (data["date"].dt.minute == 59)
        ]
        data = data.reset_index()
        return data

    @staticmethod
    def stocks():
        try:
            data = pandas.read_csv('settings/stocks.csv', header=None)
            data.columns = ['code']
        except:
            data = None
        return data

    @staticmethod
    def monitor_stocks():
        try:
            data = pandas.read_csv('settings/monitor_stocks.csv', header=None)
            data.columns = ['code']
        except:
            data = None
        return data

    @staticmethod
    def high_performance_stocks():
        try:
            data = pandas.read_csv('settings/high_performance_stocks.csv', header=None)
            data.columns = ['code']
        except:
            data = None
        return data

    @staticmethod
    def low_performance_stocks():
        try:
            data = pandas.read_csv('settings/low_performance_stocks.csv', header=None)
            data.columns = ['code']
        except:
            data = None
        return data

    @staticmethod
    def middle_stocks():
        try:
            data = pandas.read_csv('settings/middle_stocks.csv', header=None)
            data.columns = ['code']
        except:
            data = None
        return data

    @staticmethod
    def large_stocks():
        try:
            data = pandas.read_csv('settings/large_stocks.csv', header=None)
            data.columns = ['code']
        except:
            data = None
        return data

    @staticmethod
    def steady_trend_stocks(start_date, end_date, filename="rising_stocks.csv"):
        current = start_date
        data = {"all": []}
        output_format = "%Y%m"
        while int(utils.format(current, output_format=output_format)) <= int(utils.format(end_date, output_format=output_format)):
            try:
                months = utils.format(current, output_format=output_format)
                d = pandas.read_csv("%s/steady_trend_stocks/%s/%s" % (Loader.settings_dir, months, filename), header=None)
                d.columns = ['code']
                data[months] = d["code"].as_matrix().tolist()
                data["all"] = list(set(data["all"] + data[months]))
            except:
                continue
            finally:
                current = utils.to_format(utils.to_datetime(current) + utils.relativeterm(1))

        return data

    @staticmethod
    def rising_stocks():
        try:
            data = pandas.read_csv('settings/rising_stocks.csv', header=None)
            data.columns = ['code']
        except:
            data = None
        return data

    @staticmethod
    def fall_stocks():
        try:
            data = pandas.read_csv('settings/fall_stocks.csv', header=None)
            data.columns = ['code']
        except:
            data = None

        return data

    @staticmethod
    def attention_stocks():
        try:
            data = pandas.read_csv('settings/attention_stocks.csv', header=None)
            data.columns = ['code']
        except:
            data = None
        return data

    @staticmethod
    def high_attention_stocks():
        try:
            data = pandas.read_csv('settings/high_attention_stocks.csv', header=None)
            data.columns = ['code']
        except:
            data = None
        return data

    @staticmethod
    def fall_attention_stocks():
        try:
            data = pandas.read_csv('settings/fall_attention_stocks.csv', header=None)
            data.columns = ['code']
        except:
            data = None
        return data

    @staticmethod
    def fall_high_performance_attention_stocks():
        try:
            data = pandas.read_csv('settings/fall_high_performance_attention_stocks.csv', header=None)
            data.columns = ['code']
        except:
            data = None
        return data

    @staticmethod
    def simulate_setting(filename=None):
        filename = "simulate_setting.json" if filename is None else filename
        try:
            f = open("settings/%s" % (filename), "r")
            data = json.load(f)
        except:
            data = None
        return data

    @staticmethod
    def hold_stocks():
        try:
            data = pandas.read_csv('settings/hold_stocks.csv', header=None)
            data.columns = ['code', 'order', 'price', 'term', 'initial', 'strategy', 'system', 'method']
        except:
            data = None
        return data

    @staticmethod
    def new_high_stocks():
        try:
            data = pandas.read_csv('settings/new_high.csv', header=None)
            data.columns = ['code', 'high']
        except:
            data = None
        return data

    @staticmethod
    def new_score(date, filename):
        try:
            data = pandas.read_csv("%s/new_score/%s/%s" % (Loader.settings_dir, date, filename), header=None)
            data.columns = ['code', 'price']
        except:
            data = None
        return data

    @staticmethod
    def new_high_all_stocks():
        try:
            data = pandas.read_csv('settings/new_high_all.txt', header=None)
            data.columns = ['code', 'high']
        except:
            data = None
        return data

    @staticmethod
    def attention_new_high_stocks():
        try:
            data = pandas.read_csv('settings/attention_new_high_stocks.csv', header=None)
            data.columns = ['code']
        except:
            data = None
        return data

    @staticmethod
    def market_trend():
        try:
            data = pandas.read_csv('settings/market_trend.csv', header=None)
            data.columns = ['rising', 'fall']
        except:
            data = None
        return data

    @staticmethod
    def index_trend():
        try:
            data = pandas.read_csv('settings/index_trend.csv', header=None)
            data.columns = ['rising', 'fall']
        except:
            data = None
        return data

    @staticmethod
    def closing_setting():
        try:
            f = open("settings/closing_setting.json", "r")
            data = json.load(f)
        except:
            data = None
        return data

    @staticmethod
    def assets():
        try:
            f = open("settings/assets.json", "r")
            data = json.load(f)
        except:
            data = None
        return data

    @staticmethod
    def assets_history():
        try:
            data = pandas.read_csv('settings/assets_history.csv', header=None)
            data.columns = ['date', 'assets']
            data['date'] = pandas.to_datetime(data['date'], format='%Y-%m-%d')
            data.sort_values("date")
        except:
            data = None
        return data

    @staticmethod
    def ordered_stocks():
        try:
            data = pandas.read_csv("settings/ordered_stocks.csv", header=None)
            data.columns = ["code", "trade", "system"]
            return data
        except:
            return None

    @staticmethod
    def realtime_sheet_stocks(date, monitor_size=3):
        codes = []
        stocks = [
            Loader.ordered_stocks(),
            Loader.hold_stocks(),
        ]
        for stock in stocks:
            if stock is not None:
                codes.extend(stock["code"].as_matrix().tolist())

        codes = codes + Loader.before_ranking_codes(date, "volume", before=0, monitor_size=monitor_size)

        print(codes)
        return list(set(codes))

    @staticmethod
    def before_ranking(date, ranking_type, before=1):
        d = utils.to_datetime(date) - utils.relativeterm(before, tick=True)
        while not utils.is_weekday(d):
            d = d - utils.relativeterm(1, tick=True)
        d = utils.to_format(d)
        stocks = Loader.ranking(d, ranking_type)
        return stocks

    @staticmethod
    def before_ranking_codes(date, ranking_type, before=1, monitor_size=3):
        stocks = Loader.before_ranking(date, ranking_type, before=before)
        if stocks is None:
            return []
        codes = stocks["code"].iloc[:monitor_size].as_matrix().tolist()
        return codes

    @staticmethod
    def kabuplus_stock_data():
        try:
            data = pandas.read_csv("%s/kabuplus/japan-all-stock-data.csv" % Loader.base_dir, encoding="SHIFT-JIS", header=None)
            data = data.iloc[1:]
            return data
        except:
            return None

    @staticmethod
    def kabuplus_financial_data():
        try:
            data = pandas.read_csv("%s/kabuplus/japan-all-stock-financial-results.csv" % Loader.base_dir, encoding="SHIFT-JIS", header=None)
            data = data.iloc[1:]
            return data
        except:
            return None

    @staticmethod
    def stock_industry_code():
        try:
            data = pandas.read_csv("settings/stock_industry_code_map.csv", header=None)
            data.columns = ["code", "industry_code"]
            return data
        except:
            return None

    @staticmethod
    def industry_code():
         try:
            data = pandas.read_csv("settings/industry_map.csv", header=None)
            data.columns = ["name", "industry_code"]
            return data
         except:
            return None

    @staticmethod
    def load_industry_index(code):
        try:
            data = pandas.read_csv("%s/index/industry/%s.csv" % (Loader.base_dir, code), header=None)
            data.columns = ["date", "value", "volume"]
            data['date'] = pandas.to_datetime(data['date'], format='%Y-%m-%d')
            return data.sort_values(by=["date"])
        except:
            return None

    @staticmethod
    def stock_split():
        try:
            data = pandas.read_csv("settings/stock_split.csv", header=None)
            data.columns = ["date", "code"]
            data['date'] = pandas.to_datetime(data['date'], format='%Y-%m-%d')
            return data.sort_values(by=["date"])
        except:
            return None

    @staticmethod
    def reverse_stock_split():
        try:
            data = pandas.read_csv("settings/reverse_stock_split.csv", header=None)
            data.columns = ["date", "code"]
            data['date'] = pandas.to_datetime(data['date'], format='%Y-%m-%d')
            return data.sort_values(by=["date"])
        except:
            return None

    @staticmethod
    def ranking(date, ranking_type):
        d = utils.format(date)
        try:
            data = pandas.read_csv("%s/%s/%s.csv" % (Loader.ranking_dir, ranking_type, d), header=None)
            data.columns = ["code", "price", "key"]
            return data
        except:
            return None

    @staticmethod
    def eodhistoricaldata_api_token():
        try:
            f = open("settings/eodhistoricaldata.txt", "r")
            data = json.load(f)
            return data["token"]
        except:
            data = None

    @staticmethod
    def five_times():
        try:
            f = open("settings/five_times.json", "r")
            data = json.load(f)
            return data["data"]
        except:
           import traceback
           traceback.print_exc()
           data = None

