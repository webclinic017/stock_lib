# -*- coding: utf-8 -*-

def average_conditions():
    legs = ["daily", "weekly"]
    columns = ["daily_average", "weekly_average"]
    targets = ["open", "high", "low", "close"]

    conditions = []
    for leg in legs:
        for column in columns:
            for target in targets:
                conditions = conditions + [
                    lambda d, s: d.data[leg][columns].iloc[-1] > d.data[leg][target].iloc[-1], 
                    lambda d, s: d.data[leg][columns].iloc[-1] < d.data[leg][target].iloc[-1],
                ]

    return conditions

def tec_conditions():
    legs = ["daily", "weekly"]
    conditions = []
    for leg in legs:
        conditions = conditions + [
            lambda d, s: d.data["daily"]["rci"].iloc[-1] > 80,
            lambda d, s: d.data["daily"]["rci"].iloc[-1] < 80,
            lambda d, s: d.data["daily"]["rci"].iloc[-1] < -80,
            lambda d, s: d.data["daily"]["rci"].iloc[-1] > -80,
            lambda d, s: d.data["daily"]["rci_long"].iloc[-1] > 80,
            lambda d, s: d.data["daily"]["rci_long"].iloc[-1] < 80,
            lambda d, s: d.data["daily"]["rci_long"].iloc[-1] < -80,
            lambda d, s: d.data["daily"]["rci_long"].iloc[-1] > -80,
            lambda d, s: d.data["daily"]["rci"].iloc[-1] > d.data["daily"]["rci_long"].iloc[-1],
            lambda d, s: d.data["daily"]["rci"].iloc[-1] < d.data["daily"]["rci_long"].iloc[-1],
            lambda d, s: d.data["daily"]["macd"].iloc[-1] > 0,
            lambda d, s: d.data["daily"]["macd"].iloc[-1] < 0,
            lambda d, s: d.data["daily"]["macd_signal"].iloc[-1] > 0,
            lambda d, s: d.data["daily"]["macd_signal"].iloc[-1] < 0,
            lambda d, s: d.data["daily"]["macd"].iloc[-1] > d.data["daily"]["macdsignal"].iloc[-1],
            lambda d, s: d.data["daily"]["macd"].iloc[-1] < d.data["daily"]["macdsignal"].iloc[-1],
        ]
    return conditions

def band_conditions():
    return [
    ]

def safety_conditions():
    return [
    ]

def stages_conditions():
    return [
    ]

def cross_conditions():
    legs = ["daily", "weekly"]
    columns = [
        "average_cross"
        "macd_cross"
        "rci_cross"
        "env12_cross"
        "env11_cross"
        "env09_cross"
        "env08_cross"
    ]

    conditions = []
    for leg in legs:
        for column in columns:
            conditions = conditions + [
                lambda d, s: d.data[leg][columns].iloc[-1] == 1, 
                lambda d, s: d.data[leg][columns].iloc[-1] == -1,
            ]

    return conditions

def trend_conditions():
    legs = ["daily", "weekly"]
    columns = [
        "daily_average_trend", "weekly_average_trend", "volume_average_trend", "macd_trend", "macdhist_trend",
        "rci_trend", "rci_long_trend", "stages_trend", "stages_average_trend", "rising_safety_trend", "fall_safety_trend"
    ]

    conditions = []
    for leg in legs:
        for column in columns:
            conditions = conditions + [
                lambda d, s: d.data[leg][columns].iloc[-1] == 1, 
                lambda d, s: d.data[leg][columns].iloc[-1] == -1,
            ]

    return conditions
