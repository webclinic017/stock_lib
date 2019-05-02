# -*- coding: utf-8 -*-


def average_conditions():
    columns = ["daily_average", "weekly_average"]
    targets = ["open", "high", "low", "close"]

    conditions = []
    for column in columns:
        for target in targets:
            conditions = conditions + [
                lambda d: d.data.daily[column].iloc[-1] > d.data.daily[target].iloc[-1],
                lambda d: d.data.daily[column].iloc[-1] < d.data.daily[target].iloc[-1],
            ]

    return conditions

def tec_conditions():
    columns = ["rci", "macd"]
    conditions = []
    rci = [
        lambda d: d.data.daily["rci"].iloc[-1] > 80,
        lambda d: d.data.daily["rci"].iloc[-1] < 80,
        lambda d: d.data.daily["rci"].iloc[-1] < -80,
        lambda d: d.data.daily["rci"].iloc[-1] > -80,
        lambda d: d.data.daily["rci_long"].iloc[-1] > 80,
        lambda d: d.data.daily["rci_long"].iloc[-1] < 80,
        lambda d: d.data.daily["rci_long"].iloc[-1] < -80,
        lambda d: d.data.daily["rci_long"].iloc[-1] > -80,
        lambda d: d.data.daily["rci"].iloc[-1] > d.data.daily["rci_long"].iloc[-1],
        lambda d: d.data.daily["rci"].iloc[-1] < d.data.daily["rci_long"].iloc[-1]
    ]
    macd = [
        lambda d: d.data.daily["macd"].iloc[-1] > 0,
        lambda d: d.data.daily["macd"].iloc[-1] < 0,
        lambda d: d.data.daily["macdsignal"].iloc[-1] > 0,
        lambda d: d.data.daily["macdsignal"].iloc[-1] < 0,
        lambda d: d.data.daily["macd"].iloc[-1] > d.data.daily["macdsignal"].iloc[-1],
        lambda d: d.data.daily["macd"].iloc[-1] < d.data.daily["macdsignal"].iloc[-1],
        lambda d: d.data.daily["macdhist"].iloc[-1] > 0,
        lambda d: d.data.daily["macdhist"].iloc[-1] < 0,
    ]
    conditions = conditions + rci if "rci" in columns else conditions
    conditions = conditions + macd if "macd" in columns else conditions
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
    columns = [
        "average_cross", "macd_cross", "rci_cross", "env12_cross", "env11_cross", "env09_cross", "env08_cross"
    ]

    conditions = []
    for column in columns:
        conditions = conditions + [
            lambda d: d.data.daily[column].iloc[-1] == 1, 
            lambda d: d.data.daily[column].iloc[-1] == -1,
        ]

    return conditions

def trend_conditions():
    columns = [
        "daily_average_trend", "weekly_average_trend", "volume_average_trend", "macd_trend", "macdhist_trend",
        "rci_trend", "rci_long_trend", "stages_trend", "stages_average_trend", "rising_safety_trend", "fall_safety_trend"
    ]

    conditions = []
    for column in columns:
        conditions = conditions + [
            lambda d: d.data.daily[column].iloc[-1] == 1,
            lambda d: d.data.daily[column].iloc[-1] == -1,
        ]

    return conditions

def all():
    return average_conditions() + tec_conditions() + cross_conditions() + trend_conditions()

