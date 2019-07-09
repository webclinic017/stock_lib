# -*- coding: utf-8 -*-

def select(data, target="daily"):
    if target == "daily":
        return data.data.daily
    return data.data.daily

def average_conditions():
    legs = ["daily"]
    columns = ["daily_average", "weekly_average"]
    targets = ["open", "high", "low", "close"]

    conditions = []
    for leg in legs:
        for column in columns:
            for target in targets:
                conditions = conditions + [
                    lambda d: select(d, leg)[column].iloc[-1] > select(d, leg)[target].iloc[-1],
                    lambda d: select(d, leg)[column].iloc[-1] < select(d, leg)[target].iloc[-1],
                ]

        conditions = conditions + [
            lambda d: select(d, leg)["volume"].iloc[-1] > select(d, leg)["volume_average"].iloc[-1],
            lambda d: select(d, leg)["volume"].iloc[-1] < select(d, leg)["volume_average"].iloc[-1],
        ]

    return conditions

def tec_conditions():
    legs = ["daily"]
    columns = ["rci", "macd"]
    conditions = []
    for leg in legs:
        rci = [
            lambda d: select(d, leg)["rci"].iloc[-1] > 80,
            lambda d: select(d, leg)["rci"].iloc[-1] < 80,
            lambda d: select(d, leg)["rci"].iloc[-1] < -80,
            lambda d: select(d, leg)["rci"].iloc[-1] > -80,
            lambda d: select(d, leg)["rci_long"].iloc[-1] > 80,
            lambda d: select(d, leg)["rci_long"].iloc[-1] < 80,
            lambda d: select(d, leg)["rci_long"].iloc[-1] < -80,
            lambda d: select(d, leg)["rci_long"].iloc[-1] > -80,
            lambda d: select(d, leg)["rci"].iloc[-1] > select(d, leg)["rci_long"].iloc[-1],
            lambda d: select(d, leg)["rci"].iloc[-1] < select(d, leg)["rci_long"].iloc[-1]
        ]
        macd = [
            lambda d: select(d, leg)["macd"].iloc[-1] > 0,
            lambda d: select(d, leg)["macd"].iloc[-1] < 0,
            lambda d: select(d, leg)["macdsignal"].iloc[-1] > 0,
            lambda d: select(d, leg)["macdsignal"].iloc[-1] < 0,
            lambda d: select(d, leg)["macd"].iloc[-1] > select(d, leg)["macdsignal"].iloc[-1],
            lambda d: select(d, leg)["macd"].iloc[-1] < select(d, leg)["macdsignal"].iloc[-1],
            lambda d: select(d, leg)["macdhist"].iloc[-1] > 0,
            lambda d: select(d, leg)["macdhist"].iloc[-1] < 0,
            lambda d: select(d, leg)["macdhist_convert"].iloc[-1] == 1,
            lambda d: select(d, leg)["macdhist_convert"].iloc[-1] == 0,
            lambda d: select(d, leg)["macdhist_convert"].iloc[-1] == -1,
        ]
        conditions = conditions + rci if "rci" in columns else conditions
        conditions = conditions + macd if "macd" in columns else conditions
    return conditions

def band_conditions():
    legs = ["daily"]
    conditions = []
    for leg in legs:
        conditions = conditions + [
            lambda d: select(d, leg)["env_entity"].iloc[-1] < select(d, leg)["env_entity_average"].iloc[-1],
            lambda d: select(d, leg)["env_entity"].iloc[-1] > select(d, leg)["env_entity_average"].iloc[-1],
        ]
    return conditions

def safety_conditions():
    return [
    ]

def stages_conditions():
    legs = ["daily"]
    conditions = []
    for leg in legs:
        conditions = conditions + [
            lambda d: select(d, leg)["stages"].iloc[-1] == -2,
            lambda d: select(d, leg)["stages"].iloc[-1] == -1,
            lambda d: select(d, leg)["stages"].iloc[-1] == 0,
            lambda d: select(d, leg)["stages"].iloc[-1] == 1,
            lambda d: select(d, leg)["stages"].iloc[-1] == 2,
            lambda d: select(d, leg)["stages"].iloc[-1] < select(d, leg)["stages_average"].iloc[-1],
            lambda d: select(d, leg)["stages"].iloc[-1] > select(d, leg)["stages_average"].iloc[-1],
            lambda d: select(d, leg)["stages"].iloc[-1] > 0,
            lambda d: select(d, leg)["stages"].iloc[-1] < 0,
            lambda d: select(d, leg)["stages_average"].iloc[-1] > 0,
            lambda d: select(d, leg)["stages_average"].iloc[-1] < 0,
        ]
    return conditions

def cross_conditions():
    legs = ["daily"]
    columns = [
        "average_cross", "macd_cross", "rci_cross", "env12_cross", "env11_cross", "env09_cross", "env08_cross"
    ]

    conditions = []
    for leg in legs:
        for column in columns:
            conditions = conditions + [
                lambda d: select(d, leg)[column].iloc[-1] == 1,
                lambda d: select(d, leg)[column].iloc[-1] == 0,
                lambda d: select(d, leg)[column].iloc[-1] == -1,
            ]

    return conditions

def trend_conditions():
    legs = ["daily"]
    columns = [
        "daily_average_trend", "weekly_average_trend", "volume_average_trend", "macd_trend", "macdhist_trend",
        "rci_trend", "rci_long_trend", "stages_trend", "stages_average_trend", "rising_safety_trend", "fall_safety_trend"
    ]

    conditions = []
    for leg in legs:
        for column in columns:
            conditions = conditions + [
                lambda d: select(d, leg)[column].iloc[-1] == 1,
                lambda d: select(d, leg)[column].iloc[-1] == 0,
                lambda d: select(d, leg)[column].iloc[-1] == -1,
            ]

    return conditions

def cs_conditions():
    legs = ["daily"]
    columns = [
        "yang_tsutsumi", "yang_harami", "lower_kenuki", "ake_mojo", "yin_sanku", "yin_sanpei",
        "yin_tsutsumi", "yin_harami", "upper_kenuki", "yoi_mojo", "yang_sanku", "yang_sanpei",
        "long_upper_shadow", "long_lower_shadow", "yang", "yin", "long_yang", "long_yin", "low_roundup",
        "high_roundup", "low_rounddown", "high_rounddown", "yang_gap", "yin_gap", "gap", "harami", "tsutsumi"
    ]

    conditions = []
    for leg in legs:
        for column in columns:
            conditions = conditions + [
                lambda d: select(d, leg)[column].iloc[-1] == 1,
                lambda d: select(d, leg)[column].iloc[-1] == 0,
            ]

        conditions = conditions + [
            lambda d: select(d, leg)["entity"].iloc[-1] < select(d, leg)["entity_average"].iloc[-1],
            lambda d: select(d, leg)["entity"].iloc[-1] > select(d, leg)["entity_average"].iloc[-1],
        ]

    return conditions

def all():
    return average_conditions() + tec_conditions() + cross_conditions() + trend_conditions() + band_conditions() + stages_conditions() + cs_conditions()

