# -*- coding: utf-8 -*-

def selectable_data():
    return {
        "middle": lambda d: d.data.middle,
        "short": lambda d: d.data.short,
        "nikkei": lambda d: d.index.data["nikkei"].middle,
        "dow": lambda d: d.index.data["dow"].middle,
        "new_score": lambda d: d.index.data["new_score"].middle,
        "industry_score": lambda d: d.index.data["industry_score"].middle
#        "usdjpy": data.index.data["usdjpy"].middle,
#        "xbtusd": data.index.data["xbtusd"].middle
    }

def select(data, target="middle"):

    d = selectable_data()

    if target in d.keys():
        return d[target](data)

    raise Exception("unselectable: %s" % target)

def average_conditions(legs=["middle"]):
    columns = ["average", "long_average"]
    targets = ["open", "high", "low", "close"]

    conditions = []
    for leg in legs:
        for column in columns:
            for target in targets:
                conditions = conditions + [
                    lambda d, leg=leg, column=column, target=target: select(d, leg)[column].iloc[-1] > select(d, leg)[target].iloc[-1],
                    lambda d, leg=leg, column=column, target=target: select(d, leg)[column].iloc[-1] < select(d, leg)[target].iloc[-1],
                ]

        conditions = conditions + [
            lambda d, leg=leg: select(d, leg)["volume"].iloc[-1] > select(d, leg)["volume_average"].iloc[-1],
            lambda d, leg=leg: select(d, leg)["volume"].iloc[-1] < select(d, leg)["volume_average"].iloc[-1],
        ]

    return conditions

def tec_conditions(legs=["middle"]):
    columns = ["rci", "macd"]
    conditions = []
    for leg in legs:
        rci = [
            lambda d, leg=leg: select(d, leg)["rci"].iloc[-1] > 80,
            lambda d, leg=leg: select(d, leg)["rci"].iloc[-1] < 80,
            lambda d, leg=leg: select(d, leg)["rci"].iloc[-1] < -80,
            lambda d, leg=leg: select(d, leg)["rci"].iloc[-1] > -80,
            lambda d, leg=leg: select(d, leg)["rci_long"].iloc[-1] > 80,
            lambda d, leg=leg: select(d, leg)["rci_long"].iloc[-1] < 80,
            lambda d, leg=leg: select(d, leg)["rci_long"].iloc[-1] < -80,
            lambda d, leg=leg: select(d, leg)["rci_long"].iloc[-1] > -80,
            lambda d, leg=leg: select(d, leg)["rci"].iloc[-1] > select(d, leg)["rci_long"].iloc[-1],
            lambda d, leg=leg: select(d, leg)["rci"].iloc[-1] < select(d, leg)["rci_long"].iloc[-1]
        ]
        macd = [
            lambda d, leg=leg: select(d, leg)["macd"].iloc[-1] > 0,
            lambda d, leg=leg: select(d, leg)["macd"].iloc[-1] < 0,
            lambda d, leg=leg: select(d, leg)["macdsignal"].iloc[-1] > 0,
            lambda d, leg=leg: select(d, leg)["macdsignal"].iloc[-1] < 0,
            lambda d, leg=leg: select(d, leg)["macd"].iloc[-1] > select(d, leg)["macdsignal"].iloc[-1],
            lambda d, leg=leg: select(d, leg)["macd"].iloc[-1] < select(d, leg)["macdsignal"].iloc[-1],
            lambda d, leg=leg: select(d, leg)["macdhist"].iloc[-1] > 0,
            lambda d, leg=leg: select(d, leg)["macdhist"].iloc[-1] < 0,
            lambda d, leg=leg: select(d, leg)["macdhist_convert"].iloc[-1] == 1,
            lambda d, leg=leg: select(d, leg)["macdhist_convert"].iloc[-1] == 0,
            lambda d, leg=leg: select(d, leg)["macdhist_convert"].iloc[-1] == -1,
        ]
        conditions = (conditions + rci) if "rci" in columns else conditions
        conditions = (conditions + macd) if "macd" in columns else conditions
    return conditions

def band_conditions(legs=["middle"]):
    conditions = []
    for leg in legs:
        conditions = conditions + [
            lambda d, leg=leg: select(d, leg)["env_entity"].iloc[-1] < select(d, leg)["env_entity_average"].iloc[-1],
            lambda d, leg=leg: select(d, leg)["env_entity"].iloc[-1] > select(d, leg)["env_entity_average"].iloc[-1],
        ]
    return conditions

def safety_conditions(legs=["middle"]):
    return [
    ]

def stages_conditions(legs=["middle"]):
    conditions = []
    for leg in legs:
        conditions = conditions + [
            lambda d, leg=leg: select(d, leg)["stages"].iloc[-1] == -2,
            lambda d, leg=leg: select(d, leg)["stages"].iloc[-1] == -1,
            lambda d, leg=leg: select(d, leg)["stages"].iloc[-1] == 0,
            lambda d, leg=leg: select(d, leg)["stages"].iloc[-1] == 1,
            lambda d, leg=leg: select(d, leg)["stages"].iloc[-1] == 2,
            lambda d, leg=leg: select(d, leg)["stages"].iloc[-1] < select(d, leg)["stages_average"].iloc[-1],
            lambda d, leg=leg: select(d, leg)["stages"].iloc[-1] > select(d, leg)["stages_average"].iloc[-1],
            lambda d, leg=leg: select(d, leg)["stages"].iloc[-1] > 0,
            lambda d, leg=leg: select(d, leg)["stages"].iloc[-1] < 0,
            lambda d, leg=leg: select(d, leg)["stages_average"].iloc[-1] > 0,
            lambda d, leg=leg: select(d, leg)["stages_average"].iloc[-1] < 0,
            lambda d, leg=leg: select(d, leg)["stages"][-3:].min() > 0,
            lambda d, leg=leg: select(d, leg)["stages"][-3:].min() < 0,
            lambda d, leg=leg: select(d, leg)["stages"][-3:].max() > 0,
            lambda d, leg=leg: select(d, leg)["stages"][-3:].max() < 0,
            lambda d, leg=leg: select(d, leg)["stages_average"][-3:].min() > 0,
            lambda d, leg=leg: select(d, leg)["stages_average"][-3:].min() < 0,
            lambda d, leg=leg: select(d, leg)["stages_average"][-3:].max() > 0,
            lambda d, leg=leg: select(d, leg)["stages_average"][-3:].max() < 0,
        ]
    return conditions

def stages_sum_conditions(legs=["middle"]):
    conditions = []
    for leg in legs:
        conditions = conditions + [
            lambda d, leg=leg: select(d, leg)["stages_sum"].iloc[-1] <= -12,
            lambda d, leg=leg: select(d, leg)["stages_sum"].iloc[-1] <= -10,
            lambda d, leg=leg: select(d, leg)["stages_sum"].iloc[-1] <= -8,
            lambda d, leg=leg: select(d, leg)["stages_sum"].iloc[-1] <= -6,
            lambda d, leg=leg: select(d, leg)["stages_sum"].iloc[-1] <= -4,
            lambda d, leg=leg: select(d, leg)["stages_sum"].iloc[-1] <= -2,
            lambda d, leg=leg: select(d, leg)["stages_sum"].iloc[-1] <= -1,
            lambda d, leg=leg: select(d, leg)["stages_sum"].iloc[-1] == 0,
            lambda d, leg=leg: select(d, leg)["stages_sum"].iloc[-1] >= 1,
            lambda d, leg=leg: select(d, leg)["stages_sum"].iloc[-1] >= 2,
            lambda d, leg=leg: select(d, leg)["stages_sum"].iloc[-1] >= 4,
            lambda d, leg=leg: select(d, leg)["stages_sum"].iloc[-1] >= 6,
            lambda d, leg=leg: select(d, leg)["stages_sum"].iloc[-1] >= 8,
            lambda d, leg=leg: select(d, leg)["stages_sum"].iloc[-1] >= 10,
            lambda d, leg=leg: select(d, leg)["stages_sum"].iloc[-1] >= 12,
            lambda d, leg=leg: select(d, leg)["stages_sum"].iloc[-1] < select(d, leg)["stages_sum_average"].iloc[-1],
            lambda d, leg=leg: select(d, leg)["stages_sum"].iloc[-1] > select(d, leg)["stages_sum_average"].iloc[-1],
            lambda d, leg=leg: select(d, leg)["stages_sum_average"].iloc[-1] < select(d, leg)["stages_sum_average_long"].iloc[-1],
            lambda d, leg=leg: select(d, leg)["stages_sum_average"].iloc[-1] > select(d, leg)["stages_sum_average_long"].iloc[-1],
            lambda d, leg=leg: select(d, leg)["stages_sum"].iloc[-1] > 0,
            lambda d, leg=leg: select(d, leg)["stages_sum"].iloc[-1] < 0,
            lambda d, leg=leg: select(d, leg)["stages_sum_average"].iloc[-1] > 0,
            lambda d, leg=leg: select(d, leg)["stages_sum_average"].iloc[-1] < 0,
            lambda d, leg=leg: select(d, leg)["stages_sum_average_long"].iloc[-1] > 0,
            lambda d, leg=leg: select(d, leg)["stages_sum_average_long"].iloc[-1] < 0,
        ]
    return conditions

def cross_conditions(legs=["middle"], additional_columns=[]):
    columns = [
        "average_cross", "macd_cross", "rci_cross", "env12_cross", "env11_cross", "env09_cross", "env08_cross", "rising_safety_cross", "fall_safety_cross"
    ] + additional_columns

    conditions = []
    for leg in legs:
        for column in columns:
            conditions = conditions + [
                lambda d, leg=leg, column=column: select(d, leg)[column].iloc[-1] == 1,
                lambda d, leg=leg, column=column: select(d, leg)[column].iloc[-1] == 0,
                lambda d, leg=leg, column=column: select(d, leg)[column].iloc[-1] == -1,
                lambda d, leg=leg, column=column: select(d, leg)[column][-3:].max() == 1,
                lambda d, leg=leg, column=column: select(d, leg)[column][-3:].max() == 0,
                lambda d, leg=leg, column=column: select(d, leg)[column][-3:].max() == -1,
                lambda d, leg=leg, column=column: select(d, leg)[column][-3:].min() == 1,
                lambda d, leg=leg, column=column: select(d, leg)[column][-3:].min() == 0,
                lambda d, leg=leg, column=column: select(d, leg)[column][-3:].min() == -1,
            ]

    return conditions

def trend_conditions(legs=["middle"], additional_columns=[]):
    columns = [
        "average_trend", "long_average_trend", "volume_average_trend", "macd_trend", "macdhist_trend",
        "rci_trend", "rci_long_trend", "stages_trend", "stages_average_trend", "rising_safety_trend", "fall_safety_trend"
    ] + additional_columns

    conditions = []
    for leg in legs:
        for column in columns:
            conditions = conditions + [
                lambda d, leg=leg, column=column: select(d, leg)[column].iloc[-1] == 1,
                lambda d, leg=leg, column=column: select(d, leg)[column].iloc[-1] == 0,
                lambda d, leg=leg, column=column: select(d, leg)[column].iloc[-1] == -1,
                lambda d, leg=leg, column=column: select(d, leg)[column][-3:].max() == 1,
                lambda d, leg=leg, column=column: select(d, leg)[column][-3:].max() == 0,
                lambda d, leg=leg, column=column: select(d, leg)[column][-3:].max() == -1,
                lambda d, leg=leg, column=column: select(d, leg)[column][-3:].min() == 1,
                lambda d, leg=leg, column=column: select(d, leg)[column][-3:].min() == 0,
                lambda d, leg=leg, column=column: select(d, leg)[column][-3:].min() == -1,
            ]

    return conditions

def cs_conditions(legs=["middle"], additional_columns=[]):
    columns = [
        "yang_tsutsumi", "yang_harami", "lower_kenuki", "ake_mojo", "yin_sanku", "yin_sanpei",
        "yin_tsutsumi", "yin_harami", "upper_kenuki", "yoi_mojo", "yang_sanku", "yang_sanpei",
        "long_upper_shadow", "long_lower_shadow", "yang", "yin", "long_yang", "long_yin", "low_roundup",
        "high_roundup", "low_rounddown", "high_rounddown", "yang_gap", "yin_gap", "gap", "harami", "tsutsumi"
    ] + additional_columns

    conditions = []
    for leg in legs:
        for column in columns:
            conditions = conditions + [
                lambda d, leg=leg, column=column: select(d, leg)[column].iloc[-1] == 1,
                lambda d, leg=leg, column=column: select(d, leg)[column].iloc[-1] == 0,
                lambda d, leg=leg, column=column: select(d, leg)[column][-3:].max() == 1,
            ]

        conditions = conditions + [
            lambda d, leg=leg: select(d, leg)["entity"].iloc[-1] < select(d, leg)["entity_average"].iloc[-1],
            lambda d, leg=leg: select(d, leg)["entity"].iloc[-1] > select(d, leg)["entity_average"].iloc[-1],
        ]

    return conditions

def new_score_conditions():
    conditions = [
        lambda d: select(d, "new_score")["score"].iloc[-1] < select(d, "new_score")["score"].iloc[-2],
        lambda d: select(d, "new_score")["score"].iloc[-1] > select(d, "new_score")["score"].iloc[-2],
        lambda d: select(d, "new_score")["score"].iloc[-1] < -500,
        lambda d: select(d, "new_score")["score"].iloc[-1] < -1000,
        lambda d: select(d, "new_score")["score"].iloc[-1] < -2000,
        lambda d: select(d, "new_score")["score"].iloc[-5:].min() < -500,
        lambda d: select(d, "new_score")["score"].iloc[-10:].min() < -500,
        lambda d: select(d, "new_score")["score"].iloc[-20:].min() < -500,
        lambda d: select(d, "new_score")["score"].iloc[-5:].min() < -1000,
        lambda d: select(d, "new_score")["score"].iloc[-10:].min() < -1000,
        lambda d: select(d, "new_score")["score"].iloc[-20:].min() < -1000,
        lambda d: select(d, "new_score")["score"].iloc[-5:].min() < -2000,
        lambda d: select(d, "new_score")["score"].iloc[-10:].min() < -2000,
        lambda d: select(d, "new_score")["score"].iloc[-20:].min() < -2000,
        lambda d: select(d, "new_score")["score"].iloc[-1] > 50,
        lambda d: select(d, "new_score")["score"].iloc[-1] > 100,
        lambda d: select(d, "new_score")["score"].iloc[-1] > 200,
        lambda d: select(d, "new_score")["score"].iloc[-5:].min() > 50,
        lambda d: select(d, "new_score")["score"].iloc[-10:].min() > 50,
        lambda d: select(d, "new_score")["score"].iloc[-20:].min() > 50,
        lambda d: select(d, "new_score")["score"].iloc[-5:].min() > 100,
        lambda d: select(d, "new_score")["score"].iloc[-10:].min() > 100,
        lambda d: select(d, "new_score")["score"].iloc[-20:].min() > 100,
        lambda d: select(d, "new_score")["score"].iloc[-5:].min() > 200,
        lambda d: select(d, "new_score")["score"].iloc[-10:].min() > 200,
        lambda d: select(d, "new_score")["score"].iloc[-20:].min() > 200,
    ]
    return conditions

def industry_score_conditions():
    conditions = [
        lambda d: select(d, "industry_score")["featured_falling"].iloc[-5:].max() < 10,
        lambda d: select(d, "industry_score")["featured_falling"].iloc[-5:].max() > 10,
        lambda d: select(d, "industry_score")["featured_rising"].iloc[-5:].max() < 10,
        lambda d: select(d, "industry_score")["featured_rising"].iloc[-5:].max() > 10,
        lambda d: select(d, "industry_score")["rising"].iloc[-5:].max() < 20,
        lambda d: select(d, "industry_score")["rising"].iloc[-5:].max() > 20,
        lambda d: select(d, "industry_score")["falling"].iloc[-5:].max() < 20,
        lambda d: select(d, "industry_score")["falling"].iloc[-5:].max() > 20,
    ]
    return conditions

def trade_conditions():
    return [
        lambda d: d.stats.win_streak() >= 1,
        lambda d: d.stats.win_streak() >= 2,
        lambda d: d.stats.win_streak() >= 3,
        lambda d: d.stats.lose_streak() >= 1,
        lambda d: d.stats.lose_streak() >= 2,
        lambda d: d.stats.lose_streak() >= 3,
    ]

def resistance_support_conditions(legs=["middle"]):
    conditions = []
    for leg in legs:
        conditions = conditions + [
            lambda d: select(d, leg)["resistance_entity"].iloc[-2] < select(d, leg)["resistance_entity"].iloc[-1],
            lambda d: select(d, leg)["resistance_entity"].iloc[-2] > select(d, leg)["resistance_entity"].iloc[-1],
            lambda d: select(d, leg)["resistance_entity"].iloc[-3:].max() > select(d, leg)["resistance_entity"].iloc[-1],
            lambda d: select(d, leg)["resistance_entity"].iloc[-5:].max() > select(d, leg)["resistance_entity"].iloc[-1],
            lambda d: select(d, leg)["resistance_entity"].iloc[-3:-1].max() < select(d, leg)["resistance_entity"].iloc[-1],
            lambda d: select(d, leg)["resistance_entity"].iloc[-5:-1].max() < select(d, leg)["resistance_entity"].iloc[-1],
            lambda d: select(d, leg)["support_entity"].iloc[-2] < select(d, leg)["support_entity"].iloc[-1],
            lambda d: select(d, leg)["support_entity"].iloc[-2] > select(d, leg)["support_entity"].iloc[-1],
            lambda d: select(d, leg)["support_entity"].iloc[-3:].max() > select(d, leg)["support_entity"].iloc[-1],
            lambda d: select(d, leg)["support_entity"].iloc[-5:].max() > select(d, leg)["support_entity"].iloc[-1],
            lambda d: select(d, leg)["support_entity"].iloc[-3:-1].max() < select(d, leg)["support_entity"].iloc[-1],
            lambda d: select(d, leg)["support_entity"].iloc[-5:-1].max() < select(d, leg)["support_entity"].iloc[-1],
            lambda d: select(d, leg)["support_entity"].iloc[-1] > select(d, leg)["resistance_entity"].iloc[-1],
            lambda d: select(d, leg)["support_entity"].iloc[-1] < select(d, leg)["resistance_entity"].iloc[-1],
            lambda d: select(d, leg)["support_entity"].iloc[-1] == 0,
            lambda d: select(d, leg)["resistance_entity"].iloc[-1] == 0,
        ]
    return conditions

def all():
    return average_conditions() + tec_conditions() + cross_conditions() + trend_conditions() + band_conditions() + stages_conditions() + cs_conditions()

def all_with_index(targets=["middle", "nikkei", "dow", "usdjpy", "xbtusd"]):
    conditions = [
        average_conditions(targets),
        tec_conditions(targets),
        cross_conditions(targets),
        trend_conditions(targets),
        band_conditions(targets),
        stages_conditions(targets),
        cs_conditions(targets),
    ]

    return sum(conditions, [])

def by_names(targets=["middle", "nikkei", "dow", "usdjpy", "xbtusd"], names=["all"]):
    conditions = []
    conditions_map = {}
    conditions_map["all"] = lambda t: all_with_index(t)
    conditions_map["industry_score"] = lambda t: all_with_index(t) + industry_score_conditions()
    conditions_map["stages_sum"] = lambda t: all_with_index(t) + industry_score_conditions() + stages_sum_conditions(),
    conditions_map["2021-06-12"] = (lambda t: average_conditions(t) + tec_conditions(t) + cross_conditions(t)
            + trend_conditions(t, additional_columns=["stages_sum_trend", "stages_sum_average_trend", "stages_sum_average_long_trend"])
            + band_conditions(t) + stages_conditions(t) + cs_conditions(t) + industry_score_conditions() + stages_sum_conditions(t) + new_score_conditions())
    conditions_map["2021-11-13"] = lambda t: conditions_map["2021-06-12"](t) + resistance_support_conditions()
    conditions_map["2021-12-12"] = lambda t: conditions_map["2021-11-13"](t+["short"])

    for name in conditions_map.keys():
        conditions = (conditions + conditions_map[name](targets)) if name in names else conditions

    return conditions
