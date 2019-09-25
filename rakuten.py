
# price: 一日の約定金額合計
def oneday_comission(price):
    comissions = [
        [100000, 0],
        [200000, 206],
        [300000, 308],
        [500000, 463],
        [1000000, 926],
        [2000000, 2160],
        [3000000, 3240],
    ]

    comission = None
    for c in comissions:
        # 厳密ではないが片道無料なので半分に
        if price / 2 < c[0]:
            comission = c[1]
            break

    # 300万以上の場合
    if comission is None:
        comission = 3240 + (int((price / 2) / 1000000) - 3) * 1080

    return comission

# price: 取引一回の約定代金
def default_comission(price, is_credit):
    actual_comissions = [
        [50000, 50],
        [100000, 90],
        [200000, 105],
        [500000, 250],
        [1000000, 487],
        [1500000, 582],
        [30000000, 921],
    ]

    credit_comissions = [
        [100000, 90],
        [200000, 135],
        [500000, 180],
    ]

    # 最大超え
    over = 350 if is_credit else 973

    comissions = credit_comissions if is_credit else actual_comissions

    comission = None
    for c in comissions:
        if price < c[0]:
            comission = c[1]
            break

    if comission is None:
        comission = over

    return comission

