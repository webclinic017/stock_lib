
# price: 一日の約定金額合計
def oneday_commission(price):
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
def default_commission(price, is_credit):
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

