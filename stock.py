from loader import Loader

def get_industry_name(code):
    if not str(code).isdigit():
        return None

    stock_industry_code_map = Loader.stock_industry_code()
    industry_codes = Loader.industry_code()

    df = stock_industry_code_map[stock_industry_code_map["code"] == int(code)]
    if len(df) == 0:
        return None

    industry_code = df["industry_code"].iloc[0]
    df = industry_codes[industry_codes["industry_code"] == industry_code]
    if len(df) == 0:
        return None

    name = df["name"].iloc[0]
    return name

def get_name(code):
    if not str(code).isdigit():
        return None

    kabuplus_stock_data = Loader.kabuplus_stock_data()
    df = kabuplus_stock_data[kabuplus_stock_data[0] == code]
    if len(df) == 0:
        return None
    name = df[1].iloc[0]
    return name

