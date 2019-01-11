import requests
import pandas as pd
from pandas.compat import StringIO

class EODHistoricalDataClient:
    def __init__(self, token):
        self.token = token

    def get_token(self):
        return self.token

    def request(self, url, params={}):
        session = requests.Session()
        params["api_token"] = self.get_token()
        r = session.get(url, params=params)
        if r.status_code == requests.codes.ok:
            df = pd.read_csv(StringIO(r.text), skipfooter=1, parse_dates=[0], index_col=0, engine='python')
            return df
        else:
            raise Exception(r.status_code, r.reason, url)

    def get_fundamentals_data(self, symbol="AAPL.US"):
        url = "https://eodhistoricaldata.com/api/fundamentals/%s" % symbol
        return self.request(url)

    def get_eod_data(self, symbol="AAPL.US"):
        url = "https://eodhistoricaldata.com/api/eod/%s" % symbol
        return self.request(url)

    def get_codes(self, exchanges="US"):
        url = "https://eodhistoricaldata.com/api/exchanges/%s" % exchanges
        return self.request(url)

