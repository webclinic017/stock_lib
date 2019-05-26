import time
import hashlib
import hmac
import json
import requests
import urllib

class BitmexAPIClient:
    def __init__(self):
        self.public, self.secret = self.api_keys()
        print(self.public, self.secret)

    def api_keys(self):
        with open("settings/bitmex_api_key.json") as f:
            api_key = json.load(f)
        return api_key["public"], api_key["secret"]

    def body(self, params=None):
        return '' if params is None else json.dumps(params, separators=(',', ':'))

    def headers(self, method, url, data):
        expire, signature = self.generate_signature(self.secret, method, url, data)
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "api-expires": str(expire),
            "api-key": self.public,
            "api-signature": signature
        }
        return headers

    def get(self, url):
        session = requests.Session()
        body = self.body()
        headers = self.headers("GET", url, body)
        return session.get(url, headers=headers)

    def post(self, url, params):
        session = requests.Session()
        body = self.body(params)
        headers = self.headers("POST", url, body)
        return session.post(url, body, headers=headers)

    def put(self, url, params):
        session = requests.Session()
        body = self.body(params)
        headers = self.headers("PUT", url, body)
        return session.put(url, body, headers=headers)

    def delete(self, url, params):
        session = requests.Session()
        body = self.body(params)
        headers = self.headers("DELETE", url, body)
        return session.delete(url, body, headers=headers)

    # API 署名を生成。
    # 署名は HMAC_SHA256(secret, verb + path + expires + data), 16 進数変換済み。
    # 同士は大文字、URLは相対的、期限は Unix のタイムスタンプ (秒単位) である必要あり
    # データが存在する場合、キーの間に空白のない JSON であること。
    def generate_signature(self, secret, verb, url, data):
        expires = int(round(time.time()) + 5)
        """BitMEX と適合するリクエスト署名を生成。"""
        # URL を構文解析して、ベースを削除し、パスのみ抽出できるようにする。
        parsedURL = urllib.parse.urlparse(url)
        path = parsedURL.path
        if parsedURL.query:
            path = path + '?' + parsedURL.query

        if isinstance(data, (bytes, bytearray)):
            data = data.decode('utf8')

        print("Computing HMAC: %s" % verb + path + str(expires) + data)
        message = verb + path + str(expires) + data

        signature = hmac.new(bytes(secret, 'utf8'), bytes(message, 'utf8'), digestmod=hashlib.sha256).hexdigest()
        return expires, signature

