import time
import hashlib
import hmac
import json
import requests
import urllib

def api_keys():
    with open("settings/bitmex_api_key.json") as f:
        api_key = json.load(f)
    return api_key["public"], api_key["secret"]

def request(method, url, params):
    response = {}
    public, secret = api_keys()
    print(public, secret)
    expire, signature = generate_signature(secret, method, url, '')

    session = requests.Session()

    headers = {
        "api-expires": str(expire),
        "api-key": public,
        "api-signature": signature
    }

    r = session.get(url, headers=headers, params=params)
    if r.status_code == requests.codes.ok:
        return r.text
    else:
        raise Exception(r.status_code, r.reason, url)

    return response

# API 署名を生成。
# 署名は HMAC_SHA256(secret, verb + path + expires + data), 16 進数変換済み。
# 同士は大文字、URLは相対的、期限は Unix のタイムスタンプ (秒単位) である必要あり
# データが存在する場合、キーの間に空白のない JSON であること。
def generate_signature(secret, verb, url, data):
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

