# -*- coding: utf-8 -*-
import json
import subprocess

def get_token(user):
    f = open("settings/slack_token.json", "r")
    json_dict = json.load(f)
    return json_dict[user]

def get_user(user="stockd"):
    token = get_token(user)
    params = ["sh", "scripts/slack_users_list.sh", token]
    results = subprocess.check_output(params, timeout=300)
    json_str = results.splitlines()[-1].decode('utf-8')
    data = json.loads(json_str)
    return data

def get_user_name(user_id, user="stockd"):
    data = get_user(user)
    detail = list(filter(lambda x: x["id"] == user_id, data["members"]))
    if len(detail) == 0:
        print("user not found: ", user_id)
        return None
    return detail[0]["name"]

def get_user_id(name, user="stockd"):
    data = get_user(user)
    detail = list(filter(lambda x: x["name"] == name, data["members"]))
    if len(detail) == 0:
        print("user not found: ", name)
        return None
    return detail[0]["id"]

def get_channel(user="stockd"):
    token = get_token(user)
    params = ["sh", "scripts/slack_channel_list.sh", token]
    results = subprocess.check_output(params, timeout=300)
    json_str = results.splitlines()[-1].decode('utf-8')
    data = json.loads(json_str)
    return data

def get_channel_name(channel_id, user="stockd"):
    data = get_channel(user)
    detail = list(filter(lambda x: x["id"] == channel_id, data["channels"]))
    if len(detail) == 0:
        print("channel not found: ", channel_id)
        return None
    return detail[0]["name"]

def get_channel_id(name, user="stockd"):
    data = get_channel(user)
    detail = list(filter(lambda x: x["name"] == name, data["channels"]))
    if len(detail) == 0:
        print("channel not found: ", name)
        return None
    return detail[0]["id"]

def file_post(filetype, filepath, user="stockd", channel="stock", group="bottlefoxy"):
    try:
        if channel.startswith("@"):
            channel_id = get_user_id(channel[1:])
        else:
            channel_id = get_channel_id(channel)
        token = get_token(user)
        params = ["sh", "scripts/slack_file_post.sh", token, group, channel_id, filetype, filepath]
        subprocess.call(params, timeout=300)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("failed slack post")

def post(message, channel="#stock_alert"):
    try:
        subprocess.call(["sh", "scripts/slack_post.sh", channel, message], timeout=5)
    except Exception as e:
        print("failed slack post")

