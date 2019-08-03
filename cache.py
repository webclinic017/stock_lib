# -*- coding: utf-8 -*-
import os
import shutil
import pickle
import redis

class Cache:
    def __init__(self, cache_dir="/tmp"):
        self.cache_dir = cache_dir

    def dir(self):
        return self.cache_dir

    def path(self, name):
        return "%s/%s.dump" % (self.dir(), name)

    def exists(self, name):
        return os.path.exists(self.path(name))

    def create(self, name, data):
        output_dir = os.path.dirname(self.path(name))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(self.path(name), "wb") as f:
            pickle.dump(data, f)

    def get(self, name):
        with open(self.path(name), "rb") as f:
            cache = pickle.load(f)
        return cache

    def remove(self, name):
        os.remove(self.path(name))

    def remove_dir(self):
        if os.path.exists(self.dir()):
            shutil.rmtree(self.dir())

class Redis:
    def __init__(self):
        self.client = redis.Redis(host="localhost", port=6379, db=0)

    def exists(self, name):
        return self.client.exists(name)

    def create(self, name, data):
        return self.client.set(name, pickle.dumps(data))

    def get(self, name):
        return pickle.loads(self.client.get(name))

    def remove(self, name):
        return self.client.delete(name)
