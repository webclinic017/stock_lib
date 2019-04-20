# -*- coding: utf-8 -*-
import os
import shutil
import pickle

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
        if not os.path.exists(self.dir()):
            os.makedirs(self.dir())
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
