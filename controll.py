import os
from pathlib import Path

def _signal(filename):
    return "/tmp/%s.stopped" % (filename)


def is_script_stopped(filename):
    return os.path.exists(_signal(filename))


def script_stop(filename):
    path = _signal(filename)
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    Path(path).touch()

def script_start(filename):
    os.remove(_signal(filename))

def _trade_signal(filename):
    return "settings/trade_stop/%s" % (filename)

def is_trade_stopped(filename):
    return os.path.exists(_trade_signal(filename))

def trade_stop(filename):
    path = _trade_signal(filename)
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    Path(path).touch()

def trade_start(filename):
    os.remove(_trade_signal(filename))

def trade_status():
    dirname = os.path.dirname(_trade_signal(""))
    files = os.scandir(dirname)
    return [ f.name for f in files]
