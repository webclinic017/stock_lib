# -*- coding:utf-8 -*-
import os
import sys
from selenium import webdriver
from selenium.webdriver.support import expected_conditions as ec
from pyvirtualdisplay import Display
from selenium.webdriver.firefox.firefox_profile import FirefoxProfile
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities

import utils
import slack
from loader import Loader

def get_firefox(download_dir):
  display = Display(visible=0, size=(1280, 720))
  display.start()

  # enable browser logging
  d = DesiredCapabilities.FIREFOX
  d['loggingPrefs'] = {'browser': 'ALL'}

  # profile
  profile = FirefoxProfile()
  profile.set_preference("intl.accept_languages", "ja")
  profile.set_preference("general.useragent.locale", "ja-JP")
  profile.set_preference("browser.download.dir", download_dir)
  profile.set_preference("browser.download.lastDir", download_dir)
  profile.set_preference("browser.download.folderList",2)
  profile.set_preference("browser.helperApps.neverAsk.saveToDisk", "text/plain,application/octet-stream,application/pdf,application/x-pdf,application/vnd.pdf")
  profile.set_preference("browser.download.manager.showWhenStarting", False)
  profile.set_preference("browser.helperApps.neverAsk.openFile","text/plain,application/octet-stream,application/pdf,application/x-pdf,application/vnd.pdf")
  profile.set_preference("browser.helperApps.alwaysAsk.force", False)
  profile.set_preference("browser.download.manager.useWindow", False)
  profile.set_preference("browser.download.manager.focusWhenStarting", False)
  profile.set_preference("browser.helperApps.neverAsk.openFile", "")
  profile.set_preference("browser.download.manager.alertOnEXEOpen", False)
  profile.set_preference("browser.download.manager.showAlertOnComplete", False)
  profile.set_preference("browser.download.manager.closeWhenDone", True)
  profile.set_preference("pdfjs.disabled", True)
  profile.set_preference("media.navigator.enabled", True)

  # create driver
  driver = webdriver.Firefox(firefox_profile=profile, executable_path="/usr/local/bin/geckodriver", capabilities=d, timeout=300)
  return driver, display


def get_phantomjs():
  #############
  # phantomjs
  #############
  # user agent
  user_agent = 'Mozilla/5.0 (Windows NT 5.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/29.0.1547.66 Safari/537.36'
  # PhantomJS本体のパス
  pjs_path = 'scraping/node_modules/phantomjs/bin/phantomjs'
  dcap = {
      "phantomjs.page.settings.userAgent" : user_agent,
      'marionette' : True
  }
  driver = webdriver.PhantomJS(executable_path=pjs_path, desired_capabilities=dcap)
  return driver, None

def get_driver():
  download_dir = "%s/downloads" % Loader.base_dir
  if not os.path.exists(download_dir):
    os.makedirs(download_dir)
  return get_firefox(download_dir)

def screenshot(driver, channel="stock"):
    try:
        driver.save_screenshot("screen.png")
        slack.file_post("png", "screen.png", channel=channel)
    except:
        print("failed screenshot")

