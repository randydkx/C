from bs4 import BeautifulSoup
soup = BeautifulSoup('<p>hello</p>','lxml')
print(soup.p.string)

# from selenium import webdriver
# brower = webdriver.PhantomJS()
# brower.get("https://www.baidu.com")
# print(brower.current_url)

import pymysql
print(pymysql.__version__)

import tesserocr
from PIL import Image
image = Image.open('image.jpg')
print(tesserocr.image_to_text(image))

import pymongo
print(pymongo.__version__)

