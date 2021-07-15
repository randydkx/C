from io import BufferedRWPair
from selenium import webdriver
from selenium.webdriver.common.by import By
from urllib3.packages.six import b


# 通过selenium自动化登陆并获取网页源代码
def function1():
    browser = webdriver.Chrome()
    browser.get('https://www.taobao.com')
    print(browser.page_source)
    browser.close()

#  根据选择器查找节点
def function2():
    browser = webdriver.Chrome()
    browser.get('https://www.taobao.com')
    # 三种不同的选择方式，得到的结果是一样的
    input_first = browser.find_element_by_id('q')
    input_second = browser.find_element_by_css_selector('#q')
    input_third = browser.find_element_by_xpath('//*[@id="q"]')
    print(input_first)
    print(input_second)
    print(input_third)
    # 通用方法
    input = browser.find_element(By.ID,'q')
    print(input)
    # 查找多个节点，查找主题中的所有的li节点
    lis = browser.find_elements_by_css_selector('.service-bd li')
    print(lis)
    browser.close()

import time
# 与节点交互，输入信息
def function3():
    browser = webdriver.Chrome()
    browser.get('https://www.taobao.com')
    input = browser.find_element_by_id('q')
    input.send_keys('iPhone')
    time.sleep(1)
    input.clear()
    input.send_keys('iPad')
    button = browser.find_element_by_class_name('btn-search')
    button.click()

# 对浏览器进行前进和后退，并操作cookie
def function4():
    browser = webdriver.Chrome()
    browser.get('https://www.zhihu.com/')
    browser.get('https://www.taobao.com/')
    browser.get('https://www.python.org/')
    browser.back()
    time.sleep(1)
    browser.forward()
    browser.close()
    browser = webdriver.Chrome()
    browser.get('https://www.zhihu.com/explore')
    print(browser.get_cookies())
    browser.add_cookie({'name':'name','domain':'www.zhihu.com','value':'germey'})
    print(browser.get_cookies())
    browser.delete_all_cookies()
    print(browser.get_cookies())

if __name__ == '__main__':
    function4()