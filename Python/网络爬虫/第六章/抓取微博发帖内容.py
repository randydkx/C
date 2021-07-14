import requests
from urllib.parse import urlencode
from pyquery import PyQuery as pq
from pymongo import MongoClient

base_url = 'https://m.weibo.cn/api/container/getIndex?'
headers = {
    'Host': 'm.weibo.cn',
    'Referer': 'https://m.weibo.cn/u/2830678474',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36',
    # 标记是ajax请求
    'X-Requested-With': 'XMLHttpRequest',
}
client = MongoClient()
# 创建了一个weibo数据库
db = client['weibo']
# 在该数据库中创建了名为weibo的collection
collection = db['weibo']
# 最多爬取10页数据
max_page = 10


def get_page(page):
    params = {
        'type': 'uid',
        'value': '2830678474',
        'containerid': '1076032830678474',
        'page': page
    }
    url = base_url + urlencode(params)
    try:
        response = requests.get(url, headers=headers)
        # 成功则返回response的json数据格式
        if response.status_code == 200:
            return response.json(), page
    except requests.ConnectionError as e:
        print('Error', e.args)


# 对一个page进行解析
def parse_page(json, page: int):
    if json:
        items = json.get('data').get('cards')
        for index, item in enumerate(items):
            if page == 1 and index == 1:
                continue
            else:
                # 如果mblog不存在则设置为空字典数据
                item = item.get('mblog', {})
                weibo = {}
                weibo['id'] = item.get('id')
                # 去除所有的html节点，取出其中的文字等
                weibo['text'] = pq(item.get('text')).text()
                # 如果使用这种方法，将会得到html
                # weibo['text'] = item.get('text')
                weibo['attitudes'] = item.get('attitudes_count')
                weibo['comments'] = item.get('comments_count')
                weibo['reposts'] = item.get('reposts_count')
                yield weibo


# 将一条数据插入mongodb
def save_to_mongo(result):
    if collection.insert(result):
        print('Saved to Mongo')


# 抓取微博发帖的主函数
def catch_weibo_posts():
    for page in range(1, max_page + 1):
        json = get_page(page)
        # 返回的json是一个tuple结构
        print(json)
        # 将tuple解析出其中的内容
        print(*json)
        results = parse_page(*json)
        for result in results:
            print(result)
            save_to_mongo(result)

if __name__ == '__main__':
    # function19()
    catch_weibo_posts()