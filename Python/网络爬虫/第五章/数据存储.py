# 提取知乎发现中的内容到txt中
import re
from typing import Collection
from pymongo import collection, results
from pymysql import cursors
from pymysql.times import Date
import requests
from pyquery import PyQuery as pq
from bs4 import BeautifulSoup
import json
import csv
import pymysql
import pymongo

# 根据某个url获取html内容
def function1():
    url = "https://www.zhihu.com/explore"
    headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.162 Safari/537.36'
            }
    html = requests.get(url,headers=headers).text
    # 使用beantifulsoup将html美化展示
    soup = BeautifulSoup(html,'lxml')
    print(soup.prettify())
    doc = pq(html)

# 从字符串和文本中解析出json格式的数据
def function2():
    str='''
    [{
        "name":"Bob",
        "gender":"male",
        "birthday":"1992-1-12"
    },
    {
        "name":"Selina",
        "gender":"female",
        "birthday":"1994-2-1"
    }]
    '''
    data = json.loads(str)
    print(data[0])

# 从json格式的文件中解析出json对象
def function3():
    with open('data.json','r') as file:
        str = file.read()
        data = json.loads(str)
        print(data)

# 将json对象写入json文件中
def function4():
    data = [{'name': 'Bob', 'gender': 'male', 'birthday': '1992-1-12'}, {'name': 'Selina', 'gender': 'female', 'birthday': '1994-2-1'}]
    # indent代表缩进字符的个数
    with open('data1.json','w') as file:
        file.write(json.dumps(data,indent=2))

# 中文字符的处理
def function5():
    data = [{'name': '罗文水', 'gender': '男', 'birthday': '1992-1-12'}]
    with open('data1.json','w',encoding='utf-8') as file:
        file.write(json.dumps(data,indent=2,ensure_ascii=False))

# 使用csv格式写入数据
def function6():
    # w:如果文件存在就覆盖文件，如果文件不存在就创建新的文件并写入
    with open('data.csv','w',encoding='utf-8') as csvfile:
        fieldnames = ['id','name','age']
        writer = csv.DictWriter(csvfile,fieldnames=fieldnames,delimiter=',')
        writer.writeheader()
        writer.writerow({'id':'10001','name':'罗文水','age':22})

# 从文件中读取并打印出csv
def function7():
    with open('data.csv','r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            print(row)

# 关系型数据库存储
def function8():
    db = pymysql.connect(host='localhost',user='root',password='17851093886',port=3306)
    cursor = db.cursor()
    cursor.execute('SELECT VERSION()')
    data = cursor.fetchone()
    print('Database version',data)
    cursor.execute('create database spiders default character set utf8')
    db.close()

# 创建mysql数据表
def function9():
    db = pymysql.connect(host='localhost',user='root',password='17851093886',port=3306)
    cursor = db.cursor()
    cursor.execute('use spiders')
    sql = 'create table if not exists students(id varchar(255) not null,name varchar(255) not null,age int not null,primary key (id))'
    cursor.execute(sql)
    db.close()
# 向mysql数据库中插入数据
def function10():
    db = pymysql.connect(host='localhost',user='root',password='17851093886',port=3306)
    cursor = db.cursor()
    data = {'id':'42332','name':'罗文水','age':'12'}
    table = 'students'
    keys = ','.join(data.keys())
    values = ','.join(['%s']*len(data))
    cursor.execute('use spiders')
    # 最终的形式： insert into students(id,name,age) values (%s,%s,%s)
    sql = 'insert into {table}({keys}) values ({values})'.format(table=table,keys=keys,values=values)
    print(sql)
    try:
        if cursor.execute(sql,tuple(data.values())):
            print('Successful')
            db.commit()
    except:
        print('Failed')
        db.rollback()
    db.close()

# mysql数据库更新数据
def function10():
    db = pymysql.connect(host='localhost',user='root',password='17851093886',port=3306)
    cursor = db.cursor()
    data = {'id':'42332','name':'罗文水','age':'12'}
    table = 'students'
    keys = ','.join(data.keys())
    values = ','.join(['%s']*len(data))
    cursor.execute('use spiders')
    # 最终的形式： insert into students(id,name,age) values (%s,%s,%s)
    sql = 'UPDATE students SET age = %s WHERE name = %s'
    try:
        cursor.execute(sql,(25,'罗文水'))
        db.commit()
    except:
        db.rollback()
    db.close()

# 另一种数据更新方法，如果数据存在就更新数据，如果不存在就插入数据
def function11():
    db = pymysql.connect(host='localhost',user='root',password='17851093886',port=3306)
    cursor = db.cursor()
    data = {'id':'42332','name':'罗文水','age':'100'}
    table = 'students'
    keys = ','.join(data.keys())
    values = ','.join(['%s']*len(data))
    cursor.execute('use spiders')
    # 加上on duplicate key update 表示如果存在相同的主键则执行更新操作
    sql = 'insert into {table}({keys}) values ({values}) on duplicate key update'.format(table=table,keys=keys,values=values)
    updatesql = ','.join([" {key} = %s".format(key=key) for key in data.keys()])
    sql += updatesql
    print(sql)
    try:
        if cursor.execute(sql,tuple(data.values())*2):
            print('Successful')
            db.commit()
    except:
        print('Failed')
        db.rollback()
    db.close()

# mysql数据库删除数据和查询数据
def function12():
    db = pymysql.connect(host='localhost',user='root',password='17851093886',port=3306)
    cursor = db.cursor()
    data = {'id':'42332','name':'罗文水','age':100}
    table = 'students'
    condition = 'age > 99'
    cursor.execute('use spiders')
    # 加上on duplicate key update 表示如果存在相同的主键则执行更新操作
    sql = 'delete from {table} where {condition}'.format(table = table,condition = condition)
    print(sql)
    try:
        if cursor.execute(sql):
            print('Success to delete')
            db.commit()
    except:
        print('Fail to delete')
        db.rollback()
    sql = 'select * from {table}'.format(table=table)
    try:
        cursor.execute(sql)
        print('Count:',cursor.rowcount)
        # cursor在取一次数据只有偏移指针指向了吓一条数据，所以fetchall取到的数据并不是全部数据，
        # 缺少了第一条数据
        one = cursor.fetchone()
        print("One:",one)
        results = cursor.fetchall()
        print('Result:',results)
        print("results type:",type(results))
        for row in results:
            print(row)
    except:
        print('Error')
    # 推荐使用的获取方法
    db.close()

# 使用mongodb非关心型数据库进行数据存储
def function13():
    client = pymongo.MongoClient(host='localhost',port=27017)
    # 指定数据库
    db = client.test
    # 集合：类似于关系型数据库中的表
    collection = db['students']
    student={
        'id':'20170101',
        'name':'Jordan',
        'age':20,
        'gender':'male'
    }
    result = collection.insert(student)
    # 返回的是一个ObjectID
    print(result)

# 插入多条数据
def function14():
    client = pymongo.MongoClient(host='localhost',port=27017)
    # 指定数据库
    db = client.test
    # 集合：类似于关系型数据库中的表
    collection = db['students']
    student1={
        'id':'20170101',
        'name':'Jordan',
        'age':20,
        'gender':'male'
    }
    student2={
        'id':'20170202',
        'name':'Mike',
        'age':21,
        'gender':'male'
    }
    result = collection.insert_many([student1,student2])
    print(result.inserted_ids)

def function15():
    client = pymongo.MongoClient(host='localhost')
    db = client.test
    collection = db.students
    # result = collection.find_one({'name':'Mike'})
    # results = collection.find({'age':'20'})
    # 查找大于20的数据
    results = collection.find({'age':{'$gt': 20}})
    # 返回一个生成器对象
    print(type(results))
    print(results.count())
    for result in results:
        print(result)
    client.close()

from bson import ObjectId
# 通过objectID进行查询
def function16():
    client = pymongo.MongoClient(host='localhost')
    db = client.test
    collection = db.students
    result = collection.find_one(   {'_id': ObjectId('60ee5214f158fd15e7fc5c9a')})
    print(type(result))
    print(result)

# mongodb数据库中对结果进行计数和排序
def function17():
    client = pymongo.MongoClient(host='localhost')
    db = client.test
    collection = db.students
    # 计数
    count = collection.find().count()
    print(count)
    # 排序,按照name字段进行排序
    results = collection.find({}).sort('name',pymongo.ASCENDING)
    print([result['name'] for result in results])
    # 跳过开头两个数据,控制最多显示的数量
    results = collection.find({}).skip(2).limit(2)
    print([result['name'] for result in results])


# mongodb数据库的更新操作
def function18():
    client = pymongo.MongoClient(host='localhost')
    db = client.test
    collection = db.students
    condition = {'name':'Mike'}
    student = collection.find_one(condition)
    student['age']=40
    # 传入的参数是原来的条件和修改之后的数据
    result = collection.update_one(condition,{'$set':student})
    print(result)
    # 匹配条件的数量以及影响的数据条数
    print(result.matched_count,result.modified_count)

# mongodb数据库的删除操作
def function19():
    client = pymongo.MongoClient(host='localhost')
    db = client.test
    collection = db.students
    result = collection.delete_many({'name':'Mike'})
    # 删除的文档条数
    print(result.deleted_count)


import requests
from urllib.parse import urlencode
from pyquery import PyQuery as pq
from pymongo import MongoClient

base_url = 'https://m.weibo.cn/api/container/getIndex?'
headers = {
    'Host': 'm.weibo.cn',
    'Referer': 'https://m.weibo.cn/u/2830678474',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36',
    'X-Requested-With': 'XMLHttpRequest',
}
client = MongoClient()
db = client['weibo']
collection = db['weibo']
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
        if response.status_code == 200:
            return response.json(), page
    except requests.ConnectionError as e:
        print('Error', e.args)


def parse_page(json, page: int):
    if json:
        items = json.get('data').get('cards')
        for index, item in enumerate(items):
            if page == 1 and index == 1:
                continue
            else:
                item = item.get('mblog', {})
                weibo = {}
                weibo['id'] = item.get('id')
                weibo['text'] = pq(item.get('text')).text()
                weibo['attitudes'] = item.get('attitudes_count')
                weibo['comments'] = item.get('comments_count')
                weibo['reposts'] = item.get('reposts_count')
                yield weibo


def save_to_mongo(result):
    if collection.insert(result):
        print('Saved to Mongo')


def catch_weibo_posts():
    for page in range(1, max_page + 1):
        json = get_page(page)
        results = parse_page(*json)
        for result in results:
            print(result)
            save_to_mongo(result)

if __name__ == '__main__':
    # function19()
    catch_weibo_posts()