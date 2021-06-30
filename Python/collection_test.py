from collections import defaultdict
l=[('a',2),('b',3),('a',1),('b',4),('a',3),('a',1),('b',3)]
# 按照每个关键字进行收集，每个关键字对应于一个list
d = defaultdict(list)
for key,value in l:
    d[key].append(value)

# 去除重复元素，每个关键字对应于一个集合
s = defaultdict(set)
for key,value in l:
    s[key].add(value)

# 将结果转换输出
# print(dict(d.items()))

from collections import OrderedDict
d={'b':3,'a':4,'c':2,'d':1}
# 根据item，按照key进行排序
# .itmes()将对象转化成可以迭代的对象
o = OrderedDict(sorted(d.items(),key=lambda x:x[0],reverse=False))
print('按照key进行排序',o)
o2 = OrderedDict(sorted(d.items(),key=lambda x:x[1],reverse=False))
print("按照value进行排序",o2)
d={'bbbb':3,'aaa':4,'cc':2,'d':1}
o3 = OrderedDict(sorted(d.items(),key=lambda x:len(x[0]),reverse=True))
print('按照key的长度进行排序：',o3)

# a=[1,2,3,4]
# for index,value in enumerate(a):
#     print('({},{})'.format(index,value))

# 弹出第一个元素或者是最后一个元素
print(o3.popitem())
print(o3.popitem(last=False))

from collections import Counter
# 迭代计数器
counter = Counter()
for value in ['a','b','a','a']:
    counter[value]+=1
print(sorted(counter.items(),key=lambda x:x[1]))

from collections import namedtuple
named = namedtuple("helloworld",['x','y'])
n = named(1,2)
print(n)

Point = namedtuple('Point',['x','y'])
point = Point(11,22)
print(point.x)