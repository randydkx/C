import example
a = [3, 2, 1]
b = [2.1, 5, 2]

# 使用单独的函数
example.print("---- use single function -------")
example.print("a · b = " + str(example.inner_product(a, b)))
example.print("a + b = " + str(example.sum(a, b)))

# 使用类
example.print("---- use class -----------------")
vector = example.Vector() # 类实例化
example.print("a · b = " + str(vector.inner_product(a, b)))
example.print("a + b = " + str(vector.sum(a, b)))

print(help(example))