class Person(object):
    def __init__(self, name, age):
        self.__name = name
        # _age是私有变量，但是能够从全局获得，然而约定俗成不要从全局去访问
        self._age = age

    def get_age_fun(self):
        return self._age

    def set_age_fun(self, value):
        if not isinstance(value, int):
            raise ValueError('年龄必须是数字!')
        if value < 0 or value > 100:
            raise ValueError('年龄必须是0-100')
        self._age = value

    def print_info(self):
        print('%s: %s' % (self.__name, self._age))


p = Person('balala', 20)
p._age = 17
print(p._age)  # 17
print(p.get_age_fun())  # 这里是17 不再是 20,因为此时_age是全局变量,外部直接影响到类内部的更新值

p.set_age_fun(35)
print(p.get_age_fun())  # 35

print(p.print_info())  # balala: 35
