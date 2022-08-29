import sys
import example
import numpy 
# named parameters
print(example.add(i = 1, j = 2))
print(example.add(3, 4))
# default parameters
print(example.add())
print(example.the_answer)
print(type(example.World), example.World)

import example2
pet = example2.Pet("dog")
print(pet)
pet.name = 'dog_1'
print(pet.getName())
pet.setName('dog_2')
print(pet.getName())
pet.data = "new_properties"
print(pet.__dict__)

import example3
dog = example3.Dog_example3("dog_example3")
print(dog.name)
print(dog.bark())
# 在python中引发c++中定义的exception
# example3.error()
import numpy as np
example3.print_vector([1,2,3])
print()
example3.print_vector(np.arange(4))
print()
example3.print_vector(tuple([1,2,3]))
a = np.random.randn(2,2)
print(type(a[1,1]))
# float32对应c++中的float，64对应double
# a = np.float32(a)
example3.test(example3.Gemfield(a))
print(example3.getgem())

import example4
print(dir(example4))
print(example4.add_c(10,20))
print(example4.add_c(np.arange(10).reshape(1,2,5),np.arange(10).reshape(1,2,5)))