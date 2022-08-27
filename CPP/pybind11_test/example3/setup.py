from setuptools import setup, Extension 
 
functions_module = Extension( 
 name = 'functions', 
 sources = ['add.cpp', 'sub.cpp', 'mul.cpp', 'div.cpp', 'functions_wrapper.cpp'], 
 include_dirs = ['/Users/wenshuiluo/Downloads/pybind11/include', 
     '/Users/wenshuiluo/opt/anaconda3/include'] 
) 
 
setup(ext_modules = [functions_module])