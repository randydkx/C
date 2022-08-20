import pandas as pd
file = '/Users/wenshuiluo/Downloads/计算机学院EMS面单名单69(1).xlsx'
data = pd.read_excel(file, header=[0])
print(data.columns)
data_1 = data[data['学院']=='计算机科学与工程学院'].reset_index(drop=True)
names = data_1['毕业生姓名']
print(len(names))
file1 = '/Users/wenshuiluo/work/total.xlsx'
data1 = pd.read_excel(file1,header=[1])
total_names = data1['姓名']
print(len(total_names))
name_list = []
index_list = []
for index, name in enumerate(list(names)):
    if name in list(total_names):
        name_list.append(name)
        index_list.append(index)
print(name_list)
print(len(name_list))
data_tmp = data_1.iloc[index_list].reset_index(drop=True)
to = '/Users/wenshuiluo/Downloads/tmp/'
for index, name in enumerate(name_list):
    current_sel = data_tmp.iloc[index]
    id = data_tmp.iloc[index]['学号']
    current_sel.to_excel(to + id + name + '.xlsx')