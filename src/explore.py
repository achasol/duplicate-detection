from utils import load_dataset
import re 
data = load_dataset()

"""
print(data)

cols =  data['title'].str.extract(r'([a-zA-Z0-9]*(([0-9]+[ˆ0-9, ]+)|([ˆ0-9, ]+[0-9]+))[a-zA-Z0-9]*)')

cols = cols.fillna(" - ")
data['dense_title'] = cols.apply(lambda row: ''.join(str(e) for e in row), axis=1)


print(data)
"""