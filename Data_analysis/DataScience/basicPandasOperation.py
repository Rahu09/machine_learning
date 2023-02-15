import numpy as np
import pandas as pd

l1 = [1,2,3]
l2 = ['a','b','c']
arr = np.array(l1)

# pandas series
ser = pd.Series(l2,arr)
# print(ser1)
# print(pd.Series(l1,l2))

list_var = l1*2  # values are repeated two times
print(list_var) 
print(ser*2)  # values are squared

# data frames
