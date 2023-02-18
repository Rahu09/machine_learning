import numpy as np

a = np.array([1,2,3,4,5])

b = np.array([(1,2,3),(4,5,6)])

# print(a)
# print(b)

# print(a.size)
# print(b.size)

# for finding dimension of array
# print(a.ndim)
# print(b.ndim)

# for finding size of array ele
# print(a.itemsize)
# print(b.itemsize)

# for finding shape of array
# print(a.shape)
# print(b.shape)

# for finding datatype of array ele
# print(a.dtype)
# print(b.dtype)

# reshaping your array
# print(b.reshape(3,2))

c = np.array([(1,2),(3,4),(5,6),(7,8)])
# print(c[0,1])

print(c[0:2,-1])

# to get 10 no between 1 and 3
# d = np.linspace(1,3,10)
# print(d)

# arthematic operation on array
d = np.array([(1,2,3),(4,5,6)])
e = np.array([(1,2,3),(4,5,6)])

# print(d*e)
# print(d-e)
# print(d+e)
# print(d/e)

# stacking of array
# vertical stack
x = np.vstack((d,e))
# print(x)

# horizontal stack
y = np.hstack((d,e))
# print(y)