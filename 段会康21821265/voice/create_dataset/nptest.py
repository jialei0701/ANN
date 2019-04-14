import numpy as np

data_set = np.empty((3, 12), dtype=int)
a = np.empty((12, ), dtype=int)
for i in range(12):
    a[i] = i

a = np.reshape(a, (3, 4))
print(a)
print("---------")
b = a.T
print(b)
print("---------")
c = np.reshape(b, (1, -1))
print(c)
print("---------")
for i in range(3):
    data_set[i, :] = c

print(data_set)
print("---------")
d = np.reshape(data_set, (-1, 4, 3, 1))
print(d)
print("---------")
