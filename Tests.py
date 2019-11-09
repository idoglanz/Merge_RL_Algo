import numpy as np

a = np.array([])

b = np.zeros((3, 3, 3))

c = np.append(a, b, 0)

print(a.shape, b.shape, c.shape)
print(a)