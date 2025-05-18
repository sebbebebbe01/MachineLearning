import matplotlib.pyplot as plt
import numpy as np

N=5

A = np.array([[-5,-5,-4,-3],
              [-5,-4,-3,-2],
              [-4,-3,-2,-1],
              [-3,-2,-1,-1]])

print(A.shape)

plt.imshow(A)
plt.colorbar()
plt.show()