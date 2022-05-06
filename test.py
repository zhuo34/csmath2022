import numpy as np
import scipy
import scipy.linalg


A = np.array([[1, 2], [0, 0]])
idx = [0]
b = np.array([[1], [0]])
# x = scipy.linalg.solve(A, b)
# print(A[idx, idx].shape)
print(A[0, idx])