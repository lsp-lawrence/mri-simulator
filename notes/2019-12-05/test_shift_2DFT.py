from pyreconstruct import shift_2DFT
import numpy as np
A = np.array([[1,2,3],[4,5,6],[7,8,9]])
Ap = shift_2DFT(A)
print(A)
print(Ap)
