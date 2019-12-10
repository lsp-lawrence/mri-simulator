from pyreconstruct import shift_2DFT, ishift_2DFT
import numpy as np
A = np.random.randint(0,5,[5,5])
Ap = shift_2DFT(A)
App = ishift_2DFT(Ap)
print(A)
print(Ap)
print(App)
