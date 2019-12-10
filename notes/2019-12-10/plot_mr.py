import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pyreconstruct import reconstruct_from_2DFT,adjust_phase

# Opts
np.set_printoptions(precision=3)

# Import S
S_real = pd.read_csv('S_real.csv',sep=',',header=None).values
S_imag = pd.read_csv('S_imag.csv',sep=',',header=None).values
S = S_real+1j*S_imag

print(np.angle(S))
print(np.angle(adjust_phase(S)))

img = reconstruct_from_2DFT(S)
plt.figure()
plt.imshow(img)
plt.colorbar()
plt.show()
