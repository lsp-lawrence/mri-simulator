import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pyreconstruct import reconstruct_from_2DFT

mr_real = pd.read_csv('mr_real.csv',sep=' ',header=None).values
mr_imag = pd.read_csv('mr_imag.csv',sep=' ',header=None).values
S = mr_real+1j*mr_imag
img = reconstruct_from_2DFT(S)
plt.figure()
plt.imshow(np.real(img))
plt.show()

