import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('mr.csv',sep=',',header=None)
df = df.str.replace('i','j').apply(lambda x: np.complex(x))
mr = df.to_numpy()

##plt.figure()
##for sig_no in range(mr.shape[0]):
##    plt.plot(np.real(mr[sig_no,:]))
##plt.show()
