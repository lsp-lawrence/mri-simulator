import timeit


code_to_test="""
import numpy as np
N = int(1e6)
for i in range(N):
    a = np.random.random()
"""

elapsed_time = timeit.timeit(code_to_test,number=100)/100
print(elapsed_time)
