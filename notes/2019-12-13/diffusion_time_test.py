import timeit

code_to_test="""
from pyem import Em
import numpy as np

# Declare em
magnetization = np.array([0.0,0.0,1.0])
position = np.array([0.0,0.0,0.0])
gyromagnetic_ratio = 1.0
shielding_constant = 0.0
equilibrium_magnetization = 1.0
em = Em(magnetization,position,gyromagnetic_ratio,shielding_constant,equilibrium_magnetization)

# Simulate diffusion
num_steps = int(1e4)
delta_t = 1.0
diffusion_coefficient = 1.0
for step_no in range(num_steps):
    em.diffuse(diffusion_coefficient,delta_t)
"""

elapsed_time = timeit.timeit(code_to_test, number=10)/10
print(elapsed_time)
