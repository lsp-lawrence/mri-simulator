import resource
from pyem_no_methods import Em
import numpy as np

num_ems = 100000
em_list = []
magnetization = np.array([0,0,1.0])
position = np.array([1.0,0,0])
velocity = np.array([1.0,0,0])
gyromagnetic_ratio = 1.0
shielding_constant = 1e-5
equilibrium_magnetization = 1.0
for em_no in range(num_ems):
    em_list.append(Em(magnetization,position,velocity,gyromagnetic_ratio,shielding_constant,equilibrium_magnetization))
print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
