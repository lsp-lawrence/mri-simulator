from pyem import Em
import numpy as np
import matplotlib.pyplot as plt

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
r = np.empty([num_steps,3])
for step_no in range(num_steps):
    r[step_no,:] = em.r
    em.diffuse(diffusion_coefficient,delta_t)

# Plot trajectory
plt.figure()
plt.plot(r[:,0],r[:,1])
plt.xlabel('x position')
plt.ylabel('y position')
#plt.savefig('diffusion_trajectory.pdf')
plt.show()

### Plot histogram of position in x,y,z
##names = ['x','y','z']
##plt.figure(figsize=(12,4))
##for plot_no in range(3):
##    ax_name = names[plot_no]
##    plt.subplot(1,3,plot_no+1)
##    plt.hist(r[:,plot_no])
##    plt.title('Histogram of ' + ax_name + '-position visits')
##    plt.xlabel(ax_name)
##    plt.ylabel('Position visits')
##plt.tight_layout()
##plt.show()
