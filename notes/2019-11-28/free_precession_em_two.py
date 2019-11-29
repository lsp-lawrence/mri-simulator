from pyem import Em
import numpy as np
import matplotlib.pyplot as plt

# Instantiate an em
magnetization = np.array([1.,0,0])
position = np.array([0.,0,0])
velocity = np.array([0.,0,0])
gyromagnetic_ratio = 1.0
equilibrium_magnetization = 1.0
em = Em(magnetization,position,velocity,gyromagnetic_ratio,equilibrium_magnetization)

# Declare time step
delta_t = 0.01
num_steps = 500

# Simulate precession and relaxation
T1 = 1.0
T2 = 1.0
Bz = 10.0+10.0
mu = np.empty([num_steps+1,3])
mu[0,:] = em.mu
for step_no in range(num_steps):
    em.precess_and_relax(T1,T2,Bz,delta_t)
    mu[step_no+1,:] = em.mu
print()

# Plot
t = delta_t*np.arange(num_steps+1)
plt.plot(t,mu)
plt.xlabel('Time (s)')
plt.ylabel('Magnetization (a.u.)')
plt.legend(('$\mu_x$','$\mu_y$','$\mu_z$'))
plt.savefig('free_precession_em_two.pdf')
