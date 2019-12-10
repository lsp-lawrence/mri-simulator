from pysim import Sim
from pypulse import Pulse
import numpy as np
import matplotlib.pyplot as plt

# Set a single em offset along x with magnetization tipped to y
r0 = np.array([1.0e-2,0.0,0.0])
v0 = np.array([0.0,0.0,0.0])
mu0 = np.array([0.0,1.0,0.0])
gamma = 2.0
B0 = 3.0
mu_eq = 1.0
sigma = 0.0

# Generate pulse sequence
t_final = 6.0
delta_t = 1.0e-2
t_vals = np.arange(0.0,t_final,delta_t)
pulse_length = len(t_vals)
Gx = np.ones(pulse_length)
Gy = np.zeros(pulse_length)
Gz = np.zeros(pulse_length)
readout = np.ones(pulse_length,dtype=bool)
pulse = Pulse(mode='free',Gx=Gx,Gy=Gy,Gz=Gz,readout=readout,delta_t=delta_t)

# Run simulation
em_magnetizations = np.array([mu0])
em_positions = np.array([r0])
em_velocities = np.array([v0])
em_gyromagnetic_ratio = gamma
em_shielding_constants = np.array([sigma])
em_equilibrium_magnetization = mu_eq
def T1_map(position): return 1e6
def T2_map(position): return 1e6
main_field = B0
pulse_sequence = [pulse]
sim = Sim(em_magnetizations,em_positions,em_velocities,em_gyromagnetic_ratio,em_shielding_constants,em_equilibrium_magnetization,T1_map,T2_map,main_field,pulse_sequence)
ems,mrs_demodulated = sim.run_sim()

# Look at MR signal
mr = mrs[0]
plt.plot(t_vals*1e3,np.real(mr))
plt.plot(t_vals*1e3,np.imag(mr))
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude (a.u.)')
plt.show()
