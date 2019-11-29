from pypulse import Pulse
from pysim import Sim
import numpy as np
import matplotlib.pyplot as plt

# Declare a readout pulse
time_step = 0.01
pulse_length = 500
mode = 'free'
Gx = np.ones(pulse_length)
Gy = np.ones(pulse_length)
readout = np.zeros(pulse_length,dtype=bool)
for i in range(len(readout)):
    if not i%2:
        readout[i] = True
pulse = Pulse(mode=mode,Gx=Gx,Gy=Gy,readout=readout,time_step=time_step)

# Initialize simulation
num_ems = 1
main_field = 10.0
pulse_sequence = [pulse]
sim = Sim(num_ems,main_field,pulse_sequence)

# Run simulation and collect MR signal
mr_signals = sim.run_sim()
mr_signal = mr_signals[0]

# Plot MR signal (FID of one em)
t_readout = []
for step_no in range(pulse_length):
    if readout[step_no]:
        t_readout.append(time_step*step_no)
plt.plot(t_readout,mr_signal.real,'-o')
plt.xlabel('Time (s)')
plt.ylabel('MR signal (a.u.)')
plt.title('FID of a single em')
plt.savefig('fid_one_em.pdf')
