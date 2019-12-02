from pygeneratesequence import generate_kspace_pulses
import matplotlib.pyplot as plt
import numpy as np

num_fe_samples = 10
num_pe_samples = 10
kx_max = 1e-2 # 1 cm^{-1}
ky_max = 1e-2 # 1 cm^{-1}
kmove_time = 10e-6 # 10 us
adc_rate = 1e6 # 1 MHz
gyromagnetic_ratio = 2.675e8 #(s*T)^{-1}
delta_t = 10e-9 # 10 ns time step
readout_pulses = generate_kspace_pulses(num_fe_samples,num_pe_samples,kx_max,ky_max,kmove_time,adc_rate,gyromagnetic_ratio,delta_t)

t_pulse = delta_t*np.arange(readout_pulses[0].length)
num_pulses = len(readout_pulses)
plt.figure(figsize=(4,6))
for pulse_no in range(num_pulses):
    pulse = readout_pulses[pulse_no]
    plt.subplot(311)
    plt.plot(t_pulse*1e6,pulse.Gy*1e6)
    plt.xlabel('Time (us)')
    plt.ylabel('Amplitude (uT/m)')
    plt.title('Gy')
    plt.subplot(312)
    plt.plot(t_pulse*1e6,pulse.Gx*1e6)
    plt.xlabel('Time (us)')
    plt.ylabel('Amplitude (uT/m)')
    plt.title('Gx')
    plt.subplot(313)
    plt.plot(t_pulse*1e6,pulse.readout)
    plt.xlabel('Time (us)')
    plt.ylabel('T/F')
    plt.title('Readout')
plt.tight_layout()
plt.show()
