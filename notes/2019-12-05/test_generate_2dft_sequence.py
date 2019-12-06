from pygeneratesequence import generate_2DFT_sequence
import matplotlib.pyplot as plt
import numpy as np

# Running params
save_fig = False
fig_name = 'pulse_sequence_2dft.jpg'

# Declare params
gyromagnetic_ratio = 2.675e8
main_field = 3.0
Gz_amplitude = 10.0e-3
slice_width = 5e-2
delta_t = 1e-7
num_fe_samples = 11
num_pe_samples = 11
kx_max = 1e-2 
ky_max = 1e-2 
kmove_time = 1e-3
adc_rate = 5e3
gyromagnetic_ratio = 2.675e8

# Call function
pulse_sequence = generate_2DFT_sequence(main_field,Gz_amplitude,slice_width,gyromagnetic_ratio,num_fe_samples,num_pe_samples,kx_max,ky_max,kmove_time,adc_rate,delta_t)

# Plot pulse sequence
plt.figure(figsize=(8,7))
for line_no in range(num_pe_samples):
    pulse_index = line_no*2
    pulse_tip = pulse_sequence[pulse_index]
    t_tip = np.linspace(0.0,pulse_tip.length*delta_t,pulse_tip.length)
    pulse_kspace = pulse_sequence[pulse_index+1]
    t_kspace = np.linspace(0.0,pulse_kspace.length*delta_t,pulse_kspace.length)
    plt.subplot(421)
    plt.plot(t_tip*1e3,pulse_tip.Gz*1e4)
    plt.title('Gz')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude (uT/cm)')
    plt.subplot(423)
    plt.plot(t_tip*1e3,pulse_tip.B1x*1e6)
    plt.title('B1x')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude (uT)')
    plt.subplot(422)
    plt.plot(t_kspace*1e3,pulse_kspace.Gz*1e4)
    plt.title('Gz')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude (uT/cm)')
    plt.subplot(424)
    plt.plot(t_kspace*1e3,pulse_kspace.Gy*1e5)
    plt.title('Gy')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude (uT/mm)')
    plt.subplot(426)
    plt.plot(t_kspace*1e3,pulse_kspace.Gx*1e5)
    plt.title('Gx')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude (uT/mm)')
    plt.subplot(428)
    plt.plot(t_kspace*1e3,pulse_kspace.readout,'o')
    plt.title('Read ADC?')
    plt.ylabel('T/F')
    plt.xlabel('Time (ms)')
plt.tight_layout()
if save_fig:
    plt.savefig(fig_name)
else:
    plt.show()
    
