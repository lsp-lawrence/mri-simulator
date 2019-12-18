from pygeneratesequence import *
import matplotlib.pyplot as plt

# Compute the pulse sequence
grid_radius=5
Gz_amplitude = 10.0e-3
slice_width = 5e-2
pulse_delta_t = 1e-6
em_delta_t = 1e-6
xlim = 1e-2
ylim = xlim
fe_sample_radius = grid_radius
pe_sample_radius = grid_radius
kx_max = grid_radius/(2*xlim)
ky_max = grid_radius/(2*ylim)
kmove_time = 1e-3
kread_time = 8e-3
read_all = True
T1_max = 100e-3
T2_max = 10e-3
repetition_time = 100e-3
gyromagnetic_ratio = 2.675e8
main_field = 3.0
diffusion_gradients = np.ones(3)*5.0e-3
diffusion_gradient_time = 0.5e-3
diffusion_time = 4e-3
# Create pulse sequence
pulse_sequence = generate_DW_sequence(main_field,Gz_amplitude,slice_width,gyromagnetic_ratio,kread_time,pulse_delta_t,diffusion_gradients,diffusion_gradient_time,diffusion_time)
see_readout_pulses(pulse_sequence)
