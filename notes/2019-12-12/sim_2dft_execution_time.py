import timeit

code_to_test="""
from pysim import Sim
from pyem import Em
from pygeneratesequence import generate_kspace_pulses,generate_tipping_pulse,generate_2DFT_sequence,see_readout_pulses
from pyreconstruct import reconstruct_from_2DFT
import matplotlib.pyplot as plt
import numpy as np

grid_radius = 1
fig_params = 'single'
see_sequence = False
run_sim = True
plot_sim_results = False
reconstruction_name = 'reconstruction_'+fig_params+'.pdf'

# Place ems
em_positions = 1e-2*np.array([[0.5,0.0,0.0]])
num_ems = em_positions.shape[0]
em_gyromagnetic_ratio = 2.675e8
main_field = 3.0
em_equilibrium_magnetization = 1.0
em_magnetizations = np.zeros([num_ems,3])
em_magnetizations[:,1] = 1.0
em_shielding_constants = np.zeros(num_ems)
em_velocities = np.zeros([num_ems,3],dtype=float)

# Compute the pulse sequence
Gz_amplitude = 10.0e-3
slice_width = 5e-2
delta_t = 1e-6
em_delta_t = 1e-7
xlim = 1e-2
ylim = xlim
fe_sample_radius = grid_radius
pe_sample_radius = grid_radius
kx_max = grid_radius/(2*xlim)
ky_max = grid_radius/(2*ylim)
kmove_time = 1e-3
kread_time = 8e-3
read_all = False
T1_max = 100e-3
T2_max = 10e-3
repetition_time = 111e-3
gyromagnetic_ratio = em_gyromagnetic_ratio
# Create pulse sequence
pulse_sequence = generate_2DFT_sequence(main_field,Gz_amplitude,slice_width,gyromagnetic_ratio,fe_sample_radius,pe_sample_radius,kx_max,ky_max,kmove_time,kread_time,repetition_time,delta_t,read_all)
if see_sequence: see_readout_pulses(pulse_sequence)
##tip_angle = np.pi/2.0
##gyromagnetic_ratio = em_gyromagnetic_ratio
##pulse_tip = generate_tipping_pulse(gyromagnetic_ratio,main_field,Gz_amplitude,slice_width,tip_angle,delta_t)
##pulse_sequence = generate_kspace_pulses(pulse_tip,fe_sample_radius,pe_sample_radius,kx_max,ky_max,kmove_time,adc_rate,gyromagnetic_ratio,delta_t,read_all)

# Run simulation
def T1_map(position): return T1_max
def T2_map(position): return T2_max
print('beginning sim with ' + str(num_ems) + ' ems')
sim = Sim(em_magnetizations,em_positions,em_velocities,em_gyromagnetic_ratio,em_shielding_constants,em_equilibrium_magnetization,em_delta_t,T1_map,T2_map,main_field,pulse_sequence)
ems,mrs = sim.run_sim()
num_pe_samples = int(2*pe_sample_radius+1)
num_fe_samples = int(2*fe_sample_radius+1)
S = np.empty([num_pe_samples,num_fe_samples],dtype=complex)
for line_no in range(num_pe_samples):
    S[line_no,:] = mrs[line_no]
img,x,y = reconstruct_from_2DFT(S,kx_max,ky_max)
"""


elapsed_time = timeit.timeit(code_to_test, number=10)/10
print(elapsed_time)
