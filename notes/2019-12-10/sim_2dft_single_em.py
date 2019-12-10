from pysim import Sim
from pyem import Em
from pygeneratesequence import generate_kspace_pulses,generate_tipping_pulse
import matplotlib.pyplot as plt
import numpy as np

see_distribution = False
see_sequence = False
run_sim = True

# Put a single em at the origin
em_positions = np.array([[0.0,0.0,0.0]])
num_ems = em_positions.shape[0]
em_gyromagnetic_ratio = 2.675e8
main_field = 3.0
em_equilibrium_magnetization = 1.0
em_magnetizations = np.zeros([num_ems,3])
em_magnetizations[:,1] = 1.0
em_shielding_constants = np.zeros(num_ems)
em_velocities = np.zeros([num_ems,3],dtype=float)
if see_distribution:
    x,y = em_positions[:,0:2].T
    plt.scatter(x,y)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

# Compute the pulse sequence
Gz_amplitude = 10.0e-3
slice_width = 5e-2
delta_t = 1e-6
grid_radius = 1
xlim = 1e-2
ylim = xlim
fe_sample_radius = grid_radius
pe_sample_radius = grid_radius
kx_max = grid_radius/(2*xlim)
ky_max = grid_radius/(2*ylim)
kmove_time = 1e-3
adc_rate = 5e3
read_all = False
# Create pulse sequence
tip_angle = np.pi/2.0
gyromagnetic_ratio = em_gyromagnetic_ratio
pulse_tip = generate_tipping_pulse(gyromagnetic_ratio,main_field,Gz_amplitude,slice_width,tip_angle,delta_t)
pulse_sequence = generate_kspace_pulses(pulse_tip,fe_sample_radius,pe_sample_radius,kx_max,ky_max,kmove_time,adc_rate,gyromagnetic_ratio,delta_t,read_all)

if run_sim:
    # Run simulation
    def T1_map(position): return np.Inf
    def T2_map(position): return np.Inf
    print('beginning sim with ' + str(num_ems) + ' ems')
    sim = Sim(em_magnetizations,em_positions,em_velocities,em_gyromagnetic_ratio,em_shielding_constants,em_equilibrium_magnetization,T1_map,T2_map,main_field,pulse_sequence)
    ems,mrs = sim.run_sim()
##    # Look at MR signals
##    plt.figure()
##    for line_no,mr in zip(range(len(mrs)),mrs):
##        mr = np.abs(mr)
##        t_vals = np.arange(0,len(mr))*delta_t
##        plt.plot(t_vals*1e3,mr,label=str(line_no))
##    plt.xlabel('Time (ms)')
##    plt.ylabel('MR')
##    plt.legend()
##    plt.show()   
    # Save spectrum
    num_pe_samples = int(2*pe_sample_radius+1)
    num_fe_samples = int(2*fe_sample_radius+1)
    S = np.empty([num_pe_samples,num_fe_samples],dtype=complex)
    for line_no in range(num_pe_samples):
        S[line_no,:] = mrs[line_no]
    np.savetxt('S_real.csv',np.real(S),delimiter=',',fmt='%.5f')
    np.savetxt('S_imag.csv',np.imag(S),delimiter=',',fmt='%.5f')


