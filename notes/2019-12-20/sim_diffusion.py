from pysim import Sim
from pyem import Em
from pygeneratesequence import *
from pyreconstruct import reconstruct_from_2DFT
import matplotlib.pyplot as plt
import numpy as np

num_ems = 1000
params = 'num-ems-'+str(num_ems)
fig_name = 'mr-b_'+params+'.pdf'

run_no = 6
run_sim = False
see_sequence = False
plot_sim_results = True

# Place ems
origin = np.array([0.0,0.0,0.0])
em_positions = np.tile(origin,(num_ems,1))
em_gyromagnetic_ratio = 2.675e8
main_field = 3.0
em_equilibrium_magnetization = 1.0
em_magnetizations = np.zeros([num_ems,3])
em_magnetizations[:,1] = 1.0
em_shielding_constants = np.zeros(num_ems)
em_velocities = np.zeros([num_ems,3],dtype=float)

# Other parameters
grid_radius = 5
Gz_amplitude = 10.0e-3
slice_width = 5e-2
pulse_delta_t = 1e-6
em_delta_t = 1e-6
xlim = 1e-2
ylim = xlim
kread_time = 8e-3
T1_max = 1e6
T2_max = 1e6
gyromagnetic_ratio = 2.675e8
main_field = 3.0

# Diffusion weighting parameters
diffusion_gradient_time = 0.5e-3
diffusion_time = 4e-3
num_b_vals = 5
b_vals = np.linspace(0.0,1500.0,num_b_vals)*1e6
G_diff_mags = (1/(em_gyromagnetic_ratio*diffusion_gradient_time))*np.sqrt(b_vals/(diffusion_time-diffusion_gradient_time))

if run_sim:

    S = np.empty(num_b_vals)
    
    for b_val_no in range(num_b_vals):
        # Diffusion gradients
        diffusion_gradients = np.array([1.0,0.0,0.0])*G_diff_mags[b_val_no]
        # Create pulse sequence
        pulse_sequence = generate_DW_sequence(main_field,Gz_amplitude,slice_width,gyromagnetic_ratio,kread_time,pulse_delta_t,diffusion_gradients,diffusion_gradient_time,diffusion_time)
        if see_sequence:
            see_readout_pulses(pulse_sequence)
            plt.close()

        # Run simulation
        D_water = 3e-9
        def T1_map(position): return T1_max
        def T2_map(position): return T2_max
        def D_map(position): return np.array([D_water,0.0,0.0])
        print('beginning sim with b = ' + str(b_vals[b_val_no]))
        sim = Sim(em_magnetizations,em_positions,em_gyromagnetic_ratio,em_shielding_constants,em_equilibrium_magnetization,em_delta_t,T1_map,T2_map,main_field,pulse_sequence,D_map)
        ems,mrs = sim.run_sim()
        mr = mrs[0]
        S[b_val_no] = np.abs(mr[0])

    # Save MR signal versus b-value data
    dw_data = np.array([b_vals,S]).transpose()
    np.savetxt('dw-data_' + params + '.csv',dw_data,delimiter=',')
    # Save em positions from final run
    em_positions = np.empty([num_ems,3])
    for em_no in range(len(ems)):
        em_positions[em_no,:] = ems[em_no].r
    np.savetxt('final-em-positions_' + params + '.csv',em_positions,delimiter=',')

if plot_sim_results:
    plt.figure(figsize=(8,4))
    plt.subplot(121)
    dw_data = np.loadtxt('dw-data_' + params + '.csv',delimiter=',')
    b_vals = dw_data[:,0]
    S = dw_data[:,1]/dw_data[0,1]
    plt.plot(b_vals*1e-6,S,'o',label='data')
    p = np.polyfit(b_vals,np.log(S),deg=1)
    b_vals_fit = np.linspace(b_vals[0],b_vals[-1],100)
    S_fit = np.exp(b_vals_fit*p[0])
    plt.plot(b_vals_fit*1e-6,S_fit,'--',label='ADC fit')
    print('ADC = ' + str(-p[0]))
    plt.title('mr signal amplitude versus b-value')
    plt.xlabel('b value (mm$^2$/s)')
    plt.ylabel('amplitude (a.u.)')
    plt.legend()
    plt.subplot(122)
    em_positions = np.loadtxt('final-em-positions_' + params + '.csv',delimiter=',')*1e2
    plt.hist(em_positions[:,0],bins=15)
    plt.xlabel('position (cm)')
    plt.ylabel('number of ems')
    plt.title('histogram of final em x-positions')
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.show()


