from pysim import Sim
from pyem import Em
from pygeneratesequence import *
from pyreconstruct import reconstruct_from_2DFT
import matplotlib.pyplot as plt
import numpy as np

run_sim = True
plot_sim_results = True
fig_name = 'mr_vs_b.pdf'

# Place ems
origin = np.array([0.0,0.0,0.0])
em_positions = np.tile(origin,(100,1))
num_ems = em_positions.shape[0]
em_gyromagnetic_ratio = 2.675e8
main_field = 3.0
em_equilibrium_magnetization = 1.0
em_magnetizations = np.zeros([num_ems,3])
em_magnetizations[:,1] = 1.0
em_shielding_constants = np.zeros(num_ems)
em_velocities = np.zeros([num_ems,3],dtype=float)

# Diffusion weighting parameters
diffusion_gradient_time = 0.5e-3
diffusion_time = 4e-3
num_b_vals = 2
b_vals = np.linspace(0.0,1500.0,num_b_vals)
G_diff_mags = (1/(em_gyromagnetic_ratio*diffusion_gradient_time))*np.sqrt(b_vals/(diffusion_time-diffusion_gradient_time))

if run_sim:

    mr_store = []
    ems_store = []
    
    for b_val_no in range(num_b_vals):
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
        diffusion_gradients = np.array([1.0,0.0,0.0])*G_diff_mags[b_val_no]
        # Create pulse sequence
        pulse_sequence = generate_DW_sequence(main_field,Gz_amplitude,slice_width,gyromagnetic_ratio,kread_time,pulse_delta_t,diffusion_gradients,diffusion_gradient_time,diffusion_time)
        #see_readout_pulses(pulse_sequence)

        # Run simulation
        D_water = 3e-3
        def T1_map(position): return T1_max
        def T2_map(position): return T2_max
        def D_map(position): return np.array([D_water,0.0,0.0])
        print('beginning sim with b = ' + str(b_vals[b_val_no]))
        sim = Sim(em_magnetizations,em_positions,em_gyromagnetic_ratio,em_shielding_constants,em_equilibrium_magnetization,em_delta_t,T1_map,T2_map,main_field,pulse_sequence,D_map)
        ems,mr = sim.run_sim()
        mr_store.append(mr)

if plot_sim_results:
    plt.figure()
    for b_val_no in range(num_b_vals):
        plt.plot(mr_store[b_val_no],label='b='+str(b_vals[b_val_no]))
        plt.xlabel('sample number')
        plt.ylabel('amplitude')
    plt.title('mr signal for various b-values')
    plt.legend()
    plt.savefig(fig_name)
    plt.show()


