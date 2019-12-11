from pysim_memtest import Sim
from pyem import Em
from pygeneratesequence import generate_kspace_pulses,generate_tipping_pulse,generate_2DFT_sequence,see_readout_pulses
from pyreconstruct import reconstruct_from_2DFT
import matplotlib.pyplot as plt
import numpy as np
from memory_profiler import profile
from pypulse import Pulse

@profile(precision=10)
def my_func():
    grid_radius = 1
    fig_params = 'T1-100ms_T2-10ms_TE-4ms_TR-100ms_x-0.0_y-minus-0.5'
    see_sequence = False
    run_sim = True
    plot_sim_results = True
    reconstruction_name = 'reconstruction_'+fig_params+'.pdf'

    # Place ems
    r0 = np.array([[0.0,-0.5,0.0]])*1e-2
    em_positions = np.tile(r0,(8000,1))
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
    repetition_time = 100e-3
    gyromagnetic_ratio = em_gyromagnetic_ratio
    # Create pulse sequence
##    pulse_length = int(4e6)
##    Gx = np.zeros(pulse_length)
##    Gy = np.zeros(pulse_length)
##    Gz = np.zeros(pulse_length)
##    readout = np.zeros(pulse_length,dtype=bool)
##    delta_t = 1e-6
##    pulse_sequence = [Pulse(mode='free',Gx=Gx,Gy=Gy,Gz=Gz,readout=readout,delta_t=delta_t)]
    pulse_sequence = generate_2DFT_sequence(main_field,Gz_amplitude,slice_width,gyromagnetic_ratio,fe_sample_radius,pe_sample_radius,kx_max,ky_max,kmove_time,kread_time,repetition_time,delta_t,read_all)
    # Initialize simulation
    def T1_map(position): return T1_max
    def T2_map(position): return T2_max
    print('beginning sim with ' + str(num_ems) + ' ems')
    sim = Sim(em_magnetizations,em_positions,em_velocities,em_gyromagnetic_ratio,em_shielding_constants,em_equilibrium_magnetization,T1_map,T2_map,main_field,pulse_sequence)
    #ems,mr = sim.run_sim()


if __name__=='__main__':
    my_func()
