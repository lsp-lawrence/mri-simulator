from pysim import Sim
from pyem import Em
from pygeneratesequence import generate_kspace_pulses,generate_tipping_pulse,generate_DWI_sequence,see_readout_pulses
from pyreconstruct import reconstruct_from_2DFT
import matplotlib.pyplot as plt
import numpy as np

fig_params = 'diffusion'
run_sim = False
plot_sim_results = True
reconstruction_name = 'reconstruction_'+fig_params+'.pdf'

# Place ems
em_positions = 1e-2*np.array([[0.5,0.5,0.0],
                              [0.5,0.0,0.0],
                              [0.5,-0.5,0.0],
                              [-0.5,0.5,0.0],
                              [-0.5,0.0,0.0],
                              [-0.5,-0.5,0.0]])
num_ems = em_positions.shape[0]
em_gyromagnetic_ratio = 2.675e8
main_field = 3.0
em_equilibrium_magnetization = 1.0
em_magnetizations = np.zeros([num_ems,3])
em_magnetizations[:,1] = 1.0
em_shielding_constants = np.zeros(num_ems)
em_velocities = np.zeros([num_ems,3],dtype=float)

# Compute the pulse sequence
grid_radius = 5
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
read_all = False
T1_max = 100e-3
T2_max = 10e-3
repetition_time = 100e-3
gyromagnetic_ratio = em_gyromagnetic_ratio
diffusion_gradients = np.ones(3)*5.0e-3
diffusion_gradient_time = 1e-3
diffusion_time = 4e-3
# Create pulse sequence
pulse_sequence = generate_DWI_sequence(main_field,Gz_amplitude,slice_width,gyromagnetic_ratio,fe_sample_radius,pe_sample_radius,kx_max,ky_max,kmove_time,kread_time,repetition_time,pulse_delta_t,read_all,diffusion_gradients,diffusion_gradient_time,diffusion_time)



if run_sim:
    # Run simulation
    def T1_map(position): return T1_max
    def T2_map(position): return T2_max
    def D_map(position):
        if position[0]>0:
            return np.array([5e-6,5e-6,0.0])
        else:
            return np.array([0.0,0.0,0.0])
    print('beginning sim with ' + str(num_ems) + ' ems')
    sim = Sim(em_magnetizations,em_positions,em_gyromagnetic_ratio,em_shielding_constants,em_equilibrium_magnetization,em_delta_t,T1_map,T2_map,main_field,pulse_sequence,D_map)
    ems,mrs = sim.run_sim()
    num_pe_samples = int(2*pe_sample_radius+1)
    num_fe_samples = int(2*fe_sample_radius+1)
    S = np.empty([num_pe_samples,num_fe_samples],dtype=complex)
    for line_no in range(num_pe_samples):
        S[line_no,:] = mrs[line_no]
    img,x,y = reconstruct_from_2DFT(S,kx_max,ky_max)
    np.savetxt('em-positions_'+fig_params+'.csv',em_positions,delimiter=',')
    np.savetxt('img_'+fig_params+'.csv',img,delimiter=',')
    np.savetxt('x_'+fig_params+'.csv',x,delimiter=',')
    np.savetxt('y_'+fig_params+'.csv',y,delimiter=',')

if plot_sim_results:
    em_positions_0 = 1e-2*np.array([[0.5,0.5,0.0],
                              [0.5,0.0,0.0],
                              [0.5,-0.5,0.0],
                              [-0.5,0.5,0.0],
                              [-0.5,0.0,0.0],
                              [-0.5,-0.5,0.0]])
    em_positions = np.loadtxt('em-positions_'+fig_params+'.csv',delimiter=',')
    if em_positions.ndim==1:em_positions=np.array([em_positions])
    img = np.loadtxt('img_'+fig_params+'.csv',delimiter=',')
    x = np.loadtxt('x_'+fig_params+'.csv',delimiter=',')
    y = np.loadtxt('y_'+fig_params+'.csv',delimiter=',')
    x = x*1e2
    y = y*1e2
    extent = (x[0],x[-1],y[0],y[-1])
    plt.figure(figsize=(8,4))
    plt.subplot(121)
    x_pos,y_pos = em_positions[:,0:2].T
    x_0,y_0 = em_positions_0[:,0:2].T
    plt.scatter(x_0*1e2,y_0*1e2,color='red',label='initial')
    plt.scatter(x_pos*1e2,y_pos*1e2,color='blue',label='final')
    plt.legend()
    plt.xticks(x,fontsize=9)
    plt.yticks(y,fontsize=9)
    plt.title('em distribution')
    plt.xlabel('x position (cm)')
    plt.ylabel('y position (cm)')
    plt.subplot(122)
    plt.imshow(img,extent=extent)
    plt.xticks(x,fontsize=9)
    plt.yticks(y,fontsize=9)
    plt.title('reconstruction')
    plt.xlabel('x position (cm)')
    plt.ylabel('y position (cm)')
    plt.tight_layout()
    plt.savefig(reconstruction_name)
    plt.show()


