from pysim import Sim
from pyem import Em
from pygeneratesequence import generate_2DFT_sequence, see_2DFT_sequence
import matplotlib.pyplot as plt
import numpy as np

see_distribution = False
see_sequence = False
run_sim = True
save_fig = False
fig_name = 'density.pdf'

# Spread ems in Gaussian shape at z = 0
max_ems_grid_point = 20
grid_radius = 2
grid_dim = int(2*grid_radius+1)
em_positions = np.empty([0,3])
xlim = 5e-2
ylim = xlim
zlim = xlim
sigma = xlim/1.5
x_vals = np.linspace(-xlim,xlim,grid_dim)
y_vals = np.linspace(-ylim,ylim,grid_dim)
z_vals = np.linspace(-zlim,zlim,grid_dim)
rho = np.empty([grid_dim,grid_dim])
for i,x in zip(range(grid_dim),x_vals):
    for j,y in zip(range(grid_dim),y_vals):
        num_ems_here = int(np.exp(-(x**2+y**2)/(2*sigma**2))*max_ems_grid_point)
        positions = np.tile(np.array([[x,y,0.0]]),(num_ems_here,1))
        em_positions = np.concatenate((em_positions,positions))
        rho[i,j] = num_ems_here
if see_distribution:
    plt.figure()
    plt.imshow(rho,extent=[x_vals[0]*1e2,x_vals[-1]*1e2,y_vals[0]*1e2,y_vals[-1]*1e2])
    plt.colorbar(label='number of ems')
    plt.xlabel('X position (cm)')
    plt.ylabel('Y position (cm)')
    plt.show()

# Declare other em parameters
num_ems = em_positions.shape[0]
em_gyromagnetic_ratio = 2.675e8
main_field = 3.0
em_equilibrium_magnetization = 1.0
em_magnetizations = np.zeros([num_ems,3])
em_magnetizations[:,2] = em_equilibrium_magnetization
em_shielding_constants = np.zeros(num_ems)
em_velocities = np.zeros([num_ems,3],dtype=float)

# Compute the pulse sequence
Gz_amplitude = 10.0e-3
slice_width = 5e-2
delta_t = 1e-6
fe_sample_radius = int(grid_radius+2)
pe_sample_radius = int(grid_radius+2)
kx_max = grid_radius/(2*xlim) 
ky_max = grid_radius/(2*ylim)
kmove_time = 1e-3
adc_rate = 5e3
pulse_sequence = generate_2DFT_sequence(main_field,Gz_amplitude,slice_width,em_gyromagnetic_ratio,fe_sample_radius,pe_sample_radius,kx_max,ky_max,kmove_time,adc_rate,delta_t)
if see_sequence:
    see_2DFT_sequence(pulse_sequence)

# Run simulation
if run_sim:
    def T1_map(position):
        return 100.0
    def T2_map(position):
        return 100.0
    print('beginning sim with ' + str(num_ems) + ' ems')
    sim = Sim(em_magnetizations,em_positions,em_velocities,em_gyromagnetic_ratio,em_shielding_constants,em_equilibrium_magnetization,T1_map,T2_map,main_field,pulse_sequence)
    ems,mrs = sim.run_sim()

    num_pe_samples = int(2*pe_sample_radius+1)
    num_fe_samples = int(2*fe_sample_radius+1)
    S = np.empty([num_pe_samples,num_fe_samples],dtype=complex)
    for line_no in range(num_pe_samples):
        S[line_no,:] = mrs[line_no]
    np.savetxt('S_real.csv',np.real(S),delimiter=',',fmt='%.5f')
    np.savetxt('S_imag.csv',np.imag(S),delimiter=',',fmt='%.5f')

##        # Plot excitation
##        z = np.empty(num_ems,dtype=float)
##        m = np.empty(num_ems,dtype=complex)
##        for em_no in range(num_ems):
##            em = ems[em_no]
##            z[em_no] = em.r[2]
##            m[em_no] = em.mu[0]+em.mu[1]*1j
##        plt.figure()
##        plt.plot(z*1e2,np.abs(m),'o')
##        plt.xlabel('Z position (cm)')
##        plt.ylabel('Magnitude (a.u.)')
##        plt.title('Transverse magnetization profile')
##        plt.show()


