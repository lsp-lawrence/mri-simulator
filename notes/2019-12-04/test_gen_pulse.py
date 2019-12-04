from pysim import Sim
from pypulse import Pulse
import numpy as np
import matplotlib.pyplot as plt
from pygeneratesequence import generate_slice_select_pulses

see_pulse = False
run_simulation = True
sim_name = 'slice_select_90degree_small_delta_t.pdf'
save_fig = True

num_ems = 1000
em_gyromagnetic_ratio = 2.675e8
main_field = 3.0

# Set em magnetizations to equilibrium value
em_equilibrium_magnetization = 1.0
em_magnetizations = np.zeros([num_ems,3])
em_magnetizations[:,2] = em_equilibrium_magnetization
em_shielding_constants = np.zeros(num_ems)

# Place ems evenly spaced along z axis
z_width = 10e-2
z_positions = np.linspace(-z_width,z_width,num_ems)
delta_z = z_positions[1]-z_positions[0]
em_positions = np.zeros([num_ems,3])
em_positions[:,2] = z_positions

# Make ems stationary
em_velocities = np.zeros([num_ems,3],dtype=float)

# Generate slice select sequence
gyromagnetic_ratio = em_gyromagnetic_ratio
Gz_amplitude = 5.0e-3
slice_width = 5e-2
tip_angle = np.pi/2.0
delta_t = 1e-7
pulse_sequence = generate_slice_select_pulses(gyromagnetic_ratio,main_field,Gz_amplitude,slice_width,tip_angle,delta_t)

### Compute theoretical magnetization profile
##freqs = -(em_gyromagnetic_ratio/(2*np.pi))*Gz_tip_amp*z_positions
##fft_sinc = em_gyromagnetic_ratio*B1x_amplitude_scaling/sinc_scaling*np.where(abs(freqs/sinc_scaling)<=0.5,1.0,0.0)
##lab_frame_phase_factor = np.exp(-1j*omega_rf*pulse_tip_duration)
##m_theory = 1j*em_equilibrium_magnetization*fft_sinc*lab_frame_phase_factor

# Initialize simulation
# Long relaxation times
def T1_map(position):
    return 100.0
def T2_map(position):
    return 100.0
sim = Sim(em_magnetizations,em_positions,em_velocities,em_gyromagnetic_ratio,em_shielding_constants,em_equilibrium_magnetization,T1_map,T2_map,main_field,pulse_sequence)

# Run simulation
ems, mrsignals = sim.run_sim()

# Extract tranverse magnetization as a function of position
m = np.empty(num_ems,dtype=complex)
for em_no in range(num_ems):
    em = ems[em_no]
    m[em_no] = em.mu[0] + em.mu[1]*1j

# Plot
plt.figure()
plt.subplot(211)
plt.plot(z_positions*1e2,np.abs(m))
plt.title('Magnitude of transverse magnetization')
plt.xlabel('Longitudinal position (cm)')
plt.ylabel('Magnetization (a.u.)')
plt.subplot(212)
plt.plot(z_positions*1e2,np.angle(m))
plt.title('Phase of transverse magnetization')
plt.xlabel('Longitudinal position (cm)')
plt.ylabel('Phase (rad)')
plt.xlim([-10,10])
plt.tight_layout()
if save_fig:
    plt.savefig(sim_name)
else:
    plt.show()
plt.close()
