from pysim import Sim
from pypulse import Pulse
import numpy as np
import matplotlib.pyplot as plt

see_pulse = True
pulse_name = 'excitation_refocus_pulse.pdf'
run_simulation = False
sim_name = 'excitation_refocus.pdf'
save_fig = False

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

# Test different pulse durations
#pulse_tip_durations = np.array([1.0,1.5,2.0])*1e-3 # in ms
pulse_tip_duration = 1.0e-3

# Declare pulse sequence
# Params
delta_t = 1.0e-6
t_vals = np.arange(0.0,pulse_tip_duration,delta_t)
t_vals_shifted = t_vals-pulse_tip_duration/2.0
sinc_scaling = 2.0*8.0/pulse_tip_duration
gaussian_std = pulse_tip_duration/3.0

# Tipping pulse
B1x_tip = np.sinc(sinc_scaling*t_vals_shifted)*np.exp(-(t_vals_shifted**2)/(2*gaussian_std**2)) # sinc pulse
alpha_origin = em_gyromagnetic_ratio*sum(B1x_tip)*delta_t
desired_alpha_origin = 0.52
B1x_amplitude_scaling = desired_alpha_origin/alpha_origin
B1x_tip = B1x_tip*B1x_amplitude_scaling # rescale for desired tip angle
pulse_tip_length = len(B1x_tip)
B1y_tip = np.zeros(pulse_tip_length)
Gz_tip_amp = 5.0*1e-3 # T/m
Gz_tip = Gz_tip_amp*np.ones(pulse_tip_length)
omega_rf = em_gyromagnetic_ratio*main_field # omega_rf = omega_0
pulse_tip = Pulse(mode='excite',delta_t=delta_t,B1x=B1x_tip,B1y=B1y_tip,Gz=Gz_tip,omega_rf=omega_rf)

# Refocusing pulse
pulse_refocus_length = int(pulse_tip_length/2)
Gx_refocus = np.zeros(pulse_refocus_length)
Gy_refocus = np.zeros(pulse_refocus_length)
Gz_refocus = -Gz_tip_amp*np.ones(pulse_refocus_length)
Gz_no_refocus = np.zeros(pulse_refocus_length)
readout = np.zeros(pulse_refocus_length,dtype=bool)
pulse_refocus = Pulse(mode='free',delta_t=delta_t,Gx=Gx_refocus,Gy=Gy_refocus,Gz=Gz_refocus,readout=readout)
pulse_no_refocus = Pulse(mode='free',delta_t=delta_t,Gx=Gx_refocus,Gy=Gy_refocus,Gz=Gz_no_refocus,readout=readout)

# Compute theoretical magnetization profile
freqs = -(em_gyromagnetic_ratio/(2*np.pi))*Gz_tip_amp*z_positions
fft_sinc = em_gyromagnetic_ratio*B1x_amplitude_scaling/sinc_scaling*np.where(abs(freqs/sinc_scaling)<=0.5,1.0,0.0)
lab_frame_phase_factor = np.exp(-1j*omega_rf*pulse_tip_duration)
m_theory = 1j*em_equilibrium_magnetization*fft_sinc*lab_frame_phase_factor

if see_pulse: # Plots full pulse sequence
    t_vals_full = np.concatenate((t_vals,np.linspace(0.0,delta_t*pulse_refocus_length,pulse_refocus_length)+t_vals[-1]+delta_t))
    B1x_full = np.concatenate((B1x_tip,np.zeros(pulse_refocus_length)))
    Gz_full_refocus = np.concatenate((Gz_tip,Gz_refocus))
    Gz_full_no_refocus = np.concatenate((Gz_tip,Gz_no_refocus))
    plt.figure()
    plt.subplot(211)
    plt.plot(t_vals_full*1e3,B1x_full*1e6)
    plt.title('B1x')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude (uT)')
    plt.subplot(212)
    plt.plot(t_vals_full*1e3,Gz_full_refocus*1e4)
    plt.title('Gz')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude (uT/cm)')
    plt.tight_layout()
    if save_fig:
        plt.savefig(pulse_name)
    else:
        plt.show()
    plt.close()

if run_simulation: # Runs simulation with and without refocus
    # Initialize simulation
    pulse_sequence = [pulse_tip,pulse_refocus]
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
    plt.plot(z_positions*1e2,np.abs(m_theory))
    plt.title('Magnitude of transverse magnetization')
    plt.xlabel('Longitudinal position (cm)')
    plt.ylabel('Magnetization (a.u.)')
    plt.legend(('Simulation','Theory'))
    plt.subplot(212)
    plt.plot(z_positions*1e2,np.angle(m))
    plt.plot(z_positions*1e2,np.angle(m_theory))
    plt.title('Phase of transverse magnetization')
    plt.xlabel('Longitudinal position (cm)')
    plt.ylabel('Phase (rad)')
    plt.legend(('Simulation','Theory'))
    plt.xlim([-10,10])
    plt.tight_layout()
    if save_fig:
        plt.savefig(sim_name)
    else:
        plt.show()
    plt.close()
