from pysim import Sim
from pypulse import Pulse
import numpy as np
import matplotlib.pyplot as plt

debug = True

num_ems = 200
em_gyromagnetic_ratio = 1.0
main_field = 10.0

# Set em magnetizations to equilibrium value
em_equilibrium_magnetization = 1.0
em_magnetizations = np.zeros([num_ems,3])
em_magnetizations[:,2] = em_equilibrium_magnetization

if debug:
    print(em_magnetizations)

# Place ems evenly spaced along z axis
z_width = 10
z_positions = np.linspace(-z_width,z_width,num_ems)
em_positions = np.zeros([num_ems,3])
em_positions[:,2] = z_positions

if debug:
    print(em_positions)

# Make ems stationary
em_velocities = np.zeros([num_ems,3],dtype=float)

# Test different pulse durations
pulse_durations = [5.0,10.0,20.0]

for pulse_duration in pulse_durations:

    # Declare excitation pulse
    delta_t = 0.01
    t_vals = np.arange(0.0,pulse_duration,delta_t)
    t_vals_shifted = t_vals-pulse_duration/2.0
    B1x = np.sinc(pulse_duration*t_vals_shifted)*np.exp(-pulse_duration/10*(t_vals_shifted**2))
    pulse_length = len(B1x)
    B1y = np.zeros(pulse_length)
    Gz_amp = 20.0
    Gz = Gz_amp*np.ones(pulse_length)
    omega_rf = em_gyromagnetic_ratio*main_field # omega_rf = omega_0
    pulse = Pulse(mode='excite',delta_t=delta_t,B1x=B1x,B1y=B1y,Gz=Gz,omega_rf=omega_rf)

    # Compute theoretical magnetization profile
    B1x_fft = np.roll(abs(np.fft.fft(B1x)),int(len(B1x)/2))/np.sqrt(len(B1x))
    freq = np.fft.fftfreq(len(B1x),delta_t)
    freq = np.fft.fftshift(freq)
    z_theory = -2*np.pi/em_gyromagnetic_ratio*(1.0/Gz_amp)*freq
    m_theory = em_equilibrium_magnetization*B1x_fft

    # Compute theoretical tip angle at origin
    alpha_origin = em_gyromagnetic_ratio*sum(B1x)*delta_t
    print(alpha_origin)

    if debug:
        plt.subplot(1,2,1)
        plt.plot(t_vals,B1x)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.subplot(1,2,2)
        plt.plot(z_theory,m_theory)
        plt.xlabel('Z Position (m)')
        plt.ylabel('Norm of transverse magnetization (a.u.)')
        plt.show()

    if not debug:
        # Initialize simulation
        pulse_sequence = [pulse]
        sim = Sim(em_magnetizations,em_positions,em_velocities,em_gyromagnetic_ratio,em_equilibrium_magnetization,main_field,pulse_sequence)

        # Run simulation
        ems, mrsignals = sim.run_sim()

        # Extract tranverse magnetization as a function of position
        z_pos_ems = np.empty(num_ems)
        m_norm = np.empty(num_ems)
        for em_no in range(num_ems):
            em = ems[em_no]
            z_pos_ems[em_no] = em.r[2]
            m_norm[em_no] = np.sqrt(em.mu[0]**2 + em.mu[1]**2)

        # Plot
        plt.plot(z_pos_ems,m_norm)
        plt.plot(z_theory,m_theory)
        plt.xlabel('Longitudinal position (m)')
        plt.ylabel('Norm of transverse magnetization (a.u.)')
        plt.legend(('Simulation','Theory'))
        plt.savefig('excitation_pulse-duration-'+str(pulse_duration)+'.pdf')
        plt.close()
