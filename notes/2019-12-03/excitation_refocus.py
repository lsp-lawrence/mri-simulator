from pysim import Sim
from pypulse import Pulse
import numpy as np
import matplotlib.pyplot as plt

debug = False

num_ems = 200
em_gyromagnetic_ratio = 2.675e8
main_field = 3.0

# Set em magnetizations to equilibrium value
em_equilibrium_magnetization = 1.0
em_magnetizations = np.zeros([num_ems,3])
em_magnetizations[:,2] = em_equilibrium_magnetization
em_shielding_constants = np.zeros(num_ems)

if debug:
    print(em_magnetizations)

# Place ems evenly spaced along z axis
z_width = 10e-2
z_positions = np.linspace(-z_width,z_width,num_ems)
em_positions = np.zeros([num_ems,3])
em_positions[:,2] = z_positions

if debug:
    print(em_positions)

# Make ems stationary
em_velocities = np.zeros([num_ems,3],dtype=float)

# Test different pulse durations
pulse_durations = np.array([1.0,1.5,2.0])*1e-3 # in ms

for pulse_duration in pulse_durations:

    # Declare excitation pulse
    delta_t = 1e-5 # 0.1 us
    t_vals = np.arange(0.0,pulse_duration,delta_t)
    t_vals_shifted = t_vals-pulse_duration/2.0
    sinc_scaling = 2.0*8.0/pulse_duration
    gaussian_std = pulse_duration/3.0
    B1x = np.sinc(sinc_scaling*t_vals_shifted)*np.exp(-(t_vals_shifted**2)/(2*gaussian_std**2)) # sinc pulse
    alpha_origin = em_gyromagnetic_ratio*sum(B1x)*delta_t
    desired_alpha_origin = 0.52
    B1x_amplitude_scaling = desired_alpha_origin/alpha_origin
    B1x = B1x*B1x_amplitude_scaling # rescale for desired tip angle
    pulse_length = len(B1x)
    B1y = np.zeros(pulse_length)
    Gz_amp = 5.0*1e-3 # T/m
    Gz = Gz_amp*np.ones(pulse_length)
    Gz_refocus = -Gz_amp*np.ones
    omega_rf = em_gyromagnetic_ratio*main_field # omega_rf = omega_0
    pulse = Pulse(mode='excite',delta_t=delta_t,B1x=B1x,B1y=B1y,Gz=Gz,omega_rf=omega_rf)

    # Compute theoretical magnetization profile
##    B1x_fft = np.roll(np.fft.fft(B1x),int(len(B1x)/2))
##    freq = np.fft.fftfreq(len(B1x),delta_t)
##    freq = np.fft.fftshift(freq)
    freqs = -(em_gyromagnetic_ratio/(2*np.pi))*Gz_amp*z_positions
    fft_sinc = em_gyromagnetic_ratio*B1x_amplitude_scaling/sinc_scaling*np.where(abs(freqs/sinc_scaling)<=0.5,1.0,0.0)
    gradient_phase_factor = np.exp(-1j*em_gyromagnetic_ratio*Gz_amp*z_positions*pulse_duration/2.0)
    lab_frame_phase_factor = np.exp(1j*omega_rf*pulse_duration)
    m_theory = 1j*em_equilibrium_magnetization*gradient_phase_factor*fft_sinc*lab_frame_phase_factor

    if debug:
##        plt.subplot(1,2,1)
##        plt.plot(t_vals,B1x)
##        plt.xlabel('Time (s)')
##        plt.ylabel('Amplitude')
##        plt.subplot(1,2,2)
##        plt.plot(z_theory,np.abs(m_theory))
##        plt.plot(z_theory,np.angle(m_theory))
##        plt.xlabel('Z Position (m)')
##        plt.ylabel('Transverse magnetization (a.u.)')
##        plt.legend(('Mag','Phase'))
##        plt.show()
        zlims = 10
        plt.figure(figsize=(4,6))
        plt.subplot(311)
        plt.plot(t_vals*1e3,B1x)
        plt.title('B1x')
        plt.xlabel('Time (s)')
        plt.subplot(312)
        plt.plot(z_positions*1e2,np.abs(m_theory))
        plt.title('Abs(m)')
        plt.xlabel('Z Position (cm)')
        plt.xlim([-zlims,zlims])
        plt.subplot(313)
        plt.plot(z_positions*1e2,np.angle(m_theory))
        plt.xlabel('Z Position (cm)')
        plt.title('Phase(m)')
        plt.xlim([-zlims,zlims])
        plt.tight_layout()
        plt.show()
        plt.close()

    if not debug:
        # Initialize simulation
        pulse_sequence = [pulse]
        sim = Sim(em_magnetizations,em_positions,em_velocities,em_gyromagnetic_ratio,em_shielding_constants,em_equilibrium_magnetization,main_field,pulse_sequence)

        # Run simulation
        ems, mrsignals = sim.run_sim()

        # Extract tranverse magnetization as a function of position
        z_pos_ems = np.empty(num_ems)
        m = np.empty(num_ems,dtype=complex)
        for em_no in range(num_ems):
            em = ems[em_no]
            z_pos_ems[em_no] = em.r[2]
            m[em_no] = em.mu[0] + em.mu[1]*1j

        # Plot
        plt.figure(figsize=(10,8))
        plt.subplot(211)
        plt.plot(z_pos_ems*1e2,np.abs(m))
        plt.plot(z_positions*1e2,np.abs(m_theory))
        plt.xlabel('Longitudinal position (cm)')
        plt.ylabel('Magnitude (a.u.)')
        plt.title('Magnitude of transverse magnetization, pulse duration ' + str(pulse_duration) + ' s')
        plt.legend(('Simulation','Theory'))
        plt.xlim([-5,5])
        plt.subplot(212)
        plt.plot(z_pos_ems*1e2,np.angle(m))
        plt.plot(z_positions*1e2,-np.angle(m_theory))
        plt.xlabel('Longitudinal position (cm)')
        plt.ylabel('Phase (rad)')
        plt.title('Phase of transverse magnetization, pulse duration ' + str(pulse_duration) + ' s')
        plt.legend(('Simulation','Theory'))
        plt.xlim([-5,5])
        plt.tight_layout()
        #plt.show()
        #plt.close()
        plt.savefig('excitation_negative-phase_pulse-duration-'+str(pulse_duration*1e3)+'.pdf')
        plt.close()
##        plt.close()
