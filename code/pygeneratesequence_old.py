import numpy as np
from pypulse import Pulse

def generate_spin_echo_sequence():

    pulse_tip = generate_slice_select_pulse()
    

def generate_slice_select_pulses(gyromagnetic_ratio,main_field,Gz_amplitude,slice_width,tip_angle,delta_t):
    """Generates the pulses for a slice select
    Params:
        gyromagnetic_ratio: (positive float) gyromagnetic ratio of species
        main_field: (positive float) main magnetic field
        Gz_amplitude: (positive float) z-gradient amplitude
        slice_width: (postive float) width of the slice
        tip_angle: (float \in (0,pi)) angle to tip
        delta_t: (positive float) time step
    Returns:
        slice_select_pulses: (list of Pulse objects) the pulses required to excite a single slice
    """
    # Check params
    if not(isinstance(gyromagnetic_ratio,float) and gyromagnetic_ratio > 0):
        raise TypeError('gyromagnetic_ratio must be a positive float')
    if not(isinstance(main_field,float) and main_field > 0):
        raise TypeError('main_field must be a positive float')
    if not (isinstance(Gz_amplitude,float) and Gz_amplitude > 0):
        raise TypeError('Gz_amplitude must be a positive float')
    if not(isinstance(slice_width,float) and slice_width > 0):
        raise TypeError('slice_width must be a positive float')
    if not(isinstance(tip_angle,float) and tip_angle > 0 and tip_angle < np.pi):
        raise TypeError('tip_angle must be a float between 0 and \pi')
    if not(isinstance(delta_t,float) and delta_t > 0):
        raise TypeError('delta_t must be a positive float')
    # Tipping pulse
    Gz_tip_amp = Gz_amplitude
    pulse_tip_duration = 2*np.pi*8/(gyromagnetic_ratio*Gz_tip_amp*(slice_width/2.0))
    t_vals = np.arange(0.0,pulse_tip_duration,delta_t)
    t_vals_shifted = t_vals-pulse_tip_duration/2.0
    sinc_scaling = 2.0*8.0/pulse_tip_duration
    gaussian_std = pulse_tip_duration/3.0
    B1x_tip = np.sinc(sinc_scaling*t_vals_shifted)*np.exp(-(t_vals_shifted**2)/(2*gaussian_std**2)) # sinc pulse
    alpha_origin = gyromagnetic_ratio*sum(B1x_tip)*delta_t
    desired_alpha_origin = tip_angle
    B1x_amplitude_scaling = desired_alpha_origin/alpha_origin
    B1x_tip = B1x_tip*B1x_amplitude_scaling # rescale for desired tip angle
    pulse_tip_length = len(B1x_tip)
    B1y_tip = np.zeros(pulse_tip_length)
    Gz_tip = Gz_tip_amp*np.ones(pulse_tip_length)
    omega_rf = gyromagnetic_ratio*main_field # omega_rf = omega_0
    pulse_tip = Pulse(mode='excite',delta_t=delta_t,B1x=B1x_tip,B1y=B1y_tip,Gz=Gz_tip,omega_rf=omega_rf)
    # Refocusing pulse
    pulse_refocus_length = int(pulse_tip_length/2)
    Gx_refocus = np.zeros(pulse_refocus_length)
    Gy_refocus = np.zeros(pulse_refocus_length)
    Gz_refocus = -Gz_tip_amp*np.ones(pulse_refocus_length)
    readout = np.zeros(pulse_refocus_length,dtype=bool)
    pulse_refocus = Pulse(mode='free',delta_t=delta_t,Gx=Gx_refocus,Gy=Gy_refocus,Gz=Gz_refocus,readout=readout)

    return [pulse_tip,pulse_refocus]
    

def generate_kspace_pulses(num_fe_samples,num_pe_samples,kx_max,ky_max,kmove_time,adc_rate,gyromagnetic_ratio,delta_t):
    """Generates the set of pulses necessary to sample desired region of kspace using Cartesian sampling
    Params:
        num_fe_samples: (positive int) number of samples in frequency-encoding direction
        num_pe_samples: (positive int) number of samples in phase-encoding direction
        kx_max: (positive float) maximum value of kx to sample
        ky_max: (positive float) maximum value of ky to sample
        kmove_time: (positive float) time to move to each location in kspace
        adc_rate: (positive float) sampling rate of ADC in Hz
        gyromagnetic_ratio: (float) gyromagnetic ratio of species to be imaged
        delta_t: (positive float) time step for simulation
    Returns:
        kspace_pulses: (list of Pulse objects) set of readout pulses to sample given region of kspace
    """
    # Check params
    if not (isinstance(num_fe_samples,int) and num_fe_samples > 0):
        raise TypeError("num_fe_samples must be a positive int")
    if not (isinstance(num_pe_samples,int) and num_pe_samples > 0):
        raise TypeError("num_pe_samples must be a positive int")
    if not (isinstance(kx_max,float) and kx_max > 0):
        raise TypeError("kx_max must be a positive float")
    if not (isinstance(ky_max,float) and ky_max > 0):
        raise TypeError("ky_max must be a positive float")
    if not (isinstance(kmove_time,float) and kmove_time > 0):
        raise TypeError("kmove_time must be a positive float")
    if not (isinstance(adc_rate,float) and adc_rate > 0):
        raise TypeError("adc_rate must be a positive float")
    if not isinstance(gyromagnetic_ratio,float):
        raise TypeError("gyromagnetic_ratio must be a float")
    if not (isinstance(delta_t,float) and delta_t > 0):
        raise TypeError("delta_t must be a positive float")
    # Phase-encoding direction parameters
    delta_ky = (2*ky_max)/(num_pe_samples-1)
    # Frequency-encoding direction parameters
    T_adc = 1.0/adc_rate
    adc_step = int(T_adc/delta_t)
    # Common parameters
    kread_time = num_fe_samples*adc_step*delta_t
    kread_len = (num_fe_samples-1)*adc_step+1
    kmove_len = int(kmove_time/delta_t)
    # Compute pulses in frequency-encoding direction
    Gx_kmove_amp = -(2*np.pi/gyromagnetic_ratio)*kx_max/(kmove_time)
    Gx_kmove = Gx_kmove_amp*np.ones(kmove_len)
    Gx_kread_amp = (2*np.pi/gyromagnetic_ratio)*(2*kx_max)/(kread_len*delta_t)
    Gx_kread = Gx_kread_amp*np.ones(kread_len)
    Gx = np.concatenate((Gx_kmove,Gx_kread))
    # Readout indices
    readout = np.zeros(kread_len,dtype=bool)
    for fe_sample_no in range(num_fe_samples):
        fe_sample_index = fe_sample_no*adc_step
        readout[fe_sample_index] = True
    readout = np.concatenate((np.zeros(kmove_len,dtype=bool),readout))
    # Compute pulses in phase-encoding direction
    kspace_pulses = []
    for pe_sample_no in range(num_pe_samples):
        ky = ky_max - pe_sample_no*delta_ky
        # Gradients for movement to position in kspace
        Gy_kmove_amp = (2*np.pi/gyromagnetic_ratio)*ky/(kmove_time)
        Gy_kmove = Gy_kmove_amp*np.ones(kmove_len)
        # Gradients for readout movement
        Gy_kread = np.zeros(kread_len,dtype=float)
        # Gradients for sampling one line of kspace
        Gy = np.concatenate((Gy_kmove,Gy_kread))
        # Bundle into Pulse object
        pulse = Pulse(mode='free',Gx=Gx,Gy=Gy,readout=readout,delta_t=delta_t)
        kspace_pulses.append(pulse)
    return kspace_pulses

def determine_kspace_sampling(readout_pulses,gyromagnetic_ratio,delta_t):
    """Returns the grid of kx and ky samples from the readout pulses
    Params:
        readout_pulses: (list of Pulse objects) pulses for readout
    Returns:
        kx: (1D numpy array of floats) kx samples
        ky: (1D numpy array of floats) ky samples
    """
    # Check params
    if not (isinstance(readout_pulses,list) and all([isinstance(item,Pulse) for item in readout_pulses])):
        raise TypeError("readout_pulses must be a list of Pulse objects")
    # Extract ky samples
    num_pe_samples = len(readout_pulses)
    ky = np.empty(num_pe_samples)
    for pulse_no in range(num_pe_samples):
        pulse = readout_pulses[pulse_no]
        ky[pulse_no] = gyromagnetic_ratio/(2*np.pi)*np.sum(pulse.Gy)*delta_t
    # Extract kx samples
    pulse = readout_pulses[0]
    kx_trajectory = gyromagnetic_ratio/(2*np.pi)*np.cumsum(pulse.Gx)*delta_t
    kx = kx_trajectory[pulse.readout]
    return kx,ky
    
