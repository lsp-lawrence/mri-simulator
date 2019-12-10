import numpy as np
from pypulse import Pulse
import matplotlib.pyplot as plt

def generate_2DFT_sequence(main_field,Gz_amplitude,slice_width,gyromagnetic_ratio,fe_sample_radius,pe_sample_radius,kx_max,ky_max,kmove_time,adc_rate,delta_t,read_all):
    """Generates a 2DFT pulse sequence
    Params:
        main_field: (positive float) main magnetic field
        Gz_amplitude: (positive float) z-gradient amplitude
        slice_width: (postive float) width of the slice
        gyromagnetic_ratio: (float) gyromagnetic ratio of species to be imaged
        fe_sample_radius: (positive int) number of samples in frequency-encoding direction
        pe_sample_radius: (positive int) number of samples in phase-encoding direction
        kx_max: (positive float) maximum value of kx to sample
        ky_max: (positive float) maximum value of ky to sample
        kmove_time: (positive float) time to move to each location in kspace
        adc_rate: (positive float) sampling rate of ADC in Hz
        delta_t: (positive float) time step for simulation
        read_all: (bool) if true, then record the MR signal for all time steps during the read time
    Returns:
        pulse_sequence: (list of Pulse objects) a 2DFT pulse sequence
    """
    # Check params
    if not (isinstance(fe_sample_radius,int) and fe_sample_radius > 0):
        raise TypeError("fe_sample_radius must be a positive int")
    if not (isinstance(pe_sample_radius,int) and pe_sample_radius > 0):
        raise TypeError("pe_sample_radius must be a positive int")
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
    if not(isinstance(main_field,float) and main_field > 0):
        raise TypeError('main_field must be a positive float')
    if not (isinstance(Gz_amplitude,float) and Gz_amplitude > 0):
        raise TypeError('Gz_amplitude must be a positive float')
    if not(isinstance(slice_width,float) and slice_width > 0):
        raise TypeError('slice_width must be a positive float')
    if not (isinstance(read_all,bool)):
        raise TypeError("read_all must be a bool")
    # Create pulse sequence
    tip_angle = np.pi/2.0
    pulse_tip = generate_tipping_pulse(gyromagnetic_ratio,main_field,Gz_amplitude,slice_width,tip_angle,delta_t)
    kspace_pulses = generate_kspace_pulses(pulse_tip,fe_sample_radius,pe_sample_radius,kx_max,ky_max,kmove_time,adc_rate,gyromagnetic_ratio,delta_t,read_all)
    # pulse_relax
    pulse_sequence = []
    num_pe_samples = int(2*pe_sample_radius+1)
    for line_no in range(num_pe_samples):
        pulse_sequence.append(pulse_tip)
        pulse_sequence.append(kspace_pulses[line_no])
    return pulse_sequence

def generate_relaxation_pulse(T1_max,delta_t):
    """Returns a relaxation pulse
    Params:
        T1_max: (positive float) maximum T1 in image
        delta_t: (positive float) time step
    Returns:
        pulse_relax: (Pulse object) relaxation pulse
    """
    relax_length = int(5*T1_max/delta_t) # wait long enough to rebuild longitudinal magnetization
    Gx = np.zeros(relax_length)
    Gy = np.zeros(relax_length)
    Gz = np.zeros(relax_length)
    readout = np.zeros(relax_length,dtype=bool)
    pulse_relax = Pulse(mode='free',Gx=Gx,Gy=Gy,Gz=Gz,readout=readout,delta_t=delta_t)
    return pulse_relax

def generate_tipping_pulse(gyromagnetic_ratio,main_field,Gz_amplitude,slice_width,tip_angle,delta_t):
    """Generates the tipping pulse for a spin-echo sequence
    Params:
        gyromagnetic_ratio: (positive float) gyromagnetic ratio of species
        main_field: (positive float) main magnetic field
        Gz_amplitude: (positive float) z-gradient amplitude
        slice_width: (postive float) width of the slice
        tip_angle: (float \in (0,pi)) angle to tip
        delta_t: (positive float) time step
    Returns:
        pulse_tip: (list of Pulse objects) the pulse required to tip the magnetization in a slice
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

    return pulse_tip

def generate_kspace_pulses(pulse_tip,fe_sample_radius,pe_sample_radius,kx_max,ky_max,kmove_time,adc_rate,gyromagnetic_ratio,delta_t,read_all):
    """Generates the set of pulses necessary to sample desired region of kspace using Cartesian sampling
    Params:
        pulse_tip: (Pulse object) tipping pulse in spin-echo sequence
        fe_sample_radius: (positive int) radius of number of samples in frequency-encoding direction
        pe_sample_radius: (positive int) radius of number of samples in phase-encoding direction
        kx_max: (positive float) maximum value of kx to sample
        ky_max: (positive float) maximum value of ky to sample
        kmove_time: (positive float) time to move to each location in kspace
        adc_rate: (positive float) sampling rate of ADC in Hz
        gyromagnetic_ratio: (float) gyromagnetic ratio of species to be imaged
        delta_t: (positive float) time step for simulation
        read_all: (bool) if true, record the MR signal at all time steps during the read time
    Returns:
        kspace_pulses: (list of Pulse objects) set of readout pulses to sample given region of kspace
    Notes:
        kmove_time must be greater than half the tipping pulse time (see a 2DFT sequence diagram)
        the fact that num_fe_samples and num_pe_samples are odd numbers is necessary for shift_2DFT function
    """
    # Check params
    if not (isinstance(pulse_tip,Pulse)):
        raise TypeError("pulse_tip must be a Pulse object")
    if not (isinstance(fe_sample_radius,int) and fe_sample_radius > 0):
        raise TypeError("fe_sample_radius must be a positive int")
    if not (isinstance(pe_sample_radius,int) and pe_sample_radius > 0):
        raise TypeError("pe_sample_radius must be a positive int")
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
    if not (isinstance(read_all,bool)):
        raise TypeError("read_all must be a bool")
    if not (kmove_time > pulse_tip.length*delta_t/2.0):
        print("kmove_time: " + str(kmove_time*1e3))
        print("pulse_tip time: " + str(pulse_tip.length*delta_t*1e3/2.0))
        raise ValueError("kmove_time must be greater than half the tipping pulse time -- see a 2DFT sequence diagram")
    # Compute number of pe and fe samples
    num_fe_samples = int(2*fe_sample_radius+1)
    num_pe_samples =  int(2*pe_sample_radius+1)
    # Refocusing lobe in longitudinal direction
    pulse_tip_length = pulse_tip.length
    Gz_tip_amp = pulse_tip.Gz[0]
    pulse_refocus_length = int(pulse_tip_length/2)
    Gz_refocus_lobe = -Gz_tip_amp*np.ones(pulse_refocus_length)
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
    if read_all:
        readout = np.ones(kread_len,dtype=bool)
    else:
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
        Gz = np.concatenate((Gz_refocus_lobe,np.zeros(kread_len+kmove_len-pulse_refocus_length)))
        pulse = Pulse(mode='free',Gx=Gx,Gy=Gy,Gz=Gz,readout=readout,delta_t=delta_t)
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

def see_2DFT_sequence(pulse_sequence):
    """Creates a plot of the 2DFT pulse sequence
    Params:
        pulse_sequence: (list of Pulse objects) a 2DFT pulse sequence generated by generate_2DFT_sequence
    """
    # Plot pulse sequence
    plt.figure(figsize=(8,7))
    num_pe_samples = int(len(pulse_sequence)/2)
    delta_t = pulse_sequence[0].delta_t
    for line_no in range(num_pe_samples):
        pulse_index = line_no*2
        pulse_tip = pulse_sequence[pulse_index]
        t_tip = np.linspace(0.0,pulse_tip.length*delta_t,pulse_tip.length)
        pulse_kspace = pulse_sequence[pulse_index+1]
        t_kspace = np.linspace(0.0,pulse_kspace.length*delta_t,pulse_kspace.length)
        plt.subplot(421)
        plt.plot(t_tip*1e3,pulse_tip.Gz*1e4)
        plt.title('Tip-Gz')
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude (uT/cm)')
        plt.subplot(423)
        plt.plot(t_tip*1e3,pulse_tip.B1x*1e6)
        plt.title('Tip-B1x')
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude (uT)')
        plt.subplot(422)
        plt.plot(t_kspace*1e3,pulse_kspace.Gz*1e4)
        plt.title('Kspace-Gz')
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude (uT/cm)')
        plt.subplot(424)
        plt.plot(t_kspace*1e3,pulse_kspace.Gy*1e5)
        plt.title('Kspace-Gy')
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude (uT/mm)')
        plt.subplot(426)
        plt.plot(t_kspace*1e3,pulse_kspace.Gx*1e5)
        plt.title('Kspace-Gx')
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude (uT/mm)')
        plt.subplot(428)
        plt.plot(t_kspace*1e3,pulse_kspace.readout,'o')
        plt.title('Kspace-Read ADC?')
        plt.ylabel('T/F')
        plt.xlabel('Time (ms)')
    plt.tight_layout()
    plt.show()
