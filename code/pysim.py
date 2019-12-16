from pypulse import Pulse
from pyem import Em
import numpy as np

class Sim:
    """The main MR simulation object"""

    def __init__(self,em_magnetizations,em_positions,em_gyromagnetic_ratio,em_shielding_constants,em_equilibrium_magnetization,em_delta_t,T1_map,T2_map,main_field,pulse_sequence,D_map=None,conversion_dict={}):
        """Initializes the MR simulation
        Params:
            em_magnetizations: (num_ems*3 numpy array of float) the initial magnetizations of the ems
            em_positions: (num_ems*3 numpy array of floats) the initial positions of the ems
            em_gyromagnetic_ratio: (float) the gyromagnetic ratio of the ems
            em_shiedling_constants: (1D numpy array of nonnegative floats) the shielding constants of the ems
            em_equilibrium_magnetization: (positive float) the equilibrium magnetization of the ems
            em_delta_t: (positive float) the time step the ems use for diffusion, metabolic conversion, etc.
            T1_map: (function) mapping from position to T1 value
            T2_map: (function) mapping from position to T2 value
            main_field: (positive float) the main field strength
            pulse_sequence: (list of Pulse objects) the pulse sequence
            D_map: (function) mapping from position to diffusion coefficient; if None then no diffusion occurs
            conversion_dict: (dictionary) dictionary of metabolic conversion parameters
        """
        # Check params
        if (em_magnetizations.dtype != np.float64) or (em_magnetizations.ndim != 2) or (em_magnetizations.shape[1] != 3):
            raise TypeError("em_magnetizations must be a num_ems*3 numpy array of floats")
        if (em_positions.dtype != np.float64) or (em_positions.ndim != 2) or (em_positions.shape[1] != 3):
            raise TypeError("em_positions must be a num_ems*3 numpy array of floats")
        if not (em_magnetizations.shape[0] == em_positions.shape[0]):
            raise ValueError("em_magnetizations and em_positions must have the same number of rows (num_ems)")
        if not isinstance(em_gyromagnetic_ratio,float):
            raise TypeError("em_gyromagnetic_ratio must be a float")
        if not (em_shielding_constants.dtype == np.float64 and em_shielding_constants.ndim == 1 and all([item >= 0.0 for item in em_shielding_constants])):
            raise TypeError("em_shielding_constants must be a 1D numpy array of positive floats")
        if (not isinstance(em_equilibrium_magnetization,float)) or (em_equilibrium_magnetization <= 0):
            raise TypeError("em_equilibrium_magnetization must be a positive float")
        if (not isinstance(em_delta_t,float) and em_delta_t > 0):
            raise TypeError("em_delta_t must be a positive float")
        if not(em_delta_t <= min([pulse_sequence[pulse_no].delta_t for pulse_no in range(len(pulse_sequence))])):
            raise ValueError('em_delta_t must be less than or equal to the minimum time step of the pulse sequence')
        T_0 = (2*np.pi)/(em_gyromagnetic_ratio*main_field)
        if not(all([np.mod(pulse_sequence[pulse_no].delta_t,em_delta_t)<1e-9*T_0 for pulse_no in range(len(pulse_sequence))])):
            raise ValueError('em_delta_t must be a divisor of all time step durations in the pulse sequence')
        if (not isinstance(conversion_dict,dict)):
            raise TypeError('conversion_dict must be a dictionary') 
        if (not isinstance(main_field,float)) or (main_field <= 0):
            raise TypeError("main_field must be a positive float")
        if (not isinstance(pulse_sequence,list)) or (not all([isinstance(item,Pulse) for item in pulse_sequence])):
            raise TypeError("pulse_sequence must be a list of Pulse objects")
        # Store data attributes
        self.gamma = em_gyromagnetic_ratio
        self.num_ems = em_positions.shape[0]
        self.em_delta_t = em_delta_t
        self.T1_map = T1_map
        self.T2_map = T2_map
        self.diffusion_on = D_map != None
        self.D_map = D_map
        self.B0 = main_field
        self.pulse_sequence = pulse_sequence
        self.metabolism_on = len(conversion_dict)>0
        if self.metabolism_on:
            self.metabolism = conversion_dict
            # Convert metabolism rates for time step of simulation
            self.metabolism['pyr_2_lac_rate'] = self.metabolism['pyr_2_lac_rate']*em_delta_t
            self.metabolism['lac_2_pyr_rate'] = self.metabolism['lac_2_pyr_rate']*em_delta_t
        # Instantiate ems
        self.ems = []
        for em_no in range(self.num_ems):
            magnetization = em_magnetizations[em_no,:]
            position = em_positions[em_no,:]
            shielding_constant = em_shielding_constants[em_no]
            self.ems.append(Em(magnetization,position,em_gyromagnetic_ratio,shielding_constant,em_equilibrium_magnetization))

    def run_sim(self):
        """Runs the simulation
        Returns:
            ems: (list of Em objects) em objects at final time point
            mr: (list of 1D numpy arrays of floats) MR signals from acquisition
        """
        mrs = []
        sequence_length = len(self.pulse_sequence)
        for pulse_no,pulse in zip(range(sequence_length),self.pulse_sequence):
            print('applying pulse number ' + str(pulse_no+1) + '/' + str(sequence_length) + ' with mode = ' + pulse.mode)
            mr_signal = self._apply_pulse(pulse)
            if pulse.signal_collected:
                mr_signal = self._demodulate(mr_signal,pulse)
                mrs.append(mr_signal)
        ems = self.ems
        return ems, mrs
    
    def _find_T1(self,position):
        """Returns the T1 at given position
        Params:
            position: (numpy 3-vector of floats) position
        Returns:
            T1: T1 relaxation time at position
        """
        # Check params
        if (position.dtype != np.float64) or (position.shape != (3,)):
            raise TypeError("position must be a numpy 3-vector of floats")
        # Return T1
        return self.T1_map(position)

    def _find_T2(self,position):
        """Returns the T2 at given position
        Params:
            position: (numpy 3-vector of floats) position
        Returns:
            T2: T2 relaxation time at position
        """
        # Check params
        if (position.dtype != np.float64) or (position.shape != (3,)):
            raise TypeError("position must be a numpy 3-vector of floats")
        # return T2
        return self.T2_map(position)

    def _find_D(self,position):
        """Returns the diffusion coefficient at given position
        Params:
            position: (numpy 3-vector of floats) position
        Returns:
            diffusion_coefficients: diffusion coefficients at position
        """
        if self.diffusion_on:
            # Check params
            if (position.dtype != np.float64) or (position.shape != (3,)):
                raise TypeError("position must be a numpy 3-vector of floats")
            # Return D
            diffusion_coefficients = self.D_map(position)
            return diffusion_coefficients

    def _find_Bz(self,position,Gx,Gy,Gz):
        """Returns the longitudinal field at a given position with applied gradients
        Params:
            position: (numpy 3-vector of floats) position
            Gx: (float) gradient in x direction at position
            Gy: (float) gradient in y direction at position
            Gz: (float) gradient in z direction at position
        Returns:
            Bz: field in longitudinal direction at position
        """
        # Check params
        if (position.dtype != np.float64) or (position.shape != (3,)):
            raise TypeError("position must be a numpy 3-vector of floats")
        if not isinstance(Gx,float):
            raise TypeError("Gx must be a float")
        if not isinstance(Gy,float):
            raise TypeError("Gy must be a float")
        if not isinstance(Gz,float):
            raise TypeError("Gz must be a float")
        # Main
        return self.B0 + Gx*position[0] + Gy*position[1] + Gz*position[2]

    def _apply_pulse(self,pulse):
        """Simulates the effect of a pulse on the ems
        Params:
            pulse: (Pulse) the applied pulse
        Returns:
            mr_signal: mr signal collected during pulse
        """
        # Check params
        if not isinstance(pulse,Pulse):
            raise TypeError("pulse must be a Pulse object")
        # Main
        time_steps_per_sample = int(pulse.delta_t/self.em_delta_t)
        if pulse.mode == 'free':
            mr_signal = np.zeros(np.sum(pulse.readout),dtype=np.complex)
            mr_signal_index = 0
            for pulse_sample_no in range(pulse.length):
                if pulse.readout[pulse_sample_no]:
                    mr_signal[mr_signal_index] = self._readout()
                    mr_signal_index = mr_signal_index + 1
                for time_step_no in range(time_steps_per_sample):
                    # Interpolate gradients
                    if pulse_sample_no == (pulse.length-1):
                        Gx = pulse.Gx[pulse_sample_no]
                        Gy = pulse.Gy[pulse_sample_no]
                        Gz = pulse.Gz[pulse_sample_no]
                    else:
                        Gx = (pulse.Gx[pulse_sample_no]*(time_steps_per_sample-time_step_no) + pulse.Gx[pulse_sample_no+1]*time_step_no)/(time_steps_per_sample)
                        Gy = (pulse.Gy[pulse_sample_no]*(time_steps_per_sample-time_step_no) + pulse.Gy[pulse_sample_no+1]*time_step_no)/(time_steps_per_sample)
                        Gz = (pulse.Gz[pulse_sample_no]*(time_steps_per_sample-time_step_no) + pulse.Gz[pulse_sample_no+1]*time_step_no)/(time_steps_per_sample)
                    # Simulate physics of ems
                    for em in self.ems:
                        # Motion
                        r_avg = em.r
                        if self.diffusion_on:
                            em.diffuse(self._find_D(em.r),self.em_delta_t)
                            r_avg = (r_avg+em.r)/2.0
                        # Metabolic conversion
                        if self.metabolism_on:
                            if (np.random.random() < self.metabolism['pyr_2_lac_rate']): em.set_shielding_constant(self.metabolism['lac_sigma'])
                            elif (np.random.random() < self.metabolism['lac_2_pyr_rate']): em.set_shielding_constant(self.metabolism['pyr_sigma'])
                        # Precession about main field
                        T1 = self._find_T1(r_avg)
                        T2 = self._find_T2(r_avg)
                        Bz = self._find_Bz(r_avg,Gx,Gy,Gz)
                        em.precess_and_relax(T1,T2,Bz,self.em_delta_t)
        elif pulse.mode == 'excite': # Rotation of magnetization of each em only applied at the end of the excitation pulse, and no relaxation occurs during excitation
            mr_signal = None
            for pulse_sample_no in range(pulse.length):
                for time_step_no in range(time_steps_per_sample):
                    # Interpolate gradients
                    if pulse_sample_no == (pulse.length-1):
                        Gz = pulse.Gz[pulse_sample_no]
                    else:
                        Gz = (pulse.Gz[pulse_sample_no]*(time_steps_per_sample-time_step_no) + pulse.Gz[pulse_sample_no+1]*time_step_no)/(time_steps_per_sample)
                    # Simulate physics of ems
                    for em in self.ems:
                        # Motion
                        r_avg = em.r
                        if self.diffusion_on:
                            em.diffuse(self._find_D(em.r),self.em_delta_t)
                            r_avg = (r_avg+em.r)/2.0
                        # Metabolic conversion
                        if self.metabolism_on:
                            if (np.random.random() < self.metabolism['pyr_2_lac_rate']): em.set_shielding_constant(self.metabolism['lac_sigma'])
                            elif (np.random.random() < self.metabolism['lac_2_pyr_rate']): em.set_shielding_constant(self.metabolism['pyr_sigma'])
                        # Precession about main field + RF field
                        Bz = self._find_Bz(r_avg,0.0,0.0,Gz)
                        em.update_flip_quaternion(pulse.B1x[pulse_sample_no],pulse.B1y[pulse_sample_no],Bz,pulse.omega_rf,self.em_delta_t)
            # Apply flip
            pulse_duration = pulse.delta_t*pulse.length
            for em in self.ems:
                em.flip(pulse.omega_rf,pulse_duration)
        return mr_signal   

    def _readout(self):
        """Computes the MR signal by summing transverse magnetization of all ems
        Returns:
            mr_signal: the MR signal
        """
        mr_signal = 0.0+1j*0.0
        for em in self.ems:
            mr_signal = mr_signal + em.mu[0] + 1j*em.mu[1]
        return mr_signal

    def _demodulate(self,mr_signal,pulse):
        """Demodulates the MR signal
        Params:
            mr_signal: (1D numpy array of complex) MR signal acquired
            pulse: (Pulse object) pulse used to generate mr_signal
        Returns:
            mr_baseband: (1D numpy array of complex) demodulated MR signal
        """
        # Check params
        if not(mr_signal.ndim == 1 and mr_signal.dtype == np.complex):
            raise TypeError("mr_signal must be a 1D numpy array of complex")
        if not(isinstance(pulse,Pulse)):
            raise TypeError("pulse must be a Pulse object")
        # Demodulate MR signal
        num_samples = len(mr_signal)
        mr_baseband = np.empty(num_samples,dtype=complex)
        omega_0 = self.gamma*self.B0
        readout_times = np.empty(num_samples)
        time_index = 0
        for index in range(pulse.length):
            if pulse.readout[index]:
                readout_times[time_index] = index*pulse.delta_t
                time_index = time_index + 1       
        for sample_no in range(num_samples):
            mr_baseband[sample_no] = mr_signal[sample_no]*np.exp(1j*omega_0*readout_times[sample_no])
        return mr_baseband
