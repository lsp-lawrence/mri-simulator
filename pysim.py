from pypulse import Pulse
from pyem import Em
import numpy as np

class Sim:
    """The MR simulation object"""

    def __init__(self,num_ems,main_field,pulse_sequence):
        """Initializes the MR simulation
        Params:
            num_ems: number of Em objects to simulate
            time_step_seconds: time step in seconds
            gyromagnetic_ratio: gyromagnetic ratio
            main_field_tesla: main magnetic field in Tesla
            pulse_sequence: list of Pulse objects
        Returns:
            none
        """
        # Exceptions
        if not isinstance(num_ems,int):
            raise TypeError("num_ems must be an int")
        if not isinstance(main_field,float):
            raise TypeError("main_field_tesla must be a float")
        if not isinstance(pulse_sequence,list):
            raise TypeError("pulse_sequence must be a list")
        for item in pulse_sequence:
            if not isinstance(item,Pulse):
                raise TypeError("each item of pulse_sequence must be a Pulse object")
        # Main
        self.num_ems = num_ems
        self.B0 = main_field
        self.pulse_sequence = pulse_sequence
        self.ems = []
        for em_no in range(self.num_ems):
            magnetization = np.array([1.,0,0])
            position = np.array([float(em_no),0,0])
            velocity = np.array([0.,0,0])
            gyromagnetic_ratio = 1.0
            equilibrium_magnetization = 1.0
            self.ems.append(Em(magnetization,position,velocity,gyromagnetic_ratio,equilibrium_magnetization))     

    def run_sim(self):
        """Runs the simulation"""
        mr_signals = []
        for pulse in self.pulse_sequence:
            mr_signal = self._apply_pulse(pulse)
            mr_signals.append(mr_signal)
        return mr_signals

    def _find_T1(self,r):
        """Returns the T1 at given position
        Params:
            r: position
        Returns:
            T1: T1 value at position r
        """
        return 1.0

    def _find_T2(self,r):
        """Returns the T2 at given position
        Params:
            r: position
        Returns:
            T2: T1 value at position r
        """
        return 1.0

    def _find_Bz(self,r,gx,gy):
        """Returns the longitudinal field at given position
        Params:
            r: position
        Returns:
            Bz: field in longitudinal direction
        """
        return self.B0 + gx*r[0] + gy*r[1]

    def _apply_pulse(self,pulse):
        """Simulates the effect of a pulse on the ems
        Params:
            pulse: a Pulse object
        Returns:
            mr_signal: MR signal
        """
        mr_signal = np.zeros(np.sum(pulse.readout),dtype=np.complex)
        mr_signal_index = 0
        if pulse.mode == 'free':
            for step_no in range(pulse.length):
                if pulse.readout[step_no]:
                    mr_signal[mr_signal_index] = self._readout()
                    mr_signal_index = mr_signal_index + 1
                for em in self.ems:
                    old_r = em.r
                    motion_type = 'none'
                    em.move(motion_type,pulse.delta_t)
                    r_avg = (old_r+em.r)/2.0
                    T1 = self._find_T1(r_avg)
                    T2 = self._find_T2(r_avg)
                    Bz = self._find_Bz(r_avg,pulse.Gx[step_no],pulse.Gy[step_no])
                    em.precess_and_relax(T1,T2,Bz,pulse.delta_t)
        elif pulse.mode == 'excite':
            pass
        return mr_signal   

    def _readout(self):
        """Read out the MR signal by summing transverse magnetization of all ems
        Returns:
            mr_signal: the MR signal
        """
        mr_signal = 0.0+1j*0.0
        for em in self.ems:
            mr_signal = mr_signal + em.mu[0] + 1j*em.mu[1]
        return mr_signal
