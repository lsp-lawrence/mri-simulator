import numpy as np

class Pulse:
    """A pulse sequence unit"""

    def __init__(self,**kwargs):
        """Initalizes a pulse
        Params:
            kwargs.get('delta_t'): (positive float) time step duration
            kwargs.get('mode'): (string) pulse mode, either 'free' or 'excite'
            if kwargs.get('mode') == 'free':
                kwargs.get('Gx'): (1D numpy array of floats) gradient in x for each time step
                kwargs.get('Gy'): (1D numpy array of floats) gradient in y for each time step
                kwargs.get('readout'): (1D numpy array of bools) read ADC at time step?
                
        """
        # Check params
        if 'mode' not in kwargs:
            raise ValueError("mode is a necessary param")
        mode = kwargs.get('mode')
        valid_modes = ['free','excite']
        if (not isinstance(mode,str)) or (mode not in valid_modes):
            raise TypeError("mode must be a string and one of: " + ','.join(valid_modes))
        if 'time_step' not in kwargs:
            raise ValueError("time_step is a necessary param")
        delta_t = kwargs.get('time_step')
        if (not isinstance(delta_t,float)) or (delta_t <= 0):
            raise TypeError("delta_t must be a positive float")
        if self.mode == 'free':
            Gx = kwargs.get('Gx')
            Gy = kwargs.get('Gy')
            readout = kwargs.get('readout')
            if (Gx.dtype != np.float64) or (Gx.ndim != 1):
                raise TypeError("Gx must be a 1D numpy array of floats")
            if (Gy.dtype != np.float64) or (Gy.ndim != 1):
                raise TypeError("Gy must be a 1D numpy array of floats")
            if (readout.dtype != np.bool) or (readout.ndim != 1):
                raise TypeError("readout must be a 1D numpy array of bools")
            if not (len(Gx) == len(Gy) == len(readout)):
                raise ValueError("Gx, Gy, and readout must be the same length")
            # Main
            self.mode = mode
            self.delta_t = delta_t
            self.Gx = Gx
            self.Gy = Gy
            self.readout = readout
            self.length = len(Gx)
            self.signal_collected = any(readout)
        elif self.mode == 'exicte':
            self.signal_collected = False
            pass
