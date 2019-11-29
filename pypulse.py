import numpy as np

class Pulse:
    """A pulse sequence unit"""

    def __init__(self,**kwargs):
        """Initalizes a pulse
        Params:
            kwargs =
        Returns:
            none
        """
        # Exceptions
        if 'mode' not in kwargs:
            raise ValueError("mode is a necessary param")
        if 'time_step' not in kwargs:
            raise ValueError("time_step is a necessary param")
        mode = kwargs.get('mode')
        valid_modes = ['free','excite']
        if not isinstance(mode,str):
            raise TypeError("mode must be a string")
        if mode not in valid_modes:
            raise ValueError("mode must be one of: " + ','.join(valid_modes))
        #Main
        self.mode = mode
        self.delta_t = kwargs.get('time_step')
        if self.mode == 'free':
            Gx = kwargs.get('Gx')
            Gy = kwargs.get('Gy')
            readout = kwargs.get('readout')
            if not isinstance(Gx,np.ndarray):
                raise TypeError("Gx must be a numpy array")
            if not isinstance(Gy,np.ndarray):
                raise TypeError("Gy must be a numpy array")
            if not isinstance(readout,np.ndarray):
                raise TypeError("readout must be a numpy array")
            if not readout.dtype == np.bool:
                raise TypeError("readout must be an array of bools")
            if not (len(Gx) == len(Gy) == len(readout)):
                raise ValueError("Gx, Gy, readout must be the same length")
            self.Gx = Gx
            self.Gy = Gy
            self.readout = readout
            self.length = len(Gx)
        elif self.mode == 'exicte':
            pass
