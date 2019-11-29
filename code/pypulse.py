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
            elif kwargs.get('mode') == 'excite':
                kwargs.get('B1x'): (1D numpy array of floats) RF pulse modulation in x direction of rotating frame for each time step
                kwargs.get('B1y'): (1D numpy array of floats) RF pulse modulation in y direction of rotating frame for each time step
                kwargs.get('Gz'): (1D numpy array of floats) gradient in z direction for each time step
                kwargs.get('omega_rf'): (positive float) angular carrier frequency of RF pulse           
        """
        # Check params
        mode = kwargs.get('mode')
        valid_modes = ['free','excite']
        if (not isinstance(mode,str)) or (mode not in valid_modes):
            raise TypeError("mode must be a string and one of: " + ','.join(valid_modes))
        delta_t = kwargs.get('delta_t')
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
            B1x = kwargs.get('B1x')
            B1y = kwargs.get('B1y')
            omega_rf = kwargs.get('omega_rf')
            Gz = kwargs.get('Gz')
            if (B1x.dtype != np.float64) or (B1x.ndim != 1):
                raise TypeError("B1x must be a 1D numpy array of floats")
            if (B1y.dtype != np.float64) or (B1y.ndim != 1):
                raise TypeError("B1y must be a 1D numpy array of floats")
            if (Gz.dtype != np.float64) or (Gz.ndim != 1):
                raise TypeError("Gz must be a 1D numpy array of bools")
            if not (len(B1x) == len(B1y) == len(Gz)):
                raise ValueError("B1x, B1y, Gz must be the same length")
            if (not isinstance(omega_rf,float)) or (omega_rf <= 0):
                raise TypeError("omega_rf must be a positive float")
            # Main
            self.mode = mode
            self.delta_t = delta_t
            self.B1x = B1x
            self.B1y = B1y
            self.omega_rf = omega_rf
            self.Gz = Gz
            self.length = len(Gz)
            self.signal_collected = False
