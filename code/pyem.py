import numpy as np
from pyquaternion import Quaternion

class Em:
    """The basic unit of a MR simulation"""

    def __init__(self,magnetization,position,velocity,gyromagnetic_ratio,equilibrium_magnetization):
        """Initializes an em in free precession mode
        Params:
            magnetization: numpy 3-vector specifying initial magnetization
            position: numpy 3-vector specifying intial position
            velocity: numpy 3-vector specifying initial velocity
            gyromagnetic_ratio: the gyromagnetic ratio of the species
            equilibrium_magnetization: longitudinal magnetization in thermal equilibrium
        """
        # Exceptions
        if magnetization.dtype != np.float64:
            raise TypeError("magnetization must be an array of floats")
        if position.dtype != np.float64:
            raise TypeError("position must be an array of floats")
        if velocity.dtype != np.float64:
            raise TypeError("velocity must be an array of floats")
        if not isinstance(gyromagnetic_ratio,float):
            raise TypeError("gyromagnetic_ratio must be a float")
        if not isinstance(equilibrium_magnetization,float):
            raise TypeError("equilibrium_magnetization must be a float")
        if len(magnetization) != 3:
            raise ValueError("magnetization must be a 3-vector")
        if len(position) != 3:
            raise ValueError("position must be a 3-vector")
        if len(velocity) != 3:
            raise ValueError("velocity must be a 3-vector")
        # Main
        self.mu = magnetization
        self.r = position
        self.v = velocity
        self.gamma = gyromagnetic_ratio
        self.mu0 = equilibrium_magnetization

    def move(self,motion_type,delta_t):
        """Updates position according to the type of motion
        Params:
            motion_type: string specifying motion type
        """
        # Exceptions
        valid_motion_types = ['none','inertial']
        if not isinstance(motion_type,str):
            raise TypeError("motion_type must be a string")
        if motion_type not in valid_motion_types:
            raise ValueError("motion_type must be one of: " + ','.join(valid_motion_types))
        # Main
        if motion_type == 'none':
            pass
        if motion_type == 'inertial':
            self.r = self.r + self.v*delta_t

    def precess_and_relax(self,T1,T2,Bz,delta_t):
        """Updates magnetization during free precession
        Params:
            T1: T1 relaxation time
            T2: T2 relaxation time
            Bz: longitudinal field
        """
        # Exceptions
        if not isinstance(T1,float):
            raise TypeError("T1 must be a float")
        if not isinstance(T2,float):
            raise TypeError("T2 must be a float")
        if not isinstance(Bz,float):
            raise TypeError("Bz must be a float")
        if T1<=0:
            raise ValueError("T1 must be positive")
        if T2<=0:
            raise ValueError("T2 must be positive")
        # Main
        m = (self.mu[0]+1j*self.mu[1])*np.exp(-delta_t/T2 - 1j*self.gamma*Bz*delta_t)
        self.mu[0] = m.real
        self.mu[1] = m.imag
        self.mu[2] = self.mu0 + (self.mu[2] - self.mu0)*np.exp(-delta_t/T1)
