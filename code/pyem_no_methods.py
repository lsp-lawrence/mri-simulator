import numpy as np
from pyquaternion import Quaternion

class Em:
    """The atom of the MR simulation"""

    def __init__(self,magnetization,position,velocity,gyromagnetic_ratio,shielding_constant,equilibrium_magnetization):
        """Initializes an em
        Params:
            magnetization: (numpy 3-vector) initial magnetization
            position: (numpy 3-vector) intial position
            velocity: (numpy 3-vector) initial velocity
            gyromagnetic_ratio: (float) the gyromagnetic ratio of the nucleus
            shielding_constant: (positive float) shiedling constant from chemical environment
            equilibrium_magnetization: (positive float) longitudinal magnetization in thermal equilibrium
        """
        # Check params
        if (magnetization.dtype != np.float64) or (magnetization.shape != (3,)):
            raise TypeError("magnetization must be a numpy 3-vector")
        if (position.dtype != np.float64) or (position.shape != (3,)):
            raise TypeError("position must be a numpy 3-vector")
        if (velocity.dtype != np.float64) or (velocity.shape != (3,)):
            raise TypeError("velocity must be a numpy 3-vector")
        if not isinstance(gyromagnetic_ratio,float):
            raise TypeError("gyromagnetic_ratio must be a float")
        if not(isinstance(shielding_constant,float) and shielding_constant > 0):
            raise TypeError("shielding constant must be a positive float")
        if (not isinstance(equilibrium_magnetization,float)) or (equilibrium_magnetization <= 0):
            raise TypeError("equilibrium_magnetization must be a positive float")
        # Main
        self.mu = magnetization
        self.r = position
        self.v = velocity # A velocity component of None indicates exactly zero velocity in that direction
        self.gamma = gyromagnetic_ratio
        self.sigma = shielding_constant
        self.mu0 = equilibrium_magnetization
        self.flip_quaternion = Quaternion(1) # Rotation quaterion for flip from excitation
