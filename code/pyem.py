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

    def move(self,motion_type,delta_t):
        """Updates position according to the type of motion
        Params:
            motion_type: (string) type of motion
            delta_t: (positive float) time step duration
        """
        # Check params
        valid_motion_types = ['inertial']
        if (not isinstance(motion_type,str)) or (motion_type not in valid_motion_types):
            raise TypeError("motion_type must be a string and one of: " + ','.join(valid_motion_types))
        if (not isinstance(delta_t,float)) or (delta_t <= 0):
            raise TypeError("delta_t must be a positive float")
        # Main
        if motion_type == 'inertial':
            self.r = self.r + self.v*delta_t

    def precess_and_relax(self,T1,T2,Bz,delta_t):
        """Updates magnetization assuming free precession
        Params:
            T1: (positive float) T1 relaxation time
            T2: (positive float) T2 relaxation time
            Bz: (float) longitudinal field
            delta_t: (positive float) time step duration
        """
        # Check params
        if (not isinstance(T1,float))  or (T1 <= 0):
            raise TypeError("T1 must be a positive float")
        if (not isinstance(T2,float)) or (T2 <= 0):
            raise TypeError("T2 must be a positive float")
        if not isinstance(Bz,float):
            raise TypeError("Bz must be a float")
        if (not isinstance(delta_t,float)) or (delta_t <= 0):
            raise TypeError("delta_t must be a positive float")
        # Main
        Bz = (1-self.sigma)*Bz # effective field due to electron shielding
        m = (self.mu[0]+1j*self.mu[1])*np.exp(-delta_t/T2 - 1j*self.gamma*Bz*delta_t)
        self.mu[0] = m.real
        self.mu[1] = m.imag
        self.mu[2] = self.mu0 + (self.mu[2] - self.mu0)*np.exp(-delta_t/T1)

    def update_flip_quaternion(self,B1x,B1y,Bz,omega_rf,delta_t):
        """Updates flip quaternion assuming excitation
        Params:
            B1x: (float) RF pulse modulation in x direction of rotating frame
            B1y: (float) RF pulse modulaiton in y direction of rotating frame
            Bz: (float) field in z direction in lab frame
            omega_rf: (positive float) angular carrier frequency of RF pulse
            delta_t: (positive float) time step
        """
        # Check params
        if not isinstance(B1x,float):
            raise TypeError("B1x must be a float")
        if not isinstance(B1y,float):
            raise TypeError("B1y must be a float")
        if not isinstance(Bz,float):
            raise TypeError("Bz must be a float")
        if (not isinstance(omega_rf,float)) or (omega_rf <= 0):
            raise TypeError("omega_rf must be a positive float")
        if (not isinstance(delta_t,float)) or (delta_t <= 0):
            raise TypeError("delta_t must be a positive float")
        # Main
        Brot = np.array([B1x,B1y,Bz])
        Brot = Brot*(1-self.sigma) # effective field due to electron shielding
        Beff = Brot - np.array([0.0,0.0,omega_rf/self.gamma]) # effective field in rotating frame
        Beff_norm = np.linalg.norm(Beff)
        flip_axis = Beff/Beff_norm
        flip_angle = self.gamma*Beff_norm*delta_t
        self.flip_quaternion = self.flip_quaternion*Quaternion(axis=flip_axis,angle=flip_angle)

    def flip(self,omega_rf,pulse_duration):
        """Updates magnetization at end of excitation pulse and resets flip quaternion
        Params:
            omega_rf: angular carrier frequency of RF pulse
            pulse_duration: duration of RF pulse
        """
        self.flip_quaternion = self.flip_quaternion*Quaternion(axis=[0,0,1],angle=omega_rf*pulse_duration)
        self.mu = self.flip_quaternion.rotate(self.mu)
        self.flip_quaternion = Quaternion(1)
