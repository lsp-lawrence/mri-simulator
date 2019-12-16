import numpy as np
from pyquaternion import Quaternion

class Em:
    """The atom of the MR simulation"""

    def __init__(self,magnetization,position,gyromagnetic_ratio,shielding_constant,equilibrium_magnetization):
        """Initializes an em
        Params:
            magnetization: (numpy 3-vector) initial magnetization
            position: (numpy 3-vector) intial position
            gyromagnetic_ratio: (float) the gyromagnetic ratio of the nucleus
            shielding_constant: (nonnegative float) shielding constant from chemical environment
            equilibrium_magnetization: (positive float) longitudinal magnetization in thermal equilibrium
        """
        # Check params
        if not(magnetization.dtype == np.float64 and magnetization.shape == (3,)):
            raise TypeError("magnetization must be a numpy 3-vector")
        if not(position.dtype == np.float64 and position.shape == (3,)):
            raise TypeError("position must be a numpy 3-vector")
        if not isinstance(gyromagnetic_ratio,float):
            raise TypeError("gyromagnetic_ratio must be a float")
        if not(isinstance(shielding_constant,float) and shielding_constant >= 0.0):
            raise TypeError("shielding constant must be a nonnegative float")
        if not(isinstance(equilibrium_magnetization,float) and equilibrium_magnetization > 0.0):
            raise TypeError("equilibrium_magnetization must be a positive float")
        # Set data attributes
        self.mu = magnetization
        self.r = position
        self.gamma = gyromagnetic_ratio
        self.sigma = shielding_constant
        self.mu0 = equilibrium_magnetization
        self.flip_quaternion = Quaternion(1) # Rotation quaterion for flip from excitation

    def set_shielding_constant(self,new_shielding_constant):
        """Updates shielding constant
        Params:
            new_shielding_constant: (nonnegative float) new shielding constant
        """
        # Check params
        if not(isinstance(new_shielding_constant,float) and new_shielding_constant >= 0.0):
            raise TypeError("new_shielding_constant must be a nonnegative float")
        # Update data attribute
        self.sigma = new_shielding_constant

    def diffuse(self,diffusion_coefficients,delta_t):
        """Updates position according to diffusion process
        Params:
            diffusion_coefficients: (numpy 3-vector of nonnegative floats) diffusion coefficients in 3 spatial directions at current position
            delta_t: (positive float) time step
        """
        # Check params
        if not(diffusion_coefficients.shape == (3,) and diffusion_coefficients.dtype == np.float64 and all([item>=0.0 for item in diffusion_coefficients])):
            raise TypeError("diffusion_coefficients must be a numpy 3-vector of nonnegative floats")
        if not(isinstance(delta_t,float) and delta_t > 0.0):
            raise TypeError("delta_t must be a positive float")
        # Model diffusion
        for ax_no in range(3):
            if(diffusion_coefficients[ax_no]>0.0):
                self.r[ax_no] = self.r[ax_no] + np.random.normal(0.0,np.sqrt(2*diffusion_coefficients[ax_no]*delta_t))

    def precess_and_relax(self,T1,T2,Bz,delta_t):
        """Updates magnetization assuming free precession
        Params:
            T1: (positive float) T1 relaxation time
            T2: (positive float) T2 relaxation time
            Bz: (float) longitudinal field
            delta_t: (positive float) time step
        """
        # Check params
        if not(isinstance(T1,float) and T1 > 0.0):
            raise TypeError("T1 must be a positive float")
        if not(isinstance(T2,float) and T2 > 0.0):
            raise TypeError("T2 must be a positive float")
        if not isinstance(Bz,float):
            raise TypeError("Bz must be a float")
        if not(isinstance(delta_t,float) and delta_t > 0.0):
            raise TypeError("delta_t must be a positive float")
        # Rotate magnetization
        Bz = (1-self.sigma)*Bz # effective field due to electron shielding
        m = (self.mu[0]+1j*self.mu[1])*np.exp(-delta_t/T2 - 1j*self.gamma*Bz*delta_t)
        self.mu[0] = m.real
        self.mu[1] = m.imag
        self.mu[2] = self.mu0 + (self.mu[2] - self.mu0)*np.exp(-delta_t/T1)

    def update_flip_quaternion(self,B1x,B1y,Bz,omega_rf,delta_t):
        """Updates flip quaternion assuming excitation
        Params:
            B1x: (float) RF pulse modulation in x direction in rotating frame
            B1y: (float) RF pulse modulation in y direction in rotating frame
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
        if not(isinstance(omega_rf,float) and omega_rf > 0.0):
            raise TypeError("omega_rf must be a positive float")
        if not(isinstance(delta_t,float) and delta_t > 0.0):
            raise TypeError("delta_t must be a positive float")
        # Compute update to flip quaternion
        Brot = np.array([B1x,B1y,Bz])
        Brot = Brot*(1-self.sigma) # effective field due to electron shielding
        Beff = Brot - np.array([0.0,0.0,omega_rf/self.gamma]) # effective field in rotating frame
        Beff_norm = np.linalg.norm(Beff)
        flip_axis = Beff/Beff_norm
        flip_angle = -self.gamma*Beff_norm*delta_t
        self.flip_quaternion = self.flip_quaternion*Quaternion(axis=flip_axis,angle=flip_angle)

    def flip(self,omega_rf,pulse_duration):
        """Updates magnetization at end of excitation pulse and resets flip quaternion
        Params:
            omega_rf: (positive float) angular carrier frequency of RF pulse
            pulse_duration: (positive float) duration of RF pulse
        """
        # Check params
        if not(isinstance(omega_rf,float) and omega_rf > 0.0):
            raise TypeError("omega_rf must be a positive float")
        if not(isinstance(pulse_duration,float) and pulse_duration > 0.0):
            raise TypeError("pulse_duration must be a positive float")
        # Apply flip quaternion and reset
        self.flip_quaternion = self.flip_quaternion*Quaternion(axis=[0,0,1],angle=-omega_rf*pulse_duration)
        self.mu = self.flip_quaternion.rotate(self.mu)
        self.flip_quaternion = Quaternion(1)
