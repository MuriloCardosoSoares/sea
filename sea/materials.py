import numpy as np
from matplotlib import pylab as plt
import  scipy.integrate
from scipy import interpolate
from scipy.optimize import minimize
from random import uniform

class Material():
    
    def __init__(self, absorption=[], bands=[], admittance=[], freq_vec=[]):
        '''
        Set up material properties
        Inputs:
            absorption - absorption coefficients 
            bands - bands related to the absorption coefficients
            admittance - admittance of the material 
            freq_vec - frequency vector related to the admittance data
            
            Obs: all these quantities might be input data or be calculated by one of the methods
        '''
        
        self.absorption = np.array(absorption, dtype = np.float32)
        self.bands = np.array(bands, dtype = np.float32)
        self.admittance = np.array(admittance, dtype = np.complex32)
        self.freq_vec = np.array(freq_vec, dtype = np.float32)
        self.w = 2*np.pi*freq_vec
        
    def porous(self, parameters, freq_vec, rho0, c0, theta):

        """
            Computes the surface admittance for a single layer porous absorber with rigid back end

            *All the parameters of the absorber should be given together in an array*

            rf -> flow resistivity [rayl/m]
            d -> thickness of material [m]
            f_range -> the frequencies in Hz
            theta -> angle of incidence
        """
        
        self.absorber_type = "porous"
        self.flow_resistivity = parameters[0]
        self.thickness = parameters[1]

        c1=0.0978
        c2=0.7
        c3=0.189
        c4=0.595
        c5=0.0571
        c6=0.754
        c7=0.087
        c8=0.723

        X = f_range*rho0/rf
        self.characteristic_c = c0/(1+C1*np.power(X,-C2) -1j*C3*np.power(X,-C4))
        self.characteristic_rho = (rho0*c0/cc)*(1+C5*np.power(X,-C6)-1j*C7*np.power(X,-C8))

        self.characteristic_impedance = self.characteristic_rho*self.characteristic_c
        self.characteristic_k = self.w/self.characteristic_c

        theta_t = np.arctan(cc*np.sin(theta)/c0)

        self.surface_impedance = -1j*(z_c)/(np.cos(theta_t))/np.tan((k_c)*np.cos(theta_t)*d) 
    
    def __str__(self):
        
        if self.absorber_type == "porous":
            return "Single layer porous absorber with rigid back end. Flow resistivity  = " + str(self.flow_resistivity) + " and  thickness = " + str(self.thickness) 
        
        
        
