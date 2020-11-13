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
        self.admittance = np.array(admittance, dtype = np.complex64)
        self.freq_vec = np.array(freq_vec, dtype = np.float32)
        self.w = 2*np.pi*self.freq_vec
        
    def porous(self, parameters, rho0, c0, theta):

        """
            Computes the surface admittance for a single layer porous absorber with rigid back end

            *All the parameters of the absorber should be given together in an array*

            rf -> flow resistivity [rayl/m]
            d -> thickness of material [m]
            theta -> angle of incidence
        """
        
        self.absorber_type = "porous"
        self.flow_resistivity = parameters[0]
        self.thickness = parameters[1]
        self.theta = theta

        c1=0.0978
        c2=0.7
        c3=0.189
        c4=0.595
        c5=0.0571
        c6=0.754
        c7=0.087
        c8=0.723

        X = self.freq_vec*rho0/self.flow_resistivity
        self.characteristic_c = c0/(1+c1*np.power(X,-c2) -1j*c3*np.power(X,-c4))
        self.characteristic_rho = (rho0*c0/self.characteristic_c)*(1+c5*np.power(X,-c6)-1j*c7*np.power(X,-c8))

        self.characteristic_impedance = self.characteristic_rho*self.characteristic_c
        self.characteristic_k = self.w/self.characteristic_c

        theta_t = np.arctan(self.characteristic_c*np.sin(self.theta)/c0)

        self.surface_impedance = -1j*(self.characteristic_impedance)/(np.cos(theta_t))/np.tan((self.characteristic_k)*np.cos(theta_t)*self.thickness) 
    
    def __str__(self):
        
        if self.absorber_type == "porous":
            return "Single layer porous absorber with rigid back end. Flow resistivity  = " + str(self.flow_resistivity) + " and  thickness = " + str(self.thickness) 
        
        
        