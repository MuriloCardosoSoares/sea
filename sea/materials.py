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
        self.rho0 = rho0
        self.c0 = c0
        self.theta = theta

        c1=0.0978
        c2=0.7
        c3=0.189
        c4=0.595
        c5=0.0571
        c6=0.754
        c7=0.087
        c8=0.723

        X = self.freq_vec*self.rho0/self.flow_resistivity
        self.characteristic_c = self.c0/(1+c1*np.power(X,-c2) -1j*c3*np.power(X,-c4))
        self.characteristic_rho = (self.rho0*self.c0/self.characteristic_c)*(1+c5*np.power(X,-c6)-1j*c7*np.power(X,-c8))

        self.characteristic_impedance = self.characteristic_rho*self.characteristic_c
        self.characteristic_k = self.w/self.characteristic_c

        theta_t = np.arctan(self.characteristic_c*np.sin(self.theta)/self.c0)

        self.surface_impedance = -1j*(self.characteristic_impedance)/(np.cos(theta_t))/np.tan((self.characteristic_k)*np.cos(theta_t)*self.thickness) 
        self.admittance = (self.rho0*self.c0)/np.conj(self.surface_impedance)
        
        self.absorber_type = "porous"
    
    
    def porous_with_air_cavity (self, parameters, rho0, c0, theta):
    
        """
            Computes the surface impedance for a single layer porous absorber with rigid back end

            *All the parameters of the absorber should be given together in an array*

            parameters -> [flow resistivity [rayl/m], thickness of the porous absorber layer [m], depth of the air cavity]

            f_range -> the frequencies in Hz
            theta -> angle of incidence
        """
        
        self.flow_resistivity = parameters[0]
        self.thickness = parameters[1]
        self.air_cavity_depth = parameters[2]
        self.rho0 = rho0
        self.c0 = c0
        self.theta = theta
       
        Material.porous([self.flow_resistivity, self.thickness], self.rho0, self.c0, self.theta)

        theta_t_1 = np.arctan(self.characteristic_c*np.sin(self.theta)/self.c0)
        theta_t_2 = np.arctan(self.c0*np.sin(theta_t_1)/self.characteristic_c)

        air_surf_imp = -1j*(self.rho0*self.c0)/(np.cos(theta_t_2))/np.tan((self.w/self.c0)*np.cos(theta_t_2)*self.air_cavity_depth)

        self.surface_impedance = double_layer_absorber(zs2, zc1, k1, d_porous, theta_t_1)
        self.surface_impedance = (-1j*air_surf_imp*self.characteristic_impedance*np.cos(theta_t1)*1/(np.tan(self.characteristic_k*np.cos(theta_t1)*(self.thickness))) + \
                                 (self.characteristic_impedance)**2) / \
                                 (air_surf_imp*(np.cos(theta_t1))**2 - \
                                 1j*self.characteristic_impedance*np.cos(theta_t1)*1/(np.tan(self.characteristic_k*np.cos(theta_t1)*(self.thickness))))


        self.absorber_type = "porous with air cavity"


    def __str__(self):
        
        if self.absorber_type == "porous":
            return "Single layer porous absorber with rigid back end. Flow resistivity  = " + str(self.flow_resistivity) + " and  thickness = " + str(self.thickness) 
        
        elif self.absorber_type == "porous with air cavity":
            return "Porous absorber with air cavity back end. Flow resistivity  = " + str(self.flow_resistivity) + " and  material thickness = " + str(self.thickness) \
                    + ". Air cavity depth = " + str(self.air_cavity_depth)
        
        
        
