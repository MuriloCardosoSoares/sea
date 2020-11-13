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
            parameters -> [rf, d], where
                rf -> flow resistivity [rayl/m]
                d -> thickness of material [m]
            theta -> angle of incidence
        """
        
        if self.freq_vec == []:
            raise ValueError("Frequency vector is empty") 
        
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

            parameters -> [rf, d, d_air], where
                rf -> flow resistivity [rayl/m]
                d -> thickness of the porous absorber layer [m]
                d_air -> depth of the air cavity]

            theta -> angle of incidence
        """
        
        if self.freq_vec == []:
            raise ValueError("Frequency vector is empty") 
        
        self.flow_resistivity = parameters[0]
        self.thickness = parameters[1]
        self.air_cavity_depth = parameters[2]
        self.rho0 = rho0
        self.c0 = c0
        self.theta = theta
       
        self.porous([self.flow_resistivity, self.thickness], self.rho0, self.c0, self.theta)

        theta_t_1 = np.arctan(self.characteristic_c*np.sin(self.theta)/self.c0)
        theta_t_2 = np.arctan(self.c0*np.sin(theta_t_1)/self.characteristic_c)

        air_surf_imp = -1j*(self.rho0*self.c0)/(np.cos(theta_t_2))/np.tan((self.w/self.c0)*np.cos(theta_t_2)*self.air_cavity_depth)

        self.surface_impedance = double_layer(air_surf_imp, self.characteristic_impedance, self.characteristic_c, self.characteristic_k, self.thickness, self.c0, self.theta)
        
        #self.surface_impedance = (-1j*air_surf_imp*self.characteristic_impedance*np.cos(theta_t_1)*1/(np.tan(self.characteristic_k*np.cos(theta_t_1)*(self.thickness))) + \
         #                        (self.characteristic_impedance)**2) / \
          #                       (air_surf_imp*(np.cos(theta_t_1))**2 - \
           #                      1j*self.characteristic_impedance*np.cos(theta_t_1)*1/(np.tan(self.characteristic_k*np.cos(theta_t_1)*(self.thickness))))

        self.admittance = (self.rho0*self.c0)/np.conj(self.surface_impedance)
        
        self.absorber_type = "porous with air cavity"

        
    def membrane (self, parameters, rho0, c0, theta=0):

        """
            Computes the surface impedance for a membrane absorber;

            *All the parameters of the absorber should be given together in an array*

            parameters -> [m, d, rf, d_porous], where
                m -> mass per unit area of the membrane  [kg/m^2]
                d -> depth of the cavity (air + porous absorber) [m] 
                rf -> flow resistivity of the porous absorber layer [rayl/m]
                d_porous -> thickness of the porous absorber layer [m]

            theta -> angle of incidence. Here, it is assumed to be 0 degrees. It is considered an argument just 
                     to facilitate the interaction with another methods
        """

        if self.freq_vec == []:
            raise ValueError("Frequency vector is empty") 
            
        self.mass_per_unit_area = parameters[0]
        self.cavity_depth = parameters[1]
        self.flow_resistivity = parameters[2]
        self.porous_layer_thickness = parameters[3]
        self.rho0 = rho0
        self.c0 = c0
        self.k0 = self.w/c0


        self.porous([self.flow_resistivity, self.thickness], self.rho0, self.c0, self.theta)

        z_si = double_layer(self.surface_impedance, self.rho0*self.c0, self.c0, self.k0,  (self.cavity_depth - self.porous_layer_thickness), self.c0, 0)

        self.surface_impedance = 1j*self.w*self.mass_per_unit_area + z_si
        self.admittance = (self.rho0*self.c0)/np.conj(self.surface_impedance)
        
        self.absorber_type = "membrane"

    def __str__(self):
        
        if self.absorber_type == "porous":
            return ("Single layer porous absorber with rigid back end. Flow resistivity = " + str(self.flow_resistivity) + " [rayl/m] and thickness = " 
            + str(self.thickness) + " [m]")
        
        elif self.absorber_type == "porous with air cavity":
            return ("Porous absorber with air cavity back end. Flow resistivity = " + str(self.flow_resistivity) + " [rayl/m] and material thickness = " 
            + str(self.thickness) + "[m]. Air cavity depth = " + str(self.air_cavity_depth) + " [m]")
        
        elif self.absorber_type == "membrane":
            return ("Membrane absorber. The mass per unit area of the membrane is" + str(self.mass_per_unit_area) + " [kg/m^2]. The total cavity depth is " 
            + str(self.cavity_depth) + "[m], being " + str(self.porous_layer_thickness) + "[m] of porous a porous material with flow resistivity = " + str(self.flow_resistivity))
        
        
def double_layer(zs2, zc1, c1, k1,  d1, c0, theta):
    
    """
        Computes the surface impedance for a double layer absorber with rigid back end
        
        zs2 -> surface impedance of the second layer
        zc1 -> characteristic impedance of the firts layer
        c1 -> characteristic velocity in the firts layer medium
        k1 -> wave number in the firts layer medium
        d1 -> thickness of the first layer
        theta -> angle of incidence
    """
    
    theta_t1 = np.arctan(c1*np.sin(theta)/c0)
    z_si = (-1j*zs2*zc1*np.cos(theta_t1)*1/(np.tan(k1*np.cos(theta_t1)*(d1))) + (zc1)**2) / (zs2*(np.cos(theta_t1))**2 - 1j*zc1*np.cos(theta_t1)*1/(np.tan(k1*np.cos(theta_t1)*(d1))))

    return z_si
        
