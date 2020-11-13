import numpy as np
#import toml
import matplotlib.pyplot as plt
import time, sys

class Air():
    
    def __init__(self, c0 = 343.0, rho0 = 1.21, temperature = 20.0, humid = 50.0, p_atm = 101325.0):
        '''
        Set up air properties
        Inputs:
            c0 - sound speed (default 343 m/s - it can be overwriten using standardized calculation)
            rho0 - sound speed (default 1.21 kg/m3 - it can be overwriten using standardized calculation)
            temperature - temperature in degrees (default 20 C)
            humid - relative humidity (default 50 %)
            p_atm - atmospheric pressure (default 101325.0 Pa)
        '''
        
        self.c0 = np.array(c0)
        self.rho0 = np.array(rho0)
        self.temperature = np.array(temperature, dtype = np.float32)
        self.humid = np.array(humid, dtype = np.float32)
        self.p_atm = np.array(p_atm, dtype = np.float32)
       
    def standardized_c0_rho0(self,):
        '''
        This method is used to calculate the standardized value of the sound speed and
        air density based on measurements of temperature, humidity and atm pressure.
        It will overwrite the user supplied values
        '''
        # kappla = 0.026
        temp_kelvin = self.temperature + 273.16 # temperature in [K]
        R = 287.031                 # gas constant
        rvp = 461.521               # gas constant for water vapor
        # pvp from Pierce Acoustics 1955 - pag. 555
        pvp = 0.0658 * temp_kelvin**3 - 53.7558 * temp_kelvin**2 \
            + 14703.8127 * temp_kelvin - 1345485.0465
        # Air viscosity
        # vis = 7.72488e-8 * temp_kelvin - 5.95238e-11 * temp_kelvin**2
        # + 2.71368e-14 * temp_kelvin**3
        # Constant pressure specific heat
        cp = 4168.8 * (0.249679 - 7.55179e-5 * temp_kelvin \
            + 1.69194e-7 * temp_kelvin**2 \
            - 6.46128e-11 * temp_kelvin**3)
        cv = cp - R                 # Constant volume specific heat
        # b2 = vis * cp / kappla      # Prandtl number
        gam = cp / cv               # specific heat constant ratio
        # Air density
        self.rho0 = self.p_atm / (R * temp_kelvin) \
            - (1/R - 1/rvp) * self.humid/100 * pvp/temp_kelvin
        # Air sound speed
        self.c0 = (gam * self.p_atm/self.rho0)**0.5
    
    def __str__(self):
        return "Air sound speed = " + str(self.c0) + " | Air density = " + str(self.rho0) + \
                " | Temperature = " + str(self.temperature) + " | Humid = " + str(self.humid) + " | Atmospheric pressure =" + str(self.p_atm)
    
    
class Algorithm():
    
    def __init__(self, freq_init=20.0, freq_end=200.0, freq_step=1, freq_vec=[]):
        '''
        Set up algorithm controls. You set-up your frequency span:
        Inputs:
            freq_init (default - 100 Hz)
            freq_end (default - 10000 Hz)
            freq_step (default - 10 Hz)
        '''
        freq_vec = np.array(freq_vec)
        if freq_vec.size == 0:
            self.freq_init = np.array(freq_init)
            self.freq_end = np.array(freq_end)
            self.freq_step = np.array(freq_step)
            self.freq = np.arange(self.freq_init, self.freq_end + self.freq_step, self.freq_step)
        else:
            self.freq_init = np.array(freq_vec[0])
            self.freq_end = np.array(freq_vec[-1])
            self.freq = freq_vec
        self.w = 2.0 * np.pi * self.freq
     
    def __str__(self):
        return "Simulation algotithm will run from " + str(self.freq_init) + " Hz up to " + str(self.freq_end) + " Hz and a step of " + str(self.freq_step) + " Hz"

class Source():
    '''
    A sound source class to initialize the following sound source properties.
    :
    Inputs:
        cood - 3D coordinates of the sound sources
        q - volume velocity [m^3/s]
        source_type - use "monopole" if you are considering omnidiretional sound sources. 
                      Use "direciotinal" to use the spherical harmonics technique to consider the sources directivities
        
    '''
    def __init__(self, coord=[0.0, 0.0, 1.0], q = [1.0], source_type="monopole"):
        self.coord = np.reshape(np.array(coord, dtype = np.float32), (1,3))
        self.q = np.array([q], dtype = np.float32)
        self.type = source_type
        
    def __str__(self):
        return "Source coordinate is " + str(self.coord) 

        
class Receiver():
    '''
    A receiver class to initialize the following receiver properties:
    coord - 3D coordinates of a receiver
    '''
    def __init__(self, coord = [1.0, 0.0, 0.0]):
        '''
        The class constructor initializes a single receiver with a given 3D coordinates
        The default is a height of 1 [cm]. User must be sure that the receiver lies out of
        the sample being emulated. This can go wrong if we allow the sample to have a thickness
        going on z>0
        '''
        self.coord = np.reshape(np.array(coord, dtype = np.float32), (1,3))
       
    def __str__(self):
        return "Receiver coordinate is " + str(self.coord) 

    
