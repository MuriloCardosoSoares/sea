import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import sys
import pickle


from sea import directivity

class Air():
    '''
    Set up air properties.
    Inputs:
        c0 - sound speed (default 343 m/s - it can be overwriten using standardized calculation).
        rho0 - sound speed (default 1.21 kg/m3 - it can be overwriten using standardized calculation).
        temperature - temperature in degrees (default 20 C).
        humid - relative humidity (default 50 %).
        p_atm - atmospheric pressure (default 101325.0 Pa).
    '''
    
    def __init__(self, c0 = 343.0, rho0 = 1.21, temperature = 20.0, humid = 50.0, p_atm = 101325.0):
        
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
        return "Air sound speed = " + format(self.c0, ".2f") + " m/s | Air density = " + format(self.rho0, ".2f") + " kg/m^3" + \
                " | Temperature = " + format(self.temperature, ".2f") + " C | Humid = " + format(self.humid, ".2f") + " % | Atmospheric pressure = " + format(self.p_atm, ".2f") + " Pa \n"
    
    '''
    def __str__(self):
        return "Air sound speed = " + str(self.c0) + " m/s | Air density = " + str(self.rho0) + " kg/m^3" + \
                " | Temperature = " + str(self.temperature) + " C | Humid = " + str(self.humid) + " % | Atmospheric pressure = " + str(self.p_atm) + " Pa \n"
    '''
    
class Algorithm():
    '''
    Set up algorithm controls.
    Inputs:
        freq_init (default - 20 Hz).
        freq_end (default - 200 Hz).
        freq_step (default - 1 Hz).
    '''
    
    def __init__(self, freq_init=20.0, freq_end=200.0, freq_step=1, freq_vec=[]):

        freq_vec = np.array(freq_vec)
        if freq_vec.size == 0:
            self.freq_init = np.array(freq_init)
            self.freq_end = np.array(freq_end)
            self.freq_step = np.array(freq_step)
            self.freq_vec = np.arange(self.freq_init, self.freq_end + self.freq_step, self.freq_step)
        else:
            self.freq_init = np.array(freq_vec[0])
            self.freq_end = np.array(freq_vec[-1])
            self.freq_vec = freq_vec
            
        self.w = 2.0 * np.pi * self.freq_vec
     
    def __str__(self):
        return "Simulation algotithm will run from " + str(self.freq_init) + " Hz up to " + str(self.freq_end) + " Hz. To see all frequencies, use print(self.freq_vec). \n"


class Source():
    '''
    A sound source class.
    Inputs:
        freq_vec - frequencies for wich source properties will be generated.
        cood - 3D coordinates of the sound source (default is [0.0, 0.0, 1.0]).
        type - must be "monopole" or "directional". If "directional" is setted, you will need to upload a .pickle file with the spherical harmonics information. 
               
        The following keyworded arguments are optionals for a "monopole" source type:
            q - volume velocity [m^3/s].
            nws - source acoustic power. The same power will be setted for all frequencies.
            power_spec - source acoustic power in frequency bands.
            bands - frequency bands related to the argument "power_spec".
            
        The following keyworded arguments are optionals for a "directional" source type:
            elevation - characterizes source orientantion in the vertical plan. Must be given in degrees (default is 0°).
            azimuth - characterizes source orientantion in the horizontal plan. Must be given in degrees (default is 0°).
            power_correction - value to be subtracted from source power to fit desired power.
    '''
    
    def __init__(self, freq_vec, coord=[0.0, 0.0, 1.0], type="monopole", **kwargs):
        
        self.freq_vec = freq_vec
        self.coord = np.reshape(np.array(coord, dtype = np.float32), (1,3))
        
        if type == "directional":
            
            from google.colab import files
            print("Upload the file with the spherical harmonic information for this source:")
            uploaded = files.upload()
            
            for key in uploaded:
                
                sys.modules['directivity'] = directivity

                file_to_read = open(key, "rb")
                sh = pickle.load(file_to_read)
                file_to_read.close()
            
                self.sh_coefficients = sh.sh_coefficients
                self.sh_order = sh.sh_order
                self.freq_vec = sh.freq_vec
                
            try:
                self.elevation = kwargs["elevation"] * np.pi/180
            except:
                self.elevation = np.array([0.0])
                
            try:
                self.azimuth = kwargs["azimuth"] * np.pi/180
            except:
                self.azimuth = np.array([0.0])
                
            try:
                self.power_correction = kwargs["power_correction"] 
            except:
                self.power_correction = 0
                
        elif type == "monopole":
            
            if "q" in kwargs:
                self.q = np.array(kwargs["q"], dtype = np.float32)
                
            elif "power_spec" in kwargs and "bands" in kwargs:
                
                self.power_spec = kwargs["power_spec"]
                self.bands = kwargs["bands"]
                
                try:
                    rho0 = kwargs["rho0"]
                    c0 = kwargs["c0"]
                except:
                    raise ValueError("Rho0, c0 or both of them were not defined.") 

                tck_power_spec = interpolate.splrep(self.bands, self.power_spec, k=1)

                q = np.zeros(np.size(self.freq_vec), dtype = np.float32)

                for fi,f in enumerate(self.freq_vec):

                    if f < self.bands[0]:
                        q[fi] = (4*np.pi/rho0)*((rho0*c0*10**((interpolate.splev(self.bands[0], tck_power_spec, der=0))/10)*10**(-12))/(2*np.pi))**0.5

                    else:    
                        q[fi] = (4*np.pi/rho0)*((rho0*c0*10**((interpolate.splev(f, tck_power_spec, der=0))/10)*10**(-12))/(2*np.pi))**0.5

                self.q = q
                
            elif "nws" in kwargs:
                
                nws = kwargs["nws"]
                try:
                    rho0 = kwargs["rho0"]
                    c0 = kwargs["c0"]
                except:
                    raise ValueError("Rho0, c0 or both of them were not defined.") 
                
                self.q = np.ones(len(self.freq_vec), dtype = np.float32) * (4*np.pi/rho0)*((rho0*c0*10**(nws/10)*10**(-12))/(2*np.pi))**0.5
                
            else:
                self.q = np.ones(len(self.freq_vec), dtype = np.float32)  
        
        else:
            raise ValueError("Source type is not valid. It must be monopole or directional.") 
            
        self.type = type
        
    def __str__(self):
        return "Source coordinate is " + str(self.coord) + ". It is a " + str(self.type) + " source.\n"


class Receiver():
    '''
    A receiver class.
    Inputs:
        coord - 3D coordinates of the receiver (default is [1.0, 0.0, 0.0]).
        type - must be "omni" or "binaural". If "binaural" is setted, you will need to upload a .pickle file with the spherical harmonics information. 

        The following keyworded arguments are optionals for a "binaural" receiver type:
            azimuth - characterizes receiver orientantion in the horizontal plan. Must be given in degrees (default is 0°).
    '''
    
    def __init__(self, coord = [1.0, 0.0, 0.0], type="omni", **kwargs):

        self.coord = np.reshape(np.array(coord, dtype = np.float32), (1,3))
        
        if type == "binaural":
            from google.colab import files
            print("Upload the file with the spherical harmonic information for this receiver:")
            uploaded = files.upload()
            
            for key in uploaded:
                
                sys.modules['directivity'] = directivity
                
                file_to_read = open(key, "rb")
                sh = pickle.load(file_to_read)
                file_to_read.close()
            
                self.sh_coefficients_left = sh.sh_coefficients_left
                self.sh_coefficients_right = sh.sh_coefficients_right
                self.sh_order = sh.sh_order
                self.freq_vec = sh.freq_vec
                
            try:
                self.azimuth = kwargs["azimuth"] * np.pi/180
            except:
                self.azimuth = 0.0
    
        self.type = type
    
    def __str__(self):
        return "Receiver coordinate is " + str(self.coord) + ". It is a " + str(self.type) + " receiver.\n" 

    
