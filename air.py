import numpy as np
#import toml
import matplotlib.pyplot as plt
import time, sys

class AirProperties():
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
        # config = load_cfg(config_file)
        # self.c0 = np.array(c0, dtype = np.float32)
        # self.rho0 = np.array(rho0, dtype = np.float32)
        self.c0 = np.array(c0)
        self.rho0 = np.array(rho0)
        self.temperature = np.array(temperature, dtype = np.float32)
        self.hr = np.array(humid, dtype = np.float32)
        self.p_atm = np.array(p_atm, dtype = np.float32)
