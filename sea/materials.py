import numpy as np
from matplotlib import pylab as plt
import  scipy.integrate
from scipy import interpolate
from scipy.optimize import minimize
from random import uniform

class Material():
    
    #Define de lower and upper limits and center frequencies of octave and third-octave bands
    
    lower = {
            0 : np.array([11,22,44,88,177,355,710,1420,2840,5680,11360]),
            1 : np.array([11.2,14.1,17.8,22.4,28.2,35.5,44.7,56.2,70.8,89.1,112,141,178,224,282,355,447,562,708,891,1122,1413,1778,2239,2818,3548,4467,5623,7079,8913,11220,14130,17780])
            }

    upper = {
            0 : np.array([22,44,88,177,355,710,1420,2840,5680,11360,22720]),
            1 : np.array([14.1,17.8,22.4,28.2,35.5,44.7,56.2,70.8,89.1,112,141,178,224,282,355,447,562,708,891,1122,1413,1778,2239,2818,3548,4467,5623,7079,8913,11220,14130,17780,22390])
            }

    center = {
            0 : np.array([16,31.5,63,125,250,500,1000,2000,4000,8000,16000]),
            1 : np.array([12.5,16,20,25,31.5,40,50,63,80,100,125,160,200,250,315,400,500,630,800,1000,1250,1600,2000,2500,3150,4000,5000,6300,8000,10000,12500,16000,20000])
            }
            
        
    def __init__(self, normal_incidence_alpha=[], statistical_alpha=[], octave_bands_statistical_alpha=[], 
                 octave_bands=[], third_octave_bands_statistical_alpha=[], third_octave_bands=[], admittance=[], 
                 normalized_surface_impedance=[], surface_impedance=[], freq_vec=[], rmk1=[], rho0=1.21, c0=343.0):
        '''
        Set up material properties
        Inputs:
            absorption - absorption coefficients 
            bands - bands related to the absorption coefficients
            admittance - admittance of the material 
            freq_vec - frequency vector related to the admittance data
            
            Obs: all these quantities might be input data or be calculated by one of the methods
        '''
        
        self.normal_incidence_alpha = np.array(normal_incidence_alpha, dtype = np.float32)
        self.statistical_alpha = np.array(statistical_alpha, dtype = np.float32)
        self.octave_bands_statistical_alpha = np.array(octave_bands_statistical_alpha, dtype = np.float32)
        self.octave_bands = np.array(octave_bands, dtype = np.float32)
        self.third_octave_bands_statistical_alpha = np.array(third_octave_bands_statistical_alpha, dtype = np.float32)
        self.third_octave_bands = np.array(third_octave_bands, dtype = np.float32)
        self.admittance = np.array(admittance, dtype = np.complex64)
        self.surface_impedance = np.array(surface_impedance, dtype = np.complex64)
        self.normalized_surface_impedance = np.array(normalized_surface_impedance, dtype = np.complex64)
        self.freq = np.array(freq_vec, dtype = np.float32)
        self.rho0 = rho0
        self.c0 = c0
        self.w = 2*np.pi*self.freq
        self.k0 = self.w/self.c0
        
        self.adjust()

        
    def rigid(self):
        
        self.admittance = np.zeros_like(self.freq, dtype=np.complex64)
        self.statistical_alpha = np.zeros(len(self.admittance))
        self.alpha_in_bands()
                
        self.absorber_type = "rigid"
    
    
    def porous(self, parameters, theta=0):

        """
            Computes the surface admittance for a single layer porous absorber with rigid back end

            *All the parameters of the absorber should be given together in an array*
            parameters -> [rf, d], where
                rf -> flow resistivity [rayl/m]
                d -> thickness of material [m]
            theta -> angle of incidence
        """
        
        if self.freq.size == 0:
            raise ValueError("Frequency vector is empty") 
        
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

        X = self.freq*self.rho0/self.flow_resistivity
        self.characteristic_c = self.c0/(1+c1*np.power(X,-c2) -1j*c3*np.power(X,-c4))
        self.characteristic_rho = (self.rho0*self.c0/self.characteristic_c)*(1+c5*np.power(X,-c6)-1j*c7*np.power(X,-c8))

        self.characteristic_impedance = self.characteristic_rho*self.characteristic_c
        self.characteristic_k = self.w/self.characteristic_c

        theta_t = np.arctan(self.characteristic_c*np.sin(self.theta)/self.c0)

        self.surface_impedance = -1j*(self.characteristic_impedance)/(np.cos(theta_t))/np.tan((self.characteristic_k)*np.cos(theta_t)*self.thickness) 
        
        self.normalized_surface_impedance = self.surface_impedance/(self.rho0*self.c0)
        #self.admittance = 1/self.normalized_surface_impedance
        self.admittance = np.conj(1/self.normalized_surface_impedance)
        
        self.impedance2alpha()
        
        if self.flow_resistivity <= 20000:
            self.absorber_type = "soft porous"
        else:
            self.absorber_type = "hard porous"
    
    def porous_with_air_cavity (self, parameters, theta=0):
    
        """
            Computes the surface impedance for a single layer porous absorber with rigid back end

            *All the parameters of the absorber should be given together in an array*

            parameters -> [rf, d, d_air], where
                rf -> flow resistivity [rayl/m]
                d -> thickness of the porous absorber layer [m]
                d_air -> depth of the air cavity]

            theta -> angle of incidence
        """
        
        if self.freq.size == 0:
            raise ValueError("Frequency vector is empty.") 
        
        self.flow_resistivity = parameters[0]
        self.thickness = parameters[1]
        self.air_cavity_depth = parameters[2]
        self.theta = theta
       
        self.porous([self.flow_resistivity, self.thickness], self.theta)

        theta_t_1 = np.arctan(self.characteristic_c*np.sin(self.theta)/self.c0)
        theta_t_2 = np.arctan(self.c0*np.sin(theta_t_1)/self.characteristic_c)

        air_surf_imp = -1j*(self.rho0*self.c0)/(np.cos(theta_t_2))/np.tan((self.w/self.c0)*np.cos(theta_t_2)*self.air_cavity_depth)

        self.surface_impedance = double_layer(air_surf_imp, self.characteristic_impedance, self.characteristic_c, self.characteristic_k, self.thickness, self.c0, self.theta)

        self.normalized_surface_impedance = self.surface_impedance/(self.rho0*self.c0)
        #self.admittance = 1/self.normalized_surface_impedance
        self.admittance = np.conj(1/self.normalized_surface_impedance)
        
        self.impedance2alpha()
        
        self.absorber_type = "porous with air cavity"

        
    def membrane (self, parameters, theta=0):

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

        if self.freq.size == 0:
            raise ValueError("Frequency vector is empty.") 
            
        self.mass_per_unit_area = parameters[0]
        self.cavity_depth = parameters[1]
        self.flow_resistivity = parameters[2]
        self.porous_layer_thickness = parameters[3]

        self.porous([self.flow_resistivity, self.porous_layer_thickness], 0)

        z_si = double_layer(self.surface_impedance, self.rho0*self.c0, self.c0, self.k0, (self.cavity_depth - self.porous_layer_thickness), self.c0, 0)

        self.surface_impedance = 1j*self.w*self.mass_per_unit_area + z_si
        
        self.normalized_surface_impedance = self.surface_impedance/(self.rho0*self.c0)
        #self.admittance = 1/self.normalized_surface_impedance
        self.admittance = np.conj(1/self.normalized_surface_impedance)
        
        self.impedance2alpha()
        
        self.absorber_type = "membrane"

        
    def perforated_panel_old (self, parameters, theta=0):

        """
            Computes the surface impedance for a perforated panel absorber;

            *All the parameters of the absorber should be given together in an array*
            
            parameters -> [h, a, p, d, rf, d_porous], where
                h -> panel thickness [m]
                a -> radius of the circular openings [m] 
                p -> perforation rate
                d -> depth of the cavity (air + porous absorber) [m]

                rf -> flow resistivity of the porous absorber layer [rayl/m]
                d_porous -> thickness of the porous absorber layer [m]

            theta -> angle of incidence. Here, it is assumed to be 0 degrees. It is considered an argument just 
                     to facilitate the interaction with another functions

        """
        
        if self.freq.size == 0:
            raise ValueError("Frequency vector is empty.") 

        self.panel_thickness = parameters[0]
        self.openings_radius = parameters[1]
        self.perforation_rate = parameters[2]
        self.cavity_depth = parameters[3]
        self.flow_resistivity = parameters[4]
        self.porous_layer_thickness = parameters[5]

        m = self.rho0*self.perforation_rate*(self.panel_thickness + 1.7*self.openings_radius)       # superficial density of the gas in each perfuration
        z_t = 1j*self.w*m                 # impedance of a single opening

        # cavity impedance 
        self.porous([self.flow_resistivity, self.porous_layer_thickness], 0)

        zs2 = -1j*(self.rho0*self.c0) * 1/(np.tan((self.w/self.c0)*(self.cavity_depth-self.porous_layer_thickness)))
        z_cav = double_layer(zs2, self.characteristic_impedance, self.characteristic_c, self.characteristic_k, self.porous_layer_thickness, self.c0, 0)

        # The total impedance is the sum:
        self.surface_impedance = z_t/self.perforation_rate + z_cav
        
        self.normalized_surface_impedance = self.surface_impedance/(self.rho0*self.c0)
        #self.admittance = 1/self.normalized_surface_impedance
        self.admittance = np.conj(1/self.normalized_surface_impedance)
        
        self.impedance2alpha()
        
        self.absorber_type = "perforated panel"
        
        
    def perforated_panel (self, parameters, theta=0):

        """
            Computes the surface impedance for a perforated panel absorber;

            *All the parameters of the absorber should be given together in an array*
            
            parameters -> [h, a, p, d, rf, d_porous], where
                h -> panel thickness [m]
                a -> radius of the circular openings [m] 
                p -> perforation rate
                d -> depth of the cavity (air + porous absorber) [m]

                rf -> flow resistivity of the porous absorber layer [rayl/m]
                d_porous -> thickness of the porous absorber layer [m]

            theta -> angle of incidence. Here, it is assumed to be 0 degrees. It is considered an argument just 
                     to facilitate the interaction with another functions

        """
        
        if self.freq.size == 0:
            raise ValueError("Frequency vector is empty.") 

        self.panel_thickness = parameters[0]
        self.openings_radius = parameters[1]
        self.perforation_rate = parameters[2]
        self.cavity_depth = parameters[3]
        self.flow_resistivity = parameters[4]
        self.porous_layer_thickness = parameters[5]

        #m = self.rho0*self.perforation_rate*(self.panel_thickness + 1.7*self.openings_radius)       # superficial density of the gas in each perfuration
        m = self.rho0*(self.panel_thickness + 1.7*self.openings_radius)/self.perforation_rate       # superficial density of the gas in each perfuration
        z_t = 1j*self.w*m                 # impedance of a single opening
        #z_t = 1j*self.w*(self.panel_thickness + 1.7*self.openings_radius)/(self.perforation_rate)                # impedance of a single opening
        
        # cavity impedance 
        self.porous([self.flow_resistivity, self.porous_layer_thickness], 0)

        z_sar = -1j*(self.rho0*self.c0) * 1/(np.tan((self.w/self.c0)*(self.cavity_depth-self.porous_layer_thickness)))
        
        z_si = (-1j*z_sar*self.characteristic_impedance*1/(np.tan(self.characteristic_k*(self.porous_layer_thickness))) + (self.characteristic_impedance)**2) / (z_sar - 1j*self.characteristic_impedance*1/(np.tan(self.characteristic_k*(self.porous_layer_thickness))))

        # The total impedance is the sum:
        self.surface_impedance = z_t + z_si
        
        self.normalized_surface_impedance = self.surface_impedance/(self.rho0*self.c0)
        #self.admittance = 1/self.normalized_surface_impedance
        self.admittance = np.conj(1/self.normalized_surface_impedance)
        
        self.impedance2alpha()
        
        self.absorber_type = "perforated panel"
        
        
    def microperforated_panel_eric (self, parameters, mi0=1.84e-5, theta=0):

        """
            Computes the surface impedance for a microperforated panel absorber;

            *All the parameters of the absorber should be given together in an array*
            
            parameters -> [h, a, p, d_air], where
                h -> panel thickness [m]
                a -> radius of the circular openings [m] 
                p -> perforation rate
                d_air -> width of the air cavity [m]

            theta -> angle of incidence. Here, it is assumed to be 0 degrees. It is considered an argument just 
                     to facilitate the interaction with another functions

            mi0 -> dynamic viscosity of air [Pa*s]
        """
        
        if self.freq.size == 0:
            raise ValueError("Frequency vector is empty.") 
            
        self.panel_thickness = parameters[0]
        self.openings_radius = parameters[1]
        self.perforation_rate = parameters[2]
        self.air_cavity_depth = parameters[3]
        self.air_dynamic_viscosity = mi0       

        y = 2*self.openings_radius*(self.w*self.rho0/(4*self.air_dynamic_viscosity))**(1/2)

        r = 32*self.air_dynamic_viscosity*self.panel_thickness/(self.perforation_rate*(2*self.openings_radius)**2) \
            * ((1+y**2/32)**(1/2) + 2**(1/2)/32*y*2*self.openings_radius/self.panel_thickness)

        m = self.rho0*self.panel_thickness/self.perforation_rate * (1 + (9+y**2/2)**(1/2) + 0.85*2*self.openings_radius/self.panel_thickness)

        self.surface_impedance = r + 1j*self.w*m - 1j*(self.rho0*self.c0)*1/(np.tan((self.k0)*self.air_cavity_depth)) 
        
        self.normalized_surface_impedance = self.surface_impedance/(self.rho0*self.c0)
        #self.admittance = 1/self.normalized_surface_impedance
        self.admittance = 1/self.surface_impedance
        
        self.impedance2alpha()
        
        self.absorber_type = "microperforated panel"   
        
        
    def microperforated_panel (self, parameters, mi0=18.13e-6, theta=0):

        """
            Computes the surface impedance for a microperforated panel absorber;

            *All the parameters of the absorber should be given together in an array*
            
            parameters -> [h, a, p, d_air], where
                h -> panel thickness [m]
                a -> radius of the circular openings [m] 
                p -> perforation rate
                d_air -> width of the air cavity [m]

            theta -> angle of incidence. Here, it is assumed to be 0 degrees. It is considered an argument just 
                     to facilitate the interaction with another functions

            mi0 -> dynamic viscosity of air [Pa*s]
        """
        if self.freq.size == 0:
            raise ValueError("Frequency vector is empty.") 
            
        self.panel_thickness = parameters[0]
        self.openings_radius = parameters[1]
        self.perforation_rate = parameters[2]
        self.air_cavity_depth = parameters[3]
        self.air_dynamic_viscosity = mi0       

        s_2 = (self.rho0*self.w*self.openings_radius**2) / (self.air_dynamic_viscosity) 
        
        z_t = ((32*self.air_dynamic_viscosity*self.panel_thickness)/(4*self.openings_radius**2) * (1+s_2/32)**0.5 \
                + 1j*self.w*self.rho0*self.panel_thickness*(1 + (9+s_2/2)**(-0.5)))/(self.rho0*self.c0)
        
        z_e = (((self.rho0*self.air_dynamic_viscosity*self.w)/2)**0.5 + 1j*1.7*self.rho0*self.w*self.openings_radius)/(self.rho0*self.c0)
        
        self.normalized_surface_impedance = (z_t+z_e)/self.perforation_rate - 1j*1/(np.tan((self.k0)*self.air_cavity_depth))
        
        self.surface_impedance = self.normalized_surface_impedance*(self.rho0*self.c0)
        #self.admittance = 1/self.normalized_surface_impedance
        self.admittance = np.conj(1/self.normalized_surface_impedance)
        
        self.impedance2alpha()
        
        self.absorber_type = "microperforated panel"  
        
    
    def impedance2alpha(self, method="thomasson", a=11**0.5, b=11**0.5, **kwargs):

        """
        Computes absorption coeffients from complex impedances (or admittances) using Thomasson formulation or the Paris Formula
        """    
            
        ############################################################    
        if "f_list" in kwargs:
            f_list = kwargs.get("f_list")    
            k0 = 2*np.pi*f_list/self.c0
            
            if self.freq.size == 0:
                self.freq = f_list
                self.w = 2*np.pi*self.freq
                self.k0 = self.w/self.c0
                
        elif self.freq.size == 0:
            raise ValueError("Frequency vector is empty.")  
            
        else:
            f_list = self.freq
            k0 = self.k0
            
        ############################################################             
        if self.admittance.size == 0: 
            if self.normalized_surface_impedance.size == 0:
                raise ValueError("There is no information about the surface impedance (or admittance) of this material yet.") 
        else:
            if self.normalized_surface_impedance.size == 0:
                self.surface_impedance = 1/np.conj(self.admittance)
        
        self.normal_incidence_alpha = np.zeros(len(self.surface_impedance))
        
        for zsi, zs in enumerate (self.normalized_surface_impedance):

            vp =  (zs - 1)/(zs + 1)    
            self.normal_incidence_alpha[zsi] = 1 - (abs(vp))**2
            
            
        if method == "thomasson":
            
            self.statistical_alpha = np.zeros(len(self.surface_impedance))

            for zsi, zs in enumerate (self.normalized_surface_impedance):

                def alpha_fun(theta):

                    mi = np.sin(theta)
                    ke = (2*k0[zsi]*a*b) / (a+b)
                    kappa = 0.956 / ke
                    z_h = 1 / ((1 + (kappa-1j*mi)**2)**(1/2))

                    def h(q):
                        h = np.log((1+q**2)**(1/2) + q) - ((1+q**2)**(1/2)-1)/(3*q)
                        return h

                    z_l = (2*k0[zsi]*a*b / np.pi) + 1j*(2*k0[zsi] / np.pi) * (b*h(a/b) + a*h(b/a))

                    z_r = 1 / ((1/(z_l.real**2))**(1/2) + (1/(z_h.real**2))**(1/2))


                    z_hi0 = 0.67/ke
                    z_i0 = 1 / ((1/(z_l.imag**3))**(1/3) + (1/(z_hi0**3))**(1/3))
                    z_i = np.max((z_i0, z_h.imag))

                    z_radiation = z_r + 1j*z_i

                    alpha_fun = zs.real*np.sin(theta) / (abs(zs + z_radiation))**2

                    return alpha_fun

                self.statistical_alpha[zsi] = 8 * abs(scipy.integrate.quad(alpha_fun, 0, np.pi/2)[0])
                
            
        elif method == "paris":
                
            self.statistical_alpha = np.zeros(len(self.normalized_surface_impedance))

            for zsi, zs in enumerate (self.normalized_surface_impedance):

                def alpha_fun(theta):

                    vp =  (zs*np.cos(theta) - 1)/(zs*np.cos(theta) + 1)    
                    alpha = 1 - (abs(vp))**2
                    alpha_s = alpha*np.sin(2*theta)

                    return alpha_s

                self.statistical_alpha[zsi] = abs(scipy.integrate.quad(alpha_fun, 0, np.pi/2)[0])
        
        else:
            raise ValueError("Method is not valid. You must use \"paris\" or \"thomasson\".")
        
        if "f_list" in kwargs:
            self.alpha_in_bands(f_list=f_list)
        else:
            self.alpha_in_bands()

           
    def _55_rule(self):
        
        """
        Computes real valued impedances from statistical absorption coefficients in octave bands considering the 55 degree rule.
        """
        
        if self.freq.size == 0:
            raise ValueError("Frequency vector is empty.") 
            
        if self.third_octave_bands_statistical_alpha.size != 0 and self.third_octave_bands.size != 0:
            lower_limit = self.lower[1]
            upper_limit = self.upper[1]
            center_freq = self.center[1]
            
            bands = self.third_octave_bands
            alpha = self.third_octave_bands_statistical_alpha

        elif self.octave_bands_statistical_alpha.size != 0 and self.octave_bands.size != 0:
            lower_limit = self.lower[0]
            upper_limit = self.upper[0]
            center_freq = self.center[0]
            
            bands = self.octave_bands
            alpha = self.octave_bands_statistical_alpha
            
        else:
            raise ValueError("There is not enough information about this material yet.") 
            
        tck_alpha = interpolate.splrep(bands, alpha, k=1)
        
        admittance = np.zeros(len(self.freq), dtype = np.complex64)
        statistical_alpha = np.zeros(len(self.freq))
        for fi,f in enumerate(self.freq):
            
            if f < bands[0]:
                admittance[fi] = np.cos(55*np.pi/180)*(1-(1-interpolate.splev(bands[0], tck_alpha, der=0))**0.5)/(1+(1-interpolate.splev(bands[0], tck_alpha, der=0))**0.5)       
                statistical_alpha[fi] = interpolate.splev(bands[0], tck_alpha, der=0)
                
            elif f > bands[-1]:
                admittance[fi] = np.cos(55*np.pi/180)*(1-(1-interpolate.splev(bands[-1], tck_alpha, der=0))**0.5)/(1+(1-interpolate.splev(bands[-1], tck_alpha, der=0))**0.5) 
                statistical_alpha[fi] = interpolate.splev(bands[-1], tck_alpha, der=0)
                
            else:
                admittance[fi] = np.cos(55*np.pi/180)*(1-(1-interpolate.splev(f, tck_alpha, der=0))**0.5)/(1+(1-interpolate.splev(f, tck_alpha, der=0))**0.5)       
                statistical_alpha[fi] = interpolate.splev(f, tck_alpha, der=0)
                
            if np.isnan(admittance[fi]):
                admittance[fi] = np.cos(55*np.pi/180)*(1-(1-1)**0.5)/(1+(1-1)**0.5)
        
        self.admittance = admittance
        self.statistical_alpha = statistical_alpha
        self.surface_impedance = 1/self.admittance
        self.normalized_surface_impedance = self.surface_impedance/(self.rho0*self.c0)
             
        
    def impedance_thru_rmk1(self, **kwargs):
    
        """
        Computes complex surface impedances impedances using the RMK + 1 method

        Parameters = [k, r, m, g, gama]
        The parameters are normalize by their typical orders of magnitude to facilitate 
        the progress of the algorithm
        """
        ############################################################    
        if "f_list" in kwargs:
            f_list = kwargs.get("f_list")
            w = 2*np.pi*f_list
                        
            if self.freq.size == 0:
                self.freq = f_list
                self.w = 2*np.pi*self.freq
                self.k0 = self.w/self.c0
                
        elif self.freq.size == 0:
            raise ValueError("Frequency vector is empty.") 
            
        else:
            f_list = self.freq
            w = self.w
            
        ############################################################    
        if "parameters" in kwargs:
            self.rmk1 = kwargs.get("parameters")
            
        elif self.rmk1.size == 0:
            raise ValueError("It is not defined rmk+1 parameters yet.")  
        
        ############################################################    
        k = self.rmk1[0]*(10**4)
        r = self.rmk1[1]
        m = self.rmk1[2]*(10**-4)
        g = self.rmk1[3]*(10)
        gama = self.rmk1[4]

        
        self.normalized_surface_impedance = k*(1j*w)**(-1) + r + m*(1j*w) + g*(1j*w)**gama   

        self.surface_impedance = self.normalized_surface_impedance*(self.rho0*self.c0)
        self.admittance = 1/np.conj(self.normalized_surface_impedance)
        
        if "f_list" in kwargs:
            self.impedance2alpha(f_list=f_list)
        else:
            self.impedance2alpha()
            
    
    def impedance_from_alpha (self, **kwargs):

        '''
        This function computes surface impedances from statistical absorption coefficients. 
        This technique was first presented by Mondet in 2020 and is based on the solution 
        of a constrained optimization problem

        The **kwargs (optional input argument) must be passed if you have not calculated the material properties using the methods of this module.
        It must be passed as the keyword "absorber".
        absorber_type -> string that defines the type of the absorber. It is gonna be used to define 
                    the constraints of the optimization problem. The possible values are:

                        - "soft porous" -> porous absorber with flow resistivity < 20 000 [rayl/m] and
                                           thickness < 0.1 [m]           
                        - "hard porous" -> porous absorber which parameters exceeds the "soft porous" limits
                        - "perforated panel" 
                        - "microperforated panel"
                        - "membrane"
                        - "hard" -> generic hard material (the result is gonna be the same as if it is considered microperforated panel)

                    Obs: to a porous absorver with an air cavity, the values of the flow resistivity and 
                         thickness limits even decrease as the cavity depth increases                                     
        '''

        if "absorber_type" in kwargs:
            self.absorber_type = kwargs.get("absorber_type")
            
        elif hasattr(self, "absorber_type") != True:
            raise ValueError("Type of absorber is not defined.")
               
        if self.absorber_type != "soft porous" and self.absorber_type != "hard porous" and self.absorber_type != "perforated panel" and self.absorber_type != "microperforated panel" and self.absorber_type != "membrane" and self.absorber_type != "hard":
            raise ValueError("Invalid absorber; must be one of soft porous, hard porous, perforated panel, microperforated panel or membrane.")
        
        ################################################
        # Gets the needed information considering available data
        
        if self.third_octave_bands.size != 0 and self.third_octave_bands_statistical_alpha.size !=0:
            bands = self.third_octave_bands
            alpha_in = self.third_octave_bands_statistical_alpha
            upper_limit = self.upper[1]
            lower_limit = self.lower[1]
            center_freq = self.center[1]
            type = "third_octave_bands"
        
        elif self.octave_bands.size != 0 and self.octave_bands_statistical_alpha.size !=0:
            bands = self.octave_bands
            alpha_in = self.octave_bands_statistical_alpha
            upper_limit = self.upper[0]
            lower_limit = self.lower[0]
            center_freq = self.center[0]
            type = "octave_bands"
            
        else:
            raise ValueError("There is not enough information about the absorber yet. Check if it has been given octave or third-octave bands and \
                             it's corresponding statistical absorption coefficients.")
        
        
        ################################################
        # Computes a list of frequencies (f_list) containing three frequencies per octave (third-octave) band

        aux = 0
        while bands[0] > upper_limit[aux]:
            aux = aux + 1

        f_list = np.array([])
        for band in bands:

            freqs = np.array([np.mean((center_freq[aux], lower_limit[aux])), center_freq[aux], np.mean((upper_limit[aux], center_freq[aux]))])
            f_list = np.append(f_list,freqs)
            aux += 1
            
        #################################################
        # Defines cost function to be minimized
        
        def cost_fun (parameters):

            """
            This funciton defines the cost function to be minimized by the optimization process
            (The cost function is defined as the squared L2-norm between the corresponding 
            statistical absorption coefficient and the input absorption coefficient)

            Parameters = [k, r, m, g, gama]
            The parameters are normalize by their typical orders of magnitude to facilitate 
            the progress of the algorithm
            """

            self.impedance_thru_rmk1(parameters=parameters, f_list=f_list)

            if type == "third_octave_bands":
                difference = alpha_in - self.third_octave_bands_statistical_alpha
            else:
                difference = alpha_in - self.octave_bands_statistical_alpha
                
            squared_l2_norm = np.real(np.inner(difference, difference))

            return squared_l2_norm


        ################################################
        # Next condition statements define the constraints to be used in the optimization
        # problem based on the type of absorber

        if self.absorber_type == "soft porous" or self.absorber_type == "perforated panel" or self.absorber_type == "membrane":

            def constrain(parameters):

                # Parameters = [r, m, k, g, gama]
                # The parameters are normalize by their typical orders of magnitude to facilitate the progress of the algorithm
                
                self.impedance_thru_rmk1(parameters=parameters, f_list=f_list)
                zs = 2 - np.real(self.normalized_surface_impedance)

                return zs

            ineq_cons = {'type': 'ineq',
                         'fun': constrain}

            bounds = ((0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf), (-1, 1)) # (k, r, m, g, gama)
            guesses = np.array([0, 1.6, 0, 0, 0]) # np.array([k, r, m, g, gama])

            if self.absorber_type == "membrane":

                bounds = ((0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf), (0.9, 1)) # (k, r, m, g, gama)
                guesses = np.array([0, 1.6, 0, 0, 0.95]) # np.array([k, r, m, g, gama])

        if self.absorber_type == "hard porous" or self.absorber_type == "microperforated panel" or self.absorber_type == "hard":

            def constrain(parameters):

                '''
                Parameters = [r, m, k, g, gama]

                The parameters are normalize by their typical orders of magnitude to facilitate 
                the progress of the algorithm
                '''
                
                self.impedance_thru_rmk1(parameters=parameters, f_list=f_list)
                zs = np.real(self.normalized_surface_impedance) - 1

                return zs


            ineq_cons = {'type': 'ineq',
                         'fun': constrain}

            bounds = ((0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf), (-1, 1)) # (k, r, m, g, gama)
            guesses = np.array([0, 1.6, 0, 0, 0]) # np.array([k, r, m, g, gama])

        ################################################

        print ("Working on the solution of the constrained optimization problem...")

        if any(alpha > 1 for alpha in alpha_in):
            validation = 0.25
        else:
            validation = 0.1
        
        solution = minimize(cost_fun, guesses, method='SLSQP', constraints = [ineq_cons], bounds = bounds, options={'ftol': 1e-10, 'disp': True, 'maxiter': 1000})
        print(self.rmk1)
        
        i = 0
        for alpha in alpha_in:
            if alpha < 0.1:
                i += 1
        '''
        if any(alpha_in) > 1:
            validation = 0.25
        elif i >= 3:
            validation = 0.001
        else:
            validation = 0.015
        '''
        
        while cost_fun (solution.x) > validation:

            guesses = np.array([uniform(0,2), uniform(0,2), uniform(0,2), uniform(0,2), uniform(-1,1)]) # assign random values between 0 and 2 to all the normalized parameters, with the exception of the exponent, -1 <= gama <= 1
            solution = minimize(cost_fun, guesses, method='SLSQP', constraints = [ineq_cons], bounds = bounds, options={'ftol': 1e-10, 'disp': True, 'maxiter': 1000})
            print(self.rmk1)
        

        self.impedance_thru_rmk1()
        
        print("The solution of the optimization problem leads to rmk+1 parameters equal to %s.\nImpedances, admittances and everything else related to it was already calculated.\n" % self.rmk1)
        
        self.absorber_type = "Generic " + str(self.absorber_type)
        
        
    def alpha_in_bands (self, **kwargs):
    
        """
        Given data and it's corresponding frequencies, calculates these data in octave and third-octave bands.
        It is done directly: the value of a band is simply the mean value of all data inside this band.
        """
        ############################################################    
        if "f_list" in kwargs:
            f_list = kwargs.get("f_list")
            
            if self.freq.size == 0:
                self.freq = f_list
                self.w = 2*np.pi*self.freq
                self.k0 = self.w/self.c0
                
        elif self.freq.size == 0:
            raise ValueError("Frequency vector is empty.")  
            
        else:
            f_list = self.freq
            
        ############################################################ 
        first_band_aux = {0:0,
                          1:0}
        upper_limit = self.upper[0]
        while f_list[0] > upper_limit[first_band_aux[0]]:
            first_band_aux[0] = first_band_aux[0] + 1

        upper_limit = self.upper[1]
        while f_list[0] > upper_limit[first_band_aux[1]]:
            first_band_aux[1] = first_band_aux[1] + 1

        for i in np.arange(len(self.upper)):

            upper_limit = self.upper[i]
            center_freq = self.center[i]
            aux = first_band_aux[i]
            
            data_in_bands = {}
            f_aux = 0
            for fi, f in enumerate(f_list):
                
                if f < upper_limit[aux]:

                    if fi == (len(f_list) - 1):
                        data_in_bands [center_freq[aux]] = np.mean(self.statistical_alpha[f_aux:fi+1])

                    else:
                        pass   

                else:
                    data_in_bands [center_freq[aux]] = np.mean(self.statistical_alpha[f_aux:fi])

                    aux = aux + 1
                    f_aux = fi

                    if fi == (len(f_list) - 1): 
                        data_in_bands [center_freq[aux]] = self.statistical_alpha[fi]

                    while f > upper_limit[aux]:
                        aux = aux + 1

            lists = sorted(data_in_bands.items()) # sorted by key, return a list of tuples
            bands, data_in_bands = zip(*lists) # unpack a list of pairs into two tuples

            if i == 0:
                self.octave_bands = np.array(bands)
                self.octave_bands_statistical_alpha = np.array(data_in_bands)    
                
            if i == 1:
                self.third_octave_bands = np.array(bands)
                self.third_octave_bands_statistical_alpha = np.array(data_in_bands)  

    
    def plot(self, **kwargs):
        """
        Plots the absorption coeffients. If it is not yet defined, it is plotted the complex surface impedance (to be implemented)
        """        
        
        if "type" in kwargs:
            type = kwargs.get("type")
        else:
            type = "statistical in octave bands"
            
            
        if self.octave_bands.size == 0 and type == "statistical in octave bands":
            raise ValueError("Statistical absorption coefficient in octave bands is not defined yet.")

        if self.third_octave_bands.size == 0 and type == "statistical in third octave bands":
            raise ValueError("Statistical absorption coefficient in third octave bands is not defined yet.")

        if self.octave_bands.size == 0 or self.third_octave_bands.size == 0 or type == "statistical" or type == "normal incidence":
            if self.statistical_alpha.size == 0 or self.normal_incidence_alpha.size == 0:
                if self.surface_impedance.size == 0 and self.admittance.size == 0:
                    raise ValueError("There is no information about this material yet.")
            else:
                if type == "normal incidence":
                    plt.plot (self.freq, self.normal_incidence_alpha)
                    plt.title('Normal incidence absorption coefficients')
                    plt.xlabel('Frequency [Hz]')
                    plt.ylabel('Normal incidence absorption coefficient [-]')
                    plt.xscale('log')
                    plt.ylim((0,1.2))
                    plt.show()
                else: 
                    plt.plot (self.freq, self.statistical_alpha)
                    plt.title('Statistical absorption coefficients')
                    plt.xlabel('Frequency [Hz]')
                    plt.ylabel('Statistical absorption coefficient [-]')
                    plt.xscale('log')
                    plt.ylim((0,1.2))
                    plt.show()

        elif self.octave_bands_statistical_alpha.size == 0 and type == "statistical in octave bands":
            raise ValueError("Octave bands have been defined, but not the corresponding statistical absorption coefficients.")

        elif self.third_octave_bands_statistical_alpha.size == 0 and type == "statistical in third octave bands":
            raise ValueError("Third octave bands have been defined, but not the corresponding statistical absorption coefficients.")

        else:
            if type == "statistical in third octave bands":

                plt.plot (self.third_octave_bands, self.third_octave_bands_statistical_alpha, 'o-')
                plt.title('Statistical absorption coefficients in third-octave bands')
                plt.xlabel('Frequency [Hz]')
                plt.ylabel('Statistical absorption coefficient [-]')
                plt.xscale('log')
                plt.ylim((0,1.2))
                plt.show()

            else:
                plt.plot (self.octave_bands, self.octave_bands_statistical_alpha, 'o-')
                plt.title('Statistical absorption coefficients in octave bands')
                plt.xlabel('Frequency [Hz]')
                plt.ylabel('Statistical absorption coefficient [-]')
                plt.xscale('log')
                plt.ylim((0,1.2))
                plt.show()
    
    
    def adjust(self):
        '''
        Adjusts automatically information about the material based on init data.
        '''
        
        if self.admittance.size != 0:
            self.normalized_surface_impedance = 1/self.admittance
            self.surface_impedance = self.normalized_surface_impedance * (self.rho0*self.c0)
            self.impedance2alpha()
        
        elif self.normalized_surface_impedance.size != 0:
            self.surface_impedance = self.normalized_surface_impedance * (self.rho0*self.c0)
            self.admittance = 1/(self.normalized_surface_impedance)
            self.impedance2alpha()
            
        elif self.surface_impedance.size != 0:
            self.normalized_surface_impedance = np.conj(self.surface_impedance)/(self.rho0*self.c0)
            self.admittance = 1/(self.normalized_surface_impedance)
            self.impedance2alpha()                 
        
        
    def __str__(self):
        
        if hasattr(self, "absorber_type"):
            
            if self.absorber_type == "rigid":
                return ("Ideal rigid surface.\n")
            
            if self.absorber_type == "soft porous":
                return ("Single layer soft porous absorber with rigid back end. Flow resistivity = " + str(self.flow_resistivity) + " [rayl/m] and thickness = " 
                + str(self.thickness) + " [m].\n")

            elif self.absorber_type == "hard porous":
                return ("Single layer hard porous absorber with rigid back end. Flow resistivity = " + str(self.flow_resistivity) + " [rayl/m] and thickness = " 
                + str(self.thickness) + " [m].\n")

            elif self.absorber_type == "porous with air cavity":
                return ("Porous absorber with air cavity back end. Flow resistivity = " + str(self.flow_resistivity) + " [rayl/m] and material thickness = " 
                + str(self.thickness) + "[m]. Air cavity depth = " + str(self.air_cavity_depth) + " [m].\n")

            elif self.absorber_type == "membrane":
                return ("Membrane absorber. The mass per unit area of the membrane is " + str(self.mass_per_unit_area) + " [kg/m^2].\nThe total cavity depth is " 
                + str(self.cavity_depth) + " [m], being " + str(self.porous_layer_thickness) + " [m] of porous a porous material with flow resistivity = " 
                + str(self.flow_resistivity) + " [rayl/m].\n")

            elif self.absorber_type == "perforated panel":
                return ("Perforated panel absorber. The panel thickness is " + str(self.panel_thickness) + " [m].\nThe opening radius is " 
                + str(self.openings_radius) + " [m], being the perforation rate " + str(self.perforation_rate) 
                + ". \nThe total cavity depth is " + str(self.cavity_depth) + " [m] and the porous absorver layer thickness is " + str(self.porous_layer_thickness) 
                + " [m].\nThe flow resistivity of the porous absorber is " + str(self.flow_resistivity) + " [rayl/m].\n")

            elif self.absorber_type == "microperforated panel":
                return ("Microperforated panel absorber. The panel thickness is " + str(self.panel_thickness) + " [m].\nThe opening radius is " 
                + str(self.openings_radius) + " [m], being the perforation rate " + str(self.perforation_rate) 
                + ".\nThe air cavity depth is " + str(self.air_cavity_depth) + " [m].\n")
        
            elif self.absorber_type == "hard":
                return ("Generic hard material.\n")
            
            else:
                return (str(self.absorber_type) + " absorber. There is no more information about it.\n")
        
        else:
            return("Generic material. Use plot method to get more information.\n")
        
        
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


