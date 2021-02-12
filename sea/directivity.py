import numpy as npp
import scipy
import scipy.io
import pickle

from scipy.special import lpmv, spherical_jn, spherical_yn

class Directivity:
    
    def __init__(self, data_path, rho0, c0, freq_vec, simulated_ir_duration, measurement_radius, sh_order, type, sample_rate=44100, **kwargs):
        
        '''
        This script encodes the measured impulse responses, representing the
        directivity of a source into sperical harmonic coefficients 
        
        source_data_path -> path that leads to the .mat file that contains the source data
        source_name -> string. It is gonna be used as the name of the file where the solution is gonna be saved
        rho0 -> air density
        c0 -> speed of sound
        simulated_ir_duration -> length of simulation [s]                                                                       
        measurement_radius -> distance from source to measurement positions [m]
        existing_pre_delay -> delay before direct sound arrives as provided in GRAS dataset [samples]    
        
        '''
        
        self.data_path = data_path
        self.rho0 = rho0
        self.c0 = c0
        self.freq_vec = freq_vec
        self.simulated_ir_duration = simulated_ir_duration
        self.measurement_radius = measurement_radius
        self.sh_order = sh_order
        self.type = type
        self.sample_rate = sample_rate
        
        
        try:
            self.existing_pre_delay = kwargs["existing_pre_delay"]           
        except:
            pass
        
        
    def encode_directivity (self, file_name):
        
        self.file_name = file_name
        
        # Derived parameters:
        nfft = self.sample_rate*self.simulated_ir_duration  # Number of FFT points
        f_list = self.sample_rate*np.arange(nfft)/nfft    # List of FFT frequencies
        fi_lim_lo = np.argmin(np.abs(f_list - self.freq_vec[0]))  # FFT bins above which to encode
        fi_lim_hi = np.argmin(np.abs(f_list - self.freq_vec[-1]))  # FFT bins below which to encode (Hz)
        f_list = f_list[fi_lim_lo:fi_lim_hi + 1]         # Only retain frequencies to be encoded
        
        if self.type == "source":
            
            ## Load and adjust meaured impulse responses:
            
            # Load measured impulse responses:
            print ('Loading source data. It might be computationally costing...')
            source_data = scipy.io.loadmat(self.data_path) # loads variables IR, Phi, Theta
            ir = np.array (source_data['IR'])
            
            # Convert measurement angles from degrees to radians & ensure are column vectors
            beta  = np.array(source_data['Theta']) * np.pi/180
            beta = beta.reshape((np.size(beta), 1))
            alpha = np.array(source_data['Phi'])   * np.pi/180
            alpha = alpha.reshape((np.size(alpha), 1))
            
            del source_data
            
            # Correct initial time delay for measurement distance and window:                                        
            desired_pre_delay = round(self.sample_rate * self.measurement_radius / self.c0)   # Delay before direct sound arrives based on measurement radius (#samples)
            
            half_window = np.concatenate((np.array(0.5-0.5*np.cos(np.pi*np.linspace(0, 1, self.existing_pre_delay))).conj().T, np.array(np.ones((np.size(ir,0) - self.existing_pre_delay))))) # Rising window
            half_window = half_window.reshape((np.size(half_window), 1))
            ir = np.concatenate((np.zeros((desired_pre_delay - self.existing_pre_delay, np.size(ir,1))), np.multiply(ir, half_window)))
            half_window = np.concatenate((np.ones((np.ceil(np.size(ir,0)/2).astype(int),1)), 0.5+0.5*np.cos(np.pi*np.linspace(0, 1, np.floor(np.size(ir,0)/2).astype(int)).conj().T.reshape(np.floor(np.size(ir,0)/2).astype(int),1)))) # Falling window
            ir = np.multiply(ir, half_window);
                    
            # Derived parameters:
            num_meas = np.size(ir,1);   # Number of measurment points
            
            ## Fourier transform the impulse responses:
            
            # Loop over measurement points and FFT:
            phi_meas = np.zeros((fi_lim_hi-fi_lim_lo+1, num_meas), dtype = np.complex128)
            print ('Computing FFTs')
            for iMeas in range(num_meas):
            
                fft_ir = np.conj(np.fft.fft(ir[:,iMeas], n = nfft))       # conj used because project uses exp(-1i*w*t) Fourier Transform
                phi_meas[:,iMeas] = np.array([fft_ir[fi_lim_lo:fi_lim_hi + 1]]) # Only retain frequencies to be encoded
            
            del ir, fft_ir, iMeas, fi_lim_lo, fi_lim_hi
            
            # Transpose to optimise memory access for encoding step:
            print ('Transposing transfer function array...')
            phi_meas = np.transpose(phi_meas)
            print ('Complete.')
            
            ## Encoding:
            
            # Create weighting vector:
            w = np.pi/180*(np.cos(beta-np.pi/360)-np.cos(beta+np.pi/360));
            w[0] = 2*np.pi*(1-np.cos(np.pi/360));
            w[-1] = w[0];
            w = w.reshape((np.size(w), ))
            print ('Weight addition error = %s.' % abs(np.sum(w) - (4*np.pi)))
            
            # Pre-calculate spherical harmonic functions:
            y_nm, dy_dbeta, dy_dalpha = spherical_harmonic_all(self.sh_order, alpha,beta);
            
            
            # Loop over frequency:
            self.sh_coefficients = []
            i = 0
            for fi, f in enumerate (f_list):
                if f == self.freq_vec[i]:
                    
                    # Calculate spherical Hankel functions: 
                    hnOut = np.zeros(((self.sh_order+1)**2, 1), dtype = np.complex128);
                    for n in np.arange(self.sh_order+1):
                        for m in np.arange(-n, n + 1):
                            hnOut[sub2indSH(m,n),0] = spherical_hankel_out(n, self.measurement_radius*2*np.pi*f/self.c0)
                
                    # Calculate b_nm coefficients via a mode-matching approach (Eq. 9 in paper):
                    sh_coefficients_f = np.matmul(y_nm.conj().T, np.transpose(np.divide(np.multiply(w, phi_meas[:,fi]), hnOut)))
                    sh_coefficients_f = np.diagonal(sh_coefficients_f)
                    
                    self.sh_coefficients.append(sh_coefficients_f)
                    
                    i+=1
        
        
        elif self.type == "receiver":
            
            # Load measured impulse responses:
            print ('Loading receiver directionality data. It might be computationally costing...')
            receiver_data = scipy.io.loadmat(self.data_path)  # loads variables HRIR_R,HRIR_L, Phi, Theta
            
            hrir_l = np.array(receiver_data['HRIR_L'])
            hrir_r = np.array(receiver_data['HRIR_R'])
            
            azimuth = np.array(receiver_data['azimuth'])
            azimuth = azimuth.reshape((np.size(azimuth), 1))
            
            elevation = np.array(receiver_data['elevation'])
            elevation = elevation.reshape((np.size(elevation), 1))
            
            # Convert measurement angles from degrees to radians, and from elevation to polar
            alpha = np.multiply(np.divide(azimuth, 360), (2*np.pi))
            beta = np.multiply(np.divide(np.subtract(90, elevation), 360), (2*np.pi))
            
            del receiver_data, azimuth, elevation
            
            # Derived parameters:
            ir_length = np.size(hrir_l, 0) # Length of recorded impulse response (#samples)
            num_meas = np.size(hrir_l, 1)   # Number of measurment points
            
            
            ## Fourier transform the impulse responses - left:
            
            # The IR is windowed with a half-Hanning window applied to its last 25%, to
            # avoid a wrap-around discontinity of and then zero-padded to achieve the
            # required frequency resolution. 
            half_window = np.concatenate((np.ones((np.ceil(ir_length/2).astype(int),1)), np.array(0.5+0.5*np.cos(np.pi*np.linspace(0, 1, np.floor(ir_length/2).astype(int)).conj().T)).reshape((np.size(np.linspace(0, 1, np.floor(ir_length/2).astype(int))), 1))))
            half_window = half_window.reshape((np.size(half_window), ))
            
            ## Encoding:
            
            # Pre-calculate spherical harmonic functions:
            y_nm, dy_dbeta, dy_dalpha = spherical_harmonic_all(self.sh_order, alpha, beta)
            
            # Pre-calculate spherical Hankel functions:
            hnOut = np.zeros(((self.sh_order+1)**2, np.size(f_list)), dtype = np.complex128)
            for fi, f in enumerate(f_list):
                for n in np.arange(self.sh_order + 1):
                    for m in np.arange(-n, n + 1):
                        hnOut[sub2indSH(m,n),fi] = spherical_hankel_out(n, self.measurement_radius*2*np.pi*f/self.c0)
                
            
                
            # Loop over measurement points and FFT - left:
            hrtf = np.zeros((fi_lim_hi-fi_lim_lo+1, num_meas), dtype = np.complex128)
            
            for i_meas in range(num_meas):
                
                fft_hrir = np.conj(np.fft.fft(np.multiply(half_window, hrir_l[:,i_meas]), n = nfft)) # conj used because project uses exp(-1i*w*t) Fourier Transform
                hrtf[:,i_meas] = np.array([fft_hrir[fi_lim_lo:fi_lim_hi + 1]])    # Only retain frequencies to be encoded
                
            del hrir_l, fft_hrir, i_meas
                
            # Transpose to optimise memory access for encoding step:
            print('\tTransposing transfer function array...')
            hrtf = np.transpose(hrtf)
            print('Complete.\n')
            
            # Loop over frequency - left:
            self.sh_coefficients_left = []
            i = 0
            for fi, f in enumerate (f_list):
                if f == self.freq_vec[i]:           
                    # Calculate Lnm coefficients by a least-squares fit approach:
                    A = np.multiply(4*np.pi*np.transpose(hnOut[:,fi])/hnOut[0,fi], np.conj(y_nm))
                    sh_coefficients_left_f = np.linalg.lstsq (A, hrtf[:,fi])
                    
                    self.sh_coefficients_left.append(sh_coefficients_left_f[0]) 
                    
                    i+=1
                
            
            # Loop over measurement points and FFT - right:
            hrtf = np.zeros((fi_lim_hi-fi_lim_lo+1, num_meas), dtype = np.complex128)
            
            for i_meas in range(num_meas):
                
                fft_hrir = np.conj(np.fft.fft(np.multiply(half_window, hrir_r[:,i_meas]), n = nfft)) # conj used because project uses exp(-1i*w*t) Fourier Transform
                hrtf[:,i_meas] = np.array([fft_hrir[fi_lim_lo:fi_lim_hi + 1]])              # Only retain frequencies to be encoded
                
            del hrir_r, fft_hrir, i_meas
                
            # Transpose to optimise memory access for encoding right:
            print('\tTransposing transfer function array...')
            hrtf = np.transpose(hrtf)
            print('Complete.\n')
            
            
            # Loop over frequency - right:
            self.sh_coefficients_right = []
            i = 0
            for fi, f in enumerate (f_list):
                if f == self.freq_vec[i]:
                    # Calculate Lnm coefficients by a least-squares fit approach:
                    A = np.multiply(4*np.pi*np.transpose(hnOut[:,fi])/hnOut[0,fi], np.conj(y_nm))
                    sh_coefficients_right_f = np.linalg.lstsq (A, hrtf[:,fi])
                    
                    self.sh_coefficients_right.append(sh_coefficients_right_f[0]) 
                    
                    i+=1
        
        
        else:
            raise ValueError("Type is not valid. It must be source or receiver.")
            
            
        save_name = "%s.pickle" % self.file_name
        pickle_obj = open(save_name, "wb")
        pickle.dump(self, pickle_obj)
        pickle_obj.close()            
                
        print ("Saved results to %s.pickle" % self.file_name)
            
#### Functions #####

def sub2indSH (m,n):
    
	"""
	i = sub2indSH(m,n)
	
	Convert Spherical Harmonic (m,n) indices to array index i
	Assumes that i iterates from 0 (Python style)
    """
    
	i = n**2 + n + m
    
	return i


def spherical_harmonic_all (max_order, alpha, beta):
    
    """
    (y, dy_dbeta, dy_dalpha) = spherical_harmonic_all(max_order, alpha, sinbeta, cosbeta)
    	
    Computes a Spherical Harmonic function and it's angular derivatives for
    all (m,n) up to the given maximum order. The algorithm is equivalent to that
    implemented in SphericalHarmonic, but this version avoids repeated calls
    to lpmv, since that is very time consuming.
    	
    Arguments - these should all be scalars:
    r is radius
    alpha is azimuth angle (angle in radians from the positive x axis, with
    rotation around the positive z axis according to the right-hand screw rule)
    beta is polar angle, but it is specified as two arrays of its cos and sin values. 
    max_order is maximum Spherical Harmonic order and should be a non-negative real integer scalar
    	
    Returned data will be vectors of length (max_order+1)^2.
    
    """
    
    cosbeta = np.cos(beta)
    sinbeta = np.sin(beta)

    # Preallocate output arrays:
    y =  np.zeros((np.size(alpha),(max_order+1)**2), np.complex128)
    dy_dbeta = np.zeros((np.size(alpha),(max_order+1)**2), np.complex128)
    dy_dalpha = np.zeros((np.size(alpha),(max_order+1)**2), np.complex128)
    
    #% Loop over n and calculate spherical harmonic functions y_nm
    for n in range(max_order+1):
    
        # Compute Legendre function and its derivatives for all m:
        p_n = lpmv(range(0,n+1), n, cosbeta)
        #print (np.shape(p_n))
        #shape_p_n = np.shape(p_n)
        #p_n = p_n.reshape((shape_p_n[1], shape_p_n[0]))
        
        for m in range(-n, n+1):
    
            # Legendre function its derivatives for |m|:
            p_nm = p_n[:, np.absolute(m)]
            p_nm = p_nm.reshape((np.size(p_nm), ))

            if n==0:
                dPmn_dbeta = 0
            elif m==0:
                dPmn_dbeta = p_n[:,1]
            elif abs(m)<n:
                dPmn_dbeta = 0.5*p_n[:,abs(m)+1] - 0.5*(n+abs(m))*(n-abs(m)+1)*p_n[:,abs(m)-1];
                dPmn_dbeta = dPmn_dbeta.reshape((np.size(dPmn_dbeta), ))
            elif (abs(m)==1) and (n==1):
                dPmn_dbeta = -cosbeta
                dPmn_dbeta = dPmn_dbeta.reshape((np.size(dPmn_dbeta), ))
            #elif sinbeta<=np.finfo(float).eps:
                #dPmn_dbeta = 0
            else:
                dPmn_dbeta = -abs(m)*cosbeta.reshape((np.size(cosbeta), ))*p_nm/sinbeta.reshape((np.size(sinbeta), )) - (n+abs(m))*(n-abs(m)+1)*p_n[:,abs(m)-1]
                dPmn_dbeta = dPmn_dbeta.reshape((np.size(dPmn_dbeta), ))
            
            #print (dPmn_dbeta)
            #print (np.shape(dPmn_dbeta))
            # Compute scaling term, including sign factor:
            scaling_term = ((-1)**m) * np.sqrt((2 * n + 1) / (4 * np.pi * np.prod(np.float64(range(n-abs(m)+1, n+abs(m)+1)))))
    
            # Compute exponential term:
            exp_term = np.exp(1j*m*alpha)
            exp_term = exp_term.reshape((np.size(exp_term), ))
            
            # Put it all together:
            i = sub2indSH(m,n)
            #y[:,i] = np.multiply (exp_term, p_nm)
            y[:,i] = scaling_term * exp_term * p_nm
            dy_dbeta[:,i] = scaling_term * exp_term * dPmn_dbeta
            dy_dalpha[:,i] = y[:,i] * 1j * m
            
    return y, dy_dbeta, dy_dalpha


def spherical_hankel_out (n, z):
    
    """
    (h, dhdz) = spherical_hankel_out(n, z)
    	
    Computes a spherical Hankel function of the first kind (outgoing in this
    paper's lingo) and its first derivative.
    """
    
    h = spherical_jn(n,z,False) + 1j*spherical_yn(n,z,False)
    #h = h[0,0]
    #dhdz = spherical_jn(n,z,True) + 1j*spherical_yn(n,z,True)
    return h #, dhdz
