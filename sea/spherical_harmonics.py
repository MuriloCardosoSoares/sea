"""
This module contains the functions needed to use the spherical harmonic techniques
"""

import numpy as np
import scipy.sparse
from scipy.special import lpmv, spherical_jn, spherical_yn

@jit
def sub2indSH(m,n):
	"""
	i = sub2indSH(m,n)
	
	Convert Spherical Harmonic (m,n) indices to array index i
	Assumes that i iterates from 0 (Python style)
    """
    
	i = n**2 + n + m
	return i


@jit
def ind2subSH(i):
	"""
	(m,n) = ind2subSH(i)
	
	Convert array index i to Spherical Harmonic (m,n) indices
	Assumes that i iterates from 0 (Python style)
	Assumes that arguments are NumPy arrays
    """
    
	n = np.ceil(np.sqrt(i+1)-1);
	m = i - n**2 - n;	
	return (m,n)


@jit
def cart2sph(x,y,z):
    """
    (r, alpha, sinbeta, cosbeta) = Cart2Sph(x,y,z)
    	
    Converts Cartesian coordinates (x,y,z) into Spherical Polar Coordinates
    (r, alpha, beta), where alpha is azimuth angle (angle in radians from the
    positive x axis, with rotation around the positive z axis according to
    the right-hand screw rule) and beta is polar angle (angle in radians from
    the positive z axis). beta can alternatively be returned as two arrays of
    its cos and sin values.
    	
    It is assumed that x, y and z are all the same size.
    The returned arrays will be the same size as the arguments.
    """
    
    r = np.sqrt(x**2 + y**2 + z**2)
    rho = np.sqrt(x**2 + y**2)
    alpha = np.arctan2(y, x)
    cosbeta = z / r
    sinbeta = rho / r
    
    return r, alpha, sinbeta, cosbeta


@jit
def reflect_sh(Bnm, xFlag, yFlag, zFlag):
    """
    Bnm = ReflectSH(Bnm, xFlag, yFlag, zFlag)   
    
    Reflect an Spherical Harmonic representation of a sound-field in 1 to 3 cartesian axes.
    	
    Argumments:
    Bnm       Vector of Spherical Harmonic weights. Must have (Order+1)^2 entries, where Order is an integer.
    xFlag		Boolean indicating whether to flip in the x-direction
    yFlag		Boolean indicating whether to flip in the y-direction
    zFlag		Boolean indicating whether to flip in the z-direction
    """

	# Get lists of n and m values:
    (m, n) = ind2subSH(np.arange(Bnm.size))

	# Reflecting in Z:
    if zFlag:
        Bnm = Bnm * ((-1)**(n+m)).reshape((np.size(m),1))

	# Reflecting in X:
    if xFlag:
        Bnm = Bnm * ((-1)**m).reshape((np.size(m),1))

	# Reflecting in X or Y:
    if xFlag**yFlag: # XOR
        #for n in range(int(np.ceil(np.sqrt(Bnm.size)))-1):
        for n in np.arange(max(n)+1):
            i = sub2indSH(np.arange(-n,n+1),n).astype(int)
            Bnm[i,0] = np.flip(Bnm[i,0])
	
    return Bnm


@jit
def get_translation_matrix(t,k,OrderS,OrderR):
	"""
	T = GetTranslationMatrix(t,k,OrderS,OrderR)

	Computes a translation matrix T from the coefficients of a Spherical
	Harmonic source (outgoing spherical Hankel radial functions) to the
	coefficients at a Spherical Harmonic receiver (spherical Bessel radial
	functions) location at position t relative to the source. It is assumed
	that both spherical coordinate systems (source and receiver) are aligned
	to the same Cartesian system in which t is expressed. a is the polar
	angle from the postive z axis.

	Essentially computes equation 3.2.17 of: 
	Gumerov, N., & Duraiswami, R. (2005). Fast Multipole Methods for the
	Helmholtz Equation in Three Dimensions (1st ed.). Elsevier Science.

	Arguments:

	t         Cartesian translation vector (1x3 real row vector)
	k         Wavenumber (positive real scalar or vector in radians/meter)
	OrderS    Order of the source (non-negative real integer scalar)
	OrderR    Order of the receiver (non-negative real integer scalar)

	This file also contains the sub-functions
	GetStructuralTranslationCoefficients and Wigner3jSymbol.
	"""

	OrderT = OrderS + OrderR

	S = GetStructuralTranslationCoefficients(OrderS,OrderR)

	# Express t in spherical coordinates:
	[r,alpha,sinbeta,cosbeta] = cart2sph(t[0],t[1],t[2])

	# Evaluate spherical harmonic functions:
	Y, dy_dbeta, dy_dalpha = spherical_harmonic_all(OrderT, np.array([[alpha]]), np.array([[sinbeta]]), np.array([[cosbeta]]))


	# Allocated results array:
	T = np.zeros(((OrderR+1)**2, (OrderS+1)**2))


	# Loop over translation order & compute summation:
	for nT in np.arange(OrderT+1):
		h, dhdz = spherical_hankel_out(nT,k*r) # Compute radial function:

		for mT in np.arange(-nT, nT+1):
			iT = sub2indSH(mT,nT)
			T = T + h * Y[0][iT.astype(int)] * S[iT.astype(int),:,:] #!!!

	return T
                

@jit
def GetStructuralTranslationCoefficients(OrderS,OrderR):
	"""
	S = GetStructuralTranslationCoefficients(OrderS,OrderR)

	Computes the 'Structural Translation Coefficients' used in Spherical
	Harmonic translation routines, as defined in section 3.2.1 of: 
	Gumerov, N., & Duraiswami, R. (2005). Fast Multipole Methods for the
	Helmholtz Equation in Three Dimensions (1st ed.). Elsevier Science.

	Arguments:
	OrderS    Order of the source   (non-negative real integer scalar)
	OrderR    Order of the receiver (non-negative real integer scalar)

	Returned variable is a 3D array of size [(OrderR+1)**2, (OrderS+1)**2,
	(OrderR+OrderS+1)**2]. 
	"""

	# Order required for translation:
	OrderT = OrderS + OrderR

	# Allocate cell array:
	S = np.zeros(((OrderT+1)**2, (OrderR+1)**2, (OrderS+1)**2), dtype = np.complex64)

	# Loop over translation order (n2 & m2):
	for nT in  np.arange(OrderT+1, dtype = np.float64): # n'' in book
		for mT in np.arange(-nT, nT+1, dtype = np.float64): # m'' in book
			iT = sub2indSH(mT,nT)
			if mT < 0: # because m'' is negated
				epT = (-1)**mT
			else:
				epT = 1.0

			# Loop over source order (nS & mS):
			for nS in np.arange(OrderS+1, dtype = np.float64): # n in book
				for mS in np.arange(-nS, nS+1, dtype = np.float64): # m in book
					if mS > 0:
						epS = (-1)**mS
					else:
						epS = 1.0

					# Loop over recevier order (nR & mR):
					for nR in np.arange(OrderR+1, dtype = np.float64): # n' in book
						for mR in np.arange(-nR, nR+1, dtype = np.float64): # m' in book
							if mR < 0: # because m' is negated
								epR = (-1)**mR
							else:
								epR = 1.0

							# Compute coefficient if within non-zero range:
							if nT >= abs(nR-nS) and nT <= (nR+nS):
								S[iT.astype(int), sub2indSH(mR,nR).astype(int), sub2indSH(mS,nS).astype(int)] = (
											1j**(nR+nT-nS) * epS * epR * epT 
											* np.sqrt(4*np.pi*(2*nS+1)*(2*nR+1)*(2*nT+1))
											* Wigner3jSymbol(nS, nR, nT, mS, -mR, -mT)
											* Wigner3jSymbol(nS, nR, nT, 0, 0, 0) 
											)

	return S


@jit
def Wigner3jSymbol(j1, j2, j3, m1, m2, m3):
	"""
	W3jS = Wigner3j(j1, j2, j3, m1, m2, m3)

	Computes the Wigner 3j symbol following the formulation given at
	http://mathworld.wolfram.com/Wigner3j-Symbol.html.

	Arguments:

	j1, j2, j3, m1, m2 and m3     All must be scalar half-integers

	Check arguments against 'selection rules' (cited to Messiah 1962, pp. 1054-1056; Shore and Menzel 1968, p. 272)
	Nullifying any of these means the symbol equals zero.
	"""

	if abs(m1)<=abs(j1) and abs(m2)<=abs(j2) and abs(m3)<=abs(j3) and m1+m2+m3==0 and abs(j1-j2)<=j3 and j3<=(j1+j2) and np.remainder(j1+j2+j3,1)==0:

		# Evaluate the symbol using the Racah formula (Equation 7):

		# Evalaute summation:
		W3jS = 0
		for t in np.arange(min([j1+j2-j3, j1-m1, j2+m2])+1):
			if (j3-j2+t+m1>=0) and (j3-j1+t-m2>=0) and (j1+j2-j3-t>=0) and (j1-t-m1>=0) and (j2-t+m2>=0):

				# Only include term in summation if all factorials have non-negative arguments
				x = (np.math.factorial(t)
				  * np.math.factorial(j3-j2+t+m1)
				  * np.math.factorial(j3-j1+t-m2)
				  * np.math.factorial(j1+j2-j3-t)
				  * np.math.factorial(j1-t-m1)
				  * np.math.factorial(j2-t+m2)
				  )

				W3jS = W3jS + (-1)**t/x


		# Coefficients outside the summation:
		W3jS = (W3jS
			 * (-1)**(j1-j2-m3)
			 * np.sqrt(float(np.math.factorial(j1+m1)*np.math.factorial(j1-m1)*np.math.factorial(j2+m2)*np.math.factorial(j2-m2)* np.math.factorial(j3+m3)*np.math.factorial(j3-m3)))
			 * np.sqrt(float(np.math.factorial(j1 + j2 - j3)*np.math.factorial(j1 - j2 + j3)*np.math.factorial(-j1 + j2 + j3) / np.math.factorial(j1 + j2 + j3 + 1)))
			 )

	else:
		W3jS = 0 # One of the 'Selection Rules' was nullified.

	return W3jS


@jit
def get_rotation_matrix(a,b,c,Order):
    """
    [R, Q] = GetRotationMatrix(a,b,c,Order)
    
    Computes a rotation matrix R between the coefficients of Spherical
    Harmonic sound field descriptions before and after rotation. Note that R
    is block diagonal, since rotations only involve coefficients within an
    order, so it's returned as a sparse matrix. R is square & size (Order+1)^2.
    
    Essentially this is equations 3.3.37 and 3.3.39 of:
    Gumerov, N., & Duraiswami, R. (2005). Fast Multipole Methods for the
    Helmholtz Equation in Three Dimensions (1st ed.). Elsevier Science.
    
    The rotation is actually comprised of three rotations, as detailed on
    page 121 and Eq. 3.3.12:
    1) Rotation by a radians around z axis of the original coordinate system*
    2) Rotation by b radians around the y axis of the transitionary coordinate system
    3) Rotation by c radians around the z axis of the new coordinate system
    
    * note that the formulation there actually rotates by pi-a; this script
    makes that substitution so that a = 0 means no rotation (rather more intuitive!)
    
    Optionally also returns a 3 x 3 matrix Q, which is the tensor product
    between the original and transformed coordinate system unit vectors.
    
    Arguments:
    a     First rotation angle in radians (real scalar)
    b     Second rotation angle in radians (real scalar)
    c     Third rotation angle in radians (real scalar)
    Order Spherical Harmonic Order (non-negative real integer scalar)
    """
    
    # Argument checking:
    ### droped
    
    # Allocate R:
    R = np.zeros(((Order+1)**2, (Order+1)**2), dtype = np.complex128)
    
    # Loop over SH order:
    for n in np.arange(Order + 1, dtype=float):
        for m1 in np.arange(-n, n + 1, dtype=float):
            for m2 in np.arange(-n, n + 1, dtype=float):
                
                # Evalute Eq. 3.3.39:
                if m1 > 0:
                    ep1 = (-1)**m1
                else:
                    ep1 = 1
                
                if m2 > 0:
                    ep2 = (-1)**m2
                else:
                    ep2 = 1
                
                H = 0
                for s in np.arange(max(0, -(m1+m2)), min(n-m1,n-m2) + 1):                
                    H = H + (-1)**(n-s) * np.cos(b/2)**(2*s+m2+m1) * np.sin(b/2)**(2*n-2*s-m2-m1) / (np.math.factorial(s) * np.math.factorial(n-m1-s) * np.math.factorial(n-m2-s) * np.math.factorial(m1+m2+s))
                    #print(H)
                    
                H = H * ep1 * ep2 * np.sqrt(float(np.math.factorial(n+m2)*np.math.factorial(n-m2)*np.math.factorial(n+m1)*np.math.factorial(n-m1)))
                #print(H)
                # Evaluate Eq. 3.3.37:
                R[sub2indSH(m2,n).astype(int), sub2indSH(m1,n).astype(int)] = (-1)**m1 * np.exp(-1j*m1*a) * np.exp(-1j*m2*c) * H
                #print((-1)**m1 * np.exp(-1j*m1*a) * np.exp(-1j*m2*c) * H)
    R = scipy.sparse.csr_matrix(R)
    
    # # Compute Q if required, using Eq. 3.3.12:

    # Q1 = np.array([[np.sin(a), np.cos(a), 0], [-np.cos(a), np.sin(a), 0], [0, 0, 1]])
    # Q2 = np.array([[-1, 0, 0], [0, -np.cos(b), np.sin(b)], [0, np.sin(b), np.cos(b)]])
    # Q3 = np.array([[np.sin(c), np.cos(c), 0], [-np.cos(c), np.sin(c), 0], [0, 0, 1]])
    # Q = Q3 * Q2 * Q1;
    
    return R


@jit
def spherical_harmonic(n, m, alpha, sinbeta, cosbeta):
	
    """
    (Y, dY_dbeta, dY_dalpha) = SphericalHarmonic(n, m, alpha, sinbeta, cosbeta)
    	
    Computes a Spherical Harmonic function of order (m,n) and it's angular derivatives.
    	
    Arguments - these should all be scalars:
    r is radius
    alpha is azimuth angle (angle in radians from the positive x axis, with
    rotation around the positive z axis according to the right-hand screw rule)
    beta is polar angle, but it is specified as two arrays of its cos and sin values. 
    m and n should be integer scalars; n should be non-negative and m should be in the range -n<=m<=n
    	
    Returned data will be vectors of length (Order+1)^2.
    
    Associated Legendre function its derivatives for |m|:
    
    """
    
    p_nm = lpmv(abs(m), n, cosbeta)
    if n == 0:
        dPmn_dbeta = 0
    elif m == 0:
        dPmn_dbeta = lpmv(1, n, cosbeta)
    elif abs(m) < n:
        dPmn_dbeta = 0.5 * lpmv(abs(m) + 1, n, cosbeta) - 0.5 * (n + abs(m)) * (n - abs(m) + 1) * lpmv(abs(m) - 1, n, cosbeta);
    elif (abs(m) == 1) and (n == 1):
        dPmn_dbeta = -cosbeta
    elif sinbeta<=np.finfo(float).eps:
        dPmn_dbeta = 0
    else:
        dPmn_dbeta = -abs(m) * cosbeta * p_nm / sinbeta - (n + abs(m)) * (n - abs(m) + 1) * lpmv(abs(m) - 1, n, cosbeta)

    # Compute scaling term, including sign factor:
    scaling_term = ((-1) ** m) * np.sqrt((2 * n + 1) / (4 * np.pi * np.prod(np.float64(range(n - abs(m) + 1, n + abs(m) + 1)))))

    # Compute exponential term:
    exp_term = np.exp(1j * m * alpha)

    # Put it all together:
    y = scaling_term * exp_term * p_nm
    dy_dbeta = scaling_term * exp_term * dPmn_dbeta
    dy_dalpha = y * 1j * m

    return (y, dy_dbeta, dy_dalpha)


@jit
def spherical_harmonic_all (max_order, alpha, sinbeta, cosbeta):
    
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
    
    # alpha = np.array([[alpha]])
    # cosbeta = np.array([[cosbeta]])
    # sinbeta = np.array([[sinbeta]])

    # Preallocate output arrays:
    y =  np.zeros((np.size(alpha),(max_order+1)**2), np.complex128)
    dy_dbeta = np.zeros((np.size(alpha),(max_order+1)**2), np.complex128)
    dy_dalpha = np.zeros((np.size(alpha),(max_order+1)**2), np.complex128)
    
    #% Loop over n and calculate spherical harmonic functions y_nm
    for n in range(max_order+1):
    
        # Compute Legendre function and its derivatives for all m:
        p_n = lpmv(range(0,n+1), n, cosbeta)
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


@jit
def spherical_hankel_out (n, z):
    '''
    (h, dhdz) = SphericalHankelOut(n, z)
    
    Computes a spherical Hankel function of the first kind (outgoing in this
    paper's lingo) and its first derivative.
    '''
    
    h = spherical_jn(n,z,False) + 1j*spherical_yn(n,z,False)
    dhdz = spherical_jn(n,z,True) + 1j*spherical_yn(n,z,True)
    return h, dhdz


@jit
def spherical_basis_out_all(k, Bnm, pos, nUV):
    '''
    (phi, dphi_dn) = SphericalBasisOutAll(k, Bnm, x, nUV)
   	
    Returns phi and dPhi/dn for a summation of outgoing Spherical Basis functions with
   	coefficients Bnm. The algorithm is equivalent to that implemented in SphericalBasisOut,
   	but this version avoids repeated call to lpmv, since that is very time consuming.
   	
   	Arguments:
   	k     wavenumber - positive real scalar
   	Bnm   directivity coefficients - vector with a square number of elements
   	x     evaluation positions - real-valued array with 3 columns
   	nUV   unit vector defining direction in which to compute dPhi/dn - 1x3
   	
    Returned quantities are vectors with the same number of elements as x has rows.
    '''
    
    # Convert cartesison coordinates x to spherical coordinates:
    x = pos[0].reshape((1,1))
    y = pos[1].reshape((1,1))
    z = pos[2].reshape((1,1))
    (r, alpha, sinbeta, cosbeta) = cart2sph(x,y,z)

    # dot products of nUV with unit vectors of spherical coordinate system (at x):
    nUVrUV, nUValphaUV, nUVbetaUV = cart2sphUV(x,y,z,nUV)
	
    # Evaluate spherical harmonic functions and their derivatives:
    #if (Bnm.ndim != 1):
    #    raise IndexError('Bnm must be 1-dimensional')
    
    Order = int(np.sqrt(Bnm.size)) - 1
    y, dy_dbeta, dy_dalpha = spherical_harmonic_all(Order, alpha, sinbeta, cosbeta)

    # Loop over m and n and evalute phi and dPhi/dn:
    phi = np.zeros((r.size,1), np.complex64)
    dphi_dn = np.zeros((r.size,1), np.complex64)
    for n in range(Order + 1):
        R, dR_dkr = spherical_hankel_out(n, k*r)
        for m in range(-n, n + 1):
            i = sub2indSH(m,n)
            phi += Bnm[i,0] * R * y[0,i]
            dphi_dn += Bnm[i,0] * (nUVrUV * k * dR_dkr * y[0,i] + (R / r) * (nUVbetaUV * dy_dbeta[0,i] + nUValphaUV * dy_dalpha[0,i] / sinbeta))

    return (phi, dphi_dn)


@jit
def spherical_basis_out_p0_only(k, Bnm, pos):
    '''
    phi = SphericalBasisOutAll(k, Bnm, x)
   	
    Returns phi and dPhi/dn for a summation of outgoing Spherical Basis functions with
   	coefficients Bnm. The algorithm is equivalent to that implemented in SphericalBasisOut,
   	but this version avoids repeated call to lpmv, since that is very time consuming.
   	
   	Arguments:
   	k     wavenumber - positive real scalar
   	Bnm   directivity coefficients - vector with a square number of elements
   	x     evaluation positions - real-valued array with 3 columns
   	nUV   unit vector defining direction in which to compute dPhi/dn - 1x3
   	
    Returned quantities are vectors with the same number of elements as x has rows.
    '''
    
    # Convert cartesison coordinates x to spherical coordinates:
    x = pos[0].reshape((1,1))
    y = pos[1].reshape((1,1))
    z = pos[2].reshape((1,1))
    (r, alpha, sinbeta, cosbeta) = cart2sph(x,y,z)
	
    # Evaluate spherical harmonic functions and their derivatives:
    #if (Bnm.ndim != 1):
    #    raise IndexError('Bnm must be 1-dimensional')
    
    Order = int(np.sqrt(Bnm.size)) - 1
    y, dy_dbeta, dy_dalpha = spherical_harmonic_all(Order, alpha, sinbeta, cosbeta)

    # Loop over m and n and evalute phi and dPhi/dn:
    phi = np.zeros((r.size,1), np.complex128)
    for n in range(Order + 1):
        R, dR_dkr = spherical_hankel_out(n, k*r)
        for m in range(-n, n + 1):
            i = sub2indSH(m,n)
            phi += Bnm[i,0] * R * y[0,i]

    return phi


@jit
def spherical_hankel_in_p0_only(n, z):
    '''	
    h = SphericalHankelIn_pOnly(n, z)
    
    Computes a spherical Hankel function of the second kind 
    (incoming in this paper's lingo).
    Identical to SphericalHankelIn but does not compute the derivative.
    '''

    h = spherical_jn(n,z,False) - 1j*spherical_yn(n,z,False)
    return h


@jit
def spherical_hankel_in(n, z):
    '''
    (h, dhdz) = SphericalHankelIn(n, z)
    	
    Computes a spherical Hankel function of the second kind (incoming in this
    paper's lingo) and its first derivative.
    '''
  
    h = spherical_jn(n,z,False) - 1j*spherical_yn(n,z,False)
    dhdz = spherical_jn(n,z,True) - 1j*spherical_yn(n,z,True)
    return (h, dhdz)


@jit
def spherical_harmonic_p0_only(n, m, alpha, sinbeta, cosbeta):
    '''
    Y = SphericalHarmonic_pOnly(n, m, alpha, sinbeta, cosbeta)
    	
    Computes a Spherical Harmonic function of order (m,n).
    Identical to SphericalHarmonic but does not compute the derivatives.
    	
    Arguments - these should all be scalars:
    r is radius
    alpha is azimuth angle (angle in radians from the positive x axis, with
    rotation around the positive z axis according to the right-hand screw rule)
    beta is polar angle, but it is specified as two arrays of its cos and sin values. 
    m and n should be integer scalars; n should be non-negative and m should be in the range -n<=m<=n
    	
    Returned data will be a vector of length (Order+1)^2.
    
    Associated Legendre function its derivatives for |m|:
    '''
    
    p_nm = lpmv(abs(m), n, cosbeta)

    # Compute scaling term, including sign factor:
    scaling_term = ((-1)**m) * np.sqrt((2 * n + 1) / (4 * np.pi * np.prod(np.float64(range(n-abs(m)+1, n+abs(m)+1)))))

    # Compute exponential term:
    exp_term = np.exp(1j*m*alpha)
    
    # Put it all together:
    y = scaling_term * exp_term * p_nm

    return y


@jit
def spherical_basis_in_p0_only(n, m, k, pos):
    '''
	Phi = SphericalBasisIn_pOnly(n, m, k, x)
	
	Returns Phi for an incoming Spherical Basis function of order (m,n).
	Identical to SphericalBasisIn but does not compute the derivatives.
	
	Arguments:
	k     wavenumber - positive real scalar
	n     must be a non-negative real integer scalar
	m     m must be a real integer scalar in the range -n to +n
	x     evaluation positions - real-valued array with 3 columns
	
	Returned quantity is a vector with the same number of elements as x has rows.
    '''

    # Convert cartesison coordinates x to spherical coordinates:
    x = pos[0].reshape((1,1))
    y = pos[1].reshape((1,1))
    z = pos[2].reshape((1,1))
    (r, alpha, sinbeta, cosbeta) = cart2sph(x,y,z)
    	
    # Evaluate spherical harmonic and Hankel functions and their derivatives:
    Y = spherical_harmonic_p0_only(n, m, alpha, sinbeta, cosbeta)
    R = spherical_hankel_in_p0_only(n, k*r)
    
    # Evaluate Phi:
    Phi = R * Y
    	
    return Phi


@jit
def spherical_basis_in(n, m, k, pos, nUV):
    '''
   	(Phi, dPhi_dn) = SphericalBasisIn(n, m, k, x, nUV)
   	
   	Returns Phi and dPhi/dn for an incoming Spherical Basis function of order (m,n).
   	
   	Arguments:
   	k     wavenumber - positive real scalar
   	n     must be a non-negative real integer scalar
   	m     m must be a real integer scalar in the range -n to +n
   	x     evaluation positions - real-valued array with 3 columns
   	nUV   unit vector defining direction in which to compute dPhi/dn - 1x3
   	
   	Returned quantities are vectors with the same number of elements as x has rows.
    '''
    
    # Convert cartesison coordinates x to spherical coordinates:
    x = pos[0].reshape((1,1))
    y = pos[1].reshape((1,1))
    z = pos[2].reshape((1,1))
    (r, alpha, sinbeta, cosbeta) = cart2sph(x,y,z)
    
    # dot products of nUV with unit vectors of spherical coordinate system (at x):
    nUVrUV, nUValphaUV, nUVbetaUV = cart2sphUV(x,y,z,nUV)
    	
    # Evaluate spherical harmonic and Hankel functions and their derivatives:
    Y, dY_dbeta, dY_dalpha = spherical_harmonic(n, m, alpha, sinbeta, cosbeta)
    R, dR_dkr = spherical_hankel_in(n, k*r)
    
    	# Evaluate Phi and dPhi/dn:
    Phi = R * Y
    dPhi_dn = (nUVrUV * k * dR_dkr * Y + (R / r) * (nUVbetaUV * dY_dbeta + nUValphaUV * dY_dalpha / sinbeta))
    	
    return (Phi, dPhi_dn)


@jit
def cart2sphUV(x,y,z,nUV):
    '''
    [nUVrUV, nUValphaUV, nUVbetaUV] = Cart2SphUV(x,y,z,nUV)
	
    Returns the dot product between a given unit vector nUV and the unit 
    vectors of the spherical polar coordinate system (r, alpha, beta), 
    evaluated at Cartesian coordinates (x,y,z). Here alpha is
    azimuth angle (angle in radians from the positive x axis, with rotation
    around the positive z axis according to the right-hand screw rule) and
    beta is polar angle (angle in radians from the positive z axis).
    
    It is assumed that x, y and z are all the same size.
    The returned arrays will be the same size as the arguments.
    '''
    
    r = np.sqrt(x**2 + y**2 + z**2)
    rho = np.sqrt(x**2 + y**2)
    
    nUVrUV = nUV[0] * x / r + nUV[1] * y / r + nUV[2] * z / r
    nUValphaUV = - nUV[0] * y / rho + nUV[1] * x / rho 
    nUVbetaUV = nUV[0] * x * z / (rho * r) + nUV[1] * z * y / (rho * r) - nUV[2] * rho / r
	
    return nUVrUV, nUValphaUV, nUVbetaUV
