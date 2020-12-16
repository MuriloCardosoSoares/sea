"""
This module contains the functions needed to solve the acoustic field of a room using the Boundary Element Method
"""  

def impedance_bemsolve(self, device="cpu"):
    """
    Computes the bempp gridFunctions for the interior acoustic problem.

    Outputs: 

        boundP = grid_function for boundary pressure

        boundU = grid_function for boundary velocity

    """
    
    bempp.api.DEVICE_PRECISION_CPU = 'single'  
            
    self.bD={}
    if self.mu == None:
        self.mu = {}
        for i in (np.unique(self.grid.domain_indices)):
            self.mu[i] = np.zeros_like(self.f_range)

    if self.v == None:
        self.v = {}
        for i in (np.unique(self.grid.domain_indices)):
            self.v[i] = np.zeros_like(self.f_range)        

    self.space = bempp.api.function_space(self.grid, "P", 1)
    
    for fi in range(np.size(self.f_range)):
        mu_f = self.mu[fi]
        self.boundData = {}  
        f = self.f_range[fi] #Convert index to frequency
        k = 2*np.pi*f/self.c0 # Calculate wave number

        @bempp.api.callable(complex=True,jit=True,parameterized=True)
        def mu_fun(x,n,domain_index,result,mu_f):
                result[0]=np.conj(mu_f[domain_index])
                # print(mu_f[domain_index],domain_index)


        mu_op = bempp.api.MultiplicationOperator(
            bempp.api.GridFunction(self.space,fun=mu_fun,function_parameters=mu_f)
            ,self.space,self.space,self.space)

        # @bempp.api.real_callable
        # def v_data(x, n, domain_index, result):
        #     with numba.objmode():
        #         result[0] = self.v[domain_index][fi]

        identity = bempp.api.operators.boundary.sparse.identity(
            self.space, self.space, self.space)
        dlp = bempp.api.operators.boundary.helmholtz.double_layer(
            self.space, self.space, self.space, k,assembler="dense", device_interface=self.assembler)
        slp = bempp.api.operators.boundary.helmholtz.single_layer(
            self.space, self.space, self.space, k,assembler="dense", device_interface=self.assembler)

        a = 1j*k*self.c0*self.rho0
        icc=0

        for icc in range(len(self.r0.T)):
            # self.ir0 = self.r0[:,i].T

            if self.wavetype == "plane":    
                @bempp.api.callable(complex=True,jit=True,parameterized=True)
                def combined_data(r, n, domain_index, result,parameters):

                    r0 = np.real(parameters[:3])
                    k = parameters[3]
                    q = parameters[4]

                    ap = np.linalg.norm(r0) 
                    pos = (r[0]*r0[0]+r[1]*r0[1]+r[2]*r0[2])
                    # nm = (((n[0]-1)*r0[0,i]/ap)+((n[1]-1)*r0[1,i]/ap)+((n[2]-1)*r0[2,i]/ap))
                    result[0] = q*(np.exp(1j * k * pos/ap))
            elif self.wavetype == "spherical":    
                @bempp.api.callable(complex=True,jit=True,parameterized=True)
                def combined_data(r, n, domain_index, result,parameters):

                    r0 = np.real(parameters[:3])
                    k = parameters[3]
                    q = parameters[4]
                    mu = parameters[5:]


                    pos  = np.linalg.norm(r-r0)

                    val  = q*np.exp(1j*k*pos)/(4*np.pi*pos)
                    result[0]=-(1j * mu[domain_index] * k * val -
                        val / (pos**2) * (1j * k * pos - 1) * np.dot(r-r0, n))

            else:
                raise TypeError("Wavetype must be plane or spherical") 

            mnp_fun = bempp.api.GridFunction.from_zeros(self.space)
            function_parameters_mnp = np.zeros(5+len(mu_f),dtype = 'complex128')     

            function_parameters_mnp[:3] = self.r0[:,icc]
            function_parameters_mnp[3] = k
            function_parameters_mnp[4] = self.q.flat[icc]
            function_parameters_mnp[5:] = mu_f

            mnp_fun = bempp.api.GridFunction(self.space,fun=combined_data,
                                              function_parameters=function_parameters_mnp)    
            # v_fun = bempp.api.GridFunction(self.space, fun=v_data)

            monopole_fun = mnp_fun

            Y = a*(mu_op)# + monopole_fun

            lhs = (0.5*identity+dlp) - slp*Y
            rhs = monopole_fun #- slp*a*v_fun

            boundP, info = bempp.api.linalg.gmres(lhs, rhs, tol=1E-5)#, use_strong_form=True)

            boundU = Y*boundP - monopole_fun

            self.boundData[icc] = [boundP, boundU]
            # u[fi] = boundU

            self.IS = 1

            print('{} Hz - Source {}/{}'.format(self.f_range[fi],icc+1,len(self.r0.T)))


        print('{} / {}'.format(fi+1,np.size(self.f_range)))

        self.bD[fi] = self.boundData 

    return self.bD

  


def monopole(self,fi,pts,ir):

    pInc = np.zeros(pts.shape[0], dtype='complex128')
    if self.IS==1:
        pos = np.linalg.norm(pts-self.r0[:,ir])
        pInc = self.q.flat[ir]*np.exp(1j*(2*np.pi*self.f_range[fi]/self.c0)*pos)/(4*np.pi*pos)
    else:
        for i in range(len(self.r0.T)): 
            pos = np.linalg.norm(pts-self.r0[:,i].reshape(1,3),axis=1)
            pInc += self.q.flat[i]*np.exp(1j*(2*np.pi*self.f_range[fi]/self.c0)*pos)/(4*np.pi*pos)

    return pInc
