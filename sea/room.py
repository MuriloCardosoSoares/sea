import time
import warnings
import bempp.api
import numpy as np
import numba

from bemder import helpers
from matplotlib import pylab as plt
import cloudpickle
import collections
bempp.api.PLOT_BACKEND = "gmsh"

warnings.filterwarnings('ignore')

from sea.definitions import Air
from sea.definitions import Algorithm
from sea.definitions import Receiver
from sea.definitions import Source
from sea.materials import Material


class Room:   

    bempp.api.DEVICE_PRECISION_CPU = 'single'
    
    def __init__(self, air=Air(), assembler = 'numba', IS=0):
        '''
        Room object.
        This class comunicates to the other classes of this repository. 
        All information about the simulation will be set up in here.
        '''
        self.air = air
        self.IS = IS
        self.assembler = assembler
        self.EoI = 1
        self.v = 0 
        

    def algorithm_control(self, freq_init=20.0, freq_end=200.0, freq_step=1, freq_vec=[]):
        self.frequencies = Algorithm(freq_init, freq_end, freq_step, freq_vec) 
        
    
    def add_receiver(self, coord = [1.0, 0.0, 0.0]):
        self.receivers.append(Receiver(coord))
        
        
    def list_receivers(self):
        print("Receivers coordinates are:")
        for receiver in self.receivers:
            print (receiver.coord)
            
            
    def add_source(self, coord = [0.0, 0.0, 1.0], q = [1.0], source_type="monopole"):
        self.sources.append(Source(coord, q , source_type))  
        
        
    def list_sources(self):
        print("Sources coordinates are:")
        for source in self.receivers:
            print (source.coord)
     
    
    def add_mesh(path_to_msh, show_mesh=False, gmsh_filepath=None, reorder_domain_index=True):
        """
        This function imports a .msh file and orders the domain_index from 0 to len(domain_index).
        Parameters
        ----------
        path : String
            Path to .msh file.
        Returns
        -------
        Bempp Grid.
        """
        
        self.path_to_msh = path_to_msh
        
        try:  
            import gmsh
        except :
            import gmsh_api.gmsh as gmsh

        import sys
        import os

        if reorder_domain_index:
            
            gmsh.initialize(sys.argv)
            gmsh.open(path_to_msh) # Open msh
            phgr = gmsh.model.getPhysicalGroups(2)
            odph = []
            
            for i in range(len(phgr)):
                odph.append(phgr[i][1]) 
            phgr_ordered = [i for i in range(0, len(phgr))]
            phgr_ent = []
            
            for i in range(len(phgr)):
                phgr_ent.append(gmsh.model.getEntitiesForPhysicalGroup(phgr[i][0],phgr[i][1]))
                
            gmsh.model.removePhysicalGroups()
            
            for i in range(len(phgr)):
                gmsh.model.addPhysicalGroup(2, phgr_ent[i],phgr_ordered[i])

            # gmsh.fltk.run()   
            path_name = os.path.dirname(path_to_msh)
            gmsh.write(path_name+'/current_mesh.msh')   
            gmsh.finalize()    
            
            if show_mesh == True:
                
                try:
                    bempp.api.PLOT_BACKEND = "jupyter_notebook"
                    bempp.api.import_grid(path_name+'/current_mesh.msh').plot()
                    
                except:
                    bempp.api.GMSH_PATH = gmsh_filepath
                    bempp.api.PLOT_BACKEND = "gmsh"
                    bempp.api.import_grid(path_name+'/current_mesh.msh').plot()



            self.msh = bempp.api.import_grid(path_name+'/current_mesh.msh')
            os.remove(path_name+'/current_mesh.msh')
            
        else:
            self.msh = bempp.api.import_grid(path_to_msh)
     

    def add_material(self, parameters=[15000, 0.05], type="porous"):
        if type == "porous"
        self.materials.append()

        
        
    def run(self,device="cpu",individual_sources=False):
        """
        Computes the bempp gridFunctions for the interior acoustic problem.
        
        Outputs: 
            
            boundP = grid_function for boundary pressure
            
            boundU = grid_function for boundary velocity
        
        """
        if self.assembler == 'opencl':
            if device == "cpu":     
                helpers.set_cpu()
            if device == "gpu":
                helpers.set_cpu()
        self.bD={}
        if self.mu == None:
            self.mu = {}
            for i in (np.unique(self.grid.domain_indices)):
                self.mu[i] = np.zeros_like(self.f_range)

        if self.v == None:
            self.v = {}
            for i in (np.unique(self.grid.domain_indices)):
                self.v[i] = np.zeros_like(self.f_range)        
        # mu_fi = np.array([self.mu[i] for i in self.mu.keys()])
        self.space = bempp.api.function_space(self.grid, "P", 1)
        if individual_sources==True:
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
        
        elif individual_sources==False:
            self.boundData = {}  
           
            for fi in range(np.size(self.f_range)):
    
                mu_f = self.mu[fi]
                # print(mu_f[0])
                f = self.f_range[fi] #Convert index to frequency
                k = 2*np.pi*f/self.c0 # Calculate wave number
                
                @bempp.api.callable(complex=True,jit=True,parameterized=True)
                def mu_fun(x,n,domain_index,result,mu_f):
                        result[0]=np.conj(mu_f[domain_index])
                        # print(mu_f[domain_index],domain_index)
                    
                
                mu_op = bempp.api.MultiplicationOperator(
                    bempp.api.GridFunction(self.space,fun=mu_fun,function_parameters=mu_f)
                    ,self.space,self.space,self.space)
                
                
                # @bempp.api.callable(complex=True,jit=False)
                # def v_data(x, n, domain_index, result):
                #     with numba.objmode():
                #         result[0] = self.v[domain_index][fi]
            
                identity = bempp.api.operators.boundary.sparse.identity(
                    self.space, self.space, self.space)
                dlp = bempp.api.operators.boundary.helmholtz.double_layer(
                    self.space, self.space, self.space, k, assembler="dense", device_interface=self.assembler)
                slp = bempp.api.operators.boundary.helmholtz.single_layer(
                    self.space, self.space, self.space, k,assembler="dense", device_interface=self.assembler)
                
                # ni = (1j/k)
                

                

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
                        result[0]= -(1j*mu[domain_index]*k*val - val/(pos*pos) * (1j*k*pos-1)* np.dot(r-r0,n))
           
    
                # v_fun = bempp.api.GridFunction(self.space, fun=v_data)
                mnp_fun = bempp.api.GridFunction.from_zeros(self.space)
                function_parameters_mnp = np.zeros(5+len(mu_f),dtype = 'complex128')     
                for i in range(len(self.r0.T)):
                    function_parameters_mnp[:3] = self.r0[:,i]
                    function_parameters_mnp[3] = k
                    function_parameters_mnp[4] = self.q.flat[i]
                    function_parameters_mnp[5:] = mu_f
                    
                    mnp_fun += bempp.api.GridFunction(self.space,fun=combined_data,
                                                      function_parameters=function_parameters_mnp)
                a = 1j*k*self.c0*self.rho0
                monopole_fun = mnp_fun
                Y = a*(mu_op)# + monopole_fun
       
                lhs = (0.5*identity+dlp) - slp*Y
                rhs = -slp*monopole_fun #- slp*a*v_fun
            
                boundP, info = bempp.api.linalg.gmres(lhs, rhs, tol=1E-5)#, use_strong_form=True)
            
                boundU = Y*boundP - monopole_fun
                
                self.boundData[fi] = [boundP, boundU]
                # u[fi] = boundU
                
                self.IS = 0
                print('{} / {}'.format(fi+1,np.size(self.f_range)))
                
            return self.boundData
    

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
    

    
    def point_evaluate(self,boundD,R=R_init):
        
        """
        Evaluates the solution (pressure) for a point.
        
        Inputs:
            points = dict[0:numPoints] containing np arrays with receiver positions 
            
            boundData = output from bemsolve()
            
        Output:
            
           pT =  Total Pressure Field
           
           pS = Scattered Pressure Field
           
        """
        pT = {}
        pS = {}
        ppt={}
        pps={}
        pts = R.coord.reshape(len(R.coord),3)
        if self.IS==1:
            for fi in range(np.size(self.f_range)):
            
                f = self.f_range[fi] #Convert index to frequency
                k = 2*np.pi*f/self.c0
                
                
                

                dlp_pot = bempp.api.operators.potential.helmholtz.double_layer(
                    self.space, pts.T, k,assembler="dense", device_interface=self.assembler)
                slp_pot = bempp.api.operators.potential.helmholtz.single_layer(
                    self.space, pts.T, k,assembler="dense", device_interface=self.assembler)
                
                for i in range(len(self.r0.T)):
                    # self.ir0 = self.r0[:,i]
                    pScat =  -dlp_pot.evaluate(boundD[fi][i][0])+slp_pot.evaluate(boundD[fi][i][1])
                
                    if self.wavetype == "plane":
                        pInc = self.planewave(fi,pts,i)
                        
                    elif self.wavetype == "spherical":
                        pInc = self.monopole(fi,pts,i)
                    
                    pT[i] = pInc+pScat
                    pS[i] = pScat
        
                    # print(20*np.log10(np.abs(pT[fi])/2e-5))
                print('{} / {}'.format(fi+1,np.size(self.f_range)))
                ppt[fi] = np.array([pT[i] for i in pT.keys()]).reshape(len(pT),len(R.coord))
                pps[fi] = np.array([pS[i] for i in pS.keys()]).reshape(len(pS),len(R.coord))
            return ppt,pps
       
        else:
            
            for fi in range(np.size(self.f_range)):
                f = self.f_range[fi] #Convert index to frequency
                k = 2*np.pi*f/self.c0
                

                    
                dlp_pot = bempp.api.operators.potential.helmholtz.double_layer(
                    self.space, pts.T, k,assembler="dense", device_interface=self.assembler)
                slp_pot = bempp.api.operators.potential.helmholtz.single_layer(
                    self.space, pts.T, k,assembler="dense", device_interface=self.assembler)
                pScat =  -dlp_pot.evaluate(boundD[fi][0])+slp_pot.evaluate(boundD[fi][1])
                    
                if self.wavetype == "plane":
                    pInc = self.planewave(fi,pts,ir=0)
                    
                elif self.wavetype == "spherical":
                    pInc = self.monopole(fi,pts,ir=0)
                
                pT[fi] = pInc+(pScat)
                pS[fi] = pScat
    
                print(20*np.log10(np.abs(pT[fi])/2e-5))
                print('{} / {}'.format(fi+1,np.size(self.f_range)))
                
            return  np.array([pT[i] for i in pT.keys()]).reshape(len(pT),len(R.coord)),np.array([pS[i] for i in pS.keys()]).reshape(len(pS),len(R.coord))


    def bem_save(self, filename=time.strftime("%Y%m%d-%H%M%S"), ext = ".pickle"):
        # Simulation data
        gridpack = {'vertices': self.grid.vertices,
                'elements': self.grid.elements,
                'volumes': self.grid.volumes,
                'normals': self.grid.normals,
                'jacobians': self.grid.jacobians,
                'jacobian_inverse_transposed': self.grid.jacobian_inverse_transposed,
                'diameters': self.grid.diameters,
                'integration_elements': self.grid.integration_elements,
                'centroids': self.grid.centroids,
                'domain_indices': self.grid.domain_indices}

        
        bd = {}
        bbd= {}
        # incident_traces = []
        if self.IS == 1:
            
            for sol in range(len(self.f_range)):
                i=0
                bda = {}
                for i in range(len(self.r0.T)):
                    bda[i] = [(self.bD[sol][i][0].coefficients),(self.bD[sol][i][1].coefficients)]  
                bbd[sol] = bda
        elif self.IS==0:
            for sol in range(len(self.f_range)):
                bd[sol] = [(self.boundData[sol][0].coefficients),(self.boundData[sol][1].coefficients)]
                # incident_traces.append(self.simulation._incident_traces[0].coefficients)
            
        simulation_data = {'AC': self.AC,
                           "AP": self.AP,
                           'R': self.R,
                           'S': self.S,
                           'BC': self.BC,
                           'EoI': self.EoI,
                           'IS': self.IS,
                           'assembler': self.assembler,
                           'grid': gridpack,
                           'path_to_grid': self.path_to_geo,
                           'bd': bd,
                           'bbd':bbd}
                           # 'incident_traces': incident_traces}

                
        outfile = open(filename + ext, 'wb')
                
        cloudpickle.dump(simulation_data, outfile)
        outfile.close()
        print('BEM saved successfully.')
        
    def plot_room(self,S=None,R=None, opacity = 0.3, mode="element", transformation=None):
        import plotly.figure_factory as ff
        import plotly.graph_objs as go
        import numpy as np
        import plotly
        
        vertices = self.grid.vertices
        elements = self.grid.elements
        fig = ff.create_trisurf(
            x=vertices[0, :],
            y=vertices[1, :],
            z=vertices[2, :],
            simplices=elements.T,
            color_func=elements.shape[1] * ["rgb(255, 222, 173)"],
        )
        fig['data'][0].update(opacity=opacity)
        fig['layout']['scene'].update(go.layout.Scene(aspectmode='data'))
        
        if R != None:
            fig.add_trace(go.Scatter3d(x = R.coord[:,0], y = R.coord[:,1], z = R.coord[:,2],marker=dict(size=8, color='rgb(0, 0, 128)', symbol='circle'),name="Receivers"))
            
        if S != None:    
            if S.wavetype == "spherical":
                fig.add_trace(go.Scatter3d(x = S.coord[:,0], y = S.coord[:,1], z = S.coord[:,2],marker=dict(size=8, color='rgb(128, 0, 0)', symbol='square'),name="Sources"))

        # fig.add_trace(go.Mesh3d(x=[-6,6,-6,6], y=[-6,6,-6,6], z=0 * np.zeros_like([-6,6,-6,6]), color='red', opacity=0.5, showscale=False))

        return fig
