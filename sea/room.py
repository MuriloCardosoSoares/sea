import time
import warnings
import bempp.api
import numpy as np
import sys
import os
import gc

from matplotlib import pylab as plt
import matplotlib as mpl
import cloudpickle
import collections
bempp.api.PLOT_BACKEND = "gmsh"
import gmsh_api.gmsh as gmsh
#import gmsh.api.gmsh as gmsh

import plotly

from google.colab import files
import shutil 
import pickle

warnings.filterwarnings('ignore')

from sea.definitions import Air
from sea.definitions import Algorithm
from sea.definitions import Receiver
from sea.definitions import Source
from sea.materials import Material
import sea.spherical_harmonics as sh


class Room:   
    
    def __init__(self, air=Air(), **kwargs):
        '''
        Room object.
        This class comunicates to the other classes of this repository. 
        All information about the simulation will be set up in here.
        '''
        self.air = air
        try:
            self.room_name = kwargs["room_name"]
        except:
            self.room_name = "my_room"
        
        self.receivers = []
        self.sources = []
        self.materials = []
        
        self.boundary_pressure = [] 
        self.boundary_velocity = [] 
        
        self.scattered_pressure = []
        self.incident_pressure = []
        self.total_pressure = []
        
        self.simulated_freqs = []
        
        
    def air_properties(self, c0 = 343.0, rho0 = 1.21, temperature = 20.0, humid = 50.0, p_atm = 101325.0):
        self.air = Air(c0 = 343.0, rho0 = 1.21, temperature = 20.0, humid = 50.0, p_atm = 101325.0)
        if hasattr(self, "frequencies"):
            self.air.k0 = 2*np.pi*self.frequencies.freq_vec/self.air.c0
        
    
    def algorithm_control(self, freq_init=20.0, freq_end=200.0, freq_step=1, freq_vec=[]):
        self.frequencies = Algorithm(freq_init, freq_end, freq_step, freq_vec)
        self.air.k0 = 2*np.pi*self.frequencies.freq_vec/self.air.c0
        
    
    def add_receiver(self, coord = [1.0, 0.0, 0.0], type="omni", **kwargs):
        self.receivers.append(Receiver(coord, type, **kwargs))
    
    
    def del_receivers(self, positions=[]):
    
        positions = np.array(positions)
        if positions.size != 0:
            for i in sorted(positions, reverse=True):
                del self.receivers[i]
        else:
            self.receivers.clear()
        
        
    def list_receivers(self):
        print("Receivers are:")
        for receiver in self.receivers:
            print (receiver)
            
            
    def add_source(self, coord=[0.0, 0.0, 1.0], type="monopole", **kwargs):
        self.sources.append(Source(self.frequencies.freq_vec, coord, type, rho0 = self.air.rho0, c0 = self.air.c0, **kwargs))  

        
    def del_sources(self, positions=[]):
        
        positions = np.array(positions)
        if positions.size != 0:
            for i in sorted(positions, reverse=True):
                del self.sources[i]
            
        else:
            self.sources.clear()
     
    
    def list_sources(self):
        print("Sources are:")
        for source in self.sources:
            print (source)
     
    
    def add_mesh(self):
        """
        Imports a .msh file.
        This method is deprecated. Istead, consider using .add_geometry() method.
        """
        from google.colab import files
        uploaded = files.upload()
        
        for key in uploaded:
            
            path_to_msh = key
            
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
            #path_name = os.path.dirname(path_to_msh)
            #gmsh.write(path_name+'/current_mesh.msh')
            gmsh.write(path_to_msh)
            gmsh.finalize() 
            
            self.path_to_msh = key
     
    
    def generate_mesh(self, c0, freq, factor):
        """
        This function generates a .msh file from the .geo file uploaded.
        """
        
        import meshio
        
        
        gmsh.initialize(sys.argv)
        try:
            gmsh.open(self.path_to_geo) # Open .geo file
        except:
            print("Geometry was not find. Please, upload a .geo file:")
            self.add_geometry()           
            gmsh.open(self.path_to_geo) # Open .geo file
        
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", (c0/freq)/factor)
        
        #gmsh.option.setNumber("Mesh.MeshSizeMax", (c0/freq)/6)
        #gmsh.option.setNumber("Mesh.MeshSizeMin", 0)
        #gmsh.model.occ.synchronize()
        
        gmsh.model.mesh.generate(2)
        #gmsh.model.mesh.setOrder(1)
        
        gmsh.write("last_msh.msh")
        gmsh.finalize()

        #max_element_size = (c0/freq)/6
        #os.system("gmsh -clmax $max_element_size -2 $self.path_to_geo -o last_msh.msh")
        
        #import subprocess 

        #subprocess.run(["gmsh", "-clmax", "$max_element_size", "-2", "$self.path_to_geo", "-o", "last_msh.msh"])
                
        '''
        #Reorder physical groups       
        gmsh.initialize(sys.argv)
        gmsh.open("last_msh.msh")
        
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

            
        gmsh.write("last_msh.msh")
        gmsh.finalize() 
        '''
        self.path_to_msh = "last_msh.msh"
        
    
    def add_geometry(self):
        """
        This function imports a .geo file.
        """
        from google.colab import files
        uploaded = files.upload()
        
        for key in uploaded:    
            self.path_to_geo = key
    

    def add_material(self, normal_incidence_alpha=[], statistical_alpha=[], octave_bands_statistical_alpha=[], octave_bands=[],
                     third_octave_bands_statistical_alpha=[], third_octave_bands=[], admittance=[], 
                     normalized_surface_impedance=[], surface_impedance=[], rmk1=[],**kwargs):
        
        if "parameters" in kwargs and "absorber_type" in kwargs:
            
            if hasattr(self, "frequencies") != True:
                raise ValueError("Algorithm frequencies are not defined yet.")
                
            material = Material(freq_vec=self.frequencies.freq_vec, rho0=self.air.rho0, c0=self.air.c0)

            if kwargs["absorber_type"] == "porous":
                material.porous(kwargs["parameters"])
                self.materials.append(material)

            if kwargs["absorber_type"] == "porous with air cavity":
                material.porous_with_air_cavity(kwargs["parameters"])
                self.materials.append(material)
                
            if kwargs["absorber_type"] == "membrane":
                material.membrane(kwargs["parameters"])
                self.materials.append(material)
                
            if kwargs["absorber_type"] == "perforated panel":
                material.perforated_panel(kwargs["parameters"])
                self.materials.append(material)
                
            if kwargs["absorber_type"] == "microperforated panel":
                material.microperforated_panel(kwargs["parameters"])
                self.materials.append(material)

        else:
            material = Material(normal_incidence_alpha=normal_incidence_alpha, statistical_alpha=statistical_alpha, octave_bands_statistical_alpha=octave_bands_statistical_alpha, 
                                octave_bands=octave_bands, third_octave_bands_statistical_alpha=third_octave_bands_statistical_alpha, 
                                third_octave_bands=third_octave_bands, admittance=admittance, normalized_surface_impedance=normalized_surface_impedance, 
                                surface_impedance=surface_impedance, freq_vec=self.frequencies.freq_vec, rmk1=rmk1, rho0=self.air.rho0, c0=self.air.c0)

            if "absorber_type" in kwargs:
                if kwargs["absorber_type"] == "rigid":
                    material.rigid()
                    self.materials.append(material)
                
                elif material.admittance.size == 0:
                    material.impedance_from_alpha(absorber_type=kwargs["absorber_type"])
                    self.materials.append(material)
                    
            else:
                self.materials.append(material)         
           
            
    def del_materials(self, positions=[]):
    
        positions = np.array(positions)
        if positions.size != 0:
            for i in sorted(positions, reverse=True):
                del self.materials[i]
        else:
            self.materials.clear()
            
            
    def list_materials(self):
        for material in self.materials:
            print(material)          

            
    def view(self, freqs=[], opacity=0.2):
        
        from matplotlib import style
        style.use("seaborn-talk")
        
        import plotly.figure_factory as ff
        import plotly.graph_objs as go
        from bempp.api import GridFunction
        from bempp.api.grid.grid import Grid

        freqs = np.array(freqs)
        if freqs.size == 0:
            try:
                freqs = np.array([self.frequencies.freq_vec[0]])
            except:
                freqs = np.array([20])
        
        for f in freqs:
            '''
            try:
                msh_path = "meshs/msh_%s_%sHz.msh" %(self.room_name, f)
                reorder_physical_groups(msh_path)
                grid = bempp.api.import_grid(msh_path)
            except:
                raise ValueError("Mesh file for %s Hz was not found." % f)
            '''
            
            
            self.generate_mesh(self.air.c0, f, 6)

            grid = bempp.api.import_grid(self.path_to_msh)
            
            def configure_plotly_browser_state():
                import IPython
                display(IPython.core.display.HTML('''
                        <script src="/static/components/requirejs/require.js"></script>
                        <script>
                          requirejs.config({
                            paths: {
                              base: '/static/base',
                              plotly: 'https://cdn.plot.ly/plotly-1.5.1.min.js?noext',
                            },
                          });
                        </script>
                        '''))

            plotly.offline.init_notebook_mode()

            vertices = grid.vertices
            elements = grid.elements
            fig = ff.create_trisurf(
                x=vertices[0, :],
                y=vertices[1, :],
                z=vertices[2, :],
                simplices=elements.T,
                color_func=elements.shape[1] * ["rgb(255, 222, 173)"],
            )
            fig['data'][0].update(opacity=opacity)
            fig['layout']['scene'].update(go.layout.Scene(aspectmode='data'))

            if hasattr(self, "receivers"):
                x = []; y = []; z = []
                for receiver in self.receivers:
                    x = np.append(x, receiver.coord[0,0]); y = np.append(y, receiver.coord[0,1]); z = np.append(z, receiver.coord[0,2])
                fig.add_trace(go.Scatter3d(x=x, y=y, z=z, marker=dict(size=8, color='rgb(0, 0, 128)', symbol='circle'),name="Receivers"))

            if hasattr(self, "sources"):
                x = []; y = []; z = []
                for source in self.sources:
                    x = np.append(x, source.coord[0,0]); y = np.append(y, source.coord[0,1]); z = np.append(z, source.coord[0,2])
                fig.add_trace(go.Scatter3d(x=x, y=y, z=z, marker=dict(size=8, color='rgb(128, 0, 0)', symbol='square'),name="Sources"))


            fig.add_trace(go.Mesh3d(x=[-6,6,-6,6], y=[-6,6,-6,6], z=0 * np.zeros_like([-6,6,-6,6]), color='red', opacity=0.5, showscale=False))

            configure_plotly_browser_state() 
            plotly.offline.iplot(fig)
        
        
    def run(self, save=True):
  
        if hasattr(self, "frequencies") != True:
            print("Algorithm frequencies are not defined yet.")
            
            if len(self.sources) == 0:
                print("Sources were not defined yet.")
                
            raise ValueError("It is lacking some peace of information.")
            
        if len(self.receivers) == 0:
            print ("Receivers were not defined yet. Nevertheless, it will run and you will be able to perform this step later.")
        
        #for si, source in enumerate(self.sources):
            #if source.type == "directional":
                #if self.frequencies.freq_vec.all() == source.freq_vec.all():
                    #raise ValueError("The frequencies considered to calculate one of the spherical harmonic coefficients for the "
                                     #+ str(si) +  " source are not the same that you are considering in the simulation.")
        
        admittances = []
        if len(self.materials) == 0:
            for i in (np.unique(self.grid.domain_indices)):
                admittances.append(Material(admittance = np.zeros_like(self.frequencies.freq_vec, dtype=np.complex64), freq_vec=self.frequencies.freq_vec, rho0=self.air.rho0, c0=self.air.c0))
        else:
            for material in self.materials:
                admittances.append(material.admittance)
                
        bempp.api.DEVICE_PRECISION_CPU = 'single'  
        
        for fi,f in enumerate(self.frequencies.freq_vec):
            
            self.current_freq = f
            k = self.air.k0[fi]
            
            print ("Working on frequency = %0.3f Hz." % f)

            #Generate mesh for this frequency:
            if f <= 50:
                factor = 12
            elif f <= 100:
                factor = 10
            elif f <= 150:
                factor = 8 
            elif f <= 265:
                factor = 6 
            else:
                factor = 4
            
            #print("Generating mesh...")
            try:
                self.generate_mesh(self.air.c0, f, factor)
                grid = bempp.api.import_grid(self.path_to_msh)
            except:
                print("Geometry file not found. Please, upload it:")
                self.add_geometry()
                
                self.generate_mesh(self.air.c0, f, factor)
                grid = bempp.api.import_grid(self.path_to_msh)

            '''
            #Open and reorder physical groups of the .msh file for this frequency:
            try:
                msh_path = "meshs/msh_%s_%sHz.msh" %(self.room_name, f)
                reorder_physical_groups(msh_path)
                grid = bempp.api.import_grid(msh_path)
            except:
                raise ValueError("Mesh file for %s Hz was not found." % f)
            '''   
            #print("Defining space...")
            #space = bempp.api.function_space(grid, "P", 1) # como nos code do Guto
            space = bempp.api.function_space(grid, "DP", 0)  # como nos code antigos               
            
            admittance = np.array([item[fi] for item in admittances])
            
            #Generate subspaces. It is needed if any of the receivers is binaural.
            if len(self.receivers) != 0 and any(receiver.type == "binaural" for receiver in self.receivers):  
                print("Defining subspaces...")
                # Initialize approximation spaces:
                sub_spaces = [None] * len(admittance) # Initalise as empty list
                spaceNumDOF = np.zeros(len(admittance), dtype=np.int32)
                for i in np.arange(len(admittance)): # Loop over subspaces
                    sub_spaces[i] = bempp.api.function_space(grid, "DP", 0, segments=[i])  # discontinuous piecewise-constant
                    spaceNumDOF[i] = sub_spaces[i].global_dof_count
                iDOF = np.concatenate((np.array([0]), np.cumsum(spaceNumDOF)))
            '''
            @bempp.api.complex_callable(jit=False) 
            def mu_fun_r(r,n,domain_index,result):
                result[0]=np.real(admittance[domain_index-1])
            @bempp.api.complex_callable(jit=False) 
            def mu_fun_i(r,n,domain_index,result):
                result[0]=np.imag(admittance[domain_index-1])
            
            mu_op_r = bempp.api.MultiplicationOperator(bempp.api.GridFunction(space,fun=mu_fun_r),space,space,space)
            mu_op_i = bempp.api.MultiplicationOperator(bempp.api.GridFunction(space,fun=mu_fun_i),space,space,space)
            '''
            
            @bempp.api.callable(complex=True, jit=False, parameterized=True)
            def mu_fun(x, n, domain_index, result, admittance):
                    result[0]=admittance[domain_index-1]
            
            mu_op = bempp.api.MultiplicationOperator(
                bempp.api.GridFunction(space, fun=mu_fun, function_parameters=admittance)
                , space, space, space)
            
            #print("identity")
            identity = bempp.api.operators.boundary.sparse.identity(
                space, space, space)
            #print("dlp")
            dlp = bempp.api.operators.boundary.helmholtz.double_layer(
                space, space, space, k)
            #print("slp")
            slp = bempp.api.operators.boundary.helmholtz.single_layer(
                space, space, space, k)
            
            #lhs = (.5 * identity + dlp - 1j*k*slp*(mu_op_r+1j*mu_op_i))
            lhs = (.5 * identity + dlp - 1j*k*slp*mu_op)
            
            del identity, dlp
            
            for si, source in enumerate(self.sources):
                
                print ("Working on source %s of %s." % (si+1, len(self.sources)))
                
                if source.type == "monopole":
                    
                    try:
                        i = np.where(source.freq_vec == f)[0][0]
                        q = np.array([[source.q[i]]])    
                    except:
                        raise ValueError("There is no information about the power of source %s for frequency %0.3f Hz." % (si, f))
                   
                    sh_coefficients_rotated_source = 1j*k/(4*np.pi)**0.5
                    
                    @bempp.api.callable(complex=True, jit=False)
                    def source_fun(r, n, domain_index, result):
                        result[0]=0
                        pos = np.linalg.norm(r-source.coord)
                        val  = q*np.exp(1j*k*pos)/(4*np.pi*pos)
                        result[0] +=  -(1j*admittance[domain_index-1]*k*val - val/(pos*pos) * (1j*k*pos-1)* np.dot(r-source.coord,n))                  
                        
                else:             
                    
                    try:
                        i = np.where(source.freq_vec == f)[0][0]
                        sh_coefficients_source = source.sh_coefficients[i]
                    except:
                        raise ValueError("The spherical harmonic coefficients for this source were not defined for frequency %0.3f Hz." % f)
                    
                    try:
                        sh_coefficients_source = 1/(10**(source.power_correction/20)) * sh_coefficients_source
                        
                    except:
                        print("There was not found any power correction for this source.")
                    
                    
                    rot_mat_FPTP = sh.get_rotation_matrix(0, -np.pi/2, 0, source.sh_order)   # Rotation Matrix front pole to top pole
                    rot_mat_AzEl = sh.get_rotation_matrix(0, -source.elevation, source.azimuth, source.sh_order); # Rotation Matrix for Loudspeaker orientation
                                        
                    sh_coefficients_rotated_source = sh_coefficients_source.reshape((np.size(sh_coefficients_source),1))
                    sh_coefficients_rotated_source = sh.reflect_sh(rot_mat_FPTP * sh_coefficients_rotated_source, 1, 0, 0)  # Convert to top-pole format
                    sh_coefficients_rotated_source = rot_mat_AzEl * sh_coefficients_rotated_source
                    
                    #@bempp.api.callable(complex=True, jit=True, parameterized=True)
                    #def source_fun(r, n, domain_index, result, parameters):
                        
                        #result[0]=0
                        
                        #coord = np.real(parameters[:3])
                        #k = parameters[3]
                        #mu = parameters[4:]
                        
                        #val, d_val  = sh.spherical_basis_out_all(k, sh_coefficients_rotated, r-coord, n)
                        #result[0] += d_val - 1j*mu[domain_index]*k*val
                        #result[0] = d_val - 1j*mu[domain_index]*k*val
                    
                    #source_coord = source.coord.reshape(3)
                    @bempp.api.callable(complex=True, jit=False)
                    #@bempp.api.callable(complex=True, jit=True)
                    def source_fun(r, n, domain_index, result):
                        result[0]=0
                        val, d_val  = sh.spherical_basis_out_all(k, sh_coefficients_rotated_source, r-source.coord.reshape(3), n)
                        result[0] += d_val - 1j*admittance[domain_index-1]*k*val
                    
                    #source_parameters = np.zeros(4+len(admittance),dtype = 'complex128')

                    #source_parameters[:3] = source.coord
                    #source_parameters[3] = k
                    #source_parameters[4:] = admittance
                
                #rhs = bempp.api.GridFunction.from_zeros(self.space)
                source_grid = bempp.api.GridFunction(space, fun=source_fun)
                rhs =  -slp * source_grid
                    
                #print("boundary_pressure")
                boundary_pressure, info = bempp.api.linalg.gmres(lhs, rhs, tol=1E-5)
                                
                #un = 1j*(mu_op_r+1j*mu_op_i)*k*boundary_pressure - source_grid
                un = 1j*mu_op*k*boundary_pressure - source_grid
                   
                self.boundary_pressure.append (boundary_pressure.coefficients)
                
                del rhs
                try:
                    del sh_coefficients_source, rot_mat_FPTP, rot_mat_AzEl
                except:
                    pass                    
                
                if len(self.receivers) != 0:
                    for ri, receiver in enumerate(self.receivers):

                        print ("Working on receiver %s of %s." % (ri+1, len(self.receivers)))

                        if receiver.type == "omni":
                            
                            slp_pot = bempp.api.operators.potential.helmholtz.single_layer(
                                space, receiver.coord.T, k)
                            
                            dlp_pot = bempp.api.operators.potential.helmholtz.double_layer(
                                space, receiver.coord.T, k)
                            
                            pScat =  (slp_pot*un - dlp_pot*boundary_pressure)[0][0]

                            
                            distance  = np.linalg.norm(receiver.coord - source.coord)
                            if source.type == "monopole":
                                pInc = (q[0][0]*np.exp(1j*k*distance)/(4*np.pi*distance))

                            else:
                                pInc = (sh.spherical_basis_out_p0_only(k, sh_coefficients_rotated_source, receiver.coord.reshape(3) - source.coord.reshape(3)))[0][0]
                            
                            pT = pScat + pInc

                            self.incident_pressure.append(pInc)
                            self.scattered_pressure.append(pScat) 
                            self.total_pressure.append(pT) 
                            
                            del dlp_pot, slp_pot, pScat, distance, pInc, pT
                                                      
                            gc.collect(generation=0)
                            gc.collect(generation=1)
                            gc.collect(generation=2)


                        else:

                            AnmInc  = np.zeros([(receiver.sh_order + 1) ** 2], np.complex64)
                            AnmInc  = sh.get_translation_matrix((receiver.coord - source.coord).reshape((3,)), k, source.sh_order, receiver.sh_order) @ sh_coefficients_rotated_source
                            #print("AnmInc")
                            AnmScat = np.zeros([(receiver.sh_order + 1) ** 2], np.complex64)
                            #print("AnmScat")

                            for n in range(receiver.sh_order + 1):
                                for m in range(-n, n+1):
                                    #print("OpDnmFunc")
                                    # Define functions to be evaluated:
                                    @bempp.api.complex_callable(jit=False)
                                    #@bempp.api.callable(complex=True, jit=True)
                                    def OpDnmFunc(x, nUV, domain_index, result):
                                        H, dHdn = sh.spherical_basis_in(n, m, k, x - receiver.coord.reshape(3), nUV)
                                        result[0] = dHdn
                                        
                                    #print("OpSnmFunc")
                                    @bempp.api.complex_callable(jit=False)
                                    #@bempp.api.callable(complex=True, jit=True)
                                    def OpSnmFunc (x, nUV, domain_index, result):
                                        H = sh.spherical_basis_in_p0_only(n, m, k, x - receiver.coord.reshape(3))
                                        result[0] = H

                                    for i in np.arange(len(admittance)):  # loop over subspaces

                                        # Integrate the SH functions with the basis functions from the approximation spaces:
                                        #print("OpSnmGF")
                                        OpSnmGF = bempp.api.GridFunction(sub_spaces[i], fun=OpSnmFunc)
                                        #print("OpDnmGF")
                                        OpDnmGF = bempp.api.GridFunction(sub_spaces[i], fun=OpDnmFunc)


                                        # Integrate the SH functions with the basis functions from the approximation spaces:
                                        #OpSnmGF = bempp.api.GridFunction(space, fun=OpSnmFunc)
                                        #OpDnmGF = bempp.api.GridFunction(space, fun=OpDnmFunc)


                                        # Integrate the SH functions with the basis functions from the approximation spaces:
                                        #OpSnmGF =  bempp.api.MultiplicationOperator(
                                         #   bempp.api.GridFunction(space, fun=OpSnmFunc)
                                          #  , space, space, space)

                                        #OpDnmGF = bempp.api.MultiplicationOperator(
                                         #   bempp.api.GridFunction(space, fun=OpDnmFunc)
                                          #  , space, space, space)

                                        # Extract projections and conjugate to get discrete form of intended operators:
                                        #OpSnm = np.conj(OpSnmGF.projections())
                                        #OpDnm = np.conj(OpDnmGF.projections())

                                        # Extract projections and conjugate to get discrete form of intended operators:
                                        #print("OpSnm")
                                        OpSnm = np.conj(OpSnmGF.projections(sub_spaces[i]))
                                        #print("OpDnm")
                                        OpDnm = np.conj(OpDnmGF.projections(sub_spaces[i]))
                                        
                                        del OpSnmGF, OpDnmGF

                                        #AnmScat[n**2 + n + m] = 1j*k*np.sum(boundary_pressure * (OpDnm + 1j*k*mu_op * OpSnm))

                                        #AnmScat[n**2 + n + m] = 1j*k*np.sum(boundary_pressure * (OpDnmGF + 1j*k*mu_op * OpSnmGF))
                                        #print("AnmScat")
                                        AnmScat[n**2 + n + m] += 1j*k*np.sum(boundary_pressure.coefficients[iDOF[i]:iDOF[i+1]] * (OpDnm + np.complex128(1j*k*admittance[i]) * OpSnm))

                                        del OpSnm, OpDnm
                            
                            rotation_matrix = sh.get_rotation_matrix(0, 0, -receiver.azimuth, receiver.sh_order)
                            AnmInc = rotation_matrix * AnmInc
                            AnmScat = rotation_matrix * AnmScat

                            try:
                                i = np.where(receiver.freq_vec == f)[0][0]
                                sh_coefficients_receiver_left = receiver.sh_coefficients_left[i]
                                sh_coefficients_receiver_right = receiver.sh_coefficients_right[i]
                            except:
                                raise ValueError("The spherical harmonic coefficients for this receiver were not defined for frequency %0.3f Hz." % f)

                            # Scale BEM results as required by GRAS normalisation process:
                            RequiredGain = 1.9599e-05  # Calculated by applying required calibration process to a hybrid simulation
                            AnmInc  = AnmInc  * RequiredGain
                            AnmScat = AnmScat * RequiredGain

                            pInc = [np.matmul(AnmInc.reshape(1, len(AnmInc)), sh_coefficients_receiver_left)[0], np.matmul(AnmInc.reshape(1, len(AnmInc)), sh_coefficients_receiver_right)[0]]
                            pScat = [np.matmul(AnmScat.reshape(1, len(AnmScat)), sh_coefficients_receiver_left)[0], np.matmul(AnmScat.reshape(1, len(AnmScat)), sh_coefficients_receiver_right)[0]]
                            pT = [a + b for a, b in zip(pInc, pScat)]


                            self.scattered_pressure.append(pScat)
                            self.incident_pressure.append(pInc)
                            self.total_pressure.append(pT)  
            
                            del AnmInc, AnmScat, rotation_matrix, pInc, pScat, pT, sh_coefficients_receiver_left, sh_coefficients_receiver_right    

                            #print("Collecting garbage...")
                            gc.collect(generation=0)
                            gc.collect(generation=1)
                            gc.collect(generation=2)
                                
                    del boundary_pressure, un
                    
            del space, grid, mu_op, lhs, slp
            try:
                del sub_spaces, spaceNumDOF, iDOF 
            except:
                pass
            
            bempp.api.clear_fmm_cache()
            
            self.simulated_freqs.append(f)
                                        
            if save == True:
                self.save()
                            

    def receiver_evaluate (self, source, receiver, **kwargs):
        
        if "boundary_pressure" in kwargs and "boundary_velocity" in kwargs:
            
            pScat = np.zeros(len(self.frequencies.freq_vec), dtype = np.complex64)
            pInc = np.zeros(len(self.frequencies.freq_vec), dtype = np.complex64)
            pT = np.zeros(len(self.frequencies.freq_vec), dtype = np.complex64)
            
            for fi,f in enumerate(self.frequencies.freq_vec):
        
                k = self.air.k0[fi]
            
                dlp_pot = bempp.api.operators.potential.helmholtz.double_layer(
                    self.space, receiver.coord.T, k, assembler = "dense", device_interface = "numba")
                slp_pot = bempp.api.operators.potential.helmholtz.single_layer(
                    self.space, receiver.coord.T, k, assembler = "dense", device_interface = "numba")
                
                pS = -dlp_pot.evaluate(kwargs["boundary_pressure"])[0][0] + slp_pot.evaluate(kwargs["boundary_velocity"])[0][0]
                pScat[fi] = pS

                distance  = np.linalg.norm(receiver.coord - source.coord)
                pInc[fi] = q[0][0]*np.exp(1j*k*distance)/(4*np.pi*distance)

                pT[fi] = pScat[fi] + pInc[fi]
                
            self.scattered_pressure.append(pScat)
            self.incident_pressure.append(pInc)
            self.total_pressure.append(pT)  
            
    
    def plot_spl (self, sources=[], receivers=[]):

        sources = np.array(sources)-1
        receivers = np.array(receivers)-1
        
        if sources.size == 0:
            sources=np.arange(len(self.sources))
        if receivers.size == 0:
            receivers=np.arange(len(self.receivers))
        
        for source in sources:
            for receiver in receivers:
                i=0
                for s_i in np.arange(len(self.sources)):
                    for r_i, r in enumerate(self.receivers):
                        if s_i == source and r_i == receiver: 
                                                            
                            if r.type == "omni":
                                plt.plot(self.simulated_freqs, 20*np.log10(np.abs(self.total_pressure[s_i*len(self.receivers)+r_i : : len(self.sources)*len(self.receivers)])/(2e-5*np.sqrt(2))))
                                plt.legend(["Source %s, Receiver %s" % (s_i+1, r_i+1)], fontsize = 13)
                            else:                                 
                                plt.plot(self.simulated_freqs, 20*np.log10(np.abs([item[0] for item in self.total_pressure[s_i*len(self.receivers)+r_i : : len(self.sources)*len(self.receivers)]])/(2e-5*np.sqrt(2))))
                                plt.plot(self.simulated_freqs, 20*np.log10(np.abs([item[1] for item in self.total_pressure[s_i*len(self.receivers)+r_i : : len(self.sources)*len(self.receivers)]])/(2e-5*np.sqrt(2))))
                                plt.legend(["left", "right"], fontsize = 13)

                            plt.xlabel('Frequency [Hz]', fontsize = 15)
                            plt.ylabel('SPL [dB]', fontsize = 15)
                            
                            plt.xscale('log')
                            
                            plt.yticks(fontsize = 12)
                            plt.xticks([20, 30, 40, 60, 80, 100, 200, 400, 1000], ['20', '','40','60','80','100','200','400','1k'], fontsize = 12)
                            
                            plt.xlim(self.simulated_freqs[0], self.simulated_freqs[-1])
        
                            plt.savefig('SPL_LS%s_MP%s.pdf' %(s_i+1, r_i+1), bbox_inches='tight')
                            plt.show()

                        i+=1

                
    def plot_phase (self, sources=[], receivers=[]):

        sources = np.array(sources)-1
        receivers = np.array(receivers)-1
        
        if sources.size == 0:
            sources=np.arange(len(self.sources))
        if receivers.size == 0:
            receivers=np.arange(len(self.receivers))
        
        for source in sources:
            for receiver in receivers:
                i=0
                for s_i in np.arange(len(self.sources)):
                    for r_i, r in enumerate(self.receivers):
                        if s_i == source and r_i == receiver: 
                                                            
                            if r.type == "omni":
                                plt.plot(self.simulated_freqs, np.rad2deg(np.angle(np.conj(self.total_pressure[s_i*len(self.receivers)+r_i : : len(self.sources)*len(self.receivers)]))))
                                plt.legend(["Source %s, Receiver %s" % (s_i+1, r_i+1)], fontsize = 13)
                            else:                                 
                                plt.plot(self.simulated_freqs, np.rad2deg(np.angle(np.conj([item[0] for item in self.total_pressure[s_i*len(self.receivers)+r_i : : len(self.sources)*len(self.receivers)]]))))
                                plt.plot(self.simulated_freqs, np.rad2deg(np.angle(np.conj([item[1] for item in self.total_pressure[s_i*len(self.receivers)+r_i : : len(self.sources)*len(self.receivers)]]))))
                                plt.legend(["left", "right"], fontsize = 13)

                            plt.xlabel('Frequency [Hz]', fontsize = 15)
                            plt.ylabel('Phase [°]', fontsize = 15)
                            
                            plt.yticks([-180,-120,-60,0,60,120,180], ['-180','-120','-60','0','60','120','180'], fontsize = 12)
                            plt.xticks([20, 30, 40, 60, 80, 100, 200, 400, 1000], ['20', '','40','60','80','100','200','400','1k'], fontsize = 12)
                            
                            plt.xscale('log')
                            plt.xlim(self.simulated_freqs[0], self.simulated_freqs[-1])
                            
                            plt.savefig('Phase_LS%s_MP%s.pdf' %(s_i+1, r_i+1), bbox_inches='tight')
                            plt.show()

                        i+=1
                
    
    def save(self, place="drive"):
                
        saved_name = "%s.pickle" % self.room_name
        pickle_obj = open(saved_name, "wb")
        pickle.dump(self, pickle_obj)
        pickle_obj.close()
        
        if place == "drive":
            try:
                shutil.copy2(saved_name, "/content/drive/MyDrive")
            except:
                from google.colab import drive
                print("Mount your Google Drive, so that you are gonna be able to save your simulation:")
                drive.mount('/content/drive')
                
                shutil.copy2(saved_name, "/content/drive/MyDrive")
                
        elif place == "local":
            files.download(saved_name)
            


    def map(self, sources=[], freqs=[], opacity=0.5):
                
        from matplotlib import style
        import matplotlib.colors as mcolors
        import matplotlib.cm as cm
        style.use("seaborn-talk")
        
        import plotly.figure_factory as ff
        import plotly.graph_objs as go
        from bempp.api import GridFunction
        from bempp.api.grid.grid import Grid

        freqs = np.array(freqs)
        if freqs.size == 0:
            try:
                freqs = np.array([self.frequencies.freq_vec[0]])
            except:
                freqs = np.array([20])
                
        sources = np.array(sources)
        if sources.size == 0:
            sources = np.array([0])

            
        for f in freqs:
            
            if f <= 50:
                factor = 12
            elif f <= 100:
                factor = 10
            elif f <= 150:
                factor = 8 
            elif f <= 265:
                factor = 6 
            else:
                factor = 4

            self.generate_mesh(self.air.c0, f, factor)

            grid = bempp.api.import_grid(self.path_to_msh)
            
            for source in sources:
                
                def configure_plotly_browser_state():
                    import IPython
                    display(IPython.core.display.HTML('''
                            <script src="/static/components/requirejs/require.js"></script>
                            <script>
                              requirejs.config({
                                paths: {
                                  base: '/static/base',
                                  plotly: 'https://cdn.plot.ly/plotly-1.5.1.min.js?noext',
                                },
                              });
                            </script>
                            '''))
                
                plotly.offline.init_notebook_mode()

                vertices = grid.vertices
                elements = grid.elements
                x=vertices[0, :]
                y=vertices[1, :]
                z=vertices[2, :]
                
                fig = ff.create_trisurf(
                    x=x,
                    y=y,
                    z=z,
                    simplices=elements.T,
                    plot_edges=False,
                    show_colorbar=False,
                )
                fig['data'][0].update(opacity=opacity)
                fig['layout']['scene'].update(go.layout.Scene(aspectmode='data'))

                #colors
                simulated_freqs = np.array(self.simulated_freqs)
                fi = np.where(self.simulated_freqs == f)[0][0]
                
                boundary_pressure = 20*np.log10(np.abs(self.boundary_pressure[fi*len(self.sources) + source])/(2e-5*np.sqrt(2))) 
                boundary_pressure = boundary_pressure.tolist()

                val_min, val_max = min(boundary_pressure), max(boundary_pressure)
                val_center = np.average(boundary_pressure)

                offset = mcolors.TwoSlopeNorm(vmin=val_min, vcenter=val_center, vmax=val_max)

                boundary_pressure_scale = offset(boundary_pressure)

                color_list_rgba = [cm.rainbow(x) for x in boundary_pressure_scale]

                color_list_rgb = []
                for color in color_list_rgba:
                    color = [round(num, 5) for num in color]
                    color_list_rgb.append(mcolors.to_rgb(color))

                for i, element in enumerate(elements[0]):  
                    color = 'rgb(%s,%s,%s)' % (color_list_rgb[i][0], color_list_rgb[i][1], color_list_rgb[i][2])
                    fig.add_trace(go.Scatter3d(x=[x[elements[0][i]]], y=[y[elements[1][i]]], z=[z[elements[2][i]]], marker=dict(size=10, color=color, symbol='diamond'), showlegend=False)) 

                fig.add_trace(go.Mesh3d(x=[-6,6,-6,6], y=[-6,6,-6,6], z=0 * np.zeros_like([-6,6,-6,6]), color='red', opacity=0.5, showscale=False))

                configure_plotly_browser_state() 
                plotly.offline.iplot(fig)
            
            
            
def reorder_physical_groups(msh_path):
    """
    This function reorders the physical groups of the .msh file
    """

    import meshio

    #Reorder physical groups       
    gmsh.initialize(sys.argv)
    gmsh.open(msh_path)

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


    gmsh.write(msh_path)
    gmsh.finalize() 
