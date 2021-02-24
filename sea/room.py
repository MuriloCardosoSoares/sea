import time
import warnings
import bempp.api
import numpy as np

from matplotlib import pylab as plt
import cloudpickle
import collections
bempp.api.PLOT_BACKEND = "gmsh"

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
            pass
        
        self.receivers = []
        self.sources = []
        self.materials = []
        
        
    def air_properties(self, c0 = 343.0, rho0 = 1.21, temperature = 20.0, humid = 50.0, p_atm = 101325.0):
        self.air = Air(c0 = 343.0, rho0 = 1.21, temperature = 20.0, humid = 50.0, p_atm = 101325.0)
        if hasattr(self, "frequencies"):
            self.air.k0 = 2*np.pi*self.frequencies.freq_vec/self.air.c0
        
    
    def algorithm_control(self, freq_init=20.0, freq_end=200.0, freq_step=1, freq_vec=[]):
        self.frequencies = Algorithm(freq_init, freq_end, freq_step, freq_vec)
        self.air.k0 = 2*np.pi*self.frequencies.freq_vec/self.air.c0
        
    
    def add_receiver(self, coord = [1.0, 0.0, 0.0], type="omni", **kwargs):
        self.receivers.append(Receiver(coord, type, **kwargs))
    
    
    def del_receivers(self, *args):
    
        if args:
            for position in args:
                del self.receivers[position]
        else:
            self.receivers.clear()
        
        
    def list_receivers(self):
        print("Receivers coordinates are:")
        for receiver in self.receivers:
            print (receiver)
            
            
    def add_source(self, coord=[0.0, 0.0, 1.0], type="monopole", **kwargs):
        self.sources.append(Source(coord, type, **kwargs))  

        
    def del_sources(self, *args):
    
        if args:
            for position in args:
                del self.sources[position]
        else:
            self.sources.clear()
     
    
    def list_sources(self):
        print("Sources are:")
        for source in self.sources:
            print (source)
     
    
    def add_mesh(self):
        """
        This function imports a .msh file.
        """
        from google.colab import files
        uploaded = files.upload()
        
        for key in uploaded:
            self.path_to_msh = key
     

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
                
            elif material.admittance.size == 0 and "absorber_type" in kwargs:
                meterial.alpha_from_impedance(absorber_type=kwargs["absorber_type"])
                self.materials.append(material)
            
            
    def del_materials(self, *args):
    
        if args:
            for position in args:
                del self.materials[position]
        else:
            self.materials.clear()
            
    
    def list_materials(self):
        for material in self.materials:
            print(material)          

            
    def view(self, opacity = 0.2):
        
        from matplotlib import style
        style.use("seaborn-talk")
        
        import plotly.figure_factory as ff
        import plotly.graph_objs as go
        from bempp.api import GridFunction
        from bempp.api.grid.grid import Grid

        try:
            msh = bempp.api.import_grid(self.path_to_msh)
        except:
            print("Mesh file not found. Please, upload it again:")
            uploaded = files.upload()
            
            for key in uploaded:
                self.path_to_msh = key
                
            msh = bempp.api.import_grid(self.path_to_msh)
        
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

        vertices = msh.vertices
        elements = msh.elements
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
        
        
    def run(self, method='bem', save=True):
        
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
            
        self.boundary_pressure = [] 
        self.boundary_velocity = [] 
        
        self.scattered_pressure = []
        self.incident_pressure = []
        self.total_pressure = []

        try:
            msh = bempp.api.import_grid(self.path_to_msh)
        except:
            print("Mesh file not found. Please, upload it again:")
            uploaded = files.upload()
            
            for key in uploaded:
                self.path_to_msh = key
                
            msh = bempp.api.import_grid(self.path_to_msh)
            
        space = bempp.api.function_space(msh, "P", 1)
        
        for fi,f in enumerate(self.frequencies.freq_vec):
            
            print ("Working on frequency = %0.3f Hz." % f)
            
            admittance = np.array([item[fi] for item in admittances])
            k = self.air.k0[fi]
            
            @bempp.api.callable(complex=True, jit=True, parameterized=True)
            def mu_fun(x, n, domain_index, result, admittance):
                    result[0]=np.conj(admittance[domain_index])

            mu_op = bempp.api.MultiplicationOperator(
                bempp.api.GridFunction(space, fun=mu_fun, function_parameters=admittance)
                , space, space, space)

            identity = bempp.api.operators.boundary.sparse.identity(
                space, space, space)
            dlp = bempp.api.operators.boundary.helmholtz.double_layer(
                space, space, space, k, assembler="dense", device_interface='numba')
            slp = bempp.api.operators.boundary.helmholtz.single_layer(
                space, space, space, k,assembler="dense", device_interface='numba')

            for source in self.sources:
                
                if source.type == "monopole":
                
                    @bempp.api.callable(complex=True, jit=True, parameterized=True)
                    def source_fun(r, n, domain_index, result, parameters):

                        coord = np.real(parameters[:3])
                        k = parameters[3]
                        q = parameters[4]
                        mu = parameters[5:]

                        pos  = np.linalg.norm(r-coord)
                        val  = q*np.exp(1j*k*pos)/(4*np.pi*pos)

                        result[0] = -(1j * mu[domain_index] * k * val -
                            val / (pos**2) * (1j*k*pos - 1) * np.dot(r-coord, n))
                    
                    sh_coefficients_rotated = 1j*k/(4*np.pi)**0.5
                    
                    source_parameters = np.zeros(5+len(admittance),dtype = 'complex128')

                    source_parameters[:3] = source.coord
                    source_parameters[3] = k
                    source_parameters[4] = source.q
                    source_parameters[5:] = admittance
                    
                    rhs = bempp.api.GridFunction(space,fun=source_fun,
                                      function_parameters=source_parameters)
                        
                else:             
                    
                    try:
                        i = np.where(source.freq_vec == f)[0][0]
                        sh_coefficients = source.sh_coefficients[i]
                    except:
                        raise ValueError("The spherical harmonic coefficients were not defined for frequency %0.3f Hz." % f)
                    
                    try:
                        sh_coefficients = 1/(10**(source.power_correction/20)) * sh_coefficients
                        
                    except:
                        pass
                    
                    sh_coefficients_rotated = sh_coefficients.reshape((np.size(sh_coefficients),1))
                    
                    rot_mat_FPTP = sh.get_rotation_matrix(0, -np.pi/2, 0, source.sh_order)   # Rotation Matrix front pole to top pole
                    rot_mat_AzEl = sh.get_rotation_matrix(0, -source.elevation, source.azimuth, source.sh_order); # Rotation Matrix for Loudspeaker orientation

                    sh_coefficients_rotated = sh.reflect_sh(rot_mat_FPTP * sh_coefficients_rotated, 1, 0, 0)  # Convert to top-pole format
                    sh_coefficients_rotated = rot_mat_AzEl * sh_coefficients_rotated
                    
                    #@bempp.api.callable(complex=True, jit=True, parameterized=True)
                    #def source_fun(r, n, domain_index, result, parameters):
                        
                        #result[0]=0
                        
                        #coord = np.real(parameters[:3])
                        #k = parameters[3]
                        #mu = parameters[4:]
                        
                        #val, d_val  = sh.spherical_basis_out_all(k, sh_coefficients_rotated, r-coord, n)
                        #result[0] += d_val - 1j*mu[domain_index]*k*val
                        #result[0] = d_val - 1j*mu[domain_index]*k*val
                    
                    @bempp.api.complex_callable(jit=False)
                    def source_fun(r, n, domain_index, result):
                        result[0]=0
                        val, d_val  = sh.spherical_basis_out_all(k, sh_coefficients_rotated, r-source.coord.reshape(3), n)
                        result[0] += d_val - 1j*admittance[domain_index]*k*val
                    
                    #source_parameters = np.zeros(4+len(admittance),dtype = 'complex128')

                    #source_parameters[:3] = source.coord
                    #source_parameters[3] = k
                    #source_parameters[4:] = admittance
                
                #rhs = bempp.api.GridFunction.from_zeros(self.space)
                    rhs = bempp.api.GridFunction(space, fun=source_fun)

                                    
                a = 1j*k*self.air.c0*self.air.rho0
                Y = a*(mu_op)
                lhs = (0.5*identity+dlp) - slp*Y

                boundary_pressure, info = bempp.api.linalg.gmres(lhs, rhs, tol=1E-5)
                boundary_velocity = Y*boundary_pressure - rhs

                self.boundary_pressure.append (boundary_pressure.coefficients)
                self.boundary_velocity.append (boundary_velocity.coefficients)
                
                self.save()
                
            if len(self.receivers) != 0:
                for receiver in self.receivers:
                    
                    if receiver.type == "omni":
                        
                        dlp_pot = bempp.api.operators.potential.helmholtz.double_layer(
                            space, receiver.coord.T, k, assembler = "dense", device_interface = "numba")
                        slp_pot = bempp.api.operators.potential.helmholtz.single_layer(
                            space, receiver.coord.T, k, assembler = "dense", device_interface = "numba")

                        pScat = (-dlp_pot.evaluate(boundary_pressure)[0][0] + slp_pot.evaluate(boundary_velocity)[0][0]).reshape(1)
                        distance  = np.linalg.norm(receiver.coord - source.coord)
                        
                        if source.type == "monopole":
                            pInc = (source.q[0][0]*np.exp(1j*k*distance)/(4*np.pi*distance)).reshape(1)
                            
                        else:
                            pInc = (sh.spherical_basis_out_p0_only(k, sh_coefficients_rotated, receiver.coord.reshape(3) - source.coord.reshape(3))).reshape(1)
                            
                        pT = pScat + pInc
                        
                    else:
                        
                        AnmInc  = np.zeros([(receiver.sh_order + 1) ** 2], np.complex64)
                        AnmInc  = sh.get_translation_matrix((receiver.coord - source.coord).reshape((3,)), k, source.sh_order, receiver.sh_order) @ sh_coefficients_rotated
                        
                        AnmScat = np.zeros([(receiver.sh_order + 1) ** 2], np.complex64)
                        
                        for n in range(receiver.sh_order + 1):
                            for m in range(-n, n+1):

                                # Define functions to be evaluated:
                                @bempp.api.complex_callable(jit=False)
                                def OpDnmFunc(x, nUV, domain_index, result):
                                    H, dHdn = sh.spherical_basis_in(n, m, k, x - receiver.coord.reshape(3), nUV)
                                    result[0] = dHdn
                                
                                @bempp.api.complex_callable(jit=False)
                                def OpSnmFunc (x, nUV, domain_index, result):
                                    H = sh.spherical_basis_in_p0_only(n, m, k, x - receiver.coord.reshape(3))
                                    result[0] = H


                                # Integrate the SH functions with the basis functions from the approximation spaces:
                                #OpSnmGF = bempp.api.GridFunction(space, fun=OpSnmFunc)
                                #OpDnmGF = bempp.api.GridFunction(space, fun=OpDnmFunc)

                                # Extract projections and conjugate to get discrete form of intended operators:
                                #OpSnm = np.conj(OpSnmGF.projections())
                                #OpDnm = np.conj(OpDnmGF.projections())
                                
                                
                                # Integrate the SH functions with the basis functions from the approximation spaces:
                                OpSnmGF =  bempp.api.MultiplicationOperator(
                                    bempp.api.GridFunction(space, fun=OpSnmFunc)
                                    , space, space, space)
                                
                                OpDnmGF = bempp.api.MultiplicationOperator(
                                    bempp.api.GridFunction(space, fun=OpDnmFunc)
                                    , space, space, space)
                                
                                print(boundary_pressure)
                                print(OpDnm)
                                print(mu_op)
                                print(OpSnm)
                                
                                AnmScat[n**2 + n + m] = 1j*k*np.sum(boundary_pressure * (OpDnm + 1j*k*mu_op * OpSnm))
                                
                            rotation_matrix = sh.get_rotation_matrix(0, 0, -receiver.azimuth, receiver.sh_order)
                            AnmInc = rotation_matrix * AnmInc
                            AnmScat = rotation_matrix * AnmScat
                            
                            pInc = [AnmInc*receiver.sh_coefficients_left, AnmInc*receiver.sh_coefficients_right]
                            pScat = [AnmScat*receiver.sh_coefficients_left, AnmScat*receiver.sh_coefficients_right]
                            pT = c = [a + b for a, b in zip(pInc, pScat)]
                            

                    self.scattered_pressure.append(pScat)
                    self.incident_pressure.append(pInc)
                    self.total_pressure.append(pT) 

                    self.save()
                
                #self.receiver_evaluate(source, receiver, boundary_pressure = boundary_pressure, boundary_velocity = boundary_velocity)
            

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
                print(pScat)
                distance  = np.linalg.norm(receiver.coord - source.coord)
                pInc[fi] = source.q[0][0]*np.exp(1j*k*distance)/(4*np.pi*distance)
                print(pInc)
                pT[fi] = pScat[fi] + pInc[fi]
                print(pT)
                
            self.scattered_pressure.append(pScat)
            self.incident_pressure.append(pInc)
            self.total_pressure.append(pT)  
            
            
    def save(self, place="drive", name = "my_room"):
        
        self.room_name = name
        
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
            
