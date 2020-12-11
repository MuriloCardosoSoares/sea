import time
import warnings
import bempp.api
import numpy as np

from matplotlib import pylab as plt
import cloudpickle
import collections
bempp.api.PLOT_BACKEND = "gmsh"

import plotly
from matplotlib import style
style.use("seaborn-talk")

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
        self.receivers = []
        self.sources = []
        self.materials = []
        self.EoI = 1
        self.v = 0 
        

    def algorithm_control(self, freq_init=20.0, freq_end=200.0, freq_step=1, freq_vec=[]):
        self.frequencies = Algorithm(freq_init, freq_end, freq_step, freq_vec) 
        
    
    def add_receiver(self, coord = [1.0, 0.0, 0.0]):
        self.receivers.append(Receiver(coord))
    
    
    def del_receivers(self, *args):
    
        if args:
            for position in args:
                del self.receivers[position]
        else:
            self.receivers.clear()
        
        
    def list_receivers(self):
        print("Receivers coordinates are:")
        for receiver in self.receivers:
            print (receiver.coord)
            
            
    def add_source(self, coord = [0.0, 0.0, 1.0], q = [1.0], source_type="monopole"):
        self.sources.append(Source(coord, q , source_type))  

        
    def del_sources(self, *args):
    
        if args:
            for position in args:
                del self.sources[position]
        else:
            self.sources.clear()
     
    
    def list_sources(self):
        print("Sources are:")
        for source in self.sources:
            print ("Coordinate = %s, q = %s" % (source.coord, source.q))
     
    
    def add_mesh(self, show_mesh=False, gmsh_filepath=None, reorder_domain_index=False):
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
        from google.colab import files
        uploaded = files.upload()
        
        for key in uploaded:
            self.path_to_msh = key
        
        try:  
            import gmsh
        except :
            import gmsh_api.gmsh as gmsh

        import sys
        import os

        if reorder_domain_index:
            
            gmsh.initialize(sys.argv)
            gmsh.open(self.path_to_msh) # Open msh
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
            path_name = os.path.dirname(self.path_to_msh)
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
            self.msh = bempp.api.import_grid(self.path_to_msh)
     

    def add_material(self, normal_inidence_alpha=[], statistical_alpha=[], octave_bands_statistical_alpha=[], octave_bands=[],
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
                
            if kwargs["absorber_type"] == "perforated panel":
                material.perforated_panel(kwargs["parameters"])
                self.materials.append(material)
                
            if kwargs["absorber_type"] == "microperforated panel":
                material.microperforated_panel(kwargs["parameters"])
                self.materials.append(material)

        else:
            material = Material(normal_inidence_alpha=normal_inidence_alpha, statistical_alpha=statistical_alpha, octave_bands_statistical_alpha=octave_bands_statistical_alpha, 
                                octave_bands=octave_bands, third_octave_bands_statistical_alpha=third_octave_bands_statistical_alpha, 
                                third_octave_bands=third_octave_bands, admittance=admittance, normalized_surface_impedance=normalized_surface_impedance, 
                                surface_impedance=surface_impedance, freq_vec=self.frequencies.freq_vec, rmk1=rmk1, rho0=self.air.rho0, c0=self.air.c0)

            if material.admittance == 0 and "absorber_type" in kwargs:
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
            
            
    def view(self):
        
    import numpy as np
    import plotly
    import matplotlib.pyplot as plt
    from matplotlib import style
    from bemder import receivers
    style.use("seaborn-talk")

    def plot_problem(obj,S=None,R=None,grid_pts=None, pT=None, opacity = 0.75, mode="element", transformation=None):
        import plotly.figure_factory as ff
        import plotly.graph_objs as go
        from bempp.api import GridFunction
        from bempp.api.grid.grid import Grid
        import numpy as np
        obj = obj[1]
        if transformation is None:
            transformation = np.abs

        p2dB = lambda x: 20*np.log10(np.abs(x)/2e-5) 
        if transformation is None:
            transformation = np.abs
        if transformation == "dB":
            transformation = p2dB

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

        if isinstance(obj, Grid):
            vertices = obj.vertices
            elements = obj.elements
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

            fig.add_trace(go.Mesh3d(x=[-6,6,-6,6], y=[-6,6,-6,6], z=0 * np.zeros_like([-6,6,-6,6]), color='red', opacity=0.5, showscale=False))

            configure_plotly_browser_state() 
            plotly.offline.iplot(fig)

        elif isinstance(obj, GridFunction):


            grid = obj.space.grid
            vertices = grid.vertices
            elements = grid.elements

            local_coordinates = np.array([[1.0 / 3], [1.0 / 3]])
            values = np.zeros(grid.entity_count(0), dtype="float64")
            for element in grid.entity_iterator(0):
                index = element.index
                local_values = np.real(
                    transformation(obj.evaluate(index, local_coordinates))
                )
                values[index] = local_values.flatten()

            fig = ff.create_trisurf(
                x=vertices[0, :],
                y=vertices[1, :],
                z=vertices[2, :],
                color_func=values,
                colormap = "Jet",
                simplices=elements.T,
                show_colorbar = True,

            )
            fig['data'][0].update(opacity=opacity)

            fig['layout']['scene'].update(go.layout.Scene(aspectmode='data'))
            if R != None:
                fig.add_trace(go.Scatter3d(x = R.coord[:,0], y = R.coord[:,1], z = R.coord[:,2],name="Receivers"))

            if S != None:    
                if S.wavetype == "spherical":
                    fig.add_trace(go.Scatter3d(x = S.coord[:,0], y = S.coord[:,1], z = S.coord[:,2],name="Sources"))

            configure_plotly_browser_state()    
            plotly.iplot(fig)
