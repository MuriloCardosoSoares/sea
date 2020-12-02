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
     

    def add_material(self, normal_inidence_alpha=[], statistical_alpha=[], octave_bands_statistical_alpha=[], octave_bands=[],
                     third_octave_bands_statistical_alpha=[], third_octave_bands=[], admittance=[], 
                     normalized_surface_impedance=[], surface_impedance=[], rmk1=[],**kwargs):
        
        if "parameters" in kwargs and "type" in kwargs:
            
            if hasattr(self, "frequencies") != True:
                raise ValueError("Algorithm frequencies are not defined yet.")
                
            material = Material(freq_vec=self.frequencies.freq_vec, rho0=self.rho0, c0=self.c0)

            if type == "porous"
                self.materials.append(material.porous(parameters))

            if type == "porous with air cavity"
                self.materials.append(material.porous_with_air_cavity(parameters))
                
            if type == "perforated panel"
                self.materials.append(material.perforated_panel(parameters))
                
            if type == "microperforated panel"
                self.materials.append(material.microperforated_panel(parameters))

        else:
            material = Material(normal_inidence_alpha, statistical_alpha, octave_bands_statistical_alpha, octave_bands, 
                                third_octave_bands_statistical_alpha, third_octave_bands, admittance, 
                                normalized_surface_impedance, surface_impedance, freq_vec=self.frequencies.freq_vec, rmk1, rho0=self.rho0, c0=self.c0)
                                       
            if material.admittance == 0 and material.normalized_surface_impedance == 0 and material.surface_impedance == 0:
                meterial.alpha_from_impedance()
            
            self.materials.append(material)
            
    
    def list_materials(self):
        
        for material in self.materials
            print(material)
