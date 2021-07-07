#Post-processing functions that might help

#def mac (file_path, configuration_names, ref=1, source_analisys=1):
def mac (reference_path, compared_paths, sources=[], receivers=[], plot=True):
  
    '''
    Computes the modal assurance criterion (MAC) for different configurations of a simulated room.
    
    ref -> path to the .pickle file that carries the simulation results that is gonna be used as reference to compute the MAC. It must be a string.
    compared -> paths to the .pickle files that carry the simulation results that is gonna be compared to the reference to compute the MAC. 
                It must be a list of strings. Example:
                    compared = ["file_1.pickle", "file_2.pickle", "file_3.pickle"]
    sources -> numbers of the sources that you are gonna use to compare. If not given, all the sources are gonna be used to compute the MAC. 
    receivers -> numbers of the receivers that you are gonna use to compare. If not given, all the receivers are gonna be used to compute the MAC.
    
    '''
    
    import numpy as np
    from matplotlib import pylab as plt
    import pickle

    file_to_read = open(reference_path, "rb")
    ref = pickle.load(file_to_read)
    file_to_read.close()
    
    compared = []
    for i in enumerate(compared_paths):
        file_to_read = open(compared_paths[i], "rb")
        compared.append(pickle.load(file_to_read))
        file_to_read.close()
    
    for source in enumerate(sources):
        to_be_compared = []
        reference = []
        for c_i in np.arange(num_materials_configurations):
            for s_i in np.arange(num_sources):
                for r_i in np.arange(num_receivers):
                    if c_i == ref-1 and s_i == source_analisys-1:
                        reference[i] = total_pressure_receivers[i,:]
                    if c_i != ref-1 and s_i == source_analisys-1:
                        to_be_compared[i] = total_pressure_receivers[i,:]

        trash, reference =  zip(*sorted(reference.items()))
        reference = np.array(reference)

        trash, to_be_compared =  zip(*sorted(to_be_compared.items()))
        to_be_compared = np.array(to_be_compared)


        mac = np.zeros((num_materials_configurations - 1, np.size(f_range)))
        for i in np.arange(num_materials_configurations - 1):

            for fi, f in enumerate(f_range):

                ref_aux = reference[:,fi]
                to_be_compared_aux = to_be_compared[i*(num_receivers) : i*(num_receivers)+num_receivers, fi]
                mac[i,fi] = (abs(np.matmul(ref_aux.conj(), to_be_compared_aux.transpose()))**2) / np.real((np.matmul(ref_aux.conj(), ref_aux.transpose())) * np.matmul(to_be_compared_aux.conj(), to_be_compared_aux.transpose()))

    for configuration_i in np.arange(num_materials_configurations-1): 
        
        plt.plot(f_range, mac[configuration_i, :])
    
    plt.title('MAC \n Reference $\Rightarrow$ %s' % configuration_names[ref-1])
    legend = [x for i,x in enumerate(configuration_names) if i!=ref-1]
    plt.legend(legend)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('MAC')
    plt.ylim((0,1))
    plt.grid()
    plt.savefig('%s_mac.pdf' % room_name)
    plt.show()

    # Save results to a Matlab-readable.h5 file:
    with h5py.File(file_path, "r+") as fh:

        fh.create_dataset("mac", data=mac)

        print ("Saved results to %s" % file_path)
