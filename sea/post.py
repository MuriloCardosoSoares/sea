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
    
    compared_list = []
    for i in enumerate(compared_paths):
        file_to_read = open(compared_paths[i], "rb")
        compared.append(pickle.load(file_to_read))
        file_to_read.close()
    
    mac_list = []
    for source in sources:
        for compared in compared_list:
 
            reference = []
            to_be_compared = []
            
            for receiver in receivers:

                for s_i, s in enumerate(compared[0].sources):
                    for r_i, r in enumerate(compared[0].receivers):
                        if s_i == source and r_i == receiver:
                            reference.append([ref.total_pressure[s_i*len(ref.receivers)+r_i : : len(ref.sources)*len(ref.receivers)]])
                            to_be_compared.append([compared.total_pressure[s_i*len(compared.receivers)+r_i : : len(compared.sources)*len(compared.receivers)]])


            mac = []

            for fi, f in enumerate(reference[0].frequencies.freq_vec):

                ref_aux = np.array(reference)[:,fi]
                to_be_compared_aux = to_be_compared[i*(num_receivers) : i*(num_receivers)+num_receivers, fi]
                mac.append((abs(np.matmul(ref_aux.conj(), to_be_compared_aux.transpose()))**2) / np.real((np.matmul(ref_aux.conj(), ref_aux.transpose())) * np.matmul(to_be_compared_aux.conj(), to_be_compared_aux.transpose())))

            mac_list.append([mac])
'''              
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
'''
    mac_list = np.array([mac_list])
    
    return mac_list
