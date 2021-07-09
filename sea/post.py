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

    try:
        import pickle

        file_to_read = open(reference_path, "rb")
        ref = pickle.load(file_to_read)
        file_to_read.close()

        compared_list = []
        for path in compared_paths:
            file_to_read = open(path, "rb")
            compared_list.append(pickle.load(file_to_read))
            file_to_read.close()

        sources = np.array(sources)
        receivers = np.array(receivers)

        if sources.size == 0:
            sources = np.arange(len(ref.sources))
        if receivers.size == 0:
            receivers = np.arange(len(ref.receivers))

        mac_list = []
        for source in sources:
            for compared in compared_list:

                reference = []
                to_be_compared = []

                for receiver in receivers:

                    for s_i, s in enumerate(compared.sources):
                        for r_i, r in enumerate(compared.receivers):
                            if s_i == source and r_i == receiver:
                                reference.append(ref.total_pressure[s_i*len(ref.receivers)+r_i : : len(ref.sources)*len(ref.receivers)])
                                to_be_compared.append(compared.total_pressure[s_i*len(compared.receivers)+r_i : : len(compared.sources)*len(compared.receivers)])

                mac = []

                for fi, f in enumerate(ref.simulated_freqs):

                    ref_aux = np.array(reference)[:,fi]
                    to_be_compared_aux = np.array(to_be_compared)[:,fi]
                    mac.append((abs(np.matmul(ref_aux.conj(), to_be_compared_aux.transpose()))**2) / np.real((np.matmul(ref_aux.conj(), ref_aux.transpose())) * np.matmul(to_be_compared_aux.conj(), to_be_compared_aux.transpose())))

                mac_list.append(mac)

        mac_list = np.array(mac_list)
    
    
    except:

        reference = reference_path
        
        compared_list = []
        for path in compared_paths:
            file_to_read = open(path, "rb")
            compared_list.append(pickle.load(file_to_read))
            file_to_read.close()

        sources = np.array(sources)
        receivers = np.array(receivers)

        if sources.size == 0:
            sources = np.arange(1)
        if receivers.size == 0:
            receivers = np.arange(reference.shape[0])

        mac_list = []
        for source in sources:
            for compared in compared_list:

                reference = []
                to_be_compared = []

                for receiver in receivers:

                    for s_i, s in enumerate(compared.sources):
                        for r_i, r in enumerate(compared.receivers):
                            if s_i == source and r_i == receiver:
                                to_be_compared.append(compared.total_pressure[s_i*len(compared.receivers)+r_i : : len(compared.sources)*len(compared.receivers)])

                print(to_be_compared)
                mac = []

                for fi, f in enumerate(compared.simulated_freqs):

                    ref_aux = reference[:,fi]
                    to_be_compared_aux = np.array(to_be_compared)[:,fi]
                    mac.append((abs(np.matmul(ref_aux.conj(), to_be_compared_aux.transpose()))**2) / np.real((np.matmul(ref_aux.conj(), ref_aux.transpose())) * np.matmul(to_be_compared_aux.conj(), to_be_compared_aux.transpose())))

                mac_list.append(mac)

        mac_list = np.array(mac_list)
    
    return mac_list
