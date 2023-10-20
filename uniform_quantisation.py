""" Created: 13.10.2023  \\  Updated: 13.10.2023  \\   Author: Robert Sales """

#==============================================================================

import os, json, glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#==============================================================================

'''
THIS CODE IS INCOMPLETE. IT NEEDS TO BE IMPLEMENTED SUCH THAT A LIST OF WEIGHTS
CAN BE ENCODED (WITH INFORMATION ABOUT SHAPE AND ARCHITECTURE) AND SUBSEQUENTLY
DECODED FOR APPLYING TO THE NETWORK AGAIN.

THE LIST OF MATRICES NEEDS TO BE ENCODED TO THE SAME BITSTREAM OTHERWISE IT HAS
A RISK OF PADDING EACH INDIVIDUAL BITSTREAM TO BE 64*N LONG WHICH IS A WASTE OF
DISK SPACE.
'''

#==============================================================================
# Turn a list of matrices of parameters into a list of matrices of bin indices.

# Stores one global maximum and minimum -> quantises to uniform bins in between

def UniformQuantisationTogether(list_of_parameters,bits_per_value):
    
    list_of_parameters_indices = []
    list_of_parameters_range = []
    
    minimum = min([parameters.min() for parameters in list_of_parameters])
    maximum = min([parameters.max() for parameters in list_of_parameters])
    
    bins = np.linspace(start=minimum,stop=maximum,num=((2**bits_per_value)+1))
    bin_centres = ((bins[1:] + bins[:-1])/2)
        
    for parameters in list_of_parameters:

        parameters_indices = (np.digitize(x=parameters,bins=bins,right=False) - 1)
        
        index_overflows = (parameters_indices==(2**bits_per_value))
        
        parameters_indices[index_overflows] = (parameters_indices[index_overflows] - 1) 
        
        list_of_parameters_indices.append(parameters_indices)
        
    ##
    
    list_of_parameters_range.append((minimum,maximum))
    
    return list_of_parameters_indices,list_of_parameters_range


#==============================================================================
# Turn a list of matrices of parameters into a list of matrices of bin indices.

# Lists all the local maxima and minima -> quantises to uniform bins in between

def UniformQuantisationSeparate(list_of_parameters,bits_per_value):
    
    list_of_parameters_indices = []
    list_of_parameters_range = []
    
    for parameters in list_of_parameters:
        
        minimum = parameters.min()
        maximum = parameters.max()
        
        bins = np.linspace(start=minimum,stop=maximum,num=((2**bits_per_value)+1))
        bin_centres = ((bins[1:] + bins[:-1])/2)

        parameters_indices = (np.digitize(x=parameters,bins=bins,right=False) - 1)
        
        index_overflows = (parameters_indices==(2**bits_per_value))
        
        parameters_indices[index_overflows] = (parameters_indices[index_overflows] - 1) 
        
        list_of_parameters_indices.append(parameters_indices)
        
        list_of_parameters_range.append((minimum,maximum))
        
    ##
    
    return list_of_parameters_indices,list_of_parameters_range

#==============================================================================
# Turn a list of matrices of parameters into a list of matrices of bin indices.

# Note: there must be more samples than there are clusters, otherwise it fails.

def KMeansQuantisationTogether(list_of_parameters,bits_per_value):
    
    from sklearn.cluster import KMeans
    
    list_of_parameters_indices = []
    list_of_cluster_centres = []
        
    parameters = np.concatenate([x.reshape(-1,1) for x in list_of_parameters],axis=0)
            
    kmeans = KMeans(n_clusters=(2**bits_per_value),n_init="auto").fit(parameters)
    
    list_of_parameters_indices.append(kmeans.labels_.squeeze().tolist())

    list_of_cluster_centres.append(kmeans.cluster_centers_.squeeze().tolist())
    
    return list_of_parameters_indices,list_of_cluster_centres

#==============================================================================
# Turn a list of matrices of parameters into a list of matrices of bin indices.

# Note: there must be more samples than there are clusters, otherwise it fails.

def KMeansQuantisationSeparate(list_of_parameters,bits_per_value):
    
    from sklearn.cluster import KMeans
    
    list_of_parameters_indices = []
    list_of_cluster_centres = []
    
    for parameters in list_of_parameters:
    
        parameters = parameters.reshape(-1,1)
        
        kmeans = KMeans(n_clusters=(2**bits_per_value),n_init="auto").fit(parameters)
    
        list_of_parameters_indices.append(kmeans.labels_.squeeze().tolist())

        list_of_cluster_centres.append(kmeans.cluster_centers_.squeeze().tolist())
    
    return list_of_parameters_indices,list_of_cluster_centres

#==============================================================================

def SVDParameterEncoding(list_of_parameters,target_compression_ratio):

        for parameters in list_of_parameters:
            
            U,S,V = np.linalg.svd(parameters)
            
            target_size = parameters.size / target_compression_ratio

       size_of_svd = R*(M+N+1)         


#==============================================================================



'''        
        parameters_indices = np.ravel(parameters_indices,order="C")
                        
        parameters_indices_as_bytes = SupremeMinimumEncoding(indices=parameters_indices,bits_per_value=bits_per_value)
        print(len(parameters_indices_as_bytes))
        
        # parameters_indices_as_bytes = VariableBlockLengthEncoding(indices=parameters_indices,bits_per_value=bits_per_value)
        # print(len(parameters_indices_as_bytes))
        
        quantised_parameters = bin_centres[parameters_indices].reshape(parameters.shape,order="C").astype("float32")
        
        list_of_quantised_parameters.append(quantised_parameters)
    
    ##
    
    return list_of_quantised_parameters
'''    
#==============================================================================
# Note: in this format, supreme minimum encoding is mostly just a baseline case
# to compare to variable block length encoding. This is because bits_per_value 
# is specificied in the quantisation step, rather than found from the suprenum.

def SupremeMinimumEncoding(list_of_indices,bits_per_value):
    
    # Something to encode meta-information
    
    indices = list_of_indices[0].ravel()
    
    bitstring = ""
            
    for index in indices:
        index_as_binary = format(index,f'0{bits_per_value}b')
        bitstring = bitstring + index_as_binary
    ##
            
    bitstring = bitstring + ("0"* (64 - len(bitstring)%64))
    
    number_of_uint64s = len(bitstring)//64
    
    indices_as_uint64 = np.empty(shape=number_of_uint64s,dtype="uint64")
    
    for i in range(number_of_uint64s):  
        indices_as_uint64[i] = int(bitstring[:64],2)
        bitstring = bitstring[64:]
    ##

    indices_as_bytestring = indices_as_uint64.tobytes(order="C")

    return indices_as_bytestring    

#==============================================================================

def SupremeMinimumDecoding(indices_as_bytestring,bits_per_value):
    
    # Something to decode meta-information
    
    bitstring = ""
    
    indices_as_uint64 = np.frombuffer(indices_as_bytestring,dtype="uint64")
    
    for i in indices_as_uint64: bitstring = bitstring + format(i,f'0{64}b')
    
    # for index in 
        
    
    pass
    
#==============================================================================
# Note: bits_for_binary can be reduced by 1 since the lowest bits_for_binary is
# 1 instead of 0, yet b-bits ordinarily represents [0,2^b -1]. This can instead 
# be [1,2^b] (by subtracting 1) thus reducing the overall bitstring length.

def VariableBlockLengthEncoding(indices,bits_per_value):
    
    bitstring = ""
    
    bits_per_block = int(np.ceil(np.log2(bits_per_value)))
            
    for index in indices:
        index_as_binary = format(index,'b')
        length_of_block = format(len(index_as_binary),f'0{bits_per_block}b')
        bitstring = bitstring + length_of_block + index_as_binary
    ##
    
    bitstring = bitstring + ("0"* (64 - len(bitstring)%64))
    
    number_of_uint64s = len(bitstring)//64
    
    indices_as_uint64 = np.empty(shape=number_of_uint64s,dtype="uint64")
    
    for i in range(number_of_uint64s):  
        indices_as_uint64[i] = int(bitstring[:64],2)
        bitstring = bitstring[64:]
    ##

    indices_as_bytestring = indices_as_uint64.tobytes(order="C")

    return indices_as_bytestring  

#==============================================================================

list_of_parameters = [np.random.rand(100,100).astype("float32"),np.random.rand(100,100).astype("float32")]
bits_per_value = 10

# list_of_parameters_indices_separate,range_separate = UniformQuantisationSeparate(list_of_parameters=list_of_parameters,bits_per_value=bits_per_value)
# list_of_parameters_indices_together,range_together = UniformQuantisationTogether(list_of_parameters=list_of_parameters,bits_per_value=bits_per_value)

list_of_parameters_indices_separate,range_separate = KMeansQuantisationSeparate(list_of_parameters=list_of_parameters,bits_per_value=bits_per_value)
# list_of_parameters_indices_together,range_together = KMeansQuantisationTogether(list_of_parameters=list_of_parameters,bits_per_value=bits_per_value)


# indices_as_bytestring = SupremeMinimumEncoding(list_of_indices=list_of_parameters_indices,bits_per_value=bits_per_value)
