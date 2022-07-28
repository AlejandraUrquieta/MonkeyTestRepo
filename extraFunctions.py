import hdf5storage
import glob
import numpy as np


#function to take zValues array and make an accessible array
def get_zValues_array(parameters, zValstr):
    print("a")
    zVals = hdf5storage.loadmat(glob.glob(parameters.projectPath+'/Projections/*_%s.mat'%(zValstr))[0])['zValues'][0]
    print("ab")


    zValues = []
    for i in range(len(zVals)):
        print("ha")
        item = zVals[i][0]
        zValues.append(item)
    zValues = np.array(zValues)
    return zValues
def get_wavelets_and_freqs_array(parameters):
    wlets = hdf5storage.loadmat(glob.glob(parameters.projectPath+'/TSNE/*_%s.mat'%('data'))[0])['trainingSetData']
    freqs = hdf5storage.loadmat(glob.glob(parameters.projectPath+'/TSNE/*_%s.mat'%('amps'))[0])['trainingSetAmps']


    return wlets, freqs

