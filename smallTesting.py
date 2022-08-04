import pandas as pd
import numpy as np
import pickle 
import glob, os, sys
import time, copy
from datetime import datetime
import hdf5storage
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm 
from scipy.ndimage import median_filter
from matplotlib import rc
rc('animation', html='jshtml')
from pythonlib.dataset.dataset import Dataset
from pythonlib.dataset.dataset_preprocess.general import preprocessDat
import motionmapperpy as mmpy
from pythonlib.tools.expttools import makeTimeStamp, writeDictToYaml
# function to get data in the form of array dataTotal
def get_dataTotal(D):
	# x is the trial index there are 5125 trials so x < 5125
	x = 0
	# y is the stroke index- we dont know how many strokes r there
	y = 0
	# i is the index of one time frame of one stroke, 
	# this index also varies for each trial and stroke
	i = 0
	#ldataOneStroke = np.array([])
	#dataTotal = np.empty([150,1])
	ldataOneStroke = []
	ldataOneTrial = []
	ldataTotal = []
	#len(D.Dat)
	x=0
	for x in range(len(D.Dat)):
	    y=0
	    ldataOneTrial = []
	    for y in range(len(D.Dat.iloc[x]["strokes_beh"])):
	        #print(len(D.Dat.iloc[x]["strokes_beh"]))
	        i=0
	        ldataOneStroke = []
	        for i in range(i,len(D.Dat.iloc[x]["strokes_beh"][y])):
	            temp = D.Dat.iloc[x]["strokes_beh"][y][i][0:2]
	            ldataOneStroke.append(temp)
	        #print(len(ldataOneStroke))
	        dataOneStroke = np.array(ldataOneStroke)
	        #print(dataOneStroke)
	        #ldataOneTrial = np.append(ldataOneTrial,ldataOneStroke)
	        ldataOneTrial.append(dataOneStroke)
	    #print(len(ldataOneTrial))    
	    dataOneTrial = np.array(ldataOneTrial)
	    ldataTotal.append(dataOneTrial)
	    #ldataTotal = np.append(ldataTotal,ldataOneTrial)
	    #dataOneTrial = np.vstack((dataOneTrial, dataOneStroke))
	dataTotal = np.array(ldataTotal)
	return dataTotal
# function to get list of strokes indexes
def get_strokeIndexes(dataTotal):
	list_trialstroke = []
	for trial, x in enumerate(dataTotal):
	    #print(len(x))
	    for strokenum, xx in enumerate(x):
	        list_trialstroke.append((trial, strokenum))
	return list_trialstroke
# function to get list of 
# num should not be more than 5125
def get_strokes(dataTotal,num):
	#ndata = len(dataTotal)
	ndata = num
	lindependentStrokes = []
	for x in range(ndata):
	    for y in range(len(dataTotal[x])):
	        temp = dataTotal[x][y]
	        lindependentStrokes.append(temp)
	#independentStrokes = np.array(lindependentStrokes)
	return lindependentStrokes
# function to run subsampled tsne
def sub_tsne(parameters):
	mmpy.subsampled_tsne_from_projections(parameters, parameters.projectPathNots)
# function to run tsne for all data
def total_tsne(parameters):
	#tsne takes 19 mins
	tall = time.time()
	import h5py
	tfolder = parameters.projectPath+'/%s/'%parameters.method
	tfolderLoading = parameters.projectPathNots+'/%s/'%parameters.method
	# Loading training data
	with h5py.File(tfolderLoading + 'training_data.mat', 'r') as hfile:
		trainingSetData = hfile['trainingSetData'][:].T
	# Loading training embedding
	with h5py.File(tfolderLoading+ 'training_embedding.mat', 'r') as hfile:
	    trainingEmbedding= hfile['trainingEmbedding'][:].T
	if parameters.method == 'TSNE':
	    zValstr = 'zVals' 
	else:
	    zValstr = 'uVals'
	projectionFiles = glob.glob(parameters.projectPathNots+'/Projections/*notpca.mat')

	for i in range(len(projectionFiles)):
		print('Finding Embeddings')
        t1 = time.time()
        print('%i/%i : %s'%(i+1,len(projectionFiles), projectionFiles[i]))
        # Skip if embeddings already found.
        if os.path.exists(projectionFiles[i][:-4] +'_%s.mat'%(zValstr)):
        	print('Already done. Skipping.\n')
            continue

        # load projections for a dataset
        #modifying adding np.array
        projections = np.array(hdf5storage.loadmat(projectionFiles[i])['projections'])
        #print("e")

        # Find Embeddings
        #zValues, outputStatistics, wvlets = mmpy.findEmbeddings(projections,trainingSetData,trainingEmbedding,parameters)
        zValues, outputStatistics = mmpy.findEmbeddings(projections, trainingSetData, trainingEmbedding, parameters)
        #print("f")

        # Save embeddings
        hdf5storage.write(data = {'zValues':zValues}, path = '/', truncate_existing = True, filename = parameters.projectPath+'/Projections/'+'_%s.mat'%(zValstr), store_python_metadata = False, matlab_compatible = True)
        #print("g")

        '''
        # stop wavelets
        # Saving wlets from total data
        hdf5storage.write(data = {'wavelets':wlets}, path = '/', truncate_existing = True, filename = parameters.projectPath+'/Projections/'+'_%s.mat'%('wlets'), store_python_metadata = False, matlab_compatible = True)
        print("h")
        '''

        # Save output statistics
        with open(parameters.projectPath+'/Projections/'+ '_%s_outputStatistics.pkl'%(zValstr), 'wb') as hfile:
            pickle.dump(outputStatistics, hfile)
        #print("i")

        del zValues,projections,outputStatistics
	print('All Embeddings Saved in %i seconds!'%(time.time()-tall))


if __name__=="__main__":
	ts = makeTimeStamp()
	projectPath = f'content/trial1_mmpy{ts}'
	# This creates a project directory structure which will be used to store all motionmappery pipeline
	# related data in one place.
	mmpy.createProjectDirectory(projectPath)
	#projectPathNots = 'content/trial1_mmpy'
	projectPathNots = projectPath
	#%matplotlib inline
	expt = 'gridlinecircle'
	path_list = [
	    "data_030421/Pancho-gridlinecircle-baseline-210824_002447",
	    "data_030421/Pancho-gridlinecircle-circletoline-210828_100027",
	    "data_030421/Pancho-gridlinecircle-linetocircle-210828_100152",
	    "data_030421/Pancho-gridlinecircle-lolli-210903_094051",
	]
	append_list = None
	parameters = mmpy.setRunParameters()
	# %%%%%%% PARAMETERS TO CHANGE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	# These need to be revised everytime you are working with a new dataset. #
	parameters.projectPath = projectPath #% Full path to the project directory.
	parameters.projectPathNots = projectPathNots
	#parameters.method = 'UMAP' #% We can choose between 'TSNE' or 'UMAP'
	parameters.minF = 1        #% Minimum frequency for Morlet Wavelet Transform
	parameters.maxF = 25       #% Maximum frequency for Morlet Wavelet Transform,
	                           #% usually equal to the Nyquist frequency for your
	                           #% measurements.
	parameters.samplingFreq = 100    #% Sampling frequency (or FPS) of data.
	parameters.numPeriods = 25       #% No. of dyadically spaced frequencies to
	                                 #% calculate between minF and maxF.
	comps_above_thresh = 2
	parameters.pcaModes = comps_above_thresh #% Number of low-d features.
	parameters.numProcessors = -1     #% No. of processor to use when parallel
	                                 #% processing for wavelet calculation (if not using GPU)  
	                                 #% and for re-embedding. -1 to use all cores 
	                                 #% available.
	parameters.useGPU = -1           #% GPU to use for wavelet calculation, 
	                                 #% set to -1 if GPU not present.
	parameters.training_numPoints = 10000   #% Number of points in mini-trainings.

	# %%%%% NO NEED TO CHANGE THESE UNLESS MEMORY ERRORS OCCUR %%%%%%%%%%
	parameters.trainingSetSize = 5000  #% Total number of training set points to find. 
	                                 #% Increase or decrease based on
	                                 #% available RAM. For reference, 36k is a 
	                                 #% good number with 64GB RAM.
	parameters.embedding_batchSize = 30000  #% Lower this if you get a memory error when 
	                                        #% re-embedding points on a learned map.
	# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	# %%%%%%% tSNE parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	#% can be 'barnes_hut' or 'exact'. We'll use barnes_hut for this tutorial for speed.
	parameters.tSNE_method = 'barnes_hut' 
	# %2^H (H is the transition entropy)
	parameters.perplexity = 32
	# %number of neigbors to use when re-embedding
	parameters.maxNeighbors = 200
	# %local neighborhood definition in training set creation
	parameters.kdNeighbors = 5
	# %t-SNE training set perplexity
	parameters.training_perplexity = 20
	writeDictToYaml(parameters, projectPath+'/parameters.yaml')
	D = Dataset(path_list, append_list)
	dataTotal = get_dataTotal(D)
	get_strokeIndexes(dataTotal)
	projections = get_strokes(dataTotal,20)
	print('%s/Projections/test_monkey_notpca.mat'%(projectPath))
	hdf5storage.savemat('%s/Projections/test_monkey_notpca.mat'%(projectPath), {"projections" : projections})
	projectionFiles = glob.glob(parameters.projectPath+'/Projections/*test_monkey_notPCA.mat')
	sub_tsne(parameters)
	total_tsne(parameters)