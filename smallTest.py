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
from pythonlib.dataset.dataset import Dataset
from pythonlib.dataset.dataset_preprocess.general import preprocessDat
import motionmapperpy as mmpy
from pythonlib.tools.expttools import makeTimeStamp, writeDictToYaml
import extraFunctions as ef
from IPython.display import Image
import random
from pythonlib.drawmodel import strokePlots
import math
import matplotlib.image as mpimg

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
	dataTotal = ef.get_dataTotal(D)
	list_trialstroke = ef.get_strokeIndexes(dataTotal)

	projections = ef.get_strokes(dataTotal,15)
	print('%s/Projections/test_monkey_notpca.mat'%(projectPath))

	hdf5storage.savemat('%s/Projections/test_monkey_notpca.mat'%(projectPath), {"projections" : projections})

	projectionFiles = glob.glob(parameters.projectPath+'/Projections/*test_monkey_notPCA.mat')

	ef.sub_tsne(parameters)

	trainy, subdensity, subfig = ef.save_sub_tsne(parameters)

	ef.total_tsne(parameters)

	zValstr = 'zVals'

	zValues = ef.get_zValues_array(parameters, zValstr)

	density, totfig = ef.save_total_tsne(parameters, zValues)

	wlets = ef.get_wavelets(parameters)

	densityandstrokes = ef.save_density_and_strokes(parameters, density, zValues, 1)


	ef.save_watershed(parameters, 1.0, 10)

	watershedandstrokes = ef.save_watershed_and_strokes(parameters, 10, 1)

	ef.many_strokes_in_all_regions(zValues, 10, 10, 5)













