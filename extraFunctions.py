import hdf5storage
import numpy as np
import pandas as pd
import pickle 
import glob, os, sys
import time, copy
from datetime import datetime
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
import extraFunctions as ef
from IPython.display import Image
import random
from pythonlib.drawmodel import strokePlots
import math
import matplotlib.image as mpimg
import h5py

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

#function to save density maps from sub_tsne
def save_sub_tsne(parameters):
    trainy = hdf5storage.loadmat('%s/%s/training_embedding.mat'%(parameters.projectPath, parameters.method))['trainingEmbedding']
    m = np.abs(trainy).max()
    print(m)

    sigma=2.0
    _, xx, density = mmpy.findPointDensity(trainy, sigma, 511, [-m-20, m+20])

    fig, axes = plt.subplots(1, 2, figsize=(12,6))
    axes[0].scatter(trainy[:,0], trainy[:,1], marker='.', c=np.arange(trainy.shape[0]), s=1)
    axes[0].set_xlim([-m-20, m+20])
    axes[0].set_ylim([-m-20, m+20])
    axes[1].imshow(density, cmap=mmpy.gencmap(), extent=(xx[0], xx[-1], xx[0], xx[-1]), origin='lower')

    fig.savefig('%s/%s/density_sub_tsne.png'%(parameters.projectPath, parameters.method))
    return trainy, density, fig

# function to run tsne for all data
def total_tsne(parameters):
    #tsne takes 19 mins
    tall = time.time()

    tfolder = parameters.projectPath+'/%s/'%parameters.method

    tfolderLoading = parameters.projectPathNots+'/%s/'%parameters.method

    # Loading training data
    with h5py.File(tfolderLoading + 'training_data.mat', 'r') as hfile:
        trainingSetData = hfile['trainingSetData'][:].T
    print("a")
    # Loading training embedding
    with h5py.File(tfolderLoading+ 'training_embedding.mat', 'r') as hfile:
        trainingEmbedding= hfile['trainingEmbedding'][:].T
    print("b")
    if parameters.method == 'TSNE':
        zValstr = 'zVals'
        print("c")
 
    else:
        zValstr = 'uVals'



    projectionFiles = glob.glob(parameters.projectPathNots+'/Projections/*notpca.mat')
    print("d")

    #projectionFilestoSave =  glob.glob(parameters.projectPath+'/Projections/*notpca.mat')



    for i in range(len(projectionFiles)):
        print('Finding Embeddings')
        t1 = time.time()
        #print('%i/%i : %s'%(i+1,len(projectionFiles), projectionFiles[i]))


        # Skip if embeddings already found.
        if os.path.exists(projectionFiles[i][:-4] +'_%s.mat'%(zValstr)):
            print('Already done. Skipping.\n')
            continue

        # load projections for a dataset
        #modifying adding np.array
        projections = np.array(hdf5storage.loadmat(projectionFiles[i])['projections'])
        print("e")

        # Find Embeddings
        #zValues, outputStatistics, wvlets = mmpy.findEmbeddings(projections,trainingSetData,trainingEmbedding,parameters)
        zValues, outputStatistics = mmpy.findEmbeddings(projections, trainingSetData, trainingEmbedding, parameters)
        print("f")

        # Save embeddings
        hdf5storage.write(data = {'zValues':zValues}, path = '/', truncate_existing = True, filename = parameters.projectPath+'/Projections/'+'_%s.mat'%(zValstr), store_python_metadata = False, matlab_compatible = True)
        print("g")

        '''
        # stop wavelets
        # Saving wlets from total data
        hdf5storage.write(data = {'wavelets':wlets}, path = '/', truncate_existing = True, filename = parameters.projectPath+'/Projections/'+'_%s.mat'%('wlets'), store_python_metadata = False, matlab_compatible = True)
        print("h")
        '''

        # Save output statistics
        with open(parameters.projectPath+'/Projections/'+ '_%s_outputStatistics.pkl'%(zValstr), 'wb') as hfile:
            pickle.dump(outputStatistics, hfile)
        print("i")


        del zValues,projections,outputStatistics


    print('All Embeddings Saved in %i seconds!'%(time.time()-tall))


#function to take zValues array and make an accessible array
def get_zValues_array(parameters, zValstr):
    zVals = hdf5storage.loadmat(glob.glob(parameters.projectPath+'/Projections/*_%s.mat'%(zValstr))[0])['zValues'][0]

    zValues = []
    for i in range(len(zVals)):
        item = zVals[i][0]
        zValues.append(item)
    zValues = np.array(zValues)
    return zValues

# function to save density of total tsne
def save_total_tsne(parameters, ally):

    #ally = get_zValues_array(parameters, zValstr)

    m = np.abs(ally).max()

    sigma=2.0
    _, xx, density = mmpy.findPointDensity(ally, sigma, 511, [-m-20, m+20])


    fig, axes = plt.subplots(1, 2, figsize=(12,6))
    axes[0].scatter(ally[:,0], ally[:,1], marker='.', c=np.arange(ally.shape[0]), s=1)
    axes[0].set_xlim([-m-20, m+20])
    axes[0].set_ylim([-m-20, m+20])

    axes[1].imshow(density, cmap=mmpy.gencmap(), extent=(xx[0], xx[-1], xx[0], xx[-1]), origin='lower')
    
    fig.savefig('%s/%s/density_total_tsne.png'%(parameters.projectPath, parameters.method))
    return density, fig

# function to get wavelets from file
def get_wavelets(parameters):
    #used to be hdf5storage.loadmat(glob.glob(parameters.projectPath+'/Projections/*_%s.mat'%('wlets'))[0])['wavelets']
    wvlets = hdf5storage.loadmat(glob.glob(parameters.projectPath+'/Projections/*_%s.mat'%('wvlets'))[0])['wvlets'][0]
    
    wavelets = []
    for i in range(len(wvlets)):
        onewvlet = []
        for j in range(len(wvlets[i][0])):
            item = wvlets[i][0][j]
            onewvlet.append(item)
            nponewvlet = np.array(onewvlet)
        wavelets.append(nponewvlet)
    npwavelets = np.array(wavelets)
    return npwavelets

# function to divide the strokes in areas
def divide_strokes(zValues, num, origin):
    #xs = zValues[:,0]
    #arrx = np.linspace(xs.min(), xs.max(), num)
    #print(origin)
    arrx = np.linspace(origin, -origin, num)
    #print(arrx[0])
    #ys = zValues[:,1]
    #arry = np.linspace(ys.min(), ys.max(), num)
    arry = np.linspace(-origin, origin, num)
    #print(arry)
    return arrx, arry

# function to choose an stroke from the index list and return it
def choose_stroke(lsindex):
    import random
    r = random.randint(0, len(lsindex)-1)
    return lsindex[r]

# function to return array of indexes of the strokes contained in a certain grid of density
def strokes_in_area(zValues,arrx, arry, nx, ny):
    #print("ZVALUES:")
    #print(zValues)
    lsindex = []
    #nx = nare
    #ny = -1 -narea
    upx = arrx[nx+1]
    #print("upx",upx)
    lwx = arrx[nx]
    #print("lwx",lwx)
    upy = arry[ny]
    #print("upy", upy)
    lwy = arry[ny+1]
    #print("lwy", lwy)
 
    for i in range(len(zValues)):
        if (zValues[i][0]<=upx and zValues[i][0]>lwx and zValues[i][1]<=upy and zValues[i][1]>lwy):
            #print(i)
            #print(narea)
            #print("arrx",arrx[nx+1])
            #print("arry", arry[nx])
            #print(zValues[i])
            lsindex.append(zValues[i])
    return lsindex

# function to plot a single stroke given the index and ax in the subplot
def plot_stroke(indx, ax):
    #print(indx)
    trial, strokenum = list_trialstroke[indx]
    g = D.Dat.iloc[trial]["strokes_beh"][strokenum]
    strokePlots.plotDatStrokes([g],ax, clean_ordered= True)

# function to get index given the array stroke
def get_sindex(zValues, arrst):
    indx = np.where(np.all(zValues==arrst, axis=1))
    if len(indx[0]>1):
        return indx[random.randint(0, len(indx)-1)][0]
    else:
        return indx[0]

# function to plot random strokes in a grid
def save_density_and_strokes(parameters, density, zValues, number):
    num = str(number)

    uplim = round((xx[-1]),-1)

    rows = int(uplim/10)

    dcol = int(round(rows/3))

    cols = rows +dcol

    fig, axes = plt.subplots(rows, cols, figsize=(40,10))

    ax00 = plt.subplot2grid((rows,cols),(0,0),rowspan=rows, colspan=dcol)

    ax00.imshow(density, cmap=mmpy.gencmap(), extent=(xx[0],xx[-1],xx[0], xx[-1]), origin='lower')

    arrx, arry = divide_strokes(zValues, rows+1, xx[0])

    ax00.set_xticks(arrx)
    ax00.set_yticks(arry)

    ax00.grid()

    #keep calling functions to plot stuff
    for r in range(rows):
        for c in range(dcol, cols):
            ax = plt.subplot2grid((rows, cols,), (r,c))
            
            ls = strokes_in_area(zValues, arrx, arry, c-dcol, r)
            if (len(ls)>0):
                st = choose_stroke(ls)
                idx = get_sindex(zValues, st)
                plot_stroke(idx, ax)

    plt.tight_layout()  
    plt.show()
    fig.savefig('%s/%s/density_and_strokes%s.png'%(parameters.projectPath, parameters.method, num))
    return fig

# function to save watershed
def save_watershed(parameters, startsigma, minimum_regions):
    #modifying startsigma because 4.2 was too high
    #modifying minimum_regions because 50 was too high
    #startsigma = 1.0 if parameters.method == 'TSNE' else 2.0
    #minimum_regions = 50
    mmpy.findWatershedRegions(parameters, minimum_regions=minimum_regions, startsigma=startsigma, pThreshold=[0.33, 0.67],
                         saveplot=True, endident = '*_notpca.mat')

    Image(glob.glob('%s/%s/zWshed*.png'%(parameters.projectPath, parameters.method))[0])


#function to get wregions
def get_wregions(parameters, minimum_regions):
    wshedfile = hdf5storage.loadmat('%s/%s/zVals_wShed_groups%s.mat'%(parameters.projectPath, parameters.method, minimum_regions))
    wregions = wshedfile["indexesWatershedRegions"][0]
    return wregions

# function to return array of indexes of the strokes contained in a certain grid of density
def wstrokes_in_area(zValues,wregions,nr):
    lsindex = []
    for i in range(len(zValues)):
        if (wregions[i] == nr):
            lsindex.append(zValues[i])
    return lsindex

# function to plot many random strokes for all regions
def save_watershed_and_strokes(parameters, minimum_regions, num):
    minimum_regions = str(minimum_regions)
    num = str(num)
    
    wregions = get_wregions(parameters, minimum_regions)

    nregions = wregions.max()

    # calculate how many columns and rows
    # 10 rows and add more columns as needed
    # figure out how to name plot

    rows = 5
    dcol = int(round(rows/1))
    cols = math.ceil(nregions/rows) + dcol

    fig, axes = plt.subplots(rows, cols, figsize=(20,10))


    ax00 = plt.subplot2grid((rows,cols),(0,0),rowspan=rows, colspan=dcol)

    ax00.imshow(mpimg.imread(glob.glob('%s/%s/zWshed*.png'%(parameters.projectPath, parameters.method))[0]))
    ax00.axis('off')


    for r in range(rows):
        for c in range(dcol, cols):
            nr = (c-dcol)*rows + r + 1
            ax = plt.subplot2grid((rows, cols), (r, c))
            ls = wstrokes_in_area(zValues, wregions, nr)
            if (len(ls)>0):
                st = choose_stroke(ls)
                idx = get_sindex(zValues, st)
                plot_stroke(idx, ax)
                ax.set_title('Region '+str(nr))
                #ax.title()
                
    plt.tight_layout()  
    plt.show()
    fig.savefig('%s/%s/%swatershed_and_strokes%s.png'%(parameters.projectPath, parameters.method, num, minimum_regions))
    return fig

# function to choose sample of strokes from the index list and return it
def choose_sample_strokes(lsindex, ni):
    if (ni > len(lsindex)):
        ni = len(lsindex)
    lsample = random.sample(lsindex, ni)
    return lsample


# function to save strokes for a wshed region
def many_strokes_in_region(zValues, wregions, nr, ns, minimum_regions, num):
    minimum_regions = str(minimum_regions)
    num = str(num)

    wregions = get_wregions(parameters, minimum_regions)

    ls = wstrokes_in_area(zValues, wregions, nr)
    
    if (len(ls)>0):
        lsample = choose_sample_strokes(ls, ns)

        nstrokes = len(lsample)

        rows = 5
        dcol = int(round(rows/1))
        cols = math.ceil(nstrokes/rows) + dcol

        fig, axes = plt.subplots(rows, cols, figsize=(20,10), sharex=True, sharey=True)


        ax00 = plt.subplot2grid((rows,cols),(0,0),rowspan=rows, colspan=dcol)

        ax00.imshow(mpimg.imread(glob.glob('%s/%s/zWshed*.png'%(parameters.projectPath, parameters.method))[0]))
        ax00.axis('off')
        nr = str(nr)
        #print(lsample)
        ct = 0
        for r in range(rows):
            for c in range(dcol, cols):
                ax = plt.subplot2grid((rows, cols), (r, c))
                ax.set_xlim([-300, 300])
                ax.set_ylim([-200,400])
                #ax = axes.flatten()[ct]
                s = (c-dcol)*rows + r
                if (len(lsample)>s):
                    if (len(lsample)>0):
                        #print(lsample[s])
                        st = lsample[s]
                    else:
                        st = lsample[0]
                    idx = get_sindex(zValues, st)
                    plot_stroke(idx, ax)
                    ax.set_title('Region '+nr)
                    #ax.title()
                    ct=ct+1

        plt.tight_layout()  
        plt.show()
        fig.savefig('%s/%s/%swshed_%s_and_strokes%s.png'%(parameters.projectPath, parameters.method, num, nr, minimum_regions))

# function to run for all wshed regions
def many_strokes_in_all_regions(zValues, ns, minimum_regions, num):
    wregions = get_wregions(parameters, minimum_regions)
    
    nregions = wregions.max()

    for nr in range(1, nregions+1):
        many_strokes_in_all_regions(zValues, wregions, nr, ns, minimum_regions, num)


