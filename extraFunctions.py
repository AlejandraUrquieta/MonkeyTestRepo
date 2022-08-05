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
#WORKING JUST FINE!
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
#WORKING JUST FINE!
def get_strokeIndexes(dataTotal):
    list_trialstroke = []
    for trial, x in enumerate(dataTotal):
        #print(len(x))
        for strokenum, xx in enumerate(x):
            list_trialstroke.append((trial, strokenum))
    return list_trialstroke

# function to get list of 
# num should not be more than 5125
#WORKING JUST FINE!
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
#WORKING JUST FINE!
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
    #print("d")

    #projectionFilestoSave =  glob.glob(parameters.projectPath+'/Projections/*notpca.mat')



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
        zValues, outputStatistics, wvlets = mmpy.findEmbeddings(projections, trainingSetData, trainingEmbedding, parameters)
        #print("f")

        # Save embeddings
        hdf5storage.write(data = {'zValues':zValues}, path = '/', truncate_existing = True, filename = parameters.projectPath+'/Projections/'+'_%s.mat'%(zValstr), store_python_metadata = False, matlab_compatible = True)
        #print("g")

        #trying wvlets
        hdf5storage.write(data = {'wvlets':wvlets}, path = '/', truncate_existing = True, filename = parameters.projectPath+'/Projections/'+'_%s.mat'%('wvlets'), store_python_metadata = False, matlab_compatible = True)

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


        del zValues,projections,outputStatistics, wvlets


    print('All Embeddings Saved in %i seconds!'%(time.time()-tall))


#function to take zValues array and make an accessible array
#WORKING JUST FINE!
def get_zValues_array(parameters, zValstr):
    zVals = hdf5storage.loadmat(glob.glob(parameters.projectPath+'/Projections/*_%s.mat'%(zValstr))[0])['zValues'][0]

    zValues = []
    for i in range(len(zVals)):
        item = zVals[i][0]
        zValues.append(item)
    zValues = np.array(zValues)
    return zValues

# function to save density of total tsne
#WORKING JUST FINE!
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
    return density, fig, xx

# function to divide the strokes in areas
#WORKING JUST FINE!
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
#WORKING JUST FINE!
def choose_stroke(lsindex):
    import random
    r = random.randint(0, len(lsindex)-1)
    return lsindex[r]

# function to return array of indexes of the strokes contained in a certain grid of density
#WORKING JUST FINE!
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
#WORKING JUST FINE!
def plot_stroke(D, list_trialstroke,indx, ax):
    #print(indx)
    trial, strokenum = list_trialstroke[indx]
    g = D.Dat.iloc[trial]["strokes_beh"][strokenum]
    strokePlots.plotDatStrokes([g],ax, clean_ordered= True)
    ax.set_title('StrI '+str(indx)+' t'+str(trial)+' n'+str(strokenum))
    ax.set_xlim([-300, 300])
    ax.set_ylim([-200,400])

# function to get index given the array stroke
#WORKING JUST FINE!
def get_sindex(zValues, arrst):
    indx = np.where(np.all(zValues==arrst, axis=1))
    if len(indx[0]>1):
        return indx[random.randint(0, len(indx)-1)][0]
    else:
        return indx[0]

# function to plot random strokes in a grid
#WORKING JUST FINE!
def save_density_and_strokes(parameters, density, xx, zValues, D, list_trialstroke,number):
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
                plot_stroke(D, list_trialstroke,idx, ax)

    plt.tight_layout()  
    #plt.show()
    fig.savefig('%s/%s/density_and_strokes_id%s.png'%(parameters.projectPath, parameters.method, num))
    return fig

# function to save watershed
#WORKING JUST FINE!
def save_watershed(parameters, startsigma, minimum_regions):
    #modifying startsigma because 4.2 was too high
    #modifying minimum_regions because 50 was too high
    #startsigma = 1.0 if parameters.method == 'TSNE' else 2.0
    #minimum_regions = 50
    mmpy.findWatershedRegions(parameters, minimum_regions=minimum_regions, startsigma=startsigma, pThreshold=[0.33, 0.67],
                         saveplot=True, endident = '*_notpca.mat')

    #something that should be called int he jupyter notebook
    #Image(glob.glob('%s/%s/zWshed*.png'%(parameters.projectPath, parameters.method))[0])


#function to get wregions
#WORKING JUST FINE!
def get_wregions(parameters, minimum_regions):
    wshedfile = hdf5storage.loadmat('%s/%s/zVals_wShed_groups%s.mat'%(parameters.projectPath, parameters.method, minimum_regions))
    wregions = wshedfile["indexesWatershedRegions"][0]
    return wregions

# function to return list of zValues (a,b) contained in a certain grid of density
#WORKING JUST FINE!
def wstrokes_in_area(zValues,wregions,nr):
    lszvals = []
    for i in range(len(zValues)):
        if (wregions[i] == nr):
            lszvals.append(zValues[i])
    return lszvals

# function to plot many random strokes for all watershed regions
#WORKING JUST FINE!
def save_watershed_and_strokes(parameters, D, list_trialstroke, zValues, minimum_regions, num):
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
            lzvals = wstrokes_in_area(zValues, wregions, nr)
            if (len(ls)>0):
                st = choose_stroke(lzvals)
                idx = get_sindex(zValues, st)
                plot_stroke(D, list_trialstroke, idx, ax)
                #ax.set_title('Region '+str(nr))
                ax.set_ylabel('Region'+str(nr))
                #ax.title()
                #ax.legend()
                
    plt.tight_layout()  
    #plt.show()
    fig.savefig('%s/%s/allwatershed_and_strokes_(mr%s)_id%s.png'%(parameters.projectPath, parameters.method, minimum_regions, num))
    return fig

# function to choose sample of strokes from the strokes list and return it
#WORKING JUST FINE!
def choose_sample_strokes(lsindex, ni):
    if (ni > len(lsindex)):
        ni = len(lsindex)
    lsample = random.sample(lsindex, ni)
    return lsample


# function to save strokes for a wshed region
#WORKING JUST FINE!
def many_strokes_in_region(parameters, D, list_trialstroke, zValues, wregions, nr, ns, minimum_regions, num):
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
        ax00.set_title('Region '+str(nr))
        #nr = str(nr)
        #print(lsample)
        ct = 0
        for r in range(rows):
            for c in range(dcol, cols):
                ax = plt.subplot2grid((rows, cols), (r, c))
                #ax.set_xlim([-300, 300])
                #ax.set_ylim([-200,400])
                #ax = axes.flatten()[ct]
                s = (c-dcol)*rows + r
                if (len(lsample)>s):
                    if (len(lsample)>0):
                        #print(lsample[s])
                        st = lsample[s]
                    else:
                        st = lsample[0]
                    idx = get_sindex(zValues, st)
                    plot_stroke(D, list_trialstroke,idx, ax)
                    #ax.set_title('Stroke Index '+str(idx))
                    #ax.title()
                    ct=ct+1

        plt.tight_layout()  
        #plt.show()
        fig.savefig('%s/%s/wshed_reg%s_and_strokes_(mr%s)_id%s.png'%(parameters.projectPath, parameters.method, nr, minimum_regions,num))

# function to run for all wshed regions
#WORKING JUST FINE!
def many_strokes_in_all_regions(parameters, D, list_trialstroke,zValues, ns, minimum_regions, num):
    wregions = get_wregions(parameters, minimum_regions)
    
    nregions = wregions.max()

    for nr in range(1, nregions+1):
        many_strokes_in_region(parameters, D, list_trialstroke,zValues, wregions, nr, ns, minimum_regions, num)

# function to get wavelets from file
#WOKRING JUST FINE!
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


# function to plot single wavelet
#WORKING JUST FINE
def plot_single_wavelet(wavelets, idx, ax):
    ax.plot(wavelets[idx].T, label=idx)

# function to save wavelet side to side with stroke
# lindx = [1,3,6,8,9,500,..]
#WORKING JUST FINE
def strokes_and_wavelets(D, zValues, list_trialstroke, lsample, parameters, wavelets, num, nr=None):
    num = str(num)
    rows = len(lsample)    
    #wlets = ef.get_wavelets(parameters)

    fig, axes = plt.subplots(rows, 2, figsize=(10, rows*5))
    
    for i in range(len(lsample)):
        ax1 = plt.subplot2grid((rows, 2), (i,0))
        ax2 = plt.subplot2grid((rows, 2), (i,1))
        idx = get_sindex(zValues,lsample[i])
        plot_stroke(D, list_trialstroke, idx, ax1)
        if nr!=None:
            ax1.set_ylabel('Region '+str(nr))
        plot_single_wavelet(wavelets, idx, ax2)
        ax2.legend()
        
    plt.tight_layout()
    #plt.show()
    if nr!=None:
        fig.savefig('%s/%s/wreg%s_strokes_and_wavelets_id%s.png'%(parameters.projectPath, parameters.method, nr, num))
    else:
        fig.savefig('%s/%s/strokes_and_wavelets_id%s.png'%(parameters.projectPath, parameters.method, num))


    
    #plot_wavelet(ax2, wlet)    


# function to plot many wavelets for a single watershed region
#WORKING JUST FINE
def many_wavelets_in_region(parameters, D, zValues, wavelets, nr, nw, minimum_regions, num):
    #print("is this working?")
    num = str(num)
    wregions = get_wregions(parameters, minimum_regions)
    #print(wregions)
    ls = wstrokes_in_area(zValues, wregions, nr)
    #print(ls)
    rows = 1
    cols = 2
    
    if (len(ls)>0):
        #print("is this working?loop")

        lsample = choose_sample_strokes(ls, nw)
        
        nstrokes = len(lsample)
        
        fig, axes = plt.subplots(rows,cols,figsize=(20,10))

        ax1 = plt.subplot2grid((rows, cols), (0,0))

        ax1.imshow(mpimg.imread(glob.glob('%s/%s/zWshed*.png'%(parameters.projectPath, parameters.method))[0]))
        
        ax1.axis('off')
        
        ax2 = plt.subplot2grid((rows,cols), (0,1))
        
        ax2.set_title('Region '+str(nr))
        
        for s in lsample:
            i = get_sindex(zValues, s)
            plot_single_wavelet(wavelets, i, ax2)
            ax2.legend()
        #print("is this working?2")
        plt.tight_layout()  
        #plt.show()
        minimum_regions = str(minimum_regions)
        fig.savefig('%s/%s/wshed_reg%s_and_wavelets(mr%s)_id%s.png'%(parameters.projectPath, parameters.method, nr, minimum_regions, num))

#function to plot many wavelets for all regions separately
#WORKING JUST FINE
def many_wavelets_in_all_regions(parameters, D, zValues, wavelets, nw, minimum_regions, num):
    wregions = get_wregions(parameters, minimum_regions)
    nregions = wregions.max()
    for nr in range(1, nregions+1):
        many_wavelets_in_region(parameters, D, zValues, wavelets, nr, nw, minimum_regions, num)


#function to plot many strokes and wavelets comparison for a region
#WORKING JUST FINE
def many_strokes_and_wavelets_in_region(parameters, D, list_trialstroke, zValues, wavelets, nr, nsw, num, wregions=None, minimum_regions=None):
    if len(wregions)!=0:
        wregions = wregions
    else:
        wregions = get_wregions(parameters, minimum_regions)
    lszvals = wstrokes_in_area(zValues, wregions, nr)
    if (len(lszvals)>0):
        lsample = choose_sample_strokes(lszvals, nsw)
        strokes_and_wavelets(D, zValues, list_trialstroke, lsample, parameters, wavelets, num, nr=nr)

#function to plot many strokes and wavelets comparison for all regions
#WORKING JUST FINE
def many_strokes_and_wavelets_in_all_regions(parameters, D, list_trialstroke, zValues, wavelets, nsw, minimum_regions, num):
    wregions = get_wregions(parameters, minimum_regions)
    nregions = wregions.max()
    
    for nr in range(1, nregions+1):
        many_strokes_and_wavelets_in_region(parameters, D, list_trialstroke, zValues, wavelets, nr, nsw, num,wregions=wregions)


# function to acess projections from the projectionFiles[i]
#WORKING!
def plot_xy_traj(st, idx, ax, list_trialstroke):
    trial, strokenum = list_trialstroke[idx]
    ax.plot(st, label=['x','y'])
    ax.set_title('StrI '+str(idx)+' t'+str(trial)+' n'+str(strokenum))
    
# function to save wavelet and stroke
#WORKING!
def trajectories_and_wavelets(parameters, projections, zValues,list_trialstroke, wavelets, lsample, num, nr=None):
    num = str(num)
    rows = len(lsample)
    fig, axes = plt.subplots(rows, 2,figsize=(10,rows*5))
    for i in range(len(lsample)):
        ax1 = plt.subplot2grid((rows, 2), (i, 0))
        ax2 = plt.subplot2grid((rows,2), (i, 1))
        idx = get_sindex(zValues, lsample[i])
        st = projections[idx]
        plot_xy_traj(st, idx, ax1, list_trialstroke)
        plot_single_wavelet(wavelets, idx, ax2)
        ax1.legend()
        ax2.legend()
        if nr!=None:
            nr = str(nr)
            ax2.set_title('Region '+nr)

    plt.tight_layout()
    #plt.show()
    if nr!=None:
        fig.savefig('%s/%s/wreg%s_trajs_and_wavelets_id%s.png'%(parameters.projectPath, parameters.method, nr, num))
    else:
        fig.savefig('%s/%s/trajs_and_wavelets_id%s.png'%(parameters.projectPath, parameters.method, num))

# function to plot many trajectories and wavelets in a region
#WORKING!
def many_trajectories_and_wavelets_in_region(parameters, projections, zValues,list_trialstroke, wavelets,nr, ntw, num, wregions=None, minimum_regions=None):
    if len(wregions)!=0:
        wregions = wregions
    else:
        wregions = get_wregions(parameters, minimum_regions)
    ls = wstrokes_in_area(zValues, wregions, nr)
    if (len(ls)>0):
        lsample = choose_sample_strokes(ls, ntw)
        trajectories_and_wavelets(parameters, projections, zValues,list_trialstroke, wavelets, lsample, num, nr)


# function to plot many trajectories and wavelets in all regions
#WORKING!
def many_trajectories_and_wavelets_in_all_regions(parameters, projections, zValues, list_trialstroke, wavelets, ntw, num, minimum_regions):
    wregions = get_wregions(parameters, minimum_regions)
    nregions = wregions.max()
    for nr in range(1, nregions+1):
        many_trajectories_and_wavelets_in_region(parameters, projections, zValues, list_trialstroke, wavelets, nr, ntw, num, wregions)
