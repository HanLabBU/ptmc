"""
Code for pre-set pipelines of processing operations for numpy arrays of tif stacks

Requires PIL and numpy as external packages

Requires these other packages from ptmc:
io
processing
"""

#Import other packages
import numpy as np
import os
from PIL import Image
import gc
#PTMC itself
import io
import processing as pro


def makeReferenceFrame(refTif, saveDir=None, method='Median_Homomorphic'):
    '''
    Code to make a reference frame and save the output as a Tif.
    refTif is the full path to a Tif Stack to create a Reference Frame from
    refArray is a homomorphic filtered, mean intensity image that is motion corrected from refTif
    method is the method to use to make the reference frame, with the default being a Median Filter followed by a Homomorphic Filtering of the images, prior to motion correction
    A hidden output is a tif of the same name with '_MCrefImage" appended to the end, which is a saved version of refArray
    '''
    #Insert functions for different registration methods
    def Median_Homomorphic(refTif, saveDir):
        #Parse path info in reference Tif
        fileparts = os.path.split(refTif)
        if saveDir is None: #If no save directory provided
            saveDir = fileparts[0]
        
        #Loading
        print('Loading Reference file ' + refTif)
        imgstack, fileparts = io.loadImageStack(inputTif=refTif)
        
        #Processing Steps
        print('Processing Reference file ' + refTif)
        medstack = pro.doMedianFilter(imgstack, med_fsize=3)
        homomorphstack = pro.doHomomorphicFilter(medstack, sigmaVal=7)
        del medstack #Remove to save memory
        gc.collect()
        homoshift, yshift, xshift = pro.registerImages(homomorphstack)
        rawshift = pro.applyFrameShifts(imgstack, yshift, xshift)
        #Save Reference Frame
        print('Saving Reference file ' + refTif)
        refArray = homoshift.mean(axis=0).astype(np.uint16)
        refIm = Image.fromarray(refArray)
        refIm.save(saveDir+'/'+fileparts[1][:-4]+'_MCrefImage.tif')
        rawRefArray = rawshift.mean(axis=0).astype(np.uint16)
        rawIm = Image.fromarray(rawRefArray)
        rawIm.save(saveDir+'/'+fileparts[1][:-4]+'_MCrefImageRaw.tif')
        print('Completed Reference file ' + refTif + '\n')
    
        return refArray, rawRefArray
    
    #Dictionary for method selection and return
    method_select = {
        'Median_Homomorphic': Median_Homomorphic(refTif, saveDir),
    }

    #Run the selected method from the dictionary the method_select dictionary
    return method_select.get(method, "ERROR: No function defined for Provided Method")

def correctImageStack(Tif, refIm, saveDir=None, method='Median_Homomorphic'):
    '''
    Perform motion correction and save output for a tif with a provided reference frame
    Tif is the full name and path to a single multipage tif
    refIm is the reference image to correct to
    saveDir is the directory to save the corrected images and outputs in
    method is the method to use to correct the image stack, with the default being a Median Filter followed by a Homomorphic Filtering of the images.
    Hidden outputs are the x & y shifts for each frame, motion corrected homomorphic filtered image stack, and motion corrected raw image stack, all saved within saveDir
    '''
    #Insert functions for different registration methods
    def Median_Homomorphic(Tif, refIm, saveDir):
        '''
        Perform motion correction and save output for a tif with a provided reference frame
        Tif is the full name and path to a single multipage tif
        refIm is the reference image to correct to
        saveDir is the directory to save the corrected images and outputs in
        '''
        #Parse path info in reference Tif
        fileparts = os.path.split(Tif)
        if saveDir is None: #If no save directory provided
            saveDir = fileparts[0]
        
        #Loading
        print('Loading file ' + Tif)
        imgstack, fileparts = io.loadImageStack(inputTif=Tif) 
        #Processing Steps
        print('Processing file ' + Tif)
        medstack = pro.doMedianFilter(imgstack, med_fsize=3)
        homomorphstack = pro.doHomomorphicFilter(medstack, sigmaVal=7)
        homoshift, yshift, xshift = pro.registerImages(homomorphstack, Ref=refIm)
        rawshift = pro.applyFrameShifts(imgstack, yshift, xshift)
        #Save Output
        print('Saving file ' + Tif)
        io.saveFrameShifts(yshift, xshift, 
                        saveDir+'/'+fileparts[1], 
                        saveDir+'/'+fileparts[1][:-4]+'_frameShifts.hdf5')
        io.saveImageStack(homoshift, saveDir+'/m_f_'+fileparts[1])
        io.saveImageStack(rawshift, saveDir+'/m_'+fileparts[1])
        print('Completed file ' + Tif + '\n')
    
    #Dictionary for method selection and return
    method_select = {
        'Median_Homomorphic': Median_Homomorphic(Tif, refIm, saveDir),
    }

    #Run the selected method from the dictionary the method_select dictionary
    return method_select.get(method, "ERROR: No function defined for Provided Method")

