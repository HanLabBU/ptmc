"""
Code for processing operations for numpy arrays of tif stacks
"""

#Import packages
#Dependences
import numpy as np
from numpy.fft import fft2, ifft2, fftshift
from scipy.ndimage import median_filter, gaussian_filter, shift
import itertools
import gc


def doMedianFilter(imgstack, med_fsize=3):
    '''
    Median Filter (Takes 303.37 sec, 5 min 3 sec)
    imgstack is (nframes, height, width) numpy array of images
    med_fsize is the median filter size
    Returns medstack, a (nframes, height, width) numpy array of median filtered images
    '''
    medstack = np.empty(imgstack.shape, dtype=np.uint16)
    for idx, frame in enumerate(imgstack):
        medstack[idx,...] = median_filter(frame, size=med_fsize)
        
    return medstack

def doHomomorphicFilter(imgstack, sigmaVal=7):
    '''
    Homomorphic Filter (Takes 323.1 sec, 5 min 23 sec)
    imgstack is (nframes, height, width) numpy array of images
    sigmaVal is the gaussian_filter size for subtracing the low frequency component
    Returns homomorphimgs, a (nframes, height, width) numpy array of homomorphic filtered images
    '''
    #Constants to scale from between 0 and 1
    eps = 7./3 - 4./3 -1 
    maxval = imgstack.max()
    ScaleFactor = 1./maxval
    Baseline = imgstack.min()

    # Subtract minimum baseline, and multiply by scale factor.  Force minimum of eps before taking log.
    logimgs = np.log1p(np.maximum((imgstack-Baseline)*ScaleFactor, eps))

    # Get Low Frequency Component from Gaussian Filter
    lpComponent = np.empty(logimgs.shape)
    for idx, frame in enumerate(logimgs):
        lpComponent[idx,...] = gaussian_filter(frame, sigma=sigmaVal)

    # Remove Low Frequency Component and Shift Values
    adjimgs = logimgs - lpComponent
    del logimgs, lpComponent
    gc.collect()
    logmin = adjimgs.min()
    adjimgs = adjimgs - logmin #Shift by minimum logged difference value, so lowest value is 0

    #Undo the log and shift back to standard image space
    homomorphimgs = (np.expm1(adjimgs)/ScaleFactor) + Baseline
    
    return homomorphimgs

def registerImages(imgstack, Ref=None, method='CrossCorrelation'):
    '''
    Perform frame-by-frame Image Registration to a reference image using a default of Cross Correlation (465.43 sec. 7 min 45 sec)
    imgstack is (nframes, height, width) numpy array of images
    Ref is a (height, width) numpy array as a reference image to use for motion correction
    If no Ref is given, then the mean across all frames is used
    method is the method to use to register the images, with the default being cross-correlation between the Reference frame and each individual frame
    Returns stackshift, a (nframes, height, width) numpy array of motion corrected and shifted images
    Returns yshift is the number of pixels to shift each frame in the y-direction (height)
    Returns xshift is the number of pixels to shift each frame in the x-direction (width)
    '''
    #Insert functions for different registration methods
    def CrossCorrelation(imgstack, Ref):
        #Precalculate Static Values
        if Ref is None:
            Ref = imgstack.mean(axis=0)
        imshape = Ref.shape
        nframes = imgstack.shape[0]
        imcenter = np.array(imshape)/2
        yshift = np.empty((nframes,1)); xshift = np.empty((nframes,1));
        Ref_fft = fft2(Ref).conjugate()
        
        #Measure shifts from Images and apply those shifts to the Images
        stackshift = np.zeros_like(imgstack, dtype=np.uint16)
        for idx, frame in enumerate(imgstack):
            xcfft = fft2(frame) * Ref_fft
            xcim = abs(ifft2(xcfft))
            xcpeak = np.array(np.unravel_index(np.argmax(fftshift(xcim)), imshape))
            disps = imcenter - xcpeak
            stackshift[idx,...] = np.uint16(shift(frame, disps))
            yshift[idx] = disps[0]
            xshift[idx] = disps[1]
        
        return stackshift, yshift, xshift
    
    #Dictionary for method selection and return
    method_select = {
        'CrossCorrelation': CrossCorrelation(imgstack, Ref),
    }

    #Run the selected method from the dictionary the method_select dictionary
    return method_select.get(method, "ERROR: No function defined for Provided Method")

def calculateFramewiseCrossCorrelation(imgstack1, imgstack2):
    '''
    Calculate frame-by-frame Cross Correlation between two image stacks (465.43 sec. 7 min 45 sec)
    imgstack1 is (nframes, height, width) numpy array of images
    imgstack2 is (nframes, height, width) numpy array of images
    imgstack1 and imgstack2 should be the same dimensions, however if one video is shorter than the other, then the values will be calculated for all of the length of the shorter video
    Returns yshift is the number of pixels to shift each frame in the y-direction (height)
    Returns xshift is the number of pixels to shift each frame in the x-direction (width)
    '''
    #Precalculate Static Values
    nframes = imgstack1.shape[0]
    imshape = imgstack1.shape[1:]
    imcenter = np.array(imshape)/2
    yshift = np.empty((nframes,1)); xshift = np.empty((nframes,1));
    #Loop through frames and compute cross correlation between each frame in the stack
    for idx, (frame1, frame2) in enumerate(itertools.izip(imgstack1,imgstack2)):
        xcfft = fft2(frame1) * fft2(frame2).conjugate()
        xcim = abs(ifft2(xcfft))
        xcpeak = np.array(np.unravel_index(np.argmax(fftshift(xcim)), imshape))
        disps = imcenter - xcpeak
        yshift[idx] = disps[0]
        xshift[idx] = disps[1]
    
    return yshift, xshift

def applyFrameShifts(imgstack, yshift, xshift):
    '''
    Apply frame shifts to each frame of an image stack (301.28 sec.  5 min 2 sec)
    imgstack is (nframes, height, width) numpy array of images
    yshift is the number of pixels to shift each frame in the y-direction (height)
    xshift is the number of pixels to shift each frame in the x-direction (width)
    Returns stackshift, a (nframes, height, width) numpy array of images shifted according to yshift & xshift
    '''
    #Precalculate Static Values
    stackshift = np.zeros_like(imgstack, dtype=np.uint16)
    for idx, frame in enumerate(imgstack):
        stackshift[idx,...] = np.uint16(shift(frame, (yshift[idx],xshift[idx])))
    
    return stackshift

