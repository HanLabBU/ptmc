"""
Code for parallel processing operations for numpy arrays of tif stacks using python's multiprocessing tool

Default behavior is to use a pool with as many processes as CPUs

Each parallel function has a paired "unpack" function to be handled within a map call
"""

#Import packages
#Dependences
import numpy as np
from numpy.fft import fft2, ifft2, fftshift
from scipy.ndimage import median_filter, gaussian_filter, shift
import itertools
import gc
import multiprocessing as mp


#Median Filter Code
def med_filter_unpack(inList):
    #Unpack In List
    frame = inList[0]
    med_size = inList[1]
    return median_filter(frame, size=med_size)

def doMedianFilter(imgstack, pool=None, med_fsize=3):
    '''
    Median Filter
    imgstack is (nframes, height, width) numpy array of images
    pool is a parallel pool assuming the python multiprocessing package.  If None, a pool is started and closed within the function
    med_fsize is the median filter size
    Returns medstack, a (nframes, height, width) numpy array of median filtered images
    '''
    #Start Parallel Pool if one not started
    if pool is None:
        inpool = None
        n_cores = mp.cpu_count()
        pool = mp.Pool(processes=n_cores)
    else:
        inpool = 0
    
    #Create appropriate input list for parallel mapping
    imgList = [(img, med_fsize) for img in imgstack]
    medOutput = pool.map(med_filter_unpack, imgList)
    medstack = np.array(medOutput) #Output is list, so convert to np.array
    
    #Close Pool as clean-up if no pool given in
    if inpool is None:
        pool.close()
    
    return medstack


#Homomorphic Filter Code
def log_subtract_unpack(inList):
    #First step to log and subtract the low pass components
    #Unpack In List
    frame = inList[0]
    sigmaVal = inList[1]
    eps = inList[2]
    ScaleFactor = inList[3]
    Baseline = inList[4]
    
    #Process Frames
    logimg = np.log1p(np.maximum((frame-Baseline)*ScaleFactor, eps))
    lpComponent = gaussian_filter(logimg, sigma=sigmaVal)
    return logimg - lpComponent

def homo_adjust_unpack(inList):
    #Second step to subtract minimum and scale back to standard values
    #Unpack In List
    frame = inList[0]
    logmin = inList[1]
    ScaleFactor = inList[2]
    Baseline = inList[3]
    
    #Process Frames
    adjimg = frame - logmin
    return (np.expm1(adjimg)/ScaleFactor) + Baseline

def doHomomorphicFilter(imgstack, pool=None, sigmaVal=7):
    '''
    Homomorphic Filter
    imgstack is (nframes, height, width) numpy array of images
    pool is a parallel pool assuming the python multiprocessing package.  If None, a pool is started and closed within the function
    sigmaVal is the gaussian_filter size for subtracing the low frequency component
    Returns homomorphimgs, a (nframes, height, width) numpy array of homomorphic filtered images
    '''
    #Start Parallel Pool if one not started
    if pool is None:
        inpool = None
        n_cores = mp.cpu_count()
        pool = mp.Pool(processes=n_cores)
    else:
        inpool = 0
    
    #Constants to scale from between 0 and 1, calculated over whole image stack
    eps = 7./3 - 4./3 -1 
    maxval = imgstack.max()
    ScaleFactor = 1./maxval
    Baseline = imgstack.min()
    
    #Create appropriate input list for parallel mapping and apply list to parallelized function
    imgList = [(img, sigmaVal, eps, ScaleFactor, Baseline) for img in imgstack]
    adjOutput = pool.map(log_subtract_unpack, imgList)
    adjimgs = np.array(adjOutput)
    del adjOutput #Remove extra list from memory
    gc.collect()
    
    #Take minimum across all array values
    logmin = adjimgs.min()
    adjList = [(img, logmin, ScaleFactor, Baseline) for img in adjimgs]
    homomorphOutput = pool.map(homo_adjust_unpack, adjList)
    homomorphimgs = np.array(homomorphOutput)
    
    #Close Pool as clean-up if no pool given in
    if inpool is None:
        pool.close()
    
    return homomorphimgs


#Image Registration Code
def stack_shift_unpack(inList):
    #Calculate framewise Cross Correlation and apply to the image stack
    #Unpack In List
    frame = inList[0]
    imshape = inList[1]
    imcenter = inList[2]
    Ref_fft = inList[3]
    
    #Calculate Cross Correlation and Apply Shift to the frame
    xcfft = fft2(frame) * Ref_fft
    xcim = abs(ifft2(xcfft))
    xcpeak = np.array(np.unravel_index(np.argmax(fftshift(xcim)), imshape))
    disps = imcenter - xcpeak
    shiftFrame = np.uint16(shift(frame, disps))
    yshift = disps[0]
    xshift = disps[1]
    return (shiftFrame, yshift, xshift)

def registerImages(imgstack, Ref=None, pool=None, method='CrossCorrelation'):
    '''
    Perform frame-by-frame Image Registration to a reference image using a default of Cross Correlation
    imgstack is (nframes, height, width) numpy array of images
    Ref is a (height, width) numpy array as a reference image to use for motion correction
    If no Ref is given, then the mean across all frames is used
    pool is a parallel pool assuming the python multiprocessing package.  If None, a pool is started and closed within the function
    method is the method to use to register the images, with the default being cross-correlation between the Reference frame and each individual frame
    Returns stackshift, a (nframes, height, width) numpy array of motion corrected and shifted images
    Returns yshift is the number of pixels to shift each frame in the y-direction (height)
    Returns xshift is the number of pixels to shift each frame in the x-direction (width)
    '''
    #Insert functions for different registration methods
    def CrossCorrelation(imgstack, Ref, pool):
        #Start Parallel Pool if one not started
        if pool is None:
            inpool = None
            n_cores = mp.cpu_count()
            pool = mp.Pool(processes=n_cores)
        else:
            inpool = 0
        
        #Precalculate Static Values
        if Ref is None:
            Ref = imgstack.mean(axis=0)
        imshape = Ref.shape
        imcenter = np.array(imshape)/2
        Ref_fft = fft2(Ref).conjugate()
        
        #Create list for parallelization and apply function to measure shifts from Images and apply those shifts to the Images
        stackList = [(img, imshape, imcenter, Ref_fft) for img in imgstack]
        shiftOutput = pool.map(stack_shift_unpack, stackList)
        stackshift = np.array([x[0] for x in shiftOutput])
        yshift = np.array([x[1] for x in shiftOutput])
        xshift = np.array([x[2] for x in shiftOutput])
        
        #Close Pool as clean-up if no pool given in
        if inpool is None:
            pool.close()
        
        return stackshift, yshift, xshift
    
    #Dictionary for method selection and return
    method_select = {
        'CrossCorrelation': CrossCorrelation(imgstack, Ref, pool),
    }

    #Run the selected method from the dictionary the method_select dictionary
    return method_select.get(method, "ERROR: No function defined for Provided Method")


#Framewise Cross Correlation Code
def frame_shifts_unpack(inList):
    #Calculate the shifts between 2 stacks in a frame-by-frame manner from framewise Cross Correlation
    #Unpack In List
    frame1 = inList[0]
    frame2 = inList[1]
    imshape = inList[2]
    imcenter = inList[3]
    
    #Calculate shift between frames
    xcfft = fft2(frame1) * fft2(frame2).conjugate()
    xcim = abs(ifft2(xcfft))
    xcpeak = np.array(np.unravel_index(np.argmax(fftshift(xcim)), imshape))
    disps = imcenter - xcpeak
    yshift = disps[0]
    xshift = disps[1]
    return (yshift, xshift)

def calculateFramewiseCrossCorrelation(imgstack1, imgstack2, pool=None):
    '''
    Calculate frame-by-frame Cross Correlation between two image stacks
    imgstack1 is (nframes, height, width) numpy array of images
    imgstack2 is (nframes, height, width) numpy array of images
    imgstack1 and imgstack2 should be the same dimensions, however if one video is shorter than the other, then the values will be calculated for all of the length of the shorter video
    pool is a parallel pool assuming the python multiprocessing package.  If None, a pool is started and closed within the function
    Returns yshift is the number of pixels to shift each frame in the y-direction (height)
    Returns xshift is the number of pixels to shift each frame in the x-direction (width)
    '''
    #Start Parallel Pool if one not started
    if pool is None:
        inpool = None
        n_cores = mp.cpu_count()
        pool = mp.Pool(processes=n_cores)
    else:
        inpool = 0
    
    #Precalculate Static Values
    imshape = imgstack1.shape[1:]
    imcenter = np.array(imshape)/2
    
    #Create list for parallelized function to loop through frames and compute cross correlation between each frame in the stack
    pairedList = [(frame1, frame2, imshape, imcenter) for (frame1, frame2) in zip(imgstack1, imgstack2)]
    shiftOutput = pool.map(frame_shifts_unpack, pairedList)
    yshift = np.array([x[0] for x in shiftOutput])
    xshift = np.array([x[1] for x in shiftOutput])
    
    #Close Pool as clean-up if no pool given in
    if inpool is None:
        pool.close()
    
    return yshift, xshift


#Apply Frame Shifts to a Stack
def apply_shifts_unpack(inList):
    #Take input arrays of frameshifts to apply to an image stack
    #Unpack In List
    frame = inList[0]
    yshift = inList[1]
    xshift = inList[2]
    
    #Apply Shift to the Frame
    return np.uint16(shift(frame, (yshift, xshift)))

def applyFrameShifts(imgstack, yshift, xshift, pool=None):
    '''
    Apply frame shifts to each frame of an image stack
    imgstack is (nframes, height, width) numpy array of images
    yshift is the number of pixels to shift each frame in the y-direction (height)
    xshift is the number of pixels to shift each frame in the x-direction (width)
    pool is a parallel pool assuming the python multiprocessing package.  If None, a pool is started and closed within the function
    Returns stackshift, a (nframes, height, width) numpy array of images shifted according to yshift & xshift
    '''
    #Start Parallel Pool if one not started
    if pool is None:
        inpool = None
        n_cores = mp.cpu_count()
        pool = mp.Pool(processes=n_cores)
    else:
        inpool = 0
    
    #Precalculate Static Values
    imgList = [(frame, yshift[idx], xshift[idx]) for (idx, frame) in enumerate(imgstack)]
    shiftOutput = pool.map(apply_shifts_unpack, imgList)
    stackshift = np.array(shiftOutput)
    
    #Close Pool as clean-up if no pool given in
    if inpool is None:
        pool.close()
    
    return stackshift

