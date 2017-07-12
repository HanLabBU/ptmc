"""
Code for all input/output loading/saving operations for .tif stacks and associated files
"""

import os
import Tkinter as tk
import tkFileDialog as tkfd
import tifffile as tf #Much faster loading but can't save large tiffs (Only as BigTiff)
from PIL import Image #Slower loading, but can save large tiffs
import h5py

def loadImageStack():
    '''
    Loads a multipage tif file and returns the file path and info for that file.
    Returns imgstack a (nframes, height, width) numpy array of images
    Returns fileparts, a tuple as (path, filename)
    '''
    #Get File Input
    root = tk.Tk(); #Graphical Interface
    root.withdraw()
    tif_file = tkfd.askopenfilename(title='Select .Tif')
    fileparts = os.path.split(tif_file)

    #Load File (Takes 191 sec. 2.5 sec locally)
    imgstack = tf.imread(tif_file)
    
    #PIL (18.3 sec locally)
    #tiffstack = Image.open(tif_file)
    #imgs1 = np.zeros((tiffstack.n_frames, tiffstack.height, tiffstack.width))
    #for idx in range(tiffstack.n_frames):
    #    try:
    #        tiffstack.seek(idx)
    #        imgs1[idx,...] = tiffstack
    #    except EOFError:
    #        #Not enough frames in img
    #        break
    
    return imgstack, fileparts

def saveImageStack(imgstack, outname):
    '''
    #Save numpy array as multipage tiff file (203.50 sec.  3 min 24 sec)
    #imgstack is (nframes, height, width) numpy array of images to save
    #outname is the path & filename to save the file out
    #Directly saves files and returns nothing.
    '''
    imlist = []
    for frame in imgstack:
        imlist.append(Image.fromarray(frame))
    
    imlist[0].save(outname, save_all=True, append_images=imlist[1:])

def saveFrameShifts(yshift, xshift, shiftsfile, outname):
    '''
    #Save numpy array of yshifts and xshifts as HDF5 File
    #yshift is the number of pixels to shift each frame in the y-direction (height)
    #xshift is the number of pixels to shift each frame in the x-direction (width)
    #shiftsfile is the name of the file that the shifts are for (Raw Data File)
    #outname is the path & filename to save the file out
    #Directly saves files and returns nothing.
    '''
    f = h5py.File(outname)
    f.create_dataset('filename', data=shiftsfile)
    f.create_dataset('yshift', data=yshift)
    f.create_dataset('xshift', data=xshift)

