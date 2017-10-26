"""
Code for all input/output loading/saving operations for .tif stacks and associated files
"""

import os
import Tkinter as tk
import tkFileDialog as tkfd
import tkMessageBox as tkmb
import tifffile as tf #Much faster loading but can't save large tiffs (Only as BigTiff)
from PIL import Image #Slower loading, but can save large tiffs
import h5py

def loadImageStack(inputTif=None, GUItitle='Select .Tif'):
    '''
    Loads a multipage tif file and returns the file path and info for that file.
    inputTif is the full path to the tif file.  If not given, opens GUI to select it.
    Returns imgstack a (nframes, height, width) numpy array of images
    Returns fileparts, a tuple as (path, filename)
    '''
    if inputTif is None:
        #Get File Input via GUI if no input given
        root = tk.Tk(); #Graphical Interface
        root.withdraw()
        inputTif = tkfd.askopenfilename(title=GUItitle)
    
    #Get parts of file for output
    fileparts = os.path.split(inputTif)
    
    #Load File (Takes 191 sec. 2.5 sec locally)
    imgstack = tf.imread(inputTif)
    
    #PIL (18.3 sec locally)
    #tiffstack = Image.open(inputTif)
    #imgs1 = np.zeros((tiffstack.n_frames, tiffstack.height, tiffstack.width))
    #for idx in range(tiffstack.n_frames):
    #    try:
    #        tiffstack.seek(idx)
    #        imgs1[idx,...] = tiffstack
    #    except EOFError:
    #        #Not enough frames in img
    #        break
    
    return imgstack, fileparts

def getFileList(GUItitle='Select Files'):
    '''
    Gets a list of files from a GUI selection.  A wrapper for tkfd.askopenfilenames
    GUItitle allows to change the name of the GUI Title
    Returns fileList, a tuple of all the selected files with path included
    Returns mainDir, the directory where the files were pulled from
    '''
    #Get File Input via GUI
    root = tk.Tk(); #Graphical Interface
    root.withdraw()
    fileList = tkfd.askopenfilenames(title=GUItitle)
    
    fileparts = os.path.split(fileList[0]) #Assume only selected from 1 directory
    mainDir = fileparts[0]
    
    return fileList, mainDir

def getDir(GUItitle='Select Directory', initialDir=os.getcwd()):
    '''
    Gets a list of files from a GUI selection.  A wrapper for tkfd.askdirectory
    GUItitle allows to change the name of the GUI Title
    initialDir selects the starting directory to search in
    Returns selectedDir, a tuple of all the selected files with path included
    '''
    #Get directory via GUI
    root = tk.Tk(); #Graphical Interface
    root.withdraw()
    selectedDir = tkfd.askdirectory(title=GUItitle, initialdir=initialDir)

    return selectedDir 

def askYesNo(GUItitle='I have a question', GUImessage='Do you have a question?'):
    '''
    Dialog Box that asks a Yes or No question.  A wrapper for tkMessageBox
    GUItitle allows to change 3the name of the GUI Title
    Returns True if Yes is selected and False if No is selected
    '''
    #Get input via GUI
    root = tk.Tk(); #Graphical Interface
    root.withdraw()
    answer = tkmb.askyesno(title=GUItitle, message=GUImessage)
    
    return answer

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

