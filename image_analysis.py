#import relevant items
from pydoc import describe
import pandas as pd
import numpy as np
import pylab
from PIL import Image
from scipy import stats
from skimage.io import imread, imshow, imsave
from skimage.filters import threshold_otsu
from skimage import img_as_ubyte

import os
from os import listdir
import math 
import cv2 
import glob
from os.path import isfile, join

import glob
import matplotlib.pyplot as plt
import imageio as iio
import skimage.color

import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import trapz, simps


def create_slices(img, no_slices):
    # img = cv2.imread(img,0) # 0 loads in as gray scale
    img.resize(1536,1024) #need to do this if using real images?

    # print(img.shape) #debug

    inc_1=int(img.shape[0]/no_slices) #pixels per slice along x direction 
    img_area = img.shape[0]*img.shape[1]
    # print(img_area) #debug
    inc_2= int(img.shape[1]*inc_1/img.shape[0]) #pixels per slice along y direction 
    
    slice_area = inc_1*inc_2 #use dimensions to calculate area of slice 
    # print(slice_area) #debug 
    
    FC = []

    for i in range(no_slices):
        for j in range(no_slices):
            # print(i*inc_1,(i+1)*inc_1, j*inc_2,(j+1)*inc_2)
            img_slice = img[i*inc_1:(i+1)*inc_1, j*inc_2:(j+1)*inc_2] #moving along equally spaced coords per each slice i=x, j=y
            thresh = threshold_otsu(img_slice)
            # print(thresh)
            binary = img_slice > thresh

            #skimage.img_as_ubyte() method converts float point numbers back to whole numbers
            binary = img_as_ubyte(binary)
            # imshow(binary)

            # extracting only white pixels
            number_of_white_pix = np.sum(binary == 255)

            # extracting only black pixels
            number_of_black_pix = np.sum(binary == 0)

            FC.append(number_of_white_pix/slice_area)

    # this gives df of white pixel FC
    slices_df = pd.DataFrame(FC)
    slices_describe = slices_df.describe()

    # fc_total = img.all(axis=-1).mean(axis=-1).mean(axis=-1)
    # # will give average number of white pixels for entire image - do I need this or or sum of white pixels per entire image

    slices_describe = pd.DataFrame(slices_describe)
    slices_describe = slices_describe.T

    q=no_slices
    mean = slices_describe["mean"]
    std = slices_describe["std"]


    ID = (q-1)*(std)**2/(mean)
    slices_describe["ID"] = ID

    return ID, mean


def thresh_curve(impath, no_slices):
    # var=[]
    count = 0

    img = cv2.imread(impath, 0)
    img.resize(1024,1536)
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300

    thresh_quantile = [0.95, 0.94, 0.93, 0.92, 0.91, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]

    ID = []
    mean = []
    for t in thresh_quantile:
        # threshold is set to 5% brightest pixels 
        min_thres = np.quantile(img, t)
        #print(min_thres)
        # simple thresholding to create a binary image
        ret, img_array = cv2.threshold(img, min_thres, 255, cv2.THRESH_BINARY)

        ###saves images
        name = os.path.splitext(impath)[0] + "_%s.tiff" % t
        # print(name)
        # plt.imshow(impath, cmap='gray')
        # cv2.imwrite(name,img_array)

        ID1, mean1 = create_slices(img_array, no_slices)
        # df.append([ID, mean, t])        
        ID.append(ID1)
        mean.append(mean1)
        # count += 1 



    df = pd.DataFrame(np.column_stack([ID, mean, thresh_quantile]), columns=['ID', 'mean', 'thresh'])
    # df = df.set_index('thresh').T
    df["filename"] = os.path.split(impath)[1]
    df = df.set_index('filename')

    return df


def thresh_folder(folder_path, no_slices):
#################################
### Get file names from folder ##
#################################
# Initialise file name list
    filename=[]
    count = 0
    # for loop to go through the folder and check if the contents is 1) a file, and 2) a jpg and if so, add to initialised list
    for f in listdir(folder_path): #listdirectory is making a list of all files in folder
    #     Validation check: verifies folder contents are both files and jpg
        if isfile(join(folder_path, f)) and '.tif' in f:
    #         Fills the fractal_area_data list with file names
            count += 1 
            filename.append(f)

    ######initialize table i want 7 x 
   #[imname 0.5 0.6 0.7 0.8 0.9 0.95] x length of filename ($ of images per folder]

    # arr=[]
    # arr = [0 for i in range(5)]

    full_df = []
    for file in filename:
    # combined line: 
    # 1) joins folder and file to make the full path (from right to left)
    # 2) Performs the function frac_coverage_pixel on the image currently designated by file in the loop
    # 3) Adds the value to the list we initialised before the for loop
        full_df1 = thresh_curve((join(folder_path,file)), no_slices) #adds it to the list
        # full_df['filename'] = file
        # full_df = full_df + full_df1
        full_df.append(full_df1)
    full_df = pd.concat(full_df)
        # full_df = pd.concat(full_df1)
        # full_df = full_df + full_df1
    return full_df
        #### add in ID function here 


def analyze_curve(df, thresh_quantile):
    # Calculate mean and standard deviation for each threshold
    mean = df.groupby(df['thresh']).mean().rename(columns={'ID': 'avg ID', 'mean': 'avg FC'})
    std = df.groupby(df['thresh']).std().rename(columns={'ID': 'std ID', 'mean': 'std FC'})
    
    # Combine mean and standard deviation data
    df_final = pd.concat([mean, std], axis=1).sort_index(ascending=False)
    
    # # Plot the data
    x = thresh_quantile
    y = df_final["avg ID"]
    e = df_final["std ID"]

    plt.plot(x, y)
    plt.errorbar(x, y, yerr=e, linestyle='none', marker='.', markersize=10)
    plt.title('3D Reconstruction by varying FC')
    plt.xlabel('% Brightest Pixels')
    plt.ylabel('Index of Dispersion')
    plt.rcParams['figure.figsize'] = [4, 4] 
    # plt.show()
    
    # Calculate the area under the curve
    quartile_thresh = sorted(thresh_quantile)
    area_trapezoidal = trapz(y, quartile_thresh)
    area_simpson = simps(y, quartile_thresh)
    
    # return df_final, area_trapezoidal, area_simpson
    return df_final