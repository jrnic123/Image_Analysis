#import relevant items
import os
from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

from skimage.filters import threshold_otsu
from skimage import img_as_ubyte



# from pydoc import describe
# import pandas as pd
# import numpy as np
# import pylab
# from PIL import Image
# from scipy import stats
# from skimage.io import imread, imshow, imsave
# from skimage.filters import threshold_otsu
# from skimage import img_as_ubyte

# import os
# from os import listdir
# import math 
# import cv2 
# import glob
# from os.path import isfile, join

# import glob
# import matplotlib.pyplot as plt
# import imageio as iio
# import skimage.color

# import matplotlib.pyplot as plt
# from scipy.integrate import trapz, simps


def indiv_ID(img, no_slices):
    img = cv2.imread(img,0) # 0 loads in as gray scale
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

    return ID, std


# same as indiv_ID but fed through thresh_curve
def create_slices(img, no_slices):
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

    return ID, std


def thresh_curve_indiv_img(impath, no_slices):
    # var=[]
    count = 0

    img = cv2.imread(impath, 0)
    img.resize(1024,1536)
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300

    thresh_quantile = [0.95, 0.94, 0.93, 0.92, 0.91, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]

    ID = []
    std = []
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

        ID1, std1 = create_slices(img_array, no_slices)
        # df.append([ID, mean, t])        
        ID.append(ID1)
        std.append(std1)
        # count += 1 

    


    df = pd.DataFrame(np.column_stack([ID, std, thresh_quantile]), columns=['ID', 'std', 'thresh'])
    # df = df.set_index('thresh').T
    df["filename"] = os.path.split(impath)[1]
    df = df.set_index('filename')

        # # Plot the data
    x = thresh_quantile
    y = df["ID"]
    e = df["std"]

    plt.plot(x, y)
    plt.errorbar(x, y, yerr=e, linestyle='none', marker='.', markersize=10)
    plt.title(os.path.split(impath)[1])
    plt.xlabel('Threshold Parameter')
    plt.ylabel('Index of Dispersion')
    plt.rcParams['figure.figsize'] = [4, 4] 
    plt.show()
    return df

# same as thresh curve ind img but used as feed through to thresh_folder
def thresh_curve(impath, no_slices):
    # var=[]
    count = 0

    img = cv2.imread(impath, 0)
    img.resize(1024,1536)
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300

    thresh_quantile = [0.95, 0.94, 0.93, 0.92, 0.91, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]

    ID = []
    std = []
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

        ID1, std1 = create_slices(img_array, no_slices)
        # df.append([ID, mean, t])        
        ID.append(ID1)
        std.append(std1)
        # count += 1 

    
    df = pd.DataFrame(np.column_stack([ID, std, thresh_quantile]), columns=['ID', 'std', 'thresh'])
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



import pandas as pd

def summarize_max_id(df: pd.DataFrame, condition: float) -> pd.DataFrame:
    """
    For each filename in the DataFrame, extract the maximum ID value,
    then compute the average and standard deviation of those max values.
    
    Parameters:
    - df: A pandas DataFrame with columns ['ID', 'std', 'thresh'] and a MultiIndex with 'filename' as index
    - folder_name: A string representing the folder name for summary labeling
    
    Returns:
    - A one-row DataFrame with columns ['folder', 'avg_max_ID', 'std_max_ID']
    """
    # Reset index if filename is in the index
    if 'filename' not in df.columns:
        df = df.reset_index()

    # Ensure required columns exist
    if not {'filename', 'ID'}.issubset(df.columns):
        raise ValueError("DataFrame must include 'filename' and 'ID' columns")

    # Get max ID per filename
    max_id_per_file = df.groupby('filename')['ID'].max()

    # Calculate mean and std of max IDs
    avg_max_id = max_id_per_file.mean()
    std_max_id = max_id_per_file.std()

    # Return as DataFrame
    return pd.DataFrame({
        'condition': [condition],
        'avg_max_ID': [avg_max_id],
        'std_max_ID': [std_max_id]
    })

def append_dataframes(df_list, ignore_index=True):
    """
    Concatenates a list of DataFrames into a single DataFrame.

    Parameters:
    - df_list (list of pd.DataFrame): List of DataFrames to concatenate.
    - ignore_index (bool): Whether to ignore index during concatenation. Defaults to True.

    Returns:
    - pd.DataFrame: The concatenated DataFrame.
    """
    return pd.concat(df_list, ignore_index=ignore_index)


import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm
import numpy as np

def plot_stacked_id_vs_cycle(dataframes, labels, colors=None, markers=None, title="stacked_plot", ylabel="Index of Dispersion", output_path=None):
    """
    Plot stacked line plots with shaded standard deviations.
    
    Parameters:
    - dataframes: list of pd.DataFrame objects with 'Cycle Number', 'avg ID', and 'std ID'
    - labels: list of strings corresponding to each dataset
    - colors: list of hex or named colors (optional)
    - markers: list of marker styles (optional)
    - title: figure title or filename prefix
    - ylabel: y-axis label for the whole figure
    - output_path: full path to save the figure (optional)
    """

    n = len(dataframes)
    if colors is None:
        colors = [cm.viridis(i / n) for i in range(n)]
    if markers is None:
        markers = ['o', 's', '^', 'D', 'v', '*', 'X'][:n]

    fig, axes = plt.subplots(n, 1, figsize=(8, 4 * n), sharex=True)

    if n == 1:
        axes = [axes]  # Ensure it's iterable

    for ax, df, label, color, marker in zip(axes, dataframes, labels, colors, markers):
        ax.plot(df["condition"], df["avg_max_ID"], marker=marker, markersize=10,
                mec='k', label=label, linestyle=':', color=color)
        ax.fill_between(df["condition"],
                        df["avg_max_ID"] - df["std_max_ID"],
                        df["avg_max_ID"] + df["std_max_ID"],
                        color=color, alpha=0.4)
        ax.legend(loc='upper right', bbox_to_anchor=(0.4, 1), frameon=False)
        ax.set_ylabel("")  # Individual y-labels not needed; shared below

    axes[-1].set_xlabel('condition')
    fig.text(0.001, 0.5, ylabel, va='center', rotation='vertical', fontsize=24)

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
    
    plt.show()


import plotly.graph_objects as go

import plotly.graph_objects as go

# def plot_id_vs_cycle(dataframes, output_path=None):
#     """
#     Creates a line plot with markers and shaded standard deviation bands
#     from a list of DataFrames. Each DataFrame must contain:
#     - 'condition' (x-axis)
#     - 'avg_max_ID' (y-axis)
#     - 'std_max_ID' (for shading)
    
#     Parameters:
#     - dataframes (list of tuples): Each tuple should be (label, df, color) where:
#         label (str): Legend label (e.g., "3mAh")
#         df (pd.DataFrame): Must contain 'condition', 'avg_max_ID', 'std_max_ID'
#         color (str): Color name (e.g., "blue", "orange")
#     - output_path (str): If provided, saves the figure to this path.

#     Returns:
#     - fig (plotly.graph_objects.Figure): The generated figure
#     """
#     fig = go.Figure()

#     for label, df, color in dataframes:
#         # Line and markers
#         fig.add_trace(go.Scatter(
#             x=df["condition"],
#             y=df["avg_max_ID"],
#             name=label,
#             line=dict(color=color, dash='dot'),
#             mode="lines+markers",
#             marker=dict(symbol="circle", size=6)
#         ))

#         # Shaded standard deviation area
#         fig.add_trace(go.Scatter(
#             x=list(df["condition"]) + list(df["condition"])[::-1],
#             y=list(df["avg_max_ID"] + df["std_max_ID"]) + list(df["avg_max_ID"] - df["std_max_ID"])[::-1],
#             fill='toself',
#             fillcolor=f'rgba{go.Figure()._parse_color(color, 0.2)}',
#             line=dict(color='rgba(255,255,255,0)'),
#             showlegend=False,
#             name=f'{label}-shade'
#         ))

#     # Layout
#     fig.update_layout(
#         template="simple_white",
#         title="Average Maximum ID with Standard Deviation",
#         legend=dict(yanchor="top", y=1.0, xanchor="left"),
#         legend_title_text="Condition",
#         width=700,
#         height=600,
#         xaxis_title="Condition",
#         yaxis_title="Average Max ID"
#     )

#     fig.show()

#     if output_path:
#         fig.write_image(output_path)
    
#     return fig

# def append_dataframes(df_list, ignore_index=True):
#     """
#     Concatenates a list of DataFrames into a single DataFrame.

#     Parameters:
#     - df_list (list of pd.DataFrame): List of DataFrames to concatenate.
#     - ignore_index (bool): Whether to ignore index during concatenation. Defaults to True.

#     Returns:
#     - pd.DataFrame: The concatenated DataFrame.
#     """
#     return pd.concat(df_list, ignore_index=ignore_index)

# def color_to_rgba(color, alpha=0.2):
#     """
#     Converts basic color names to RGBA strings with specified alpha transparency.
#     Supports 'blue', 'orange', 'green', etc.
#     """
#     named_colors = {
#         "blue": "0,0,255",
#         "orange": "255,165,0",
#         "green": "0,255,0",
#         "red": "255,0,0",
#         "purple": "128,0,128",
#         "gray": "128,128,128",
#         "black": "0,0,0"
#     }
#     rgb = named_colors.get(color.lower(), "0,0,0")
#     return f"rgba({rgb},{alpha})"


# def analyze_curve(df, thresh_quantile):
#     # Calculate mean and standard deviation for each threshold
#     mean = df.groupby(df['thresh']).mean().rename(columns={'ID': 'avg ID', 'mean': 'avg FC'})
#     std = df.groupby(df['thresh']).std().rename(columns={'ID': 'std ID', 'mean': 'std FC'})
    
#     # Combine mean and standard deviation data
#     df_final = pd.concat([mean, std], axis=1).sort_index(ascending=False)
    
#     # # Plot the data
#     x = thresh_quantile
#     y = df_final["avg ID"]
#     e = df_final["std ID"]

#     plt.plot(x, y)
#     plt.errorbar(x, y, yerr=e, linestyle='none', marker='.', markersize=10)
#     plt.title('3D Reconstruction by varying FC')
#     plt.xlabel('% Brightest Pixels')
#     plt.ylabel('Index of Dispersion')
#     plt.rcParams['figure.figsize'] = [4, 4] 
#     # plt.show()
    
#     # Calculate the area under the curve
#     quartile_thresh = sorted(thresh_quantile)

#     return df_final