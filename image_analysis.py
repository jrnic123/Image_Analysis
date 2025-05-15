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


#used for synthetic image set which doesn't needed to be thresholded
def indiv_ID(img, no_slices):
    """
    Calculates the Index of Dispersion (ID) and standard deviation of pixel intensities across slices of a binary image.

    This function is designed for synthetic images that are already in black and white and do not require pre-thresholding.
    It divides the image into a grid of `no_slices` × `no_slices` subregions, then calculates the fractional coverage
    of white pixels in each slice and uses these values to compute the Index of Dispersion.

    Parameters:
    -----------
    img : str
        Path to the binary image file (already black and white).
    no_slices : int
        Number of vertical and horizontal slices to divide the image into (e.g., 4 results in 4x4 = 16 slices).

    Returns:
    --------
    ID : float
        Index of Dispersion calculated across the slices.
    std : float
        Standard deviation of fractional white pixel coverage across slices.
    """

    # Load the image in grayscale (0 flag means grayscale)
    img = cv2.imread(img, 0)
    
    # Resize the image to a fixed shape for consistent analysis
    img.resize(1536, 1024)  

    # Compute the number of pixels per slice along the vertical (x) direction
    inc_1 = int(img.shape[0] / no_slices)

    # Calculate the total image area
    img_area = img.shape[0] * img.shape[1]

    # Compute the number of pixels per slice along the horizontal (y) direction
    inc_2 = int(img.shape[1] * inc_1 / img.shape[0])

    # Calculate the area of each slice
    slice_area = inc_1 * inc_2

    # Initialize list to store the fractional coverage (FC) of white pixels for each slice
    FC = []

    # Iterate over the grid of image slices
    for i in range(no_slices):  # along vertical axis
        for j in range(no_slices):  # along horizontal axis
            # Extract each slice of the image using calculated increments
            img_slice = img[i*inc_1:(i+1)*inc_1, j*inc_2:(j+1)*inc_2]

            # Apply Otsu's thresholding to convert grayscale slice to binary
            thresh = threshold_otsu(img_slice)
            binary = img_slice > thresh

            # Convert the binary image to 8-bit unsigned integers (0 or 255)
            binary = img_as_ubyte(binary)

            # Count the number of white and black pixels in the binary slice
            number_of_white_pix = np.sum(binary == 255)
            number_of_black_pix = np.sum(binary == 0)

            # Calculate and store fractional coverage of white pixels in this slice
            FC.append(number_of_white_pix / slice_area)

    # Create a DataFrame from the list of white pixel FC values
    slices_df = pd.DataFrame(FC)

    # Get statistical summary (mean, std, etc.) of fractional coverages
    slices_describe = slices_df.describe().T

    # Compute the Index of Dispersion (ID) using formula: ID = (q-1)*(std^2)/mean
    q = no_slices
    mean = slices_describe["mean"]
    std = slices_describe["std"]
    ID = (q - 1) * (std ** 2) / mean

    # Add the Index of Dispersion to the summary DataFrame
    slices_describe["ID"] = ID

    # Return both the ID and the standard deviation
    return ID, std


# same as indiv_ID but embedded in thresh_curve function
# only difference is image has already been loaded, no imread step
def create_slices(img, no_slices):
    """
    Calculates the Index of Dispersion (ID) and standard deviation of pixel intensities across slices of an image.

    This function is intended for use within other functions (e.g., `thresh_curve`) where the image has already
    been loaded (as a NumPy array). The input image is resized and divided into a grid of `no_slices` × `no_slices`
    regions. For each slice, Otsu's thresholding is applied, and the fractional coverage (FC) of white pixels is computed.
    The ID and standard deviation are calculated from these FC values.

    Parameters:
    -----------
    img : numpy.ndarray
        Grayscale image array (already loaded, e.g., using `cv2.imread(..., 0)`).
    no_slices : int
        Number of vertical and horizontal slices to divide the image into (e.g., 4 results in 4x4 = 16 slices).

    Returns:
    --------
    ID : float
        Index of Dispersion calculated across the slices.
    std : float
        Standard deviation of fractional white pixel coverage across slices.
    """
    # Resize the image to a fixed shape for consistent analysis
    img.resize(1536, 1024)  

    # Compute the number of pixels per slice along the vertical (x) direction
    inc_1 = int(img.shape[0] / no_slices)

    # Calculate the total image area
    img_area = img.shape[0] * img.shape[1]

    # Compute the number of pixels per slice along the horizontal (y) direction
    inc_2 = int(img.shape[1] * inc_1 / img.shape[0])

    # Calculate the area of each slice
    slice_area = inc_1 * inc_2

    # Initialize list to store the fractional coverage (FC) of white pixels for each slice
    FC = []

    # Iterate over the grid of image slices
    for i in range(no_slices):  # along vertical axis
        for j in range(no_slices):  # along horizontal axis
            # Extract each slice of the image using calculated increments
            img_slice = img[i*inc_1:(i+1)*inc_1, j*inc_2:(j+1)*inc_2]

            # Apply Otsu's thresholding to convert grayscale slice to binary
            thresh = threshold_otsu(img_slice)
            binary = img_slice > thresh

            # Convert the binary image to 8-bit unsigned integers (0 or 255)
            binary = img_as_ubyte(binary)

            # Count the number of white and black pixels in the binary slice
            number_of_white_pix = np.sum(binary == 255)
            number_of_black_pix = np.sum(binary == 0)

            # Calculate and store fractional coverage of white pixels in this slice
            FC.append(number_of_white_pix / slice_area)

    # Create a DataFrame from the list of white pixel FC values
    slices_df = pd.DataFrame(FC)

    # Get statistical summary (mean, std, etc.) of fractional coverages
    slices_describe = slices_df.describe().T

    # Compute the Index of Dispersion (ID) using formula: ID = (q-1)*(std^2)/mean
    q = no_slices
    mean = slices_describe["mean"]
    std = slices_describe["std"]
    ID = (q - 1) * (std ** 2) / mean

    # Add the Index of Dispersion to the summary DataFrame
    slices_describe["ID"] = ID

    # Return both the ID and the standard deviation
    return ID, std


def thresh_curve_indiv_img(impath, no_slices):
    """
    This function loads a grayscale image and applies a series of threshold values based on brightness quantiles.
    For each thresholded image, it calculates the Index of Dispersion (ID) and standard deviation of white pixel
    fractions across equally-sized slices. The results are plotted and returned as a DataFrame.

    Parameters:
        impath (str): Path to the input image file.
        no_slices (int): Number of slices along each image dimension (e.g., 4 would divide image into 4x4 grid).

    Returns:
        df (pd.DataFrame): DataFrame with columns for threshold value, ID, standard deviation, and filename.
    """

    # Counter (not currently used, but could be helpful for debugging or indexing)
    count = 0

    # Load the image in grayscale mode (0 flag)
    img = cv2.imread(impath, 0)

    # Resize image to a fixed size (height=1024, width=1536)
    # Important for consistent slice area calculations across images
    img.resize(1024, 1536)

    # Set high-resolution plotting for display and saved figures
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300

    # List of quantile thresholds (top X% brightest pixels retained in binary image)
    thresh_quantile = [0.95, 0.94, 0.93, 0.92, 0.91, 0.9,
                       0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]

    # Lists to store results for each threshold
    ID = []   # Index of Dispersion
    std = []  # Standard deviation of white pixel fraction per slice

    # Loop through each quantile threshold
    for t in thresh_quantile:
        # Compute brightness threshold value corresponding to quantile
        min_thres = np.quantile(img, t)

        # Apply binary thresholding using OpenCV
        # Pixels above min_thres become white (255), others black (0)
        ret, img_array = cv2.threshold(img, min_thres, 255, cv2.THRESH_BINARY)

        # Optional: construct filename for saving the thresholded binary image
        name = os.path.splitext(impath)[0] + "_%s.tiff" % t
        # cv2.imwrite(name, img_array)  # Uncomment to save thresholded image

        # Compute dispersion statistics across image slices
        ID1, std1 = create_slices(img_array, no_slices)

        # Store results
        ID.append(ID1)
        std.append(std1)

    # Combine results into a DataFrame
    df = pd.DataFrame(np.column_stack([ID, std, thresh_quantile]),
                      columns=['ID', 'std', 'thresh'])

    # Add the filename column for traceability
    df["filename"] = os.path.split(impath)[1]

    # Set the filename as the DataFrame index
    df = df.set_index('filename')

    # Plot the threshold parameter (x-axis) vs. Index of Dispersion (y-axis) with error bars
    x = thresh_quantile
    y = df["ID"]
    e = df["std"]

    plt.plot(x, y, label='ID vs. Threshold')
    plt.errorbar(x, y, yerr=e, linestyle='none', marker='.', markersize=10, label='Std Error')
    plt.title(os.path.split(impath)[1])
    plt.xlabel('Threshold Parameter')
    plt.ylabel('Index of Dispersion')
    plt.rcParams['figure.figsize'] = [4, 4] 
    plt.legend()
    plt.show()

    return df


# same as thresh curve ind img but used as feed through to thresh_folder
def thresh_curve(impath, no_slices):
    
    # Counter (not currently used, but could be helpful for debugging or indexing)
    count = 0

    # Load the image in grayscale mode (0 flag)
    img = cv2.imread(impath, 0)

    # Resize image to a fixed size (height=1024, width=1536)
    # Important for consistent slice area calculations across images
    img.resize(1024, 1536)

    # Set high-resolution plotting for display and saved figures
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300

    # List of quantile thresholds (top X% brightest pixels retained in binary image)
    thresh_quantile = [0.95, 0.94, 0.93, 0.92, 0.91, 0.9,
                       0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]

    # Lists to store results for each threshold
    ID = []   # Index of Dispersion
    std = []  # Standard deviation of white pixel fraction per slice

    # Loop through each quantile threshold
    for t in thresh_quantile:
        # Compute brightness threshold value corresponding to quantile
        min_thres = np.quantile(img, t)

        # Apply binary thresholding using OpenCV
        # Pixels above min_thres become white (255), others black (0)
        ret, img_array = cv2.threshold(img, min_thres, 255, cv2.THRESH_BINARY)

        # Optional: construct filename for saving the thresholded binary image
        name = os.path.splitext(impath)[0] + "_%s.tiff" % t
        # cv2.imwrite(name, img_array)  # Uncomment to save thresholded image

        # Compute dispersion statistics across image slices
        ID1, std1 = create_slices(img_array, no_slices)

        # Store results
        ID.append(ID1)
        std.append(std1)

    # Combine results into a DataFrame
    df = pd.DataFrame(np.column_stack([ID, std, thresh_quantile]),
                      columns=['ID', 'std', 'thresh'])

    # Add the filename column for traceability
    df["filename"] = os.path.split(impath)[1]

    # Set the filename as the DataFrame index
    df = df.set_index('filename')

    return df


def thresh_folder(folder_path, no_slices):
    """
    Processes all .tif images in a folder to compute threshold-based Index of Dispersion (ID) curves.

    For each image, the function applies a series of intensity thresholds, performs binary segmentation,
    computes the Index of Dispersion (ID) and standard deviation for each threshold, and collects the
    results into a combined DataFrame.

    Parameters:
    -----------
    folder_path : str
        Path to the folder containing .tif images.
    no_slices : int
        Number of vertical and horizontal slices to divide each image into (e.g., 4 results in 4x4 = 16 slices).

    Returns:
    --------
    full_df : pandas.DataFrame
        A concatenated DataFrame containing threshold values, ID, and standard deviation
        for each image in the folder. Includes the filename as an index.
    """

    filename = []  # Initialise list to store .tif file names
    count = 0      # Count of valid .tif images

    # Loop through all files in the folder
    for f in listdir(folder_path):
        # Validation check: ensures item is a file and ends with '.tif'
        if isfile(join(folder_path, f)) and '.tif' in f:
            count += 1
            filename.append(f)  # Add valid image to the list

    full_df = []  # List to hold DataFrames for each image

    for file in filename:
        # Construct full file path
        full_path = join(folder_path, file)

        # Run threshold curve analysis on the image
        # Returns a DataFrame with 'ID', 'std', and 'thresh' columns
        full_df1 = thresh_curve(full_path, no_slices)

        # Append the result to the list
        full_df.append(full_df1)

    # Combine all individual image DataFrames into a single DataFrame
    full_df = pd.concat(full_df)

    # Return the final merged DataFrame
    return full_df


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

def plot_stacked_id_vs_cycle(dataframes, labels, colors=None, markers=None, 
                              title="stacked_plot", ylabel="Index of Dispersion", output_path=None):
    """
    Plot vertically stacked line plots showing average Index of Dispersion (ID)
    across different conditions or cycles, with shaded areas representing standard deviation.

    Parameters:
    - dataframes: list of pd.DataFrame objects.
        Each DataFrame must have columns: 
        'condition' (e.g., cycle number), 'avg_max_ID', and 'std_max_ID'.
    - labels: list of strings to label each subplot (same length as dataframes).
    - colors: optional list of colors for each plot line (default: viridis colormap).
    - markers: optional list of markers for data points (default: unique markers).
    - title: string, title of the figure or filename prefix if saving.
    - ylabel: string, y-axis label for the whole figure.
    - output_path: optional full file path (including filename) to save the figure.
    """

    n = len(dataframes)  # Number of plots to stack vertically

    # Generate default color list if none provided
    if colors is None:
        colors = [cm.viridis(i / n) for i in range(n)]
    
    # Use default marker styles if none provided
    if markers is None:
        markers = ['o', 's', '^', 'D', 'v', '*', 'X'][:n]

    # Create n vertically stacked subplots, sharing x-axis
    fig, axes = plt.subplots(n, 1, figsize=(8, 4 * n), sharex=True)

    # If only one subplot, wrap it in a list for consistent iteration
    if n == 1:
        axes = [axes]

    # Plot each dataset in its corresponding subplot
    for ax, df, label, color, marker in zip(axes, dataframes, labels, colors, markers):
        # Plot average ID vs. condition/cycle, with line and marker
        ax.plot(df["condition"], df["avg_max_ID"], marker=marker, markersize=10,
                mec='k', label=label, linestyle=':', color=color)

        # Shade ±1 standard deviation around the average
        ax.fill_between(df["condition"],
                        df["avg_max_ID"] - df["std_max_ID"],
                        df["avg_max_ID"] + df["std_max_ID"],
                        color=color, alpha=0.4)

        # Add legend with no frame
        ax.legend(loc='upper right', bbox_to_anchor=(0.4, 1), frameon=False)

        # Remove individual y-axis labels (we use a shared one instead)
        ax.set_ylabel("")

    # Label x-axis on the bottom plot only
    axes[-1].set_xlabel('Capacity (mAh/cm²)')

    # Shared y-axis label for the entire figure
    fig.text(0.001, 0.5, ylabel, va='center', rotation='vertical', fontsize=24)

    # Save figure if output path is provided
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')

    # Display the plot
    plt.show()
