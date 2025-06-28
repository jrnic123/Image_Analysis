# Image_Analysis
 Code used to carry out image analysis from Nicolas et al. 2025 (in review).

# Abstract
(To be added upon publication)
----------------------------

# Functions
indiv_ID(img, no_slices)
Calculates the Index of Dispersion (ID) and standard deviation of white pixel coverage from a binary (black-and-white) image divided into equal-sized grid slices. 

create_slices(img, no_slices)
Same as indiv_ID, but operates on a NumPy image array that has already been loaded and optionally thresholded. Used internally by other functions for reusability.

thresh_curve_indiv_img(impath, no_slices)
Applies a series of brightness quantile thresholds to a grayscale image, computes ID and standard deviation for each, and plots ID vs. threshold. Returns a DataFrame summarizing the results.

thresh_curve(impath, no_slices)
Performs the same threshold-curve analysis as thresh_curve_indiv_img but without plotting. Designed for use in batch processing workflows.

thresh_folder(folder_path, no_slices)
Applies thresh_curve to all .tif images in a folder and concatenates the results into a single DataFrame for comparison and analysis.

summarize_max_id(df, condition)
Extracts the maximum ID value for each image in a DataFrame and returns the average and standard deviation of those maxima, labeled by a specified experimental condition.

append_dataframes(df_list, ignore_index=True)
Concatenates a list of DataFrames into one unified DataFrame, optionally resetting the index. Simplifies merging results across conditions or folders.

plot_stacked_id_vs_cycle(dataframes, labels, ...)
Generates a vertically stacked plot of average max ID values across multiple experimental conditions, with shaded regions showing ±1 reflecting the global uniformity. 
