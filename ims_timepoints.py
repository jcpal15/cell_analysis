import numpy as np
import h5py
from skimage import filters, measure, exposure
import matplotlib.pyplot as plt
from scipy import ndimage, optimize, stats
import os
import pandas as pd
from tqdm import tqdm  # Import tqdm for progress bars
import datetime
from pathlib import Path
import glob
import re

def extract_identifier(filename):
    """Extract identifier from the end of the filename (e.g. 'F0' from 'baseline-try3_1_F0.ims')"""
    # Get the stem (filename without extension)
    stem = Path(filename).stem
    
    # Try to find an underscore followed by one or more characters at the end
    match = re.search(r'_([^_]+)$', stem)
    if match:
        return match.group(1)
    else:
        # If no underscore pattern found, just return the last 2-3 characters
        return stem[-3:] if len(stem) > 3 else stem

def extract_folder_name(folder_path):
    """Extract just the folder name from a full path"""
    return os.path.basename(os.path.normpath(folder_path))
    
def exponential_decay(x, a, b, c):
    """Exponential decay function for fitting: f(x) = a * exp(-b * x) + c"""
    return a * np.exp(-b * x) + c

def linear_model(x, a, b):
    """Linear model function for fitting: f(x) = a * x + b"""
    return a * x + b

def polynomial_model(x, *params):
    """Polynomial model function for fitting: f(x) = a + b*x + c*x^2 + ...
    The order is determined by the number of parameters.
    """
    result = 0
    for i, a in enumerate(params):
        result += a * (x ** i)
    return result

def fit_bleaching_model(time_points, intensities, model_type='exponential', poly_order=2):
    """
    Fit a model to the bleaching curve
    
    Parameters:
    -----------
    time_points : array-like
        The time points
    intensities : array-like
        The intensity values at each time point
    model_type : str
        The type of model to fit ('exponential', 'linear', or 'polynomial')
    poly_order : int
        The order of the polynomial (only used if model_type='polynomial')
        
    Returns:
    --------
    model_params : tuple
        The fitted model parameters
    """
    # Convert to numpy arrays if they aren't already
    time_points = np.array(time_points)
    intensities = np.array(intensities)
    
    # Remove any NaN values
    valid_indices = ~np.isnan(intensities)
    time_points = time_points[valid_indices]
    intensities = intensities[valid_indices]
    
    if len(time_points) < 2:
        print("Warning: Not enough data points for fitting. Returning default parameters.")
        if model_type == 'exponential':
            return (1.0, 0.0, intensities[0] if len(intensities) > 0 else 1.0)
        elif model_type == 'linear':
            return (0.0, intensities[0] if len(intensities) > 0 else 1.0)
        else:  # polynomial
            return tuple([intensities[0] if len(intensities) > 0 else 1.0] + [0.0] * poly_order)
    
    try:
        if model_type == 'exponential':
            # Initial guess for parameters (a, b, c)
            # a: amplitude (max - min), b: decay rate, c: offset
            max_val = np.max(intensities)
            min_val = np.min(intensities)
            p0 = (max_val - min_val, 0.1, min_val)
            
            # Fit the model
            popt, _ = optimize.curve_fit(exponential_decay, time_points, intensities, p0=p0, maxfev=10000)
            return popt
        
        elif model_type == 'linear':
            # Use linear regression for linear model
            slope, intercept, _, _, _ = stats.linregress(time_points, intensities)
            return (slope, intercept)
        
        else:  # polynomial
            # Fit a polynomial of the specified order
            popt = np.polyfit(time_points, intensities, poly_order)
            return tuple(reversed(popt))  # Reverse to match the polynomial_model function parameter order
    
    except Exception as e:
        print(f"Error fitting model: {e}")
        print("Using default parameters instead.")
        if model_type == 'exponential':
            return (1.0, 0.0, intensities[0])
        elif model_type == 'linear':
            return (0.0, intensities[0])
        else:  # polynomial
            return tuple([intensities[0]] + [0.0] * poly_order)

def calculate_correction_factors(time_points, model_params, model_type='exponential', poly_order=2, normalization_point=0):
    """
    Calculate correction factors based on the fitted model
    
    Parameters:
    -----------
    time_points : array-like
        The time points
    model_params : tuple
        The fitted model parameters
    model_type : str
        The type of model ('exponential', 'linear', or 'polynomial')
    poly_order : int
        The order of the polynomial (only used if model_type='polynomial')
    normalization_point : int
        The time point to normalize to (default: 0, i.e., the first time point)
        
    Returns:
    --------
    correction_factors : array
        The correction factors for each time point
    """
    # Convert to numpy array if it isn't already
    time_points = np.array(time_points)
    
    # Calculate the predicted intensities at each time point
    if model_type == 'exponential':
        predicted = exponential_decay(time_points, *model_params)
    elif model_type == 'linear':
        predicted = linear_model(time_points, *model_params)
    else:  # polynomial
        predicted = polynomial_model(time_points, *model_params)
    
    # Calculate the intensity at the normalization point
    if model_type == 'exponential':
        normalization_value = exponential_decay(normalization_point, *model_params)
    elif model_type == 'linear':
        normalization_value = linear_model(normalization_point, *model_params)
    else:  # polynomial
        normalization_value = polynomial_model(normalization_point, *model_params)
    
    # Calculate correction factors (normalization_value / predicted)
    correction_factors = np.ones_like(time_points, dtype=float)
    valid_indices = (predicted > 0)  # Avoid division by zero
    correction_factors[valid_indices] = normalization_value / predicted[valid_indices]
    
    return correction_factors

def apply_bleaching_correction(df, model_type='exponential', poly_order=2, normalization_point=0):
    """
    Apply bleaching correction to a dataframe of cell data
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the cell data
    model_type : str
        The type of model to fit ('exponential', 'linear', or 'polynomial')
    poly_order : int
        The order of the polynomial (only used if model_type='polynomial')
    normalization_point : int
        The time point to normalize to (default: 0, i.e., the first time point)
        
    Returns:
    --------
    df_corrected : pandas.DataFrame
        The dataframe with corrected values
    model_params : tuple
        The fitted model parameters
    correction_factors : array
        The correction factors for each time point
    """
    # Create a copy of the dataframe to avoid modifying the original
    df_corrected = df.copy()
    
    # Group by time point
    time_groups = df.groupby('time_point')
    
    # Calculate mean intensities per time point
    mean_ref = time_groups['reference_intensity'].mean()
    
    # Get time points and intensities as arrays
    time_points = np.array(mean_ref.index)
    intensities = np.array(mean_ref.values)
    
    # Fit the model to the bleaching curve
    model_params = fit_bleaching_model(time_points, intensities, model_type, poly_order)
    
    # Calculate correction factors
    correction_factors = calculate_correction_factors(time_points, model_params, model_type, poly_order, normalization_point)
    
    # Apply correction factors to the reference channel
    for time_point, factor in zip(time_points, correction_factors):
        df_corrected.loc[df_corrected['time_point'] == time_point, 'reference_intensity_corrected'] = \
            df_corrected.loc[df_corrected['time_point'] == time_point, 'reference_intensity'] * factor
    
    # Calculate corrected ratio
    df_corrected['ratio_corrected'] = df_corrected['measurement_intensity'] / df_corrected['reference_intensity_corrected']
    
    # Handle any invalid values (NaN, Inf) in the corrected ratio
    df_corrected['ratio_corrected'].replace([np.inf, -np.inf], np.nan, inplace=True)
    
    return df_corrected, model_params, correction_factors
    
def analyze_multichannel_depletion_h5(h5_path, ref_channel=1, meas_channel=0, 
                                    percentile_low=0.5, percentile_high=99.5,
                                    crop_size=50, min_cell_size=100,
                                    resolution_level=0, z_slice=0,
                                    threshold_method=None, save_cell_images=True,
                                    bleach_correction=True, bleach_model='exponential',
                                    poly_order=2, normalization_point=0):
    
    # Extract filename without extension
    file_name = Path(h5_path).stem
    
    # Get current time and date
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directories
    output_dir = f'cell_analysis_results/{file_name}_analysis_{current_time}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Only create cell_crops directory if needed
    cell_crops_dir = None
    if save_cell_images:
        cell_crops_dir = f'cell_crops/{file_name}_cell_crops_{current_time}'
        if not os.path.exists(cell_crops_dir):
            os.makedirs(cell_crops_dir)
    
    try:
        # Open h5 file
        with h5py.File(h5_path, 'r') as f:
            # Get number of time points
            time_points_path = f'DataSet/ResolutionLevel {resolution_level}'
            num_time_points = len([k for k in f[time_points_path].keys() if k.startswith('TimePoint')])
            
            # Check the first time point to get data dimensions
            test_path = f'DataSet/ResolutionLevel {resolution_level}/TimePoint 0/Channel {ref_channel}/Data'
            data_shape = f[test_path].shape
            
            print(f"Data shape: {data_shape}")
            
            # Validate z_slice is within range
            max_z = data_shape[0] - 1  # Subtract 1 because indices start at 0
            if z_slice > max_z:
                print(f"Warning: Requested z_slice {z_slice} is out of range. Using z_slice = 0 instead.")
                z_slice = 0
                
            print(f"Using z_slice: {z_slice}")
            
            # Create data structures to store results over time
            all_time_data = []
            cells_over_time = {}
            
            # Process each time point
            for time_point in tqdm(range(num_time_points), desc="Processing time points"):
                # Load reference and measurement channels for this time point
                ref_path = f'DataSet/ResolutionLevel {resolution_level}/TimePoint {time_point}/Channel {ref_channel}/Data'
                meas_path = f'DataSet/ResolutionLevel {resolution_level}/TimePoint {time_point}/Channel {meas_channel}/Data'
                
                reference_channel = f[ref_path][z_slice, :, :]
                measurement_channel = f[meas_path][z_slice, :, :]
                
                # Use the first time point for cell segmentation
                if time_point == 0:
                    # Enhanced preprocessing
                    print("Applying Gaussian smoothing...")
                    smoothed = filters.gaussian(reference_channel, sigma=2.0)
                    
                    # Enhance contrast
                    print("Enhancing image contrast...")
                    p_low, p_high = np.percentile(smoothed, (percentile_low, percentile_high))
                    reference_enhanced = exposure.rescale_intensity(smoothed, in_range=(p_low, p_high))
                    
                    # Apply threshold based on provided method
                    if threshold_method == 'otsu':
                        print("Calculating Otsu threshold...")
                        thresh = filters.threshold_otsu(reference_enhanced)
                    elif threshold_method == 'li':
                        print("Calculating Li threshold...")
                        thresh = filters.threshold_li(reference_enhanced)
                    else:
                        try:
                            # Try to interpret as a float value
                            thresh = float(threshold_method)
                        except (ValueError, TypeError):
                            print("Invalid threshold method, using Otsu's method")
                            thresh = filters.threshold_otsu(reference_enhanced)
                    
                    print(f"Threshold value: {thresh}")
                    
                    # Create binary mask with progress indication
                    print("Creating binary mask...")
                    binary = reference_enhanced > thresh
                    
                    print("Filling holes...")
                    binary = ndimage.binary_fill_holes(binary)
                    
                    # Morphological operations to clean up mask
                    print("Applying morphological operations...")
                    binary = ndimage.binary_closing(binary, iterations=3)
                    binary = ndimage.binary_opening(binary, iterations=2)
                    binary = ndimage.binary_fill_holes(binary)
                    binary = ndimage.binary_opening(binary, structure=np.ones((3,3)))
                    
                    # Label regions
                    print("Labeling regions...")
                    labels = measure.label(binary)
                    
                    # Get region properties
                    print("Calculating region properties...")
                    props = measure.regionprops(labels, reference_channel)
                    
                    # Filter regions based on criteria
                    print(f"Filtering regions from {len(props)} initial detections...")
                    valid_labels = []
                    
                    for prop in tqdm(props, desc="Filtering cells"):
                        # Skip cells touching the image edges
                        if (prop.bbox[0] == 0 or prop.bbox[1] == 0 or 
                            prop.bbox[2] == binary.shape[0] or prop.bbox[3] == binary.shape[1]):
                            continue
                            
                        # Skip cells that are too small or too large
                        if prop.area < min_cell_size or prop.area > min_cell_size * 10:
                            continue
                            
                        # Skip highly elongated cells
                        if prop.eccentricity > 0.8:
                            continue
                            
                        # Skip cells with low solidity (irregular shapes)
                        if prop.solidity < 0.8:
                            continue
                            
                        # Skip cells that are too close to other cells
                        centroid = prop.centroid
                        too_close = False
                        for other_prop in props:
                            if other_prop.label != prop.label:
                                dist = np.sqrt((centroid[0] - other_prop.centroid[0])**2 + 
                                             (centroid[1] - other_prop.centroid[1])**2)
                                if dist < crop_size/2:  # If centers are closer than half crop size
                                    too_close = True
                                    break
                        if too_close:
                            continue
                            
                        valid_labels.append(prop.label)
                    
                    print(f"Found {len(valid_labels)} valid cells after filtering")
                    
                    # Create new labels image with only valid cells
                    print("Creating filtered label mask...")
                    filtered_labels = np.zeros_like(labels)
                    for label in valid_labels:
                        filtered_labels[labels == label] = label
                    
                    # Save this mask for future use
                    # We'll use the same cells for all time points
                    master_labels = filtered_labels
                    
                    # Save the segmentation mask with unique filename
                    segmentation_mask_filename = f"{file_name}_segmentation_mask_{current_time}.png"
                    print(f"Saving segmentation mask as {segmentation_mask_filename}...")
                    plt.figure(figsize=(10, 10))
                    plt.imshow(filtered_labels, cmap='nipy_spectral')
                    plt.colorbar(label='Cell ID')
                    plt.title(f'Cell Segmentation Mask - {file_name}')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, segmentation_mask_filename))
                    plt.close()
                    
                    # Only save individual cell images if requested
                    if save_cell_images:
                        # Crop and save individual cell images
                        print("Creating individual cell images...")
                        
                        # Prepare for cropping
                        half_crop = crop_size // 2
                        
                        # Pad images
                        pad_width = ((half_crop, half_crop), (half_crop, half_crop))
                        ref_padded = np.pad(reference_channel, pad_width, mode='constant')
                        meas_padded = np.pad(measurement_channel, pad_width, mode='constant')
                        
                        # Get filtered region properties
                        valid_props = measure.regionprops(master_labels, reference_channel)
                        meas_props = measure.regionprops(master_labels, measurement_channel)
                        
                        for i, prop in enumerate(tqdm(valid_props, desc="Saving cell images")):
                            if prop.label == 0:  # Skip background
                                continue
                                
                            # Get padded coordinates
                            y, x = [int(c) + half_crop for c in prop.centroid]
                            
                            # Extract crops
                            ref_crop = ref_padded[y-half_crop:y+half_crop, x-half_crop:x+half_crop]
                            meas_crop = meas_padded[y-half_crop:y+half_crop, x-half_crop:x+half_crop]
                            
                            # Calculate intensity statistics for each crop
                            ref_stats = {
                                'mean': np.mean(ref_crop),
                                'max': np.max(ref_crop),
                                'total': np.sum(ref_crop)
                            }
                            
                            meas_stats = {
                                'mean': np.mean(meas_crop),
                                'max': np.max(meas_crop),
                                'total': np.sum(meas_crop)
                            }
                            
                            # Create figure with intensity information
                            plt.figure(figsize=(12, 6))
                            
                            # Reference channel
                            plt.subplot(1, 2, 1)
                            plt.imshow(ref_crop)
                            plt.title('Reference Channel\n' + 
                                    f'Mean: {ref_stats["mean"]:.1f}\n' +
                                    f'Max: {ref_stats["max"]:.1f}\n' +
                                    f'Total: {ref_stats["total"]:.1f}')
                            plt.axis('off')
                            
                            # Measurement channel
                            plt.subplot(1, 2, 2)
                            plt.imshow(meas_crop)
                            plt.title('Measurement Channel\n' + 
                                    f'Mean: {meas_stats["mean"]:.1f}\n' +
                                    f'Max: {meas_stats["max"]:.1f}\n' +
                                    f'Total: {meas_stats["total"]:.1f}')
                            plt.axis('off')
                            
                            plt.suptitle(f'Cell {prop.label} (Area: {prop.area} px)')
                            plt.tight_layout()
                            plt.savefig(os.path.join(cell_crops_dir, f'cell_{prop.label}.png'))
                            plt.close()
                    else:
                        print("Skipping individual cell image generation as requested")
                
                # Get properties for each time point using the master labels
                props = measure.regionprops(master_labels, reference_channel)
                meas_props = measure.regionprops(master_labels, measurement_channel)
                
                # Process each cell at this time point
                for i, prop in enumerate(props):
                    if prop.label == 0:  # Skip background
                        continue
                    
                    cell_id = prop.label
                    
                    # Calculate measurements
                    ref_intensity = prop.mean_intensity
                    meas_intensity = meas_props[i].mean_intensity
                    
                    # Calculate ratio
                    ratio = meas_intensity / ref_intensity if ref_intensity > 0 else 0
                    
                    # Store data for this cell
                    cell_data = {
                        'time_point': time_point,
                        'cell_id': cell_id,
                        'area': prop.area,
                        'centroid_y': prop.centroid[0],
                        'centroid_x': prop.centroid[1],
                        'reference_intensity': ref_intensity,
                        'measurement_intensity': meas_intensity,
                        'ratio': ratio
                    }
                    
                    all_time_data.append(cell_data)
                    
                    # Also organize by cell ID for time course analysis
                    if cell_id not in cells_over_time:
                        cells_over_time[cell_id] = {
                            'time_points': [],
                            'reference_intensities': [],
                            'measurement_intensities': [],
                            'ratios': []
                        }
                    
                    cells_over_time[cell_id]['time_points'].append(time_point)
                    cells_over_time[cell_id]['reference_intensities'].append(ref_intensity)
                    cells_over_time[cell_id]['measurement_intensities'].append(meas_intensity)
                    cells_over_time[cell_id]['ratios'].append(ratio)
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(all_time_data)
        
        # Apply bleaching correction if requested
        if bleach_correction:
            print("Applying photobleaching correction to reference channel...")
            df_corrected, model_params, correction_factors = apply_bleaching_correction(
                df, bleach_model, poly_order, normalization_point
            )
            
            # Replace the original DataFrame with the corrected one
            df = df_corrected
            
            # Save the correction factors and model parameters
            correction_info = pd.DataFrame({
                'time_point': np.arange(num_time_points),
                'correction_factor': correction_factors if len(correction_factors) >= num_time_points else 
                                     np.pad(correction_factors, (0, num_time_points - len(correction_factors)), 'constant', constant_values=1.0)
            })
            
            # Add the model parameters as a dictionary in the info
            model_info = {
                'model_type': bleach_model,
                'poly_order': poly_order if bleach_model == 'polynomial' else None,
                'normalization_point': normalization_point
            }
            
            # Add model parameters to the info
            if bleach_model == 'exponential':
                model_info.update({
                    'amplitude': model_params[0],
                    'decay_rate': model_params[1],
                    'offset': model_params[2]
                })
            elif bleach_model == 'linear':
                model_info.update({
                    'slope': model_params[0],
                    'intercept': model_params[1]
                })
            else:  # polynomial
                for i, param in enumerate(model_params):
                    model_info[f'coef_{i}'] = param
            
            # Save the correction info to a CSV
            correction_csv = os.path.join(output_dir, f"{file_name}_bleaching_correction_info_{current_time}.csv")
            correction_info.to_csv(correction_csv, index=False)
            
            # Save the model info to a separate CSV
            model_info_df = pd.DataFrame([model_info])
            model_csv = os.path.join(output_dir, f"{file_name}_bleaching_model_info_{current_time}.csv")
            model_info_df.to_csv(model_csv, index=False)
            
            print(f"Photobleaching correction applied using {bleach_model} model.")
            print(f"Correction info saved to {correction_csv}")
            print(f"Model info saved to {model_csv}")
            
            # Create a plot showing the bleaching curve and the fitted model
            plt.figure(figsize=(12, 8))
            
            # Group data by time point for plotting
            time_groups = df.groupby('time_point')
            
            # Original reference channel
            mean_ref_orig = time_groups['reference_intensity'].mean()
            std_ref_orig = time_groups['reference_intensity'].std()
            time_points = np.array(mean_ref_orig.index)
            
            # Corrected reference channel
            mean_ref_corr = time_groups['reference_intensity_corrected'].mean()
            std_ref_corr = time_groups['reference_intensity_corrected'].std()
            
            # Fit curve
            if bleach_model == 'exponential':
                fit_curve = exponential_decay(time_points, *model_params)
            elif bleach_model == 'linear':
                fit_curve = linear_model(time_points, *model_params)
            else:  # polynomial
                fit_curve = polynomial_model(time_points, *model_params)
            
            # Plot original data with fit curve
            plt.errorbar(time_points, mean_ref_orig, yerr=std_ref_orig, fmt='bo-', label='Original Reference', alpha=0.7)
            plt.plot(time_points, fit_curve, 'r-', label=f'Fitted {bleach_model.capitalize()} Model', linewidth=2)
            plt.errorbar(time_points, mean_ref_corr, yerr=std_ref_corr, fmt='go-', label='Corrected Reference', alpha=0.7)
            
            plt.title(f'Reference Channel Bleaching Correction - {file_name}', fontsize=16)
            plt.xlabel('Time Point', fontsize=14)
            plt.ylabel('Mean Intensity', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(fontsize=12)
            
            # Save the plot
            bleach_plot = os.path.join(output_dir, f"{file_name}_bleaching_correction_plot_{current_time}.png")
            plt.savefig(bleach_plot, dpi=300)
            plt.close()
            print(f"Bleaching correction plot saved to {bleach_plot}")
        
        # Save the complete dataset with unique filename
        csv_filename = f"{file_name}_cell_time_course_data_{current_time}.csv"
        print(f"Saving data to CSV as {csv_filename}...")
        df.to_csv(os.path.join(output_dir, csv_filename), index=False)
        
        # Create summary plots
        print("Creating summary plots...")
        
        # Group data by time point
        time_groups = df.groupby('time_point')
        
        # Calculate mean intensities and ratios per time point
        mean_ref = time_groups['reference_intensity'].mean()
        std_ref = time_groups['reference_intensity'].std()
        mean_meas = time_groups['measurement_intensity'].mean()
        std_meas = time_groups['measurement_intensity'].std()
        
        # Get ratio data - use corrected if available
        if bleach_correction:
            mean_ratios = time_groups['ratio_corrected'].mean()
            std_ratios = time_groups['ratio_corrected'].std()
            mean_ref_corr = time_groups['reference_intensity_corrected'].mean()
            std_ref_corr = time_groups['reference_intensity_corrected'].std()
        else:
            mean_ratios = time_groups['ratio'].mean()
            std_ratios = time_groups['ratio'].std()
        
        # 1. Plot showing intensity of each channel over time with unique filename
        intensities_filename = f"{file_name}_channel_intensities_{current_time}.png"
        plt.figure(figsize=(10, 6))
        
        plt.errorbar(mean_ref.index, mean_ref, yerr=std_ref, fmt='b-o', label='Reference Channel')
        
        if bleach_correction:
            plt.errorbar(mean_ref_corr.index, mean_ref_corr, yerr=std_ref_corr, fmt='g-o', label='Reference (Corrected)')
            
        plt.errorbar(mean_meas.index, mean_meas, yerr=std_meas, fmt='r-o', label='Measurement Channel')
        
        plt.title(f'Mean Channel Intensities Over Time - {file_name}')
        plt.xlabel('Time Point')
        plt.ylabel('Mean Intensity')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, intensities_filename))
        plt.close()
        
        # 2. Plot showing intensity ratio over time with unique filename
        ratio_filename = f"{file_name}_mean_ratio_{current_time}.png"
        plt.figure(figsize=(10, 6))
        plt.errorbar(mean_ratios.index, mean_ratios, yerr=std_ratios, fmt='g-o')
        ratio_title = 'Mean Ratio Over Time (All Cells)'
        if bleach_correction:
            ratio_title += ' - Bleach Corrected'
        plt.title(f'{ratio_title} - {file_name}')
        plt.xlabel('Time Point')
        plt.ylabel('Mean Ratio (Measurement/Reference)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, ratio_filename))
        plt.close()
        
        # Print info about results location
        output_info = f"- {output_dir} (summary data and plots)"
        if save_cell_images:
            output_info += f"\n- {cell_crops_dir} (individual cell images)"
            
        print(f"Analysis complete! Results saved in:\n{output_info}")
        
        # Return the dataframe, including identifier and output paths
        return {
            'df': df,
            'identifier': extract_identifier(h5_path),
            'output_dir': output_dir,
            'original_path': h5_path,
            'mean_ratios': mean_ratios,
            'std_ratios': std_ratios,
            'mean_ref': mean_ref,
            'std_ref': std_ref,
            'mean_meas': mean_meas,
            'std_meas': std_meas,
            'time_points': list(mean_ratios.index),
            'num_time_points': num_time_points,
            'bleach_corrected': bleach_correction,
            'mean_ref_corr': mean_ref_corr if bleach_correction else None,
            'std_ref_corr': std_ref_corr if bleach_correction else None
        }
        
    except Exception as e:
        print(f"Error in analysis of {h5_path}: {e}")
        # Print a more detailed traceback
        import traceback
        traceback.print_exc()
        return None

def process_ims_folder(folder_path, ref_channel=1, meas_channel=0, 
                      resolution_level=0, z_slice=0, 
                      save_cell_images=True, bleach_correction=True,
                      bleach_model='exponential', poly_order=2,
                      normalization_point=0):
    """
    Process all IMS files in a folder and create combined analysis
    """
    # Extract the folder name for use in outputs
    folder_name = extract_folder_name(folder_path)
    
    # Find all IMS files
    ims_files = glob.glob(os.path.join(folder_path, "*.ims"))
    
    if not ims_files:
        print(f"No IMS files found in {folder_path}")
        return
    
    print(f"Found {len(ims_files)} IMS files to process in folder: {folder_name}")
    for i, file in enumerate(ims_files):
        print(f"{i+1}. {os.path.basename(file)}")
    
    # Ask for threshold method once and apply to all files
    threshold_choice = input("Enter threshold method for ALL files (otsu/li) or a manual value (0-1): ")
    
    # Create timestamp for this batch analysis
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create folder-named output directory
    folder_output_dir = f'{folder_name}_analysis_{current_time}'
    if not os.path.exists(folder_output_dir):
        os.makedirs(folder_output_dir)
    
    # Process each file and collect results
    all_results = []
    
    for file_path in ims_files:
        print(f"\n{'='*50}")
        print(f"Processing {os.path.basename(file_path)}...")
        print(f"{'='*50}")
        
        result = analyze_multichannel_depletion_h5(
            file_path, 
            ref_channel=ref_channel,
            meas_channel=meas_channel,
            resolution_level=resolution_level,
            z_slice=z_slice,
            threshold_method=threshold_choice,
            save_cell_images=save_cell_images,
            bleach_correction=bleach_correction,
            bleach_model=bleach_model,
            poly_order=poly_order,
            normalization_point=normalization_point
        )
        
        if result is not None:
            all_results.append(result)
    
    # Check if we have results to plot
    if not all_results:
        print("No successful analyses to combine")
        return
    
    # Create combined ratio plot with individual file data
    create_combined_ratio_plot(all_results, current_time, folder_name, folder_output_dir, bleach_correction)
    
    # Create averaged ratio plot across all files
    create_averaged_ratio_plot(all_results, current_time, folder_name, folder_output_dir, bleach_correction)
    
    # Create averaged channel plots across all files
    create_averaged_channel_plots(all_results, current_time, folder_name, folder_output_dir, bleach_correction)
    
    print(f"\nBatch processing complete! Combined results saved in: {folder_output_dir}")

def create_combined_ratio_plot(all_results, timestamp, folder_name, output_dir, bleach_corrected=True):
    """
    Create a plot showing ratio data from all analyzed files
    """
    print("\nCreating combined ratio plot...")
    
    # Create figure for ratio plot
    plt.figure(figsize=(12, 8))
    
    # Use a different color for each dataset
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
    
    # Plot each ratio line
    for i, result in enumerate(all_results):
        color = colors[i % len(colors)]
        identifier = result['identifier']
        mean_ratios = result['mean_ratios']
        std_ratios = result['std_ratios']
        
        # Plot with error bars
        plt.errorbar(
            mean_ratios.index, 
            mean_ratios, 
            yerr=std_ratios, 
            fmt=f'-o', 
            color=color,
            label=identifier,
            capsize=4,
            markersize=6,
            linewidth=2,
            elinewidth=1,
            alpha=0.8
        )
    
    # Add plot details
    title = f'Comparison of Mean Ratios - {folder_name}'
    if bleach_corrected:
        title += " (Bleach Corrected)"
    plt.title(title, fontsize=16)
    plt.xlabel('Time Point', fontsize=14)
    plt.ylabel('Mean Ratio (Measurement/Reference)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Sample ID', loc='best', fontsize=12)
    
    # Enhance the plot appearance
    plt.tight_layout()
    
    # Save the combined plot
    label = 'corrected' if bleach_corrected else 'orig'
    output_file = os.path.join(output_dir, f'{folder_name}_combined_ratio_plot_{label}_{timestamp}.png')
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    print(f"Combined ratio plot saved as {output_file}")
    
    # Also save the data as CSV for future reference
    combined_data = pd.DataFrame()
    
    for result in all_results:
        identifier = result['identifier']
        mean_ratios = result['mean_ratios']
        std_ratios = result['std_ratios']
        
        # Create a dataframe for this result
        df = pd.DataFrame({
            'sample': identifier,
            'time_point': mean_ratios.index,
            'mean_ratio': mean_ratios.values,
            'std_ratio': std_ratios.values
        })
        
        combined_data = pd.concat([combined_data, df])
    
    # Save the combined data
    output_csv = os.path.join(output_dir, f'{folder_name}_combined_ratio_data_{label}_{timestamp}.csv')
    combined_data.to_csv(output_csv, index=False)
    print(f"Combined ratio data saved as {output_csv}")

def create_averaged_ratio_plot(all_results, timestamp, folder_name, output_dir, bleach_corrected=True):
    """
    Create a plot showing the average ratio data across all files
    """
    print("\nCreating averaged ratio plot across all files...")
    
    # First, determine the maximum number of time points
    max_time_points = max([result['num_time_points'] for result in all_results])
    
    # Create arrays to store data for each time point
    all_means = [[] for _ in range(max_time_points)]
    
    # Collect the ratio values for each time point
    for result in all_results:
        mean_ratios = result['mean_ratios']
        
        for time_point in range(max_time_points):
            if time_point in mean_ratios:
                all_means[time_point].append(mean_ratios[time_point])
    
    # Calculate the mean and standard deviation across all files for each time point
    time_points = list(range(max_time_points))
    avg_means = [np.mean(means) if means else np.nan for means in all_means]
    std_means = [np.std(means) if len(means) > 1 else 0 for means in all_means]
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    plt.errorbar(
        time_points,
        avg_means,
        yerr=std_means,
        fmt='-o',
        color='black',
        capsize=4,
        markersize=8,
        linewidth=2,
        elinewidth=1
    )
    
    # Add plot details
    title = f'Average Ratio Across All Samples - {folder_name}'
    if bleach_corrected:
        title += " (Bleach Corrected)"
    plt.title(title, fontsize=16)
    plt.xlabel('Time Point', fontsize=14)
    plt.ylabel('Mean Ratio (Measurement/Reference)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Enhance the plot appearance
    plt.tight_layout()
    
    # Save the averaged plot
    label = 'corrected' if bleach_corrected else 'orig'
    output_file = os.path.join(output_dir, f'{folder_name}_averaged_ratio_plot_{label}_{timestamp}.png')
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    print(f"Averaged ratio plot saved as {output_file}")
    
    # Save the averaged data as CSV
    avg_data = pd.DataFrame({
        'time_point': time_points,
        'average_ratio': avg_means,
        'std_ratio': std_means,
        'num_samples': [len(means) for means in all_means]
    })
    
    output_csv = os.path.join(output_dir, f'{folder_name}_averaged_ratio_data_{label}_{timestamp}.csv')
    avg_data.to_csv(output_csv, index=False)
    print(f"Averaged ratio data saved as {output_csv}")

def create_averaged_channel_plots(all_results, timestamp, folder_name, output_dir, bleach_corrected=True):
    """
    Create plots showing the average reference and measurement channel intensities
    across all files
    """
    print("\nCreating averaged channel plots across all files...")
    
    # First, determine the maximum number of time points
    max_time_points = max([result['num_time_points'] for result in all_results])
    
    # Create arrays to store data for each time point
    all_ref_means = [[] for _ in range(max_time_points)]
    all_meas_means = [[] for _ in range(max_time_points)]
    
    # If bleach correction is enabled, also collect corrected reference data
    if bleach_corrected:
        all_ref_corr_means = [[] for _ in range(max_time_points)]
    
    # Collect the channel intensity values for each time point
    for result in all_results:
        mean_ref = result['mean_ref']
        mean_meas = result['mean_meas']
        
        for time_point in range(max_time_points):
            if time_point in mean_ref:
                all_ref_means[time_point].append(mean_ref[time_point])
            if time_point in mean_meas:
                all_meas_means[time_point].append(mean_meas[time_point])
                
        # Collect corrected reference data if available
        if bleach_corrected and result.get('mean_ref_corr') is not None:
            mean_ref_corr = result['mean_ref_corr']
            for time_point in range(max_time_points):
                if time_point in mean_ref_corr:
                    all_ref_corr_means[time_point].append(mean_ref_corr[time_point])
    
    # Calculate the mean and standard deviation across all files for each time point
    time_points = list(range(max_time_points))
    
    # Reference channel
    avg_ref_means = [np.mean(means) if means else np.nan for means in all_ref_means]
    std_ref_means = [np.std(means) if len(means) > 1 else 0 for means in all_ref_means]
    
    # Measurement channel
    avg_meas_means = [np.mean(means) if means else np.nan for means in all_meas_means]
    std_meas_means = [np.std(means) if len(means) > 1 else 0 for means in all_meas_means]
    
    # Corrected reference channel (if applicable)
    if bleach_corrected:
        avg_ref_corr_means = [np.mean(means) if means else np.nan for means in all_ref_corr_means]
        std_ref_corr_means = [np.std(means) if len(means) > 1 else 0 for means in all_ref_corr_means]
    
    # Create the plots
    
    # 1. Combined plot with both channels
    plt.figure(figsize= (12, 8))
    
    plt.errorbar(
        time_points,
        avg_ref_means,
        yerr=std_ref_means,
        fmt='-o',
        color='red',
        label='Reference Channel',
        capsize=4,
        markersize=8,
        linewidth=2,
        elinewidth=1
    )
    
    if bleach_corrected:
        plt.errorbar(
            time_points,
            avg_ref_corr_means,
            yerr=std_ref_corr_means,
            fmt='-o',
            color='yellow',
            label='Reference Channel (Corrected)',
            capsize=4,
            markersize=8,
            linewidth=2,
            elinewidth=1
        )
    
    plt.errorbar(
        time_points,
        avg_meas_means,
        yerr=std_meas_means,
        fmt='-o',
        color='green',
        label='Measurement Channel',
        capsize=4,
        markersize=8,
        linewidth=2,
        elinewidth=1
    )
    
    # Add plot details
    title = f'Average Channel Intensities - {folder_name}'
    if bleach_corrected:
        title += " (with Bleach Correction)"
    plt.title(title, fontsize=16)
    plt.xlabel('Time Point', fontsize=14)
    plt.ylabel('Mean Intensity', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Enhance the plot appearance
    plt.tight_layout()
    
    # Save the combined channels plot
    label = 'with_correction' if bleach_corrected else 'orig'
    combined_output_file = os.path.join(output_dir, f'{folder_name}_avg_channels_plot_{label}_{timestamp}.png')
    plt.savefig(combined_output_file, dpi=300)
    plt.close()
    
    print(f"Combined channels plot saved as {combined_output_file}")
    
    # Save the averaged data as CSV
    if bleach_corrected:
        avg_data = pd.DataFrame({
            'time_point': time_points,
            'average_reference': avg_ref_means,
            'std_reference': std_ref_means,
            'average_reference_corrected': avg_ref_corr_means,
            'std_reference_corrected': std_ref_corr_means,
            'average_measurement': avg_meas_means,
            'std_measurement': std_meas_means,
            'num_ref_samples': [len(means) for means in all_ref_means],
            'num_meas_samples': [len(means) for means in all_meas_means]
        })
    else:
        avg_data = pd.DataFrame({
            'time_point': time_points,
            'average_reference': avg_ref_means,
            'std_reference': std_ref_means,
            'average_measurement': avg_meas_means,
            'std_measurement': std_meas_means,
            'num_ref_samples': [len(means) for means in all_ref_means],
            'num_meas_samples': [len(means) for means in all_meas_means]
        })
    
    output_csv = os.path.join(output_dir, f'{folder_name}_averaged_channels_data_{label}_{timestamp}.csv')
    avg_data.to_csv(output_csv, index=False)
    print(f"Averaged channels data saved as {output_csv}")

# Usage example:
if __name__ == "__main__":
    # Process a folder of IMS files
    folder_path = input("Enter folder path containing IMS files: ")
    
    # Use default channels unless specified otherwise
    ref_channel = 1
    meas_channel = 0
    
    # Allow user to override defaults
    channel_override = input("Use default channels (ref=1, meas=0)? (y/n): ")
    if channel_override.lower() == 'n':
        ref_channel = int(input("Enter reference channel number: "))
        meas_channel = int(input("Enter measurement channel number: "))
    
    # Ask whether to save individual cell images
    save_images = input("Save individual cell images? (y/n, default=n): ").lower()
    save_cell_images = save_images == 'y'
    
    # Ask about bleaching correction
    bleach_correct = input("Apply bleaching correction to reference channel? (y/n, default=y): ").lower()
    bleach_correction = bleach_correct != 'n'  # Default to yes
    
    bleach_model = 'exponential'
    poly_order = 2
    normalization_point = 0
    
    if bleach_correction:
        # Ask about correction model
        model_choice = input("Choose bleaching model (exponential/linear/polynomial, default=exponential): ").lower()
        if model_choice in ['linear', 'polynomial']:
            bleach_model = model_choice
            
        if bleach_model == 'polynomial':
            try:
                poly_order = int(input("Enter polynomial order (2-5, default=2): "))
                if poly_order < 1 or poly_order > 5:
                    print("Invalid order. Using default (2).")
                    poly_order = 2
            except ValueError:
                print("Invalid input. Using default polynomial order (2).")
        
        try:
            normalization_point = int(input("Enter time point to normalize to (default=0, i.e., first frame): "))
        except ValueError:
            print("Invalid input. Using default normalization point (0).")
    
    # Process the folder
    process_ims_folder(
        folder_path, 
        ref_channel=ref_channel,
        meas_channel=meas_channel,
        save_cell_images=save_cell_images,
        bleach_correction=bleach_correction,
        bleach_model=bleach_model,
        poly_order=poly_order,
        normalization_point=normalization_point
    )