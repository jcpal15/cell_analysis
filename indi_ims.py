import numpy as np
import h5py
from skimage import filters, measure, exposure
import matplotlib.pyplot as plt
from scipy import ndimage, optimize, stats
import os
import pandas as pd
from tqdm import tqdm
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
        # If no underscore pattern found, just return the stem as is
        return stem

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
            
        print(f"Analysis of {file_name} complete! Results saved in:\n{output_info}")
        
        # Return the result info
        return {
            'file_name': file_name,
            'output_dir': output_dir,
            'cell_crops_dir': cell_crops_dir,
            'time_points': num_time_points,
            'cells_detected': len(valid_labels),
            'bleach_corrected': bleach_correction
        }
        
    except Exception as e:
        print(f"Error in analysis of {h5_path}: {e}")
        # Print a more detailed traceback
        import traceback
        traceback.print_exc()
        return None

def process_ims_files_individually(folder_path, ref_channel=1, meas_channel=0, 
                               resolution_level=0, z_slice=0, 
                               save_cell_images=True, bleach_correction=True,
                               bleach_model='exponential', poly_order=2,
                               normalization_point=0):
    """
    Process each IMS file in a folder individually
    """
    # Find all IMS files
    ims_files = glob.glob(os.path.join(folder_path, "*.ims"))
    
    if not ims_files:
        print(f"No IMS files found in {folder_path}")
        return
    
    print(f"Found {len(ims_files)} IMS files to process:")
    for i, file in enumerate(ims_files):
        print(f"{i+1}. {os.path.basename(file)}")
    
    # Setup common parameters for all files
    common_params = {
        'ref_channel': ref_channel,
        'meas_channel': meas_channel, 
        'resolution_level': resolution_level,
        'z_slice': z_slice,
        'save_cell_images': save_cell_images,
        'bleach_correction': bleach_correction,
        'bleach_model': bleach_model,
        'poly_order': poly_order,
        'normalization_point': normalization_point
    }
    
    # Create a summary of all analyses
    all_results = []
    
    # Process each file independently
    for i, file_path in enumerate(ims_files):
        print(f"\n{'='*50}")
        print(f"Processing file {i+1}/{len(ims_files)}: {os.path.basename(file_path)}")
        print(f"{'='*50}")
        
        # Ask for threshold method for this specific file
        threshold_choice = input(f"Enter threshold method for {os.path.basename(file_path)} (otsu/li) or a manual value (0-1): ")
        
        # Process this file
        result = analyze_multichannel_depletion_h5(
            file_path, 
            threshold_method=threshold_choice,
            **common_params
        )
        
        if result is not None:
            all_results.append(result)
            print(f"\nCompleted analysis of file {i+1}/{len(ims_files)}")
        else:
            print(f"\nFailed to analyze file {i+1}/{len(ims_files)}")
    
    # Create a summary report
    if all_results:
        # Create timestamp for the summary report
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create summary table
        summary_data = []
        for result in all_results:
            summary_data.append({
                'File Name': result['file_name'],
                'Output Directory': result['output_dir'],
                'Time Points': result['time_points'],
                'Cells Detected': result['cells_detected'],
                'Bleach Correction': 'Yes' if result['bleach_corrected'] else 'No'
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save the summary
        summary_dir = 'analysis_summaries'
        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir)
            
        summary_file = os.path.join(summary_dir, f"analysis_summary_{current_time}.csv")
        summary_df.to_csv(summary_file, index=False)
        
        print(f"\nAnalysis summary saved to {summary_file}")
        print("\nSummary of analyses:")
        print(summary_df.to_string())
    else:
        print("\nNo successful analyses to summarize.")
    
    print("\nAll processing complete!")

# Usage example:
if __name__ == "__main__":
    # Process all IMS files individually
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
    
    # Process all files in the folder individually
    process_ims_files_individually(
        folder_path, 
        ref_channel=ref_channel,
        meas_channel=meas_channel,
        save_cell_images=save_cell_images,
        bleach_correction=bleach_correction,
        bleach_model=bleach_model,
        poly_order=poly_order,
        normalization_point=normalization_point
    )