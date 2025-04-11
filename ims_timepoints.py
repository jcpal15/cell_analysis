import numpy as np
import h5py
from skimage import filters, measure, exposure
import matplotlib.pyplot as plt
from scipy import ndimage
import os
import pandas as pd
from tqdm import tqdm  # Import tqdm for progress bars

def analyze_multichannel_depletion_h5(h5_path, ref_channel=1, meas_channel=0, 
                                    percentile_low=0.5, percentile_high=99.5,
                                    crop_size=50, min_cell_size=100,
                                    resolution_level=0, z_slice=0):
    
    # Create output directories
    output_dir = 'cell_analysis_results'
    cell_crops_dir = 'cell_crops'
    for directory in [output_dir, cell_crops_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
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
                    
                    # Threshold selection
                    threshold_choice = input("Enter threshold method (otsu/li) or a manual value (0-1): ")
                    
                    if threshold_choice == 'otsu':
                        print("Calculating Otsu threshold...")
                        thresh = filters.threshold_otsu(reference_enhanced)
                    elif threshold_choice == 'li':
                        print("Calculating Li threshold...")
                        thresh = filters.threshold_li(reference_enhanced)
                    else:
                        try:
                            thresh = float(threshold_choice)
                        except ValueError:
                            print("Invalid input, using Otsu's method")
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
                    
                    # Save the segmentation mask
                    print("Saving segmentation mask...")
                    plt.figure(figsize=(10, 10))
                    plt.imshow(filtered_labels, cmap='nipy_spectral')
                    plt.colorbar(label='Cell ID')
                    plt.title('Cell Segmentation Mask')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, 'cell_segmentation_mask.png'))
                    plt.close()
                    
                    # Crop and save individual cell images (like original script)
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
        
        # Save the complete dataset
        print("Saving data to CSV...")
        df.to_csv(os.path.join(output_dir, 'cell_time_course_data.csv'), index=False)
        
        # Create just the two key summary plots
        print("Creating summary plots...")
        
        # Group data by time point
        time_groups = df.groupby('time_point')
        
        # Calculate mean intensities and ratios per time point
        mean_ref = time_groups['reference_intensity'].mean()
        std_ref = time_groups['reference_intensity'].std()
        mean_meas = time_groups['measurement_intensity'].mean()
        std_meas = time_groups['measurement_intensity'].std()
        mean_ratios = time_groups['ratio'].mean()
        std_ratios = time_groups['ratio'].std()
        
        # 1. Plot showing intensity of each channel over time
        plt.figure(figsize=(10, 6))
        plt.errorbar(mean_ref.index, mean_ref, yerr=std_ref, fmt='b-o', label='Reference Channel')
        plt.errorbar(mean_meas.index, mean_meas, yerr=std_meas, fmt='r-o', label='Measurement Channel')
        plt.title('Mean Channel Intensities Over Time')
        plt.xlabel('Time Point')
        plt.ylabel('Mean Intensity')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'channel_intensities_over_time.png'))
        plt.close()
        
        # 2. Plot showing intensity ratio over time
        plt.figure(figsize=(10, 6))
        plt.errorbar(mean_ratios.index, mean_ratios, yerr=std_ratios, fmt='g-o')
        plt.title('Mean Ratio Over Time (All Cells)')
        plt.xlabel('Time Point')
        plt.ylabel('Mean Ratio (Measurement/Reference)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'mean_ratio_over_time.png'))
        plt.close()
        
        print("Analysis complete!")
        
        # Return the dataframe and the segmentation mask
        return df, master_labels
        
    except Exception as e:
        print(f"Error in analysis: {e}")
        # Print a more detailed traceback
        import traceback
        traceback.print_exc()
        return None, None

# Usage example:
if __name__ == "__main__":
    cell_data, labels = analyze_multichannel_depletion_h5('control.ims', 
                                                       ref_channel=1, 
                                                       meas_channel=0,
                                                       resolution_level=0, 
                                                       z_slice=0)  # Use z_slice=0 as default