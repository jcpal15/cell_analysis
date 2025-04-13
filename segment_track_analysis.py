import numpy as np
import tifffile
from skimage import filters, measure, exposure
from scipy import ndimage, optimize, stats
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm
import datetime
from pathlib import Path
from scipy.spatial.distance import cdist

def segment_track_analyze_tif(tif_path, output_dir=None, ref_channel=1, meas_channel=0, 
                           min_cell_size=100, max_tracking_distance=20,
                           use_adaptive=True, adaptive_block_size=35, 
                           use_watershed=True, watershed_min_distance=10,
                           percentile_low=0.1, percentile_high=99.9,
                           bleach_correction=True, bleach_model='exponential'):
    """
    Segment, track, and analyze cells in a time-lapse TIF file.
    
    Parameters:
    -----------
    tif_path : str
        Path to the time-lapse TIF file
    output_dir : str or None
        Directory to save results (if None, creates timestamped directory)
    ref_channel : int
        Index of the reference channel (default: 1)
    meas_channel : int
        Index of the measurement channel (default: 0)
    min_cell_size : int
        Minimum size of cells in pixels (default: 100)
    max_tracking_distance : float
        Maximum distance for cell tracking between frames (default: 20)
    use_adaptive : bool
        Whether to use adaptive thresholding (default: True)
    adaptive_block_size : int
        Block size for adaptive thresholding (default: 35)
    use_watershed : bool
        Whether to use watershed segmentation (default: True)
    watershed_min_distance : int
        Minimum distance between peaks for watershed (default: 10)
    percentile_low, percentile_high : float
        Percentiles for contrast enhancement (default: 0.1, 99.9)
    bleach_correction : bool
        Whether to apply bleaching correction (default: True)
    bleach_model : str
        Type of bleaching model ('exponential', 'linear', 'polynomial')
    """
    # Create timestamp and output directory
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = Path(tif_path).stem
    
    if output_dir is None:
        output_dir = f'analysis_results/{file_name}_{current_time}'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Step 1: Load the TIF file
    print(f"Loading TIF file: {tif_path}")
    tif_data = tifffile.imread(tif_path)
    
    # Determine data dimensions
    print(f"TIF data shape: {tif_data.shape}")
    
    # Interpret the shape based on dimensions
    if len(tif_data.shape) == 5:  # TZCYX format
        num_frames = tif_data.shape[0]
        num_channels = tif_data.shape[2] if tif_data.shape[2] > 1 else 1
        height, width = tif_data.shape[3], tif_data.shape[4]
        # Use first Z-slice
        tif_data = tif_data[:, 0, :, :, :]
    elif len(tif_data.shape) == 4:
        if tif_data.shape[1] <= 5:  # Likely TCYX (T, channels, Y, X)
            num_frames = tif_data.shape[0]
            num_channels = tif_data.shape[1]
            height, width = tif_data.shape[2], tif_data.shape[3]
        else:  # May be TZYX or similar
            num_frames = tif_data.shape[0]
            num_channels = 1
            height, width = tif_data.shape[2], tif_data.shape[3]
            # Reshape to add channel dimension
            tif_data = tif_data[:, np.newaxis, :, :]
    elif len(tif_data.shape) == 3:
        # Could be TYX or YXC
        if tif_data.shape[2] <= 5:  # Likely YXC
            num_frames = 1
            num_channels = tif_data.shape[2]
            height, width = tif_data.shape[0], tif_data.shape[1]
            # Reshape to TCYX
            tif_data = tif_data[np.newaxis, :, :, :].transpose(0, 3, 1, 2)
        else:  # Likely TYX
            num_frames = tif_data.shape[0]
            num_channels = 1
            height, width = tif_data.shape[1], tif_data.shape[2]
            # Reshape to add channel dimension
            tif_data = tif_data[:, np.newaxis, :, :]
    else:
        raise ValueError(f"Unexpected data shape: {tif_data.shape}")
    
    print(f"Interpreted data: {num_frames} frames, {num_channels} channels, {height}x{width} pixels")
    
    # Validate channel indices
    if ref_channel >= num_channels or meas_channel >= num_channels:
        raise ValueError(f"Channel indices out of range (max: {num_channels-1})")
    
    # Initialize storage for cell tracking
    cell_tracks = {}   # Will store tracking information for each cell
    all_frames_data = []  # Will store measurements for each cell in each frame
    
    # Step 2: Segment the first frame
    print("\nSegmenting first frame...")
    
    # Get first frame reference channel
    first_ref_img = tif_data[0, ref_channel, :, :]
    
    # Normalize image
    print("Normalizing image...")
    first_ref_img = (first_ref_img - np.min(first_ref_img)) / (np.max(first_ref_img) - np.min(first_ref_img))
    
    # Apply Gaussian smoothing
    smoothed = filters.gaussian(first_ref_img, sigma=1.0)
    
    # Enhance contrast
    p_low, p_high = np.percentile(smoothed, (percentile_low, percentile_high))
    reference_enhanced = exposure.rescale_intensity(smoothed, in_range=(p_low, p_high))
    
    # Apply thresholding
    if use_adaptive:
        # Adaptive thresholding for uneven illumination
        local_thresh = filters.threshold_local(reference_enhanced, block_size=adaptive_block_size)
        binary = reference_enhanced > local_thresh
    else:
        # Global Otsu thresholding
        thresh = filters.threshold_otsu(reference_enhanced)
        binary = reference_enhanced > thresh
    
    # Clean up binary mask
    binary = ndimage.binary_fill_holes(binary)
    binary = ndimage.binary_closing(binary, iterations=3)
    binary = ndimage.binary_opening(binary, iterations=2)
    binary = ndimage.binary_fill_holes(binary)
    binary = ndimage.binary_opening(binary, structure=np.ones((3,3)))
    
    # Apply watershed segmentation if requested
    if use_watershed:
        from scipy import ndimage as ndi
        from skimage.feature import peak_local_max
        from skimage import segmentation
        
        # Compute the distance map
        distance = ndi.distance_transform_edt(binary)
        
        # Find local maxima (cell centers)
        try:
            # Try newer scikit-image version with indices=False
            local_max_coords = peak_local_max(distance, min_distance=watershed_min_distance,
                                          labels=binary)
            # Convert coordinates to mask
            local_max = np.zeros_like(binary, dtype=bool)
            for coord in local_max_coords:
                local_max[tuple(coord)] = True
        except TypeError:
            # Older versions
            local_max_coords = peak_local_max(distance, min_distance=watershed_min_distance, 
                                          labels=binary)
            # Convert coordinates to mask
            local_max = np.zeros_like(binary, dtype=bool)
            for coord in local_max_coords:
                local_max[coord[0], coord[1]] = True
        
        # Mark each local maximum with a unique label
        markers = measure.label(local_max)
        
        # Apply watershed to find cell boundaries
        labels = segmentation.watershed(-distance, markers, mask=binary)
    else:
        # Standard connected component labeling
        labels = measure.label(binary)
    
    # Get region properties
    props = measure.regionprops(labels, first_ref_img)
    
    # Filter regions based on size
    valid_labels = []
    for prop in props:
        if prop.area >= min_cell_size and prop.area <= min_cell_size * 10:
            valid_labels.append(prop.label)
    
    # Create filtered labels for first frame
    first_frame_labels = np.zeros_like(labels)
    for label in valid_labels:
        first_frame_labels[labels == label] = label
    
    print(f"Found {len(valid_labels)} valid cells in first frame")
    
    # Save the segmentation overlay
    plt.figure(figsize=(12, 10))
    plt.imshow(first_ref_img, cmap='gray')
    
    # Create a semi-transparent overlay
    masked_labels = np.ma.masked_where(first_frame_labels == 0, first_frame_labels)
    cmap = plt.cm.get_cmap('nipy_spectral', len(valid_labels) + 1)
    plt.imshow(masked_labels, cmap=cmap, alpha=0.5)
    
    # Add cell ID labels
    for label in valid_labels:
        props = measure.regionprops(np.asarray(first_frame_labels == label, dtype=int))
        if props:
            y, x = props[0].centroid
            plt.text(x, y, str(label), color='white', 
                   fontsize=8, ha='center', va='center',
                   bbox=dict(facecolor='black', alpha=0.5, pad=1))
    
    plt.title('First Frame Segmentation')
    plt.savefig(os.path.join(output_dir, f"{file_name}_first_frame_segmentation.png"))
    plt.close()
    
    # Step 3: Initialize tracking
    # Create master labels array to store cell IDs across all frames
    master_labels = np.zeros((num_frames, height, width), dtype=np.int32)
    master_labels[0] = first_frame_labels
    
    # Initialize cell tracking data for first frame
    for label in valid_labels:
        prop = measure.regionprops(np.asarray(first_frame_labels == label, dtype=int))[0]
        cell_tracks[label] = {
            'frames': [0],
            'centroids': [prop.centroid],
            'areas': [prop.area],
            'in_view': True  # Assume initially in view
        }
    
    # Step 4: Process all frames and track cells
    print("\nProcessing all frames and tracking cells...")
    
    for frame in tqdm(range(num_frames), desc="Processing frames"):
        # Get reference and measurement channels for this frame
        ref_img = tif_data[frame, ref_channel]
        meas_img = tif_data[frame, meas_channel]
        
        # For first frame, we already have the labels
        if frame == 0:
            frame_labels = first_frame_labels
        else:
            # For subsequent frames, segment again and match with previous
            # Normalize image
            ref_img_norm = (ref_img - np.min(ref_img)) / (np.max(ref_img) - np.min(ref_img))
            
            # Apply same preprocessing as first frame
            smoothed = filters.gaussian(ref_img_norm, sigma=1.0)
            p_low, p_high = np.percentile(smoothed, (percentile_low, percentile_high))
            reference_enhanced = exposure.rescale_intensity(smoothed, in_range=(p_low, p_high))
            
            # Apply thresholding
            if use_adaptive:
                local_thresh = filters.threshold_local(reference_enhanced, block_size=adaptive_block_size)
                binary = reference_enhanced > local_thresh
            else:
                binary = reference_enhanced > filters.threshold_otsu(reference_enhanced)
            
            # Clean up binary mask
            binary = ndimage.binary_fill_holes(binary)
            binary = ndimage.binary_closing(binary, iterations=3)
            binary = ndimage.binary_opening(binary, iterations=2)
            binary = ndimage.binary_fill_holes(binary)
            binary = ndimage.binary_opening(binary, structure=np.ones((3,3)))
            
            # Apply watershed or standard labeling
            if use_watershed:
                distance = ndi.distance_transform_edt(binary)
                
                try:
                    local_max_coords = peak_local_max(distance, min_distance=watershed_min_distance,
                                                 labels=binary)
                    local_max = np.zeros_like(binary, dtype=bool)
                    for coord in local_max_coords:
                        local_max[tuple(coord)] = True
                except TypeError:
                    local_max_coords = peak_local_max(distance, min_distance=watershed_min_distance, 
                                                 labels=binary)
                    local_max = np.zeros_like(binary, dtype=bool)
                    for coord in local_max_coords:
                        local_max[coord[0], coord[1]] = True
                
                markers = measure.label(local_max)
                labels = segmentation.watershed(-distance, markers, mask=binary)
            else:
                labels = measure.label(binary)
            
            # Filter regions based on size
            props = measure.regionprops(labels)
            valid_labels_current = []
            
            for prop in props:
                if prop.area >= min_cell_size and prop.area <= min_cell_size * 10:
                    valid_labels_current.append(prop.label)
            
            # Create filtered labels for this frame
            frame_labels = np.zeros_like(labels)
            for label in valid_labels_current:
                frame_labels[labels == label] = label
            
            # Match cells with previous frame
            # Get centroids of current frame cells
            if len(valid_labels_current) > 0:
                current_props = measure.regionprops(frame_labels)
                current_centroids = np.array([prop.centroid for prop in current_props if prop.label > 0])
                current_labels = np.array([prop.label for prop in current_props if prop.label > 0])
                current_areas = np.array([prop.area for prop in current_props if prop.label > 0])
                
                # Get centroids from previous frame tracked cells
                prev_frame = frame - 1
                prev_centroids = []
                prev_cell_ids = []
                prev_areas = []
                
                for cell_id, track in cell_tracks.items():
                    if prev_frame in track['frames']:
                        idx = track['frames'].index(prev_frame)
                        prev_centroids.append(track['centroids'][idx])
                        prev_cell_ids.append(cell_id)
                        prev_areas.append(track['areas'][idx])
                
                # Calculate distances between cell centroids
                if len(prev_centroids) > 0 and len(current_centroids) > 0:
                    prev_centroids = np.array(prev_centroids)
                    distances = cdist(prev_centroids, current_centroids)
                    
                    # Match cells based on minimum distance
                    for prev_idx in range(len(prev_cell_ids)):
                        prev_id = prev_cell_ids[prev_idx]
                        
                        # Find closest cell within max distance
                        min_dist_idx = np.argmin(distances[prev_idx])
                        min_dist = distances[prev_idx, min_dist_idx]
                        
                        if min_dist <= max_tracking_distance:
                            # Match found, update tracking
                            curr_label = current_labels[min_dist_idx]
                            
                            # Update master labels for this frame
                            master_labels[frame][frame_labels == curr_label] = prev_id
                            
                            # Update tracking data
                            cell_tracks[prev_id]['frames'].append(frame)
                            cell_tracks[prev_id]['centroids'].append(current_centroids[min_dist_idx])
                            cell_tracks[prev_id]['areas'].append(current_areas[min_dist_idx])
                            
                            # Mark this cell as matched so we don't match it again
                            distances[:, min_dist_idx] = float('inf')
                    
                    # For unmatched cells in current frame, assign new IDs
                    for i, curr_label in enumerate(current_labels):
                        if not np.any(master_labels[frame] == curr_label):
                            # Check if label is already used
                            if not np.any(master_labels[frame] > 0):
                                new_id = curr_label + frame * 10000  # Ensure unique ID
                            else:
                                new_id = np.max(master_labels[frame]) + 1
                            
                            # Update master labels
                            master_labels[frame][frame_labels == curr_label] = new_id
                            
                            # Create new tracking data
                            cell_tracks[new_id] = {
                                'frames': [frame],
                                'centroids': [current_centroids[i]],
                                'areas': [current_areas[i]],
                                'in_view': True
                            }
        
        # Measure cell properties for this frame
        for cell_id in list(cell_tracks.keys()):
            # Check if this cell exists in the current frame
            cell_mask = master_labels[frame] == cell_id
            
            if np.any(cell_mask):
                # Cell found in this frame, measure properties
                ref_intensity = np.mean(ref_img[cell_mask])
                meas_intensity = np.mean(meas_img[cell_mask])
                
                # Calculate ratio
                ratio = meas_intensity / ref_intensity if ref_intensity > 0 else 0
                
                # Store measurements
                all_frames_data.append({
                    'time_point': frame,
                    'cell_id': cell_id,
                    'reference_intensity': ref_intensity,
                    'measurement_intensity': meas_intensity,
                    'ratio': ratio,
                    'area': np.sum(cell_mask)
                })
            else:
                # Check if cell was in previous frame but not this one
                if cell_id in cell_tracks and frame - 1 in cell_tracks[cell_id]['frames']:
                    # Cell has left the field of view
                    cell_tracks[cell_id]['in_view'] = False
    
    # Convert measurements to DataFrame
    df = pd.DataFrame(all_frames_data)
    
    # Step 5: Apply bleaching correction if requested
    if bleach_correction:
        print("\nApplying photobleaching correction...")
        df_corrected, model_params, correction_factors = apply_bleaching_correction(
            df, bleach_model)
        
        # Replace original DataFrame
        df = df_corrected
    
    # Step 6: Filter cells that leave field of view
    in_view_cells = [cell_id for cell_id, track in cell_tracks.items() if track['in_view']]
    
    # Filter to only keep cells visible through all frames
    full_track_cells = [cell_id for cell_id, track in cell_tracks.items() 
                     if len(track['frames']) == num_frames]
    
    print(f"\nTracking statistics:")
    print(f"  Total cells detected: {len(cell_tracks)}")
    print(f"  Cells that stayed in view: {len(in_view_cells)}")
    print(f"  Cells tracked through all frames: {len(full_track_cells)}")
    
    # Filter DataFrame to only include full-track cells
    df_filtered = df[df['cell_id'].isin(full_track_cells)]
    
    # Save the full data and filtered data
    df.to_csv(os.path.join(output_dir, f"{file_name}_all_cells_data.csv"), index=False)
    df_filtered.to_csv(os.path.join(output_dir, f"{file_name}_tracked_cells_data.csv"), index=False)
    
    # Step 7: Generate analysis plots
    print("\nGenerating analysis plots...")
    
    # 1. Uncorrected ratio plot
    create_ratio_plot(df_filtered, corrected=False, output_dir=output_dir, file_name=file_name)
    
    # 2. Corrected ratio plot (if bleach correction applied)
    if bleach_correction:
        create_ratio_plot(df_filtered, corrected=True, output_dir=output_dir, file_name=file_name)
    
    # 3. Channel intensity plots
    create_channel_plots(df_filtered, bleach_correction, output_dir, file_name)
    
    # 4. Individual cell plots
    create_individual_cell_plots(df_filtered, bleach_correction, output_dir, file_name)
    
    # 5. Create cell tracking visualization
    create_tracking_visualization(master_labels, cell_tracks, full_track_cells, output_dir, file_name)
    
    print(f"\nAnalysis complete. Results saved to: {output_dir}")
    return df_filtered, cell_tracks, master_labels


def exponential_decay(x, a, b, c):
    """Exponential decay function for fitting"""
    return a * np.exp(-b * x) + c


def linear_model(x, a, b):
    """Linear model function for fitting"""
    return a * x + b


def polynomial_model(x, *params):
    """Polynomial model function for fitting"""
    result = 0
    for i, a in enumerate(params):
        result += a * (x ** i)
    return result


def apply_bleaching_correction(df, model_type='exponential', poly_order=2, normalization_point=0):
    """
    Apply bleaching correction to the reference channel
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with cell measurements
    model_type : str
        Type of model to fit ('exponential', 'linear', 'polynomial')
    poly_order : int
        Order of polynomial (only used for 'polynomial' model)
    normalization_point : int
        Time point to normalize to
        
    Returns:
    --------
    df_corrected : pandas.DataFrame
        DataFrame with corrected values
    model_params : tuple
        Model parameters
    correction_factors : array
        Correction factors for each time point
    """
    # Create a copy of the DataFrame
    df_corrected = df.copy()
    
    # Group by time point
    time_groups = df.groupby('time_point')
    
    # Calculate mean intensities per time point
    mean_ref = time_groups['reference_intensity'].mean()
    
    # Get time points and intensities
    time_points = np.array(mean_ref.index)
    intensities = np.array(mean_ref.values)
    
    # Fit the model
    if model_type == 'exponential':
        # Initial guess for parameters (a, b, c)
        max_val = np.max(intensities)
        min_val = np.min(intensities)
        p0 = (max_val - min_val, 0.1, min_val)
        
        try:
            # Fit the model
            popt, _ = optimize.curve_fit(exponential_decay, time_points, intensities, p0=p0, maxfev=10000)
            model_params = popt
            
            # Calculate predicted values
            predicted = exponential_decay(time_points, *model_params)
            normalization_value = exponential_decay(normalization_point, *model_params)
        except:
            print("Exponential fit failed, using linear model instead")
            model_type = 'linear'
    
    if model_type == 'linear':
        # Linear regression
        slope, intercept, _, _, _ = stats.linregress(time_points, intensities)
        model_params = (slope, intercept)
        
        # Calculate predicted values
        predicted = linear_model(time_points, *model_params)
        normalization_value = linear_model(normalization_point, *model_params)
    
    if model_type == 'polynomial':
        # Fit polynomial
        popt = np.polyfit(time_points, intensities, poly_order)
        model_params = tuple(reversed(popt))
        
        # Calculate predicted values using numpy polynomial
        predicted = np.polyval(popt, time_points)
        normalization_value = np.polyval(popt, normalization_point)
    
    # Calculate correction factors
    correction_factors = np.ones_like(time_points, dtype=float)
    valid_indices = (predicted > 0)
    correction_factors[valid_indices] = normalization_value / predicted[valid_indices]
    
    # Apply correction to reference channel
    for time_point, factor in zip(time_points, correction_factors):
        mask = df_corrected['time_point'] == time_point
        df_corrected.loc[mask, 'reference_intensity_corrected'] = \
            df_corrected.loc[mask, 'reference_intensity'] * factor
    
    # Calculate corrected ratio
    df_corrected['ratio_corrected'] = df_corrected['measurement_intensity'] / df_corrected['reference_intensity_corrected']
    
    # Handle invalid values
    df_corrected['ratio_corrected'].replace([np.inf, -np.inf], np.nan, inplace=True)
    
    return df_corrected, model_params, correction_factors


def create_ratio_plot(df, corrected=False, output_dir='.', file_name='output'):
    """Create plot of measurement/reference ratio over time"""
    plt.figure(figsize=(12, 8))
    
    # Group by time point
    time_groups = df.groupby('time_point')
    
    # Calculate stats
    if corrected and 'ratio_corrected' in df.columns:
        ratio_column = 'ratio_corrected'
        title_suffix = 'Corrected'
    else:
        ratio_column = 'ratio'
        title_suffix = 'Uncorrected'
    
    mean_ratios = time_groups[ratio_column].mean()
    std_ratios = time_groups[ratio_column].std()
    
    # Plot with error bars
    plt.errorbar(mean_ratios.index, mean_ratios, yerr=std_ratios, 
               fmt='o-', linewidth=2, capsize=4)
    
    plt.title(f'Mean {title_suffix} Ratio Over Time (Measurement/Reference)', fontsize=14)
    plt.xlabel('Time Point', fontsize=12)
    plt.ylabel(f'{title_suffix} Ratio', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, f"{file_name}_mean_{title_suffix.lower()}_ratio.png"))
    plt.close()


def create_channel_plots(df, bleach_correction, output_dir='.', file_name='output'):
    """Create plots of channel intensities over time"""
    plt.figure(figsize=(12, 8))
    
    # Group by time point
    time_groups = df.groupby('time_point')
    
    # Calculate stats for each channel
    mean_ref = time_groups['reference_intensity'].mean()
    std_ref = time_groups['reference_intensity'].std()
    
    mean_meas = time_groups['measurement_intensity'].mean()
    std_meas = time_groups['measurement_intensity'].std()
    
    # Plot reference channel
    plt.errorbar(mean_ref.index, mean_ref, yerr=std_ref, 
               fmt='o-', linewidth=2, capsize=4, 
               label='Reference Channel', color='blue')
    
    # If bleach correction applied, also plot corrected reference
    if bleach_correction and 'reference_intensity_corrected' in df.columns:
        mean_ref_corr = time_groups['reference_intensity_corrected'].mean()
        std_ref_corr = time_groups['reference_intensity_corrected'].std()
        
        plt.errorbar(mean_ref_corr.index, mean_ref_corr, yerr=std_ref_corr, 
                   fmt='o-', linewidth=2, capsize=4, 
                   label='Reference Channel (Corrected)', color='cyan')
    
    # Plot measurement channel
    plt.errorbar(mean_meas.index, mean_meas, yerr=std_meas, 
               fmt='o-', linewidth=2, capsize=4, 
               label='Measurement Channel', color='red')
    
    plt.title('Channel Intensities Over Time', fontsize=14)
    plt.xlabel('Time Point', fontsize=12)
    plt.ylabel('Mean Intensity', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, f"{file_name}_channel_intensities.png"))
    plt.close()


def create_individual_cell_plots(df, bleach_correction, output_dir='.', file_name='output'):
    """Create plots for individual cell traces"""
    # Create directory for individual cell plots
    cell_plots_dir = os.path.join(output_dir, 'cell_plots')
    if not os.path.exists(cell_plots_dir):
        os.makedirs(cell_plots_dir)
    
    # Get unique cell IDs
    cell_ids = df['cell_id'].unique()
    
    # Plot all cells together for ratio
    plt.figure(figsize=(12, 8))
    
    ratio_column = 'ratio_corrected' if bleach_correction and 'ratio_corrected' in df.columns else 'ratio'
    
    # Use a different color for each cell
    cmap = plt.cm.get_cmap('tab20', len(cell_ids))
    
    for i, cell_id in enumerate(cell_ids):
        cell_data = df[df['cell_id'] == cell_id]
        plt.plot(cell_data['time_point'], cell_data[ratio_column], 
               'o-', color=cmap(i), alpha=0.7, linewidth=1, 
               label=f'Cell {cell_id}')
    
    plt.title('All Cells Ratio Traces', fontsize=14)
    plt.xlabel('Time Point', fontsize=12)
    plt.ylabel('Ratio (Measurement/Reference)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # If there are many cells, don't show individual legend entries
    if len(cell_ids) <= 20:
        plt.legend(fontsize=8)
    else:
        plt.legend(['Individual cells'], fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{file_name}_all_cells_ratio.png"))
    plt.close()
    
    # Create individual cell plots (batches of 20 cells per plot)
    max_cells_per_plot = 20
    num_plots = (len(cell_ids) + max_cells_per_plot - 1) // max_cells_per_plot
    
    for plot_idx in range(num_plots):
        plt.figure(figsize=(15, 10))
        
        start_idx = plot_idx * max_cells_per_plot
        end_idx = min((plot_idx + 1) * max_cells_per_plot, len(cell_ids))
        batch_cell_ids = cell_ids[start_idx:end_idx]
        
        for i, cell_id in enumerate(batch_cell_ids):
            cell_data = df[df['cell_id'] == cell_id]
            
            plt.subplot(4, 5, i % 20 + 1)
            
            plt.plot(cell_data['time_point'], cell_data['reference_intensity'], 
                   'b-', label='Reference')
            
            if bleach_correction and 'reference_intensity_corrected' in df.columns:
                plt.plot(cell_data['time_point'], cell_data['reference_intensity_corrected'], 
                       'c-', label='Ref (Corr)')
            
            plt.plot(cell_data['time_point'], cell_data['measurement_intensity'], 
                   'r-', label='Measurement')
            
            plt.title(f'Cell {cell_id}', fontsize=10)
            plt.grid(True, alpha=0.3)
            
            if i % 5 == 0:  # Add y-label to leftmost plots
                plt.ylabel('Intensity', fontsize=8)
            
            if i >= 15:  # Add x-label to bottom plots
                plt.xlabel('Time Point', fontsize=8)
            
            if i == 0:  # Add legend to first plot only
                plt.legend(fontsize=6)
        
        plt.tight_layout()
        plt.savefig(os.path.join(cell_plots_dir, f"{file_name}_cell_intensities_batch{plot_idx+1}.png"))
        plt.close()
        
    # Create individual cell ratio plots
    for plot_idx in range(num_plots):
        plt.figure(figsize=(15, 10))
        
        start_idx = plot_idx * max_cells_per_plot
        end_idx = min((plot_idx + 1) * max_cells_per_plot, len(cell_ids))
        batch_cell_ids = cell_ids[start_idx:end_idx]
        
        for i, cell_id in enumerate(batch_cell_ids):
            cell_data = df[df['cell_id'] == cell_id]
            
            plt.subplot(4, 5, i % 20 + 1)
            
            plt.plot(cell_data['time_point'], cell_data['ratio'], 
                   'g-', label='Ratio')
            
            if bleach_correction and 'ratio_corrected' in df.columns:
                plt.plot(cell_data['time_point'], cell_data['ratio_corrected'], 
                       'm-', label='Ratio (Corr)')
            
            plt.title(f'Cell {cell_id}', fontsize=10)
            plt.grid(True, alpha=0.3)
            
            if i % 5 == 0:  # Add y-label to leftmost plots
                plt.ylabel('Ratio', fontsize=8)
            
            if i >= 15:  # Add x-label to bottom plots
                plt.xlabel('Time Point', fontsize=8)
            
            if i == 0:  # Add legend to first plot only
                plt.legend(fontsize=6)
        
        plt.tight_layout()
        plt.savefig(os.path.join(cell_plots_dir, f"{file_name}_cell_ratios_batch{plot_idx+1}.png"))
        plt.close()


def create_tracking_visualization(master_labels, cell_tracks, full_track_cells, output_dir='.', file_name='output'):
    """Create visualization of cell tracking"""
    # Get dimensions
    num_frames = master_labels.shape[0]
    
    if num_frames <= 1:
        print("Not enough frames for tracking visualization")
        return
    
    # Select frames to visualize (first, last, and equally spaced in between)
    if num_frames <= 5:
        frames_to_show = list(range(num_frames))
    else:
        frames_to_show = [0, 
                        num_frames // 4, 
                        num_frames // 2, 
                        3 * num_frames // 4, 
                        num_frames - 1]
    
    # Create a colormap for cell tracking
    num_colors = len(full_track_cells)
    cmap = plt.cm.get_cmap('tab20', num_colors)
    
    # Create a mapping of cell IDs to colors
    color_map = {}
    for i, cell_id in enumerate(full_track_cells):
        color_map[cell_id] = i % num_colors
    
    # Create the visualization
    plt.figure(figsize=(20, 5))
    
    for i, frame_idx in enumerate(frames_to_show):
        plt.subplot(1, len(frames_to_show), i+1)
        
        # Create an RGB image
        height, width = master_labels[frame_idx].shape
        vis_img = np.zeros((height, width, 3))
        
        # Add each tracked cell with its color
        for cell_id in full_track_cells:
            if frame_idx in cell_tracks[cell_id]['frames']:
                idx = cell_tracks[cell_id]['frames'].index(frame_idx)
                
                # Get cell mask
                mask = master_labels[frame_idx] == cell_id
                
                if np.any(mask):
                    # Apply cell color
                    color_idx = color_map[cell_id]
                    color = np.array(cmap(color_idx)[:3])
                    
                    for c in range(3):
                        vis_img[:, :, c][mask] = color[c]
        
        plt.imshow(vis_img)
        plt.title(f'Frame {frame_idx}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{file_name}_tracking_visualization.png"))
    plt.close()


if __name__ == "__main__":
    print("Cell Segmentation, Tracking and Analysis")
    print("=======================================")
    
    # Get input from user
    tif_path = input("Enter path to TIF file: ")
    
    # Use preset parameters
    ref_channel = 1
    meas_channel = 0
    min_cell_size = 100
    use_adaptive = True
    adaptive_block_size = 35
    use_watershed = True
    watershed_min_distance = 10
    percentile_low = 0.1
    percentile_high = 99.9
    bleach_correction = True
    
    print("\nUsing automatic settings:")
    print(f"  Reference channel: {ref_channel}")
    print(f"  Measurement channel: {meas_channel}")
    print(f"  Minimum cell size: {min_cell_size} pixels")
    print(f"  Adaptive thresholding: Enabled (block size {adaptive_block_size})")
    print(f"  Watershed segmentation: Enabled (min distance {watershed_min_distance})")
    print(f"  Contrast enhancement: {percentile_low} to {percentile_high} percentiles")
    print(f"  Bleach correction: {'Enabled' if bleach_correction else 'Disabled'}")
    print()
    
    # Run the analysis
    segment_track_analyze_tif(tif_path, ref_channel=ref_channel, meas_channel=meas_channel,
                           min_cell_size=min_cell_size, 
                           use_adaptive=use_adaptive, adaptive_block_size=adaptive_block_size,
                           use_watershed=use_watershed, watershed_min_distance=watershed_min_distance,
                           percentile_low=percentile_low, percentile_high=percentile_high,
                           bleach_correction=bleach_correction)