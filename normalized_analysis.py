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
from scipy.spatial import cKDTree

def segment_track_analyze_tif(tif_path, output_dir=None, ref_channel=1, meas_channel=0, 
                           min_cell_size=100, max_tracking_distance=20,
                           use_adaptive=True, adaptive_block_size=35, 
                           use_watershed=True, watershed_min_distance=10,
                           percentile_low=0.1, percentile_high=99.9,
                           bleach_correction=True, bleach_model='exponential',
                           baseline_frames=None):
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
    baseline_frames : int or None
        Number of initial frames to use as baseline for ratio normalization.
        If provided, ratio values will be normalized to the average of these frames.
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
            
            # Clean up binary mask - optimized by reducing iterations
            binary = ndimage.binary_fill_holes(binary)
            binary = ndimage.binary_closing(binary, iterations=2)
            binary = ndimage.binary_opening(binary, iterations=1)
            binary = ndimage.binary_fill_holes(binary)
            
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
            
            # Match cells with previous frame using KD-Tree for nearest-neighbor search
            if len(valid_labels_current) > 0:
                # Get properties of current frame cells
                current_props = measure.regionprops(frame_labels)
                current_centroids = np.array([prop.centroid for prop in current_props if prop.label > 0])
                current_labels = np.array([prop.label for prop in current_props if prop.label > 0])
                current_areas = np.array([prop.area for prop in current_props if prop.label > 0])
                
                # Get properties from previous frame tracked cells
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
                
                # Use a simple nearest-neighbor approach
                if len(prev_centroids) > 0 and len(current_centroids) > 0:
                    prev_centroids_arr = np.array(prev_centroids)
                    
                    # Use KD-Tree for efficient distance calculation
                    tree = cKDTree(current_centroids)
                    
                    # Store which current cells have been matched
                    matched_current_cells = set()
                    
                    # For each previous cell, find the closest current cell within max_tracking_distance
                    for i, prev_id in enumerate(prev_cell_ids):
                        # Query for the closest point within max_tracking_distance
                        distances, indices = tree.query(prev_centroids_arr[i], k=1, 
                                                      distance_upper_bound=max_tracking_distance)
                        
                        # Check if a valid match was found (distance < max_tracking_distance)
                        # If no match is found, distance will be inf and indices will be out of bounds
                        if isinstance(distances, np.ndarray):
                            if len(distances) > 0 and distances[0] < max_tracking_distance and indices[0] < len(current_centroids):
                                idx = indices[0]
                                
                                # Skip if this current cell is already matched to a closer previous cell
                                if idx in matched_current_cells:
                                    continue
                                
                                # Get the current cell label
                                curr_label = current_labels[idx]
                                
                                # Update master labels
                                master_labels[frame][frame_labels == curr_label] = prev_id
                                
                                # Update tracking data
                                cell_tracks[prev_id]['frames'].append(frame)
                                cell_tracks[prev_id]['centroids'].append(current_centroids[idx])
                                cell_tracks[prev_id]['areas'].append(current_areas[idx])
                                
                                # Mark this current cell as matched
                                matched_current_cells.add(idx)
                        else:  # Single value case
                            if distances < max_tracking_distance and indices < len(current_centroids):
                                idx = indices
                                
                                # Skip if this current cell is already matched to a closer previous cell
                                if idx in matched_current_cells:
                                    continue
                                
                                # Get the current cell label
                                curr_label = current_labels[idx]
                                
                                # Update master labels
                                master_labels[frame][frame_labels == curr_label] = prev_id
                                
                                # Update tracking data
                                cell_tracks[prev_id]['frames'].append(frame)
                                cell_tracks[prev_id]['centroids'].append(current_centroids[idx])
                                cell_tracks[prev_id]['areas'].append(current_areas[idx])
                                
                                # Mark this current cell as matched
                                matched_current_cells.add(idx)
                    
                    # Create new IDs for unmatched current cells
                    for i, curr_label in enumerate(current_labels):
                        if i not in matched_current_cells:
                            # Create a new unique ID
                            new_id = curr_label + frame * 10000
                            
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
        frame_cells = np.unique(master_labels[frame])
        frame_cells = frame_cells[frame_cells > 0]  # Remove background
        
        # Use vectorized operations for efficiency
        for cell_id in frame_cells:
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
    if bleach_correction and not df.empty:
        print("\nApplying photobleaching correction...")
        df_corrected, model_params, correction_factors = apply_bleaching_correction(
            df, bleach_model)
        
        # Replace original DataFrame
        df = df_corrected
    
    # Step 6: Apply baseline normalization if requested
    if baseline_frames is not None and baseline_frames > 0 and baseline_frames < num_frames and not df.empty:
        print(f"\nNormalizing ratio values to first {baseline_frames} frames...")
        df = normalize_to_baseline(df, baseline_frames, bleach_correction)
    
    # Step 7: Filter cells that leave field of view
    in_view_cells = [cell_id for cell_id, track in cell_tracks.items() if track['in_view']]
    
    # Filter to only keep cells visible through all frames
    full_track_cells = [cell_id for cell_id, track in cell_tracks.items() 
                     if len(track['frames']) == num_frames]
    
    print(f"\nTracking statistics:")
    print(f"  Total cells detected: {len(cell_tracks)}")
    print(f"  Cells that stayed in view: {len(in_view_cells)}")
    print(f"  Cells tracked through all frames: {len(full_track_cells)}")
    
    # Filter DataFrame to only include full-track cells
    if full_track_cells:
        df_filtered = df[df['cell_id'].isin(full_track_cells)]
    else:
        df_filtered = df.copy()
        print("Warning: No cells were tracked through all frames.")
    
    # Save the full data and filtered data
    df.to_csv(os.path.join(output_dir, f"{file_name}_all_cells_data.csv"), index=False)
    df_filtered.to_csv(os.path.join(output_dir, f"{file_name}_tracked_cells_data.csv"), index=False)
    
    # Step 8: Generate analysis plots
    print("\nGenerating analysis plots...")
    
    # Create plots
    if not df_filtered.empty:
        # 1. Ratio plots
        create_ratio_plot(df_filtered, corrected=False, output_dir=output_dir, file_name=file_name)
        
        if bleach_correction:
            create_ratio_plot(df_filtered, corrected=True, output_dir=output_dir, file_name=file_name)
        
        # 2. Channel intensity plots
        create_channel_plots(df_filtered, bleach_correction, output_dir, file_name)
        
        # 3. Create normalized ratio plots if baseline normalization was applied
        if baseline_frames is not None and 'ratio_normalized' in df_filtered.columns:
            create_normalized_ratio_plot(df_filtered, bleach_correction, output_dir, file_name, baseline_frames)
        
        # 4. Create cell tracking visualization
        create_tracking_visualization(master_labels, cell_tracks, full_track_cells, output_dir, file_name)
    else:
        print("Warning: No cell data available for plotting.")
    
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


def normalize_to_baseline(df, baseline_frames, bleach_correction=True):
    """
    Normalize ratio values to the average of the first baseline_frames
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with cell measurements
    baseline_frames : int
        Number of initial frames to use as baseline
    bleach_correction : bool
        Whether bleaching correction was applied
        
    Returns:
    --------
    df_normalized : pandas.DataFrame
        DataFrame with normalized ratio values
    """
    # Create a copy of the DataFrame
    df_normalized = df.copy()
    
    # Normalize each cell individually
    for cell_id in df_normalized['cell_id'].unique():
        cell_df = df_normalized[df_normalized['cell_id'] == cell_id]
        
        # Get baseline frames for this cell
        baseline_df = cell_df[cell_df['time_point'] < baseline_frames]
        
        if len(baseline_df) > 0:
            # Calculate baseline average for the original ratio
            baseline_avg = baseline_df['ratio'].mean()
            if baseline_avg > 0:
                # Apply normalization
                df_normalized.loc[df_normalized['cell_id'] == cell_id, 'ratio_normalized'] = \
                    df_normalized.loc[df_normalized['cell_id'] == cell_id, 'ratio'] / baseline_avg
            
            # Also normalize corrected ratio if available
            if bleach_correction and 'ratio_corrected' in cell_df.columns:
                baseline_avg_corrected = baseline_df['ratio_corrected'].mean()
                if baseline_avg_corrected > 0:
                    # Apply normalization
                    df_normalized.loc[df_normalized['cell_id'] == cell_id, 'ratio_corrected_normalized'] = \
                        df_normalized.loc[df_normalized['cell_id'] == cell_id, 'ratio_corrected'] / baseline_avg_corrected
    
    return df_normalized


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
    counts = time_groups[ratio_column].count()
    std_ratios = time_groups[ratio_column].std() / np.sqrt(counts)
    
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
    counts_ref = time_groups['reference_intensity'].count()
    std_ref = time_groups['reference_intensity'].std() / np.sqrt(counts_ref)
    
    mean_meas = time_groups['measurement_intensity'].mean()
    counts_meas = time_groups['measurement_intensity'].count()
    std_meas = time_groups['measurement_intensity'].std() / np.sqrt(counts_meas)
    
    # Plot reference channel
    plt.errorbar(mean_ref.index, mean_ref, yerr=std_ref, 
               fmt='o-', linewidth=2, capsize=4, 
               label='Reference Channel', color='blue')
    
    # If bleach correction applied, also plot corrected reference
    if bleach_correction and 'reference_intensity_corrected' in df.columns:
        mean_ref_corr = time_groups['reference_intensity_corrected'].mean()
        std_ref_corr = time_groups['reference_intensity_corrected'].std() / np.sqrt(counts_ref)
        
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


def create_normalized_ratio_plot(df, bleach_correction, output_dir='.', file_name='output', baseline_frames=None):
    """Create plots of normalized ratio values"""
    plt.figure(figsize=(12, 8))
    
    # Group by time point
    time_groups = df.groupby('time_point')
    
    # Calculate stats for normalized ratio
    mean_ratio_norm = time_groups['ratio_normalized'].mean()
    counts_norm = time_groups['ratio_normalized'].count()
    std_ratio_norm = time_groups['ratio_normalized'].std() / np.sqrt(counts_norm)
    
    # Plot normalized ratio
    plt.errorbar(mean_ratio_norm.index, mean_ratio_norm, yerr=std_ratio_norm, 
               fmt='o-', linewidth=2, capsize=4, 
               label='Normalized Ratio', color='green')
    
    # Also plot corrected normalized ratio if available
    if bleach_correction and 'ratio_corrected_normalized' in df.columns:
        mean_ratio_corr_norm = time_groups['ratio_corrected_normalized'].mean()
        counts_corr_norm = time_groups['ratio_corrected_normalized'].count()
        std_ratio_corr_norm = time_groups['ratio_corrected_normalized'].std() / np.sqrt(counts_corr_norm)
        
        plt.errorbar(mean_ratio_corr_norm.index, mean_ratio_corr_norm, yerr=std_ratio_corr_norm, 
                   fmt='o-', linewidth=2, capsize=4, 
                   label='Normalized Corrected Ratio', color='purple')
    
    # Add vertical line at baseline/treatment transition if specified
    if baseline_frames is not None:
        plt.axvline(x=baseline_frames-0.5, color='red', linestyle='--', 
                  label='Baseline → Treatment')
    
    plt.title(f'Normalized Ratio Values (Baseline: First {baseline_frames} Frames)', fontsize=14)
    plt.xlabel('Time Point', fontsize=12)
    plt.ylabel('Normalized Ratio (Relative to Baseline)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Draw horizontal line at y=1.0 to indicate baseline level
    plt.axhline(y=1.0, color='gray', linestyle='-', alpha=0.5)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, f"{file_name}_normalized_ratio.png"))
    plt.close()


def create_all_cells_ratio_plot(df, bleach_correction, output_dir='.', file_name='output'):
    """Create a single plot showing all cell ratio traces"""
    plt.figure(figsize=(12, 8))
    
    ratio_column = 'ratio_corrected' if bleach_correction and 'ratio_corrected' in df.columns else 'ratio'
    
    # Get unique cell IDs
    cell_ids = df['cell_id'].unique()
    
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
    
    # Ask if user wants to use baseline normalization
    try:
        baseline_input = input("Enter number of baseline frames (or press Enter to skip normalization): ")
        baseline_frames = int(baseline_input) if baseline_input.strip() else None
    except ValueError:
        print("Invalid input. Proceeding without baseline normalization.")
        baseline_frames = None
    
    print("\nUsing settings:")
    print(f"  Reference channel: {ref_channel}")
    print(f"  Measurement channel: {meas_channel}")
    print(f"  Minimum cell size: {min_cell_size} pixels")
    print(f"  Adaptive thresholding: Enabled (block size {adaptive_block_size})")
    print(f"  Watershed segmentation: Enabled (min distance {watershed_min_distance})")
    print(f"  Contrast enhancement: {percentile_low} to {percentile_high} percentiles")
    print(f"  Bleach correction: {'Enabled' if bleach_correction else 'Disabled'}")
    print(f"  Baseline normalization: {'First ' + str(baseline_frames) + ' frames' if baseline_frames else 'Disabled'}")
    print()
    
    # Run the analysis
    segment_track_analyze_tif(tif_path, ref_channel=ref_channel, meas_channel=meas_channel,
                           min_cell_size=min_cell_size, 
                           use_adaptive=use_adaptive, adaptive_block_size=adaptive_block_size,
                           use_watershed=use_watershed, watershed_min_distance=watershed_min_distance,
                           percentile_low=percentile_low, percentile_high=percentile_high,
                           bleach_correction=bleach_correction,
                           baseline_frames=baseline_frames)