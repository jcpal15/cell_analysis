import numpy as np
import tifffile
from skimage import filters, measure, exposure, transform
from scipy import ndimage, optimize, stats
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm
import datetime
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
import multiprocessing
from functools import partial

def segment_track_analyze_tif(tif_path, output_dir=None, ref_channel=1, meas_channel=0, 
                           min_cell_size=100, max_tracking_distance=20,
                           use_adaptive=True, adaptive_block_size=35, 
                           use_watershed=True, watershed_min_distance=10,
                           percentile_low=0.1, percentile_high=99.9,
                           bleach_correction=True, bleach_model='exponential',
                           baseline_frames=None, segmentation_interval=5,
                           use_hungarian=True, n_processes=None):
    """
    Optimized version of segment_track_analyze_tif with several speed improvements
    
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
        Number of frames considered as baseline for ratio normalization
    segmentation_interval : int
        Interval between frames to perform full segmentation (default: 5)
        A value of 1 means segment every frame, 5 means segment every 5th frame
    use_hungarian : bool
        Whether to use the Hungarian algorithm for cell assignment (default: True)
    n_processes : int or None
        Number of processes to use for parallel processing (default: None = auto)
    
    Returns:
    --------
    tuple
        DataFrame with cell measurements, cell_tracks dictionary, and labels array
    """
    # Create timestamp for output directory
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
    
    # Create master labels array to store cell IDs across all frames
    master_labels = np.zeros((num_frames, height, width), dtype=np.int32)
    
    # Determine which frames to fully segment
    segmentation_frames = list(range(0, num_frames, segmentation_interval))
    if (num_frames - 1) not in segmentation_frames:
        segmentation_frames.append(num_frames - 1)  # Always segment the last frame
    
    print(f"Will perform full segmentation on frames: {segmentation_frames}")
    
    # Step 2: Process segmentation frames
    for frame_idx, frame in enumerate(tqdm(segmentation_frames, desc="Segmenting frames")):
        # Get reference channel image
        ref_img = tif_data[frame, ref_channel]
        
        # Skip segmentation for very first frame as it needs special handling
        if frame_idx == 0:
            # First frame - perform thorough segmentation
            labels = segment_frame(ref_img, min_cell_size, use_adaptive, adaptive_block_size, 
                                use_watershed, watershed_min_distance, 
                                percentile_low, percentile_high)
            
            # Save segmentation for first frame
            plt.figure(figsize=(10, 8))
            plt.imshow(ref_img, cmap='gray')
            plt.imshow(labels > 0, alpha=0.5, cmap='viridis')
            plt.title(f"Segmentation - Frame {frame}")
            plt.savefig(os.path.join(output_dir, f"{file_name}_segmentation_frame{frame}.png"))
            plt.close()
            
            # Store in master labels
            master_labels[frame] = labels
            
            # Initialize cell tracking for first frame
            props = measure.regionprops(labels)
            for prop in props:
                cell_id = prop.label
                if cell_id > 0:  # Skip background
                    cell_tracks[cell_id] = {
                        'frames': [frame],
                        'centroids': [prop.centroid],
                        'areas': [prop.area],
                        'bounding_boxes': [prop.bbox],
                        'in_view': True
                    }
        else:
            # Subsequent segmentation frames
            # Get previous segmentation frame
            prev_frame = segmentation_frames[frame_idx - 1]
            prev_labels = master_labels[prev_frame]
            
            # Transfer previous masks and refine them for the current frame
            # This is more efficient than segmenting from scratch
            labels = propagate_and_refine_segmentation(
                ref_img, prev_labels, prev_frame, frame, 
                min_cell_size, use_adaptive, adaptive_block_size, 
                use_watershed, watershed_min_distance, 
                percentile_low, percentile_high
            )
            
            # Store in master labels
            master_labels[frame] = labels
            
            # Match cells with previous frame
            match_cells_between_frames(
                prev_frame, frame, prev_labels, labels, 
                cell_tracks, master_labels, max_tracking_distance, use_hungarian
            )
    
    # Step 3: Fill in the intermediate frames using interpolation
    print("Processing intermediate frames...")
    for i in range(len(segmentation_frames) - 1):
        start_frame = segmentation_frames[i]
        end_frame = segmentation_frames[i + 1]
        
        # Skip if frames are consecutive
        if end_frame - start_frame <= 1:
            continue
        
        # Interpolate for frames between start_frame and end_frame
        for frame in range(start_frame + 1, end_frame):
            print(f"Interpolating frame {frame} between {start_frame} and {end_frame}")
            
            # Calculate interpolation factor (0 at start_frame, 1 at end_frame)
            alpha = (frame - start_frame) / (end_frame - start_frame)
            
            # Get reference channel for current frame
            current_ref_img = tif_data[frame, ref_channel]
            
            # Interpolate masks between start and end frames
            master_labels[frame] = interpolate_labels(
                master_labels[start_frame], master_labels[end_frame], 
                alpha, current_ref_img, min_cell_size
            )
            
            # Match cells and update tracking data
            found_cells = np.unique(master_labels[frame])
            found_cells = found_cells[found_cells > 0]  # Remove background
            
            for cell_id in found_cells:
                if cell_id in cell_tracks:
                    # Calculate interpolated centroid
                    start_idx = cell_tracks[cell_id]['frames'].index(start_frame) if start_frame in cell_tracks[cell_id]['frames'] else None
                    end_idx = cell_tracks[cell_id]['frames'].index(end_frame) if end_frame in cell_tracks[cell_id]['frames'] else None
                    
                    if start_idx is not None and end_idx is not None:
                        # Cell exists in both start and end frames, interpolate centroid
                        start_centroid = cell_tracks[cell_id]['centroids'][start_idx]
                        end_centroid = cell_tracks[cell_id]['centroids'][end_idx]
                        
                        # Linear interpolation
                        interpolated_centroid = (
                            start_centroid[0] * (1 - alpha) + end_centroid[0] * alpha,
                            start_centroid[1] * (1 - alpha) + end_centroid[1] * alpha
                        )
                        
                        # Update tracking data
                        cell_tracks[cell_id]['frames'].append(frame)
                        cell_tracks[cell_id]['centroids'].append(interpolated_centroid)
                        
                        # Get area from mask
                        area = np.sum(master_labels[frame] == cell_id)
                        cell_tracks[cell_id]['areas'].append(area)
                        
                        # Calculate bounding box
                        props = measure.regionprops(np.asarray(master_labels[frame] == cell_id, dtype=np.int32))
                        if props:
                            cell_tracks[cell_id]['bounding_boxes'].append(props[0].bbox)
                        else:
                            # Use interpolated bounding box as fallback
                            start_bbox = cell_tracks[cell_id]['bounding_boxes'][start_idx]
                            end_bbox = cell_tracks[cell_id]['bounding_boxes'][end_idx]
                            interpolated_bbox = tuple(
                                start_bbox[i] * (1 - alpha) + end_bbox[i] * alpha for i in range(len(start_bbox))
                            )
                            cell_tracks[cell_id]['bounding_boxes'].append(interpolated_bbox)
    
    # Step 4: Measure cell properties for all frames
    print("Measuring cell properties for all frames...")
    
    # Function to process a single frame
    def process_frame(frame):
        ref_img = tif_data[frame, ref_channel]
        meas_img = tif_data[frame, meas_channel]
        
        frame_cells = np.unique(master_labels[frame])
        frame_cells = frame_cells[frame_cells > 0]  # Remove background
        
        frame_data = []
        
        for cell_id in frame_cells:
            cell_mask = master_labels[frame] == cell_id
            
            if np.any(cell_mask):
                # Cell found in this frame, measure properties
                ref_intensity = np.mean(ref_img[cell_mask])
                meas_intensity = np.mean(meas_img[cell_mask])
                
                # Calculate ratio
                ratio = meas_intensity / ref_intensity if ref_intensity > 0 else 0
                
                # Store measurements
                frame_data.append({
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
        
        return frame_data
    
    # Use parallel processing for measurement
    if n_processes is None:
        n_processes = max(1, multiprocessing.cpu_count() - 1)  # Use all but one CPU
    
    print(f"Using {n_processes} processes for parallel measurement")
    
    with multiprocessing.Pool(processes=n_processes) as pool:
        all_frame_data_lists = list(tqdm(
            pool.imap(process_frame, range(num_frames)),
            total=num_frames,
            desc="Measuring cells"
        ))
    
    # Flatten the list of lists
    for frame_data in all_frame_data_lists:
        all_frames_data.extend(frame_data)
    
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

def segment_frame(ref_img, min_cell_size=100, use_adaptive=True, adaptive_block_size=35, 
               use_watershed=True, watershed_min_distance=10, 
               percentile_low=0.1, percentile_high=99.9):
    """
    Segment a single frame to identify cells
    
    Parameters:
    -----------
    ref_img : numpy.ndarray
        Reference channel image
    min_cell_size : int
        Minimum size of cells in pixels
    use_adaptive : bool
        Whether to use adaptive thresholding
    adaptive_block_size : int
        Block size for adaptive thresholding
    use_watershed : bool
        Whether to use watershed segmentation
    watershed_min_distance : int
        Minimum distance between peaks for watershed
    percentile_low, percentile_high : float
        Percentiles for contrast enhancement
        
    Returns:
    --------
    labels : numpy.ndarray
        Labeled image with cell regions
    """
    # Normalize image
    ref_img_norm = (ref_img - np.min(ref_img)) / (np.max(ref_img) - np.min(ref_img))
    
    # Apply Gaussian smoothing
    smoothed = filters.gaussian(ref_img_norm, sigma=1.0)
    
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
    binary = ndimage.binary_closing(binary, iterations=2)
    binary = ndimage.binary_opening(binary, iterations=1)
    binary = ndimage.binary_fill_holes(binary)
    
    # Apply watershed or standard labeling
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
    
    # Filter small regions
    props = measure.regionprops(labels)
    
    filtered_labels = np.zeros_like(labels)
    for prop in props:
        if prop.area >= min_cell_size:
            filtered_labels[labels == prop.label] = prop.label
    
    return filtered_labels

def propagate_and_refine_segmentation(ref_img, prev_labels, prev_frame, current_frame, 
                                   min_cell_size=100, use_adaptive=True, adaptive_block_size=35, 
                                   use_watershed=True, watershed_min_distance=10, 
                                   percentile_low=0.1, percentile_high=99.9):
    """
    Propagate previous segmentation to current frame and refine
    
    Parameters:
    -----------
    ref_img : numpy.ndarray
        Reference channel image for current frame
    prev_labels : numpy.ndarray
        Labels from previous frame
    prev_frame : int
        Previous frame index
    current_frame : int
        Current frame index
    ... (other parameters same as segment_frame)
        
    Returns:
    --------
    labels : numpy.ndarray
        Labeled image with cell regions for current frame
    """
    # First perform standard segmentation on current frame
    current_labels = segment_frame(ref_img, min_cell_size, use_adaptive, adaptive_block_size, 
                                use_watershed, watershed_min_distance, 
                                percentile_low, percentile_high)
    
    # For small frame gaps, we can use the previous segmentation as a guide
    frame_gap = current_frame - prev_frame
    
    if frame_gap <= 3:  # Only use previous labels if frames are close
        # Get properties of cells in previous frame
        prev_props = measure.regionprops(prev_labels)
        
        # For each cell in previous frame, use it as a seed for the current frame
        for prop in prev_props:
            cell_id = prop.label
            if cell_id > 0:  # Skip background
                # Get previous cell mask
                prev_mask = prev_labels == cell_id
                
                # Estimate movement between frames (can be refined with optical flow)
                # Here we use a simple dilation to account for potential movement
                dilated_mask = ndimage.binary_dilation(prev_mask, iterations=frame_gap * 2)
                
                # Find overlapping regions in current segmentation
                overlap_labels = np.unique(current_labels[dilated_mask])
                overlap_labels = overlap_labels[overlap_labels > 0]  # Remove background
                
                if len(overlap_labels) > 0:
                    # Find best match based on overlap
                    max_overlap = 0
                    best_label = None
                    
                    for label in overlap_labels:
                        overlap = np.sum((current_labels == label) & dilated_mask)
                        if overlap > max_overlap:
                            max_overlap = overlap
                            best_label = label
                    
                    if best_label is not None:
                        # Replace the current label with the cell ID from previous frame
                        current_labels[current_labels == best_label] = cell_id
    
    return current_labels

def interpolate_labels(start_labels, end_labels, alpha, current_img=None, min_cell_size=100):
    """
    Interpolate between two labeled images
    
    Parameters:
    -----------
    start_labels : numpy.ndarray
        Labels from start frame
    end_labels : numpy.ndarray
        Labels from end frame
    alpha : float
        Interpolation factor (0-1)
    current_img : numpy.ndarray or None
        Reference image for the current frame (for refinement)
    min_cell_size : int
        Minimum size of cells in pixels
        
    Returns:
    --------
    interp_labels : numpy.ndarray
        Interpolated labels
    """
    # Get all cell IDs present in either start or end frame
    all_cell_ids = np.unique(np.concatenate([
        np.unique(start_labels), 
        np.unique(end_labels)
    ]))
    all_cell_ids = all_cell_ids[all_cell_ids > 0]  # Remove background
    
    # Create output array
    interp_labels = np.zeros_like(start_labels)
    
    # Process each cell ID
    for cell_id in all_cell_ids:
        # Check if cell exists in both frames
        start_mask = (start_labels == cell_id)
        end_mask = (end_labels == cell_id)
        
        if np.any(start_mask) and np.any(end_mask):
            # Cell exists in both frames
            
            # Get centroids
            start_props = measure.regionprops(start_mask.astype(np.int32))
            end_props = measure.regionprops(end_mask.astype(np.int32))
            
            if start_props and end_props:
                start_centroid = start_props[0].centroid
                end_centroid = end_props[0].centroid
                
                # Calculate interpolated centroid
                interp_centroid = (
                    start_centroid[0] * (1 - alpha) + end_centroid[0] * alpha,
                    start_centroid[1] * (1 - alpha) + end_centroid[1] * alpha
                )
                
                # Create interpolated mask
                # Option 1: Morphological interpolation
                if alpha <= 0.5:
                    # Closer to start frame
                    base_mask = start_mask.copy()
                    morph_steps = int(alpha * 10)  # Scale number of steps based on alpha
                else:
                    # Closer to end frame
                    base_mask = end_mask.copy()
                    morph_steps = int((1 - alpha) * 10)
                
                # Apply morphological operations to approximate intermediate shape
                dilated = ndimage.binary_dilation(base_mask, iterations=morph_steps)
                interp_mask = ndimage.binary_erosion(dilated, iterations=morph_steps//2)
                
                # Ensure minimum size
                if np.sum(interp_mask) < min_cell_size:
                    continue
                
                # Add to interpolated labels
                interp_labels[interp_mask] = cell_id
        elif np.any(start_mask) and alpha < 0.5:
            # Cell only exists in start frame and we're closer to start
            interp_labels[start_mask] = cell_id
        elif np.any(end_mask) and alpha >= 0.5:
            # Cell only exists in end frame and we're closer to end
            interp_labels[end_mask] = cell_id
    
    # Optional: Refine using current image if provided
    if current_img is not None:
        # Simple refinement - ensure each cell is a connected component
        for cell_id in all_cell_ids:
            cell_mask = (interp_labels == cell_id)
            if np.any(cell_mask):
                # Label connected components
                labeled_components = measure.label(cell_mask)
                
                # Keep only the largest component
                largest_size = 0
                largest_label = 0
                
                for i, region in enumerate(measure.regionprops(labeled_components)):
                    if region.area > largest_size:
                        largest_size = region.area
                        largest_label = region.label
                
                # Update mask
                if largest_label > 0:
                    interp_labels[(interp_labels == cell_id) & (labeled_components != largest_label)] = 0
    
    return interp_labels

def match_cells_between_frames(prev_frame, current_frame, prev_labels, current_labels,
                           cell_tracks, master_labels, max_tracking_distance=20,
                           use_hungarian=True):
    """
    Match cells between two frames
    
    Parameters:
    -----------
    prev_frame : int
        Previous frame index
    current_frame : int
        Current frame index
    prev_labels : numpy.ndarray
        Labels from previous frame
    current_labels : numpy.ndarray
        Labels from current frame
    cell_tracks : dict
        Cell tracking information
    master_labels : numpy.ndarray
        Master labels array for all frames
    max_tracking_distance : float
        Maximum distance for tracking
    use_hungarian : bool
        Whether to use Hungarian algorithm for assignment
    """
    # Get properties of cells in both frames
    prev_props = measure.regionprops(prev_labels)
    curr_props = measure.regionprops(current_labels)
    
    # Skip if either frame has no cells
    if not prev_props or not curr_props:
        print(f"Warning: No cells found in either previous frame {prev_frame} or current frame {current_frame}")
        return
    
    # Extract centroids and IDs
    prev_centroids = np.array([prop.centroid for prop in prev_props])
    prev_ids = np.array([prop.label for prop in prev_props])
    
    curr_centroids = np.array([prop.centroid for prop in curr_props])
    curr_ids = np.array([prop.label for prop in curr_props])
    
    # Calculate distance matrix between all pairs of centroids
    distance_matrix = np.zeros((len(prev_ids), len(curr_ids)))
    
    for i, prev_centroid in enumerate(prev_centroids):
        for j, curr_centroid in enumerate(curr_centroids):
            dist = np.sqrt(np.sum((prev_centroid - curr_centroid)**2))
            distance_matrix[i, j] = dist
    
    # Set large distances to infinity (beyond max tracking distance)
    distance_matrix[distance_matrix > max_tracking_distance] = np.inf
    
    # Check if we can use Hungarian algorithm (feasible cost matrix)
    if use_hungarian:
        # Check if the distance matrix contains any finite values
        if np.any(np.isfinite(distance_matrix)):
            try:
                # Use Hungarian algorithm for optimal assignment
                # This minimizes the total distance between matched cells
                row_indices, col_indices = linear_sum_assignment(distance_matrix)
                
                # Process assignments
                for i, j in zip(row_indices, col_indices):
                    if distance_matrix[i, j] < np.inf:
                        prev_id = prev_ids[i]
                        curr_id = curr_ids[j]
                        
                        # Update master labels
                        master_labels[current_frame][current_labels == curr_id] = prev_id
                        
                        # Update tracking data
                        if prev_id in cell_tracks:
                            cell_tracks[prev_id]['frames'].append(current_frame)
                            cell_tracks[prev_id]['centroids'].append(curr_centroids[j])
                            
                            # Calculate area
                            area = np.sum(current_labels == curr_id)
                            cell_tracks[prev_id]['areas'].append(area)
                            
                            # Calculate bounding box
                            cell_tracks[prev_id]['bounding_boxes'].append(curr_props[j].bbox)
                
                # Create new cells for unmatched current cells
                matched_curr_indices = set(col_indices[distance_matrix[row_indices, col_indices] < np.inf])
                
                for j, curr_id in enumerate(curr_ids):
                    if j not in matched_curr_indices:
                        # Create a new cell ID
                        new_id = curr_id + current_frame * 10000
                        
                        # Update master labels
                        master_labels[current_frame][current_labels == curr_id] = new_id
                        
                        # Create new tracking data
                        cell_tracks[new_id] = {
                            'frames': [current_frame],
                            'centroids': [curr_centroids[j]],
                            'areas': [curr_props[j].area],
                            'bounding_boxes': [curr_props[j].bbox],
                            'in_view': True
                        }
            except ValueError as e:
                # Fall back to nearest neighbor if Hungarian algorithm fails
                print(f"Warning: Hungarian algorithm failed ({e}), falling back to nearest neighbor")
                use_hungarian = False
        else:
            # No feasible assignments, fall back to nearest neighbor
            print(f"Warning: No valid cell matches found between frames {prev_frame} and {current_frame}, falling back to nearest neighbor")
            use_hungarian = False
    
    # Use nearest neighbor approach if Hungarian algorithm is disabled or failed
    if not use_hungarian:
        # Use a simple greedy nearest-neighbor approach
        # Process each previous cell
        matched_curr_indices = set()
        
        for i, prev_id in enumerate(prev_ids):
            # Find the closest unmatched current cell
            min_dist = np.inf
            best_j = None
            
            for j, curr_id in enumerate(curr_ids):
                if j not in matched_curr_indices and distance_matrix[i, j] < min_dist:
                    min_dist = distance_matrix[i, j]
                    best_j = j
            
            if best_j is not None and min_dist < np.inf:
                # Match found
                curr_id = curr_ids[best_j]
                
                # Update master labels
                master_labels[current_frame][current_labels == curr_id] = prev_id
                
                # Update tracking data
                if prev_id in cell_tracks:
                    cell_tracks[prev_id]['frames'].append(current_frame)
                    cell_tracks[prev_id]['centroids'].append(curr_centroids[best_j])
                    
                    # Calculate area
                    area = np.sum(current_labels == curr_id)
                    cell_tracks[prev_id]['areas'].append(area)
                    
                    # Calculate bounding box
                    cell_tracks[prev_id]['bounding_boxes'].append(curr_props[best_j].bbox)
                
                # Mark as matched
                matched_curr_indices.add(best_j)
        
        # Create new cells for unmatched current cells
        for j, curr_id in enumerate(curr_ids):
            if j not in matched_curr_indices:
                # Create a new cell ID
                new_id = curr_id + current_frame * 10000
                
                # Update master labels
                master_labels[current_frame][current_labels == curr_id] = new_id
                
                # Create new tracking data
                cell_tracks[new_id] = {
                    'frames': [current_frame],
                    'centroids': [curr_centroids[j]],
                    'areas': [curr_props[j].area],
                    'bounding_boxes': [curr_props[j].bbox],
                    'in_view': True
                }

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
    std_ratios = time_groups[ratio_column].std() / np.sqrt(counts)  # Standard error
    
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
    std_ref = time_groups['reference_intensity'].std() / np.sqrt(counts_ref)  # Standard error
    
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
    num_colors = len(full_track_cells) if full_track_cells else 20
    cmap = plt.cm.get_cmap('tab20', num_colors)
    
    # Create a mapping of cell IDs to colors
    color_map = {}
    for i, cell_id in enumerate(full_track_cells or cell_tracks.keys()):
        color_map[cell_id] = i % num_colors
    
    # Create the visualization
    plt.figure(figsize=(20, 5))
    
    for i, frame_idx in enumerate(frames_to_show):
        plt.subplot(1, len(frames_to_show), i+1)
        
        # Create an RGB image
        height, width = master_labels[frame_idx].shape
        vis_img = np.zeros((height, width, 3))
        
        # Add each tracked cell with its color
        cells_in_frame = np.unique(master_labels[frame_idx])
        cells_in_frame = cells_in_frame[cells_in_frame > 0]  # Remove background
        
        for cell_id in cells_in_frame:
            if full_track_cells is None or cell_id in full_track_cells:
                # Get cell mask
                mask = master_labels[frame_idx] == cell_id
                
                if np.any(mask):
                    # Apply cell color if cell is in color map
                    if cell_id in color_map:
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
    
    # Create simplified tracking visualization for a few selected cells
    # This can help to visualize the movement of individual cells
    if full_track_cells and len(full_track_cells) > 0:
        # Select up to 5 cells to visualize trajectories
        selected_cells = full_track_cells[:min(5, len(full_track_cells))]
        
        plt.figure(figsize=(12, 10))
        
        # Plot the last frame as background
        last_frame_idx = frames_to_show[-1]
        
        # Create grayscale image of last frame
        background = np.zeros((master_labels.shape[1], master_labels.shape[2]))
        
        # Add each cell to the background
        cells_in_frame = np.unique(master_labels[last_frame_idx])
        cells_in_frame = cells_in_frame[cells_in_frame > 0]  # Remove background
        
        for cell_id in cells_in_frame:
            mask = master_labels[last_frame_idx] == cell_id
            background[mask] = 0.5  # Medium gray
            
        plt.imshow(background, cmap='gray', alpha=0.5)
        
        # Plot trajectories for selected cells
        for i, cell_id in enumerate(selected_cells):
            if cell_id in cell_tracks:
                # Get cell trajectory
                frames = cell_tracks[cell_id]['frames']
                centroids = cell_tracks[cell_id]['centroids']
                
                # Convert centroids to x, y arrays
                y_coords = [c[0] for c in centroids]
                x_coords = [c[1] for c in centroids]
                
                # Plot trajectory
                color = cmap(i % num_colors)
                plt.plot(x_coords, y_coords, '-o', color=color, linewidth=2, markersize=4, 
                       label=f'Cell {cell_id}')
                
                # Mark the start and end points
                plt.plot(x_coords[0], y_coords[0], 'o', color=color, markersize=8, markeredgecolor='white')
                plt.plot(x_coords[-1], y_coords[-1], 's', color=color, markersize=8, markeredgecolor='white')
        
        plt.title('Cell Trajectories', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, f"{file_name}_cell_trajectories.png"))
        plt.close()
    
    return

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Optimized cell tracking analysis for time-lapse TIF files')
    
    # Required arguments
    parser.add_argument('tif_path', type=str, help='Path to the TIF file')
    
    # Optional arguments
    parser.add_argument('--output', type=str, default=None, help='Output directory (default: auto-generated)')
    parser.add_argument('--ref', type=int, default=1, help='Reference channel index (default: 1)')
    parser.add_argument('--meas', type=int, default=0, help='Measurement channel index (default: 0)')
    parser.add_argument('--min-size', type=int, default=100, help='Minimum cell size in pixels (default: 100)')
    parser.add_argument('--max-distance', type=float, default=20, help='Maximum tracking distance (default: 20)')
    parser.add_argument('--no-adaptive', action='store_false', dest='adaptive', help='Disable adaptive thresholding')
    parser.add_argument('--block-size', type=int, default=35, help='Block size for adaptive thresholding (default: 35)')
    parser.add_argument('--no-watershed', action='store_false', dest='watershed', help='Disable watershed segmentation')
    parser.add_argument('--watershed-distance', type=int, default=10, help='Minimum distance for watershed peaks (default: 10)')
    parser.add_argument('--percentile-low', type=float, default=0.1, help='Lower percentile for contrast enhancement (default: 0.1)')
    parser.add_argument('--percentile-high', type=float, default=99.9, help='Upper percentile for contrast enhancement (default: 99.9)')
    parser.add_argument('--no-bleach-correction', action='store_false', dest='bleach_correction', help='Disable bleaching correction')
    parser.add_argument('--bleach-model', type=str, choices=['exponential', 'linear', 'polynomial'], default='exponential', help='Bleaching model (default: exponential)')
    parser.add_argument('--baseline-frames', type=int, default=None, help='Number of frames for baseline normalization (default: None)')
    parser.add_argument('--segmentation-interval', type=int, default=5, help='Interval between frames for full segmentation (default: 5)')
    parser.add_argument('--no-hungarian', action='store_false', dest='hungarian', help='Disable Hungarian algorithm for cell matching')
    parser.add_argument('--processes', type=int, default=None, help='Number of processes for parallel processing (default: auto)')
    
    # Set defaults for optional arguments
    parser.set_defaults(adaptive=True, watershed=True, bleach_correction=True, hungarian=True)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run analysis
    segment_track_analyze_tif(
        args.tif_path,
        output_dir=args.output,
        ref_channel=args.ref,
        meas_channel=args.meas,
        min_cell_size=args.min_size,
        max_tracking_distance=args.max_distance,
        use_adaptive=args.adaptive,
        adaptive_block_size=args.block_size,
        use_watershed=args.watershed,
        watershed_min_distance=args.watershed_distance,
        percentile_low=args.percentile_low,
        percentile_high=args.percentile_high,
        bleach_correction=args.bleach_correction,
        bleach_model=args.bleach_model,
        baseline_frames=args.baseline_frames,
        segmentation_interval=args.segmentation_interval,
        use_hungarian=args.hungarian,
        n_processes=args.processes
    )