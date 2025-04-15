# Helper function for downsampling images
def downsample_image(image, factor):
    """Downsample an image by the given factor using block reducing"""
    if factor <= 1:
        return image
    
    # Use block reduce from skimage for clean downsampling
    from skimage.measure import block_reduce
    return block_reduce(image, block_size=(factor, factor), func=np.mean)

# Helper function for upsampling masks
def upsample_mask(mask, original_shape):
    """Upsample a mask to the original image shape"""
    from skimage.transform import resize
    return resize(mask.astype(float), original_shape, order=0, preserve_range=True, anti_aliasing=False).astype(int)

import numpy as np
import tifffile
from skimage import filters, measure, exposure, segmentation, feature, morphology
from scipy import ndimage, optimize, stats
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm
import datetime
from pathlib import Path
from scipy.spatial import cKDTree
import warnings

# Suppress specific warnings that might occur during processing
warnings.filterwarnings("ignore", category=UserWarning, module="skimage")

def segment_track_analyze_tif(tif_path, output_dir=None, ref_channel=1, meas_channel=0, 
                           min_cell_size=100, max_tracking_distance=20,
                           enhance_method='adaptive', denoise_method='tv', sigma=2.0,
                           threshold_method='li', marker_method='distance', 
                           min_distance=15, segmentation_method='watershed',
                           split_touching=True, border_clearing=False,
                           n_clusters=4, max_holes_size=50,
                           bleach_correction=True, bleach_model='exponential',
                           baseline_frames=None, enable_tracking=True,
                           analyze_all_frames=True, downsample_factor=1,
                           fast_mode_method='otsu', use_cached_segmentation=True):
    """
    Segment, track, and analyze cells in a time-lapse TIF file using improved segmentation.
    
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
    enhance_method : str
        Method for contrast enhancement ('rescale', 'adaptive', 'hist_eq')
    denoise_method : str
        Method for denoising ('gaussian', 'median', 'bilateral', 'tv')
    sigma : float
        Sigma parameter for Gaussian denoising
    threshold_method : str
        Method for thresholding ('otsu', 'multiotsu', 'local', 'li', 'absolute')
    marker_method : str
        Method for finding cell markers ('distance', 'log', 'dog', 'h_maxima')
    min_distance : int
        Minimum distance between cell markers
    segmentation_method : str
        Method for segmentation ('watershed', 'random_walker')
    split_touching : bool
        Whether to attempt splitting touching cells
    border_clearing : bool
        Whether to remove cells touching image borders
    n_clusters : int
        Number of clusters for intensity-based grouping
    max_holes_size : int
        Maximum size of holes to fill in binary mask
    bleach_correction : bool
        Whether to apply bleaching correction (default: True)
    bleach_model : str
        Type of bleaching model ('exponential', 'linear', 'polynomial')
    baseline_frames : int or None
        Number of initial frames to use as baseline for ratio normalization.
        If provided, ratio values will be normalized to the average of these frames.
    enable_tracking : bool
        Whether to enable full cell tracking (default: True). If False, only cells
        present in both first and last frames will be analyzed, which is faster.
    analyze_all_frames : bool
        Whether to analyze all frames for ratio measurements (default: True).
        If True, ratios from all frames will be calculated even when tracking is disabled.
    downsample_factor : int
        Factor by which to downsample images for faster processing (default: 1, no downsampling).
        For example, 2 will reduce image dimensions by half, significantly speeding up processing.
    fast_mode_method : str
        Segmentation method to use in fast mode ('otsu', 'li', 'local')
        Simpler methods are faster but may be less accurate.
    use_cached_segmentation : bool
        Whether to reuse the segmentation from first frame as a starting point for other frames
        when processing without tracking. Can speed up processing but may reduce accuracy.
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
    
    # Store original dimensions for upsampling later if needed
    original_dims = (height, width)
    
    # Validate channel indices
    if ref_channel >= num_channels or meas_channel >= num_channels:
        raise ValueError(f"Channel indices out of range (max: {num_channels-1})")
    
    # Initialize storage for cell tracking
    cell_tracks = {}   # Will store tracking information for each cell
    all_frames_data = []  # Will store measurements for each cell in each frame
    
    # Step 2: Segment the first frame using the improved method
    print("\nSegmenting first frame...")
    
    # Get first frame reference channel
    first_ref_img = tif_data[0, ref_channel, :, :]
    
    # Apply downsampling if requested
    if downsample_factor > 1:
        print(f"Downsampling images by factor of {downsample_factor}...")
        first_ref_img_ds = downsample_image(first_ref_img, downsample_factor)
        height_ds, width_ds = first_ref_img_ds.shape
        print(f"Downsampled dimensions: {height_ds}x{width_ds}")
    else:
        first_ref_img_ds = first_ref_img
    
    # Normalize image for processing
    first_ref_img_norm = (first_ref_img_ds - np.min(first_ref_img_ds)) / (np.max(first_ref_img_ds) - np.min(first_ref_img_ds))
    
    # Apply contrast enhancement
    enhanced = enhance_contrast(first_ref_img_norm, method=enhance_method)
    
    # Apply denoising
    denoised = denoise_image(enhanced, method=denoise_method, sigma=sigma)
    
    # Detect cell regions
    binary = detect_cell_regions(denoised, method=threshold_method, 
                                min_size=min_cell_size // (downsample_factor**2), 
                                max_holes_size=max_holes_size // (downsample_factor**2))
    
    # Find cell markers
    markers = find_cell_markers(denoised, binary, method=marker_method, 
                               min_distance=min_distance // downsample_factor)
    
    # Segment cells
    labels = segment_cells(denoised, binary, markers, method=segmentation_method)
    
    # Refine segmentation
    first_frame_labels_ds = refine_segmentation(
        labels, denoised, 
        min_size=min_cell_size // (downsample_factor**2), 
        max_size=(min_cell_size*10) // (downsample_factor**2),
        split_touching=split_touching, 
        border_clearing=border_clearing
    )
    
    # If we downsampled, upsample the result back to original size
    if downsample_factor > 1:
        first_frame_labels = upsample_mask(first_frame_labels_ds, original_dims)
    else:
        first_frame_labels = first_frame_labels_ds
    
    # Extract valid labels (cell IDs)
    valid_labels = np.unique(first_frame_labels)
    valid_labels = valid_labels[valid_labels > 0]  # Remove background (0)
    
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
    
    # Step 4: Process frames based on tracking mode
    if enable_tracking:
        print("\nProcessing all frames and tracking cells...")
        
        for frame in tqdm(range(num_frames), desc="Processing frames"):
            # Get reference and measurement channels for this frame
            ref_img = tif_data[frame, ref_channel]
            meas_img = tif_data[frame, meas_channel]
            
            # For first frame, we already have the labels
            if frame == 0:
                frame_labels = first_frame_labels
            else:
                # For subsequent frames, segment again using the improved method
                # Normalize image
                ref_img_norm = (ref_img - np.min(ref_img)) / (np.max(ref_img) - np.min(ref_img))
                
                # Apply the same segmentation pipeline as for the first frame
                enhanced = enhance_contrast(ref_img_norm, method=enhance_method)
                denoised = denoise_image(enhanced, method=denoise_method, sigma=sigma)
                binary = detect_cell_regions(denoised, method=threshold_method, 
                                            min_size=min_cell_size, max_holes_size=max_holes_size)
                markers = find_cell_markers(denoised, binary, method=marker_method, min_distance=min_distance)
                labels = segment_cells(denoised, binary, markers, method=segmentation_method)
                frame_labels = refine_segmentation(
                    labels, denoised, min_size=min_cell_size, max_size=min_cell_size*10,
                    split_touching=split_touching, border_clearing=border_clearing
                )
                
                # Match cells with previous frame using KD-Tree for nearest-neighbor search
                current_labels = np.unique(frame_labels)
                current_labels = current_labels[current_labels > 0]  # Remove background
                
                if len(current_labels) > 0:
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
                    
                    # Use a nearest-neighbor approach
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
                            
                            # Check if a valid match was found
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
    else:
        # Fast mode: Process frames in different ways based on analyze_all_frames parameter
        if analyze_all_frames:
            # Process all frames individually (no tracking), but extract ratio measurements from each
            print("\nFast mode: Analyzing all frames individually (no tracking)...")
            
            # Initialize frame labels for storing segmentation of each frame
            frame_labels_all = np.zeros((num_frames, height, width), dtype=np.int32)
            frame_labels_all[0] = first_frame_labels
            
            # List to store cell IDs for each frame
            frame_cell_ids = [[] for _ in range(num_frames)]
            frame_cell_ids[0] = list(np.unique(first_frame_labels)[1:])  # Skip background
            
            # Process each frame
            for frame in tqdm(range(num_frames), desc="Processing frames"):
                # Get reference and measurement channels for this frame
                ref_img = tif_data[frame, ref_channel]
                meas_img = tif_data[frame, meas_channel]
                
                # For first frame, we already have the labels
                if frame == 0:
                    # Use existing first frame labels
                    pass
                else:
                    # Segment each frame independently
                    # Normalize image
                    ref_img_norm = (ref_img - np.min(ref_img)) / (np.max(ref_img) - np.min(ref_img))
                    
                    # Apply the same segmentation pipeline as for the first frame
                    enhanced = enhance_contrast(ref_img_norm, method=enhance_method)
                    denoised = denoise_image(enhanced, method=denoise_method, sigma=sigma)
                    binary = detect_cell_regions(denoised, method=threshold_method, 
                                                min_size=min_cell_size, max_holes_size=max_holes_size)
                    markers = find_cell_markers(denoised, binary, method=marker_method, min_distance=min_distance)
                    labels = segment_cells(denoised, binary, markers, method=segmentation_method)
                    
                    # Refine segmentation for this frame
                    curr_frame_labels = refine_segmentation(
                        labels, denoised, min_size=min_cell_size, max_size=min_cell_size*10,
                        split_touching=split_touching, border_clearing=border_clearing
                    )
                    
                    # Create unique cell IDs for this frame (frame_number * 100000 + cell_label)
                    unique_ids = np.unique(curr_frame_labels)[1:]  # Skip background
                    for label in unique_ids:
                        new_id = label + frame * 100000  # Create frame-specific unique ID
                        frame_labels_all[frame][curr_frame_labels == label] = new_id
                        frame_cell_ids[frame].append(new_id)
                
                # Measure properties for all cells in this frame
                for cell_id in frame_cell_ids[frame]:
                    cell_mask = (frame_labels_all[frame] == cell_id)
                    
                    if np.any(cell_mask):
                        # Measure intensity values
                        ref_intensity = np.mean(ref_img[cell_mask])
                        meas_intensity = np.mean(meas_img[cell_mask])
                        
                        # Calculate ratio
                        ratio = meas_intensity / ref_intensity if ref_intensity > 0 else 0
                        
                        # Create cell tracks entry for visualization
                        if cell_id not in cell_tracks:
                            # Get centroid and area
                            props = measure.regionprops(cell_mask.astype(int))
                            if props:
                                centroid = props[0].centroid
                                area = props[0].area
                                
                                cell_tracks[cell_id] = {
                                    'frames': [frame],
                                    'centroids': [centroid],
                                    'areas': [area],
                                    'in_view': True
                                }
                        
                        # Store measurements
                        all_frames_data.append({
                            'time_point': frame,
                            'cell_id': cell_id,
                            'reference_intensity': ref_intensity,
                            'measurement_intensity': meas_intensity,
                            'ratio': ratio,
                            'area': np.sum(cell_mask)
                        })
            
            # Update master_labels
            master_labels = frame_labels_all
            
        else:
            # Fast mode: Only process first and last frames, skip tracking
            print("\nFast mode: Processing only first and last frames...")
            
            # We already processed the first frame
            if num_frames > 1:
                # Process the last frame
                last_frame = num_frames - 1
                
                # Get reference and measurement channels for last frame
                ref_img = tif_data[last_frame, ref_channel]
                meas_img = tif_data[last_frame, meas_channel]
                
                # Normalize image
                ref_img_norm = (ref_img - np.min(ref_img)) / (np.max(ref_img) - np.min(ref_img))
                
                # Apply the same segmentation pipeline as for the first frame
                enhanced = enhance_contrast(ref_img_norm, method=enhance_method)
                denoised = denoise_image(enhanced, method=denoise_method, sigma=sigma)
                binary = detect_cell_regions(denoised, method=threshold_method, 
                                            min_size=min_cell_size, max_holes_size=max_holes_size)
                markers = find_cell_markers(denoised, binary, method=marker_method, min_distance=min_distance)
                labels = segment_cells(denoised, binary, markers, method=segmentation_method)
                last_frame_labels = refine_segmentation(
                    labels, denoised, min_size=min_cell_size, max_size=min_cell_size*10,
                    split_touching=split_touching, border_clearing=border_clearing
                )
                
                # Get properties of cells in last frame
                last_frame_props = measure.regionprops(last_frame_labels)
                
                # Match cells from first frame to last frame using centroids
                first_frame_props = measure.regionprops(first_frame_labels)
                first_centroids = np.array([prop.centroid for prop in first_frame_props])
                last_centroids = np.array([prop.centroid for prop in last_frame_props])
                
                if len(first_centroids) > 0 and len(last_centroids) > 0:
                    # Use KD-Tree for efficient matching
                    tree = cKDTree(last_centroids)
                    
                    # Map from first frame labels to last frame labels for matched cells
                    matched_cells = {}
                    
                    # For each cell in first frame, find the closest cell in last frame
                    for i, prop in enumerate(first_frame_props):
                        first_label = prop.label
                        distances, indices = tree.query(first_centroids[i], k=1, 
                                                      distance_upper_bound=max_tracking_distance*2)  # Use larger distance
                        
                        # Check if a valid match was found
                        if isinstance(distances, np.ndarray):
                            if len(distances) > 0 and indices[0] < len(last_frame_props):
                                matched_cells[first_label] = last_frame_props[indices[0]].label
                        else:  # Single value case
                            if indices < len(last_frame_props):
                                matched_cells[first_label] = last_frame_props[indices].label
                    
                    # Store matched cells in master_labels
                    master_labels[0] = first_frame_labels
                    master_labels[last_frame] = np.zeros_like(last_frame_labels)
                    
                    for first_label, last_label in matched_cells.items():
                        # Map the labels to keep IDs consistent
                        master_labels[last_frame][last_frame_labels == last_label] = first_label
                        
                        # Initialize cell tracking data
                        first_prop = first_frame_props[first_label-1] if first_label-1 < len(first_frame_props) else None
                        last_prop = last_frame_props[last_label-1] if last_label-1 < len(last_frame_props) else None
                        
                        if first_prop and last_prop:
                            cell_tracks[first_label] = {
                                'frames': [0, last_frame],
                                'centroids': [first_prop.centroid, last_prop.centroid],
                                'areas': [first_prop.area, last_prop.area],
                                'in_view': True
                            }
                    
                    # Collect measurements for first and last frames
                    for frame in [0, last_frame]:
                        # Get reference and measurement channels
                        ref_img = tif_data[frame, ref_channel]
                        meas_img = tif_data[frame, meas_channel]
                        
                        # Measure properties of matched cells
                        for cell_id in matched_cells.keys():
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
                
                # Save the last frame segmentation image
                plt.figure(figsize=(12, 10))
                plt.imshow(ref_img, cmap='gray')
                
                # Create a semi-transparent overlay
                masked_labels = np.ma.masked_where(master_labels[last_frame] == 0, master_labels[last_frame])
                matched_ids = list(matched_cells.keys())
                cmap = plt.cm.get_cmap('nipy_spectral', len(matched_ids) + 1)
                plt.imshow(masked_labels, cmap=cmap, alpha=0.5)
                
                # Add cell ID labels for matched cells
                for cell_id in matched_ids:
                    mask = master_labels[last_frame] == cell_id
                    if np.any(mask):
                        y, x = ndimage.center_of_mass(mask)
                        plt.text(x, y, str(cell_id), color='white', 
                               fontsize=8, ha='center', va='center',
                               bbox=dict(facecolor='black', alpha=0.5, pad=1))
                
                plt.title('Last Frame Segmentation')
                plt.savefig(os.path.join(output_dir, f"{file_name}_last_frame_segmentation.png"))
                plt.close()
    
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
    
    # Step 7: Filter cells based on tracking mode
    if enable_tracking:
        # When tracking is enabled, filter cells that leave field of view
        in_view_cells = [cell_id for cell_id, track in cell_tracks.items() if track['in_view']]
        
        # Filter to only keep cells visible through all frames
        full_track_cells = [cell_id for cell_id, track in cell_tracks.items() 
                         if len(track['frames']) == num_frames]
        
        print(f"\nTracking statistics:")
        print(f"  Total cells detected: {len(cell_tracks)}")
        print(f"  Cells that stayed in view: {len(in_view_cells)}")
        print(f"  Cells tracked through all frames: {len(full_track_cells)}")
    else:
        # When tracking is disabled (fast mode), only cells matched between first and last frames are considered
        full_track_cells = list(cell_tracks.keys())
        in_view_cells = full_track_cells
        
        print(f"\nFast mode statistics:")
        print(f"  Total cells detected in first frame: {len(np.unique(first_frame_labels)) - 1}")  # Subtract background
        print(f"  Total cells detected in last frame: {len(np.unique(master_labels[num_frames-1])) - 1}")
        print(f"  Cells matched between first and last frames: {len(full_track_cells)}")
    
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


def enhance_contrast(image, percentiles=(1, 99), method='adaptive'):
    """
    Enhance contrast of the image using different methods.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image
    percentiles : tuple
        Percentiles for contrast stretching (lower, upper)
    method : str
        Method for contrast enhancement:
        - 'rescale': Simple percentile-based rescaling
        - 'adaptive': Adaptive histogram equalization
        - 'hist_eq': Histogram equalization
        
    Returns:
    --------
    enhanced : numpy.ndarray
        Contrast-enhanced image
    """
    # Ensure float image
    img = image.astype(float)
    
    if method == 'rescale':
        # Simple contrast stretching
        p_low, p_high = np.percentile(img, percentiles)
        img = exposure.rescale_intensity(img, in_range=(p_low, p_high))
    elif method == 'adaptive':
        # Normalize to [0, 1] for adaptive histogram equalization
        min_val, max_val = img.min(), img.max()
        if max_val > min_val:
            img = (img - min_val) / (max_val - min_val)
        # Adaptive histogram equalization
        img = exposure.equalize_adapthist(img, clip_limit=0.03)
    elif method == 'hist_eq':
        # Normalize to [0, 1] for histogram equalization
        min_val, max_val = img.min(), img.max()
        if max_val > min_val:
            img = (img - min_val) / (max_val - min_val)
        # Histogram equalization
        img = exposure.equalize_hist(img)
    
    return img


def denoise_image(image, method='tv', sigma=2.0):
    """
    Apply denoising to the image.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image
    method : str
        Denoising method: 'gaussian', 'median', 'bilateral', or 'tv'
    sigma : float
        Sigma parameter for Gaussian filter
        
    Returns:
    --------
    denoised : numpy.ndarray
        Denoised image
    """
    # Ensure float image in [0, 1]
    img = image.copy()
    if img.dtype != np.float64 and img.dtype != np.float32:
        if img.max() > 1.0:
            img = img.astype(float) / 255.0
    
    if method == 'gaussian':
        return filters.gaussian(img, sigma=sigma)
    elif method == 'median':
        return filters.median(img)
    elif method == 'bilateral':
        # Bilateral filter preserves edges better
        from skimage.restoration import denoise_bilateral
        return denoise_bilateral(img, sigma_spatial=sigma, sigma_color=0.1)
    elif method == 'tv':
        # Total variation filter good for preserving edges
        from skimage.restoration import denoise_tv_chambolle
        return denoise_tv_chambolle(img, weight=0.1)
    else:
        return img


def detect_cell_regions(image, method='li', min_size=100, max_holes_size=50, threshold_abs=None):
    """
    Detect foreground cell regions in the image.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image
    method : str
        Thresholding method:
        - 'otsu': Otsu's method
        - 'multiotsu': Multi-Otsu thresholding
        - 'local': Local thresholding
        - 'li': Li's minimum cross entropy
    min_size : int
        Minimum size of objects to keep
    max_holes_size : int
        Maximum size of holes to fill
        
    Returns:
    --------
    binary : numpy.ndarray
        Binary mask of detected foreground regions
    """
    # Choose thresholding method
    if method == 'otsu':
        thresh = filters.threshold_otsu(image)
        binary = image > thresh
    elif method == 'multiotsu':
        thresholds = filters.threshold_multiotsu(image, classes=3)
        # Use the higher threshold to separate bright cells
        binary = image > thresholds[1]
    elif method == 'local':
        # Use sauvola instead of simple local for better adaptation to variations
        thresh = filters.threshold_sauvola(image, window_size=51)
        binary = image > thresh
    elif method == 'li':
        thresh = filters.threshold_li(image)
        binary = image > thresh
    elif method == 'absolute':
        if threshold_abs is None:
            threshold_abs = 0.5
        binary = image > threshold_abs
    else:
        # Default to Otsu
        thresh = filters.threshold_otsu(image)
        binary = image > thresh
    
    # Clean up binary image
    binary = morphology.remove_small_holes(binary, area_threshold=max_holes_size)
    binary = morphology.remove_small_objects(binary, min_size=min_size)
    
    # Apply morphological operations to refine the mask
    binary = morphology.binary_opening(binary, morphology.disk(2))
    binary = morphology.binary_closing(binary, morphology.disk(3))
    
    return binary


def find_cell_markers(image, binary_mask, method='distance', min_distance=15, 
                      h_maxima_threshold=0.05, footprint_size=3):
    """
    Find markers for each cell in the image.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image
    binary_mask : numpy.ndarray
        Binary mask of cell regions
    method : str
        Method for finding markers:
        - 'distance': Distance transform + peak detection
        - 'log': Laplacian of Gaussian blob detection
        - 'dog': Difference of Gaussian
        - 'h_maxima': H-maxima transform
    min_distance : int
        Minimum distance between cell markers
    h_maxima_threshold : float
        Threshold for h-maxima transform
    footprint_size : int
        Size of footprint for local maxima detection
        
    Returns:
    --------
    markers : numpy.ndarray
        Labeled array with markers for each cell
    """
    if method == 'distance':
        # Distance transform approach
        distance = ndimage.distance_transform_edt(binary_mask)
        
        # Find local maxima (peaks)
        footprint = morphology.disk(footprint_size)
        peak_idx = feature.peak_local_max(
            distance, 
            min_distance=min_distance,
            footprint=footprint,
            labels=binary_mask,
            exclude_border=False
        )
        
        # Create markers
        markers = np.zeros_like(distance, dtype=int)
        markers[tuple(peak_idx.T)] = np.arange(1, len(peak_idx) + 1)
    
    elif method == 'log':
        # Laplacian of Gaussian blob detection
        # Invert the image if cells are bright (make them dark blobs)
        inv_image = 1.0 - image if np.mean(image[binary_mask]) > np.mean(image) else image.copy()
        
        # Apply LoG filter
        blobs = feature.blob_log(
            inv_image, 
            min_sigma=3, 
            max_sigma=15, 
            num_sigma=10, 
            threshold=0.05
        )
        
        # Create markers from blobs
        markers = np.zeros_like(image, dtype=int)
        if len(blobs) > 0:
            y, x, _ = blobs.T
            markers[np.round(y).astype(int), np.round(x).astype(int)] = np.arange(1, len(blobs) + 1)
    
    elif method == 'dog':
        # Difference of Gaussian blob detection
        # Invert the image if cells are bright (make them dark blobs)
        inv_image = 1.0 - image if np.mean(image[binary_mask]) > np.mean(image) else image.copy()
        
        # Apply DoG filter
        blobs = feature.blob_dog(
            inv_image, 
            min_sigma=3, 
            max_sigma=15, 
            threshold=0.05
        )
        
        # Create markers from blobs
        markers = np.zeros_like(image, dtype=int)
        if len(blobs) > 0:
            y, x, _ = blobs.T
            markers[np.round(y).astype(int), np.round(x).astype(int)] = np.arange(1, len(blobs) + 1)
    
    elif method == 'h_maxima':
        # H-maxima transform
        # Use binary mask to limit computation
        masked_image = image.copy()
        masked_image[~binary_mask] = 0
        
        # Compute h-maxima
        h_max = morphology.h_maxima(masked_image, h_maxima_threshold)
        
        # Label connected regions in h-maxima
        markers, _ = ndimage.label(h_max)
    
    else:
        # Default to distance transform
        distance = ndimage.distance_transform_edt(binary_mask)
        
        # Find local maxima (peaks)
        footprint = morphology.disk(footprint_size)
        peak_idx = feature.peak_local_max(
            distance, 
            min_distance=min_distance,
            footprint=footprint,
            labels=binary_mask
        )
        
        # Create markers
        markers = np.zeros_like(distance, dtype=int)
        markers[tuple(peak_idx.T)] = np.arange(1, len(peak_idx) + 1)
    
    return markers


def segment_cells(image, binary_mask, markers, method='watershed'):
    """
    Segment individual cells based on markers.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image
    binary_mask : numpy.ndarray
        Binary mask of cell regions
    markers : numpy.ndarray
        Array with markers for each cell
    method : str
        Segmentation method:
        - 'watershed': Watershed segmentation
        - 'random_walker': Random walker segmentation
        
    Returns:
    --------
    labels : numpy.ndarray
        Labeled image with segmented cells
    """
    if method == 'watershed':
        # Prepare gradient for watershed
        gradient = filters.sobel(image)
        
        # Apply watershed
        labels = segmentation.watershed(gradient, markers, mask=binary_mask)
    
    elif method == 'random_walker':
        # Prepare markers for random walker
        # Background is 1, foreground starts from 2
        rw_markers = np.zeros_like(markers, dtype=int)
        rw_markers[markers > 0] = markers[markers > 0] + 1
        rw_markers[~binary_mask] = 1
        
        # Apply random walker
        try:
            labels = segmentation.random_walker(image, rw_markers)
            
            # Subtract 1 to make background 0
            labels = labels - 1
            labels[labels < 0] = 0
        except Exception as e:
            print(f"Random walker failed: {e}. Falling back to watershed.")
            # Fallback to watershed if random_walker fails
            gradient = filters.sobel(image)
            labels = segmentation.watershed(gradient, markers, mask=binary_mask)
    
    else:
        # Default to watershed
        gradient = filters.sobel(image)
        labels = segmentation.watershed(gradient, markers, mask=binary_mask)
    
    return labels


def refine_segmentation(labels, image, min_size=100, max_size=1000, 
                        split_touching=True, border_clearing=False):
    """
    Refine segmentation results by removing small objects, splitting touching cells, etc.
    
    Parameters:
    -----------
    labels : numpy.ndarray
        Labeled image with segmented cells
    image : numpy.ndarray
        Original input image
    min_size : int
        Minimum size of objects to keep
    max_size : int
        Maximum size of objects (larger objects will be split)
    split_touching : bool
        Whether to attempt splitting of touching cells
    border_clearing : bool
        Whether to remove objects touching the border
        
    Returns:
    --------
    refined_labels : numpy.ndarray
        Refined labeled image
    """
    refined_labels = labels.copy()
    
    # Remove small objects
    refined_labels = morphology.remove_small_objects(refined_labels, min_size=min_size)
    
    if border_clearing:
        # Clear border objects
        refined_labels = segmentation.clear_border(refined_labels)
    
    if split_touching:
        # For each label, check if it's too large and try to split it
        props = measure.regionprops(refined_labels)
        
        for prop in props:
            if prop.area > max_size:
                # Get the object mask
                mask = refined_labels == prop.label
                
                # Create a distance transform of the mask
                distance = ndimage.distance_transform_edt(mask)
                
                # Apply watershed to split the large object
                local_max = feature.peak_local_max(
                    distance, 
                    min_distance=15,
                    labels=mask,
                    exclude_border=False
                )
                
                # If we found multiple peaks, split the object
                if len(local_max) > 1:
                    # Create markers for watershed
                    local_markers = np.zeros_like(distance, dtype=int)
                    local_markers[tuple(local_max.T)] = np.arange(1, len(local_max) + 1)
                    
                    # Apply watershed to split the object
                    local_gradient = filters.sobel(image * mask)
                    local_labels = segmentation.watershed(local_gradient, local_markers, mask=mask)
                    
                    # Update the refined labels
                    # First, remove the original label
                    refined_labels[mask] = 0
                    
                    # Then, add the new labels with appropriate offsets
                    max_label = np.max(refined_labels)
                    for i in range(1, np.max(local_labels) + 1):
                        refined_labels[local_labels == i] = max_label + i
    
    # Relabel to ensure consecutive labels
    refined_labels, _ = ndimage.label(refined_labels > 0)
    
    return refined_labels


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
    cmap = plt.cm.get_cmap('tab20', num_colors) if num_colors > 0 else plt.cm.get_cmap('tab20')
    
    # Create a mapping of cell IDs to colors
    color_map = {}
    for i, cell_id in enumerate(full_track_cells):
        color_map[cell_id] = i % (max(num_colors, 1))
    
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
    
    # Default parameters (can be customized)
    ref_channel = 1
    meas_channel = 0
    min_cell_size = 100
    max_tracking_distance = 20
    enhance_method = 'adaptive'
    denoise_method = 'tv'
    sigma = 2.0
    threshold_method = 'li'
    marker_method = 'distance'
    min_distance = 15
    segmentation_method = 'watershed'
    split_touching = True
    border_clearing = False
    bleach_correction = True
    enable_tracking = True
    analyze_all_frames = True
    
    # Ask if user wants to enable full tracking or use fast mode
    try:
        tracking_input = input("Enable full cell tracking? (y/n, default: n): ").lower()
        enable_tracking = tracking_input == 'y'
        
        if not enable_tracking:
            analyze_input = input("Analyze all frames for ratio measurements? (y/n, default: y): ").lower()
            analyze_all_frames = analyze_input != 'n'
    except:
        enable_tracking = False
        analyze_all_frames = True
    
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
    print(f"  Segmentation method: {segmentation_method} with {threshold_method} thresholding")
    print(f"  Image enhancement: {enhance_method} contrast, {denoise_method} denoising")
    print(f"  Cell detection: {marker_method} method, min distance {min_distance}")
    print(f"  Cell splitting: {'Enabled' if split_touching else 'Disabled'}")
    print(f"  Border clearing: {'Enabled' if border_clearing else 'Disabled'}")
    print(f"  Full cell tracking: {'Enabled' if enable_tracking else 'Disabled (fast mode)'}")
    if not enable_tracking:
        print(f"  Analyze all frames: {'Yes' if analyze_all_frames else 'No, only first and last'}")
    print(f"  Bleach correction: {'Enabled' if bleach_correction else 'Disabled'}")
    print(f"  Baseline normalization: {'First ' + str(baseline_frames) + ' frames' if baseline_frames else 'Disabled'}")
    print()
    
    # Run the analysis
    segment_track_analyze_tif(
        tif_path, 
        ref_channel=ref_channel, 
        meas_channel=meas_channel,
        min_cell_size=min_cell_size, 
        max_tracking_distance=max_tracking_distance,
        enhance_method=enhance_method,
        denoise_method=denoise_method,
        sigma=sigma,
        threshold_method=threshold_method,
        marker_method=marker_method,
        min_distance=min_distance,
        segmentation_method=segmentation_method,
        split_touching=split_touching,
        border_clearing=border_clearing,
        bleach_correction=bleach_correction,
        baseline_frames=baseline_frames,
        enable_tracking=enable_tracking,
        analyze_all_frames=analyze_all_frames
    )