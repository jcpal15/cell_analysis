import numpy as np
import tifffile
from skimage import filters, measure, exposure, io, segmentation, registration
from scipy import ndimage, optimize, stats
import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm import tqdm
import datetime
from pathlib import Path
import glob
from scipy.spatial.distance import cdist
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
import matplotlib.cm as cm

def analyze_tif_with_tracking(tif_path, ref_channel=1, meas_channel=0, 
                            percentile_low=0.5, percentile_high=99.5,
                            crop_size=50, min_cell_size=100, threshold_method='otsu',
                            max_tracking_distance=20, bleach_correction=True,
                            bleach_model='exponential', poly_order=2, 
                            normalization_point=0, require_full_track=True,
                            baseline_frames=None):
    """
    Analyze TIF file with tracking of cells across time points.
    
    Parameters:
    -----------
    tif_path : str
        Path to the TIF file to analyze
    ref_channel : int
        Index of the reference channel
    meas_channel : int
        Index of the measurement channel
    percentile_low, percentile_high : float
        Percentiles for contrast enhancement
    crop_size : int
        Size of the crop for individual cell images
    min_cell_size : int
        Minimum size of cells to consider
    threshold_method : str or float
        Method for thresholding ('otsu', 'li') or a manual value
    max_tracking_distance : float
        Maximum distance between centroids to consider as the same cell
    bleach_correction : bool
        Whether to apply bleaching correction
    bleach_model : str
        Type of bleaching model ('exponential', 'linear', 'polynomial')
    poly_order : int
        Order of polynomial for polynomial bleaching model
    normalization_point : int
        Time point to normalize to for bleaching correction
    require_full_track : bool
        If True, only keep cells that are tracked from start to end
    baseline_frames : int or None
        Number of frames considered baseline (for separated analysis)
        
    Returns:
    --------
    result_dict : dict
        Dictionary with analysis results
    """
    # Extract filename without extension for output naming
    file_name = Path(tif_path).stem
    
    # Create timestamp for unique output directories
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directories
    output_dir = f'tracking_results/{file_name}_tracking_{current_time}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cell_crops_dir = f'{output_dir}/cell_crops'
    if not os.path.exists(cell_crops_dir):
        os.makedirs(cell_crops_dir)
    
    # Load the TIF file
    print(f"Loading TIF file: {tif_path}")
    tif_data = tifffile.imread(tif_path)
    
    # Determine data dimensions
    if tif_data.ndim == 5:  # TZCYX format (multi-channel)
        num_frames = tif_data.shape[0]
        num_channels = tif_data.shape[2]
        height, width = tif_data.shape[3], tif_data.shape[4]
        # Extract z=0 for 2D analysis
        tif_data = tif_data[:, 0, :, :, :]
    elif tif_data.ndim == 4:  # TZYX format (single channel)
        num_frames = tif_data.shape[0]
        num_channels = 1
        height, width = tif_data.shape[2], tif_data.shape[3]
        # Add channel dimension
        tif_data = tif_data[:, 0, :, :, np.newaxis]
    else:
        print(f"Unexpected data dimensions: {tif_data.shape}")
        return None
    
    print(f"Data dimensions: {tif_data.shape}")
    print(f"Number of frames: {num_frames}")
    print(f"Number of channels: {num_channels}")
    
    # Initialize structures to store tracking results
    cells_over_time = {}  # Will store cell data across time points
    cell_tracks = {}      # Will link cell IDs across frames
    all_cells_data = []   # Will store all cell measurements
    
    # Process first frame to identify cells
    print("\nSegmenting cells in first frame...")
    
    # Get reference channel for first frame
    if num_channels > 1:
        first_ref_img = tif_data[0, ref_channel, :, :]
        first_meas_img = tif_data[0, meas_channel, :, :]
    else:
        first_ref_img = tif_data[0, 0, :, :]
        first_meas_img = first_ref_img.copy()
    
    # Save the raw reference image for debugging
    plt.figure(figsize=(10, 8))
    plt.imshow(first_ref_img)
    plt.colorbar(label='Intensity')
    plt.title('Raw Reference Channel - First Frame')
    plt.savefig(os.path.join(output_dir, f"{file_name}_raw_reference_first_frame.png"))
    plt.close()
    
    print(f"Raw image shape: {first_ref_img.shape}")
    print(f"Raw image min: {np.min(first_ref_img)}, max: {np.max(first_ref_img)}")
    
    # Preprocess the image
    smoothed = filters.gaussian(first_ref_img, sigma=1.5)  # Reduced sigma for less smoothing
    
    # Adjust percentiles for better contrast
    p_low, p_high = np.percentile(smoothed, (percentile_low, percentile_high))
    print(f"Contrast adjustment range: {p_low} to {p_high}")
    
    reference_enhanced = exposure.rescale_intensity(smoothed, in_range=(p_low, p_high))
    
    # Save the enhanced image for debugging
    plt.figure(figsize=(10, 8))
    plt.imshow(reference_enhanced)
    plt.colorbar(label='Enhanced Intensity')
    plt.title('Enhanced Reference Channel - First Frame')
    plt.savefig(os.path.join(output_dir, f"{file_name}_enhanced_reference_first_frame.png"))
    plt.close()
    
    # Try multiple thresholding methods and let user select
    try_all_thresholds = True  # Set to True to try multiple methods
    
    if try_all_thresholds:
        # Try multiple thresholding methods
        thresh_otsu = filters.threshold_otsu(reference_enhanced)
        thresh_li = filters.threshold_li(reference_enhanced)
        thresh_mean = filters.threshold_mean(reference_enhanced)
        thresh_triangle = filters.threshold_triangle(reference_enhanced)
        thresh_yen = filters.threshold_yen(reference_enhanced)
        
        # Also try a fixed percentile-based threshold
        thresh_percentile = np.percentile(reference_enhanced, 75)  # 75th percentile
        
        print(f"Threshold values:")
        print(f"  Otsu: {thresh_otsu}")
        print(f"  Li: {thresh_li}")
        print(f"  Mean: {thresh_mean}")
        print(f"  Triangle: {thresh_triangle}")
        print(f"  Yen: {thresh_yen}")
        print(f"  75th Percentile: {thresh_percentile}")
        
        # Create figure with all thresholding methods
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Otsu
        axes[0].imshow(reference_enhanced > thresh_otsu)
        axes[0].set_title(f'Otsu Threshold: {thresh_otsu:.4f}')
        
        # Li
        axes[1].imshow(reference_enhanced > thresh_li)
        axes[1].set_title(f'Li Threshold: {thresh_li:.4f}')
        
        # Mean
        axes[2].imshow(reference_enhanced > thresh_mean)
        axes[2].set_title(f'Mean Threshold: {thresh_mean:.4f}')
        
        # Triangle
        axes[3].imshow(reference_enhanced > thresh_triangle)
        axes[3].set_title(f'Triangle Threshold: {thresh_triangle:.4f}')
        
        # Yen
        axes[4].imshow(reference_enhanced > thresh_yen)
        axes[4].set_title(f'Yen Threshold: {thresh_yen:.4f}')
        
        # Percentile
        axes[5].imshow(reference_enhanced > thresh_percentile)
        axes[5].set_title(f'75th Percentile: {thresh_percentile:.4f}')
        
        for ax in axes:
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{file_name}_threshold_comparison.png"))
        plt.close()
        
        # Ask user which threshold method they prefer
        print("\nComparison of thresholding methods saved.")
        print("Check the output directory for 'threshold_comparison.png'")
        
        # Set the actual threshold to use based on input method
        if threshold_method == 'otsu':
            thresh = thresh_otsu
        elif threshold_method == 'li':
            thresh = thresh_li
        elif threshold_method == 'mean':
            thresh = thresh_mean
        elif threshold_method == 'triangle':
            thresh = thresh_triangle
        elif threshold_method == 'yen':
            thresh = thresh_yen
        elif threshold_method == 'percentile':
            thresh = thresh_percentile
        else:
            try:
                # Try to interpret as a float value
                thresh = float(threshold_method)
            except (ValueError, TypeError):
                print("Invalid threshold method, using Otsu's method")
                thresh = thresh_otsu
        
        # Allow user to override the threshold
        override = input(f"Current threshold: {thresh}. Enter a new threshold value, or press Enter to continue: ")
        if override.strip():
            try:
                thresh = float(override)
                print(f"Using manual threshold: {thresh}")
            except ValueError:
                print("Invalid input. Keeping original threshold.")
    else:
        # Just use the provided method
        if threshold_method == 'otsu':
            thresh = filters.threshold_otsu(reference_enhanced)
        elif threshold_method == 'li':
            thresh = filters.threshold_li(reference_enhanced)
        else:
            try:
                # Try to interpret as a float value
                thresh = float(threshold_method)
            except (ValueError, TypeError):
                print("Invalid threshold method, using Otsu's method")
                thresh = filters.threshold_otsu(reference_enhanced)
    
    print(f"Final threshold value: {thresh}")
    
    # Create binary mask
    binary = reference_enhanced > thresh
    
    # Save the initial binary mask for debugging
    plt.figure(figsize=(10, 8))
    plt.imshow(binary)
    plt.title('Initial Binary Mask (Before Cleanup)')
    plt.savefig(os.path.join(output_dir, f"{file_name}_initial_binary_mask.png"))
    plt.close()
    
    # Clean up binary mask
    binary = ndimage.binary_fill_holes(binary)
    binary = ndimage.binary_closing(binary, iterations=3)
    binary = ndimage.binary_opening(binary, iterations=2)
    binary = ndimage.binary_fill_holes(binary)
    binary = ndimage.binary_opening(binary, structure=np.ones((3,3)))
    
    # Save the cleaned binary mask for debugging
    plt.figure(figsize=(10, 8))
    plt.imshow(binary)
    plt.title('Cleaned Binary Mask')
    plt.savefig(os.path.join(output_dir, f"{file_name}_cleaned_binary_mask.png"))
    plt.close()
    
    # Label regions
    labels = measure.label(binary)
    
    print(f"Number of regions before filtering: {len(np.unique(labels)) - 1}")  # -1 to exclude background
    
    # Get region properties
    props = measure.regionprops(labels, first_ref_img)
    
    # Display histogram of region areas to help with min_cell_size adjustment
    areas = [prop.area for prop in props]
    if areas:
        plt.figure(figsize=(10, 6))
        plt.hist(areas, bins=30)
        plt.axvline(x=min_cell_size, color='r', linestyle='--', 
                  label=f'Min Size: {min_cell_size}')
        plt.axvline(x=min_cell_size*10, color='g', linestyle='--', 
                  label=f'Max Size: {min_cell_size*10}')
        plt.xlabel('Region Area (pixels)')
        plt.ylabel('Count')
        plt.legend()
        plt.title('Histogram of Region Areas')
        plt.savefig(os.path.join(output_dir, f"{file_name}_region_area_histogram.png"))
        plt.close()
    
    # Allow user to adjust min_cell_size if needed
    print(f"Current min_cell_size: {min_cell_size}")
    new_min_size = input("Enter new minimum cell size, or press Enter to keep current value: ")
    if new_min_size.strip():
        try:
            min_cell_size = int(new_min_size)
            print(f"Using new minimum cell size: {min_cell_size}")
        except ValueError:
            print("Invalid input. Keeping original minimum cell size.")
    
    # Count how many regions would be filtered by each criterion
    edge_filtered = 0
    size_filtered = 0
    elongation_filtered = 0
    solidity_filtered = 0
    proximity_filtered = 0
    
    # Filter regions based on criteria but track which criteria are filtering cells
    valid_labels = []
    
    # Filter with more lenient parameters initially
    edge_filtering = True       # Set to False to keep cells at the edges
    size_filtering = True       # Set to False to ignore size constraints
    elongation_threshold = 0.9  # More lenient than original 0.8
    solidity_threshold = 0.7    # More lenient than original 0.8
    proximity_filtering = True  # Set to False to keep cells that are close together
    
    for prop in props:
        # Track which criteria would filter this region
        is_edge_cell = (prop.bbox[0] == 0 or prop.bbox[1] == 0 or 
                       prop.bbox[2] == binary.shape[0] or prop.bbox[3] == binary.shape[1])
        
        is_size_issue = (prop.area < min_cell_size or prop.area > min_cell_size * 10)
        
        is_elongated = (prop.eccentricity > elongation_threshold)
        
        is_irregular = (prop.solidity < solidity_threshold)
        
        # Check proximity to other cells
        centroid = prop.centroid
        is_too_close = False
        if proximity_filtering:
            for other_prop in props:
                if other_prop.label != prop.label:
                    dist = np.sqrt((centroid[0] - other_prop.centroid[0])**2 + 
                                 (centroid[1] - other_prop.centroid[1])**2)
                    if dist < crop_size/2:  # If centers are closer than half crop size
                        is_too_close = True
                        break
        
        # Count regions filtered by each criterion
        if is_edge_cell:
            edge_filtered += 1
            if edge_filtering:
                continue
                
        if is_size_issue:
            size_filtered += 1
            if size_filtering:
                continue
                
        if is_elongated:
            elongation_filtered += 1
            if elongation_threshold < 1.0:
                continue
                
        if is_irregular:
            solidity_filtered += 1
            if solidity_threshold > 0:
                continue
                
        if is_too_close:
            proximity_filtered += 1
            if proximity_filtering:
                continue
        
        valid_labels.append(prop.label)
    
    # Print filtering statistics
    print(f"\nCell filtering statistics:")
    print(f"  Total regions detected: {len(props)}")
    print(f"  Regions touching edges: {edge_filtered}")
    print(f"  Regions with size issues: {size_filtered}")
    print(f"  Highly elongated regions: {elongation_filtered}")
    print(f"  Regions with low solidity: {solidity_filtered}")
    print(f"  Regions too close to others: {proximity_filtered}")
    print(f"  Valid regions after filtering: {len(valid_labels)}")
    
    # Try again with more lenient filtering if no valid cells found
    if len(valid_labels) == 0:
        print("\nWARNING: No valid cells detected with current filtering criteria.")
        print("Trying again with more lenient criteria...")
        
        valid_labels = []
        
        # Disable all filtering
        for prop in props:
            if prop.area >= min_cell_size / 2:  # Only keep minimal size filtering
                valid_labels.append(prop.label)
        
        print(f"Found {len(valid_labels)} cells with lenient filtering criteria.")
    
    print(f"Found {len(valid_labels)} valid cells in first frame after filtering")
    
    # Create filtered labels for first frame
    first_frame_labels = np.zeros_like(labels)
    for label in valid_labels:
        first_frame_labels[labels == label] = label
    
    # Get properties of filtered regions
    first_frame_props = measure.regionprops(first_frame_labels, first_ref_img)
    
    # Create dictionary to store data for first frame cells
    first_frame_cells = {}
    for i, prop in enumerate(first_frame_props):
        if prop.label == 0:  # Skip background
            continue
        
        # Assign new cell ID
        cell_id = prop.label
        
        # Store cell properties
        first_frame_cells[cell_id] = {
            'centroid': prop.centroid,
            'area': prop.area,
            'eccentricity': prop.eccentricity,
            'solidity': prop.solidity,
            'frame': 0,
            'label': cell_id
        }
        
        # Initialize tracking data
        cell_tracks[cell_id] = {
            'frames': [0],
            'centroids': [prop.centroid],
            'labels': [cell_id]
        }
    
    # Save the segmentation mask for the first frame
    plt.figure(figsize=(10, 10))
    # Use a better colormap for segmentation visualization
    cmap = plt.cm.get_cmap('nipy_spectral', np.max(first_frame_labels) + 1)
    plt.imshow(first_frame_labels, cmap=cmap, interpolation='nearest')
    plt.colorbar(label='Cell ID')
    plt.title(f'Cell Segmentation Mask - First Frame')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{file_name}_first_frame_segmentation_{current_time}.png"))
    plt.close()
    
    # Also save an overlay of the segmentation on the original image for better context
    plt.figure(figsize=(10, 10))
    plt.imshow(first_ref_img, cmap='gray')
    
    # Create a semi-transparent overlay of the segmentation
    labeled_overlay = np.zeros((height, width, 4))  # RGBA
    unique_labels = np.unique(first_frame_labels)
    unique_labels = unique_labels[unique_labels > 0]  # Remove background
    
    # Generate random colors for each label
    import matplotlib.colors as mcolors
    import random
    colors = list(mcolors.TABLEAU_COLORS.values())
    random.shuffle(colors)
    
    # Create the overlay
    for i, label in enumerate(unique_labels):
        mask = first_frame_labels == label
        color_idx = i % len(colors)
        color = mcolors.to_rgba(colors[color_idx])
        # Make it semi-transparent
        color = list(color[:3]) + [0.5]  # RGB + alpha
        labeled_overlay[mask] = color
    
    plt.imshow(labeled_overlay, interpolation='nearest')
    
    # Add cell ID labels as text
    for label in unique_labels:
        # Get centroid of this cell
        props = measure.regionprops(np.asarray(first_frame_labels == label, dtype=int))
        if props:
            y, x = props[0].centroid
            plt.text(x, y, str(label), color='white', 
                   fontsize=8, ha='center', va='center',
                   bbox=dict(facecolor='black', alpha=0.5, pad=1))
    
    plt.title(f'Cell Segmentation Overlay - First Frame')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{file_name}_first_frame_overlay_{current_time}.png"))
    plt.close()
    
    # Process all frames, starting with frame 0 again for consistent handling
    print("\nProcessing all frames and tracking cells...")
    
    # Copy first frame labels to master_labels to start
    # The cells in master_labels will have consistent IDs across all frames
    master_labels = np.zeros((num_frames, height, width), dtype=np.int32)
    master_labels[0] = first_frame_labels
    
    # Process each frame
    for frame in tqdm(range(num_frames), desc="Processing frames"):
        # Get reference and measurement images for this frame
        if num_channels > 1:
            ref_img = tif_data[frame, ref_channel, :, :]
            meas_img = tif_data[frame, meas_channel, :, :]
        else:
            ref_img = tif_data[frame, 0, :, :]
            meas_img = ref_img.copy()
        
        # For first frame, use the already processed labels
        if frame == 0:
            frame_labels = first_frame_labels
        else:
            # For subsequent frames, segment the cells again
            # This is to account for cell movement, deformation, etc.
            
            # Preprocess the image
            smoothed = filters.gaussian(ref_img, sigma=2.0)
            p_low, p_high = np.percentile(smoothed, (percentile_low, percentile_high))
            reference_enhanced = exposure.rescale_intensity(smoothed, in_range=(p_low, p_high))
            
            # Apply the same threshold as first frame for consistency
            binary = reference_enhanced > thresh
            
            # Clean up binary mask
            binary = ndimage.binary_fill_holes(binary)
            binary = ndimage.binary_closing(binary, iterations=3)
            binary = ndimage.binary_opening(binary, iterations=2)
            binary = ndimage.binary_fill_holes(binary)
            binary = ndimage.binary_opening(binary, structure=np.ones((3,3)))
            
            # Label regions
            labels = measure.label(binary)
            
            # Get region properties
            props = measure.regionprops(labels, ref_img)
            
            # Filter regions based on same criteria as first frame
            valid_labels = []
            
            for prop in props:
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
                
                valid_labels.append(prop.label)
            
            # Create filtered labels for this frame
            frame_labels = np.zeros_like(labels)
            for label in valid_labels:
                frame_labels[labels == label] = label
            
            # Get properties of filtered regions
            frame_props = measure.regionprops(frame_labels, ref_img)
            
            # Match cells with previous frame
            # Get centroids and additional properties of cells in current frame
            current_centroids = np.array([prop.centroid for prop in frame_props if prop.label > 0])
            current_labels = np.array([prop.label for prop in frame_props if prop.label > 0])
            current_areas = np.array([prop.area for prop in frame_props if prop.label > 0])
            
            # Get the most recent centroids and properties for each tracked cell
            prev_frame = frame - 1
            prev_track_centroids = []
            prev_track_ids = []
            prev_track_areas = []
            
            for cell_id, track in cell_tracks.items():
                if prev_frame in track['frames']:
                    idx = track['frames'].index(prev_frame)
                    prev_track_centroids.append(track['centroids'][idx])
                    prev_track_ids.append(cell_id)
                    
                    # Get the area from the previous frame's properties
                    prev_area = None
                    for prop in measure.regionprops(master_labels[prev_frame]):
                        if prop.label == cell_id:
                            prev_area = prop.area
                            break
                    prev_track_areas.append(prev_area if prev_area is not None else 0)
            
            # Calculate distances between cell centroids
            if len(prev_track_centroids) > 0 and len(current_centroids) > 0:
                prev_track_centroids = np.array(prev_track_centroids)
                distances = cdist(prev_track_centroids, current_centroids)
                
                # Adjust distances based on area similarity to improve tracking
                # This helps with cells that change appearance but maintain position
                if len(prev_track_areas) > 0 and len(current_areas) > 0:
                    prev_track_areas = np.array(prev_track_areas)
                    for i in range(len(prev_track_centroids)):
                        for j in range(len(current_centroids)):
                            # Compute area ratio (always <= 1.0)
                            if prev_track_areas[i] > 0 and current_areas[j] > 0:
                                area_ratio = min(prev_track_areas[i], current_areas[j]) / max(prev_track_areas[i], current_areas[j])
                                # Give a bonus to cells with similar areas (reduce distance)
                                area_bonus = (1.0 - area_ratio) * max_tracking_distance * 0.5
                                distances[i, j] += area_bonus
                
                # Match cells based on minimum distance
                for prev_idx in range(len(prev_track_ids)):
                    prev_id = prev_track_ids[prev_idx]
                    
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
                        cell_tracks[prev_id]['labels'].append(curr_label)
                        
                        # Mark this cell as matched so we don't match it again
                        distances[:, min_dist_idx] = float('inf')
                        
                # Add debug output to show tracking success rate
                tracked_count = sum(1 for cell_id in prev_track_ids if frame in cell_tracks[cell_id]['frames'])
                print(f"Frame {frame}: Successfully tracked {tracked_count}/{len(prev_track_ids)} cells ({tracked_count/len(prev_track_ids)*100:.1f}%)")
            
            # For new cells that appear in this frame (not matched with any previous), 
            # assign a new unique ID
            for i, curr_label in enumerate(current_labels):
                # Check if this cell was matched (exists in master_labels for this frame)
                if not np.any(master_labels[frame] == curr_label):
                    # This is a new cell, assign a new ID
                    # Use a formula to ensure unique IDs across frames
                    new_id = curr_label + frame * 10000  # This ensures unique IDs
                    
                    # Update master labels
                    master_labels[frame][frame_labels == curr_label] = new_id
                    
                    # Create new tracking data
                    cell_tracks[new_id] = {
                        'frames': [frame],
                        'centroids': [current_centroids[i]],
                        'labels': [curr_label]
                    }
        
        # Measure cell properties for this frame using master_labels
        frame_master_props = measure.regionprops(master_labels[frame], ref_img)
        frame_master_meas_props = measure.regionprops(master_labels[frame], meas_img)
        
        # Store measurements for each cell
        for i, prop in enumerate(frame_master_props):
            if prop.label == 0:  # Skip background
                continue
            
            cell_id = prop.label
            
            # Calculate reference and measurement intensities
            ref_intensity = prop.mean_intensity
            # Find the matching measurement region
            meas_intensity = 0
            for meas_prop in frame_master_meas_props:
                if meas_prop.label == cell_id:
                    meas_intensity = meas_prop.mean_intensity
                    break
            
            # Calculate ratio
            ratio = meas_intensity / ref_intensity if ref_intensity > 0 else 0
            
            # Store measurements
            cell_data = {
                'time_point': frame,
                'cell_id': cell_id,
                'area': prop.area,
                'centroid_y': prop.centroid[0],
                'centroid_x': prop.centroid[1],
                'reference_intensity': ref_intensity,
                'measurement_intensity': meas_intensity,
                'ratio': ratio
            }
            
            all_cells_data.append(cell_data)
            
            # Also organize by cell ID for time course analysis
            if cell_id not in cells_over_time:
                cells_over_time[cell_id] = {
                    'time_points': [],
                    'reference_intensities': [],
                    'measurement_intensities': [],
                    'ratios': []
                }
            
            cells_over_time[cell_id]['time_points'].append(frame)
            cells_over_time[cell_id]['reference_intensities'].append(ref_intensity)
            cells_over_time[cell_id]['measurement_intensities'].append(meas_intensity)
            cells_over_time[cell_id]['ratios'].append(ratio)
    
    # Convert all cell data to DataFrame
    df = pd.DataFrame(all_cells_data)
    
    # Filter cells based on tracking requirements
    if require_full_track:
        print("\nFiltering cells to keep only those tracked from start to end...")
        
        # Identify cells that are present in all frames
        full_track_cells = []
        
        for cell_id, track in cell_tracks.items():
            if len(track['frames']) == num_frames:
                # This cell is present in all frames
                full_track_cells.append(cell_id)
        
        print(f"Found {len(full_track_cells)} cells tracked through all {num_frames} frames")
        
        # Filter the DataFrame to keep only these cells
        df = df[df['cell_id'].isin(full_track_cells)]
        
        # Also filter cells_over_time
        cells_over_time = {k: v for k, v in cells_over_time.items() if k in full_track_cells}
    
    # Apply bleaching correction if requested
    if bleach_correction:
        print("\nApplying photobleaching correction to reference channel...")
        df_corrected, model_params, correction_factors = apply_bleaching_correction(
            df, bleach_model, poly_order, normalization_point
        )
        
        # Replace the original DataFrame with the corrected one
        df = df_corrected
    
    # Save the complete dataset with unique filename
    csv_filename = f"{file_name}_tracking_data_{current_time}.csv"
    print(f"\nSaving data to CSV as {csv_filename}...")
    df.to_csv(os.path.join(output_dir, csv_filename), index=False)
    
    # Create visualization of cell tracks over time
    print("\nCreating visualization of cell tracks...")
    create_tracking_visualization(master_labels, cell_tracks, full_track_cells if require_full_track else None, 
                              output_dir, file_name, current_time)
    
    # Create summary plots
    print("\nCreating summary plots...")
    
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
    
    # Draw vertical line at baseline/treatment transition if specified
    if baseline_frames is not None and baseline_frames < num_frames:
        plt.axvline(x=baseline_frames-0.5, color='red', linestyle='--', 
                  label='Baseline -> Treatment')
        plt.legend()
    
    plt.title(f'{ratio_title} - {file_name}')
    plt.xlabel('Time Point')
    plt.ylabel('Mean Ratio (Measurement/Reference)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, ratio_filename))
    plt.close()
    
    # 3. If we have baseline and treatment frames specified, show comparison
    if baseline_frames is not None and baseline_frames < num_frames:
        # Create box plots comparing baseline and treatment ratios
        baseline_data = []
        treatment_data = []
        
        # Collect data for each cell
        for cell_id in cells_over_time:
            if cell_id in df['cell_id'].unique():
                cell_df = df[df['cell_id'] == cell_id]
                
                # Get average ratio in baseline and treatment periods
                if bleach_correction:
                    ratio_column = 'ratio_corrected'
                else:
                    ratio_column = 'ratio'
                
                baseline_mean = cell_df[cell_df['time_point'] < baseline_frames][ratio_column].mean()
                treatment_mean = cell_df[cell_df['time_point'] >= baseline_frames][ratio_column].mean()
                
                baseline_data.append(baseline_mean)
                treatment_data.append(treatment_mean)
        
        # Create box plot
        plt.figure(figsize=(8, 10))
        box_data = [baseline_data, treatment_data]
        bp = plt.boxplot(box_data, notch=True, patch_artist=True, 
                       labels=['Baseline', 'Treatment'])
        
        # Add individual data points
        for i, data in enumerate([baseline_data, treatment_data]):
            # Add scatter points
            x = np.random.normal(i+1, 0.1, size=len(data))
            plt.scatter(x, data, alpha=0.4, color='black')
        
        # Calculate and add statistics
        baseline_mean = np.mean(baseline_data)
        treatment_mean = np.mean(treatment_data)
        
        # Paired t-test
        from scipy import stats
        t_stat, p_value = stats.ttest_rel(baseline_data, treatment_data)
        
        plt.title(f'Ratio Comparison: Baseline vs Treatment\n' +
                f'Baseline Mean: {baseline_mean:.3f}, Treatment Mean: {treatment_mean:.3f}\n' +
                f'Paired t-test: p = {p_value:.4f}')
        plt.ylabel('Ratio (Measurement/Reference)')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{file_name}_baseline_treatment_comparison_{current_time}.png"))
        plt.close()
        
        # Also create a paired dot plot
        plt.figure(figsize=(10, 10))
        
        # Draw lines connecting pairs
        for i in range(len(baseline_data)):
            plt.plot([1, 2], [baseline_data[i], treatment_data[i]], 'k-', alpha=0.3)
        
        # Plot the dots
        plt.scatter([1] * len(baseline_data), baseline_data, s=80, alpha=0.7, c='blue', label='Baseline')
        plt.scatter([2] * len(treatment_data), treatment_data, s=80, alpha=0.7, c='red', label='Treatment')
        
        # Draw means
        plt.scatter([1], [baseline_mean], s=150, c='darkblue', marker='_', linewidth=3)
        plt.scatter([2], [treatment_mean], s=150, c='darkred', marker='_', linewidth=3)
        
        plt.title(f'Paired Comparison: Baseline vs Treatment\n' +
                f'Baseline Mean: {baseline_mean:.3f}, Treatment Mean: {treatment_mean:.3f}\n' +
                f'Paired t-test: p = {p_value:.4f}')
        plt.ylabel('Ratio (Measurement/Reference)')
        plt.xticks([1, 2], ['Baseline', 'Treatment'])
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{file_name}_paired_dot_plot_{current_time}.png"))
        plt.close()
    
    # Create individual cell traces
    print("\nCreating individual cell traces...")
    create_individual_cell_traces(df, cells_over_time, bleach_correction, 
                               output_dir, file_name, current_time, 
                               baseline_frames)
    
    # Report analysis completion
    print(f"\nAnalysis complete. Results saved to: {output_dir}")
    
    # Return results dictionary
    return {
        'file_name': file_name,
        'output_dir': output_dir,
        'num_frames': num_frames,
        'num_channels': num_channels,
        'total_cells_detected': len(cell_tracks),
        'cells_tracked_full': len(full_track_cells) if require_full_track else None,
        'bleach_corrected': bleach_correction,
        'mean_ratios': mean_ratios,
        'std_ratios': std_ratios,
        'baseline_frames': baseline_frames,
        'treatment_frames': num_frames - baseline_frames if baseline_frames is not None else None
    }

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

def create_tracking_visualization(master_labels, cell_tracks, full_track_cells=None, 
                              output_dir='.', file_name='output', current_time=None):
    """
    Create visualizations of cell tracking
    
    Parameters:
    -----------
    master_labels : numpy.ndarray
        3D array with labeled cells for each frame
    cell_tracks : dict
        Dictionary mapping cell IDs to track information
    full_track_cells : list or None
        List of cell IDs that have full tracks (if None, show all)
    output_dir : str
        Directory to save output
    file_name : str
        Base name for output files
    current_time : str
        Timestamp for unique filenames
    """
    if current_time is None:
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    num_frames = master_labels.shape[0]
    height, width = master_labels.shape[1], master_labels.shape[2]
    
    # Create a color map for cell tracking
    # Use a specific colormap for better visualization
    num_colors = 256
    cmap = plt.cm.get_cmap('tab20', num_colors)
    
    # Create a random mapping of cell IDs to colors
    cell_ids = list(cell_tracks.keys())
    np.random.shuffle(cell_ids)  # Shuffle to distribute colors
    
    color_map = {}
    for i, cell_id in enumerate(cell_ids):
        color_map[cell_id] = i % num_colors
    
    # Create image to visualize tracking
    tracking_image = np.zeros((height, width, 3, num_frames), dtype=np.float32)
    
    # First, add all cells to the tracking image (faded)
    for frame in range(num_frames):
        frame_labels = master_labels[frame]
        for cell_id in cell_tracks:
            # Make non-tracked cells appear faded
            mask = (frame_labels == cell_id)
            if mask.any():
                color_idx = color_map[cell_id]
                color = np.array(cmap(color_idx)[:3])
                
                if full_track_cells is not None and cell_id not in full_track_cells:
                    # Make non-tracked cells appear faded
                    color = color * 0.3
                
                # Apply color to the mask area
                for c in range(3):
                    tracking_image[:, :, c, frame][mask] = color[c]
    
    # Save a combined tracking visualization
    fig, axes = plt.subplots(1, min(5, num_frames), figsize=(15, 8))
    if min(5, num_frames) == 1:
        axes = [axes]  # Make it iterable if only one frame
    
    # Select frames to show (first, 25%, 50%, 75%, last)
    if num_frames <= 5:
        frames_to_show = range(num_frames)
    else:
        frames_to_show = [0, 
                        num_frames // 4, 
                        num_frames // 2, 
                        3 * num_frames // 4, 
                        num_frames - 1]
    
    for i, frame_idx in enumerate(frames_to_show[:len(axes)]):
        axes[i].imshow(tracking_image[:, :, :, frame_idx])
        axes[i].set_title(f'Frame {frame_idx}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{file_name}_tracking_visualization_{current_time}.png"))
    plt.close()
    
    # Also create a movie if we have many frames
    if num_frames > 5:
        from matplotlib.animation import FuncAnimation
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.axis('off')
        
        # Initialize with the first frame
        im = ax.imshow(tracking_image[:, :, :, 0])
        title = ax.set_title(f'Frame 0/{num_frames-1}')
        
        def update(frame):
            im.set_array(tracking_image[:, :, :, frame])
            title.set_text(f'Frame {frame}/{num_frames-1}')
            return [im, title]
        
        # Create animation
        anim = FuncAnimation(fig, update, frames=range(num_frames), 
                           interval=200, blit=True)
        
        # Save as mp4 using ffmpeg if available, otherwise save sampled frames
        try:
            anim.save(os.path.join(output_dir, f"{file_name}_tracking_movie_{current_time}.mp4"), 
                    writer='ffmpeg', dpi=100)
        except:
            print("Could not create movie (ffmpeg may not be available). Saving sampled frames instead.")
            # Save a series of frames instead
            num_samples = min(20, num_frames)
            sample_frames = np.linspace(0, num_frames-1, num_samples, dtype=int)
            
            for i, frame in enumerate(sample_frames):
                plt.figure(figsize=(10, 10))
                plt.imshow(tracking_image[:, :, :, frame])
                plt.title(f'Frame {frame}/{num_frames-1}')
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{file_name}_tracking_frame{frame:03d}_{current_time}.png"))
                plt.close()
        
        plt.close()
    
    return

def create_individual_cell_traces(df, cells_over_time, bleach_correction=True, 
                               output_dir='.', file_name='output', current_time=None,
                               baseline_frames=None):
    """
    Create plots showing individual cell traces over time
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with cell measurements
    cells_over_time : dict
        Dictionary mapping cell IDs to time course data
    bleach_correction : bool
        Whether bleaching correction was applied
    output_dir : str
        Directory to save output
    file_name : str
        Base name for output files
    current_time : str
        Timestamp for unique filenames
    baseline_frames : int or None
        Number of frames in baseline period
    """
    if current_time is None:
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create individual cell trace plots
    # We'll show reference and measurement intensities, and ratio
    
    # Calculate overall min/max for consistent y-axis scales
    all_ref = []
    all_meas = []
    all_ratios = []
    
    for cell_id, cell_data in cells_over_time.items():
        all_ref.extend(cell_data['reference_intensities'])
        all_meas.extend(cell_data['measurement_intensities'])
        all_ratios.extend(cell_data['ratios'])
    
    ref_min, ref_max = np.min(all_ref), np.max(all_ref)
    meas_min, meas_max = np.min(all_meas), np.max(all_meas)
    ratio_min, ratio_max = np.min(all_ratios), np.max(all_ratios)
    
    # Add some padding to the limits
    ref_pad = (ref_max - ref_min) * 0.1
    meas_pad = (meas_max - meas_min) * 0.1
    ratio_pad = (ratio_max - ratio_min) * 0.1
    
    ref_lim = (ref_min - ref_pad, ref_max + ref_pad)
    meas_lim = (meas_min - meas_pad, meas_max + meas_pad)
    ratio_lim = (ratio_min - ratio_pad, ratio_max + ratio_pad)
    
    # Get also corrected reference and ratio data if available
    if bleach_correction:
        all_ref_corr = []
        all_ratios_corr = []
        
        for cell_id in cells_over_time:
            cell_df = df[df['cell_id'] == cell_id]
            if 'reference_intensity_corrected' in cell_df.columns:
                all_ref_corr.extend(cell_df['reference_intensity_corrected'].values)
            if 'ratio_corrected' in cell_df.columns:
                all_ratios_corr.extend(cell_df['ratio_corrected'].values)
        
        if all_ref_corr:
            ref_corr_min, ref_corr_max = np.min(all_ref_corr), np.max(all_ref_corr)
            ref_corr_pad = (ref_corr_max - ref_corr_min) * 0.1
            ref_corr_lim = (ref_corr_min - ref_corr_pad, ref_corr_max + ref_corr_pad)
        else:
            ref_corr_lim = ref_lim
            
        if all_ratios_corr:
            ratio_corr_min, ratio_corr_max = np.min(all_ratios_corr), np.max(all_ratios_corr)
            ratio_corr_pad = (ratio_corr_max - ratio_corr_min) * 0.1
            ratio_corr_lim = (ratio_corr_min - ratio_corr_pad, ratio_corr_max + ratio_corr_pad)
        else:
            ratio_corr_lim = ratio_lim
    
    # Create a multi-plot figure with all cells
    max_cells_per_figure = 20
    num_cells = len(cells_over_time)
    num_figures = (num_cells + max_cells_per_figure - 1) // max_cells_per_figure
    
    cell_ids = sorted(cells_over_time.keys())
    
    for fig_num in range(num_figures):
        start_idx = fig_num * max_cells_per_figure
        end_idx = min((fig_num + 1) * max_cells_per_figure, num_cells)
        cells_to_plot = cell_ids[start_idx:end_idx]
        
        num_cols = 4  # Number of cells per row
        num_rows = (len(cells_to_plot) + num_cols - 1) // num_cols
        
        plt.figure(figsize=(16, 4 * num_rows))
        
        for i, cell_id in enumerate(cells_to_plot):
            ax = plt.subplot(num_rows, num_cols, i + 1)
            
            # Get cell data
            cell_data = cells_over_time[cell_id]
            time_points = cell_data['time_points']
            ref_intensities = cell_data['reference_intensities']
            meas_intensities = cell_data['measurement_intensities']
            ratios = cell_data['ratios']
            
            # Plot reference and measurement intensities
            ax.plot(time_points, ref_intensities, 'b-', label='Reference')
            ax.plot(time_points, meas_intensities, 'r-', label='Measurement')
            
            # If we have bleach correction, also plot corrected values
            if bleach_correction:
                cell_df = df[df['cell_id'] == cell_id]
                if 'reference_intensity_corrected' in cell_df.columns:
                    ref_corr = cell_df.sort_values('time_point')['reference_intensity_corrected'].values
                    ax.plot(time_points, ref_corr, 'g-', label='Ref (Corrected)')
            
            # Draw vertical line at baseline/treatment transition if specified
            if baseline_frames is not None and baseline_frames < max(time_points):
                ax.axvline(x=baseline_frames-0.5, color='red', linestyle='--')
            
            ax.set_title(f'Cell {cell_id}')
            ax.set_xlabel('Time Point')
            ax.set_ylabel('Intensity')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(fontsize='small')
            
            # Set consistent y-axis limits
            # ax.set_ylim(0, max(ref_max, meas_max) * 1.1)
            
            # Create a twin axis for ratio
            ax2 = ax.twinx()
            
            # Plot ratio
            ratio_label = 'Ratio'
            if bleach_correction:
                if 'ratio_corrected' in cell_df.columns:
                    ratios = cell_df.sort_values('time_point')['ratio_corrected'].values
                    ratio_label = 'Ratio (Corrected)'
            
            ax2.plot(time_points, ratios, 'g--', label=ratio_label)
            ax2.set_ylabel(ratio_label, color='g')
            ax2.tick_params(axis='y', labelcolor='g')
            
            # Set consistent y-axis limits for ratio
            if bleach_correction:
                ax2.set_ylim(ratio_corr_lim)
            else:
                ax2.set_ylim(ratio_lim)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{file_name}_individual_cells_{fig_num+1}of{num_figures}_{current_time}.png"))
        plt.close()
    
    return

def process_folder_of_tifs(folder_path, ref_channel=1, meas_channel=0, 
                         threshold_method='otsu', bleach_correction=True, 
                         max_tracking_distance=20, require_full_track=True,
                         bleach_model='exponential', baseline_frames=None):
    """
    Process all TIF files in a folder
    
    Parameters:
    -----------
    folder_path : str
        Path to folder containing TIF files
    ref_channel, meas_channel : int
        Indices of reference and measurement channels
    threshold_method : str or float
        Method for thresholding ('otsu', 'li') or a manual value
    bleach_correction : bool
        Whether to apply bleaching correction
    max_tracking_distance : float
        Maximum distance for cell tracking
    require_full_track : bool
        If True, only keep cells tracked from start to end
    bleach_model : str
        Type of bleaching model ('exponential', 'linear', 'polynomial')
    baseline_frames : int or None
        Number of frames considered as baseline
        
    Returns:
    --------
    results : list
        List of result dictionaries for each file
    """
    # Find all TIF files in the folder
    tif_files = glob.glob(os.path.join(folder_path, "*.tif")) + glob.glob(os.path.join(folder_path, "*.tiff"))
    
    if not tif_files:
        print(f"No TIF files found in {folder_path}")
        return []
    
    print(f"Found {len(tif_files)} TIF files to process:")
    for i, file in enumerate(tif_files):
        print(f"{i+1}. {os.path.basename(file)}")
    
    # Process each file
    results = []
    
    for i, tif_path in enumerate(tif_files):
        print(f"\n{'='*50}")
        print(f"Processing file {i+1}/{len(tif_files)}: {os.path.basename(tif_path)}")
        print(f"{'='*50}")
        
        # Check if user wants to override baseline frames for this file
        this_baseline_frames = baseline_frames
        if baseline_frames is None:
            try:
                input_bf = input(f"Enter number of baseline frames for {os.path.basename(tif_path)} (or press Enter to skip): ")
                if input_bf.strip():
                    this_baseline_frames = int(input_bf)
            except:
                print("Invalid input, proceeding without baseline/treatment separation")
        
        # Process the file
        result = analyze_tif_with_tracking(
            tif_path,
            ref_channel=ref_channel,
            meas_channel=meas_channel,
            threshold_method=threshold_method,
            bleach_correction=bleach_correction,
            max_tracking_distance=max_tracking_distance,
            require_full_track=require_full_track,
            bleach_model=bleach_model,
            baseline_frames=this_baseline_frames
        )
        
        if result is not None:
            results.append(result)
    
    # Create a summary report
    if results:
        # Create timestamp for the summary
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create summary directory
        summary_dir = f'tif_analysis_summary_{current_time}'
        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir)
        
        # Create summary table
        summary_data = []
        for result in results:
            summary_data.append({
                'File Name': result['file_name'],
                'Output Directory': result['output_dir'],
                'Frames': result['num_frames'],
                'Channels': result['num_channels'],
                'Total Cells': result['total_cells_detected'],
                'Tracked Cells': result['cells_tracked_full'] if result['cells_tracked_full'] is not None else 'All',
                'Bleach Correction': 'Yes' if result['bleach_corrected'] else 'No',
                'Baseline Frames': result['baseline_frames'] if result['baseline_frames'] is not None else 'N/A',
                'Treatment Frames': result['treatment_frames'] if result['treatment_frames'] is not None else 'N/A'
            })
        
        # Save summary as CSV
        summary_df = pd.DataFrame(summary_data)
        summary_csv = os.path.join(summary_dir, f"tif_analysis_summary_{current_time}.csv")
        summary_df.to_csv(summary_csv, index=False)
        
        # Also create a combined plot showing ratio changes across all files
        plt.figure(figsize=(12, 8))
        
        colors = plt.cm.tab10.colors
        for i, result in enumerate(results):
            color = colors[i % len(colors)]
            file_name = result['file_name']
            mean_ratios = result['mean_ratios']
            std_ratios = result['std_ratios']
            
            plt.errorbar(
                mean_ratios.index,
                mean_ratios,
                yerr=std_ratios,
                fmt=f'-o',
                color=color,
                label=file_name,
                capsize=4,
                markersize=6,
                linewidth=2,
                alpha=0.8
            )
        
        plt.title('Comparison of Mean Ratios Across Files', fontsize=16)
        plt.xlabel('Time Point', fontsize=14)
        plt.ylabel('Mean Ratio (Measurement/Reference)', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(title='File Name', loc='best', fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(summary_dir, f"combined_ratio_comparison_{current_time}.png"))
        plt.close()
        
        print(f"\nSummary report saved to {summary_dir}")
        print(f"Summary CSV: {summary_csv}")
    
    return results

# Main execution block
if __name__ == "__main__":
    print("\n" + "="*50)
    print("TIF Cell Tracking and Analysis")
    print("="*50)
    
    # Get input from user
    mode = input("Process a single file (s) or a folder of TIF files (f)? [s/f]: ").lower()
    
    if mode == 'f':
        # Process folder mode
        folder_path = input("Enter folder path containing TIF files: ")
        
        # Get channel settings
        ref_channel = int(input("Enter reference channel index (default 1): ") or 1)
        meas_channel = int(input("Enter measurement channel index (default 0): ") or 0)
        
        # Get threshold method
        threshold_input = input("Enter threshold method (otsu/li) or a manual value (0-1) [default: otsu]: ") or 'otsu'
        
        # Get tracking settings
        max_tracking_distance = float(input("Enter maximum tracking distance (default 20): ") or 20)
        full_track_only = input("Keep only cells tracked from start to end? [y/n, default: y]: ").lower() != 'n'
        
        # Get bleaching correction settings
        bleach_correction = input("Apply bleaching correction? [y/n, default: y]: ").lower() != 'n'
        bleach_model = 'exponential'
        if bleach_correction:
            bleach_model = input("Choose bleaching model (exponential/linear/polynomial) [default: exponential]: ") or 'exponential'
        
        # Get baseline frames
        baseline_input = input("Enter number of baseline frames (or press Enter if not applicable): ")
        baseline_frames = int(baseline_input) if baseline_input.strip() else None
        
        # Process the folder
        process_folder_of_tifs(
            folder_path,
            ref_channel=ref_channel,
            meas_channel=meas_channel,
            threshold_method=threshold_input,
            bleach_correction=bleach_correction,
            max_tracking_distance=max_tracking_distance,
            require_full_track=full_track_only,
            bleach_model=bleach_model,
            baseline_frames=baseline_frames
        )
        
    else:
        # Single file mode
        tif_path = input("Enter path to TIF file: ")
        
        # Get channel settings
        ref_channel = int(input("Enter reference channel index (default 1): ") or 1)
        meas_channel = int(input("Enter measurement channel index (default 0): ") or 0)
        
        # Get threshold method
        threshold_input = input("Enter threshold method (otsu/li) or a manual value (0-1) [default: otsu]: ") or 'otsu'
        
        # Get tracking settings
        max_tracking_distance = float(input("Enter maximum tracking distance (default 20): ") or 20)
        full_track_only = input("Keep only cells tracked from start to end? [y/n, default: y]: ").lower() != 'n'
        
        # Get bleaching correction settings
        bleach_correction = input("Apply bleaching correction? [y/n, default: y]: ").lower() != 'n'
        bleach_model = 'exponential'
        if bleach_correction:
            bleach_model = input("Choose bleaching model (exponential/linear/polynomial) [default: exponential]: ") or 'exponential'
        
        # Get baseline frames
        baseline_input = input("Enter number of baseline frames (or press Enter if not applicable): ")
        baseline_frames = int(baseline_input) if baseline_input.strip() else None
        
        # Process the file
        analyze_tif_with_tracking(
            tif_path,
            ref_channel=ref_channel,
            meas_channel=meas_channel,
            threshold_method=threshold_input,
            bleach_correction=bleach_correction,
            max_tracking_distance=max_tracking_distance,
            require_full_track=full_track_only,
            bleach_model=bleach_model,
            baseline_frames=baseline_frames
        )
    
    print("\nAnalysis complete!")