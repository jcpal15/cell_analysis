import numpy as np
import tifffile
from skimage import filters, measure, exposure
from scipy import ndimage
import matplotlib.pyplot as plt
import os
from pathlib import Path
import datetime

def segment_first_frame(tif_path, ref_channel=1, min_cell_size=100, 
                        percentile_low=0.5, percentile_high=99.5,
                        threshold_method='otsu', use_adaptive=False, 
                        use_watershed=True, watershed_min_distance=15,
                        adaptive_block_size=35):
    """
    Simple script to segment the first frame of a TIF file.
    
    Parameters:
    -----------
    tif_path : str
        Path to the TIF file to analyze
    ref_channel : int
        Index of the reference channel (default: 1)
    min_cell_size : int
        Minimum size of cells to consider (default: 100)
    percentile_low, percentile_high : float
        Percentiles for contrast enhancement (default: 0.5, 99.5)
    threshold_method : str or float
        Method for thresholding ('otsu', 'li') or a manual value (default: 'otsu')
    use_adaptive : bool
        Whether to use adaptive thresholding (default: False)
    use_watershed : bool
        Whether to use watershed segmentation (default: True)
    watershed_min_distance : int
        Minimum distance between peaks for watershed (default: 15)
    adaptive_block_size : int
        Block size for adaptive thresholding (default: 35)
    """
    # Extract filename for outputs
    file_name = Path(tif_path).stem
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory
    output_dir = f'segmentation_output_{current_time}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Loading TIF file: {tif_path}")
    
    # Step 1: Load the TIF file
    try:
        tif_data = tifffile.imread(tif_path)
        print(f"TIF data shape: {tif_data.shape}")
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    
    # Step 2: Extract first frame and reference channel
    # Handle different dimension formats
    first_frame = None
    
    # Try to determine dimensionality and extract first frame's reference channel
    try:
        if len(tif_data.shape) == 5:  # TZCYX format
            first_frame = tif_data[0, 0, ref_channel, :, :]
            print(f"Extracted from 5D format (TZCYX)")
        elif len(tif_data.shape) == 4:
            if tif_data.shape[2] <= 5:  # Likely TZYX with small Z dimension
                first_frame = tif_data[0, 0, :, :]
                print(f"Extracted from 4D format (TZYX)")
            else:  # Likely TCYX with C > Z
                first_frame = tif_data[0, ref_channel, :, :]
                print(f"Extracted from 4D format (TCYX)")
        elif len(tif_data.shape) == 3:
            if tif_data.shape[2] <= 5:  # Likely YXC format
                first_frame = tif_data[:, :, ref_channel]
                print(f"Extracted from 3D format (YXC)")
            else:  # Likely TYX format
                first_frame = tif_data[0, :, :]
                print(f"Extracted from 3D format (TYX)")
        else:
            first_frame = tif_data
            print(f"Using as-is - shape: {first_frame.shape}")
            
        print(f"First frame shape: {first_frame.shape}")
        
        # Save the raw image
        plt.figure(figsize=(10, 8))
        plt.imshow(first_frame, cmap='gray')
        plt.colorbar(label='Intensity')
        plt.title('Raw Reference Channel - First Frame')
        plt.savefig(os.path.join(output_dir, f"{file_name}_raw.png"))
        plt.close()
        
    except Exception as e:
        print(f"Error extracting first frame: {e}")
        
        # Try alternate formats if the standard ones fail
        try:
            # Just take the first slice of whatever we have
            if len(tif_data.shape) > 2:
                first_frame = tif_data[0]
                if len(first_frame.shape) > 2 and first_frame.shape[-1] > 1:
                    # This might be channel data
                    ref_idx = min(ref_channel, first_frame.shape[-1] - 1)
                    first_frame = first_frame[..., ref_idx]
            else:
                first_frame = tif_data
                
            print(f"Using alternate extraction, shape: {first_frame.shape}")
            
            # Save the raw image
            plt.figure(figsize=(10, 8))
            plt.imshow(first_frame, cmap='gray')
            plt.colorbar(label='Intensity')
            plt.title('Raw Reference Channel - First Frame (Alternative Format)')
            plt.savefig(os.path.join(output_dir, f"{file_name}_raw_alt.png"))
            plt.close()
            
        except Exception as e2:
            print(f"Error with alternate extraction: {e2}")
            return
    
    # Check if we have a valid image
    if first_frame is None or first_frame.size == 0:
        print("Failed to extract a valid frame from the TIF file.")
        return
    
    # Step 3: Preprocess the image
    print("Preprocessing image...")
    
    # Apply Gaussian smoothing
    smoothed = filters.gaussian(first_frame, sigma=2.0)
    print(f"Smoothed image min/max: {np.min(smoothed)}/{np.max(smoothed)}")
    
    # Enhance contrast
    p_low, p_high = np.percentile(smoothed, (percentile_low, percentile_high))
    print(f"Contrast stretch range: {p_low} to {p_high}")
    
    reference_enhanced = exposure.rescale_intensity(smoothed, in_range=(p_low, p_high))
    print(f"Enhanced image min/max: {np.min(reference_enhanced)}/{np.max(reference_enhanced)}")
    
    # Save the enhanced image
    plt.figure(figsize=(10, 8))
    plt.imshow(reference_enhanced, cmap='gray')
    plt.colorbar(label='Enhanced Intensity')
    plt.title('Enhanced Reference Channel')
    plt.savefig(os.path.join(output_dir, f"{file_name}_enhanced.png"))
    plt.close()
    
    # Step 4: Apply threshold
    print("Applying threshold...")
    
    # Different thresholding approaches
    if use_adaptive:
        # Adaptive thresholding for uneven illumination
        from skimage.filters import threshold_local
        print(f"Using adaptive thresholding with block size {adaptive_block_size}")
        local_thresh = threshold_local(reference_enhanced, block_size=adaptive_block_size)
        binary = reference_enhanced > local_thresh
    else:
        # Global thresholding
        if threshold_method == 'otsu':
            thresh = filters.threshold_otsu(reference_enhanced)
        elif threshold_method == 'li':
            thresh = filters.threshold_li(reference_enhanced)
        elif threshold_method == 'triangle':
            thresh = filters.threshold_triangle(reference_enhanced)
        elif threshold_method == 'yen':
            thresh = filters.threshold_yen(reference_enhanced)
        else:
            try:
                # Try to interpret as a float value
                thresh = float(threshold_method)
            except (ValueError, TypeError):
                print("Invalid threshold method, using Otsu's method")
                thresh = filters.threshold_otsu(reference_enhanced)
        
        print(f"Threshold value: {thresh}")
        binary = reference_enhanced > thresh
    
    # Save the initial binary mask
    plt.figure(figsize=(10, 8))
    plt.imshow(binary, cmap='gray')
    plt.title('Initial Binary Mask (Before Cleanup)')
    plt.savefig(os.path.join(output_dir, f"{file_name}_initial_binary.png"))
    plt.close()
    
    # Step 5: Clean up the binary mask
    print("Cleaning up binary mask...")
    
    # Apply morphological operations
    binary = ndimage.binary_fill_holes(binary)
    binary = ndimage.binary_closing(binary, iterations=3)
    binary = ndimage.binary_opening(binary, iterations=2)
    binary = ndimage.binary_fill_holes(binary)
    binary = ndimage.binary_opening(binary, structure=np.ones((3,3)))
    
    # Save the cleaned binary mask
    plt.figure(figsize=(10, 8))
    plt.imshow(binary, cmap='gray')
    plt.title('Cleaned Binary Mask')
    plt.savefig(os.path.join(output_dir, f"{file_name}_cleaned_binary.png"))
    plt.close()
    
    # Step 6: Label regions
    print("Labeling and filtering regions...")
    
    # Watershed segmentation to separate close cells
    if use_watershed:
        from scipy import ndimage as ndi
        from skimage.feature import peak_local_max
        
        print(f"Applying watershed segmentation with min_distance={watershed_min_distance}")
        
        # Compute the distance map
        distance = ndi.distance_transform_edt(binary)
        
        # Find local maxima (cell centers)
        try:
            # Try with indices=False first (newer versions)
            local_max_coords = peak_local_max(distance, min_distance=watershed_min_distance,
                                          labels=binary)
            # Convert coordinates to mask
            local_max = np.zeros_like(binary, dtype=bool)
            for coord in local_max_coords:
                local_max[tuple(coord)] = True
        except TypeError:
            # Older versions might return coordinates directly
            print("Using alternative peak_local_max method")
            local_max_coords = peak_local_max(distance, min_distance=watershed_min_distance, 
                                          labels=binary)
            # Convert coordinates to mask
            local_max = np.zeros_like(binary, dtype=bool)
            for coord in local_max_coords:
                local_max[coord[0], coord[1]] = True
        
        # Mark each local maximum with a unique label
        markers = measure.label(local_max)
        
        # Apply watershed to find cell boundaries
        from skimage import segmentation
        labels = segmentation.watershed(-distance, markers, mask=binary)
        
        print(f"Watershed segmentation found {len(np.unique(labels)) - 1} regions")
    else:
        # Standard connected component labeling
        labels = measure.label(binary)
        print(f"Connected component labeling found {len(np.unique(labels)) - 1} regions")
    
    # Get region properties
    props = measure.regionprops(labels, first_frame)
    
    print(f"Found {len(props)} initial regions")
    
    # Filter regions based on criteria
    valid_labels = []
    for prop in props:
        # Simple size-based filtering
        if prop.area >= min_cell_size and prop.area <= min_cell_size * 10:
            valid_labels.append(prop.label)
    
    print(f"Found {len(valid_labels)} valid cells after size filtering")
    
    # Create filtered labels for visualization
    filtered_labels = np.zeros_like(labels)
    for label in valid_labels:
        filtered_labels[labels == label] = label
    
    # Step 7: Visualize the segmentation results
    
    # Create segmentation mask
    plt.figure(figsize=(12, 10))
    
    # Use a discrete colormap with enough colors
    from matplotlib.colors import ListedColormap
    num_labels = len(valid_labels) + 1  # +1 for background
    cmap = plt.cm.get_cmap('nipy_spectral', num_labels)
    
    plt.imshow(filtered_labels, cmap=cmap)
    plt.colorbar(label='Cell ID')
    plt.title('Cell Segmentation Mask')
    plt.savefig(os.path.join(output_dir, f"{file_name}_segmentation_mask.png"))
    plt.close()
    
    # Create overlay of the segmentation on the original image
    plt.figure(figsize=(12, 10))
    plt.imshow(first_frame, cmap='gray')
    
    # Create a semi-transparent overlay
    masked_labels = np.ma.masked_where(filtered_labels == 0, filtered_labels)
    plt.imshow(masked_labels, cmap=cmap, alpha=0.5)
    
    # Add cell ID labels to each cell
    for label in valid_labels:
        # Get centroid of this cell
        props = measure.regionprops(np.asarray(filtered_labels == label, dtype=int))
        if props:
            y, x = props[0].centroid
            plt.text(x, y, str(label), color='white', 
                   fontsize=8, ha='center', va='center',
                   bbox=dict(facecolor='black', alpha=0.5, pad=1))
    
    plt.title('Segmentation Overlay on Original Image')
    plt.savefig(os.path.join(output_dir, f"{file_name}_segmentation_overlay.png"))
    plt.close()
    
    print(f"Segmentation complete. Results saved to {output_dir}")
    return filtered_labels, first_frame

if __name__ == "__main__":
    print("Simple TIF First Frame Segmentation")
    print("===================================")
    
    # Get input from user
    tif_path = input("Enter path to TIF file: ")
    
    # Use automatic settings as requested
    ref_channel = 1  # Use the 1st channel for reference
    threshold_method = 'otsu'  # Use Otsu thresholding
    min_cell_size = 100  # Set minimum cell size to 100
    use_adaptive = True  # Use adaptive thresholding
    adaptive_block_size = 35  # Block size = 35
    use_watershed = True  # Use watershed segmentation
    watershed_min_distance = 10  # Set minimum distance = 10
    percentile_low = 0.1  # Lower percentile = 0.1
    percentile_high = 99.9  # Higher percentile = 99.9
    
    print("\nUsing automatic settings:")
    print(f"  Reference channel: {ref_channel}")
    print(f"  Threshold method: {threshold_method}")
    print(f"  Minimum cell size: {min_cell_size} pixels")
    print(f"  Adaptive thresholding: Enabled (block size {adaptive_block_size})")
    print(f"  Watershed segmentation: Enabled (min distance {watershed_min_distance})")
    print(f"  Contrast enhancement: {percentile_low} to {percentile_high} percentiles")
    print()
    
    # Run the segmentation
    segment_first_frame(tif_path, ref_channel, min_cell_size, 
                      percentile_low, percentile_high,
                      threshold_method=threshold_method,
                      use_adaptive=use_adaptive, 
                      use_watershed=use_watershed,
                      watershed_min_distance=watershed_min_distance,
                      adaptive_block_size=adaptive_block_size)