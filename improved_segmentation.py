import numpy as np
import tifffile
import matplotlib.pyplot as plt
from skimage import filters, morphology, feature, segmentation, measure, exposure, util, color
from scipy import ndimage
from pathlib import Path
import os
import pandas as pd
import argparse

def load_first_frame(tiff_path, channel=0):
    """
    Load the first frame from a TIFF file, extracting a specific channel.
    
    Parameters:
    -----------
    tiff_path : str
        Path to the TIFF file
    channel : int
        Channel index to extract
    
    Returns:
    --------
    image : numpy.ndarray
        First frame, single channel image
    """
    with tifffile.TiffFile(tiff_path) as tif:
        # Try to get metadata
        metadata = getattr(tif, 'imagej_metadata', {}) or {}
        axes = metadata.get('axes', 'TZCYX')
        
        # Read the first timepoint
        if 'T' in axes and axes.startswith('T'):
            # For TZCYX format
            if len(axes) >= 5 and 'C' in axes:
                c_idx = axes.index('C')
                image = tif.asarray(0)  # First timepoint
                
                # Get the correct channel
                if image.ndim > c_idx-1:
                    channel_idx = min(channel, image.shape[c_idx-1]-1)
                    
                    # Handle different dimensions
                    if image.ndim == 5:  # TZCYX (we've already sliced T)
                        image = image[0, channel_idx]  # Take first Z, requested channel
                    elif image.ndim == 4:  # TZYX or TCYX
                        if 'Z' in axes:
                            image = image[0, channel_idx]  # Take first Z, requested channel
                        else:
                            image = image[channel_idx]  # Take requested channel
                    elif image.ndim == 3:  # TZX or TCX
                        image = image[channel_idx]  # Take requested channel
                else:
                    # Fallback if dimensions don't match expected format
                    image = tif.asarray(0)
                    if image.ndim > 2:
                        image = image[0]  # Take first slice of whatever dimension
            else:
                # No channel dimension
                image = tif.asarray(0)  # First timepoint
                if image.ndim > 2:
                    image = image[0]  # Take first Z or whatever is the first dimension
        else:
            # No time dimension, just read first page
            image = tif.asarray()
            if image.ndim > 2:
                # Try to extract channel if it exists
                if len(axes) >= 3 and 'C' in axes:
                    c_idx = axes.index('C')
                    if image.ndim > c_idx-1:
                        channel_idx = min(channel, image.shape[c_idx-1]-1)
                        slices = [0] * image.ndim
                        slices[c_idx-1] = channel_idx
                        image = image[tuple(slices)]
                    else:
                        image = image[0]  # Fallback to first slice
                else:
                    image = image[0]  # Take first Z or channel
    
    # Ensure 2D
    if image.ndim > 2:
        image = image.reshape(image.shape[-2], image.shape[-1])
    
    return image

def enhance_contrast(image, percentiles=(1, 99), method='rescale'):
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

def denoise_image(image, method='gaussian', sigma=1.0):
    """
    Apply denoising to the image.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image
    method : str
        Denoising method: 'gaussian', 'median', or 'bilateral'
    sigma : float
        Sigma parameter for Gaussian filter
        
    Returns:
    --------
    denoised : numpy.ndarray
        Denoised image
    """
    if method == 'gaussian':
        return filters.gaussian(image, sigma=sigma)
    elif method == 'median':
        return filters.median(image)
    elif method == 'bilateral':
        return filters.gaussian(image, sigma=sigma, mode='reflect')
    else:
        return image

def detect_cell_regions(image, method='multiotsu', min_size=50, threshold_abs=None):
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
        thresh = filters.threshold_local(image, block_size=51)
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
    binary = morphology.remove_small_holes(binary, area_threshold=min_size)
    binary = morphology.remove_small_objects(binary, min_size=min_size)
    
    return binary

def find_cell_markers(image, binary_mask, method='distance', min_distance=10, 
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
        labels = segmentation.random_walker(image, rw_markers)
        
        # Subtract 1 to make background 0
        labels = labels - 1
        labels[labels < 0] = 0
    
    else:
        # Default to watershed
        gradient = filters.sobel(image)
        labels = segmentation.watershed(gradient, markers, mask=binary_mask)
    
    return labels

def refine_segmentation(labels, image, min_size=50, max_size=5000, 
                        split_touching=True, border_clearing=True):
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
                    min_distance=10,
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

def extract_features(labels, image):
    """
    Extract features from segmented cells.
    
    Parameters:
    -----------
    labels : numpy.ndarray
        Labeled image with segmented cells
    image : numpy.ndarray
        Original input image
        
    Returns:
    --------
    props : list
        List of region properties for each cell
    df : pandas.DataFrame
        DataFrame with extracted features
    """
    # Calculate properties of labeled regions
    props = measure.regionprops(labels, intensity_image=image)
    
    # Extract features
    features = []
    for prop in props:
        cell_features = {
            'label': prop.label,
            'y': prop.centroid[0],
            'x': prop.centroid[1],
            'area': prop.area,
            'perimeter': prop.perimeter,
            'major_axis_length': prop.major_axis_length,
            'minor_axis_length': prop.minor_axis_length,
            'eccentricity': prop.eccentricity,
            'solidity': prop.solidity,
            'extent': prop.extent,
            'mean_intensity': prop.mean_intensity,
            'max_intensity': prop.max_intensity,
            'min_intensity': prop.min_intensity
        }
        features.append(cell_features)
    
    # Convert to DataFrame
    df = pd.DataFrame(features)
    
    return props, df

def cluster_cells_by_intensity(df, n_clusters=4):
    """
    Cluster cells based on intensity features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with cell features
    n_clusters : int
        Number of clusters
        
    Returns:
    --------
    df : pandas.DataFrame
        DataFrame with cluster assignments
    """
    # Simple quantile-based clustering
    if 'mean_intensity' in df.columns:
        # Get mean intensity
        intensities = df['mean_intensity'].values
        
        # Create clusters based on quantiles
        quantiles = np.linspace(0, 1, n_clusters+1)[1:-1]
        thresholds = np.quantile(intensities, quantiles)
        
        # Initialize cluster column
        df['cluster'] = 0
        
        # Assign clusters
        for i, threshold in enumerate(thresholds):
            df.loc[df['mean_intensity'] > threshold, 'cluster'] = i + 1
    
    return df

def plot_segmentation(image, labels, df=None, output_path=None, figsize=(12, 12),
                     show_labels=True, show_outlines=True, cmap='viridis'):
    """
    Plot segmentation results with colored labels.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Original input image
    labels : numpy.ndarray
        Labeled segmentation mask
    df : pandas.DataFrame or None
        DataFrame with cell features (used for coloring)
    output_path : str or None
        Path to save the output figure
    figsize : tuple
        Figure size
    show_labels : bool
        Whether to show label numbers
    show_outlines : bool
        Whether to show cell outlines
    cmap : str
        Colormap name
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Display original image
    ax.imshow(image, cmap='gray')
    
    # Color segmentation based on clusters if available
    if df is not None and 'cluster' in df.columns:
        # Get cluster colors with compatibility for different matplotlib versions
        try:
            # For newer matplotlib versions (3.7+)
            cluster_colors = plt.colormaps[cmap].resampled(len(df['cluster'].unique()))
        except (AttributeError, KeyError):
            # Fallback for older matplotlib versions
            cluster_colors = plt.cm.get_cmap(cmap, len(df['cluster'].unique()))
        
        # Draw colored labels or outlines
        for _, row in df.iterrows():
            label = int(row['label'])
            y, x = row['y'], row['x']
            cluster = int(row['cluster'])
            color = cluster_colors(cluster)
            
            # Show label numbers
            if show_labels:
                ax.text(x, y, str(label), color='white', fontsize=8, 
                        ha='center', va='center', 
                        bbox=dict(facecolor=color, alpha=0.7, edgecolor='none', pad=1))
            
            # Show outlines
            if show_outlines:
                # Get cell boundary
                mask = labels == label
                boundary = morphology.dilation(mask, morphology.disk(1)) ^ mask
                
                # Draw boundary
                for coord in np.argwhere(boundary):
                    ax.plot(coord[1], coord[0], '.', color=color, markersize=1)
    else:
        # Create a color overlay of the segmentation
        if show_outlines:
            # Draw boundaries
            boundaries = segmentation.find_boundaries(labels, mode='outer')
            ax.imshow(boundaries, cmap='gray', alpha=0.7)
        
        # Show label numbers
        if show_labels:
            for region in measure.regionprops(labels):
                y, x = region.centroid
                ax.text(x, y, str(region.label), color='white', fontsize=8, 
                        ha='center', va='center', 
                        bbox=dict(facecolor='blue', alpha=0.7, edgecolor='none', pad=1))
    
    ax.set_title('Cell Segmentation')
    ax.axis('on')
    
    # Save figure if output_path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Segmentation image saved to: {output_path}")
    
    plt.close(fig)
    
    return fig

def save_segmentation_data(df, output_csv=None):
    """
    Save segmentation data to CSV file.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with segmentation data
    output_csv : str or None
        Path to save the output CSV file
    
    Returns:
    --------
    df : pandas.DataFrame
        DataFrame with segmentation data
    """
    if df.empty:
        print("No objects found in segmentation")
        return df
    
    # Save to CSV if output_csv is provided
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Segmentation data saved to: {output_csv}")
    
    return df

def process_tiff_file(tiff_path, output_dir=None, channel=0, **params):
    """
    Process a single TIFF file and segment cells.
    
    Parameters:
    -----------
    tiff_path : str
        Path to the TIFF file
    output_dir : str or None
        Directory to save output files
    channel : int
        Channel index to process
    params : dict
        Additional parameters for processing
        
    Returns:
    --------
    results : dict
        Dictionary with processing results
    """
    # Create output directory if needed
    if output_dir is None:
        output_dir = os.path.dirname(tiff_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Get file name without extension
    file_stem = Path(tiff_path).stem
    
    # Load first frame
    print(f"Loading first frame from {tiff_path}, channel {channel}")
    image = load_first_frame(tiff_path, channel=channel)
    
    if image is None or image.size == 0:
        print(f"Failed to load image from {tiff_path}")
        return None
    
    # Extract parameters with defaults
    enhance_method = params.get('enhance_method', 'rescale')
    denoise_method = params.get('denoise_method', 'gaussian')
    sigma = params.get('sigma', 1.0)
    threshold_method = params.get('threshold_method', 'multiotsu')
    min_size = params.get('min_size', 50)
    marker_method = params.get('marker_method', 'distance')
    min_distance = params.get('min_distance', 10)
    segmentation_method = params.get('segmentation_method', 'watershed')
    split_touching = params.get('split_touching', True)
    border_clearing = params.get('border_clearing', False)
    n_clusters = params.get('n_clusters', 4)
    
    # Process image
    print("Enhancing contrast...")
    enhanced = enhance_contrast(image, method=enhance_method)
    
    print("Denoising image...")
    denoised = denoise_image(enhanced, method=denoise_method, sigma=sigma)
    
    print("Detecting cell regions...")
    binary = detect_cell_regions(denoised, method=threshold_method, min_size=min_size)
    
    print("Finding cell markers...")
    markers = find_cell_markers(
        denoised, binary, 
        method=marker_method, 
        min_distance=min_distance
    )
    
    print("Segmenting cells...")
    labels = segment_cells(denoised, binary, markers, method=segmentation_method)
    
    print("Refining segmentation...")
    refined_labels = refine_segmentation(
        labels, denoised, 
        min_size=min_size, 
        split_touching=split_touching,
        border_clearing=border_clearing
    )
    
    print("Extracting features...")
    props, df = extract_features(refined_labels, image)
    
    print("Clustering cells...")
    df = cluster_cells_by_intensity(df, n_clusters=n_clusters)
    
    # Define output paths
    img_output = os.path.join(output_dir, f"{file_stem}_segmentation.png")
    csv_output = os.path.join(output_dir, f"{file_stem}_segmentation.csv")
    
    # Plot segmentation
    print("Plotting segmentation...")
    plot_segmentation(image, refined_labels, df, output_path=img_output)
    
    # Save segmentation data
    print("Saving segmentation data...")
    save_segmentation_data(df, output_csv=csv_output)
    
    results = {
        'image': image,
        'enhanced': enhanced,
        'denoised': denoised,
        'binary': binary,
        'markers': markers,
        'labels': labels,
        'refined_labels': refined_labels,
        'props': props,
        'df': df,
        'img_output': img_output,
        'csv_output': csv_output,
        'n_objects': len(props)
    }
    
    print(f"Segmentation complete. Found {len(props)} objects.")
    return results

def run_pipeline(tiff_path, output_dir=None, channel=0, parameter_sets=None):
    """
    Run segmentation pipeline with different parameter sets.
    
    Parameters:
    -----------
    tiff_path : str
        Path to the TIFF file
    output_dir : str or None
        Directory to save output files
    channel : int
        Channel index to process
    parameter_sets : list or None
        List of parameter dictionaries to try
        
    Returns:
    --------
    best_result : dict
        Best segmentation result
    """
    if parameter_sets is None:
        # Default parameter sets to try
        parameter_sets = [
            # Set 1: Baseline parameters
            {
                'enhance_method': 'rescale',
                'denoise_method': 'gaussian',
                'sigma': 1.0,
                'threshold_method': 'multiotsu',
                'min_size': 50,
                'marker_method': 'distance',
                'min_distance': 10,
                'segmentation_method': 'watershed',
                'split_touching': True,
                'border_clearing': False,
                'n_clusters': 4
            },
            # Set 2: More aggressive denoising and local thresholding
            {
                'enhance_method': 'adaptive',
                'denoise_method': 'gaussian',
                'sigma': 2.0,
                'threshold_method': 'local',
                'min_size': 100,
                'marker_method': 'distance',
                'min_distance': 15,
                'segmentation_method': 'watershed',
                'split_touching': True,
                'border_clearing': True,
                'n_clusters': 4
            },
            # Set 3: LoG marker detection
            {
                'enhance_method': 'rescale',
                'denoise_method': 'gaussian',
                'sigma': 1.5,
                'threshold_method': 'multiotsu',
                'min_size': 50,
                'marker_method': 'log',
                'min_distance': 10,
                'segmentation_method': 'watershed',
                'split_touching': True,
                'border_clearing': False,
                'n_clusters': 4
            }
        ]
    
    best_result = None
    max_objects = 0
    
    # Create output directory if needed
    if output_dir is None:
        output_dir = os.path.dirname(tiff_path)
    
    # Create a subdirectory for parameter exploration
    explore_dir = os.path.join(output_dir, "parameter_exploration")
    os.makedirs(explore_dir, exist_ok=True)
    
    # Try each parameter set
    for i, params in enumerate(parameter_sets):
        print(f"\n{'='*50}")
        print(f"Parameter Set {i+1}/{len(parameter_sets)}")
        print(f"{'='*50}")
        
        # Create subdirectory for this parameter set
        set_dir = os.path.join(explore_dir, f"set_{i+1}")
        os.makedirs(set_dir, exist_ok=True)
        
        # Process with current parameters
        result = process_tiff_file(tiff_path, output_dir=set_dir, channel=channel, **params)
        
        if result is not None:
            n_objects = result['n_objects']
            print(f"Parameter Set {i+1}: Found {n_objects} objects")
            
            # Check if this result is better
            if n_objects > max_objects:
                max_objects = n_objects
                best_result = result
                best_result['parameter_set'] = i + 1
    
    if best_result is not None:
        print(f"\nBest Parameter Set: {best_result['parameter_set']} with {max_objects} objects")
        
        # Copy best result to main output directory
        best_set = best_result['parameter_set']
        best_dir = os.path.join(explore_dir, f"set_{best_set}")
        
        # Get file name without extension
        file_stem = Path(tiff_path).stem
        
        # Define source and destination paths
        src_img = os.path.join(best_dir, f"{file_stem}_segmentation.png")
        dst_img = os.path.join(output_dir, f"{file_stem}_segmentation.png")
        src_csv = os.path.join(best_dir, f"{file_stem}_segmentation.csv")
        dst_csv = os.path.join(output_dir, f"{file_stem}_segmentation.csv")
        
        # Copy files
        import shutil
        if os.path.exists(src_img):
            shutil.copy2(src_img, dst_img)
        if os.path.exists(src_csv):
            shutil.copy2(src_csv, dst_csv)
    
    return best_result

def main():
    parser = argparse.ArgumentParser(description='Cell segmentation for microscopy images')
    parser.add_argument('tiff_path', type=str, help='Path to TIFF file or directory')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    parser.add_argument('--channel', type=int, default=0, help='Channel to process')
    parser.add_argument('--enhance-method', type=str, default='rescale', 
                        choices=['rescale', 'adaptive', 'hist_eq'], help='Method for contrast enhancement')
    parser.add_argument('--denoise-method', type=str, default='gaussian', 
                        choices=['gaussian', 'median', 'bilateral'], help='Method for denoising')
    parser.add_argument('--sigma', type=float, default=1.0, help='Sigma for Gaussian filtering')
    parser.add_argument('--threshold-method', type=str, default='multiotsu', 
                        choices=['otsu', 'multiotsu', 'local', 'li', 'absolute'], 
                        help='Method for thresholding')
    parser.add_argument('--min-size', type=int, default=50, help='Minimum object size')
    parser.add_argument('--marker-method', type=str, default='distance', 
                        choices=['distance', 'log', 'dog', 'h_maxima'], 
                        help='Method for cell marker detection')
    parser.add_argument('--min-distance', type=int, default=10, 
                        help='Minimum distance between cell markers')
    parser.add_argument('--segmentation-method', type=str, default='watershed', 
                        choices=['watershed', 'random_walker'], help='Method for cell segmentation')
    parser.add_argument('--no-split-touching', dest='split_touching', action='store_false', 
                        help='Disable splitting of touching cells')
    parser.add_argument('--border-clearing', action='store_true', 
                        help='Enable clearing of objects touching the border')
    parser.add_argument('--n-clusters', type=int, default=4, 
                        help='Number of clusters for intensity-based clustering')
    parser.add_argument('--batch', action='store_true', 
                        help='Process all TIFF files in a directory')
    parser.add_argument('--explore-parameters', action='store_true', 
                        help='Explore different parameter sets')
    parser.set_defaults(split_touching=True, border_clearing=False)
    
    args = parser.parse_args()
    
    # Create parameter dictionary
    params = {
        'enhance_method': args.enhance_method,
        'denoise_method': args.denoise_method,
        'sigma': args.sigma,
        'threshold_method': args.threshold_method,
        'min_size': args.min_size,
        'marker_method': args.marker_method,
        'min_distance': args.min_distance,
        'segmentation_method': args.segmentation_method,
        'split_touching': args.split_touching,
        'border_clearing': args.border_clearing,
        'n_clusters': args.n_clusters
    }
    
    if args.batch:
        # Process all TIFF files in directory
        tiff_dir = args.tiff_path  # In batch mode, tiff_path is a directory
        output_dir = args.output_dir or tiff_dir
        
        for tiff_file in Path(tiff_dir).glob('*.tif*'):
            try:
                if args.explore_parameters:
                    run_pipeline(
                        str(tiff_file),
                        output_dir=os.path.join(output_dir, tiff_file.stem),
                        channel=args.channel
                    )
                else:
                    process_tiff_file(
                        str(tiff_file),
                        output_dir=os.path.join(output_dir, tiff_file.stem),
                        channel=args.channel,
                        **params
                    )
            except Exception as e:
                print(f"Error processing {tiff_file}: {e}")
                import traceback
                traceback.print_exc()
    else:
        # Process single file
        if args.explore_parameters:
            run_pipeline(
                args.tiff_path,
                output_dir=args.output_dir,
                channel=args.channel
            )
        else:
            process_tiff_file(
                args.tiff_path,
                output_dir=args.output_dir,
                channel=args.channel,
                **params
            )

if __name__ == "__main__":
    main()