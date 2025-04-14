import numpy as np
import tifffile
import matplotlib.pyplot as plt
from skimage import filters, morphology, measure, segmentation, feature
from skimage.exposure import rescale_intensity
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
        metadata = tif.imagej_metadata
        axes = metadata.get('axes', 'TZCYX')
        
        # Read the first timepoint
        if 'T' in axes and axes.startswith('T'):
            # For TZCYX format
            if len(axes) >= 5 and 'C' in axes:
                c_idx = axes.index('C')
                image = tif.asarray(0)  # First timepoint
                
                # Get the correct channel
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
                # No channel dimension
                image = tif.asarray(0)  # First timepoint
                if image.ndim > 2:
                    image = image[0]  # Take first Z
        else:
            # No time dimension, just read first page
            image = tif.asarray()
            if image.ndim > 2:
                if len(axes) >= 3 and 'C' in axes:
                    c_idx = axes.index('C')
                    channel_idx = min(channel, image.shape[c_idx-1]-1)
                    slices = [0] * image.ndim
                    slices[c_idx-1] = channel_idx
                    image = image[tuple(slices)]
                else:
                    image = image[0]  # Take first Z or channel
    
    # Ensure 2D
    if image.ndim > 2:
        image = image.reshape(image.shape[-2], image.shape[-1])
    
    return image

def segment_cells(image, min_distance=10, footprint_size=5, threshold_method='otsu', 
                  min_size=20, apply_watershed=True, sigma=1.0, contrast_enhance=True):
    """
    Segment cells or particles in a fluorescence microscopy image.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image
    min_distance : int
        Minimum distance between peaks in peak detection
    footprint_size : int
        Size of the local footprint for peak detection
    threshold_method : str
        Method for thresholding ('otsu', 'adaptive', or 'li')
    min_size : int
        Minimum size of objects to keep
    apply_watershed : bool
        Whether to apply watershed segmentation
    sigma : float
        Sigma for Gaussian filtering
    contrast_enhance : bool
        Whether to enhance contrast before processing
        
    Returns:
    --------
    labels : numpy.ndarray
        Labeled segmentation mask
    props : list
        Region properties for each labeled object
    """
    # Normalize and enhance contrast if needed
    if contrast_enhance:
        p2, p98 = np.percentile(image, (2, 98))
        image = rescale_intensity(image, in_range=(p2, p98))
    
    # Apply Gaussian filter to reduce noise
    smoothed = filters.gaussian(image, sigma=sigma)
    
    # Thresholding
    if threshold_method == 'otsu':
        thresh = filters.threshold_otsu(smoothed)
    elif threshold_method == 'adaptive':
        thresh = filters.threshold_local(smoothed, block_size=51)
    elif threshold_method == 'li':
        thresh = filters.threshold_li(smoothed)
    else:
        thresh = filters.threshold_otsu(smoothed)
    
    # Create binary mask
    binary = smoothed > thresh
    
    # Clean up binary image
    binary = morphology.remove_small_holes(binary, area_threshold=min_size*2)
    binary = morphology.remove_small_objects(binary, min_size=min_size)
    
    # Refine segmentation
    if apply_watershed:
        # Distance transform for watershed
        distance = morphology.distance_transform_edt(binary)
        
        # Find local maxima (peaks)
        footprint = morphology.disk(footprint_size)
        peaks = feature.peak_local_max(
            distance, 
            min_distance=min_distance,
            footprint=footprint,
            labels=binary
        )
        
        # Create markers for watershed
        markers = np.zeros_like(distance, dtype=int)
        markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
        
        # Apply watershed
        labels = segmentation.watershed(-distance, markers, mask=binary)
    else:
        # Use connected components if not using watershed
        labels = measure.label(binary)
    
    # Remove small objects again
    labels = morphology.remove_small_objects(labels, min_size=min_size)
    
    # Calculate properties of labeled regions
    props = measure.regionprops(labels, intensity_image=image)
    
    return labels, props

def plot_segmentation(image, labels, props, output_path=None, figsize=(12, 12), 
                      n_clusters=5, cmap='viridis'):
    """
    Plot segmentation results with colored labels and save to file.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Original input image
    labels : numpy.ndarray
        Labeled segmentation mask
    props : list
        Region properties for each labeled object
    output_path : str or None
        Path to save the output figure
    figsize : tuple
        Figure size
    n_clusters : int
        Number of clusters to group objects by intensity
    cmap : str
        Colormap name
    """
    # Extract mean intensities from props
    if len(props) == 0:
        print("No objects found in segmentation")
        return
    
    intensities = np.array([prop.mean_intensity for prop in props])
    
    # Create clusters based on intensities
    if len(intensities) >= n_clusters:
        # Use quantiles for clustering
        thresholds = np.percentile(intensities, np.linspace(0, 100, n_clusters+1)[1:-1])
        clusters = np.zeros(len(intensities), dtype=int)
        
        for i, threshold in enumerate(thresholds):
            clusters[intensities > threshold] = i + 1
    else:
        # Not enough objects for requested clusters
        clusters = np.zeros(len(intensities), dtype=int)
    
    # Define colors for clusters
    cluster_colors = plt.cm.get_cmap('tab10', n_clusters)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Display original image
    ax.imshow(image, cmap='gray')
    
    # Create colored labels based on clusters
    for i, prop in enumerate(props):
        y, x = prop.centroid
        label = prop.label
        cluster = clusters[i]
        color = cluster_colors(cluster)
        
        # Draw text with colored background based on cluster
        ax.text(x, y, str(label), color='white', fontsize=8, 
                ha='center', va='center', 
                bbox=dict(facecolor=color, alpha=0.7, edgecolor='none', pad=1))
    
    ax.set_title('First Frame Segmentation')
    ax.axis('on')
    
    # Save figure if output_path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Segmentation image saved to: {output_path}")
    
    plt.close(fig)
    
    return fig

def save_segmentation_data(props, output_csv=None):
    """
    Save segmentation data to CSV file.
    
    Parameters:
    -----------
    props : list
        Region properties for each labeled object
    output_csv : str or None
        Path to save the output CSV file
    
    Returns:
    --------
    df : pandas.DataFrame
        DataFrame with segmentation data
    """
    if len(props) == 0:
        print("No objects found in segmentation")
        return pd.DataFrame()
    
    # Extract properties
    data = []
    for prop in props:
        y, x = prop.centroid
        data.append({
            'label': prop.label,
            'x': x,
            'y': y,
            'area': prop.area,
            'perimeter': prop.perimeter,
            'eccentricity': prop.eccentricity,
            'mean_intensity': prop.mean_intensity,
            'max_intensity': prop.max_intensity,
            'min_intensity': prop.min_intensity
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV if output_csv is provided
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Segmentation data saved to: {output_csv}")
    
    return df

def process_tiff_file(tiff_path, output_dir=None, channel=0, **segmentation_params):
    """
    Process a single TIFF file and generate segmentation results.
    
    Parameters:
    -----------
    tiff_path : str
        Path to the TIFF file
    output_dir : str or None
        Directory to save output files
    channel : int
        Channel index to process
    **segmentation_params : dict
        Additional parameters for segmentation
    
    Returns:
    --------
    results : dict
        Dictionary with segmentation results
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
    
    # Segment cells
    print("Segmenting objects...")
    labels, props = segment_cells(image, **segmentation_params)
    
    # Define output paths
    img_output = os.path.join(output_dir, f"{file_stem}_segmentation.png")
    csv_output = os.path.join(output_dir, f"{file_stem}_segmentation.csv")
    
    # Plot segmentation
    print("Plotting segmentation...")
    plot_segmentation(image, labels, props, output_path=img_output)
    
    # Save segmentation data
    print("Saving segmentation data...")
    df = save_segmentation_data(props, output_csv=csv_output)
    
    results = {
        'image': image,
        'labels': labels,
        'props': props,
        'df': df,
        'img_output': img_output,
        'csv_output': csv_output,
        'n_objects': len(props)
    }
    
    print(f"Segmentation complete. Found {len(props)} objects.")
    return results

def main():
    parser = argparse.ArgumentParser(description='Segment objects in the first frame of a TIFF file')
    parser.add_argument('tiff_path', type=str, help='Path to TIFF file')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    parser.add_argument('--channel', type=int, default=0, help='Channel to process')
    parser.add_argument('--min-distance', type=int, default=10, help='Minimum distance between peaks')
    parser.add_argument('--footprint-size', type=int, default=5, help='Size of footprint for peak detection')
    parser.add_argument('--threshold-method', type=str, default='otsu', 
                        choices=['otsu', 'adaptive', 'li'], help='Thresholding method')
    parser.add_argument('--min-size', type=int, default=20, help='Minimum object size')
    parser.add_argument('--sigma', type=float, default=1.0, help='Sigma for Gaussian filtering')
    parser.add_argument('--no-watershed', action='store_false', dest='apply_watershed', 
                        help='Disable watershed segmentation')
    parser.add_argument('--no-contrast-enhance', action='store_false', dest='contrast_enhance',
                        help='Disable contrast enhancement')
    parser.add_argument('--batch', action='store_true', help='Process all TIFF files in a directory')
    
    args = parser.parse_args()
    
    segmentation_params = {
        'min_distance': args.min_distance,
        'footprint_size': args.footprint_size,
        'threshold_method': args.threshold_method,
        'min_size': args.min_size,
        'apply_watershed': args.apply_watershed,
        'sigma': args.sigma,
        'contrast_enhance': args.contrast_enhance
    }
    
    if args.batch:
        # Process all TIFF files in directory
        tiff_dir = args.tiff_path  # In batch mode, tiff_path is a directory
        output_dir = args.output_dir or tiff_dir
        
        for tiff_file in Path(tiff_dir).glob('*.tif'):
            try:
                process_tiff_file(
                    str(tiff_file),
                    output_dir=output_dir,
                    channel=args.channel,
                    **segmentation_params
                )
            except Exception as e:
                print(f"Error processing {tiff_file}: {e}")
    else:
        # Process single file
        process_tiff_file(
            args.tiff_path,
            output_dir=args.output_dir,
            channel=args.channel,
            **segmentation_params
        )

if __name__ == "__main__":
    main()