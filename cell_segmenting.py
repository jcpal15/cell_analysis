import numpy as np
from skimage import io, filters, measure, segmentation, exposure
import matplotlib.pyplot as plt
from scipy import ndimage
import os

def analyze_multichannel_depletion(tif_path, ref_channel=1, meas_channel=0, 
                                 percentile_low=0.5, percentile_high=99.5,
                                 crop_size=50, min_cell_size=100):
    
    # Create output directory
    output_dir = 'cell_crops'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load image
    img = io.imread(tif_path)
    reference_channel = img[ref_channel] if img.ndim == 3 else img[:, :, ref_channel]
    measurement_channel = img[meas_channel] if img.ndim == 3 else img[:, :, meas_channel]
    
    # Enhanced preprocessing
    # Apply stronger Gaussian smoothing to merge nearby spots
    smoothed = filters.gaussian(reference_channel, sigma=2.0)
    
    # Enhance contrast
    p_low, p_high = np.percentile(smoothed, (percentile_low, percentile_high))
    reference_enhanced = exposure.rescale_intensity(smoothed, in_range=(p_low, p_high))
    
    # Threshold selection
    threshold_choice = input("Enter threshold method (otsu/li) or a manual value (0-1): ")
    
    if threshold_choice == 'otsu':
        thresh = filters.threshold_otsu(reference_enhanced)
    elif threshold_choice == 'li':
        thresh = filters.threshold_li(reference_enhanced)
    else:
        try:
            thresh = float(threshold_choice)
        except ValueError:
            print("Invalid input, using Otsu's method")
            thresh = filters.threshold_otsu(reference_enhanced)
    
    # Create binary mask with more aggressive cleaning
    binary = reference_enhanced > thresh
    binary = ndimage.binary_fill_holes(binary)
    
    # More aggressive morphological operations
    # This helps merge nearby spots that belong to the same cell
    binary = ndimage.binary_closing(binary, iterations=3)
    binary = ndimage.binary_opening(binary, iterations=2)
    
    # Remove small objects and fill holes again
    binary = ndimage.binary_fill_holes(binary)
    binary = ndimage.binary_opening(binary, structure=np.ones((3,3)))
    
    # Label regions
    labels = measure.label(binary)
    
    # Get region properties
    props = measure.regionprops(labels, reference_channel)
    
    # Filter regions based on improved criteria
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
    
    # Create new labels image with only valid cells
    filtered_labels = np.zeros_like(labels)
    for label in valid_labels:
        filtered_labels[labels == label] = label
    
    # Get properties of filtered regions
    props = measure.regionprops(filtered_labels, reference_channel)
    
    # Crop and analyze cells
    cell_data = []
    half_crop = crop_size // 2
    
    # Pad images
    pad_width = ((half_crop, half_crop), (half_crop, half_crop))
    ref_padded = np.pad(reference_channel, pad_width, mode='constant')
    meas_padded = np.pad(measurement_channel, pad_width, mode='constant')
    
    for prop in props:
        # Get padded coordinates
        y, x = [int(c) + half_crop for c in prop.centroid]
        
        # Extract crops
        ref_crop = ref_padded[y-half_crop:y+half_crop, x-half_crop:x+half_crop]
        meas_crop = meas_padded[y-half_crop:y+half_crop, x-half_crop:x+half_crop]
        
        # Save crops
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
        plt.savefig(os.path.join(output_dir, f'cell_{prop.label}.png'))
        plt.close()
        
        # Store measurements
        cell_data.append({
            'cell_id': prop.label,
            'area': prop.area,
            'centroid_y': prop.centroid[0],
            'centroid_x': prop.centroid[1],
            'eccentricity': prop.eccentricity,
            'solidity': prop.solidity,
            'reference_intensity': prop.mean_intensity,
            'measurement_intensity': measure.regionprops(filtered_labels, measurement_channel)[props.index(prop)].mean_intensity
        })
    
    # Create summary statistics
    print("\nCell Analysis Summary:")
    print("-" * 50)
    print(f"Total cells analyzed: {len(cell_data)}")
    
    # Calculate summary statistics
    areas = [cell['area'] for cell in cell_data]
    ref_intensities = [cell['reference_intensity'] for cell in cell_data]
    meas_intensities = [cell['measurement_intensity'] for cell in cell_data]
    
    # Create summary figure
    plt.figure(figsize=(15, 10))
    
    # Area distribution
    plt.subplot(2, 2, 1)
    plt.hist(areas, bins=20)
    plt.title('Cell Area Distribution')
    plt.xlabel('Area (pixels)')
    plt.ylabel('Count')
    
    # Reference channel intensity
    plt.subplot(2, 2, 2)
    plt.hist(ref_intensities, bins=20)
    plt.title('Reference Channel Intensity Distribution')
    plt.xlabel('Mean Intensity')
    plt.ylabel('Count')
    
    # Measurement channel intensity
    plt.subplot(2, 2, 3)
    plt.hist(meas_intensities, bins=20)
    plt.title('Measurement Channel Intensity Distribution')
    plt.xlabel('Mean Intensity')
    plt.ylabel('Count')
    
    # Scatter plot of intensities
    plt.subplot(2, 2, 4)
    plt.scatter(ref_intensities, meas_intensities, alpha=0.5)
    plt.title('Reference vs Measurement Intensity')
    plt.xlabel('Reference Channel Intensity')
    plt.ylabel('Measurement Channel Intensity')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_statistics.png'))
    plt.close()
    
    # Print summary statistics
    print(f"\nArea Statistics:")
    print(f"Mean: {np.mean(areas):.1f} px")
    print(f"Std Dev: {np.std(areas):.1f} px")
    print(f"Min: {np.min(areas):.1f} px")
    print(f"Max: {np.max(areas):.1f} px")
    
    print(f"\nReference Channel Intensity Statistics:")
    print(f"Mean: {np.mean(ref_intensities):.1f}")
    print(f"Std Dev: {np.std(ref_intensities):.1f}")
    print(f"Min: {np.min(ref_intensities):.1f}")
    print(f"Max: {np.max(ref_intensities):.1f}")
    
    print(f"\nMeasurement Channel Intensity Statistics:")
    print(f"Mean: {np.mean(meas_intensities):.1f}")
    print(f"Std Dev: {np.std(meas_intensities):.1f}")
    print(f"Min: {np.min(meas_intensities):.1f}")
    print(f"Max: {np.max(meas_intensities):.1f}")
    
    return cell_data, filtered_labels

# Usage example:
cell_data, labels = analyze_multichannel_depletion('control.tif', crop_size=50, min_cell_size=100)