import h5py
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# File paths - CHANGE THESE to your actual file paths
ilastik_h5_path = "x.h5"
original_image_path = "ampk-cyto-1_FusionStitcher_F0-ex-a7.tif"
output_dir = "results_ex"

# Create output directory if it doesn't exist
Path(output_dir).mkdir(exist_ok=True, parents=True)

# Load the mask from ilastik export
print(f"Loading mask from {ilastik_h5_path}")
with h5py.File(ilastik_h5_path, "r") as f:
    # Print keys to see the structure
    print("Keys in ilastik file:", list(f.keys()))
    
    # Try to find the mask dataset - you may need to adjust this path
    # based on your specific ilastik export format
    try:
        # Common ilastik export paths - try a few possibilities
        if "exported_data" in f:
            mask_data = f["exported_data"][:]
            print("Found mask in 'exported_data'")
        elif "segmentation" in f:
            mask_data = f["segmentation"][:]
            print("Found mask in 'segmentation'")
        else:
            # If we can't find a common key, use the first available dataset
            first_key = list(f.keys())[0]
            mask_data = f[first_key][:]
            print(f"Using first available key: '{first_key}'")
    except Exception as e:
        print(f"Error loading mask: {e}")
        # Try deeper exploration if the top level doesn't work
        for key in f.keys():
            print(f"Exploring key '{key}':")
            if isinstance(f[key], h5py.Group):
                print(f"  Subkeys: {list(f[key].keys())}")
                if len(list(f[key].keys())) > 0:
                    first_subkey = list(f[key].keys())[0]
                    mask_data = f[key][first_subkey][:]
                    print(f"  Using '{key}/{first_subkey}'")
                    break
            else:
                print(f"  This is a dataset, shape: {f[key].shape}")

# Print mask information
print("Mask data shape:", mask_data.shape)
print("Mask data type:", mask_data.dtype)

# Convert to binary mask if needed
if mask_data.dtype != bool:
    if np.max(mask_data) > 1:
        # Multi-class segmentation
        mask_data = mask_data > 0
    else:
        # Probability map
        mask_data = mask_data > 0.5

print("Mask is now binary with shape:", mask_data.shape)

# Load original timelapse
print(f"Loading timelapse from {original_image_path}")
try:
    timelapse = tifffile.imread(original_image_path)
    print("Timelapse loaded with shape:", timelapse.shape)
except Exception as e:
    print(f"Error loading timelapse: {e}")
    # Try alternative loading method
    try:
        with h5py.File(original_image_path, "r") as f:
            print("Timelapse H5 keys:", list(f.keys()))
            first_key = list(f.keys())[0]
            timelapse = f[first_key][:]
            print("Timelapse loaded from H5 with shape:", timelapse.shape)
    except Exception as e2:
        print(f"Alternative loading also failed: {e2}")
        raise

# Print all dimensions to help understand the data structure
print(f"Timelapse dimensions: {timelapse.shape}")
print(f"Mask dimensions: {mask_data.shape}")

# Try to determine the data structure based on common patterns
# This is a simplified approach and may need to be adjusted
if len(timelapse.shape) == 4:
    # Possible formats: (t, z, y, x), (t, y, x, c), (c, t, y, x)
    if timelapse.shape[3] <= 5:  # Likely (t, y, x, c)
        num_timepoints = timelapse.shape[0]
        num_channels = timelapse.shape[3]
        print(f"Detected format: (t, y, x, c) with {num_timepoints} timepoints and {num_channels} channels")
        
        # Function to extract channels for a timepoint
        def get_channels(t):
            return timelapse[t, :, :, 0], timelapse[t, :, :, 1]
    elif timelapse.shape[0] <= 5:  # Likely (c, t, y, x)
        num_timepoints = timelapse.shape[1]
        num_channels = timelapse.shape[0]
        print(f"Detected format: (c, t, y, x) with {num_timepoints} timepoints and {num_channels} channels")
        
        # Function to extract channels for a timepoint
        def get_channels(t):
            return timelapse[0, t], timelapse[1, t]
    else:  # Default to (t, c, y, x)
        num_timepoints = timelapse.shape[0]
        num_channels = timelapse.shape[1]
        print(f"Detected format: (t, c, y, x) with {num_timepoints} timepoints and {num_channels} channels")
        
        # Function to extract channels for a timepoint
        def get_channels(t):
            return timelapse[t, 0], timelapse[t, 1]
elif len(timelapse.shape) == 3:
    # Possible formats: (t, y, x) or (y, x, c)
    if timelapse.shape[2] <= 5:  # Likely (y, x, c)
        num_timepoints = 1
        num_channels = timelapse.shape[2]
        print(f"Detected format: (y, x, c) with 1 timepoint and {num_channels} channels")
        
        # Function to extract channels for a timepoint
        def get_channels(t):
            return timelapse[:, :, 0], timelapse[:, :, 1]
    else:  # Likely (t, y, x)
        num_timepoints = timelapse.shape[0]
        num_channels = 1
        print(f"Detected format: (t, y, x) with {num_timepoints} timepoints and 1 channel")
        
        # Cannot calculate ratio with only 1 channel
        raise ValueError("Cannot calculate ratio with only 1 channel")
else:
    raise ValueError(f"Unexpected timelapse shape: {timelapse.shape}")

# Check if mask has time dimension
mask_has_time = len(mask_data.shape) >= 3 and mask_data.shape[0] == num_timepoints
if mask_has_time:
    print("Mask has time dimension")
    def get_mask(t):
        if len(mask_data.shape) == 3:  # (t, y, x)
            return mask_data[t]
        else:  # More dimensions - try to get the right slice
            return mask_data[t, :, :]
else:
    print("Mask does not have time dimension, using same mask for all timepoints")
    if len(mask_data.shape) == 2:  # Simple (y, x)
        def get_mask(t):
            return mask_data
    else:  # More dimensions - try to get a 2D slice
        print(f"Warning: Mask has unusual shape {mask_data.shape}, will try to use first 2D slice")
        if len(mask_data.shape) == 3:
            mask_data = mask_data[0]  # Use first slice if 3D
        else:
            mask_data = mask_data[0, 0]  # Use first slice if 4D or more
        def get_mask(t):
            return mask_data

# Calculate ratios for each timepoint
time_points = []
ch1_means = []
ch2_means = []
ratios = []
normalized_ratios = []

for t in range(num_timepoints):
    try:
        # Get channels and mask for this timepoint
        ch1, ch2 = get_channels(t)
        current_mask = get_mask(t)
        
        # Ensure mask and channel data have compatible shapes
        if current_mask.shape != ch1.shape:
            print(f"Warning: Mask shape {current_mask.shape} doesn't match channel shape {ch1.shape}")
            # Try to resize the mask
            if len(current_mask.shape) == 2 and len(ch1.shape) == 2:
                if current_mask.shape[0] > ch1.shape[0] or current_mask.shape[1] > ch1.shape[1]:
                    # Crop mask
                    current_mask = current_mask[:ch1.shape[0], :ch1.shape[1]]
                else:
                    # Pad mask
                    temp_mask = np.zeros(ch1.shape, dtype=bool)
                    temp_mask[:current_mask.shape[0], :current_mask.shape[1]] = current_mask
                    current_mask = temp_mask
                print(f"Adjusted mask to shape {current_mask.shape}")
        
        # Calculate means
        ch1_mean = np.mean(ch1[current_mask])
        ch2_mean = np.mean(ch2[current_mask])
        ratio = ch1_mean / ch2_mean if ch2_mean != 0 else np.nan
        
        time_points.append(t)
        ch1_means.append(ch1_mean)
        ch2_means.append(ch2_mean)
        ratios.append(ratio)
        
        print(f"Time point {t}: Ch1 mean = {ch1_mean:.2f}, Ch2 mean = {ch2_mean:.2f}, Ratio = {ratio:.2f}")
    except Exception as e:
        print(f"Error processing time point {t}: {e}")
        # Try a different approach with flattened arrays
        try:
            ch1, ch2 = get_channels(t)
            current_mask = get_mask(t)
            
            # Flatten everything
            flat_ch1 = ch1.flatten()
            flat_ch2 = ch2.flatten()
            flat_mask = current_mask.flatten()
            
            # Calculate on flattened arrays
            ch1_mean = np.mean(flat_ch1[flat_mask])
            ch2_mean = np.mean(flat_ch2[flat_mask])
            ratio = ch1_mean / ch2_mean if ch2_mean != 0 else np.nan
            
            time_points.append(t)
            ch1_means.append(ch1_mean)
            ch2_means.append(ch2_mean)
            ratios.append(ratio)
            
            print(f"Time point {t} (flattened approach): Ch1 mean = {ch1_mean:.2f}, Ch2 mean = {ch2_mean:.2f}, Ratio = {ratio:.2f}")
        except Exception as e2:
            print(f"Flattened approach also failed for time point {t}: {e2}")
            # Add placeholder values
            time_points.append(t)
            ch1_means.append(np.nan)
            ch2_means.append(np.nan)
            ratios.append(np.nan)

# Calculate normalization based on first 15 frames
if len(ratios) >= 15:
    # Calculate the mean of the first 15 frames, excluding any NaN values
    first_15_valid = [r for r in ratios[:15] if not np.isnan(r)]
    if first_15_valid:
        baseline = np.mean(first_15_valid)
        print(f"Normalizing to baseline value: {baseline:.4f} (mean of first 15 frames)")
        normalized_ratios = [r / baseline if not np.isnan(r) else np.nan for r in ratios]
    else:
        print("Warning: No valid ratios in first 15 frames for normalization")
        normalized_ratios = ratios.copy()
else:
    print(f"Warning: Only {len(ratios)} frames available, using all for normalization")
    valid_ratios = [r for r in ratios if not np.isnan(r)]
    if valid_ratios:
        baseline = np.mean(valid_ratios)
        print(f"Normalizing to baseline value: {baseline:.4f}")
        normalized_ratios = [r / baseline if not np.isnan(r) else np.nan for r in ratios]
    else:
        print("Warning: No valid ratios for normalization")
        normalized_ratios = ratios.copy()

# Create DataFrame with results
results_df = pd.DataFrame({
    'Time_Point': time_points,
    'Channel1_Mean': ch1_means,
    'Channel2_Mean': ch2_means,
    'Ratio_Ch1_Ch2': ratios,
    'Normalized_Ratio': normalized_ratios
})

# Save results
csv_path = f"{output_dir}/channel_ratio_analysis.csv"
results_df.to_csv(csv_path, index=False)
print(f"Results saved to {csv_path}")

# Plot both raw and normalized ratios
plt.figure(figsize=(12, 10))

# Raw ratios subplot
plt.subplot(2, 1, 1)
plt.plot(time_points, ratios, 'o-', linewidth=2, color='blue')
plt.xlabel('Time Point')
plt.ylabel('Ratio (Channel 1 / Channel 2)')
plt.title('Raw Channel Ratio Over Time')
plt.grid(True)

# Normalized ratios subplot
plt.subplot(2, 1, 2)
plt.plot(time_points, normalized_ratios, 'o-', linewidth=2, color='green')
plt.xlabel('Time Point')
plt.ylabel('Normalized Ratio')
plt.title('Normalized Channel Ratio (Relative to First 15 Frames)')
plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7)  # Add a reference line at y=1
plt.grid(True)

plt.tight_layout()  # Adjust spacing between subplots

# Save plot
plot_path = f"{output_dir}/channel_ratio_plot.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to {plot_path}")

# Show plot
plt.show()