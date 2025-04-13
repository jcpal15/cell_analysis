import numpy as np
import h5py
import tifffile
import os
import glob
from pathlib import Path
import re
from tqdm import tqdm

def extract_identifier(filename):
    """
    Extract identifier (e.g. 'F0', 'F1') from the filename
    Examples:
    - "baseline-try6-epi_F0.ims" -> "F0"
    - "epi-druged_F0.ims" -> "F0"
    """
    # Get the stem (filename without extension)
    stem = Path(filename).stem
    
    # Try to find an underscore followed by one or more characters at the end
    match = re.search(r'_([^_]+)$', stem)
    if match:
        return match.group(1)
    else:
        # If no underscore pattern found, return the whole stem as identifier
        return stem

def ims_to_numpy(ims_path, resolution_level=0, channel=None, z_slice=0):
    """
    Read all time points from an IMS file into a numpy array
    
    Parameters:
    -----------
    ims_path : str
        Path to the IMS file
    resolution_level : int
        Resolution level to read
    channel : int or None
        If specified, read only this channel. If None, read all channels.
    z_slice : int
        Z-slice to read
    
    Returns:
    --------
    data : numpy.ndarray
        Array with dimensions [time_points, (channels), height, width]
    """
    with h5py.File(ims_path, 'r') as f:
        # Get the number of time points
        time_points_path = f'DataSet/ResolutionLevel {resolution_level}'
        time_point_keys = [k for k in f[time_points_path].keys() if k.startswith('TimePoint')]
        num_time_points = len(time_point_keys)
        
        # Get number of channels
        channel_keys = [k for k in f[f'{time_points_path}/TimePoint 0'].keys() if k.startswith('Channel')]
        num_channels = len(channel_keys)
        
        # Read the data shape from the first time point and channel
        first_data_path = f'{time_points_path}/TimePoint 0/Channel 0/Data'
        data_shape = f[first_data_path].shape
        
        # Create array to hold all data
        if channel is None:
            # All channels
            full_data = np.zeros((num_time_points, num_channels, data_shape[1], data_shape[2]), dtype=np.float32)
        else:
            # Single channel
            full_data = np.zeros((num_time_points, data_shape[1], data_shape[2]), dtype=np.float32)
        
        # Read data for each time point
        for time_point in tqdm(range(num_time_points), desc=f"Reading {Path(ims_path).name}"):
            if channel is None:
                # Read all channels
                for ch in range(num_channels):
                    data_path = f'{time_points_path}/TimePoint {time_point}/Channel {ch}/Data'
                    full_data[time_point, ch] = f[data_path][z_slice]
            else:
                # Read specific channel
                data_path = f'{time_points_path}/TimePoint {time_point}/Channel {channel}/Data'
                full_data[time_point] = f[data_path][z_slice]
    
    return full_data

def concatenate_ims_files(baseline_file, drugged_file, output_file, channels=[0, 1], z_slice=0, resolution_level=0):
    """
    Concatenate two IMS files along the time dimension and save as a TIF file
    
    Parameters:
    -----------
    baseline_file : str
        Path to the baseline IMS file
    drugged_file : str
        Path to the drugged IMS file
    output_file : str
        Path to the output TIF file
    channels : list
        List of channels to include
    z_slice : int
        Z-slice to extract
    resolution_level : int
        Resolution level to read
    """
    # Read data from both files
    print(f"Reading baseline file: {baseline_file}")
    baseline_data = ims_to_numpy(baseline_file, resolution_level, None, z_slice)
    
    print(f"Reading drugged file: {drugged_file}")
    drugged_data = ims_to_numpy(drugged_file, resolution_level, None, z_slice)
    
    # Check that the files have the same number of channels and dimensions
    if baseline_data.shape[1:] != drugged_data.shape[1:]:
        raise ValueError(f"Files have different dimensions: {baseline_data.shape[1:]} vs {drugged_data.shape[1:]}")
    
    # Filter to include only the specified channels
    if channels is not None:
        baseline_data = baseline_data[:, channels]
        drugged_data = drugged_data[:, channels]
    
    # Concatenate the data along the time dimension
    combined_data = np.concatenate([baseline_data, drugged_data], axis=0)
    
    # Normalize to 16-bit range for TIFF
    # Find global min and max
    data_min = np.min(combined_data)
    data_max = np.max(combined_data)
    
    # Scale to 0-65535 range (16-bit)
    if data_max > data_min:
        combined_data = ((combined_data - data_min) / (data_max - data_min) * 65535).astype(np.uint16)
    else:
        combined_data = np.zeros_like(combined_data, dtype=np.uint16)
    
    # Save as multi-page TIFF
    print(f"Saving combined data to: {output_file}")
    
    # For multi-channel data, we arrange as TZCYX (time, z=1, channel, y, x)
    if combined_data.ndim == 4:  # time, channel, height, width
        combined_data = combined_data[:, np.newaxis, :, :, :]  # Add Z dimension
    
    tifffile.imwrite(output_file, combined_data, imagej=True, metadata={
        'axes': 'TZCYX' if combined_data.ndim == 5 else 'TZYX',
        'channels': len(channels) if channels is not None else combined_data.shape[2],
        'slices': 1,  # We're only saving one Z slice
        'frames': combined_data.shape[0],  # Number of time points
    })
    
    print(f"Successfully concatenated {baseline_data.shape[0]} baseline frames with "
          f"{drugged_data.shape[0]} drugged frames for a total of {combined_data.shape[0]} frames")
    
    return {
        'baseline_frames': baseline_data.shape[0],
        'drugged_frames': drugged_data.shape[0],
        'total_frames': combined_data.shape[0],
        'output_file': output_file
    }

def find_and_concatenate_matching_files(baseline_folder, drugged_folder, output_folder, 
                                       channels=[0, 1], z_slice=0, resolution_level=0):
    """
    Find matching IMS files in baseline and drugged folders and concatenate them
    
    Parameters:
    -----------
    baseline_folder : str
        Path to folder containing baseline IMS files
    drugged_folder : str
        Path to folder containing drugged IMS files
    output_folder : str
        Path to folder for output TIF files
    channels : list
        List of channels to include
    z_slice : int
        Z-slice to extract
    resolution_level : int
        Resolution level to read
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Find all IMS files in both folders
    baseline_files = glob.glob(os.path.join(baseline_folder, "*.ims"))
    drugged_files = glob.glob(os.path.join(drugged_folder, "*.ims"))
    
    if not baseline_files:
        print(f"No IMS files found in baseline folder: {baseline_folder}")
        return
    
    if not drugged_files:
        print(f"No IMS files found in drugged folder: {drugged_folder}")
        return
    
    print(f"Found {len(baseline_files)} baseline files and {len(drugged_files)} drugged files")
    
    # Create a mapping of identifiers to files
    baseline_map = {}
    for file in baseline_files:
        identifier = extract_identifier(file)
        baseline_map[identifier] = file
        
    drugged_map = {}
    for file in drugged_files:
        identifier = extract_identifier(file)
        drugged_map[identifier] = file
    
    # Find matching pairs
    matched_pairs = []
    for identifier in baseline_map:
        if identifier in drugged_map:
            matched_pairs.append({
                'identifier': identifier,
                'baseline': baseline_map[identifier],
                'drugged': drugged_map[identifier]
            })
    
    print(f"Found {len(matched_pairs)} matching pairs of files")
    
    if not matched_pairs:
        print("No matching files found! Check file naming patterns.")
        return
    
    # Process each matched pair
    results = []
    for pair in matched_pairs:
        identifier = pair['identifier']
        baseline_file = pair['baseline']
        drugged_file = pair['drugged']
        
        output_file = os.path.join(output_folder, f"combined_{identifier}.tif")
        
        print(f"\n{'='*50}")
        print(f"Processing pair {identifier}: {os.path.basename(baseline_file)} + {os.path.basename(drugged_file)}")
        print(f"Output: {output_file}")
        print(f"{'='*50}")
        
        try:
            result = concatenate_ims_files(
                baseline_file, 
                drugged_file, 
                output_file,
                channels=channels,
                z_slice=z_slice,
                resolution_level=resolution_level
            )
            results.append({
                'identifier': identifier,
                'baseline_file': baseline_file,
                'drugged_file': drugged_file,
                'output_file': output_file,
                'baseline_frames': result['baseline_frames'],
                'drugged_frames': result['drugged_frames'],
                'total_frames': result['total_frames']
            })
        except Exception as e:
            print(f"Error processing pair {identifier}: {e}")
            import traceback
            traceback.print_exc()
    
    # Create summary report
    if results:
        # Create a DataFrame with the results
        import pandas as pd
        df = pd.DataFrame(results)
        summary_file = os.path.join(output_folder, "concatenation_summary.csv")
        df.to_csv(summary_file, index=False)
        
        print(f"\nConcatenation summary saved to: {summary_file}")
        print("\nSummary of concatenated files:")
        for result in results:
            print(f"{result['identifier']}: {result['baseline_frames']} baseline frames + "
                  f"{result['drugged_frames']} drugged frames = {result['total_frames']} total frames")
    else:
        print("\nNo files were successfully concatenated.")
        
    return results

def main():
    print("=" * 50)
    print("IMS File Concatenator - Baseline + Drugged")
    print("=" * 50)
    
    baseline_folder = input("Enter path to baseline folder: ")
    drugged_folder = input("Enter path to drugged folder: ")
    output_folder = input("Enter output folder for combined TIF files: ")
    
    # Optional parameters
    include_channels = input("Enter channels to include (comma-separated, default: 0,1): ")
    if include_channels:
        channels = [int(ch.strip()) for ch in include_channels.split(',')]
    else:
        channels = [0, 1]
    
    z_slice = input("Enter z-slice to extract (default: 0): ")
    z_slice = int(z_slice) if z_slice else 0
    
    resolution_level = input("Enter resolution level (default: 0): ")
    resolution_level = int(resolution_level) if resolution_level else 0
    
    find_and_concatenate_matching_files(
        baseline_folder,
        drugged_folder,
        output_folder,
        channels=channels,
        z_slice=z_slice,
        resolution_level=resolution_level
    )

if __name__ == "__main__":
    main()