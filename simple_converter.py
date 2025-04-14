import numpy as np
import h5py
import tifffile
import os
import glob
from pathlib import Path
import re
from tqdm import tqdm
import argparse

def ims_to_numpy(ims_path, resolution_level=0, channels=None, z_slice=0):
    """
    Read data from an IMS file into a numpy array
    
    Parameters:
    -----------
    ims_path : str
        Path to the IMS file
    resolution_level : int
        Resolution level to read
    channels : list or None
        If specified, read only these channels. If None, read all channels.
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
        
        # If channels not specified, use all channels
        if channels is None:
            channels = list(range(num_channels))
        
        # Read the data shape from the first time point and channel
        first_data_path = f'{time_points_path}/TimePoint 0/Channel 0/Data'
        data_shape = f[first_data_path].shape
        
        # Create array to hold all data
        if len(channels) > 1:
            # Multiple channels
            full_data = np.zeros((num_time_points, len(channels), data_shape[1], data_shape[2]), dtype=np.float32)
        else:
            # Single channel
            full_data = np.zeros((num_time_points, data_shape[1], data_shape[2]), dtype=np.float32)
        
        # Read data for each time point
        for time_point in tqdm(range(num_time_points), desc=f"Reading {Path(ims_path).name}"):
            if len(channels) > 1:
                # Read specified channels
                for ch_idx, ch in enumerate(channels):
                    data_path = f'{time_points_path}/TimePoint {time_point}/Channel {ch}/Data'
                    full_data[time_point, ch_idx] = f[data_path][z_slice]
            else:
                # Read single channel
                data_path = f'{time_points_path}/TimePoint {time_point}/Channel {channels[0]}/Data'
                full_data[time_point] = f[data_path][z_slice]
    
    return full_data

def convert_ims_to_tif(ims_file, output_file, channels=None, z_slice=0, resolution_level=0):
    """
    Convert an IMS file to a TIF file
    
    Parameters:
    -----------
    ims_file : str
        Path to the IMS file
    output_file : str
        Path to the output TIF file
    channels : list or None
        List of channels to include. If None, include all channels.
    z_slice : int
        Z-slice to extract
    resolution_level : int
        Resolution level to read
    
    Returns:
    --------
    dict
        Dictionary with information about the conversion
    """
    try:
        # Read data from the IMS file
        print(f"Reading file: {ims_file}")
        data = ims_to_numpy(ims_file, resolution_level, channels, z_slice)
        
        # Normalize to 16-bit range for TIFF
        # Find global min and max
        data_min = np.min(data)
        data_max = np.max(data)
        
        # Scale to 0-65535 range (16-bit)
        if data_max > data_min:
            data = ((data - data_min) / (data_max - data_min) * 65535).astype(np.uint16)
        else:
            data = np.zeros_like(data, dtype=np.uint16)
        
        # Save as multi-page TIFF
        print(f"Saving data to: {output_file}")
        
        # For multi-channel data, we arrange as TZCYX (time, z=1, channel, y, x)
        if data.ndim == 4:  # time, channel, height, width
            data = data[:, np.newaxis, :, :, :]  # Add Z dimension
            axes = 'TZCYX'
            num_channels = data.shape[2]
        else:  # time, height, width
            data = data[:, np.newaxis, :, :]  # Add Z dimension
            axes = 'TZYX'
            num_channels = 1
        
        tifffile.imwrite(output_file, data, imagej=True, metadata={
            'axes': axes,
            'channels': num_channels,
            'slices': 1,  # We're only saving one Z slice
            'frames': data.shape[0],  # Number of time points
        })
        
        print(f"Successfully converted {Path(ims_file).name} to {Path(output_file).name} "
              f"with {data.shape[0]} frames")
        
        return {
            'input_file': ims_file,
            'output_file': output_file,
            'frames': data.shape[0],
            'channels': num_channels,
            'success': True
        }
    
    except Exception as e:
        print(f"Error converting {Path(ims_file).name}: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'input_file': ims_file,
            'output_file': output_file,
            'error': str(e),
            'success': False
        }

def batch_convert_ims_to_tif(input_folder, output_folder, channels=None, z_slice=0, resolution_level=0):
    """
    Convert all IMS files in a folder to TIF files
    
    Parameters:
    -----------
    input_folder : str
        Path to folder containing IMS files
    output_folder : str
        Path to folder for output TIF files
    channels : list or None
        List of channels to include. If None, include all channels.
    z_slice : int
        Z-slice to extract
    resolution_level : int
        Resolution level to read
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Find all IMS files in the input folder
    ims_files = glob.glob(os.path.join(input_folder, "*.ims"))
    
    if not ims_files:
        print(f"No IMS files found in folder: {input_folder}")
        return []
    
    print(f"Found {len(ims_files)} IMS files in {input_folder}")
    
    # Process each IMS file
    results = []
    for ims_file in ims_files:
        base_name = Path(ims_file).stem
        output_file = os.path.join(output_folder, f"{base_name}.tif")
        
        print(f"\n{'='*50}")
        print(f"Processing file: {os.path.basename(ims_file)}")
        print(f"Output: {output_file}")
        print(f"{'='*50}")
        
        result = convert_ims_to_tif(
            ims_file,
            output_file,
            channels=channels,
            z_slice=z_slice,
            resolution_level=resolution_level
        )
        
        results.append(result)
    
    # Create summary report
    if results:
        # Count successes and failures
        successes = [r for r in results if r['success']]
        failures = [r for r in results if not r['success']]
        
        print(f"\nConversion complete!")
        print(f"Successfully converted: {len(successes)} files")
        print(f"Failed conversions: {len(failures)} files")
        
        # Create a DataFrame with the results
        import pandas as pd
        df = pd.DataFrame(results)
        summary_file = os.path.join(output_folder, "conversion_summary.csv")
        df.to_csv(summary_file, index=False)
        
        print(f"\nConversion summary saved to: {summary_file}")
        
        if failures:
            print("\nFiles that failed conversion:")
            for fail in failures:
                print(f"  {Path(fail['input_file']).name}: {fail.get('error', 'Unknown error')}")
    else:
        print("\nNo files were processed.")
        
    return results

def main():
    parser = argparse.ArgumentParser(description='Convert IMS files to TIF files')
    parser.add_argument('--input_folder', type=str, help='Folder containing IMS files')
    parser.add_argument('--output_folder', type=str, help='Folder for output TIF files')
    parser.add_argument('--channels', type=str, help='Comma-separated list of channels to include (default: all)')
    parser.add_argument('--z_slice', type=int, default=0, help='Z-slice to extract (default: 0)')
    parser.add_argument('--resolution_level', type=int, default=0, help='Resolution level (default: 0)')
    
    args = parser.parse_args()
    
    # If no command line arguments, prompt for them
    input_folder = args.input_folder
    if input_folder is None:
        input_folder = input("Enter path to folder containing IMS files: ")
    
    output_folder = args.output_folder
    if output_folder is None:
        output_folder = input("Enter output folder for TIF files: ")
    
    # Optional parameters
    if args.channels:
        channels = [int(ch.strip()) for ch in args.channels.split(',')]
    else:
        include_channels = input("Enter channels to include (comma-separated, leave blank for all): ")
        if include_channels:
            channels = [int(ch.strip()) for ch in include_channels.split(',')]
        else:
            channels = None
    
    z_slice = args.z_slice
    if args.z_slice is None:
        z_slice_input = input("Enter z-slice to extract (default: 0): ")
        z_slice = int(z_slice_input) if z_slice_input else 0
    
    resolution_level = args.resolution_level
    if args.resolution_level is None:
        resolution_input = input("Enter resolution level (default: 0): ")
        resolution_level = int(resolution_input) if resolution_input else 0
    
    print("=" * 50)
    print("IMS to TIF Converter")
    print("=" * 50)
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Channels: {channels if channels is not None else 'All'}")
    print(f"Z-slice: {z_slice}")
    print(f"Resolution level: {resolution_level}")
    print("=" * 50)
    
    batch_convert_ims_to_tif(
        input_folder,
        output_folder,
        channels=channels,
        z_slice=z_slice,
        resolution_level=resolution_level
    )

if __name__ == "__main__":
    main()