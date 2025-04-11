import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use Agg backend
import matplotlib.pyplot as plt

def view_ims_data(file_path, resolution_level=0, time_point=0, channel=0, z_slice=0):
    """
    Save an image slice from the IMS file to disk
    """
    with h5py.File(file_path, 'r') as f:
        data_path = f'DataSet/ResolutionLevel {resolution_level}/TimePoint {time_point}/Channel {channel}/Data'
        data = f[data_path][:]
        image = data[z_slice, :, :]
        
        plt.figure(figsize=(10, 10))
        plt.imshow(image, cmap='gray')
        plt.colorbar()
        plt.title(f'Resolution Level {resolution_level}, Channel {channel}, Z-slice {z_slice}')
        
        # Save instead of show
        plt.savefig(f'single_channel_r{resolution_level}_c{channel}_z{z_slice}.png')
        plt.close()  # Close the figure to free memory
        
        return image

def view_both_channels(file_path, resolution_level=0, time_point=0, z_slice=0):
    """
    Save both channels side by side to disk
    """
    plt.figure(figsize=(20, 10))
    
    with h5py.File(file_path, 'r') as f:
        for channel in [0, 1]:
            data_path = f'DataSet/ResolutionLevel {resolution_level}/TimePoint {time_point}/Channel {channel}/Data'
            data = f[data_path][:]
            image = data[z_slice, :, :]
            
            plt.subplot(1, 2, channel + 1)
            plt.imshow(image, cmap='gray')
            plt.colorbar()
            plt.title(f'Channel {channel}')
    
    plt.suptitle(f'Resolution Level {resolution_level}, Z-slice {z_slice}')
    # Save instead of show
    plt.savefig(f'both_channels_r{resolution_level}_z{z_slice}.png')
    plt.close()

# Example usage - replace with your file path
file_path = 'data/control.ims'
view_ims_data(file_path)
view_both_channels(file_path)