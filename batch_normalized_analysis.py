import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import datetime
from tqdm import tqdm
import sys

# Import the necessary functions from normalized_analysis.py
try:
    from normalized_analysis import segment_track_analyze_tif
except ImportError:
    print("Error: normalized_analysis.py module not found in the current directory.")
    print("Please make sure normalized_analysis.py is in the same directory as this script.")
    sys.exit(1)

def process_folder(folder_path, output_dir=None, mode='individual', 
                 ref_channel=1, meas_channel=0, min_cell_size=100,
                 max_tracking_distance=20, use_adaptive=True, 
                 adaptive_block_size=35, use_watershed=True, 
                 watershed_min_distance=10, percentile_low=0.1, 
                 percentile_high=99.9, bleach_correction=True, 
                 bleach_model='exponential', baseline_frames=None,
                 file_extension='.tif'):
    """
    Process all TIF files in a folder using the specified mode.
    
    Parameters:
    -----------
    folder_path : str
        Path to the folder containing TIF files
    output_dir : str or None
        Directory to save results (if None, creates timestamped directory)
    mode : str
        Processing mode - 'individual' or 'combined'
        - 'individual': Process each file separately and create comparative plots
        - 'combined': Process each file separately but average them as replicates
    ref_channel : int
        Index of the reference channel (default: 1)
    meas_channel : int
        Index of the measurement channel (default: 0)
    min_cell_size : int
        Minimum size of cells in pixels (default: 100)
    max_tracking_distance : float
        Maximum distance for cell tracking between frames (default: 20)
    use_adaptive : bool
        Whether to use adaptive thresholding (default: True)
    adaptive_block_size : int
        Block size for adaptive thresholding (default: 35)
    use_watershed : bool
        Whether to use watershed segmentation (default: True)
    watershed_min_distance : int
        Minimum distance between peaks for watershed (default: 10)
    percentile_low, percentile_high : float
        Percentiles for contrast enhancement (default: 0.1, 99.9)
    bleach_correction : bool
        Whether to apply bleaching correction (default: True)
    bleach_model : str
        Type of bleaching model ('exponential', 'linear', 'polynomial')
    baseline_frames : int or None
        Number of frames considered as baseline for ratio normalization
    file_extension : str
        File extension to filter by (default: '.tif')
    
    Returns:
    --------
    dict
        Dictionary containing processing results
    """
    # Create timestamp for output directory
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = os.path.basename(os.path.normpath(folder_path))
    
    if output_dir is None:
        if mode == 'individual':
            output_dir = f'batch_analysis_{folder_name}_comparison_{current_time}'
        else:
            output_dir = f'batch_analysis_{folder_name}_combined_{current_time}'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Find all matching files in the folder
    if file_extension.startswith('.'):
        file_pattern = os.path.join(folder_path, f'*{file_extension}')
    else:
        file_pattern = os.path.join(folder_path, f'*.{file_extension}')
    
    files = glob.glob(file_pattern)
    
    if not files:
        print(f"No {file_extension} files found in {folder_path}")
        return None
    
    # Sort files to ensure consistent ordering
    files.sort()
    
    print(f"Found {len(files)} {file_extension} files to process:")
    for i, file in enumerate(files):
        print(f"{i+1}. {os.path.basename(file)}")
    
    # Process each file
    all_results = []
    all_dataframes = []
    
    for i, file_path in enumerate(files):
        print(f"\n{'='*50}")
        print(f"Processing file {i+1}/{len(files)}: {os.path.basename(file_path)}")
        print(f"{'='*50}")
        
        # Create file-specific output directory inside the main output directory
        file_name = Path(file_path).stem
        file_output_dir = os.path.join(output_dir, file_name)
        
        try:
            # Process the file using the normalized_analysis function
            df_filtered, cell_tracks, master_labels = segment_track_analyze_tif(
                file_path,
                output_dir=file_output_dir,
                ref_channel=ref_channel,
                meas_channel=meas_channel,
                min_cell_size=min_cell_size,
                max_tracking_distance=max_tracking_distance,
                use_adaptive=use_adaptive,
                adaptive_block_size=adaptive_block_size,
                use_watershed=use_watershed,
                watershed_min_distance=watershed_min_distance,
                percentile_low=percentile_low,
                percentile_high=percentile_high,
                bleach_correction=bleach_correction,
                bleach_model=bleach_model,
                baseline_frames=baseline_frames
            )
            
            # Add a column to identify the source file
            df_filtered['file_name'] = file_name
            
            # Store results
            all_results.append({
                'file_path': file_path,
                'file_name': file_name,
                'dataframe': df_filtered,
                'cell_tracks': cell_tracks,
                'master_labels': master_labels
            })
            
            all_dataframes.append(df_filtered)
            
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            import traceback
            traceback.print_exc()
    
    # Combine results based on the selected mode
    if mode == 'combined' and all_dataframes:
        print("\nGenerating combined analysis...")
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        combined_df.to_csv(os.path.join(output_dir, "combined_all_data.csv"), index=False)
        
        # Generate combined plots
        create_combined_plots(combined_df, output_dir, folder_name, bleach_correction, baseline_frames)
    
    elif mode == 'individual' and all_dataframes:
        print("\nGenerating comparative analysis...")
        # Combine all dataframes for comparison
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        combined_df.to_csv(os.path.join(output_dir, "comparison_all_data.csv"), index=False)
        
        # Generate comparison plots
        create_comparison_plots(all_results, output_dir, folder_name, bleach_correction, baseline_frames)
    
    print(f"\nBatch processing complete. Results saved to: {output_dir}")
    return {
        'output_dir': output_dir,
        'all_results': all_results,
        'combined_df': combined_df if all_dataframes else None
    }

def create_combined_plots(df, output_dir, folder_name, bleach_correction=True, baseline_frames=None):
    """
    Create plots from combined data (treating files as replicates)
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Combined DataFrame with cell measurements from all files
    output_dir : str
        Directory to save plots
    folder_name : str
        Name of the source folder
    bleach_correction : bool
        Whether bleaching correction was applied
    baseline_frames : int or None
        Number of frames considered as baseline
    """
    # Group by time point
    time_groups = df.groupby('time_point')
    
    # Create ratio plot (raw or corrected)
    plt.figure(figsize=(12, 8))
    
    # Determine which ratio column to use
    if bleach_correction and 'ratio_corrected' in df.columns:
        ratio_column = 'ratio_corrected'
        title_suffix = 'Corrected'
    else:
        ratio_column = 'ratio'
        title_suffix = 'Uncorrected'
    
    # Calculate statistics
    mean_ratios = time_groups[ratio_column].mean()
    counts = time_groups[ratio_column].count()
    std_ratios = time_groups[ratio_column].std() / np.sqrt(counts)  # Standard error
    
    # Plot with error bars
    plt.errorbar(mean_ratios.index, mean_ratios, yerr=std_ratios, 
               fmt='o-', linewidth=2, capsize=4, color='blue')
    
    # Add baseline line if specified
    if baseline_frames is not None:
        plt.axvline(x=baseline_frames-0.5, color='red', linestyle='--', 
                  label='Baseline → Treatment')
        plt.legend(fontsize=10)
    
    plt.title(f'Combined Mean {title_suffix} Ratio\n({len(df["file_name"].unique())} Files)', fontsize=14)
    plt.xlabel('Time Point', fontsize=12)
    plt.ylabel(f'{title_suffix} Ratio (Measurement/Reference)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, f"{folder_name}_combined_ratio.png"))
    plt.close()
    
    # Create normalized ratio plot if baseline normalization was applied
    if baseline_frames is not None and 'ratio_normalized' in df.columns:
        plt.figure(figsize=(12, 8))
        
        # Determine which normalized ratio column to use
        if bleach_correction and 'ratio_corrected_normalized' in df.columns:
            ratio_norm_column = 'ratio_corrected_normalized'
        else:
            ratio_norm_column = 'ratio_normalized'
        
        # Calculate statistics
        mean_norm_ratios = time_groups[ratio_norm_column].mean()
        counts_norm = time_groups[ratio_norm_column].count()
        std_norm_ratios = time_groups[ratio_norm_column].std() / np.sqrt(counts_norm)  # Standard error
        
        # Plot with error bars
        plt.errorbar(mean_norm_ratios.index, mean_norm_ratios, yerr=std_norm_ratios, 
                   fmt='o-', linewidth=2, capsize=4, color='green')
        
        # Add baseline line
        plt.axvline(x=baseline_frames-0.5, color='red', linestyle='--', 
                  label='Baseline → Treatment')
        
        # Add horizontal line at y=1.0 to indicate baseline level
        plt.axhline(y=1.0, color='gray', linestyle='-', alpha=0.5)
        
        plt.title(f'Combined Normalized Ratio\n({len(df["file_name"].unique())} Files)', fontsize=14)
        plt.xlabel('Time Point', fontsize=12)
        plt.ylabel('Normalized Ratio (Relative to Baseline)', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(output_dir, f"{folder_name}_combined_normalized_ratio.png"))
        plt.close()

    # Create channel intensity plots
    plt.figure(figsize=(12, 8))
    
    # Calculate statistics for channel intensities
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
    
    # Also plot corrected reference if available
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
    
    # Add baseline line if specified
    if baseline_frames is not None:
        plt.axvline(x=baseline_frames-0.5, color='red', linestyle='--', 
                  label='Baseline → Treatment')
    
    plt.title(f'Combined Channel Intensities\n({len(df["file_name"].unique())} Files)', fontsize=14)
    plt.xlabel('Time Point', fontsize=12)
    plt.ylabel('Mean Intensity', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, f"{folder_name}_combined_channel_intensities.png"))
    plt.close()

def create_comparison_plots(all_results, output_dir, folder_name, bleach_correction=True, baseline_frames=None):
    """
    Create comparison plots across different files
    
    Parameters:
    -----------
    all_results : list
        List of dictionaries containing results for each file
    output_dir : str
        Directory to save plots
    folder_name : str
        Name of the source folder
    bleach_correction : bool
        Whether bleaching correction was applied
    baseline_frames : int or None
        Number of frames considered as baseline
    """
    # Extract file names and data frames
    file_names = [result['file_name'] for result in all_results]
    dataframes = [result['dataframe'] for result in all_results]
    
    # Determine which columns to use
    if bleach_correction and 'ratio_corrected' in dataframes[0].columns:
        ratio_column = 'ratio_corrected'
        title_suffix = 'Corrected'
    else:
        ratio_column = 'ratio'
        title_suffix = 'Uncorrected'
    
    if baseline_frames is not None:
        if bleach_correction and 'ratio_corrected_normalized' in dataframes[0].columns:
            ratio_norm_column = 'ratio_corrected_normalized'
        elif 'ratio_normalized' in dataframes[0].columns:
            ratio_norm_column = 'ratio_normalized'
        else:
            ratio_norm_column = None
    else:
        ratio_norm_column = None
    
    # Create comparison plot for raw/corrected ratios
    plt.figure(figsize=(12, 8))
    
    # Use a different color for each file
    colors = plt.cm.tab10.colors  # Get a color cycle
    
    # Plot each file's data
    for i, df in enumerate(dataframes):
        # Group by time point
        time_groups = df.groupby('time_point')
        
        # Calculate statistics
        mean_ratios = time_groups[ratio_column].mean()
        counts = time_groups[ratio_column].count()
        std_ratios = time_groups[ratio_column].std() / np.sqrt(counts)  # Standard error
        
        # Plot with error bars
        plt.errorbar(mean_ratios.index, mean_ratios, yerr=std_ratios, 
                   fmt='o-', linewidth=2, capsize=4, 
                   label=file_names[i], color=colors[i % len(colors)])
    
    # Add baseline line if specified
    if baseline_frames is not None:
        plt.axvline(x=baseline_frames-0.5, color='red', linestyle='--', 
                  label='Baseline → Treatment')
    
    plt.title(f'Comparison of {title_suffix} Ratios Across Files', fontsize=14)
    plt.xlabel('Time Point', fontsize=12)
    plt.ylabel(f'{title_suffix} Ratio (Measurement/Reference)', fontsize=12)
    plt.legend(fontsize=10, title='File')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, f"{folder_name}_comparison_ratio.png"))
    plt.close()
    
    # Create comparison plot for normalized ratios if available
    if ratio_norm_column is not None:
        plt.figure(figsize=(12, 8))
        
        # Plot each file's normalized data
        for i, df in enumerate(dataframes):
            # Group by time point
            time_groups = df.groupby('time_point')
            
            # Calculate statistics
            if ratio_norm_column in df.columns:
                mean_norm_ratios = time_groups[ratio_norm_column].mean()
                counts_norm = time_groups[ratio_norm_column].count()
                std_norm_ratios = time_groups[ratio_norm_column].std() / np.sqrt(counts_norm)
                
                # Plot with error bars
                plt.errorbar(mean_norm_ratios.index, mean_norm_ratios, yerr=std_norm_ratios, 
                           fmt='o-', linewidth=2, capsize=4, 
                           label=file_names[i], color=colors[i % len(colors)])
        
        # Add baseline line
        plt.axvline(x=baseline_frames-0.5, color='red', linestyle='--', 
                  label='Baseline → Treatment')
        
        # Add horizontal line at y=1.0 to indicate baseline level
        plt.axhline(y=1.0, color='gray', linestyle='-', alpha=0.5)
        
        plt.title(f'Comparison of Normalized Ratios Across Files', fontsize=14)
        plt.xlabel('Time Point', fontsize=12)
        plt.ylabel('Normalized Ratio (Relative to Baseline)', fontsize=12)
        plt.legend(fontsize=10, title='File')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(output_dir, f"{folder_name}_comparison_normalized_ratio.png"))
        plt.close()
    
    # Create comparison plot for reference channel
    plt.figure(figsize=(12, 8))
    
    # Plot each file's reference channel data
    for i, df in enumerate(dataframes):
        # Group by time point
        time_groups = df.groupby('time_point')
        
        # Calculate statistics
        mean_ref = time_groups['reference_intensity'].mean()
        counts_ref = time_groups['reference_intensity'].count()
        std_ref = time_groups['reference_intensity'].std() / np.sqrt(counts_ref)
        
        # Plot with error bars
        plt.errorbar(mean_ref.index, mean_ref, yerr=std_ref, 
                   fmt='o-', linewidth=2, capsize=4, 
                   label=file_names[i], color=colors[i % len(colors)])
    
    # Add baseline line if specified
    if baseline_frames is not None:
        plt.axvline(x=baseline_frames-0.5, color='red', linestyle='--', 
                  label='Baseline → Treatment')
    
    plt.title(f'Comparison of Reference Channel Intensities', fontsize=14)
    plt.xlabel('Time Point', fontsize=12)
    plt.ylabel('Mean Reference Intensity', fontsize=12)
    plt.legend(fontsize=10, title='File')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, f"{folder_name}_comparison_reference.png"))
    plt.close()
    
    # Create comparison plot for measurement channel
    plt.figure(figsize=(12, 8))
    
    # Plot each file's measurement channel data
    for i, df in enumerate(dataframes):
        # Group by time point
        time_groups = df.groupby('time_point')
        
        # Calculate statistics
        mean_meas = time_groups['measurement_intensity'].mean()
        counts_meas = time_groups['measurement_intensity'].count()
        std_meas = time_groups['measurement_intensity'].std() / np.sqrt(counts_meas)
        
        # Plot with error bars
        plt.errorbar(mean_meas.index, mean_meas, yerr=std_meas, 
                   fmt='o-', linewidth=2, capsize=4, 
                   label=file_names[i], color=colors[i % len(colors)])
    
    # Add baseline line if specified
    if baseline_frames is not None:
        plt.axvline(x=baseline_frames-0.5, color='red', linestyle='--', 
                  label='Baseline → Treatment')
    
    plt.title(f'Comparison of Measurement Channel Intensities', fontsize=14)
    plt.xlabel('Time Point', fontsize=12)
    plt.ylabel('Mean Measurement Intensity', fontsize=12)
    plt.legend(fontsize=10, title='File')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, f"{folder_name}_comparison_measurement.png"))
    plt.close()
    
    # Create a summary table
    summary_data = []
    
    for i, df in enumerate(dataframes):
        # Get number of cells tracked in this file
        num_cells = len(df['cell_id'].unique())
        
        # Calculate average ratio for the entire time course
        avg_ratio = df[ratio_column].mean()
        
        # If baseline normalization was applied, calculate average fold change
        fold_change = None
        if ratio_norm_column is not None and ratio_norm_column in df.columns:
            treatment_data = df[df['time_point'] >= baseline_frames][ratio_norm_column]
            if not treatment_data.empty:
                fold_change = treatment_data.mean()
        
        summary_data.append({
            'File': file_names[i],
            'Cells Tracked': num_cells,
            'Average Ratio': avg_ratio,
            'Fold Change': fold_change
        })
    
    # Create summary DataFrame and save to CSV
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, f"{folder_name}_summary.csv"), index=False)
    
    print(f"Created summary file: {folder_name}_summary.csv")
    print("\nSummary of analyzed files:")
    for i, row in summary_df.iterrows():
        fold_change_str = f", Fold Change: {row['Fold Change']:.3f}" if pd.notna(row['Fold Change']) else ""
        print(f"  {row['File']}: {row['Cells Tracked']} cells, Avg Ratio: {row['Average Ratio']:.3f}{fold_change_str}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Batch process TIF files for cell tracking and analysis.')
    
    # Required arguments
    parser.add_argument('folder_path', type=str, help='Path to the folder containing TIF files')
    
    # Optional arguments
    parser.add_argument('--mode', type=str, choices=['individual', 'combined'], default='individual',
                      help='Processing mode - individual: compare files, combined: average files (default: individual)')
    parser.add_argument('--output', type=str, default=None, 
                      help='Output directory (default: auto-generated)')
    parser.add_argument('--ref', type=int, default=1,
                      help='Reference channel index (default: 1)')
    parser.add_argument('--meas', type=int, default=0,
                      help='Measurement channel index (default: 0)')
    parser.add_argument('--min-size', type=int, default=100,
                      help='Minimum cell size in pixels (default: 100)')
    parser.add_argument('--max-distance', type=float, default=20,
                      help='Maximum tracking distance (default: 20)')
    parser.add_argument('--no-adaptive', action='store_false', dest='adaptive',
                      help='Disable adaptive thresholding')
    parser.add_argument('--block-size', type=int, default=35,
                      help='Block size for adaptive thresholding (default: 35)')
    parser.add_argument('--no-watershed', action='store_false', dest='watershed',
                      help='Disable watershed segmentation')
    parser.add_argument('--watershed-distance', type=int, default=10,
                      help='Minimum distance for watershed peaks (default: 10)')
    parser.add_argument('--percentile-low', type=float, default=0.1,
                      help='Lower percentile for contrast enhancement (default: 0.1)')
    parser.add_argument('--percentile-high', type=float, default=99.9,
                      help='Upper percentile for contrast enhancement (default: 99.9)')
    parser.add_argument('--no-bleach-correction', action='store_false', dest='bleach_correction',
                      help='Disable bleaching correction')
    parser.add_argument('--bleach-model', type=str, choices=['exponential', 'linear', 'polynomial'],
                      default='exponential', help='Bleaching model (default: exponential)')
    parser.add_argument('--baseline-frames', type=int, default=None,
                      help='Number of frames for baseline normalization (default: None)')
    parser.add_argument('--extension', type=str, default='.tif',
                      help='File extension to process (default: .tif)')
    
    # Set defaults for optional arguments
    parser.set_defaults(adaptive=True, watershed=True, bleach_correction=True)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run batch processing
    process_folder(
        args.folder_path,
        output_dir=args.output,
        mode=args.mode,
        ref_channel=args.ref,
        meas_channel=args.meas,
        min_cell_size=args.min_size,
        max_tracking_distance=args.max_distance,
        use_adaptive=args.adaptive,
        adaptive_block_size=args.block_size,
        use_watershed=args.watershed,
        watershed_min_distance=args.watershed_distance,
        percentile_low=args.percentile_low,
        percentile_high=args.percentile_high,
        bleach_correction=args.bleach_correction,
        bleach_model=args.bleach_model,
        baseline_frames=args.baseline_frames,
        file_extension=args.extension
    )