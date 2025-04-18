import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# File paths - update these to your actual file paths
ex_file_path = "ex_channel_ratio_analysis.csv"
ta_file_path = "ta_channel_ratio_analysis.csv"
output_dir = "combined_plots"

# Create output directory if it doesn't exist
Path(output_dir).mkdir(exist_ok=True, parents=True)

# Load the data
print(f"Loading data from {ex_file_path} and {ta_file_path}")
ex_data = pd.read_csv(ex_file_path)
ta_data = pd.read_csv(ta_file_path)

# Add a column to identify the source
ex_data['Source'] = 'EX'
ta_data['Source'] = 'TA'

# Check if 'Normalized_Ratio' column exists in both dataframes
if 'Normalized_Ratio' not in ex_data.columns or 'Normalized_Ratio' not in ta_data.columns:
    print("Warning: 'Normalized_Ratio' column not found in one or both files.")
    print(f"Columns in {ex_file_path}: {ex_data.columns.tolist()}")
    print(f"Columns in {ta_file_path}: {ta_data.columns.tolist()}")
    
    # Try to calculate normalized ratios if they don't exist but we have the raw ratios
    if 'Ratio_Ch1_Ch2' in ex_data.columns:
        print("Calculating normalized ratios for EX data...")
        ex_ratios = ex_data['Ratio_Ch1_Ch2'].values
        ex_baseline = np.nanmean(ex_ratios[:15])
        ex_data['Normalized_Ratio'] = ex_ratios / ex_baseline
        print(f"EX data normalized to baseline: {ex_baseline:.4f}")
    
    if 'Ratio_Ch1_Ch2' in ta_data.columns:
        print("Calculating normalized ratios for TA data...")
        ta_ratios = ta_data['Ratio_Ch1_Ch2'].values
        ta_baseline = np.nanmean(ta_ratios[:15])
        ta_data['Normalized_Ratio'] = ta_ratios / ta_baseline
        print(f"TA data normalized to baseline: {ta_baseline:.4f}")

# Combine the data
combined_data = pd.concat([ex_data, ta_data], ignore_index=True)
print(f"Combined data shape: {combined_data.shape}")

# Create the plot
plt.figure(figsize=(12, 8))

# Plot each dataset
for source, color, marker in [('EX', 'blue', 'o'), ('TA', 'red', 's')]:
    source_data = combined_data[combined_data['Source'] == source]
    plt.plot(source_data['Time_Point'], source_data['Normalized_Ratio'], 
             marker=marker, linestyle='-', color=color, label=source,
             markersize=6, linewidth=2, alpha=0.8)

# Add reference line at y=1.0
plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)

# Set axis limits
plt.ylim(0.95, 1.05)

# Add labels and title
plt.xlabel('Time Point', fontsize=12)
plt.ylabel('Normalized Ratio', fontsize=12)
plt.title('Comparison of Normalized Channel Ratios (EX vs TA)', fontsize=14)

# Add grid and legend
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)

# Save the plot
output_path = f"{output_dir}/combined_normalized_ratios.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to {output_path}")

# Show the plot
plt.show()

# Optional: Create a detailed plot showing both datasets side by side
plt.figure(figsize=(16, 10))

# Create two subplots
plt.subplot(2, 1, 1)
for source, color, marker in [('EX', 'blue', 'o'), ('TA', 'red', 's')]:
    source_data = combined_data[combined_data['Source'] == source]
    plt.plot(source_data['Time_Point'], source_data['Normalized_Ratio'], 
             marker=marker, linestyle='-', color=color, label=source,
             markersize=4, linewidth=1.5, alpha=0.8)
plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
plt.ylim(0.95, 1.05)
plt.xlabel('Time Point', fontsize=12)
plt.ylabel('Normalized Ratio', fontsize=12)
plt.title('Combined View of Normalized Ratios', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)

# Now add individual plots for clearer viewing
plt.subplot(2, 2, 3)
ex_only = combined_data[combined_data['Source'] == 'EX']
plt.plot(ex_only['Time_Point'], ex_only['Normalized_Ratio'], 
         'o-', color='blue', markersize=4, linewidth=1.5)
plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
plt.ylim(0.95, 1.05)
plt.xlabel('Time Point', fontsize=12)
plt.ylabel('Normalized Ratio', fontsize=12)
plt.title('EX Normalized Ratio', fontsize=14)
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
ta_only = combined_data[combined_data['Source'] == 'TA']
plt.plot(ta_only['Time_Point'], ta_only['Normalized_Ratio'], 
         's-', color='red', markersize=4, linewidth=1.5)
plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
plt.ylim(0.95, 1.05)
plt.xlabel('Time Point', fontsize=12)
plt.ylabel('Normalized Ratio', fontsize=12)
plt.title('TA Normalized Ratio', fontsize=14)
plt.grid(True, alpha=0.3)

plt.tight_layout()

# Save the detailed plot
detailed_output_path = f"{output_dir}/detailed_normalized_ratios.png"
plt.savefig(detailed_output_path, dpi=300, bbox_inches='tight')
print(f"Detailed plot saved to {detailed_output_path}")

# Show the detailed plot
plt.show()

# Optional: Save the combined data to a CSV
combined_csv_path = f"{output_dir}/combined_channel_ratios.csv"
combined_data.to_csv(combined_csv_path, index=False)
print(f"Combined data saved to {combined_csv_path}")