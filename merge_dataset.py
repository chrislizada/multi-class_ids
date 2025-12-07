"""
Script to merge CIC IoT-DIAD 2024 dataset CSV files
Combines all attack type subdirectories into a single dataset
"""

import pandas as pd
import glob
import os
from pathlib import Path
import sys

def merge_dataset(dataset_dir="data/ciciot_idad_2024", output_file="data/merged_flow_dataset.csv"):
    """
    Merge all CSV files from CIC IoT-DIAD 2024 dataset
    
    Args:
        dataset_dir: Path to the dataset directory containing attack subdirectories
        output_file: Path where merged dataset will be saved
    """
    
    print("="*80)
    print("MERGING CIC IoT-DIAD 2024 DATASET")
    print("="*80)
    
    # Check if dataset directory exists
    if not os.path.exists(dataset_dir):
        print(f"\nError: Dataset directory not found: {dataset_dir}")
        print("Please download the dataset first!")
        sys.exit(1)
    
    # Attack type directories
    attack_dirs = ['Benign', 'BruteForce', 'DDoS', 'DoS', 'Mirai', 'Recon', 'Spoofing', 'Web-Based']
    
    all_dataframes = []
    total_samples = 0
    attack_summary = {}
    
    # Process each attack type directory
    for attack_type in attack_dirs:
        attack_path = os.path.join(dataset_dir, attack_type)
        
        if not os.path.exists(attack_path):
            print(f"\nWarning: {attack_type} directory not found, skipping...")
            continue
        
        # Find all CSV files in this attack type directory (recursive)
        csv_files = glob.glob(os.path.join(attack_path, '**/*.csv'), recursive=True)
        
        if len(csv_files) == 0:
            print(f"\nWarning: No CSV files found in {attack_type}, skipping...")
            continue
        
        print(f"\n{attack_type}:")
        print(f"  Found {len(csv_files)} CSV file(s)")
        
        attack_samples = 0
        attack_dfs = []
        
        for csv_file in csv_files:
            try:
                # Read CSV file
                df = pd.read_csv(csv_file)
                
                # Handle label column
                if 'Label' not in df.columns and 'label' not in df.columns:
                    # If no label column, add the attack type as label
                    df['Label'] = attack_type
                    print(f"    Adding Label column with value: {attack_type}")
                elif 'label' in df.columns:
                    # Rename to uppercase Label for consistency
                    df.rename(columns={'label': 'Label'}, inplace=True)
                
                # Ensure Label matches directory name for consistency
                # (in case CSV has different labels)
                if 'Label' in df.columns:
                    # Only override if all labels are the same or empty
                    unique_labels = df['Label'].unique()
                    if len(unique_labels) == 1 and pd.isna(unique_labels[0]):
                        df['Label'] = attack_type
                
                attack_dfs.append(df)
                attack_samples += len(df)
                
                file_name = os.path.basename(csv_file)
                print(f"    ✓ {file_name}: {len(df):,} samples, {len(df.columns)} columns")
                
            except Exception as e:
                print(f"    ✗ Error reading {csv_file}: {e}")
                continue
        
        if len(attack_dfs) > 0:
            all_dataframes.extend(attack_dfs)
            total_samples += attack_samples
            attack_summary[attack_type] = attack_samples
            print(f"  Subtotal for {attack_type}: {attack_samples:,} samples")
        else:
            print(f"  No valid data loaded for {attack_type}")
    
    # Check if we have any data
    if len(all_dataframes) == 0:
        print("\nError: No CSV files were successfully loaded!")
        print("Please check the dataset directory structure.")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("MERGING ALL DATAFRAMES...")
    print("="*80)
    
    # Concatenate all dataframes
    try:
        merged_df = pd.concat(all_dataframes, ignore_index=True)
    except Exception as e:
        print(f"\nError during merge: {e}")
        print("This might be due to inconsistent column names across files.")
        sys.exit(1)
    
    print(f"\nMerge successful!")
    print(f"Total samples: {len(merged_df):,}")
    print(f"Total features: {len(merged_df.columns)}")
    
    # Display column names
    print(f"\nColumn names ({len(merged_df.columns)} total):")
    print("  First 10:", merged_df.columns.tolist()[:10])
    if len(merged_df.columns) > 10:
        print("  Last 5:", merged_df.columns.tolist()[-5:])
    
    # Check Label column
    if 'Label' not in merged_df.columns:
        print("\nWarning: 'Label' column not found!")
        print("Available columns:", merged_df.columns.tolist())
    else:
        print(f"\nLabel distribution:")
        print("-" * 40)
        label_counts = merged_df['Label'].value_counts()
        for label, count in label_counts.items():
            percentage = (count / len(merged_df)) * 100
            print(f"  {label:<15} {count:>10,} ({percentage:>5.2f}%)")
        print("-" * 40)
        print(f"  {'TOTAL':<15} {len(merged_df):>10,} (100.00%)")
    
    # Check for missing values
    print(f"\nMissing values analysis:")
    missing_counts = merged_df.isnull().sum()
    cols_with_missing = missing_counts[missing_counts > 0]
    
    if len(cols_with_missing) > 0:
        print(f"  Columns with missing values: {len(cols_with_missing)}")
        
        # Show columns with high missing percentage
        missing_percentage = (missing_counts / len(merged_df) * 100)
        high_missing = missing_percentage[missing_percentage > 50].sort_values(ascending=False)
        
        if len(high_missing) > 0:
            print(f"\n  Columns with >50% missing values: {len(high_missing)}")
            print("  (These will be handled by the preprocessing pipeline)")
            if len(high_missing) <= 10:
                for col, pct in high_missing.items():
                    print(f"    - {col}: {pct:.1f}% missing")
    else:
        print("  No missing values found!")
    
    # Data type summary
    print(f"\nData type summary:")
    dtype_counts = merged_df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count} columns")
    
    # Memory usage
    memory_mb = merged_df.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"\nMemory usage: {memory_mb:.2f} MB")
    
    # Save merged dataset
    print(f"\n{'='*80}")
    print(f"SAVING MERGED DATASET")
    print(f"{'='*80}")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")
    
    print(f"\nSaving to: {output_file}")
    print("This may take a few minutes for large datasets...")
    
    try:
        merged_df.to_csv(output_file, index=False)
        file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
        
        print(f"\n✓ Successfully saved!")
        print(f"  File: {os.path.abspath(output_file)}")
        print(f"  Size: {file_size:.2f} MB")
        
    except Exception as e:
        print(f"\nError saving file: {e}")
        sys.exit(1)
    
    # Final summary
    print("\n" + "="*80)
    print("DATASET SUMMARY")
    print("="*80)
    print(f"\nAttack Types Processed: {len(attack_summary)}")
    for attack, count in sorted(attack_summary.items(), key=lambda x: x[1], reverse=True):
        print(f"  {attack:<15} {count:>10,} samples")
    
    print(f"\nTotal Samples:  {len(merged_df):,}")
    print(f"Total Features: {len(merged_df.columns)}")
    print(f"Output File:    {output_file}")
    
    print("\n" + "="*80)
    print("DATASET READY FOR TRAINING!")
    print("="*80)
    print(f"\nNext steps:")
    print(f"1. Verify the dataset:")
    print(f"   python -c \"import pandas as pd; df=pd.read_csv('{output_file}'); print(df.info())\"")
    print(f"\n2. Start training with full pipeline:")
    print(f"   python main.py --data {output_file}")
    print(f"\n3. Or quick test (no optimization):")
    print(f"   python main.py --data {output_file} --no-optimize")
    print()
    
    return merged_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Merge CIC IoT-DIAD 2024 dataset CSV files')
    parser.add_argument('--dataset-dir', type=str, default='data/ciciot_idad_2024',
                       help='Path to dataset directory (default: data/ciciot_idad_2024)')
    parser.add_argument('--output', type=str, default='data/merged_flow_dataset.csv',
                       help='Output file path (default: data/merged_flow_dataset.csv)')
    
    args = parser.parse_args()
    
    # Run merge
    merge_dataset(args.dataset_dir, args.output)
