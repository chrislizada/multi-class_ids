"""
Merge script for CIC IoT-DIAD 2024 Packet-Based Dataset
Handles flexible directory structure (both flat and nested)
Supports all attack types with 5% sampling per file
"""

import pandas as pd
import glob
import os
from pathlib import Path
import sys

def merge_packet_dataset(dataset_dir="data/ciciot_idad_2024_packet", 
                         output_file="data/merged_packet_dataset.csv",
                         sample_fraction=0.05,
                         random_state=42):
    """
    Merge packet-based CSV files with percentage sampling
    
    Args:
        dataset_dir: Path to the packet-based dataset directory
        output_file: Output file path
        sample_fraction: Fraction of samples to keep per file (default: 0.05 = 5%)
        random_state: Random seed for reproducibility
    """
    
    print("="*80)
    print("MERGING PACKET-BASED DATASET - CIC IoT-DIAD 2024")
    print("="*80)
    print(f"\nDataset directory: {dataset_dir}")
    print(f"Sampling: {sample_fraction*100:.1f}% per file")
    print(f"Random seed: {random_state}")
    
    if not os.path.exists(dataset_dir):
        print(f"\nError: Dataset directory not found: {dataset_dir}")
        sys.exit(1)
    
    # Find all CSV files (both in root and subdirectories)
    csv_files = []
    csv_files.extend(glob.glob(os.path.join(dataset_dir, "*.csv")))
    csv_files.extend(glob.glob(os.path.join(dataset_dir, "*/*.csv")))
    csv_files.extend(glob.glob(os.path.join(dataset_dir, "*/*/*.csv")))
    
    # Remove duplicates
    csv_files = list(set(csv_files))
    
    if len(csv_files) == 0:
        print(f"\nError: No CSV files found in {dataset_dir}")
        sys.exit(1)
    
    print(f"\nFound {len(csv_files)} CSV files\n")
    
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"Removed existing file: {output_file}\n")
    
    all_dataframes = []
    total_original = 0
    total_sampled = 0
    file_summary = {}
    
    for i, csv_file in enumerate(csv_files, 1):
        try:
            file_name = os.path.basename(csv_file)
            rel_path = os.path.relpath(csv_file, dataset_dir)
            
            print(f"[{i}/{len(csv_files)}] Processing: {rel_path}")
            
            df = pd.read_csv(csv_file)
            
            if len(df) == 0:
                print(f"  ⚠ Skipping (0 samples)\n")
                continue
            
            original_count = len(df)
            
            # Sample 5% from this file
            n_samples = max(1, int(len(df) * sample_fraction))
            sampled = df.sample(n=n_samples, random_state=random_state)
            
            all_dataframes.append(sampled)
            total_original += original_count
            total_sampled += len(sampled)
            
            # Track by attack type (from filename or directory)
            attack_type = file_name.replace('.csv', '')
            if attack_type not in file_summary:
                file_summary[attack_type] = {'files': 0, 'samples': 0}
            file_summary[attack_type]['files'] += 1
            file_summary[attack_type]['samples'] += len(sampled)
            
            print(f"  Sampled: {len(sampled):,} from {original_count:,} rows")
            
            # Show label distribution if available
            label_col = None
            if 'Label' in df.columns:
                label_col = 'Label'
            elif 'label' in df.columns:
                label_col = 'label'
            
            if label_col:
                unique_labels = df[label_col].unique()
                if len(unique_labels) <= 3:
                    print(f"  Labels: {', '.join(map(str, unique_labels))}")
            
            print()
            
        except Exception as e:
            print(f"  ✗ Error: {e}\n")
            continue
    
    if len(all_dataframes) == 0:
        print("\nError: No data was processed!")
        sys.exit(1)
    
    print("="*80)
    print("MERGING ALL DATAFRAMES...")
    print("="*80)
    
    try:
        merged_df = pd.concat(all_dataframes, ignore_index=True)
    except Exception as e:
        print(f"\nError during merge: {e}")
        sys.exit(1)
    
    print(f"\n✓ Merge successful!")
    print(f"  Original samples: {total_original:,}")
    print(f"  Sampled: {total_sampled:,} ({sample_fraction*100:.1f}%)")
    print(f"  Features: {len(merged_df.columns)}")
    
    # Analyze columns
    print(f"\nColumn analysis:")
    print(f"  Total columns: {len(merged_df.columns)}")
    print(f"  First 10: {merged_df.columns.tolist()[:10]}")
    if len(merged_df.columns) > 10:
        print(f"  Last 5: {merged_df.columns.tolist()[-5:]}")
    
    # Label distribution
    label_col = None
    if 'Label' in merged_df.columns:
        label_col = 'Label'
    elif 'label' in merged_df.columns:
        label_col = 'label'
    
    if label_col:
        print(f"\n{label_col} distribution:")
        print("-" * 50)
        label_counts = merged_df[label_col].value_counts()
        for label, count in label_counts.items():
            percentage = (count / len(merged_df)) * 100
            print(f"  {str(label):<30} {count:>8,} ({percentage:>5.2f}%)")
        print("-" * 50)
        print(f"  {'TOTAL':<30} {len(merged_df):>8,} (100.00%)")
        print(f"\n  Unique classes: {len(label_counts)}")
    
    # Missing values
    print(f"\nMissing values:")
    missing_counts = merged_df.isnull().sum()
    cols_with_missing = missing_counts[missing_counts > 0]
    
    if len(cols_with_missing) > 0:
        print(f"  Columns with missing: {len(cols_with_missing)}")
        total_missing = missing_counts.sum()
        print(f"  Total missing cells: {total_missing:,}")
    else:
        print("  ✓ No missing values!")
    
    # Memory usage
    memory_mb = merged_df.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"\nMemory usage: {memory_mb:.2f} MB")
    
    # Save
    print(f"\n{'='*80}")
    print(f"SAVING MERGED DATASET")
    print(f"{'='*80}")
    
    print(f"\nSaving to: {output_file}")
    print("This may take a few minutes...")
    
    try:
        merged_df.to_csv(output_file, index=False)
        file_size = os.path.getsize(output_file) / (1024 * 1024)
        
        print(f"\n✓ Successfully saved!")
        print(f"  File: {os.path.abspath(output_file)}")
        print(f"  Size: {file_size:.2f} MB")
        
    except Exception as e:
        print(f"\n✗ Error saving file: {e}")
        sys.exit(1)
    
    # Summary
    print("\n" + "="*80)
    print("DATASET SUMMARY")
    print("="*80)
    print(f"\nFiles processed: {len(csv_files)}")
    print(f"Total samples: {len(merged_df):,}")
    print(f"Total features: {len(merged_df.columns)}")
    if label_col:
        print(f"Unique classes: {merged_df[label_col].nunique()}")
    print(f"\nOutput: {output_file}")
    
    print("\n" + "="*80)
    print("✓ DATASET READY FOR TRAINING!")
    print("="*80)
    print(f"\nNext steps:")
    print(f"1. Train without SMOTE (recommended for packet-based):")
    print(f"   python main.py --data {output_file} --no-optimize --no-smote")
    print(f"\n2. Train with optimization (slower):")
    print(f"   python main.py --data {output_file} --no-smote")
    print()
    
    return merged_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Merge CIC IoT-DIAD 2024 Packet-Based Dataset')
    parser.add_argument('--dataset-dir', type=str, default='data/ciciot_idad_2024_packet',
                       help='Packet-based dataset directory')
    parser.add_argument('--output', type=str, default='data/merged_packet_dataset.csv',
                       help='Output merged CSV file')
    parser.add_argument('--sample-fraction', type=float, default=0.05,
                       help='Fraction to sample per file (default: 0.05 = 5%%)')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    merge_packet_dataset(args.dataset_dir, args.output, args.sample_fraction, args.random_state)
