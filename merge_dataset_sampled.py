"""
Sample-limited script to merge CIC IoT-DIAD 2024 dataset CSV files
Limits samples per attack type to avoid memory issues
"""

import pandas as pd
import glob
import os
from pathlib import Path
import sys

def merge_dataset_sampled(dataset_dir="data/ciciot_idad_2024", 
                          output_file="data/merged_flow_dataset.csv",
                          sample_fraction=0.1,
                          random_state=42):
    """
    Merge CSV files with percentage sampling per attack type
    
    Args:
        dataset_dir: Path to the dataset directory
        output_file: Output file path
        sample_fraction: Fraction of samples to keep per class (default: 0.1 = 10%)
        random_state: Random seed for reproducibility
    """
    
    print("="*80)
    print("SAMPLE-LIMITED MERGING - CIC IoT-DIAD 2024 DATASET")
    print("="*80)
    print(f"\nSampling: {sample_fraction*100:.1f}% of each attack class")
    print(f"Random seed: {random_state}")
    
    if not os.path.exists(dataset_dir):
        print(f"\nError: Dataset directory not found: {dataset_dir}")
        sys.exit(1)
    
    attack_dirs = ['Benign', 'BruteForce', 'DDoS', 'DoS', 'Mirai', 'Recon', 'Spoofing', 'Web-Based']
    
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"Removed existing file: {output_file}\n")
    
    all_dataframes = []
    total_samples = 0
    attack_summary = {}
    
    for attack_type in attack_dirs:
        attack_path = os.path.join(dataset_dir, attack_type)
        
        if not os.path.exists(attack_path):
            print(f"\nWarning: {attack_type} directory not found, skipping...")
            continue
        
        csv_files = glob.glob(os.path.join(attack_path, '**/*.csv'), recursive=True)
        
        if len(csv_files) == 0:
            print(f"\nWarning: No CSV files found in {attack_type}, skipping...")
            continue
        
        print(f"\n{attack_type}:")
        print(f"  Found {len(csv_files)} file(s)")
        
        attack_dfs = []
        attack_samples = 0
        
        for csv_file in csv_files:
            try:
                file_name = os.path.basename(csv_file)
                print(f"  Reading {file_name}...", end=' ', flush=True)
                
                df = pd.read_csv(csv_file)
                
                if 'Label' not in df.columns and 'label' not in df.columns:
                    df['Label'] = attack_type
                elif 'label' in df.columns:
                    df.rename(columns={'label': 'Label'}, inplace=True)
                
                attack_dfs.append(df)
                attack_samples += len(df)
                
                print(f"{len(df):,} samples")
                
            except Exception as e:
                print(f"Error: {e}")
                continue
        
        if len(attack_dfs) == 0:
            print(f"  No valid data loaded for {attack_type}")
            continue
        
        print(f"\n  Combining {len(attack_dfs)} file(s) for {attack_type}...")
        combined_df = pd.concat(attack_dfs, ignore_index=True)
        
        print(f"  Total available: {len(combined_df):,} samples")
        
        n_samples = max(1, int(len(combined_df) * sample_fraction))
        print(f"  Sampling {sample_fraction*100:.1f}% = {n_samples:,} samples...")
        
        sampled_df = combined_df.sample(n=n_samples, random_state=random_state)
        all_dataframes.append(sampled_df)
        attack_summary[attack_type] = n_samples
        total_samples += n_samples
        print(f"  Kept: {n_samples:,} samples")
    
    if len(all_dataframes) == 0:
        print("\nError: No data was processed!")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("MERGING ALL DATAFRAMES...")
    print("="*80)
    
    try:
        merged_df = pd.concat(all_dataframes, ignore_index=True)
    except Exception as e:
        print(f"\nError during merge: {e}")
        sys.exit(1)
    
    print(f"\nMerge successful!")
    print(f"Total samples: {len(merged_df):,}")
    print(f"Total features: {len(merged_df.columns)}")
    
    print(f"\nColumn names ({len(merged_df.columns)} total):")
    print("  First 10:", merged_df.columns.tolist()[:10])
    if len(merged_df.columns) > 10:
        print("  Last 5:", merged_df.columns.tolist()[-5:])
    
    if 'Label' in merged_df.columns:
        print(f"\nLabel distribution:")
        print("-" * 40)
        label_counts = merged_df['Label'].value_counts()
        for label, count in sorted(label_counts.items()):
            percentage = (count / len(merged_df)) * 100
            print(f"  {label:<15} {count:>10,} ({percentage:>5.2f}%)")
        print("-" * 40)
        print(f"  {'TOTAL':<15} {len(merged_df):>10,} (100.00%)")
    
    print(f"\nMissing values analysis:")
    missing_counts = merged_df.isnull().sum()
    cols_with_missing = missing_counts[missing_counts > 0]
    
    if len(cols_with_missing) > 0:
        print(f"  Columns with missing values: {len(cols_with_missing)}")
        missing_percentage = (missing_counts / len(merged_df) * 100)
        high_missing = missing_percentage[missing_percentage > 50].sort_values(ascending=False)
        
        if len(high_missing) > 0:
            print(f"  Columns with >50% missing: {len(high_missing)}")
            print("  (These will be handled by the preprocessing pipeline)")
    else:
        print("  No missing values found!")
    
    memory_mb = merged_df.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"\nMemory usage: {memory_mb:.2f} MB")
    
    print(f"\n{'='*80}")
    print(f"SAVING MERGED DATASET")
    print(f"{'='*80}")
    
    print(f"\nSaving to: {output_file}")
    print("This may take a few minutes...")
    
    try:
        merged_df.to_csv(output_file, index=False)
        file_size = os.path.getsize(output_file) / (1024 * 1024)
        
        print(f"\nSuccessfully saved!")
        print(f"  File: {os.path.abspath(output_file)}")
        print(f"  Size: {file_size:.2f} MB")
        
    except Exception as e:
        print(f"\nError saving file: {e}")
        sys.exit(1)
    
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
    print(f"\n2. Start training:")
    print(f"   python main.py --data {output_file}")
    print(f"\n3. Quick test (no optimization):")
    print(f"   python main.py --data {output_file} --no-optimize")
    print()
    
    return merged_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Sample-limited merge of CIC IoT-DIAD 2024 dataset')
    parser.add_argument('--dataset-dir', type=str, default='data/ciciot_idad_2024',
                       help='Dataset directory (default: data/ciciot_idad_2024)')
    parser.add_argument('--output', type=str, default='data/merged_flow_dataset.csv',
                       help='Output file (default: data/merged_flow_dataset.csv)')
    parser.add_argument('--sample-fraction', type=float, default=0.1,
                       help='Fraction of samples to keep per class (default: 0.1 = 10%%)')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    merge_dataset_sampled(args.dataset_dir, args.output, args.sample_fraction, args.random_state)
