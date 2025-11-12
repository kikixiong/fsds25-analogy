#!/usr/bin/env python3
"""
Batch testing script for analogy testing platform
Processes CSV files with analogies and outputs results
"""

import argparse
import sys
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import ModelManager

# Try to import openpyxl for Excel support (optional dependency)
EXCEL_SUPPORT = False
try:
    import openpyxl  # type: ignore
    EXCEL_SUPPORT = True
except ImportError:
    pass


def detect_encoding(file_path):
    """Detect file encoding"""
    try:
        import chardet  # type: ignore
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            return result.get('encoding', 'utf-8')
    except ImportError:
        # If chardet is not available, try common encodings
        return None
    except Exception:
        return None


def validate_csv(input_file):
    """Validate CSV/Excel file format"""
    file_path = Path(input_file)
    file_ext = file_path.suffix.lower()
    
    df = None
    last_error = None
    
    # Try to detect if file is actually Excel (even if extension is .csv)
    # Check file signature (magic bytes)
    is_excel = False
    try:
        with open(input_file, 'rb') as f:
            header = f.read(8)
            # Excel files start with PK (ZIP signature) or D0 CF 11 E0 (OLE2 signature)
            if header.startswith(b'PK') or header.startswith(b'\xd0\xcf\x11\xe0'):
                is_excel = True
    except Exception:
        pass
    
    # Handle Excel files (by extension or detected format)
    if file_ext in ['.xlsx', '.xls'] or is_excel:
        if not EXCEL_SUPPORT:
            print("❌ Error: Excel file support requires 'openpyxl' package")
            print("   Install it with: pip install openpyxl")
            return None
        
        try:
            df = pd.read_excel(input_file, engine='openpyxl')
            if is_excel and file_ext == '.csv':
                print(f"📝 Detected Excel format (despite .csv extension): {input_file}")
            else:
                print(f"📝 Reading Excel file: {input_file}")
        except Exception as e:
            print(f"❌ Error reading Excel file: {e}")
            return None
    
    # Handle CSV files
    elif file_ext == '.csv' and not is_excel:
        # Try to detect encoding
        encodings_to_try = []
        
        # First, try to detect encoding
        detected_encoding = detect_encoding(input_file)
        if detected_encoding:
            encodings_to_try.append(detected_encoding)
        
        # Add common encodings
        encodings_to_try.extend(['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'latin-1', 'cp1252', 'iso-8859-1'])
        
        # Remove duplicates while preserving order
        encodings_to_try = list(dict.fromkeys(encodings_to_try))
        
        for encoding in encodings_to_try:
            try:
                df = pd.read_csv(input_file, encoding=encoding)
                if detected_encoding and encoding == detected_encoding:
                    print(f"📝 Detected encoding: {encoding}")
                elif encoding != 'utf-8':
                    print(f"📝 Using encoding: {encoding}")
                break
            except UnicodeDecodeError as e:
                last_error = e
                continue
            except Exception as e:
                last_error = e
                continue
        
        if df is None:
            print(f"❌ Error reading CSV file: Could not decode with any encoding")
            print(f"   Tried encodings: {', '.join(encodings_to_try)}")
            if last_error:
                print(f"   Last error: {last_error}")
            return None
    
    else:
        print(f"❌ Error: Unsupported file format: {file_ext}")
        print(f"   Supported formats: .csv, .xlsx, .xls")
        return None
    
    # Validate columns
    try:
        # Check required columns
        required_cols = ['word1', 'word2', 'word3', 'word4']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"❌ Error: File is missing required columns: {', '.join(missing_cols)}")
            print(f"   Required columns: {', '.join(required_cols)}")
            print(f"   Found columns: {', '.join(df.columns)}")
            return None
        
        # Add category column if missing
        if 'category' not in df.columns:
            df['category'] = 'unknown'
            print("⚠️  Warning: 'category' column not found, using 'unknown' for all rows")
        
        return df
    except Exception as e:
        print(f"❌ Error processing file: {e}")
        return None


def batch_test(input_file, model_type, output_file, top_n=10, verbose=False):
    """
    Batch test analogies from CSV file
    
    Args:
        input_file: Path to input CSV file
        model_type: Model type (word2vec, bert, qwen)
        output_file: Path to output CSV file
        top_n: Number of top predictions to record
        verbose: Print detailed progress
    """
    print("=" * 70)
    print("🔍 Batch Analogy Testing")
    print("=" * 70)
    print()
    
    # Validate input file
    print(f"📂 Reading input file: {input_file}")
    df = validate_csv(input_file)
    if df is None:
        return 1
    
    print(f"✅ Found {len(df)} analogies to test")
    print()
    
    # Load model
    print(f"🔄 Loading model: {model_type}")
    try:
        manager = ModelManager()
        model = manager.load_model(model_type)
        model_name = model.model_name
        print(f"✅ Model loaded: {model_name}")
        print()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return 1
    
    # Test analogies
    print("🧪 Testing analogies...")
    print("-" * 70)
    
    results = []
    success_count = 0
    error_count = 0
    
    # Use tqdm for progress bar
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing", unit="analogy"):
        word_a = str(row['word1']).strip()
        word_b = str(row['word2']).strip()
        word_c = str(row['word3']).strip()
        target = str(row['word4']).strip()
        category = row.get('category', 'unknown')
        
        if verbose:
            print(f"\n[{idx + 1}/{len(df)}] Testing: {word_a}:{word_b}::{word_c}:{target}")
        
        try:
            result = model.test_analogy(word_a, word_b, word_c, target, top_n=top_n)
            
            if result['success']:
                success_count += 1
                
                # Get top 10 predictions (pad with empty if less than 10)
                top_predictions = result['top_predictions'][:10]
                
                # Build result dict with separate columns for each prediction
                result_dict = {
                    'word1': word_a,
                    'word2': word_b,
                    'word3': word_c,
                    'word4': target,
                    'category': category,
                    'similarity': result['target_similarity'],
                    'rank': result['rank'],
                    'model_used': model_name,
                    'found_in_top_1': result.get('found_in_top_1', False),
                    'found_in_top_5': result.get('found_in_top_5', False),
                    'found_in_top_10': result.get('found_in_top_10', False),
                    'error': ''
                }
                
                # Add top 10 predictions as separate columns
                for i in range(10):
                    if i < len(top_predictions):
                        word, sim = top_predictions[i]
                        result_dict[f'top_{i+1}_word'] = word
                        result_dict[f'top_{i+1}_similarity'] = sim
                    else:
                        result_dict[f'top_{i+1}_word'] = ''
                        result_dict[f'top_{i+1}_similarity'] = ''
                
                results.append(result_dict)
                
                if verbose:
                    rank_str = f"#{result['rank']}" if result['rank'] else "N/A"
                    print(f"  ✅ Rank: {rank_str}, Similarity: {result['target_similarity']:.4f}")
            else:
                error_count += 1
                
                # Build error result dict with empty prediction columns
                error_dict = {
                    'word1': word_a,
                    'word2': word_b,
                    'word3': word_c,
                    'word4': target,
                    'category': category,
                    'similarity': None,
                    'rank': None,
                    'model_used': model_name,
                    'found_in_top_1': False,
                    'found_in_top_5': False,
                    'found_in_top_10': False,
                    'error': result.get('error', 'Unknown error')
                }
                
                # Add empty prediction columns
                for i in range(10):
                    error_dict[f'top_{i+1}_word'] = ''
                    error_dict[f'top_{i+1}_similarity'] = ''
                
                results.append(error_dict)
                
                if verbose:
                    print(f"  ❌ Error: {result.get('error', 'Unknown error')}")
        
        except Exception as e:
            error_count += 1
            error_msg = str(e)
            
            # Build exception result dict with empty prediction columns
            exception_dict = {
                'word1': word_a,
                'word2': word_b,
                'word3': word_c,
                'word4': target,
                'category': category,
                'similarity': None,
                'rank': None,
                'model_used': model_name,
                'found_in_top_1': False,
                'found_in_top_5': False,
                'found_in_top_10': False,
                'error': error_msg
            }
            
            # Add empty prediction columns
            for i in range(10):
                exception_dict[f'top_{i+1}_word'] = ''
                exception_dict[f'top_{i+1}_similarity'] = ''
            
            results.append(exception_dict)
            
            if verbose:
                print(f"  ❌ Exception: {error_msg}")
    
    print()
    print("-" * 70)
    
    # Create output dataframe
    output_df = pd.DataFrame(results)
    
    # Reorder columns for better readability
    # Basic columns first, then prediction columns in order
    basic_columns = ['word1', 'word2', 'word3', 'word4', 'category', 
                     'similarity', 'rank', 'model_used',
                     'found_in_top_1', 'found_in_top_5', 'found_in_top_10', 'error']
    
    # Prediction columns in order (top_1_word, top_1_similarity, top_2_word, ...)
    prediction_columns = []
    for i in range(1, 11):
        prediction_columns.extend([f'top_{i}_word', f'top_{i}_similarity'])
    
    # Create ordered column list (only include columns that exist)
    ordered_columns = [col for col in basic_columns + prediction_columns if col in output_df.columns]
    
    # Reorder DataFrame columns
    output_df = output_df[ordered_columns]
    
    # Save results
    print(f"💾 Saving results to: {output_file}")
    output_dir = os.path.dirname(output_file) if os.path.dirname(output_file) else '.'
    os.makedirs(output_dir, exist_ok=True)
    output_df.to_csv(output_file, index=False)
    print(f"✅ Results saved!")
    print()
    
    # Print summary statistics
    print("=" * 70)
    print("📊 Summary Statistics")
    print("=" * 70)
    print(f"Model Used:        {model_name}")
    print(f"Total Tests:       {len(df)}")
    print(f"Successful:        {success_count} ({success_count/len(df)*100:.1f}%)")
    print(f"Errors:            {error_count} ({error_count/len(df)*100:.1f}%)")
    print()
    
    if success_count > 0:
        successful_df = output_df[output_df['rank'].notna()]
        top1_count = successful_df['found_in_top_1'].sum()
        top5_count = successful_df['found_in_top_5'].sum()
        top10_count = successful_df['found_in_top_10'].sum()
        
        print("Success Rate:")
        print(f"  Top 1:           {top1_count}/{len(df)} ({top1_count/len(df)*100:.1f}%)")
        print(f"  Top 5:           {top5_count}/{len(df)} ({top5_count/len(df)*100:.1f}%)")
        print(f"  Top 10:          {top10_count}/{len(df)} ({top10_count/len(df)*100:.1f}%)")
        print()
        print(f"Average Similarity: {successful_df['similarity'].mean():.3f}")
        print(f"Median Rank:        {successful_df['rank'].median():.0f}")
        print()
        
        # Category breakdown
        if 'category' in output_df.columns and output_df['category'].nunique() > 1:
            print("Category Breakdown:")
            category_stats = output_df.groupby('category').agg({
                'rank': ['count', lambda x: (x == 1).sum(), lambda x: x.median()],
                'similarity': 'mean'
            }).round(3)
            category_stats.columns = ['Total', 'Top1', 'Median Rank', 'Avg Similarity']
            print(category_stats.to_string())
            print()
    
    print("=" * 70)
    print(f"✅ Batch testing complete! Results saved to: {output_file}")
    print("=" * 70)
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Batch test analogies from CSV file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python batch_test.py input.csv --model word2vec --output results.csv
  
  # Use BERT model
  python batch_test.py input.csv --model bert --output results.csv
  
  # Use QWEN model with verbose output
  python batch_test.py input.csv --model qwen --output results.csv --verbose
  
  # Specify top N predictions
  python batch_test.py input.csv --model word2vec --output results.csv --top-n 20

CSV Format:
  The input CSV file must have the following columns:
  - word1, word2, word3, word4 (required)
  - category (optional)
  
  Example:
    word1,word2,word3,word4,category
    man,woman,king,queen,gender
    Paris,France,London,England,geography
    good,better,bad,worse,grammar
        """
    )
    
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to input CSV file'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='word2vec',
        choices=['word2vec', 'bert', 'qwen'],
        help='Model type to use (default: word2vec)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to output CSV file (default: <input_file>_results.csv)'
    )
    
    parser.add_argument(
        '--top-n',
        type=int,
        default=10,
        help='Number of top predictions to record (default: 10)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress for each analogy'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"❌ Error: Input file not found: {args.input_file}")
        return 1
    
    # Set output file
    if args.output is None:
        input_path = Path(args.input_file)
        output_file = input_path.parent / f"{input_path.stem}_results.csv"
    else:
        output_file = args.output
    
    # Run batch testing
    return batch_test(
        input_file=args.input_file,
        model_type=args.model,
        output_file=str(output_file),
        top_n=args.top_n,
        verbose=args.verbose
    )


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Batch testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

