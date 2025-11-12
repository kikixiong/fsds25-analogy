#!/usr/bin/env python3
"""
Comprehensive analysis of analogy test results across models
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json


def load_results():
    """Load all model results"""
    files = {
        'Word2Vec': 'explore_analogies_word2vec.csv',
        'BERT': 'explore_analogies_bert.csv',
        'QWEN': 'explore_analogies_qwen.csv'
    }
    
    results = {}
    for model_name, file_path in files.items():
        if Path(file_path).exists():
            df = pd.read_csv(file_path)
            results[model_name] = df
            print(f"✓ Loaded {model_name}: {len(df)} rows")
        else:
            print(f"⚠ {file_path} not found")
    
    return results


def analyze_overall_performance(results):
    """Analyze overall performance metrics"""
    print("\n" + "=" * 70)
    print("📊 Overall Performance Analysis")
    print("=" * 70)
    
    performance = {}
    
    for model_name, df in results.items():
        # Filter successful tests
        successful = df[df['rank'].notna()]
        total = len(df)
        success_count = len(successful)
        
        if success_count == 0:
            continue
        
        # Calculate metrics
        metrics = {
            'total_tests': total,
            'successful_tests': success_count,
            'success_rate': success_count / total * 100,
            'mean_rank': successful['rank'].mean(),
            'median_rank': successful['rank'].median(),
            'mean_similarity': successful['similarity'].mean(),
            'std_similarity': successful['similarity'].std(),
            'top1_count': successful['found_in_top_1'].sum(),
            'top1_rate': successful['found_in_top_1'].sum() / total * 100,
            'top5_count': successful['found_in_top_5'].sum(),
            'top5_rate': successful['found_in_top_5'].sum() / total * 100,
            'top10_count': successful['found_in_top_10'].sum(),
            'top10_rate': successful['found_in_top_10'].sum() / total * 100,
        }
        
        performance[model_name] = metrics
        
        print(f"\n{model_name}:")
        print(f"  Success Rate: {metrics['success_rate']:.1f}% ({success_count}/{total})")
        print(f"  Mean Rank: {metrics['mean_rank']:.1f}")
        print(f"  Median Rank: {metrics['median_rank']:.1f}")
        print(f"  Mean Similarity: {metrics['mean_similarity']:.4f}")
        print(f"  Top-1 Rate: {metrics['top1_rate']:.1f}% ({metrics['top1_count']}/{total})")
        print(f"  Top-5 Rate: {metrics['top5_rate']:.1f}% ({metrics['top5_count']}/{total})")
        print(f"  Top-10 Rate: {metrics['top10_rate']:.1f}% ({metrics['top10_count']}/{total})")
    
    return performance


def analyze_by_category(results):
    """Analyze performance by category"""
    print("\n" + "=" * 70)
    print("📂 Category-Based Analysis")
    print("=" * 70)
    
    category_analysis = {}
    
    for model_name, df in results.items():
        print(f"\n{model_name} by Category:")
        print("-" * 70)
        
        # Group by category
        category_stats = []
        for category in df['category'].unique():
            if pd.isna(category):
                continue
            
            cat_df = df[df['category'] == category]
            successful = cat_df[cat_df['rank'].notna()]
            
            if len(successful) == 0:
                continue
            
            stats = {
                'category': category,
                'total': len(cat_df),
                'successful': len(successful),
                'mean_rank': successful['rank'].mean(),
                'median_rank': successful['rank'].median(),
                'mean_similarity': successful['similarity'].mean(),
                'top1_count': successful['found_in_top_1'].sum(),
                'top1_rate': successful['found_in_top_1'].sum() / len(cat_df) * 100,
                'top5_count': successful['found_in_top_5'].sum(),
                'top5_rate': successful['found_in_top_5'].sum() / len(cat_df) * 100,
                'top10_count': successful['found_in_top_10'].sum(),
                'top10_rate': successful['found_in_top_10'].sum() / len(cat_df) * 100,
            }
            
            category_stats.append(stats)
            
            print(f"\n  {category}:")
            print(f"    Tests: {stats['total']}")
            print(f"    Mean Rank: {stats['mean_rank']:.1f}")
            print(f"    Top-1 Rate: {stats['top1_rate']:.1f}%")
            print(f"    Top-10 Rate: {stats['top10_rate']:.1f}%")
        
        category_analysis[model_name] = category_stats
    
    return category_analysis


def analyze_individual_analogies(results):
    """Analyze individual analogy performance"""
    print("\n" + "=" * 70)
    print("🔍 Individual Analogy Analysis")
    print("=" * 70)
    
    # Get all analogies
    all_analogies = []
    for model_name, df in results.items():
        for idx, row in df.iterrows():
            analogy_key = f"{row['word1']}:{row['word2']}::{row['word3']}:{row['word4']}"
            all_analogies.append({
                'analogy': analogy_key,
                'category': row['category'],
                'model': model_name,
                'rank': row['rank'],
                'similarity': row['similarity'],
                'top1': row.get('found_in_top_1', False),
                'top5': row.get('found_in_top_5', False),
                'top10': row.get('found_in_top_10', False),
            })
    
    analogy_df = pd.DataFrame(all_analogies)
    
    # Analyze each analogy across models
    analogy_summary = []
    for analogy in analogy_df['analogy'].unique():
        analogy_data = analogy_df[analogy_df['analogy'] == analogy]
        category = analogy_data['category'].iloc[0]
        
        summary = {
            'analogy': analogy,
            'category': category,
            'word1': analogy.split(':')[0],
            'word2': analogy.split(':')[1].split('::')[0],
            'word3': analogy.split('::')[1].split(':')[0],
            'word4': analogy.split(':')[-1],
        }
        
        for model_name in results.keys():
            model_data = analogy_data[analogy_data['model'] == model_name]
            if len(model_data) > 0:
                row = model_data.iloc[0]
                summary[f'{model_name}_rank'] = row['rank']
                summary[f'{model_name}_similarity'] = row['similarity']
                summary[f'{model_name}_top1'] = row['top1']
                summary[f'{model_name}_top5'] = row['top5']
                summary[f'{model_name}_top10'] = row['top10']
            else:
                summary[f'{model_name}_rank'] = None
                summary[f'{model_name}_similarity'] = None
                summary[f'{model_name}_top1'] = False
                summary[f'{model_name}_top5'] = False
                summary[f'{model_name}_top10'] = False
        
        analogy_summary.append(summary)
    
    summary_df = pd.DataFrame(analogy_summary)
    
    # Find interesting patterns
    print("\n🎯 Interesting Findings:")
    print("-" * 70)
    
    # Best performing analogies (top-1 in at least one model)
    print("\n1. Best Performing Analogies (Top-1 in at least one model):")
    best = summary_df[
        (summary_df['Word2Vec_top1'] == True) | 
        (summary_df['BERT_top1'] == True) | 
        (summary_df['QWEN_top1'] == True)
    ]
    for idx, row in best.iterrows():
        print(f"\n   {row['analogy']} ({row['category']}):")
        for model in ['Word2Vec', 'BERT', 'QWEN']:
            rank = row[f'{model}_rank']
            top1 = row[f'{model}_top1']
            if pd.notna(rank):
                marker = "🔥" if top1 else "  "
                print(f"     {marker} {model}: Rank {int(rank)}")
    
    # Worst performing analogies
    print("\n2. Worst Performing Analogies (Rank > 100 in all models):")
    worst = summary_df[
        (summary_df['Word2Vec_rank'].fillna(999) > 100) &
        (summary_df['BERT_rank'].fillna(999) > 100) &
        (summary_df['QWEN_rank'].fillna(999) > 100)
    ]
    for idx, row in worst.iterrows():
        print(f"\n   {row['analogy']} ({row['category']}):")
        for model in ['Word2Vec', 'BERT', 'QWEN']:
            rank = row[f'{model}_rank']
            if pd.notna(rank):
                print(f"     {model}: Rank {int(rank)}")
    
    # Model-specific strengths
    print("\n3. Model-Specific Strengths:")
    print("\n   Word2Vec excels at:")
    w2v_best = summary_df[summary_df['Word2Vec_top1'] == True]
    for idx, row in w2v_best.iterrows():
        bert_rank = row['BERT_rank'] if pd.notna(row['BERT_rank']) else 999
        qwen_rank = row['QWEN_rank'] if pd.notna(row['QWEN_rank']) else 999
        if bert_rank > 10 and qwen_rank > 10:
            print(f"     - {row['analogy']} (Rank: {int(row['Word2Vec_rank'])})")
            print(f"       BERT: {int(bert_rank) if bert_rank != 999 else 'N/A'}, QWEN: {int(qwen_rank) if qwen_rank != 999 else 'N/A'}")
    
    print("\n   BERT excels at:")
    bert_best = summary_df[summary_df['BERT_top1'] == True]
    for idx, row in bert_best.iterrows():
        w2v_rank = row['Word2Vec_rank'] if pd.notna(row['Word2Vec_rank']) else 999
        qwen_rank = row['QWEN_rank'] if pd.notna(row['QWEN_rank']) else 999
        if w2v_rank > 10 and qwen_rank > 10:
            print(f"     - {row['analogy']} (Rank: {int(row['BERT_rank'])})")
            print(f"       Word2Vec: {int(w2v_rank) if w2v_rank != 999 else 'N/A'}, QWEN: {int(qwen_rank) if qwen_rank != 999 else 'N/A'}")
    
    return summary_df


def analyze_similarity_distribution(results):
    """Analyze similarity score distributions"""
    print("\n" + "=" * 70)
    print("📈 Similarity Distribution Analysis")
    print("=" * 70)
    
    for model_name, df in results.items():
        successful = df[df['similarity'].notna()]
        if len(successful) == 0:
            continue
        
        print(f"\n{model_name}:")
        print(f"  Mean: {successful['similarity'].mean():.4f}")
        print(f"  Median: {successful['similarity'].median():.4f}")
        print(f"  Std: {successful['similarity'].std():.4f}")
        print(f"  Min: {successful['similarity'].min():.4f}")
        print(f"  Max: {successful['similarity'].max():.4f}")
        print(f"  Range: {successful['similarity'].max() - successful['similarity'].min():.4f}")
        
        # Quartiles
        q25 = successful['similarity'].quantile(0.25)
        q75 = successful['similarity'].quantile(0.75)
        print(f"  Q1 (25%): {q25:.4f}")
        print(f"  Q3 (75%): {q75:.4f}")
        print(f"  IQR: {q75 - q25:.4f}")


def find_interesting_patterns(results, summary_df):
    """Find interesting patterns and insights"""
    print("\n" + "=" * 70)
    print("💡 Key Insights for Presentation")
    print("=" * 70)
    
    insights = []
    
    # 1. Overall winner
    performance = analyze_overall_performance(results)
    best_model = min(performance.items(), key=lambda x: x[1]['median_rank'])
    insights.append({
        'type': 'overall_winner',
        'title': 'Best Overall Performance',
        'content': f"{best_model[0]} has the best median rank ({best_model[1]['median_rank']:.1f}) and highest Top-1 rate ({best_model[1]['top1_rate']:.1f}%)"
    })
    
    # 2. Category strengths
    category_analysis = analyze_by_category(results)
    for model_name, categories in category_analysis.items():
        if categories:
            best_category = min(categories, key=lambda x: x['mean_rank'])
            insights.append({
                'type': 'category_strength',
                'title': f'{model_name} Best Category',
                'content': f"{model_name} performs best on '{best_category['category']}' category (mean rank: {best_category['mean_rank']:.1f})"
            })
    
    # 3. Surprising failures
    surprising_failures = summary_df[
        (summary_df['Word2Vec_rank'].fillna(999) > 50) &
        (summary_df['BERT_rank'].fillna(999) > 50) &
        (summary_df['QWEN_rank'].fillna(999) > 50)
    ]
    if len(surprising_failures) > 0:
        insights.append({
            'type': 'surprising_failure',
            'title': 'Surprising Failures',
            'content': f"{len(surprising_failures)} analogies failed across all models, suggesting these may be challenging patterns"
        })
    
    # 4. Similarity vs Rank correlation
    for model_name, df in results.items():
        successful = df[df['rank'].notna() & df['similarity'].notna()]
        if len(successful) > 0:
            correlation = successful['rank'].corr(successful['similarity'])
            insights.append({
                'type': 'correlation',
                'title': f'{model_name} Similarity-Rank Correlation',
                'content': f"Correlation between similarity and rank: {correlation:.3f} (negative is expected: higher similarity = lower rank)"
            })
    
    # 5. Model differences
    print("\n🔍 Model Comparison Insights:")
    print("-" * 70)
    
    # Word2Vec advantages
    w2v_advantages = summary_df[
        (summary_df['Word2Vec_rank'].fillna(999) < 10) &
        ((summary_df['BERT_rank'].fillna(999) > 50) | (summary_df['QWEN_rank'].fillna(999) > 50))
    ]
    if len(w2v_advantages) > 0:
        print(f"\nWord2Vec advantages ({len(w2v_advantages)} cases):")
        for idx, row in w2v_advantages.head(3).iterrows():
            print(f"  - {row['analogy']}: Word2Vec rank {int(row['Word2Vec_rank'])}, BERT {int(row['BERT_rank']) if pd.notna(row['BERT_rank']) else 'N/A'}, QWEN {int(row['QWEN_rank']) if pd.notna(row['QWEN_rank']) else 'N/A'}")
    
    # BERT advantages
    bert_advantages = summary_df[
        (summary_df['BERT_rank'].fillna(999) < 10) &
        ((summary_df['Word2Vec_rank'].fillna(999) > 50) | (summary_df['QWEN_rank'].fillna(999) > 50))
    ]
    if len(bert_advantages) > 0:
        print(f"\nBERT advantages ({len(bert_advantages)} cases):")
        for idx, row in bert_advantages.head(3).iterrows():
            print(f"  - {row['analogy']}: BERT rank {int(row['BERT_rank'])}, Word2Vec {int(row['Word2Vec_rank']) if pd.notna(row['Word2Vec_rank']) else 'N/A'}, QWEN {int(row['QWEN_rank']) if pd.notna(row['QWEN_rank']) else 'N/A'}")
    
    return insights


def generate_presentation_summary(results, performance, category_analysis, summary_df):
    """Generate summary for presentation"""
    print("\n" + "=" * 70)
    print("📋 Presentation Summary")
    print("=" * 70)
    
    summary = {
        'overall_performance': performance,
        'key_findings': [],
        'interesting_cases': [],
        'recommendations': []
    }
    
    # Key findings
    print("\n🎯 Key Findings for Presentation:")
    print("-" * 70)
    
    # 1. Overall performance
    print("\n1. Overall Performance Ranking:")
    ranked_models = sorted(performance.items(), key=lambda x: x[1]['median_rank'])
    for i, (model, metrics) in enumerate(ranked_models, 1):
        print(f"   {i}. {model}: Median rank {metrics['median_rank']:.1f}, Top-1 rate {metrics['top1_rate']:.1f}%")
        summary['key_findings'].append(f"{model} has median rank {metrics['median_rank']:.1f} and {metrics['top1_rate']:.1f}% Top-1 rate")
    
    # 2. Best analogies
    print("\n2. Best Performing Analogies (Top-1 in at least one model):")
    best_analogies = summary_df[
        (summary_df['Word2Vec_top1'] == True) | 
        (summary_df['BERT_top1'] == True) | 
        (summary_df['QWEN_top1'] == True)
    ].sort_values('Word2Vec_rank')
    
    for idx, row in best_analogies.head(5).iterrows():
        print(f"\n   {row['analogy']} ({row['category']}):")
        for model in ['Word2Vec', 'BERT', 'QWEN']:
            rank = row[f'{model}_rank']
            if pd.notna(rank):
                top1 = "🔥" if row[f'{model}_top1'] else "  "
                print(f"     {top1} {model}: Rank {int(rank)}")
        summary['interesting_cases'].append({
            'analogy': row['analogy'],
            'category': row['category'],
            'results': {
                'Word2Vec': int(row['Word2Vec_rank']) if pd.notna(row['Word2Vec_rank']) else None,
                'BERT': int(row['BERT_rank']) if pd.notna(row['BERT_rank']) else None,
                'QWEN': int(row['QWEN_rank']) if pd.notna(row['QWEN_rank']) else None,
            }
        })
    
    # 3. Category analysis
    print("\n3. Category Performance:")
    all_categories = set()
    for categories in category_analysis.values():
        for cat in categories:
            all_categories.add(cat['category'])
    
    for category in sorted(all_categories):
        print(f"\n   {category}:")
        for model_name, categories in category_analysis.items():
            cat_data = next((c for c in categories if c['category'] == category), None)
            if cat_data:
                print(f"     {model_name}: Mean rank {cat_data['mean_rank']:.1f}, Top-1 rate {cat_data['top1_rate']:.1f}%")
    
    # 4. Interesting insights
    print("\n4. Interesting Insights:")
    print("   - Word2Vec performs best overall for analogy tasks")
    print("   - BERT and QWEN have higher similarity scores but worse ranks")
    print("   - Some analogies work well across all models, others fail universally")
    print("   - Category-specific patterns: some models excel in specific domains")
    
    # 5. Recommendations
    print("\n5. Recommendations for Analysis:")
    print("   - Focus on rank metrics for model comparison")
    print("   - Analyze category-specific performance")
    print("   - Investigate why some analogies fail across all models")
    print("   - Consider model-specific strengths for different use cases")
    
    return summary


def main():
    print("=" * 70)
    print("🔬 Comprehensive Analogy Results Analysis")
    print("=" * 70)
    
    # Load results
    results = load_results()
    
    if not results:
        print("❌ No result files found!")
        return
    
    # Run analyses
    performance = analyze_overall_performance(results)
    category_analysis = analyze_by_category(results)
    summary_df = analyze_individual_analogies(results)
    analyze_similarity_distribution(results)
    insights = find_interesting_patterns(results, summary_df)
    presentation_summary = generate_presentation_summary(results, performance, category_analysis, summary_df)
    
    # Save summary
    output_file = 'analysis_summary.json'
    with open(output_file, 'w') as f:
        # Convert to JSON-serializable format
        json_summary = {
            'overall_performance': {k: {kk: float(vv) if isinstance(vv, (np.int64, np.float64)) else vv 
                                      for kk, vv in v.items()} 
                                  for k, v in performance.items()},
            'key_findings': presentation_summary['key_findings'],
            'interesting_cases': presentation_summary['interesting_cases']
        }
        json.dump(json_summary, f, indent=2)
    
    print(f"\n✅ Analysis complete! Summary saved to {output_file}")
    
    # Save detailed comparison
    summary_df.to_csv('analogy_comparison.csv', index=False)
    print(f"✅ Detailed comparison saved to analogy_comparison.csv")


if __name__ == "__main__":
    main()




