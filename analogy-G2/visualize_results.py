#!/usr/bin/env python3
"""
Generate visualizations for presentation
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from pathlib import Path


def load_data():
    """Load all result files"""
    results = {}
    files = {
        'Word2Vec': 'explore_analogies_word2vec.csv',
        'BERT': 'explore_analogies_bert.csv',
        'QWEN': 'explore_analogies_qwen.csv'
    }
    
    for model_name, file_path in files.items():
        if Path(file_path).exists():
            results[model_name] = pd.read_csv(file_path)
    
    return results


def plot_overall_performance(results):
    """Plot overall performance comparison"""
    models = []
    top1_rates = []
    top5_rates = []
    top10_rates = []
    median_ranks = []
    
    for model_name, df in results.items():
        successful = df[df['rank'].notna()]
        total = len(df)
        
        if len(successful) > 0:
            models.append(model_name)
            top1_rates.append(successful['found_in_top_1'].sum() / total * 100)
            top5_rates.append(successful['found_in_top_5'].sum() / total * 100)
            top10_rates.append(successful['found_in_top_10'].sum() / total * 100)
            median_ranks.append(successful['rank'].median())
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Top-K Success Rates', 'Median Rank (lower is better)'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Top-K rates
    fig.add_trace(
        go.Bar(name='Top-1', x=models, y=top1_rates, marker_color='#1f77b4'),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(name='Top-5', x=models, y=top5_rates, marker_color='#ff7f0e'),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(name='Top-10', x=models, y=top10_rates, marker_color='#2ca02c'),
        row=1, col=1
    )
    
    # Median ranks
    fig.add_trace(
        go.Bar(name='Median Rank', x=models, y=median_ranks, marker_color='#d62728'),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Model", row=1, col=1)
    fig.update_xaxes(title_text="Model", row=1, col=2)
    fig.update_yaxes(title_text="Success Rate (%)", row=1, col=1)
    fig.update_yaxes(title_text="Median Rank", row=1, col=2)
    
    fig.update_layout(
        title_text="Overall Performance Comparison",
        barmode='group',
        height=500,
        showlegend=True
    )
    
    fig.write_html('visualization_overall_performance.html')
    print("✅ Saved: visualization_overall_performance.html")


def plot_similarity_vs_rank(results):
    """Plot similarity vs rank scatter plot"""
    fig = go.Figure()
    
    colors = {'Word2Vec': '#1f77b4', 'BERT': '#ff7f0e', 'QWEN': '#2ca02c'}
    
    for model_name, df in results.items():
        successful = df[df['rank'].notna() & df['similarity'].notna()]
        if len(successful) > 0:
            fig.add_trace(go.Scatter(
                x=successful['similarity'],
                y=successful['rank'],
                mode='markers',
                name=model_name,
                marker=dict(
                    color=colors.get(model_name, '#000000'),
                    size=10,
                    opacity=0.6
                ),
                text=successful.apply(lambda row: f"{row['word1']}:{row['word2']}::{row['word3']}:{row['word4']}", axis=1),
                hovertemplate='<b>%{text}</b><br>Similarity: %{x:.3f}<br>Rank: %{y}<extra></extra>'
            ))
    
    fig.update_xaxes(title_text="Similarity Score")
    fig.update_yaxes(title_text="Rank (log scale)", type="log")
    fig.update_layout(
        title_text="Similarity vs Rank (The Similarity Paradox)",
        height=600,
        showlegend=True
    )
    
    fig.write_html('visualization_similarity_vs_rank.html')
    print("✅ Saved: visualization_similarity_vs_rank.html")


def plot_category_performance(results):
    """Plot category performance heatmap"""
    # Collect category data
    category_data = []
    
    for model_name, df in results.items():
        for category in df['category'].unique():
            if pd.isna(category):
                continue
            
            cat_df = df[df['category'] == category]
            successful = cat_df[cat_df['rank'].notna()]
            
            if len(successful) > 0:
                top1_rate = successful['found_in_top_1'].sum() / len(cat_df) * 100
                median_rank = successful['rank'].median()
                
                category_data.append({
                    'model': model_name,
                    'category': category,
                    'top1_rate': top1_rate,
                    'median_rank': median_rank
                })
    
    if not category_data:
        print("⚠ No category data found")
        return
    
    cat_df = pd.DataFrame(category_data)
    
    # Create heatmap for Top-1 rate
    pivot_top1 = cat_df.pivot(index='category', columns='model', values='top1_rate')
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_top1.values,
        x=pivot_top1.columns,
        y=pivot_top1.index,
        colorscale='RdYlGn',
        text=pivot_top1.values,
        texttemplate='%{text:.1f}%',
        textfont={"size": 12},
        colorbar=dict(title="Top-1 Rate (%)")
    ))
    
    fig.update_layout(
        title_text="Category Performance - Top-1 Rate",
        xaxis_title="Model",
        yaxis_title="Category",
        height=400
    )
    
    fig.write_html('visualization_category_performance.html')
    print("✅ Saved: visualization_category_performance.html")


def plot_analogy_matrix(results):
    """Plot analogy success/failure matrix"""
    # Load comparison data
    if Path('analogy_comparison.csv').exists():
        df = pd.read_csv('analogy_comparison.csv')
    else:
        print("⚠ analogy_comparison.csv not found, skipping matrix plot")
        return
    
    # Prepare data for heatmap
    analogies = df['analogy'].tolist()
    models = ['Word2Vec', 'BERT', 'QWEN']
    
    # Create rank matrix
    rank_matrix = []
    for model in models:
        ranks = []
        for idx, row in df.iterrows():
            rank = row[f'{model}_rank']
            if pd.notna(rank):
                ranks.append(rank)
            else:
                ranks.append(999)  # Use 999 for failed cases
        rank_matrix.append(ranks)
    
    # Convert to log scale for better visualization
    rank_matrix_log = np.log10(np.array(rank_matrix) + 1)
    
    fig = go.Figure(data=go.Heatmap(
        z=rank_matrix_log,
        x=analogies,
        y=models,
        colorscale='RdYlGn_r',  # Reversed: green = good (low rank), red = bad (high rank)
        text=np.array(rank_matrix),
        texttemplate='%{text:.0f}',
        textfont={"size": 8},
        colorbar=dict(title="Rank (log scale)")
    ))
    
    fig.update_layout(
        title_text="Analogy Success/Failure Matrix",
        xaxis_title="Analogy",
        yaxis_title="Model",
        height=500,
        xaxis=dict(tickangle=45)
    )
    
    fig.write_html('visualization_analogy_matrix.html')
    print("✅ Saved: visualization_analogy_matrix.html")


def plot_similarity_distribution(results):
    """Plot similarity score distribution"""
    fig = go.Figure()
    
    colors = {'Word2Vec': '#1f77b4', 'BERT': '#ff7f0e', 'QWEN': '#2ca02c'}
    
    for model_name, df in results.items():
        successful = df[df['similarity'].notna()]
        if len(successful) > 0:
            fig.add_trace(go.Violin(
                y=successful['similarity'],
                name=model_name,
                box_visible=True,
                meanline_visible=True,
                fillcolor=colors.get(model_name, '#000000'),
                line_color=colors.get(model_name, '#000000'),
                opacity=0.6
            ))
    
    fig.update_layout(
        title_text="Similarity Score Distribution",
        yaxis_title="Similarity Score",
        xaxis_title="Model",
        height=500
    )
    
    fig.write_html('visualization_similarity_distribution.html')
    print("✅ Saved: visualization_similarity_distribution.html")


def main():
    print("=" * 70)
    print("📊 Generating Visualizations")
    print("=" * 70)
    print()
    
    # Load data
    results = load_data()
    
    if not results:
        print("❌ No result files found!")
        return
    
    # Generate visualizations
    print("Generating visualizations...")
    plot_overall_performance(results)
    plot_similarity_vs_rank(results)
    plot_category_performance(results)
    plot_analogy_matrix(results)
    plot_similarity_distribution(results)
    
    print()
    print("=" * 70)
    print("✅ All visualizations generated!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - visualization_overall_performance.html")
    print("  - visualization_similarity_vs_rank.html")
    print("  - visualization_category_performance.html")
    print("  - visualization_analogy_matrix.html")
    print("  - visualization_similarity_distribution.html")
    print("\nOpen these HTML files in a browser to view the visualizations.")


if __name__ == "__main__":
    main()




