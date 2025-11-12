#!/usr/bin/env python3
"""
Command-line interface for testing analogies
Useful for quick tests without launching the GUI
"""

import argparse
from models import ModelManager


def main():
    parser = argparse.ArgumentParser(
        description="Test word analogies using different embedding models"
    )
    
    parser.add_argument(
        "word_a",
        help="First word in base pair (e.g., 'man')"
    )
    
    parser.add_argument(
        "word_b",
        help="Second word in base pair (e.g., 'woman')"
    )
    
    parser.add_argument(
        "word_c",
        help="First word in target pair (e.g., 'king')"
    )
    
    parser.add_argument(
        "target_word",
        help="Expected target word (e.g., 'queen')"
    )
    
    parser.add_argument(
        "--model",
        choices=["word2vec", "bert", "qwen"],
        default="word2vec",
        help="Embedding model to use (default: word2vec)"
    )
    
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top predictions to show (default: 10)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print(f"Testing Analogy: {args.word_a}:{args.word_b}::{args.word_c}:?")
    print("="*70 + "\n")
    
    # Load model
    manager = ModelManager()
    print(f"Loading {args.model}...")
    model = manager.load_model(args.model)
    
    print("\n" + "-"*70 + "\n")
    
    # Test analogy
    result = model.test_analogy(
        args.word_a,
        args.word_b,
        args.word_c,
        args.target_word,
        top_n=args.top_n
    )
    
    if not result['success']:
        print(f"❌ Error: {result['error']}")
        return
    
    # Display results
    print(f"Vector Arithmetic: {result['vector_equation']}")
    print(f"\nTop {args.top_n} Predictions:")
    print("-"*70)
    
    for i, (word, sim) in enumerate(result['top_predictions'], 1):
        marker = " ← TARGET!" if word.lower() == args.target_word.lower() else ""
        print(f"{i:2d}. {word:<20} (similarity: {sim:.4f}){marker}")
    
    print("-"*70)
    print(f"\nTarget Word: '{args.target_word}'")
    print(f"Rank: #{result['rank']}" if result['rank'] else "Rank: Not found in top 1000")
    print(f"Similarity: {result['target_similarity']:.4f}")
    
    print("\n" + "="*70)
    
    # Interpretation
    if result['rank'] == 1:
        print("🔥 PERFECT MATCH! The target word is #1 prediction.")
    elif result['found_in_top_5']:
        print("✅ Strong match - target in top 5")
    elif result['found_in_top_10']:
        print("✓ Good match - target in top 10")
    elif result['rank'] and result['rank'] <= 50:
        print("~ Weak match - target in top 50")
    else:
        print("❌ Poor match - target not strongly associated")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    main()


