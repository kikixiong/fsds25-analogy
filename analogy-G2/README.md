# 🔍 Analogy Testing Platform

A command-line tool for testing word analogies across multiple embedding models: **Word2Vec**, **BERT**, and **QWEN**.

## ✨ Features

- **Multi-Model Support**: Word2Vec, BERT, QWEN
- **CLI Tools**: Single test and batch processing
- **Comprehensive Results**: Top-N predictions, ranks, similarities
- **CSV/Excel Support**: Batch processing with automatic format detection
- **Fair Comparison**: All models use the same vocabulary search space

---

## 🚀 Quick Start

### Installation

```bash
cd analogy-platform
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Usage

**Single test:**
```bash
python test_cli.py man woman king queen
python test_cli.py --model bert man woman doctor nurse
```

**Batch testing:**
```bash
python batch_test.py explore_analogies.csv --model word2vec --output results.csv
```

---

## 📖 Usage Guide

### Single Test

```bash
python test_cli.py <word_a> <word_b> <word_c> <target_word> [options]

Options:
  --model MODEL     Model type: word2vec, bert, qwen (default: word2vec)
  --top-n N         Number of top predictions to show (default: 10)
```

**Examples:**
```bash
python test_cli.py man woman king queen
python test_cli.py --model bert man woman doctor nurse
python test_cli.py --top-n 20 Paris France London England
```

### Batch Testing

```bash
python batch_test.py <input.csv> [options]

Options:
  --model MODEL     Model type: word2vec, bert, qwen (default: word2vec)
  --output FILE     Output CSV file (default: <input>_results.csv)
  --top-n N         Number of top predictions (default: 10)
  --verbose         Print detailed progress
```

**Input File Format:**

CSV or Excel file with columns: `word1`, `word2`, `word3`, `word4`, `category` (optional)

```csv
word1,word2,word3,word4,category
man,woman,king,queen,gender
Paris,France,London,England,geography
```

- **CSV files**: Auto-detects encoding (UTF-8, GBK, GB2312, Latin-1, etc.)
- **Excel files**: Automatically detected even if file extension is `.csv`
- **Output**: Contains input words, rank, similarity, top-10 predictions (separate columns), and error messages

---

## 🔬 How It Works

For analogy **A:B::C:D**, the system computes:

```
result_vector = embedding(C) - embedding(A) + embedding(B)
```

Then finds words with embeddings closest to `result_vector` using **cosine similarity**.

**Example:** `man:woman::king:queen`
1. Computes `king - man + woman`
2. Calculates cosine similarity with vocabulary words
3. Ranks words by similarity
4. Returns top 10 predictions + target word's rank

---

## 📊 Interpreting Results

| Rank | Meaning | Interpretation |
|------|---------|----------------|
| 🔥 #1 | Perfect match | Target word is the top prediction |
| ✅ Top 5 | Strong match | Pattern well learned |
| ✓ Top 10 | Moderate match | Some association exists |
| ~ Top 50 | Weak match | Loose association |
| ❌ >50 | Poor match | No strong pattern |

**Note:** Rank is the most reliable metric for comparison. Similarity scores vary across models (Word2Vec: 0.3-0.7, BERT/QWEN: 0.75-0.95) due to different embedding dimensions and training objectives, but this does not indicate better performance.

---

## 🎯 Example Analogies

```bash
# Grammar patterns
python test_cli.py good better bad worse
python test_cli.py walking walked swimming swam

# Geographic relations
python test_cli.py Paris France London England
python test_cli.py Tokyo Japan Berlin Germany

# Social science patterns
python test_cli.py man woman doctor nurse
python test_cli.py young old attractive unattractive
python test_cli.py poor rich lazy hardworking
```

---

## 🤖 Models

### Model Comparison

| Metric | Word2Vec | BERT | QWEN |
|--------|----------|------|------|
| **Dimension** | 300 | 768 | 1024 |
| **Architecture** | Skip-gram | Transformer encoder | Transformer decoder |
| **Data Source** | Google News | BooksCorpus + Wikipedia | Multilingual mix |
| **Case** | Sensitive | Uncased | Sensitive |
| **Speed** | ⚡⚡⚡ Fast (~2s) | ⚡⚡ Medium (~8s) | ⚡⚡ Medium (~9s) |
| **Performance** | ⭐⭐⭐ Best (rank 9.0) | ⭐⭐ Moderate (rank 35.0) | ⭐ Weak (rank 586.5) |
| **Similarity Range** | 0.30-0.70 | 0.75-0.97 | 0.41-0.99 |

### Fair Comparison Setup

All models use the **same search space** (50,000 most frequent words from Word2Vec vocabulary) for fair comparison. BERT and QWEN automatically load Word2Vec vocabulary when initialized.

**Important:** BERT and QWEN show higher similarity scores due to:
- High-dimensional dense spaces (768D/1024D vs 300D)
- Different training objectives (contextual understanding vs word similarity)
- Curse of dimensionality in high-dimensional spaces

**Rank is the more reliable metric** for comparison, not similarity scores.

---

## 🛠️ Troubleshooting

### Module Not Found
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Model Loading Slow
- First download takes time (Word2Vec: ~1-2 min, BERT: ~2-5 min, QWEN: ~3-5 min)
- Subsequent loads are fast (cached)

### Word Not in Vocabulary
- Try lowercase: `King` → `king`
- Try singular form: `doctors` → `doctor`
- Check if word exists in model vocabulary

### Batch Test Errors
- Ensure CSV has required columns: `word1`, `word2`, `word3`, `word4`
- Check file encoding (use UTF-8)
- Verify all words are valid (no empty cells)

### Clean Up Old Models
```bash
python cleanup_old_models.py
```

---

## 📁 Project Structure

```
analogy-platform/
├── models/                  # Model implementations
│   ├── __init__.py          # Model manager
│   ├── base_embedder.py     # Abstract base class
│   ├── word2vec_embedder.py
│   ├── bert_embedder.py
│   └── qwen_embedder.py
├── batch_test.py            # Batch testing CLI
├── test_cli.py              # Single test CLI
├── analyze_results.py       # Results analysis
├── visualize_results.py     # Visualization
├── cleanup_old_models.py    # Cleanup tool
├── explore_analogies.csv    # Example input file
├── requirements.txt         # Dependencies
├── setup.sh                 # Setup script
└── README.md                # This file
```

---

## 💻 API Reference

### ModelManager

```python
from models import ModelManager

manager = ModelManager()
model = manager.load_model("word2vec")  # or "bert", "qwen"

result = model.test_analogy("man", "woman", "king", "queen", top_n=10)

print(f"Rank: {result['rank']}")
print(f"Similarity: {result['target_similarity']}")
print(f"Top predictions: {result['top_predictions']}")
```

### Result Dictionary

```python
{
    'success': True,
    'word_a': 'man',
    'word_b': 'woman',
    'word_c': 'king',
    'target_word': 'queen',
    'rank': 1,
    'target_similarity': 0.6543,
    'top_predictions': [('queen', 0.6543), ('princess', 0.6234), ...],
    'found_in_top_1': True,
    'found_in_top_5': True,
    'found_in_top_10': True,
    'vector_equation': 'king - man + woman',
    'model_name': 'Word2Vec (Google News)'
}
```

---

## 🔧 Advanced Usage

### QWEN API Mode

```bash
export QWEN_API_KEY="your-api-key-here"
```

```python
from models import ModelManager

manager = ModelManager()
model = manager.load_model("qwen", mode="api", api_key="your-key")
```

### Custom Batch Processing

```python
from models import ModelManager
import pandas as pd

manager = ModelManager()
model = manager.load_model("word2vec")

df = pd.read_csv("input.csv")
results = []

for _, row in df.iterrows():
    result = model.test_analogy(
        row['word1'], row['word2'], 
        row['word3'], row['word4']
    )
    results.append(result)
```

---

## 📝 License

See LICENSE file for details.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

---

## 🎉 Acknowledgments

- **Word2Vec**: Google News embeddings
- **BERT**: HuggingFace transformers
- **QWEN**: Qwen team

---

**Happy Testing! 🎯**
