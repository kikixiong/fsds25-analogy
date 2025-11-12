# 🔍 Analogy Test

A command-line tool for testing word analogies across multiple embedding models: **Word2Vec**, **BERT**, and **QWEN**.


## 🚀 Quick Start

### Installation

**Using Conda (Recommended):**

```bash
# Option 1: Using environment.yml (recommended)
conda env create -f environment.yml
conda activate analogy

# Option 2: Manual setup
conda create -n analogy python=3.9 -y
conda activate analogy
pip install --upgrade pip
pip install -r requirements.txt
```

**Using venv (Alternative):**

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Usage

**Activate environment:**
```bash
# Using conda (recommended)
conda activate analogy

# Using venv (alternative)
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

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

