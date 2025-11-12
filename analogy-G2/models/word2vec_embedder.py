"""
Word2Vec Embedder
Uses pre-trained Google News Word2Vec model
"""

import numpy as np
import gensim.downloader as api
from typing import List, Tuple, Optional
from .base_embedder import BaseEmbedder


class Word2VecEmbedder(BaseEmbedder):
    """Word2Vec embedding model using gensim"""
    
    def __init__(self):
        super().__init__("Word2Vec (Google News)")
        self.search_space = 50000  # Search top 50k words for efficiency
        
    def load_model(self):
        """Load pre-trained Word2Vec model"""
        print(f"Loading {self.model_name}...")
        print("This may take a few minutes on first download...")
        self.model = api.load('word2vec-google-news-300')
        print(f"✓ Loaded successfully!")
        print(f"  Vocabulary: {len(self.model.index_to_key):,} words")
        print(f"  Dimensions: {self.model.vector_size}")
        
    def get_embedding(self, word: str) -> Optional[np.ndarray]:
        """Get embedding vector for a word"""
        if word in self.model:
            return self.model[word]
        return None
    
    def get_vocabulary_size(self) -> int:
        """Get vocabulary size"""
        return len(self.model.index_to_key)
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension"""
        return self.model.vector_size
    
    def get_vocabulary(self, limit: int = None, filter_special: bool = True) -> List[str]:
        """
        Get vocabulary list from Word2Vec model (ordered by frequency)
        
        Args:
            limit: Maximum number of words to return (default: search_space)
            filter_special: If True, filter out special tokens (</s>, <unk>, etc.)
        
        Returns:
            List of words from Word2Vec vocabulary
        """
        if limit is None:
            limit = self.search_space
        search_limit = min(limit, len(self.model.index_to_key))
        words = list(self.model.index_to_key[:search_limit])
        
        # Filter out special tokens if requested
        if filter_special:
            filtered = []
            
            for word in words:
                # Skip special tokens (XML-like tags)
                if word.startswith('<') and word.endswith('>'):
                    continue
                # Skip tokens that contain < or > (HTML/formatting markers)
                if '<' in word or '>' in word:
                    continue
                # Skip tokens that contain ## (subword pieces or formatting tokens)
                if '##' in word:
                    continue
                # Skip tokens that contain # (formatting patterns)
                if '#' in word:
                    continue
                # Skip tokens that are only special characters
                if word in ['##', '__', '###', '####']:
                    continue
                # Skip very short tokens (likely noise, except single letters which are valid)
                if len(word) < 1:
                    continue
                # Skip tokens that are mostly punctuation (but allow words with apostrophes/hyphens)
                if not any(c.isalnum() for c in word):
                    continue
                # Skip tokens that are only numbers
                if word.isdigit():
                    continue
                filtered.append(word)
            
            return filtered
        
        return words
    
    def get_nearest_neighbors(self, vector: np.ndarray, top_n: int = 10,
                             exclude_words: List[str] = None) -> List[Tuple[str, float]]:
        """Find nearest neighbors to a vector"""
        if exclude_words is None:
            exclude_words = []
        
        # Normalize exclude words for case-insensitive comparison
        exclude_lower = [w.lower() for w in exclude_words]
        
        # Calculate similarities for all words in search space
        similarities = []
        search_limit = min(self.search_space, len(self.model.index_to_key))
        
        for word in self.model.index_to_key[:search_limit]:
            if word.lower() not in exclude_lower:
                similarity = self.cosine_similarity(vector, self.model[word])
                similarities.append((word, similarity))
        
        # Sort by similarity (descending) and return top_n
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_n]


