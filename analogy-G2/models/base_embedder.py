"""
Base class for embedding models
Provides a unified interface for all embedding models
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict, Tuple, Optional


class BaseEmbedder(ABC):
    """Abstract base class for all embedding models"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        
    @abstractmethod
    def load_model(self):
        """Load the embedding model"""
        pass
    
    @abstractmethod
    def get_embedding(self, word: str) -> Optional[np.ndarray]:
        """
        Get embedding vector for a single word
        
        Args:
            word: Input word
            
        Returns:
            numpy array of embedding vector, or None if word not found
        """
        pass
    
    @abstractmethod
    def get_vocabulary_size(self) -> int:
        """Get the vocabulary size of the model"""
        pass
    
    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Get the dimension of embeddings"""
        pass
    
    @abstractmethod
    def get_nearest_neighbors(self, vector: np.ndarray, top_n: int = 10, 
                             exclude_words: List[str] = None) -> List[Tuple[str, float]]:
        """
        Find nearest neighbors to a given vector
        
        Args:
            vector: Input vector
            top_n: Number of neighbors to return
            exclude_words: Words to exclude from results
            
        Returns:
            List of (word, similarity) tuples
        """
        pass
    
    def word_in_vocab(self, word: str) -> bool:
        """Check if word is in vocabulary"""
        embedding = self.get_embedding(word)
        return embedding is not None
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (0 to 1)
        """
        from scipy.spatial.distance import cosine
        return 1 - cosine(vec1, vec2)
    
    def test_analogy(self, word_a: str, word_b: str, word_c: str, 
                    target_word: str, top_n: int = 10) -> Dict:
        """
        Test analogy: word_a:word_b::word_c:?
        
        Args:
            word_a: First word in base pair
            word_b: Second word in base pair
            word_c: First word in target pair
            target_word: Expected result word
            top_n: Number of top results to return
            
        Returns:
            Dictionary with test results
        """
        # Check if all words are in vocabulary
        missing_words = []
        for word in [word_a, word_b, word_c, target_word]:
            if not self.word_in_vocab(word):
                missing_words.append(word)
        
        if missing_words:
            return {
                'success': False,
                'error': f"Words not in vocabulary: {', '.join(missing_words)}",
                'missing_words': missing_words
            }
        
        # Get embeddings
        vec_a = self.get_embedding(word_a)
        vec_b = self.get_embedding(word_b)
        vec_c = self.get_embedding(word_c)
        vec_target = self.get_embedding(target_word)
        
        # Perform vector arithmetic: c - a + b
        result_vector = vec_c - vec_a + vec_b
        
        # Find nearest neighbors
        exclude_words = [word_a, word_b, word_c]
        neighbors = self.get_nearest_neighbors(result_vector, top_n=top_n, 
                                              exclude_words=exclude_words)
        
        # Find rank of target word
        target_rank = None
        target_similarity = self.cosine_similarity(result_vector, vec_target)
        
        for i, (word, sim) in enumerate(neighbors, 1):
            if word.lower() == target_word.lower():
                target_rank = i
                break
        
        # If not in top_n, search further
        if target_rank is None:
            extended_neighbors = self.get_nearest_neighbors(
                result_vector, top_n=1000, exclude_words=exclude_words
            )
            for i, (word, sim) in enumerate(extended_neighbors, 1):
                if word.lower() == target_word.lower():
                    target_rank = i
                    break
        
        return {
            'success': True,
            'word_a': word_a,
            'word_b': word_b,
            'word_c': word_c,
            'target_word': target_word,
            'rank': target_rank,
            'target_similarity': target_similarity,
            'top_predictions': neighbors,
            'vector_equation': f"{word_c} - {word_a} + {word_b}",
            'found_in_top_10': target_rank is not None and target_rank <= 10,
            'found_in_top_5': target_rank is not None and target_rank <= 5,
            'found_in_top_1': target_rank == 1,
            'model_name': self.model_name
        }
    
    def get_model_info(self) -> Dict:
        """Get information about the model"""
        return {
            'name': self.model_name,
            'vocab_size': self.get_vocabulary_size(),
            'embedding_dim': self.get_embedding_dim()
        }


