"""
Models package
Provides unified interface for different embedding models
"""

from typing import List, Optional
from .base_embedder import BaseEmbedder
from .word2vec_embedder import Word2VecEmbedder
from .bert_embedder import BERTEmbedder
from .qwen_embedder import QWENEmbedder

__all__ = [
    'BaseEmbedder',
    'Word2VecEmbedder',
    'BERTEmbedder',
    'QWENEmbedder',
    'ModelManager'
]


class ModelManager:
    """Manager for loading and switching between different embedding models"""
    
    def __init__(self):
        self.models = {}
        self.current_model = None
        self._word2vec_vocabulary = None  # Cache Word2Vec vocabulary for other models
        
    def load_model(self, model_type: str, **kwargs):
        """
        Load a specific model
        
        Args:
            model_type: One of "word2vec", "bert", "qwen"
            **kwargs: Additional arguments for model initialization
        """
        if model_type in self.models:
            print(f"Model '{model_type}' already loaded.")
            self.current_model = self.models[model_type]
            return self.current_model
        
        print(f"\nInitializing {model_type}...")
        
        if model_type == "word2vec":
            model = Word2VecEmbedder()
            model.load_model()
            # Cache Word2Vec vocabulary for other models to use
            self._word2vec_vocabulary = model.get_vocabulary(limit=50000)
        elif model_type == "bert":
            bert_model = kwargs.get("bert_model", "bert-base-uncased")
            model = BERTEmbedder(model_name=bert_model)
            model.load_model()
            # Get Word2Vec vocabulary for fair comparison (load Word2Vec if needed)
            word2vec_vocab = self._ensure_word2vec_vocabulary()
            if word2vec_vocab is not None:
                model.set_word2vec_vocabulary(word2vec_vocab)
        elif model_type == "qwen":
            mode = kwargs.get("mode", "local")
            api_key = kwargs.get("api_key", None)
            qwen_model = kwargs.get("qwen_model", "Qwen/Qwen3-0.6B")
            model = QWENEmbedder(mode=mode, api_key=api_key, model_name=qwen_model)
            model.load_model()
            # Get Word2Vec vocabulary for fair comparison (load Word2Vec if needed)
            word2vec_vocab = self._ensure_word2vec_vocabulary()
            if word2vec_vocab is not None:
                model.set_word2vec_vocabulary(word2vec_vocab)
        else:
            raise ValueError(f"Unknown model type: {model_type}. Choose from: word2vec, bert, qwen")
        
        self.models[model_type] = model
        self.current_model = model
        
        return model
    
    def get_model(self, model_type: str = None):
        """Get a loaded model"""
        if model_type is None:
            return self.current_model
        return self.models.get(model_type)
    
    def list_loaded_models(self):
        """List all loaded models"""
        return list(self.models.keys())
    
    def get_available_models(self):
        """Get list of available model types"""
        return ["word2vec", "bert", "qwen"]
    
    def _get_word2vec_vocabulary(self, limit: int = 50000) -> Optional[List[str]]:
        """
        Get Word2Vec vocabulary (internal method)
        
        Args:
            limit: Maximum number of words to return
        
        Returns:
            List of words from Word2Vec vocabulary, or None if not available
        """
        if "word2vec" in self.models:
            return self.models["word2vec"].get_vocabulary(limit=limit)
        elif self._word2vec_vocabulary is not None:
            return self._word2vec_vocabulary[:limit] if limit else self._word2vec_vocabulary
        return None
    
    def _ensure_word2vec_vocabulary(self, limit: int = 50000) -> Optional[List[str]]:
        """
        Ensure Word2Vec vocabulary is available (load Word2Vec if needed)
        
        This method automatically loads Word2Vec model if it's not already loaded,
        to get the vocabulary for fair comparison with BERT/QWEN.
        
        Args:
            limit: Maximum number of words to return
        
        Returns:
            List of words from Word2Vec vocabulary, or None if loading fails
        """
        # Check if already available
        vocab = self._get_word2vec_vocabulary(limit=limit)
        if vocab is not None:
            return vocab
        
        # Try to load Word2Vec to get vocabulary
        print("  🔍 Word2Vec vocabulary not available. Loading Word2Vec to get vocabulary...")
        print("  💡 This ensures fair comparison - all models use the same search space.")
        try:
            word2vec_model = Word2VecEmbedder()
            word2vec_model.load_model()
            vocab = word2vec_model.get_vocabulary(limit=limit)
            self._word2vec_vocabulary = vocab
            # Save model to models dict so it can be reused
            self.models["word2vec"] = word2vec_model
            print(f"  ✓ Word2Vec vocabulary loaded: {len(vocab):,} words")
            print(f"  ✓ Word2Vec model cached for future use")
            return vocab
        except Exception as e:
            print(f"  ⚠️  Could not load Word2Vec vocabulary: {e}")
            print("  → Using fallback vocabulary (comparison may not be fair)")
            return None
    
    def get_word2vec_vocabulary(self, limit: int = 50000) -> Optional[List[str]]:
        """
        Get Word2Vec vocabulary for use by other models
        
        Args:
            limit: Maximum number of words to return
        
        Returns:
            List of words from Word2Vec vocabulary, or None if Word2Vec not loaded
        """
        return self._get_word2vec_vocabulary(limit=limit)


