"""
QWEN Embedder
Uses QWEN API for embeddings (or local QWEN model)
"""

import numpy as np
from typing import List, Tuple, Optional
from .base_embedder import BaseEmbedder
import os

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class QWENEmbedder(BaseEmbedder):
    """
    QWEN embedding model
    
    Supports two modes:
    1. API mode: Uses QWEN API (requires API key)
    2. Local mode: Uses local QWEN model via transformers
    """
    
    def __init__(self, mode: str = "local", api_key: str = None, 
                 model_name: str = "Qwen/Qwen3-0.6B"):
        """
        Initialize QWEN embedder
        
        Args:
            mode: "api" or "local"
            api_key: API key for QWEN API (if mode="api")
            model_name: Model name for local mode (default: Qwen/Qwen3-0.6B)
        """
        super().__init__(f"QWEN ({mode})")
        self.mode = mode
        self.api_key = api_key or os.getenv("QWEN_API_KEY")
        self.hf_model_name = model_name
        self.client = None
        self.tokenizer = None
        self.device = "cuda" if TRANSFORMERS_AVAILABLE and torch.cuda.is_available() else "cpu"
        self.embedding_cache = {}
        self.word2vec_vocabulary = None  # Will be set from Word2Vec vocabulary
        self._filtered_vocabulary = None  # Cached filtered vocabulary
        
    def load_model(self):
        """Load QWEN model"""
        print(f"Loading {self.model_name}...")
        
        if self.mode == "api":
            if not OPENAI_AVAILABLE:
                raise ImportError("openai package not installed. Run: pip install openai")
            if not self.api_key:
                raise ValueError("API key required for API mode. Set QWEN_API_KEY environment variable or pass api_key parameter")
            
            # Initialize OpenAI-compatible client for QWEN
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # QWEN API endpoint
            )
            print(f"✓ API client initialized")
            print(f"  Note: Vocabulary and dimensions depend on QWEN API")
            
        else:  # local mode
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("transformers package not installed. Run: pip install transformers torch")
            
            print(f"Loading local model: {self.hf_model_name}")
            print(f"Device: {self.device}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name, 
                                                          trust_remote_code=True)
            self.model = AutoModel.from_pretrained(self.hf_model_name,
                                                   trust_remote_code=True)
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✓ Loaded successfully!")
            print(f"  Vocabulary: {len(self.tokenizer):,} tokens")
            print(f"  Dimensions: {self.model.config.hidden_size}")
    
    def get_embedding(self, word: str) -> Optional[np.ndarray]:
        """Get embedding vector for a word"""
        # Check cache
        if word in self.embedding_cache:
            return self.embedding_cache[word]
        
        if self.mode == "api":
            return self._get_embedding_api(word)
        else:
            return self._get_embedding_local(word)
    
    def _get_embedding_api(self, word: str) -> Optional[np.ndarray]:
        """Get embedding using QWEN API"""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-v1",  # QWEN embedding model
                input=word
            )
            embedding = np.array(response.data[0].embedding)
            self.embedding_cache[word] = embedding
            return embedding
        except Exception as e:
            print(f"API error for word '{word}': {e}")
            return None
    
    def _get_embedding_local(self, word: str) -> Optional[np.ndarray]:
        """Get embedding using local QWEN model"""
        # Check if word can be tokenized
        tokens = self.tokenizer.tokenize(word)
        if not tokens:
            return None
        
        with torch.no_grad():
            inputs = self.tokenizer(word, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            
            # Use mean pooling of last hidden state
            embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
        
        self.embedding_cache[word] = embedding
        return embedding
    
    def get_vocabulary_size(self) -> int:
        """Get vocabulary size"""
        if self.mode == "api":
            return -1  # Unknown for API mode
        return len(self.tokenizer)
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension"""
        if self.mode == "api":
            # Try to get a sample embedding to determine dimension
            sample_embedding = self.get_embedding("test")
            return len(sample_embedding) if sample_embedding is not None else -1
        return self.model.config.hidden_size
    
    def get_nearest_neighbors(self, vector: np.ndarray, top_n: int = 10,
                             exclude_words: List[str] = None) -> List[Tuple[str, float]]:
        """Find nearest neighbors to a vector"""
        if exclude_words is None:
            exclude_words = []
        
        exclude_lower = [w.lower() for w in exclude_words]
        
        # Use common vocabulary (same as BERT)
        common_words = self._get_common_vocabulary()
        
        similarities = []
        for word in common_words:
            if word.lower() not in exclude_lower:
                word_embedding = self.get_embedding(word)
                if word_embedding is not None:
                    similarity = self.cosine_similarity(vector, word_embedding)
                    similarities.append((word, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_n]
    
    def set_word2vec_vocabulary(self, word2vec_vocab: List[str]):
        """
        Set Word2Vec vocabulary to use for search space (for fair comparison)
        
        Args:
            word2vec_vocab: List of words from Word2Vec vocabulary
        """
        self.word2vec_vocabulary = word2vec_vocab
        self._filtered_vocabulary = None  # Reset cache
        print(f"  ✓ Using Word2Vec vocabulary: {len(word2vec_vocab):,} words")
    
    def _get_common_vocabulary(self, use_extended: bool = True) -> List[str]:
        """
        Get vocabulary list for nearest neighbor search
        
        Priority:
        1. Word2Vec vocabulary (if set) - for fair comparison
        2. Fallback to BERT's vocabulary method
        
        Args:
            use_extended: Ignored if Word2Vec vocabulary is available
        
        Returns:
            List of words for search
        """
        # Use Word2Vec vocabulary if available (for fair comparison)
        if self.word2vec_vocabulary is not None:
            if self._filtered_vocabulary is None:
                # Use Word2Vec vocabulary directly (QWEN can handle subword tokenization)
                self._filtered_vocabulary = self.word2vec_vocabulary
                print(f"  ✓ Using Word2Vec vocabulary: {len(self._filtered_vocabulary):,} words "
                      f"(for fair comparison with Word2Vec)")
            return self._filtered_vocabulary
        
        # Fallback to BERT's method if Word2Vec vocabulary not available
        from .bert_embedder import BERTEmbedder
        bert = BERTEmbedder()
        return bert._get_common_vocabulary(use_extended=use_extended)


