"""
BERT Embedder
Uses pre-trained BERT model for contextual embeddings
"""

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple, Optional
from .base_embedder import BaseEmbedder


class BERTEmbedder(BaseEmbedder):
    """BERT embedding model using HuggingFace transformers"""
    
    def __init__(self, model_name: str = "bert-base-uncased"):
        super().__init__(f"BERT ({model_name})")
        self.hf_model_name = model_name
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Cache for word embeddings (since BERT is contextual, we use single-word context)
        self.embedding_cache = {}
        self.word2vec_vocabulary = None  # Will be set from Word2Vec vocabulary
        self._filtered_vocabulary = None  # Cached filtered vocabulary
        
    def load_model(self):
        """Load pre-trained BERT model"""
        print(f"Loading {self.model_name}...")
        print(f"Device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name)
        self.model = AutoModel.from_pretrained(self.hf_model_name)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Loaded successfully!")
        print(f"  Vocabulary: {len(self.tokenizer):,} tokens")
        print(f"  Dimensions: {self.model.config.hidden_size}")
        
    def get_embedding(self, word: str) -> Optional[np.ndarray]:
        """
        Get embedding vector for a word
        Uses [CLS] token representation from single-word input
        """
        # Check cache first
        if word in self.embedding_cache:
            return self.embedding_cache[word]
        
        # Tokenize and check if word is in vocabulary
        tokens = self.tokenizer.tokenize(word)
        if not tokens:
            return None
        
        # Get embedding
        with torch.no_grad():
            inputs = self.tokenizer(word, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            
            # Use [CLS] token embedding
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
            
        # Cache the embedding
        self.embedding_cache[word] = embedding
        return embedding
    
    def get_vocabulary_size(self) -> int:
        """Get vocabulary size"""
        return len(self.tokenizer)
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension"""
        return self.model.config.hidden_size
    
    def get_nearest_neighbors(self, vector: np.ndarray, top_n: int = 10,
                             exclude_words: List[str] = None) -> List[Tuple[str, float]]:
        """
        Find nearest neighbors to a vector
        
        Note: For BERT, this searches through a common vocabulary list
        rather than the full tokenizer vocabulary, as BERT uses subword tokens
        """
        if exclude_words is None:
            exclude_words = []
        
        exclude_lower = [w.lower() for w in exclude_words]
        
        # Use a common English vocabulary for search
        # In production, you might want to use a larger vocabulary file
        common_words = self._get_common_vocabulary()
        
        similarities = []
        for word in common_words:
            if word.lower() not in exclude_lower:
                word_embedding = self.get_embedding(word)
                if word_embedding is not None:
                    similarity = self.cosine_similarity(vector, word_embedding)
                    similarities.append((word, similarity))
        
        # Sort by similarity and return top_n
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
    
    def _filter_vocabulary_for_bert(self, words: List[str]) -> List[str]:
        """
        Filter Word2Vec vocabulary to only include words BERT can handle
        
        Since BERT uses subword tokenization, it can handle most words.
        We'll use all words from Word2Vec vocabulary and let BERT's
        get_embedding method handle cases where words can't be processed.
        
        Args:
            words: List of words from Word2Vec vocabulary
        
        Returns:
            List of words (same as input, filtering happens during embedding)
        """
        # BERT can handle any word through subword tokenization
        # No need to pre-filter - let get_embedding handle it
        return words
    
    def _get_common_vocabulary(self, use_extended: bool = True) -> List[str]:
        """
        Get vocabulary list for nearest neighbor search
        
        Priority:
        1. Word2Vec vocabulary (if set) - filtered for BERT compatibility
        2. Fallback to basic vocabulary
        
        Args:
            use_extended: Ignored if Word2Vec vocabulary is available
        
        Returns:
            List of words for search
        """
        # Use Word2Vec vocabulary if available (for fair comparison)
        if self.word2vec_vocabulary is not None:
            if self._filtered_vocabulary is None:
                # Use Word2Vec vocabulary directly (BERT can handle subword tokenization)
                self._filtered_vocabulary = self._filter_vocabulary_for_bert(self.word2vec_vocabulary)
                print(f"  ✓ Using Word2Vec vocabulary: {len(self._filtered_vocabulary):,} words "
                      f"(for fair comparison with Word2Vec)")
            return self._filtered_vocabulary
        
        # Fallback to basic vocabulary if Word2Vec vocabulary not available
        base_words = [
            # Professions
            "doctor", "nurse", "teacher", "engineer", "lawyer", "architect",
            "scientist", "artist", "musician", "writer", "programmer", "designer",
            "manager", "director", "consultant", "analyst", "researcher",
            "professor", "student", "chef", "waiter", "pilot", "driver",
            "farmer", "builder", "mechanic", "electrician", "plumber",
            "accountant", "banker", "trader", "investor", "entrepreneur",
            
            # Gender & People
            "man", "woman", "boy", "girl", "male", "female", "person",
            "father", "mother", "son", "daughter", "brother", "sister",
            "husband", "wife", "parent", "child", "baby", "toddler",
            "teenager", "adult", "elder", "elderly", "senior",
            
            # Royalty & Titles
            "king", "queen", "prince", "princess", "duke", "duchess",
            "lord", "lady", "sir", "madam", "emperor", "empress",
            
            # Adjectives - Qualities
            "good", "bad", "great", "terrible", "excellent", "poor",
            "beautiful", "ugly", "attractive", "unattractive", "pretty", "handsome",
            "smart", "stupid", "intelligent", "dumb", "wise", "foolish",
            "strong", "weak", "powerful", "powerless", "mighty", "feeble",
            "rich", "poor", "wealthy", "broke", "affluent", "impoverished",
            "happy", "sad", "joyful", "miserable", "cheerful", "depressed",
            "healthy", "sick", "fit", "unfit", "able", "disabled",
            "young", "old", "youthful", "aged", "new", "ancient",
            "big", "small", "large", "tiny", "huge", "minuscule",
            "fast", "slow", "quick", "sluggish", "rapid", "gradual",
            "hot", "cold", "warm", "cool", "freezing", "burning",
            "bright", "dark", "light", "dim", "brilliant", "gloomy",
            "clean", "dirty", "pure", "filthy", "spotless", "soiled",
            "safe", "dangerous", "secure", "risky", "protected", "vulnerable",
            "easy", "difficult", "simple", "complex", "hard", "effortless",
            "high", "low", "tall", "short", "elevated", "deep",
            "wide", "narrow", "broad", "thin", "thick", "slim",
            "loud", "quiet", "noisy", "silent", "deafening", "hushed",
            "full", "empty", "complete", "incomplete", "whole", "partial",
            "right", "wrong", "correct", "incorrect", "accurate", "false",
            "true", "false", "real", "fake", "genuine", "artificial",
            "natural", "artificial", "organic", "synthetic", "wild", "domestic",
            
            # Emotions & Mental States
            "love", "hate", "like", "dislike", "adore", "despise",
            "hope", "fear", "trust", "doubt", "faith", "skepticism",
            "calm", "anxious", "relaxed", "stressed", "peaceful", "worried",
            "confident", "insecure", "bold", "timid", "brave", "cowardly",
            "proud", "ashamed", "honored", "humiliated", "dignified", "embarrassed",
            "excited", "bored", "thrilled", "indifferent", "eager", "apathetic",
            "angry", "peaceful", "furious", "serene", "enraged", "tranquil",
            "jealous", "content", "envious", "satisfied", "resentful", "grateful",
            
            # Actions & Verbs (present participle for adjective-like usage)
            "running", "walking", "swimming", "flying", "climbing", "jumping",
            "thinking", "feeling", "knowing", "learning", "teaching", "studying",
            "working", "playing", "resting", "sleeping", "eating", "drinking",
            "talking", "listening", "reading", "writing", "speaking", "hearing",
            "seeing", "watching", "looking", "observing", "noticing", "perceiving",
            "building", "creating", "making", "destroying", "breaking", "fixing",
            "helping", "hurting", "healing", "harming", "aiding", "injuring",
            "loving", "hating", "caring", "neglecting", "nurturing", "abandoning",
            
            # Places
            "city", "town", "village", "country", "nation", "state",
            "home", "house", "apartment", "building", "office", "factory",
            "school", "university", "college", "library", "museum", "theater",
            "hospital", "clinic", "pharmacy", "restaurant", "cafe", "bar",
            "park", "garden", "forest", "mountain", "valley", "river",
            "ocean", "sea", "lake", "beach", "desert", "island",
            "street", "road", "highway", "avenue", "lane", "path",
            "urban", "rural", "suburban", "metropolitan", "provincial", "cosmopolitan",
            
            # Geography
            "north", "south", "east", "west", "central", "remote",
            "local", "foreign", "domestic", "international", "global", "regional",
            "near", "far", "close", "distant", "adjacent", "remote",
            
            # Time
            "past", "present", "future", "ancient", "modern", "contemporary",
            "early", "late", "timely", "delayed", "prompt", "overdue",
            "morning", "afternoon", "evening", "night", "dawn", "dusk",
            "day", "night", "week", "month", "year", "decade",
            "yesterday", "today", "tomorrow", "now", "then", "soon",
            
            # Abstract Concepts
            "freedom", "slavery", "liberty", "oppression", "independence", "subjugation",
            "justice", "injustice", "fairness", "bias", "equality", "inequality",
            "truth", "lie", "honesty", "deception", "integrity", "corruption",
            "peace", "war", "harmony", "conflict", "cooperation", "competition",
            "knowledge", "ignorance", "wisdom", "folly", "education", "illiteracy",
            "science", "art", "logic", "emotion", "reason", "feeling",
            "success", "failure", "achievement", "defeat", "victory", "loss",
            "progress", "regression", "advancement", "decline", "improvement", "deterioration",
            "tradition", "innovation", "custom", "novelty", "conventional", "revolutionary",
            
            # Social & Political
            "democracy", "dictatorship", "freedom", "tyranny", "republic", "monarchy",
            "liberal", "conservative", "progressive", "reactionary", "radical", "moderate",
            "public", "private", "common", "exclusive", "shared", "individual",
            "society", "community", "group", "individual", "collective", "personal",
            "citizen", "immigrant", "native", "foreigner", "local", "outsider",
            "legal", "illegal", "lawful", "unlawful", "legitimate", "illicit",
            
            # Economic
            "capitalism", "socialism", "market", "economy", "trade", "commerce",
            "profit", "loss", "gain", "debt", "credit", "investment",
            "employment", "unemployment", "work", "labor", "career", "job",
            "salary", "wage", "income", "expense", "cost", "price",
            "valuable", "worthless", "expensive", "cheap", "costly", "affordable",
            
            # Education & Intelligence
            "educated", "uneducated", "literate", "illiterate", "learned", "ignorant",
            "skilled", "unskilled", "talented", "untalented", "gifted", "ordinary",
            "genius", "idiot", "brilliant", "mediocre", "exceptional", "average",
            "rational", "irrational", "logical", "illogical", "reasonable", "absurd",
            
            # Health & Body
            "healthy", "unhealthy", "fit", "unfit", "strong", "weak",
            "able", "disabled", "capable", "incapable", "competent", "incompetent",
            "sane", "insane", "normal", "abnormal", "typical", "atypical",
            "physical", "mental", "bodily", "psychological", "somatic", "cognitive",
            "illness", "wellness", "disease", "health", "sickness", "fitness",
            
            # Morality & Ethics
            "moral", "immoral", "ethical", "unethical", "virtuous", "wicked",
            "honest", "dishonest", "truthful", "deceitful", "sincere", "hypocritical",
            "kind", "cruel", "gentle", "harsh", "compassionate", "callous",
            "generous", "selfish", "altruistic", "egotistical", "charitable", "greedy",
            "humble", "arrogant", "modest", "pretentious", "meek", "haughty",
            
            # More common words for better coverage
            "thing", "stuff", "object", "item", "element", "component",
            "idea", "concept", "thought", "notion", "belief", "opinion",
            "problem", "solution", "issue", "answer", "question", "response",
            "cause", "effect", "reason", "result", "consequence", "outcome",
            "begin", "end", "start", "finish", "commence", "conclude",
            "increase", "decrease", "grow", "shrink", "expand", "contract",
            "create", "destroy", "build", "demolish", "construct", "ruin",
            "improve", "worsen", "enhance", "degrade", "better", "worse",
        ]
        
        return base_words


