# transformer_manager.py
"""
Manages transformer models for cross-lingual NLP tasks.

This module:
- Provides a unified interface for transformer models
- Handles model loading and caching
- Supports text embeddings, classification, and other transformer tasks
- Works with multilingual models for cross-language analysis
"""

import logging
from typing import List, Optional, Tuple, Any
import torch

class TransformerManager:
    """Manages transformer models for NLP tasks across languages."""

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the transformer manager.
        
        Args:
            cache_dir: Directory for caching models (defaults to ~/.cache/huggingface)
        """
        self.logger = logging.getLogger(__name__)
        self.cache_dir = cache_dir
        self._loaded_models = {}

        # Check for available hardware acceleration
        self.device = self._get_optimal_device()
        self.logger.info("Using device: %s", self.device)

    def _get_optimal_device(self) -> str:
        """
        Determine the optimal device for model inference.
        
        Returns:
            Device string ("cuda", "mps", or "cpu")
        """
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch, 'mps') and torch.mps.is_available():
            # Apple Silicon (M1/M2) support
            return "mps"
        else:
            return "cpu"


    def get_embedding_model(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Get a sentence embedding model.
        
        Args:
            model_name: Name of the sentence transformer model
            
        Returns:
            Sentence transformer model
        """
        # Import here to avoid dependencies if not using this function
        try:
            from sentence_transformers import SentenceTransformer
            
            model_key = f"embedding_{model_name}"
           
            # Return cached model if available
            if model_key in self._loaded_models:
                return self._loaded_models[model_key]

            # Load and cache the model
            model = SentenceTransformer(model_name, cache_folder=self.cache_dir)
            model.to(self.device)

            self._loaded_models[model_key] = model
            self.logger.info("Loaded embedding model: %s", model_name)

            return model

        except ImportError:
            self.logger.error("sentence-transformers package not installed. "
                             "Please install with: pip install sentence-transformers")
            return None
        except OSError as e:
            self.logger.error("Error loading embedding model - file not found: %s", e)
            return None
        except RuntimeError as e:
            self.logger.error("Runtime error loading model (possibly CUDA/memory related): %s", e)
            self.logger.info("Falling back to CPU")
            # Try again with CPU
            try:
                model = SentenceTransformer(model_name, cache_folder=self.cache_dir)
                model.to("cpu")
                self._loaded_models[model_key] = model
                return model
            except ImportError as fallback_err:
                self.logger.error("Import error in fallback: %s", fallback_err)
                return None
            except OSError as fallback_err:
                self.logger.error("File error in fallback: %s", fallback_err)
                return None
            except RuntimeError as fallback_err:
                self.logger.error("Runtime error in CPU fallback: %s", fallback_err)
                return None
            

    def get_classification_model(self, model_name: str = "nlptown/bert-base-multilingual-uncased-sentiment") -> Tuple[Any, Any]:
        """
        Get a classification model (e.g., for sentiment analysis).
        
        Args:
            model_name: Name of the transformer model
            
        Returns:
            Transformer model and tokenizer as a tuple
        """
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            
            model_key = f"classification_{model_name}"
            
            # Return cached model if available
            if model_key in self._loaded_models:
                return self._loaded_models[model_key]
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=self.cache_dir)
            model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=self.cache_dir)
            model.to(self.device)
            
            # Store as a tuple (tokenizer, model)
            self._loaded_models[model_key] = (tokenizer, model)
            self.logger.info("Loaded classification model: %s", model_name)
            
            return tokenizer, model
            
        except ImportError:
            self.logger.error("transformers package not installed. "
                             "Please install with: pip install transformers")
            return None, None
        except OSError as e:
            self.logger.error("Error loading classification model - file not found: %s", e)
            return None, None
        except RuntimeError as e:
            self.logger.error("Runtime error loading model (possibly CUDA/memory related): %s", e)
            self.logger.info("Falling back to CPU")
            # Try again with CPU
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=self.cache_dir)
                model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=self.cache_dir)
                model.to("cpu")
                self._loaded_models[model_key] = (tokenizer, model)
                return tokenizer, model
            except ImportError as fallback_err:
                self.logger.error("Import error in fallback: %s", fallback_err)
                return None, None
            except OSError as fallback_err:
                self.logger.error("File error in fallback: %s", fallback_err)
                return None, None
            except RuntimeError as fallback_err:
                self.logger.error("Runtime error in CPU fallback: %s", fallback_err)
                return None, None
    

    def embed_texts(self, texts: List[str], model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Embed a list of texts using a sentence transformer.
        
        Args:
            texts: List of texts to embed
            model_name: Name of the embedding model
            
        Returns:
            Numpy array of embeddings
        """
        model = self.get_embedding_model(model_name)
        
        if model is None:
            return None
            
        try:
            # Create embeddings
            embeddings = model.encode(texts, convert_to_numpy=True)
            return embeddings
        except RuntimeError as e:
            self.logger.error("Runtime error embedding texts (possibly memory-related): %s", e)
            # Try with smaller batch size if possible
            try:
                embeddings = model.encode(texts, convert_to_numpy=True, batch_size=1)
                return embeddings
            except RuntimeError as batch_err:
                self.logger.error("Failed with batch size 1 as well: %s", batch_err)
                return None
        except ValueError as e:
            self.logger.error("Value error embedding texts (possibly incorrect input): %s", e)
            return None
    
    def calculate_similarity(self, text1: str, text2: str, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Calculate semantic similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            model_name: Name of the embedding model
            
        Returns:
            Similarity score (0-1)
        """
        try:
            import numpy as np
            
            # Get embeddings
            embeddings = self.embed_texts([text1, text2], model_name)
            
            if embeddings is None or len(embeddings) < 2:
                return 0.0
                
            # Calculate cosine similarity
            emb1 = embeddings[0]
            emb2 = embeddings[1]
            
            # Handle zero vectors to avoid division by zero
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            sim_score = np.dot(emb1, emb2) / (norm1 * norm2)
            
            return float(sim_score)
            
        except ImportError:
            self.logger.error("NumPy not installed")
            return 0.0
        except ValueError as e:
            self.logger.error("Value error calculating similarity: %s", e)
            return 0.0
            

    def classify_sentiment(self, 
                      text: str, 
                      model_name: str = "nlptown/bert-base-multilingual-uncased-sentiment"):
        """
        Classify the sentiment of a text.
        
        Args:
            text: Text to classify
            model_name: Name of the sentiment model
            
        Returns:
            Dict with sentiment score and label
        """
        try:
            import torch.nn.functional as F
            
            # Handle non-string inputs
            if text is None:
                self.logger.warning("Received None text input, returning neutral sentiment")
                return {'label': 'neutral', 'score': 0.5}
                
            # Ensure text is a string (convert if possible, otherwise handle error)
            if not isinstance(text, str):
                # Try to convert dictionary's 'text' field if available
                if isinstance(text, dict) and 'text' in text:
                    text = text['text']
                    if not isinstance(text, str):
                        self.logger.warning(f"Text field is not a string: {type(text)}")
                        return {'label': 'neutral', 'score': 0.5}
                # Try to convert to string as a fallback for simple types
                elif isinstance(text, (int, float, bool)):
                    text = str(text)
                else:
                    self.logger.warning(f"Cannot process input of type: {type(text)}")
                    return {'label': 'neutral', 'score': 0.5}
            
            # Now text should be a string, continue with normal processing
            tokenizer, model = self.get_classification_model(model_name)
            
            if tokenizer is None or model is None:
                return {'label': 'neutral', 'score': 0.5}
                    
            # Encode the text
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get model prediction
            with torch.no_grad():
                outputs = model(**inputs)
                
            # Process outputs according to the model type
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs[0]
                
            # Convert logits to probabilities
            probs = F.softmax(logits, dim=1)
            
            # Get the predicted class and its probability
            predicted_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, predicted_class].item()
            
            # Map the predicted class to a label (specific to the model)
            if model_name == "nlptown/bert-base-multilingual-uncased-sentiment":
                # This model outputs 1-5 stars
                sentiment_map = {
                    0: "very negative",
                    1: "negative",
                    2: "neutral",
                    3: "positive",
                    4: "very positive"
                }
                label = sentiment_map.get(predicted_class, "neutral")
                
                # Normalize score to 0-1 range
                score = (predicted_class / 4.0)  # 0-4 to 0-1
            else:
                # Generic handling for other models
                # Assuming standard negative (0), neutral (1), positive (2)
                sentiment_map = {
                    0: "negative",
                    1: "neutral", 
                    2: "positive"
                }
                label = sentiment_map.get(predicted_class, "neutral")
                score = confidence
            
            return {
                'label': label,
                'score': score,
                'confidence': confidence
            }
                
        except ImportError:
            self.logger.error("Required package not installed")
            return {'label': 'neutral', 'score': 0.5}
        except RuntimeError as e:
            self.logger.error("Runtime error classifying sentiment: %s", e)
            return {'label': 'neutral', 'score': 0.5}
        except ValueError as e:
            self.logger.error("Value error classifying sentiment: %s", e)
            return {'label': 'neutral', 'score': 0.5}