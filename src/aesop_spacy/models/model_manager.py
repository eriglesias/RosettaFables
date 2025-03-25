from typing import Optional, Dict, Any
import logging
import spacy
from pathlib import Path
import importlib.util

# Define language-model mapping
LANGUAGE_MODELS = {
    'nl': 'nl_core_news_lg',    # Dutch
    'de': 'de_core_news_lg',    # German
    'en': 'en_core_web_lg',     # English
    'es': 'es_core_news_md',    # Spanish
    'xx': 'xx_sent_ud_sm'       # Multilingual (fallback)
}

# Cache loaded models
_loaded_models = {}

class StanzaWrapper:
    """
    Wrapper class for Stanza NLP models to make them compatible with the pipeline.
    This provides a consistent interface similar to spaCy models.
    """
    
    def __init__(self, nlp, language: str):
        """
        Initialize the wrapper with a Stanza processor.
        
        Args:
            nlp: The Stanza processor
            language: Language code
        """
        self.nlp = nlp
        self.language = language
        self.logger = logging.getLogger(__name__)
        
        # Add meta attribute for compatibility with spaCy
        self.meta = {
            'name': f'stanza_{language}',
            'source': 'stanza',
            'language': language
        }
        
        # Add pipe_names for compatibility with spaCy
        self.pipe_names = ['tokenize', 'pos', 'lemma']
        
        # Add other spaCy-like attributes
        self.vocab = type('DummyVocab', (), {'strings': {}})
    
    def __call__(self, text: str) -> Any:
        """
        Process text using the Stanza processor and return a document object.
        
        Args:
            text: Input text
            
        Returns:
            Processed document
        """
        # Process the text with Stanza
        doc = self.nlp(text)
        
        # Create a compatible document with necessary attributes
        doc = self._create_compatible_doc(doc)
        
        return doc
    
    def _create_compatible_doc(self, stanza_doc):
        """
        Create a compatible document with spaCy-like attributes.
        
        Args:
            stanza_doc: Original Stanza document
            
        Returns:
            Compatible document with necessary attributes
        """
        # For debugging purposes - understand what we're dealing with
        self.logger.debug(f"Document type: {type(stanza_doc).__name__}")
        
        # Check if we're dealing with our special wrapper already
        if hasattr(stanza_doc, 'tokens') and hasattr(stanza_doc, 'sentence_spans'):
            # This is already our custom CompatibleDoc from stanza_wrapper.py
            # Just ensure it has all required attributes
            if not hasattr(stanza_doc, 'sentences'):
                # Use sentence_spans as sentences attribute for compatibility
                stanza_doc.sentences = stanza_doc.sentence_spans
            
            # Apply any other missing attributes
            self._enhance_tokens(stanza_doc)
            return stanza_doc
        
        # Create a new compatible document class
        class CompatibleDoc:
            def __init__(self, stanza_doc):
                self.stanza_doc = stanza_doc
                
                # Set text attribute
                self.text = stanza_doc.text if hasattr(stanza_doc, 'text') else ""
                
                # Extract sentences and tokens
                if hasattr(stanza_doc, 'sentences'):
                    self.sentences = stanza_doc.sentences
                    self.tokens = []
                    self.words = []
                    
                    # Process sentences and tokens
                    for sent in self.sentences:
                        for token in sent.tokens:
                            self.tokens.append(token)
                            for word in token.words:
                                self.words.append(word)
                    
                    # Add sentence spans for compatibility with alternative structure
                    self.sentence_spans = self.sentences
                else:
                    # Fallback for documents without sentences attribute
                    self.sentences = []
                    self.tokens = getattr(stanza_doc, 'tokens', [])
                    self.words = getattr(stanza_doc, 'words', [])
                    self.sentence_spans = getattr(stanza_doc, 'sentence_spans', [])
                
                # Add spaCy-like attributes
                self.ents = getattr(stanza_doc, 'ents', [])
                self.vector = getattr(stanza_doc, 'vector', [])
                
                # Add sentence iterator for spaCy compatibility
                self.sents = self.sentences if hasattr(self, 'sentences') and self.sentences else self.sentence_spans
        
        # Create and return the compatible document
        try:
            compat_doc = CompatibleDoc(stanza_doc)
            
            # Enhance tokens for full compatibility
            self._enhance_tokens(compat_doc)
            
            return compat_doc
        except Exception as e:
            self.logger.error(f"Error creating compatible document: {e}")
            # Create a minimal compatible document
            doc = CompatibleDoc.__new__(CompatibleDoc)
            doc.text = getattr(stanza_doc, 'text', "")
            doc.sentences = []
            doc.tokens = []
            doc.words = []
            doc.ents = []
            doc.vector = []
            doc.sents = []
            doc.sentence_spans = []
            return doc

    def _enhance_tokens(self, doc):
        """
        Add spaCy-like attributes to Stanza tokens for compatibility.
        
        Args:
            doc: Compatible document with Stanza content
        """
        # Check which type of document we're dealing with
        if hasattr(doc, 'stanza_doc'):
            # This is our CompatibleDoc from Stanza
            if hasattr(doc, 'words') and doc.words:
                for word in doc.words:
                    # Add spaCy-like attributes
                    if not hasattr(word, 'pos_'):
                        word.pos_ = getattr(word, 'pos', "")
                    if not hasattr(word, 'lemma_'):
                        word.lemma_ = getattr(word, 'lemma', "")
                    if not hasattr(word, 'i'):
                        word.i = getattr(word, 'id', 0)
                    
                    # Additional attributes for compatibility
                    if not hasattr(word, 'is_stop'):
                        word.is_stop = False
                    if not hasattr(word, 'is_alpha'):
                        word.is_alpha = word.text.isalpha() if hasattr(word, 'text') else False
                    if not hasattr(word, 'ent_type_'):
                        word.ent_type_ = ""
                    if not hasattr(word, 'ent_iob_'):
                        word.ent_iob_ = "O"
                    if not hasattr(word, 'dep_'):
                        word.dep_ = ""
                    if not hasattr(word, 'head'):
                        word.head = word
            
            # If we have tokens but no words array
            elif hasattr(doc, 'tokens') and doc.tokens:
                for token in doc.tokens:
                    self._ensure_token_attributes(token)
        
        elif hasattr(doc, 'sentences'):
            # This is a direct Stanza document
            for sent in doc.sentences:
                if hasattr(sent, 'tokens'):
                    for token in sent.tokens:
                        if hasattr(token, 'words'):
                            for word in token.words:
                                self._ensure_token_attributes(word)
                        else:
                            self._ensure_token_attributes(token)
        
        elif hasattr(doc, 'tokens'):
            # Direct token list
            for token in doc.tokens:
                self._ensure_token_attributes(token)
    
    def _ensure_token_attributes(self, token):
        """Helper method to ensure all tokens have necessary attributes."""
        # Add spaCy-like attributes
        if not hasattr(token, 'pos_'):
            token.pos_ = getattr(token, 'pos', "")
        if not hasattr(token, 'lemma_'):
            token.lemma_ = getattr(token, 'lemma', "")
        if not hasattr(token, 'i'):
            token.i = getattr(token, 'id', 0)
        
        # Additional attributes
        if not hasattr(token, 'is_stop'):
            token.is_stop = False
        if not hasattr(token, 'is_alpha'):
            token.is_alpha = token.text.isalpha() if hasattr(token, 'text') else False
        if not hasattr(token, 'ent_type_'):
            token.ent_type_ = ""
        if not hasattr(token, 'ent_iob_'):
            token.ent_iob_ = "O"
        if not hasattr(token, 'dep_'):
            token.dep_ = ""
        if not hasattr(token, 'head'):
            token.head = token


def get_model(language_code: str) -> Optional[Any]:
    """
    Get the appropriate NLP model for a language.
    
    Args:
        language_code: ISO language code
        
    Returns:
        spaCy Language model or StanzaWrapper
    """
    # Set up logging
    logger = logging.getLogger(__name__)
    
    # Handle empty language code FIRST
    if not language_code:
        logger.warning("Empty language code encountered, using multilingual model")
        language_code = 'xx'

    # Return cached model if available
    if language_code in _loaded_models:
        return _loaded_models[language_code]

    # Handle Ancient Greek
    if language_code == 'grc':
        try:
            # Check if Stanza is installed
            if importlib.util.find_spec("stanza") is None:
                logger.error("Stanza not installed. Please install with: pip install stanza")
                return None
                
            # Import the Stanza wrapper - use absolute import if relative fails
            try:
                from .stanza_wrapper import get_stanza_greek_processor
            except ImportError:
                # Try to find the module in the current package
                try:
                    current_module = __name__.split(".")[0]
                    module_path = f"{current_module}.stanza_wrapper"
                    stanza_module = importlib.import_module(module_path)
                    get_stanza_greek_processor = stanza_module.get_stanza_greek_processor
                except ImportError:
                    # Last resort - direct import (assumes it's in PYTHONPATH)
                    from stanza_wrapper import get_stanza_greek_processor
            
            # Get the processor
            processor = get_stanza_greek_processor()
            
            # Check if we got a StanzaWrapper or just a pipeline
            if hasattr(processor, '__call__') and not isinstance(processor, StanzaWrapper):
                # Wrap the processor in our StanzaWrapper
                wrapped_processor = StanzaWrapper(processor, 'grc')
                _loaded_models['grc'] = wrapped_processor
                logger.info("Successfully loaded Stanza processor for Ancient Greek")
                return wrapped_processor
            else:
                # Already a wrapper or similar
                _loaded_models['grc'] = processor
                logger.info("Successfully loaded Stanza processor for Ancient Greek")
                return processor
            
        except Exception as e:
            logger.error(f"Error loading Stanza processor for Ancient Greek: {e}")
            logger.warning("Falling back to multilingual model for Ancient Greek")
            language_code = 'xx'

    # Get model name AFTER handling all cases
    model_name = LANGUAGE_MODELS.get(language_code)
    if not model_name:
        logger.warning(f"No model defined for language code: {language_code}")
        # Fall back to multilingual model
        model_name = LANGUAGE_MODELS['xx']

    try:
        nlp = spacy.load(model_name)
        _loaded_models[language_code] = nlp
        logger.info(f"Successfully loaded model {model_name} for language {language_code}")
        return nlp
    except OSError:
        logger.error(f"Model {model_name} not found. Trying fallback...")
        
        # Try multilingual model as fallback
        if model_name != LANGUAGE_MODELS['xx']:
            try:
                fallback_model = LANGUAGE_MODELS['xx']
                nlp = spacy.load(fallback_model)
                _loaded_models[language_code] = nlp
                logger.warning(f"Using fallback model {fallback_model} for language {language_code}")
                return nlp
            except OSError:
                logger.error(f"Fallback model {fallback_model} not found. Please run: python -m spacy download {fallback_model}")
                return None
        else:
            logger.error(f"Please run: python -m spacy download {model_name}")
            return None