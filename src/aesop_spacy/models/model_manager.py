from typing import Optional, Dict, Any
import logging
import spacy
from pathlib import Path
import importlib.util

# Define language-model mapping
LANGUAGE_MODELS = {
    'nl': 'nl_core_news_lg',    # Dutch
    'de': 'de_dep_news_trf',    # German
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
  
        # Add spaCy-like attributes to the document
        if not hasattr(doc, 'sents'):
            # Make doc.sents point to doc.sentences for compatibility
            doc.sents = doc.sentences
 
        # Add additional properties for compatibility
        if not hasattr(doc, 'ents'):
            doc.ents = []
  
        # Add vector property
        if not hasattr(doc, 'vector'):
            doc.vector = []
    
        # Add any token enhancements needed for pipeline compatibility
        self._enhance_tokens(doc)
      
        return doc

    def _enhance_tokens(self, doc):
        """
        Add spaCy-like attributes to Stanza tokens for compatibility.
        
        Args:
            doc: Stanza document
        """
        # Process each sentence
        for sent in doc.sentences:
            for token in sent.tokens:
                # Add spaCy-like token attributes if they don't exist
                for word in token.words:
                    # These attributes are expected by the processor
                    if not hasattr(word, 'pos_'):
                        word.pos_ = word.pos
                    if not hasattr(word, 'lemma_'):
                        word.lemma_ = word.lemma
                    if not hasattr(word, 'text'):
                        word.text = word.text
                    if not hasattr(word, 'i'):
                        word.i = word.id
                    
                    # Additional attributes that might be needed
                    if not hasattr(word, 'is_stop'):
                        word.is_stop = False
                    if not hasattr(word, 'is_alpha'):
                        word.is_alpha = word.text.isalpha()
                    if not hasattr(word, 'ent_type_'):
                        word.ent_type_ = ""
                    if not hasattr(word, 'ent_iob_'):
                        word.ent_iob_ = "O"


def get_model(language_code: str) -> Optional[Any]:
    """
    Get the appropriate NLP model for a language.
    
    Args:
        language_code: ISO language code
        
    Returns:
        spaCy Language model or StanzaWrapper
    """
    # Handle empty language code FIRST
    if not language_code:
        logging.warning("Empty language code encountered, using multilingual model")
        language_code = 'xx'

    # Return cached model if available
    if language_code in _loaded_models:
        return _loaded_models[language_code]

    # Handle Ancient Greek
    if language_code == 'grc':
        try:
            # Check if Stanza is installed
            if importlib.util.find_spec("stanza") is None:
                logging.error("Stanza not installed. Please install with: pip install stanza")
                return None
                
            # Import the Stanza wrapper
            from .stanza_wrapper import get_stanza_greek_processor
            processor = get_stanza_greek_processor()
            
            # Wrap the processor in our StanzaWrapper
            wrapped_processor = StanzaWrapper(processor, 'grc')
            _loaded_models['grc'] = wrapped_processor
            
            logging.info("Successfully loaded Stanza processor for Ancient Greek")
            return wrapped_processor
            
        except Exception as e:
            logging.error(f"Error loading Stanza processor for Ancient Greek: {e}")
            logging.warning("Falling back to multilingual model for Ancient Greek")
            language_code = 'xx'

    # Get model name AFTER handling all cases
    model_name = LANGUAGE_MODELS.get(language_code)
    if not model_name:
        logging.warning(f"No model defined for language code: {language_code}")
        # Fall back to multilingual model
        model_name = LANGUAGE_MODELS['xx']

    try:
        nlp = spacy.load(model_name)
        _loaded_models[language_code] = nlp
        logging.info(f"Successfully loaded model {model_name} for language {language_code}")
        return nlp
    except OSError:
        logging.error(f"Model {model_name} not found. Trying fallback...")
        
        # Try multilingual model as fallback
        if model_name != LANGUAGE_MODELS['xx']:
            try:
                fallback_model = LANGUAGE_MODELS['xx']
                nlp = spacy.load(fallback_model)
                _loaded_models[language_code] = nlp
                logging.warning(f"Using fallback model {fallback_model} for language {language_code}")
                return nlp
            except OSError:
                logging.error(f"Fallback model {fallback_model} not found. Please run: python -m spacy download {fallback_model}")
                return None
        else:
            logging.error(f"Please run: python -m spacy download {model_name}")
            return None