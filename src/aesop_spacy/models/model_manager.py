import spacy
from typing import Optional
import logging

# Define language-model mapping
LANGUAGE_MODELS = {
    'nl': 'nl_core_news_lg',    # Dutch
    'de': 'de_dep_news_trf',    # German
    'en': 'en_core_web_lg',     # English
    'es': 'es_core_news_md',  
    'xx': 'xx_sent_ud_sm'       # Multilingual (fallback)
}

# Cache loaded models
_loaded_models = {}

def get_model(language_code: str) -> Optional[spacy.language.Language]:
    """
    Get spaCy model for the specified language.
    
    Args:
        language_code: ISO language code (nl, de, en, es, grc)
        
    Returns:
        Loaded spaCy model or None if not available
    """
    # Handle Ancient Greek specially
    if language_code == 'grc':
        # For Ancient Greek, use the multilingual model
        language_code = 'xx'

    # Return cached model if already loaded
    if language_code in _loaded_models:
        return _loaded_models[language_code]

    # Get model name
    model_name = LANGUAGE_MODELS.get(language_code)
    if not model_name:
        logging.warning(f"No model defined for language code: {language_code}")
        # Fall back to multilingual model
        model_name = LANGUAGE_MODELS['xx']

    # Load the model
    try:
        nlp = spacy.load(model_name)
        _loaded_models[language_code] = nlp
        return nlp
    except OSError:
        logging.error(f"Model {model_name} not found. Try running python -m spacy download {model_name}")
        return None