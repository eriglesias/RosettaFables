from typing import Optional
import logging
import spacy

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

def get_model(language_code: str) -> Optional[spacy.language.Language]:
    """ Docstring """
    # Handle empty language code FIRST
    if not language_code:
        logging.warning("Empty language code encountered, using multilingual model")
        language_code = 'xx'

    # Handle Ancient Greek
    if language_code == 'grc':
        language_code = 'xx'

    # Return cached model if available
    if language_code in _loaded_models:
        return _loaded_models[language_code]

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