#model_manager.py
""" Handles models"""
import importlib.util
from typing import Optional, Any
import logging
import spacy


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

def verify_models(languages=None, check_optional=False):
    """
    Verify installed spaCy models for the specified languages.
    
    Args:
        languages: List of language codes to check, or None for default set
        check_optional: If True, check additional optional models
        
    Returns:
        Dict with verification results
    """
    logger = logging.getLogger(__name__)
    
    # Default to checking only required languages if none specified
    if languages is None:
        languages = list(LANGUAGE_MODELS.keys())
    
    # If check_optional, add Greek to the check list
    if check_optional and 'grc' not in languages:
        languages.append('grc')
    
    logger.info("Verifying models for languages: %s", languages)
    
    # Initialize results
    results = {
        'installed': [],
        'missing': [],
        'install_commands': []
    }
    
    # Check each language
    for lang in languages:
        if lang == 'grc':
            # Special handling for Ancient Greek using Stanza
            try:
                greek_processor = _load_greek_processor(logger)
                if greek_processor:
                    results['installed'].append(lang)
                else:
                    results['missing'].append(lang)
                    results['install_commands'].append("pip install stanza")
            except Exception as e:
                logger.error("Error checking Greek model: %s", e)
                results['missing'].append(lang)
                results['install_commands'].append("pip install stanza")
        else:
            # Regular spaCy model check
            model_name = LANGUAGE_MODELS.get(lang)
            if not model_name:
                logger.warning("No model defined for language code: %s", lang)
                continue
                
            try:
                # Attempt to load the model (will be cached if successful)
                model = spacy.load(model_name)
                results['installed'].append(lang)
            except OSError:
                results['missing'].append(lang)
                results['install_commands'].append(f"python -m spacy download {model_name}")
    
    # Format a verification summary
    if results['missing']:
        logger.warning("Missing models for languages: %s", ", ".join(results['missing']))
        for cmd in results['install_commands']:
            logger.info("Install command: %s", cmd)
    else:
        logger.info("All required models are installed")
    
    return results


def get_model(language_code: str) -> Optional[Any]:
    """
    Get the appropriate NLP model for a language.
    
    Args:
        language_code: ISO language code
        
    Returns:
        spaCy Language model or GreekProcessor for Ancient Greek
    """
    # Set up logging
    logger = logging.getLogger(__name__)
    
    # Handle empty language code
    if not language_code:
        logger.warning("Empty language code encountered, using multilingual model")
        language_code = 'xx'

    # Return cached model if available
    if language_code in _loaded_models:
        return _loaded_models[language_code]

    # Handle Ancient Greek (grc) separately
    if language_code == 'grc':
        greek_processor = _load_greek_processor(logger)
        if greek_processor:
            return greek_processor
        logger.warning("Falling back to multilingual model for Ancient Greek")
        language_code = 'xx'

    # Get model name for other languages
    model_name = LANGUAGE_MODELS.get(language_code)
    if not model_name:
        logger.warning("No model defined for language code: %s", language_code)
        # Fall back to multilingual model
        model_name = LANGUAGE_MODELS['xx']

    # Try to load the model
    try:
        nlp = spacy.load(model_name)
        _loaded_models[language_code] = nlp
        logger.info("Successfully loaded model %s for language %s", model_name, language_code)
        return nlp
    except OSError:
        logger.error("Model %s not found. Trying fallback...", model_name)

        # Try multilingual model as fallback
        if model_name != LANGUAGE_MODELS['xx']:
            try:
                fallback_model = LANGUAGE_MODELS['xx']
                nlp = spacy.load(fallback_model)
                _loaded_models[language_code] = nlp
                logger.warning("Using fallback model %s for language %s", fallback_model, language_code)
                return nlp
            except OSError:
                logger.error("Fallback model %s not found. Please run: python -m spacy download %s", 
                            fallback_model, fallback_model)
                return None
        else:
            logger.error("Please run: python -m spacy download %s", model_name)
            return None


def _load_greek_processor(logger):
    """
    Helper function to load the Ancient Greek processor using Stanza.
    
    Args:
        logger: Logger instance
        
    Returns:
        GreekProcessor if successful, None otherwise
    """
    try:
        # Check if Stanza is installed
        if importlib.util.find_spec("stanza") is None:
            logger.error("Stanza not installed. Please install with: pip install stanza")
            return None
   
        # Import the Greek processor
        try:
            from .stanza_wrapper import get_stanza_greek_processor
        except ImportError:
            # Try different import paths if the relative import fails
            try:
                current_module = __name__.split(".")[0]
                module_path = f"{current_module}.stanza_wrapper"
                stanza_module = importlib.import_module(module_path)
                get_stanza_greek_processor = stanza_module.get_stanza_greek_processor
            except ImportError:
                # Direct import as last resort
                from stanza_wrapper import get_stanza_greek_processor

        # Get the processor and cache it
        processor = get_stanza_greek_processor()
        _loaded_models['grc'] = processor
        logger.info("Successfully loaded Greek processor for Ancient Greek")
        return processor

    except (ImportError, ModuleNotFoundError) as e:
        logger.error("Error importing Stanza modules: %s", e)
        return None
    except OSError as e:
        logger.error("Error accessing Stanza resources: %s", e)
        return None