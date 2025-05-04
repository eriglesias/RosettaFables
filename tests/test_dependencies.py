# test_dependencies.py
import unittest
import logging
import sys
from pathlib import Path
from aesop_spacy.models.model_manager import LANGUAGE_MODELS, get_model

# Add the parent directory to the path so we can import the model_manager
sys.path.append(str(Path(__file__).parent.parent))

class TestModelDependencies(unittest.TestCase):
    """Test dependency parsing capabilities of spaCy models for different languages."""
    
    def setUp(self):
        """Set up logging for the tests."""
        logging.basicConfig(level=logging.INFO, 
                           format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def test_dependency_capabilities(self):
        """Test if all models have dependency parsing capabilities."""
        # Sample sentences for each language
        sample_sentences = {
            'nl': "Dit is een testzin voor afhankelijkheidsanalyse.",
            'de': "Dies ist ein Testsatz für Abhängigkeitsanalyse.",
            'en': "This is a test sentence for dependency analysis.",
            'es': "Esta es una oración de prueba para análisis de dependencias.",
            'xx': "This is a test sentence for the multilingual model.",
            'grc': "τοῦτό ἐστι δοκιμαστικὴ πρότασις διὰ τὴν ἀνάλυσιν τῶν ἐξαρτήσεων."
        }
        
        # Test each language in LANGUAGE_MODELS
        for lang_code in LANGUAGE_MODELS:
            with self.subTest(language=lang_code):
                self._test_single_model(lang_code, sample_sentences.get(lang_code, sample_sentences['en']))
    
    def _test_single_model(self, language, sample_sentence):
        """Test if a specific model has dependency parsing capabilities."""
        try:
            # Load the model using our model_manager
            self.logger.info(f"Testing model for language: {language}")
            nlp = get_model(language)
            
            # Skip if model couldn't be loaded
            if nlp is None:
                self.logger.warning(f"Couldn't load model for {language}, skipping test")
                self.skipTest(f"Model for {language} could not be loaded")
            
            # Check if the parser is in the pipeline
            self.assertIn('parser', nlp.pipe_names, 
                         f"Parser not found in pipeline for {language}")
            
            # Process the sample sentence
            doc = nlp(sample_sentence)
            
            # Check for dependencies
            deps_found = False
            for token in doc:
                if token.dep_ != '':
                    deps_found = True
                    self.logger.info(f"Found dependency: {token.text} -{token.dep_}-> {token.head.text}")
            
            # Ensure dependencies were found
            self.assertTrue(deps_found, f"No dependencies found for {language}")
            
            # Print dependency tree for debugging
            self.logger.info(f"\nDependency Tree for {language}:")
            for token in doc:
                self.logger.info(f"{token.text:{15}} {token.dep_:{10}} {token.head.text:{15}}")
            
        except Exception as e:
            self.logger.error(f"Error testing model for {language}: {e}")
            self.fail(f"Exception occurred while testing {language}: {e}")


if __name__ == '__main__':
    unittest.main()