"""
Tests for Ancient Greek processing using Stanza.
This test suite isolates and diagnoses the Greek dependency parsing issues.
"""
import unittest
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the modules we need to test
try:
    from src.aesop_spacy.models.model_manager import get_model
    from src.aesop_spacy.models.stanza_wrapper import GreekProcessor
    MODEL_IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Model imports failed: {e}")
    MODEL_IMPORTS_AVAILABLE = False

# Check if Stanza is available
try:
    import stanza
    STANZA_AVAILABLE = True
except ImportError:
    STANZA_AVAILABLE = False


class TestGreekProcessing(unittest.TestCase):
    """Test suite for diagnosing Greek processing issues."""
    
    @classmethod
    def setUpClass(cls):
        """Set up logging and test data."""
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        cls.logger = logging.getLogger(__name__)
        
        # Greek text from your actual fables
        cls.test_texts = {
            'simple': "Λύκος θεασάμενος ἄρνα",
            'complex': """Λύκος θεασάμενος ἄρνα ἀπό τινος ποταμοῦ πίνοντα, τοῦτον ἐβουλήθη 
                         μετά τινος εὐλόγου αἰτίας καταθοινήσασθαι.""",
            'full_fable': """Λύκος θεασάμενος ἄρνα ἀπό τινος ποταμοῦ πίνοντα, τοῦτον ἐβουλήθη 
                           μετά τινος εὐλόγου αἰτίας καταθοινήσασθαι. Διόπερ στὰς ἀνωτέρω ᾐτιᾶτο 
                           αὐτὸν ὡς θολοῦντα τὸ ὕδωρ καὶ πιεῖν αὐτὸν μὴ ἐῶντα."""
        }

    @unittest.skipIf(not STANZA_AVAILABLE, "Stanza not installed")
    def test_01_stanza_installation(self):
        """Test if Stanza is properly installed and can import."""
        self.logger.info("=== Testing Stanza Installation ===")
        
        import stanza
        self.logger.info(f"✅ Stanza version: {stanza.__version__}")
        
        # Test if we can access the download function
        self.assertTrue(hasattr(stanza, 'download'))
        self.assertTrue(hasattr(stanza, 'Pipeline'))

    @unittest.skipIf(not STANZA_AVAILABLE, "Stanza not installed") 
    def test_02_greek_model_download(self):
        """Test downloading the Greek model."""
        self.logger.info("=== Testing Greek Model Download ===")
        
        import stanza
        
        try:
            # Try to download Greek model
            self.logger.info("Attempting to download Greek model...")
            stanza.download('grc', verbose=True)
            self.logger.info("✅ Greek model download completed")
            
        except Exception as e:
            self.logger.error(f"❌ Greek model download failed: {e}")
            self.fail(f"Could not download Greek model: {e}")

    @unittest.skipIf(not STANZA_AVAILABLE, "Stanza not installed")
    def test_03_basic_greek_pipeline(self):
        """Test creating a basic Greek pipeline without dependencies."""
        self.logger.info("=== Testing Basic Greek Pipeline ===")
        
        import stanza
        
        try:
            # Create basic pipeline
            nlp = stanza.Pipeline('grc', processors='tokenize,pos,lemma', verbose=True)
            self.logger.info("✅ Basic Greek pipeline created")
            
            # Test with simple text
            doc = nlp(self.test_texts['simple'])
            self.logger.info(f"✅ Processed text: {self.test_texts['simple']}")
            
            # Check what we got
            self.assertTrue(hasattr(doc, 'sentences'))
            self.assertGreater(len(doc.sentences), 0)
            
            # Check sentence structure
            sent = doc.sentences[0]
            self.assertTrue(hasattr(sent, 'words'))
            self.assertGreater(len(sent.words), 0)
            
            # Log what we found
            for word in sent.words:
                self.logger.info(f"  Word: {word.text}, POS: {word.upos}, Lemma: {word.lemma}")
            
        except Exception as e:
            self.logger.error(f"❌ Basic Greek pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            self.fail(f"Basic Greek pipeline failed: {e}")

    @unittest.skipIf(not STANZA_AVAILABLE, "Stanza not installed")
    def test_04_greek_dependency_parsing(self):
        """Test Greek pipeline WITH dependency parsing - this is the critical test."""
        self.logger.info("=== Testing Greek Dependency Parsing ===")
        
        import stanza
        
        try:
            # Try to create pipeline with dependency parsing
            self.logger.info("Creating Greek pipeline with dependency parsing...")
            nlp = stanza.Pipeline(
                'grc', 
                processors='tokenize,pos,lemma,depparse',  # This is what we need!
                verbose=True
            )
            self.logger.info("✅ Greek pipeline with depparse created")
            
            # Test with complex text
            doc = nlp(self.test_texts['complex'])
            self.logger.info("✅ Processed complex Greek text")
            
            # Count dependencies
            total_deps = 0
            for sent in doc.sentences:
                for word in sent.words:
                    if hasattr(word, 'head') and word.head > 0:
                        total_deps += 1
                        self.logger.info(f"  Dependency: {word.text} --{word.deprel}--> "
                                       f"{sent.words[word.head-1].text}")
            
            self.logger.info(f"✅ Found {total_deps} dependencies")
            
            # This is the critical assertion - we MUST have dependencies
            self.assertGreater(total_deps, 0, 
                             "No dependencies found - this is why your syntax analysis fails!")
            
        except Exception as e:
            self.logger.error(f"❌ Greek dependency parsing failed: {e}")
            self.logger.error("This is likely why your Greek syntax analysis produces no output!")
            
            # Try to provide helpful diagnostic info
            if "depparse" in str(e).lower():
                self.logger.error("💡 The Greek model might not support dependency parsing")
            elif "model" in str(e).lower():
                self.logger.error("💡 The Greek model files might be corrupted or missing")
            
            import traceback
            traceback.print_exc()
            self.fail(f"Greek dependency parsing failed: {e}")

    @unittest.skipIf(not MODEL_IMPORTS_AVAILABLE, "Model imports not available")
    def test_05_pipeline_integration(self):
        """Test Greek processing through your actual pipeline."""
        self.logger.info("=== Testing Pipeline Integration ===")
        
        # Test your model manager
        greek_model = get_model('grc')
        
        if greek_model is None:
            self.fail("get_model('grc') returned None - check your model_manager.py")
        
        self.logger.info(f"✅ Got Greek model from pipeline: {type(greek_model)}")
        
        # Test processing
        try:
            doc = greek_model(self.test_texts['simple'])
            self.logger.info("✅ Greek text processed through pipeline")
            
            # Check document format
            self.logger.info(f"Document type: {type(doc)}")
            
            # Check for dependencies in whatever format we got back
            deps_found = 0
            
            if hasattr(doc, 'sentences'):  # Stanza format
                for sent in doc.sentences:
                    if hasattr(sent, 'words'):
                        for word in sent.words:
                            if hasattr(word, 'head') and word.head > 0:
                                deps_found += 1
                self.logger.info(f"Dependencies found (Stanza format): {deps_found}")
                
            elif hasattr(doc, 'sents'):  # spaCy format  
                for sent in doc.sents:
                    for token in sent:
                        if hasattr(token, 'dep_') and token.dep_ not in ['', 'ROOT']:
                            deps_found += 1
                self.logger.info(f"Dependencies found (spaCy format): {deps_found}")
                
            else:
                self.logger.warning(f"Unknown document format: {type(doc)}")
                # Check if it's your CompatibleDoc
                if hasattr(doc, 'sentence_spans'):
                    self.logger.info("Found CompatibleDoc format")
                    for span in doc.sentence_spans:
                        self.logger.info(f"  Span: {span.text}")
            
            # The critical check - are we getting dependencies through the pipeline?
            if deps_found == 0:
                self.logger.warning("⚠️ No dependencies found through pipeline!")
                self.logger.warning("This explains why your Greek syntax files are empty!")
            else:
                self.logger.info(f"✅ Found {deps_found} dependencies through pipeline")
                
        except Exception as e:
            self.logger.error(f"❌ Pipeline integration failed: {e}")
            import traceback
            traceback.print_exc()
            self.fail(f"Pipeline integration failed: {e}")

    @unittest.skipIf(not MODEL_IMPORTS_AVAILABLE, "Model imports not available")
    def test_06_greek_processor_direct(self):
        """Test the GreekProcessor class directly."""
        self.logger.info("=== Testing GreekProcessor Directly ===")
        
        try:
            processor = GreekProcessor()
            self.assertIsNotNone(processor.processor, "GreekProcessor.processor should not be None")
            
            # Test processing
            result = processor(self.test_texts['simple'])
            
            self.logger.info(f"✅ GreekProcessor returned: {type(result)}")
            self.logger.info(f"Result text: {result.text}")
            
            # Check structure
            if hasattr(result, 'sentences') or hasattr(result, 'sentence_spans'):
                sentences = getattr(result, 'sentences', getattr(result, 'sentence_spans', []))
                self.logger.info(f"Found {len(sentences)} sentences")
                
                for i, sent in enumerate(sentences):
                    self.logger.info(f"  Sentence {i}: {getattr(sent, 'text', 'No text')}")
                    
        except Exception as e:
            self.logger.error(f"❌ GreekProcessor test failed: {e}")
            import traceback
            traceback.print_exc()
            self.fail(f"GreekProcessor failed: {e}")

    def test_07_diagnosis_summary(self):
        """Provide a summary of what we found."""
        self.logger.info("=== DIAGNOSIS SUMMARY ===")
        
        if not STANZA_AVAILABLE:
            self.logger.error("❌ STANZA NOT INSTALLED - Install with: pip install stanza")
            return
            
        if not MODEL_IMPORTS_AVAILABLE:
            self.logger.error("❌ MODEL IMPORTS FAILED - Check your import paths")
            return
            
        self.logger.info("✅ Basic setup appears to be working")
        self.logger.info("🔍 Run individual tests to see where Greek processing breaks")
        self.logger.info("💡 Most likely issue: Greek model doesn't support dependency parsing")


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)