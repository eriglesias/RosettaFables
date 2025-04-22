import unittest
import json
from pathlib import Path
import sys
import os

from aesop_spacy.analysis.nlp_techniques import NLPTechniques

class TestNLPTechniques(unittest.TestCase):
    """Test cases for NLPTechniques class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path('test_output')
        self.test_dir.mkdir(exist_ok=True)
        self.analyzer = NLPTechniques(self.test_dir)
        
        # Create sample fables for testing
        self.sample_fable_en = {
            'fable_id': '1',
            'language': 'en',
            'body': 'The fox saw a crow sitting on a tree with a piece of cheese in its beak. The fox wanted the cheese, so he said to the crow, "What a beautiful voice you have! Won\'t you sing for me?" The crow was flattered and opened his beak to sing, dropping the cheese. The fox quickly grabbed the cheese and said, "That\'s what happens when you believe everything you hear."'
        }
        
        self.sample_fable_es = {
            'fable_id': '1',
            'language': 'es',
            'body': 'El zorro vio un cuervo sentado en un árbol con un trozo de queso en el pico. El zorro quería el queso, así que le dijo al cuervo: "¡Qué hermosa voz tienes! ¿No cantarías para mí?" El cuervo se sintió halagado y abrió el pico para cantar, dejando caer el queso. El zorro rápidamente agarró el queso y dijo: "Esto es lo que sucede cuando crees todo lo que escuchas."'
        }
        
        # Sample fables for multi-language comparison
        self.fables_by_language = {
            'en': self.sample_fable_en,
            'es': self.sample_fable_es
        }
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove test directory if needed
        # If you want to keep test outputs for inspection, comment this out
        import shutil
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_tfidf_analysis(self):
        """Test TF-IDF analysis."""
        try:
            result = self.analyzer.tfidf_analysis(self.fables_by_language)
            
            # Basic validation of the result structure
            self.assertIn('document_info', result)
            self.assertIn('top_terms_overall', result)
            
            # Check if we have entries for both fables
            self.assertEqual(len(result['document_info']), 2)
            
            # Check if top terms are present
            self.assertTrue(len(result['top_terms_overall']) > 0)
            
            # Ensure each document has top terms
            for doc_info in result['document_info']:
                self.assertIn('top_terms', doc_info)
                self.assertTrue(len(doc_info['top_terms']) > 0)
        except ImportError:
            self.skipTest("scikit-learn not installed")
    
    def test_topic_modeling(self):
        """Test topic modeling."""
        try:
            # Test LDA
            result_lda = self.analyzer.topic_modeling(
                self.fables_by_language, n_topics=2, method='lda'
            )
            
            # Basic validation
            self.assertIn('topics', result_lda)
            self.assertIn('document_topics', result_lda)
            self.assertEqual(result_lda['model_type'], 'LDA')
            self.assertEqual(len(result_lda['topics']), 2)  # 2 topics
            
            # Test NMF
            result_nmf = self.analyzer.topic_modeling(
                self.fables_by_language, n_topics=2, method='nmf'
            )
            
            self.assertEqual(result_nmf['model_type'], 'NMF')
            self.assertEqual(len(result_nmf['topics']), 2)
        except ImportError:
            self.skipTest("scikit-learn not installed")
    
    def test_word_embeddings(self):
        """Test word embeddings."""
        try:
            # Skip if gensim is not available
            import gensim
            
            result = self.analyzer.word_embeddings(
                self.fables_by_language, embedding_size=50, window=3
            )
            
            # Basic validation
            self.assertIn('language_models', result)
            self.assertIn('similar_words', result)
            
            # Should have models for both languages
            self.assertIn('en', result['language_models'])
            self.assertIn('es', result['language_models'])
            
            # Each language model should have vocabulary information
            self.assertIn('vocabulary_size', result['language_models']['en'])
            self.assertIn('most_common_words', result['language_models']['en'])
            
        except ImportError:
            self.skipTest("gensim not installed")
    
    def test_save_analysis(self):
        """Test saving analysis results to file."""
        test_results = {'test': 'data', 'value': 123}
        self.analyzer.save_analysis('test_fable', 'test_analysis', test_results)
        
        # Check if file was created
        output_path = self.test_dir / 'nlp' / 'test_fable_test_analysis.json'
        self.assertTrue(output_path.exists())
        
        # Check if file content matches
        with open(output_path, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
        
        self.assertEqual(saved_data, test_results)


if __name__ == '__main__':
    unittest.main()