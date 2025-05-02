import unittest
import json
from pathlib import Path
from aesop_spacy.analysis.stats_analyzer import StatsAnalyzer


class TestStatsAnalyzer(unittest.TestCase):
    """Test cases for StatsAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path('test_output')
        self.test_dir.mkdir(exist_ok=True)
        self.analyzer = StatsAnalyzer(self.test_dir)
        
        # Create a sample fable for testing
        self.sample_fable_en = {
            'fable_id': '1',
            'language': 'en',
            'body': 'The fox saw a crow sitting on a tree with a piece of cheese in its beak. The fox wanted the cheese, so he said to the crow, "What a beautiful voice you have! Won\'t you sing for me?" The crow was flattered and opened his beak to sing, dropping the cheese. The fox quickly grabbed the cheese and said, "That\'s what happens when you believe everything you hear."',
            'sentences': [
                {'text': 'The fox saw a crow sitting on a tree with a piece of cheese in its beak.', 
                 'pos_tags': [('The', 'DET'), ('fox', 'NOUN'), ('saw', 'VERB'), ('a', 'DET'), 
                              ('crow', 'NOUN'), ('sitting', 'VERB'), ('on', 'ADP'), ('a', 'DET'), 
                              ('tree', 'NOUN'), ('with', 'ADP'), ('a', 'DET'), ('piece', 'NOUN'), 
                              ('of', 'ADP'), ('cheese', 'NOUN'), ('in', 'ADP'), ('its', 'PRON'), 
                              ('beak', 'NOUN'), ('.', 'PUNCT')]},
                {'text': 'The fox wanted the cheese, so he said to the crow, "What a beautiful voice you have!',
                 'pos_tags': [('The', 'DET'), ('fox', 'NOUN'), ('wanted', 'VERB'), ('the', 'DET'), 
                              ('cheese', 'NOUN'), (',', 'PUNCT'), ('so', 'ADV'), ('he', 'PRON'), 
                              ('said', 'VERB'), ('to', 'ADP'), ('the', 'DET'), ('crow', 'NOUN'), 
                              (',', 'PUNCT'), ('"', 'PUNCT'), ('What', 'PRON'), ('a', 'DET'), 
                              ('beautiful', 'ADJ'), ('voice', 'NOUN'), ('you', 'PRON'), 
                              ('have', 'VERB'), ('!', 'PUNCT')]}
            ]
        }
        
        self.sample_fable_es = {
            'fable_id': '1',
            'language': 'es',
            'body': 'El zorro vio un cuervo sentado en un árbol con un trozo de queso en el pico. El zorro quería el queso, así que le dijo al cuervo: "¡Qué hermosa voz tienes! ¿No cantarías para mí?" El cuervo se sintió halagado y abrió el pico para cantar, dejando caer el queso. El zorro rápidamente agarró el queso y dijo: "Esto es lo que sucede cuando crees todo lo que escuchas."',
            'sentences': [
                {'text': 'El zorro vio un cuervo sentado en un árbol con un trozo de queso en el pico.',
                 'pos_tags': [('El', 'DET'), ('zorro', 'NOUN'), ('vio', 'VERB'), ('un', 'DET'), 
                              ('cuervo', 'NOUN'), ('sentado', 'VERB'), ('en', 'ADP'), ('un', 'DET'), 
                              ('árbol', 'NOUN'), ('con', 'ADP'), ('un', 'DET'), ('trozo', 'NOUN'), 
                              ('de', 'ADP'), ('queso', 'NOUN'), ('en', 'ADP'), ('el', 'DET'), 
                              ('pico', 'NOUN'), ('.', 'PUNCT')]}
            ]
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
    
    def test_word_frequency(self):
        """Test word frequency analysis."""
        result = self.analyzer.word_frequency(self.sample_fable_en)
        
        # Basic validation of the result structure
        self.assertIn('total_word_count', result)
        self.assertIn('unique_word_count', result)
        self.assertIn('top_words', result)
        self.assertIn('hapax_legomena_count', result)
        self.assertIn('entropy', result)
        
        # Check if we have a reasonable number of words detected
        self.assertGreater(result['total_word_count'], 30)
        self.assertGreater(result['unique_word_count'], 20)
        
        # Check that common words are present in top words
        top_words = [item['word'] for item in result['top_words']]
        self.assertTrue(any(word in top_words for word in ['fox', 'crow', 'cheese']))
    
    def test_compare_word_usage(self):
        """Test comparison of word usage across languages."""
        result = self.analyzer.compare_word_usage(self.fables_by_language)
        
        # Validate result structure
        self.assertIn('languages', result)
        self.assertIn('word_counts', result)
        self.assertIn('frequent_words', result)
        self.assertIn('hapax_comparison', result)
        self.assertIn('correlation', result)
        
        # Check if we have entries for both languages
        self.assertIn('en', result['word_counts'])
        self.assertIn('es', result['word_counts'])
        
        # Check correlation data
        self.assertIn('en_es', result['correlation'])
        self.assertIn('cosine_similarity', result['correlation']['en_es'])
    
    def test_chi_square_test(self):
        """Test chi-square test for POS distribution."""
        result = self.analyzer.chi_square_test(self.fables_by_language, feature='pos')
        
        # Check result structure
        self.assertIn('languages', result)
        self.assertIn('feature', result)
        self.assertIn('contingency_table', result)
        self.assertEqual(result['feature'], 'pos')
        
        # Check if we have data for both languages in the contingency table
        self.assertIn('en', result['contingency_table'])
        self.assertIn('es', result['contingency_table'])
    
    def test_pos_variation(self):
        """Test POS variation analysis."""
        result = self.analyzer.pos_variation(self.sample_fable_en)
        
        # Check result structure
        self.assertIn('total_pos_tags', result)
        self.assertIn('unique_pos_tags', result)
        self.assertIn('pos_distribution', result)
        self.assertIn('most_common_bigrams', result)
        
        # Verify POS counts
        self.assertGreater(result['total_pos_tags'], 0)
        self.assertGreater(result['unique_pos_tags'], 0)
        
        # Check if common POS tags are present
        self.assertIn('NOUN', result['pos_distribution'])
        self.assertIn('VERB', result['pos_distribution'])
    
    def test_compare_lexical_diversity(self):
        """Test comparison of lexical diversity across languages."""
        result = self.analyzer.compare_lexical_diversity(self.fables_by_language)
        
        # Check result structure
        self.assertIn('languages', result)
        self.assertIn('type_token_ratio', result)
        self.assertIn('hapax_ratio', result)
        self.assertIn('entropy', result)
        
        # Check if we have data for both languages
        self.assertIn('en', result['type_token_ratio'])
        self.assertIn('es', result['type_token_ratio'])
        
        # Type-token ratio should be between 0 and 1
        self.assertGreater(result['type_token_ratio']['en'], 0)
        self.assertLess(result['type_token_ratio']['en'], 1)
    
    def test_save_analysis(self):
        """Test saving analysis results to file."""
        test_results = {'test': 'data', 'value': 123}
        self.analyzer.save_analysis('test_fable', 'test_analysis', test_results)
        
        # Check if file was created
        output_path = self.test_dir / 'stats' / 'test_fable_test_analysis.json'
        self.assertTrue(output_path.exists())
        
        # Check if file content matches
        with open(output_path, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
        
        self.assertEqual(saved_data, test_results)


if __name__ == '__main__':
    unittest.main()