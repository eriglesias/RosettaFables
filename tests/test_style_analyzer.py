# tests/test_style_analyzer.py
import unittest
from pathlib import Path
from unittest.mock import patch, mock_open

# Note: The class doesn't exist yet, but we're defining how it should work
from src.aesop_spacy.analysis.style_analyzer import StyleAnalyzer

class TestStyleAnalyzer(unittest.TestCase):

    def setUp(self):
        # Create a mock analysis directory
        self.analysis_dir = Path("mock/analysis")
        self.analyzer = StyleAnalyzer(self.analysis_dir)

        # Sample processed fable data
        self.mock_fable = {
            'id': '1',
            'title': 'The Fox and the Grapes',
            'sentences': [
                {'start': 0, 'end': 10, 'text': 'A fox saw some grapes.'},
                {'start': 11, 'end': 25, 'text': 'He tried to reach them.'},
                {'start': 26, 'end': 45, 'text': 'But they were too high for him.'}
            ]
        }

    def test_sentence_complexity(self):
        # Test calculating sentence complexity
        metrics = self.analyzer.sentence_complexity(self.mock_fable)

        # Assertions about what we expect
        self.assertIn('avg_sentence_length', metrics)
        self.assertIn('sentence_count', metrics)
        self.assertEqual(metrics['sentence_count'], 3)

        # For the sentences:
        # "A fox saw some grapes." - 5 tokens
        # "He tried to reach them." - 5 tokens
        # "But they were too high for him." - 7 tokens
        # Average: (5 + 5 + 7) / 3 = 5.67
        self.assertAlmostEqual(metrics['avg_sentence_length'], 5.67, places=2)

    def test_sentence_complexity_clauses(self):
        metrics = self.analyzer.sentence_complexity(self.mock_fable)
        self.assertIn('avg_clauses_per_sentence', metrics)
        self.assertTrue(1.0 <= metrics['avg_clauses_per_sentence'] <= 1.5)

    def test_sentence_complexity_depth(self):
        metrics = self.analyzer.sentence_complexity(self.mock_fable)
        self.assertIn('avg_dependency_depth', metrics)
        self.assertTrue(metrics['avg_dependency_depth'] > 0)

    def test_with_realistic_fable(self):
        """Test with a more realistic fable structure."""
        # Create a mock fable based on a real Dutch fable structure
        realistic_fable = {
            'id': '1',
            'title': 'De Wolf en het Geitje',
            'language': 'nl',
            'body': """Er was eens een klein geitje dat hoorntjes begon te krijgen en daarom dacht dat hij nu al een grote geit was. 
                    Hij liep in de wei, samen met zijn moeder en een grote kudde geiten, en zei tegen iedereen dat hij nu wel 
                    voor zichzelf kon zorgen. Elke avond gingen de geiten naar hun stal om er te slapen. Op een avond bleef 
                    het klein geitje op de wei staan.""",
            'sentences': [
                {
                    'text': 'Er was eens een klein geitje dat hoorntjes begon te krijgen en daarom dacht dat hij nu al een grote geit was.',
                    'pos_tags': [('Er', 'ADV'), ('was', 'VERB'), ('eens', 'ADV'), ('een', 'DET'), ('klein', 'ADJ'), 
                                ('geitje', 'NOUN'), ('dat', 'PRON'), ('hoorntjes', 'NOUN'), ('begon', 'VERB'),
                                ('te', 'PART'), ('krijgen', 'VERB'), ('en', 'CCONJ'), ('daarom', 'ADV'), ('dacht', 'VERB'),
                                ('dat', 'SCONJ'), ('hij', 'PRON'), ('nu', 'ADV'), ('al', 'ADV'), ('een', 'DET'),
                                ('grote', 'ADJ'), ('geit', 'NOUN'), ('was', 'VERB')]
                },
                {
                    'text': 'Hij liep in de wei, samen met zijn moeder en een grote kudde geiten, en zei tegen iedereen dat hij nu wel voor zichzelf kon zorgen.',
                    'pos_tags': [('Hij', 'PRON'), ('liep', 'VERB'), ('in', 'ADP'), ('de', 'DET'), ('wei', 'NOUN'),
                                (',', 'PUNCT'), ('samen', 'ADV'), ('met', 'ADP'), ('zijn', 'PRON'), ('moeder', 'NOUN'),
                                ('en', 'CCONJ'), ('een', 'DET'), ('grote', 'ADJ'), ('kudde', 'NOUN'), ('geiten', 'NOUN'),
                                (',', 'PUNCT'), ('en', 'CCONJ'), ('zei', 'VERB'), ('tegen', 'ADP'), ('iedereen', 'PRON'),
                                ('dat', 'SCONJ'), ('hij', 'PRON'), ('nu', 'ADV'), ('wel', 'ADV'), ('voor', 'ADP'),
                                ('zichzelf', 'PRON'), ('kon', 'AUX'), ('zorgen', 'VERB')]
                }
            ],
            'tokens': [('Er', 0), ('was', 1), ('eens', 2), ('een', 3), ('klein', 4), ('geitje', 5)],
            'lemmas': [('er', 0), ('zijn', 1), ('eens', 2), ('een', 3), ('klein', 4), ('geitje', 5)]
        }
        
        # Test lexical richness
        rich_metrics = self.analyzer.lexical_richness(realistic_fable)
         # Debug prints
        print("\nDEBUG INFO:")
        print(f"Token count: {rich_metrics['token_count']}")
        print(f"Type count: {rich_metrics['type_count']}")
        print(f"TTR: {rich_metrics['ttr']}")
        # Basic assertions
        self.assertIn('ttr', rich_metrics)
        self.assertIn('mattr', rich_metrics)
        self.assertIn('hapax_ratio', rich_metrics)
        self.assertIn('vocab_growth', rich_metrics)
        
        # Test that values are reasonable
        self.assertTrue(0 < rich_metrics['ttr'] <= 1.0)
        self.assertTrue(0 < rich_metrics['mattr'] <= 1.0)
        
        # Test rhetorical devices
        devices = self.analyzer.rhetorical_devices(realistic_fable)
        
        # Basic assertions
        self.assertIn('repetition', devices)
        self.assertIn('alliteration', devices)
        self.assertIn('parallelism', devices)
        self.assertIn('possible_metaphors', devices)