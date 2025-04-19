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
        