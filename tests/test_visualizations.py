"""
Test visualization components with unittest framework.
"""

import unittest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Update this import path if you've saved the class in pos_distribution.py
from src.aesop_spacy.visualization.plots.pos_comparison import POSDistributionPlot


class TestVisualizations(unittest.TestCase):
    """Test visualization components"""
    
    def test_single_language_plots(self):
        """Test POS distribution for single languages"""
        pos_plotter = POSDistributionPlot()
        
        for lang in ['en', 'de', 'nl', 'es', 'grc']:
            with self.subTest(language=lang):
                fig, ax = pos_plotter.plot_single_language(lang)
                self.assertIsNotNone(fig)
                self.assertIsNotNone(ax)
                # Clean up figure to avoid memory leaks
                fig.clear()
    
    def test_language_comparison(self):
        """Test POS language comparison plot"""
        pos_plotter = POSDistributionPlot()
        fig, ax = pos_plotter.plot_language_comparison()
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
        fig.clear()
    
    def test_heatmap(self):
        """Test POS heatmap visualization"""
        pos_plotter = POSDistributionPlot()
        fig, ax = pos_plotter.plot_pos_heatmap()
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
        fig.clear()


if __name__ == "__main__":
    unittest.main()