# test_visualizations.py
import os
import sys
import pytest
import matplotlib.pyplot as plt

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.aesop_spacy.visualization.plots.pos_comparison import POSComparisonPlot

@pytest.fixture
def pos_plotter():
    """Create a POSComparisonPlot instance for testing."""
    return POSComparisonPlot(analysis_file='comparison_1.json')

def test_pos_comparison_creation(pos_plotter):
    """Test that the POS comparison plot can be created without errors."""
    fig, ax = pos_plotter.plot_pos_distribution()
    
    # Check basic properties
    assert fig is not None
    assert ax is not None
    assert 'Part-of-Speech Distribution' in ax.get_title()
    
    # Clean up
    plt.close(fig)

def test_pos_comparison_languages(pos_plotter):
    """Test that the POS comparison plot works with specific languages."""
    # Test with a subset of languages
    fig, ax = pos_plotter.plot_pos_distribution(languages=['en', 'nl'])
    
    # Check that only the specified languages are in the legend
    legend_texts = [text.get_text() for text in ax.get_legend().get_texts()]
    assert set(legend_texts) == {'en', 'nl'}
    
    # Clean up
    plt.close(fig)