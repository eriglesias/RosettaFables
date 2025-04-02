# test_simple_visualization.py
import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.aesop_spacy.visualization.plots.pos_comparison import POSComparisonPlot

def test_pos_comparison_plot():
    """Test creating a POS comparison visualization."""
    print("Creating POS comparison visualization...")
    plotter = POSComparisonPlot(analysis_file='comparison_1.json')
    fig, ax = plotter.plot_pos_distribution()
    
    # Save the figure
    output_path = os.path.join(project_root, 'data', 'visualizations', 'pos_comparison.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plotter.save_figure(fig, 'pos_comparison.png')
    print(f"Visualization saved to: {output_path}")
    
    return fig, ax

# When run directly, execute the test function
if __name__ == "__main__":
    test_pos_comparison_plot()