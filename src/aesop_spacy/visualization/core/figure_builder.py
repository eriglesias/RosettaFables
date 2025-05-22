#figure_builder.py
"""
Figure building utilities for data visualization.

This module provides base classes and utilities for creating standardized
data visualizations from analysis results. It handles loading data, applying
consistent styling, and saving figures to the appropriate directory.
"""

import json
import os
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import seaborn as sns


class FigureBuilder:
    """Base class for creating visualizations of analysis results."""
  
    def __init__(self, theme='default', fig_size=(10, 6), use_mock_data=True, output_dir=None):
        self.logger = logging.getLogger(__name__)
        self.theme = theme
        self.fig_size = fig_size
        self.use_mock_data = use_mock_data
        self.output_dir = output_dir
        # Simple, direct palette definitions
        self.palettes = {
            'languages': sns.color_palette('colorblind', 10),
            'categories': sns.color_palette('Set2', 10),
            'sequential': sns.color_palette('Blues', 10),
            'diverging': sns.color_palette('RdBu_r', 10)
        }
        
        # Initialize mock data for POS distributions
        self.pos_mock_data = {
            'en': {
                'NOUN': 11.96, 'VERB': 13.40, 'PUNCT': 12.48, 'PRON': 12.09, 
                'PROPN': 9.59, 'DET': 8.12, 'ADP': 7.56, 'ADJ': 6.45
            },
            'de': {
                'NOUN': 16.23, 'PUNCT': 13.80, 'DET': 12.68, 'ADV': 12.61, 
                'VERB': 11.56, 'PRON': 10.84, 'ADP': 6.96, 'CCONJ': 4.07
            },
            'nl': {
                'VERB': 14.60, 'NOUN': 13.46, 'ADP': 11.34, 'PRON': 10.94, 
                'PUNCT': 9.51, 'DET': 9.22, 'ADV': 7.89, 'ADJ': 7.11
            },
            'es': {
                'PUNCT': 15.46, 'VERB': 14.77, 'NOUN': 14.37, 'ADP': 12.69, 
                'DET': 11.99, 'PRON': 7.18, 'ADJ': 6.87, 'ADV': 5.98
            },
            'grc': {
                'VERB': 20.27, 'NOUN': 19.17, 'DET': 12.74, 'PUNCT': 11.53, 
                'ADJ': 8.86, 'ADP': 8.21, 'PRON': 7.32, 'ADV': 5.41
            }
        }
        
        self._apply_theme()
    
    def _apply_theme(self):
        """Apply theme settings to matplotlib."""
        themes = {
            'default': lambda: (
                sns.set_style('whitegrid'),
                setattr(plt.rcParams, 'axes.facecolor', 'white'),
                setattr(plt.rcParams, 'figure.facecolor', 'white')
            ),
            'dark': lambda: (
                sns.set_style('darkgrid'),
                setattr(plt.rcParams, 'axes.facecolor', '#2E3440'),
                setattr(plt.rcParams, 'figure.facecolor', '#2E3440'),
                setattr(plt.rcParams, 'text.color', 'white'),
                setattr(plt.rcParams, 'axes.labelcolor', 'white'),
                setattr(plt.rcParams, 'xtick.color', 'white'),
                setattr(plt.rcParams, 'ytick.color', 'white')
            ),
            'paper': lambda: (
                sns.set_style('ticks'),
                setattr(plt.rcParams, 'font.family', 'serif')
            )
        }
        
        # Apply theme if it exists, otherwise use default
        themes.get(self.theme, themes['default'])()
    
    def create_figure(self, figsize=None):
        """Create a matplotlib figure with the configured styling."""
        fig, ax = plt.subplots(figsize=figsize or self.fig_size)
        # Register for cleanup
        self._open_figures.append(fig)
        return fig, ax 

    def cleanup_figures(self):
        """Close all figures to prevent memory leaks"""
        for fig in getattr(self, '_open_figures', []):
            plt.close(fig)
        self._open_figures = []

    def load_analysis_data(self, filename):
        """
        Load analysis JSON data from the analysis directory.
        
        Tries multiple potential paths and falls back to mock data if needed.
        
        Args:
            filename: Name of the file to load
            
        Returns:
            Dictionary containing the loaded data, or mock data if file not found
        """
        # Use mock data if requested
        if self.use_mock_data and filename.startswith('pos_'):
            lang = filename.split('_')[1].split('.')[0]
            if lang in self.pos_mock_data:
                self.logger.info(f"Using mock POS data for {lang}")
                return self.pos_mock_data.get(lang, {})
        
        # Try multiple potential paths for the file
        potential_paths = [
            # Try from project root
            Path(__file__).resolve().parents[3] / 'data' / 'analysis' / filename,
            Path(__file__).resolve().parents[3] / 'data_handled' / 'analysis' / filename,
            Path(__file__).resolve().parents[3] / 'data' / 'data_handled' / 'analysis' / filename,
            
            # Try from current directory
            Path("data_handled/analysis") / filename,
            Path("data/data_handled/analysis") / filename,
            Path("data/analysis") / filename,
            
            # Try from src directory
            Path("src/data/data_handled/analysis") / filename,
            Path("src/data/analysis") / filename,
            
            # Try from the source file location
            Path(__file__).resolve().parent.parent.parent.parent / "data_handled" / "analysis" / filename,
        ]
        
        # Try each path
        for file_path in potential_paths:
            if file_path.exists():
                self.logger.info(f"Found analysis file at: {file_path}")
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except json.JSONDecodeError as e:
                    self.logger.error(f"Invalid JSON in {file_path}: {e}")
                except IOError as e:
                    self.logger.error(f"IO error reading {file_path}: {e}")
                    
                # If we got to this point, there was an error in loading, try next path
                continue
        
        # If we get here, we couldn't find or load the file
        self.logger.warning(f"File not found in any search paths: {filename}")
        
        # Fall back to mock data for POS files
        if filename.startswith('pos_'):
            lang = filename.split('_')[1].split('.')[0]
            if lang in self.pos_mock_data:
                self.logger.info(f"Using mock POS data for {lang}")
                return self.pos_mock_data.get(lang, {})
        
        # Return empty dictionary if all else fails
        return {}
    
    def save_figure(self, fig, filename, dpi=300):
        """
        Save figure to the figures directory.
        
        Args:
            fig: Matplotlib figure to save
            filename: Filename for the saved figure
            dpi: DPI (dots per inch) resolution for the saved figure
            
        Returns:
            Boolean indicating success
        """
        # If an explicit output directory was provided, use it
        if self.output_dir is not None:
            try:
                os.makedirs(self.output_dir, exist_ok=True)
                output_path = Path(self.output_dir) / filename
                fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
                self.logger.info(f"Figure saved: {output_path}")
                return True
            except (ValueError, IOError) as e:
                self.logger.error(f"Error saving to explicit path {output_path}: {e}")
                # Fall through to use fallback paths
        
        # Fallback to other paths if output_dir wasn't specified or failed
        # Try standard path first
        standard_path = Path("/Users/enriqueviv/Coding/coding_projects/data_science_projects/ds_sandbox/nlp_sandbox/aesop_spacy/data/data_handled/figures")
        try:
            os.makedirs(standard_path, exist_ok=True)
            output_path = standard_path / filename
            fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
            self.logger.info(f"Figure saved to standard path: {output_path}")
            return True
        except (ValueError, IOError) as e:
            self.logger.error(f"Error saving to standard path: {e}")
            
            # Last resort: save to current directory
            try:
                output_path = Path('.') / filename
                fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
                self.logger.info(f"Figure saved to current directory: {output_path}")
                return True
            except (ValueError, IOError) as e:
                self.logger.error(f"Failed to save figure to any location: {e}")
                return False