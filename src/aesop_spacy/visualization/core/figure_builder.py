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
    
    def __init__(self, theme='default', fig_size=(10, 6)):
        self.logger = logging.getLogger(__name__)
        self.theme = theme
        self.fig_size = fig_size
        
        # Simple, direct palette definitions
        self.palettes = {
            'languages': sns.color_palette('colorblind', 10),
            'categories': sns.color_palette('Set2', 10),
            'sequential': sns.color_palette('Blues', 10),
            'diverging': sns.color_palette('RdBu_r', 10)
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
        return plt.subplots(figsize=figsize or self.fig_size)
    
    def load_analysis_data(self, filename):
        """Load analysis JSON data from the analysis directory."""
        # Get project root in a cleaner way
        project_dir = Path(__file__).resolve().parents[3]
        file_path = project_dir / 'data' / 'analysis' / filename
        
        if not file_path.exists():
            self.logger.warning("File not found: %s", file_path)
            return {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            self.logger.error("Invalid JSON in %s: %s", file_path, e)
            return {}
        except IOError as e:
            self.logger.error("IO error reading %s: %s", file_path, e)
            return {}
    
    def save_figure(self, fig, filename, dpi=300):
        """Save figure to the figures directory."""
        project_dir = Path(__file__).resolve().parents[3]
        figures_dir = project_dir / 'data' / 'figures'
        
        # Create directory if needed
        os.makedirs(figures_dir, exist_ok=True)
        
        # Save the figure
        output_path = figures_dir / filename
        try:
            fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
            self.logger.info("Figure saved: %s", output_path)
            return True
        except (ValueError, IOError) as e:
            self.logger.error("Error saving figure to %s: %s", output_path, e)
            return False