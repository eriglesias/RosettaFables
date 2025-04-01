#class figure builder
import matplotlib.pyplot as plt
import seaborn as sns
class FigureBuilder:
    """Base class for creating visualization figures with consistent styling"""
    # Handle common visualization setup needs
    # Set consistent styling parameters
    # Provide utility methods for loading data
    # Establish patterns used in specific visualizations
    def __init__(self, theme='default', fig_size=(10,6)):
        """Initialize with styling parameters."""
        self.theme = theme
        self.fig_size = fig_size
        self._configure_styling()

    def _configure_styling(self):
        """Apply consistent styling based on a selected theme"""
        # Set the seaborn style
        if self.theme == 'default':
            sns.set_theme(style="whitegrid")
        elif self.theme == 'dark':
            sns.set_theme(style="darkgrid")
        elif self.theme == 'minimal':
            sns.set_theme(style="tikcs")

        self.palettes = {
            'languages': sns.color_palette("husl", 8),
            'pos': sns.color_palette("muted"),
            'categorical': sns.color_palette("Set2"),
            'sequential': sns.color_palette("Yl0Br")
        }

        #font properties
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman', 'Computer Modern Roman']
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['xtick.labelsize'] = 11
        plt.rcParams['ytick.labelsize'] = 11

        plt.rcParams['axes.spines.top'] = False
        plt.rcParams['axes.spines.right'] = False

    def load_analysis_data(self, filename):
        """Load analysis data from the standard location."""

    def create_figure(self):
        """Create a new figure with standard sizing and styling"""

    def save_figure(self, fig, filename, dpi=300):
        """Save a figure to the output locatin with standard paramenters"""
