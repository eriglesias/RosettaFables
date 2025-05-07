from ..core.figure_builder import FigureBuilder
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class POSComparisonPlot(FigureBuilder):
    """Creates visualizations comparing POS tag distributions across languages."""
 
    def __init__(self, analysis_files=None, theme='default', fig_size=(12, 8)):
        """
        Initialize with multiple analysis files or a single file.
        
        Args:
            analysis_files: String or list of strings with analysis filenames
            theme: Visual theme to apply
            fig_size: Default figure size
        """
        super().__init__(theme=theme, fig_size=fig_size)
        
        # Handle both single file and list of files
        if analysis_files is None:
            analysis_files = ['pos_en.json', 'pos_de.json', 'pos_nl.json', 
                            'pos_es.json', 'pos_grc.json']
        elif isinstance(analysis_files, str):
            analysis_files = [analysis_files]
            
        # Load and merge data from all files
        self.data = {'pos_distribution': {}}
        for file in analysis_files:
            file_data = self.load_analysis_data(file)
            if file_data:
                # Extract language code from filename
                lang = file.split('_')[1].split('.')[0]
                if 'pos_distribution' in file_data:
                    self.data['pos_distribution'][lang] = file_data['pos_distribution']
        
        # If data couldn't be loaded, use mock data for tests
        if not self.data['pos_distribution']:
            self._create_mock_data()
        
        # Define language names for better labels
        self.language_names = {
            'en': 'English',
            'de': 'German',
            'nl': 'Dutch',
            'es': 'Spanish',
            'grc': 'Ancient Greek'
        }
     
        # Define linguistic terminology for more professional labels
        self.pos_full_names = {
            'NOUN': 'Nouns',
            'VERB': 'Verbs',
            'ADJ': 'Adjectives',
            'ADV': 'Adverbs',
            'PRON': 'Pronouns',
            'DET': 'Determiners',
            'ADP': 'Adpositions',
            'NUM': 'Numerals',
            'CCONJ': 'Coordinating Conjunctions',
            'SCONJ': 'Subordinating Conjunctions',
            'INTJ': 'Interjections',
            'PROPN': 'Proper Nouns',
            'PUNCT': 'Punctuation',
            'SYM': 'Symbols',
            'X': 'Other',
            'AUX': 'Auxiliary Verbs',
            'PART': 'Particles'
        }

    # Mock data generation method (keep as is)
    # ...

    def plot_pos_distribution(self, languages=None, top_n=8, normalize=True):
        """Create a bar chart comparing POS distributions across languages.
        
        Args:
            languages: List of language codes to include. If None, use all languages.
            top_n: Number of POS categories to show (most frequent)
            normalize: Whether to normalize percentages to make languages directly comparable
            
        Returns:
            tuple: (figure, axes) The created matplotlib figure and axes
        """
        # Rest of the implementation with your enhancements
        # ...