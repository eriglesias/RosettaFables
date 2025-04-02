from ..core.figure_builder import FigureBuilder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class POSComparisonPlot(FigureBuilder):
    """Creates visualizations comparing POS tag distributions across languages."""
    
    def __init__(self, analysis_file='comparison_1.json', theme='default', fig_size=(12, 8)):
        super().__init__(theme=theme, fig_size=fig_size)
        self.data = self.load_analysis_data(analysis_file)
        
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
    
    def plot_pos_distribution(self, languages=None, top_n=8):
        """Create a bar chart comparing POS distributions across languages.
        
        Args:
            languages: List of language codes to include. If None, use all languages.
            top_n: Number of POS categories to show (most frequent)
            
        Returns:
            tuple: (figure, axes) The created matplotlib figure and axes
        """
        # Prepare the data
        pos_data = self.data['pos_distribution']
        if languages is None:
            languages = list(pos_data.keys())
        
        # Determine which POS tags to include (take the top_n most common across languages)
        all_pos_freqs = {}
        for pos_tag in set().union(*[set(pos_data[lang].keys()) for lang in languages]):
            all_pos_freqs[pos_tag] = sum(pos_data[lang].get(pos_tag, 0) for lang in languages)
        
        top_pos = sorted(all_pos_freqs.keys(), key=lambda x: all_pos_freqs[x], reverse=True)[:top_n]
        
        # Create DataFrame for easier plotting
        plot_data = []
        for lang in languages:
            for pos in top_pos:
                plot_data.append({
                    'Language': lang,
                    'POS': pos,
                    'Full POS Name': self.pos_full_names.get(pos, pos),
                    'Frequency (%)': pos_data[lang].get(pos, 0)
                })
        
        df = pd.DataFrame(plot_data)
        
        # Create the figure with a larger size for this specific visualization
        fig, ax = self.create_figure(figsize=(14, 8))
        
        # Create a grouped bar chart
        sns.barplot(
            x='Full POS Name', 
            y='Frequency (%)', 
            hue='Language', 
            data=df,
            palette=self.palettes['languages'][:len(languages)],
            ax=ax
        )
        
        # Enhance the visualization
        ax.set_title('Part-of-Speech Distribution Across Languages', fontsize=18, pad=20)
        ax.set_xlabel('Part of Speech', fontsize=14, labelpad=10)
        ax.set_ylabel('Frequency (%)', fontsize=14, labelpad=10)
        
        # Improve readability
        plt.xticks(rotation=45, ha='right')
        ax.legend(title='Language', title_fontsize=12, bbox_to_anchor=(1.02, 1), loc='upper left')
        
        # Add a subtle grid only on the y-axis for readability
        ax.grid(axis='y', alpha=0.3)
        
        # Ensure layout fits everything
        plt.tight_layout()
        
        return fig, ax