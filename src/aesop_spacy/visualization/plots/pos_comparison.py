#pos_comparison.py
""" Does this and that"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from ..core.figure_builder import FigureBuilder
class POSDistributionPlot(FigureBuilder):
    """Creates visualizations of Part-of-Speech distributions for single languages and comparisons."""
 
    def __init__(self, theme='default', fig_size=(12, 8)):
        """Initialize the POS distribution plotter with default settings."""
        super().__init__(theme=theme, fig_size=fig_size)
        
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
        
        # Define language names for better labels
        self.language_names = {
            'en': 'English',
            'de': 'German',
            'nl': 'Dutch',
            'es': 'Spanish',
            'grc': 'Ancient Greek'
        }
        
        # Default color palette for POS categories
        self.pos_palette = sns.color_palette('viridis', n_colors=len(self.pos_full_names))
        
    def plot_single_language(self, language_code, top_n=10, sort_by='frequency'):
        """
        Create a bar chart for POS distribution of a single language.
        
        Args:
            language_code: Two-letter language code (e.g., 'en', 'de')
            top_n: Number of top POS categories to display
            sort_by: How to sort the bars ('frequency' or 'alphabetical')
            
        Returns:
            tuple: (figure, axes) The created matplotlib figure and axes
        """
        # Load the POS data for this language
        file_name = f"pos_{language_code}.json"
        pos_data = self.load_analysis_data(file_name)
        
        if not pos_data:
            # If no data, create an empty figure with a message
            fig, ax = self.create_figure()
            ax.text(0.5, 0.5, f"No POS data available for {self.language_names.get(language_code, language_code)}", 
                    ha='center', va='center', fontsize=14)
            ax.set_axis_off()
            return fig, ax
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame({
            'POS': list(pos_data.keys()),
            'Frequency': list(pos_data.values())
        })
        
        # Add full names for better readability
        df['Full Name'] = df['POS'].map(lambda x: self.pos_full_names.get(x, x))
        
        # Sort as requested
        if sort_by == 'frequency':
            df = df.sort_values('Frequency', ascending=False)
        else:  # alphabetical
            df = df.sort_values('Full Name')
        
        # Take only top N categories
        df = df.head(top_n)
        
        # Create the figure
        fig, ax = self.create_figure()
        
        # Create horizontal bar chart for better readability with long labels
        bars = ax.barh(df['Full Name'], df['Frequency'], 
                color=self.pos_palette[:len(df)])
        
        # Add percentage labels to the bars
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width + 0.5
            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.1f}%',
                   va='center', fontsize=10)
        
        # Set titles and labels
        language_name = self.language_names.get(language_code, language_code)
        ax.set_title(f'Part-of-Speech Distribution in {language_name}', fontsize=16, pad=20)
        ax.set_xlabel('Frequency (%)', fontsize=12, labelpad=10)
        ax.set_ylabel('Part of Speech', fontsize=12, labelpad=10)
        
        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add a subtle grid only on the x-axis
        ax.grid(axis='x', alpha=0.3)
        
        # Ensure layout fits everything
        plt.tight_layout()
        
        return fig, ax
        
    def plot_language_comparison(self, languages=None, top_n=8):
        """
        Create a grouped bar chart comparing POS distributions across multiple languages.
        
        Args:
            languages: List of language codes to include. If None, use all available.
            top_n: Number of POS categories to show (most frequent)
            
        Returns:
            tuple: (figure, axes) The created matplotlib figure and axes
        """
        # Default to all languages if none specified
        if languages is None:
            languages = ['en', 'de', 'nl', 'es', 'grc']
        
        # Load data for each language
        data_by_lang = {}
        for lang in languages:
            file_name = f"pos_{lang}.json"
            lang_data = self.load_analysis_data(file_name)
            if lang_data:
                data_by_lang[lang] = lang_data
        
        # If no data found, show a message
        if not data_by_lang:
            fig, ax = self.create_figure()
            ax.text(0.5, 0.5, "No POS data available for the requested languages", 
                    ha='center', va='center', fontsize=14)
            ax.set_axis_off()
            return fig, ax
        
        # Find the most common POS tags across all languages
        all_pos = {}
        for lang, pos_data in data_by_lang.items():
            for pos, freq in pos_data.items():
                if pos in all_pos:
                    all_pos[pos] += freq
                else:
                    all_pos[pos] = freq
        
        # Get top N most frequent overall
        top_pos = sorted(all_pos.keys(), key=lambda x: all_pos[x], reverse=True)[:top_n]
        
        # Create DataFrame for plotting
        plot_data = []
        for lang, pos_data in data_by_lang.items():
            lang_name = self.language_names.get(lang, lang)
            for pos in top_pos:
                plot_data.append({
                    'Language': lang_name,
                    'POS': pos,
                    'Full Name': self.pos_full_names.get(pos, pos),
                    'Frequency (%)': pos_data.get(pos, 0)
                })
        
        df = pd.DataFrame(plot_data)
        
        # Create the figure with a larger size for this visualization
        fig, ax = self.create_figure(figsize=(14, 8))
        
        # Create a grouped bar chart
        g = sns.barplot(
            x='Full Name', 
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
        
    def plot_pos_heatmap(self, languages=None):
        """
        Create a heatmap showing POS distributions across languages for visual comparison.
        
        Args:
            languages: List of language codes to include. If None, use all available.
            
        Returns:
            tuple: (figure, axes) The created matplotlib figure and axes
        """
        # Default to all languages if none specified
        if languages is None:
            languages = ['en', 'de', 'nl', 'es', 'grc']
        
        # Load data for each language
        data_by_lang = {}
        for lang in languages:
            file_name = f"pos_{lang}.json"
            lang_data = self.load_analysis_data(file_name)
            if lang_data:
                data_by_lang[lang] = lang_data
        
        # If no data found, show a message
        if not data_by_lang:
            fig, ax = self.create_figure()
            ax.text(0.5, 0.5, "No POS data available for the requested languages", 
                    ha='center', va='center', fontsize=14)
            ax.set_axis_off()
            return fig, ax
        
        # Get all unique POS tags
        all_pos = set()
        for pos_data in data_by_lang.values():
            all_pos.update(pos_data.keys())
        
        # Create a matrix for the heatmap
        matrix = []
        rows = []
        for lang, pos_data in data_by_lang.items():
            row = [pos_data.get(pos, 0) for pos in sorted(all_pos)]
            matrix.append(row)
            rows.append(self.language_names.get(lang, lang))
        
        # Convert to numpy array
        heat_data = np.array(matrix)
        
        # Create figure
        fig, ax = self.create_figure(figsize=(14, 10))
        
        # Create heatmap
        sns.heatmap(
            heat_data, 
            annot=True, 
            fmt='.1f',
            cmap='viridis',
            xticklabels=[self.pos_full_names.get(pos, pos) for pos in sorted(all_pos)],
            yticklabels=rows,
            ax=ax
        )
        
        # Enhance the visualization
        ax.set_title('Part-of-Speech Distribution Heatmap', fontsize=18, pad=20)
        
        # Improve readability
        plt.xticks(rotation=45, ha='right')
        
        # Add colorbar title
        cbar = ax.collections[0].colorbar
        cbar.set_label('Frequency (%)', rotation=270, labelpad=20)
        
        # Ensure layout fits everything
        plt.tight_layout()
        
        return fig, ax