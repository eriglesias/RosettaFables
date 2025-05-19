# word_frequency_plot.py
"""
Visualizations for word frequency analysis across languages.

This module provides visualizations to compare word usage patterns
across different language versions of the same fable, including:
- Most frequent words by language
- Shared vocabulary across translations
- Word frequency distributions
- Cross-language comparison of key terms
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import json
from ..core.figure_builder import FigureBuilder


class WordFrequencyPlot(FigureBuilder):
    """Visualizations for comparing word frequencies across languages."""

    def __init__(self, theme='default', fig_size=(12, 8), output_dir=None):
        """Initialize the word frequency plotter with default settings."""
        super().__init__(theme=theme, fig_size=fig_size, output_dir=output_dir)
        
        # Define language names for better labels
        self.language_names = {
            'en': 'English',
            'de': 'German',
            'nl': 'Dutch',
            'es': 'Spanish',
            'grc': 'Ancient Greek'
        }
        
        # Words to exclude from visualizations (common stopwords)
        self.exclude_words = {
            'en': {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'of', 'for', 'with'},
            'de': {'der', 'die', 'das', 'und', 'oder', 'aber', 'in', 'auf', 'bei', 'zu', 'von', 'für'},
            'nl': {'de', 'het', 'een', 'en', 'of', 'maar', 'in', 'op', 'bij', 'tot', 'van', 'voor'},
            'es': {'el', 'la', 'los', 'las', 'un', 'una', 'y', 'o', 'pero', 'en', 'a', 'de', 'por'}
        }
    
    def load_comparison_data(self, fable_id):
        """
        Load comparison data for a specific fable.
        
        Args:
            fable_id: ID of the fable to analyze
            
        Returns:
            Dictionary containing the loaded comparison data or None if not found
        """
        file_name = f"comparison_{fable_id}.json"
        data = self.load_analysis_data(file_name)
        
        if not data:
            self.logger.warning(f"No comparison data found for fable {fable_id}")
            return None
            
        return data
    
    def load_word_frequency_data(self, fable_id, language):
        """
        Load word frequency data for a specific fable and language.
        This would ideally come from a precomputed source, but we'll
        simulate it for now based on available data.
        
        Args:
            fable_id: ID of the fable
            language: Language code
            
        Returns:
            Dictionary of word frequencies
        """
        # In a real implementation, you'd load precomputed word frequencies
        # For now, we'll create mock data based on the comparison data
        comparison = self.load_comparison_data(fable_id)
        if not comparison:
            return {}
            
        # If you have actual word frequency data, use that instead
        # For demonstration purposes, we'll create mock data
        language_key = {'en': 'English', 'de': 'German', 'nl': 'Dutch', 'es': 'Spanish'}
        mock_freq = {
            'en': {
                'wolf': 8.2, 'lamb': 7.5, 'water': 5.3, 'drink': 4.9, 'river': 4.2,
                'said': 3.8, 'replied': 3.5, 'accused': 3.3, 'upstream': 3.0, 'downstream': 2.8
            },
            'de': {
                'wolf': 9.1, 'lamm': 8.6, 'wasser': 6.2, 'trinken': 5.7, 'fluss': 4.9,
                'sagte': 4.3, 'antwortete': 3.8, 'beschuldigte': 3.5, 'oben': 3.2, 'unten': 2.9
            },
            'nl': {
                'wolf': 8.5, 'geitje': 7.8, 'water': 5.6, 'drinken': 5.1, 'rivier': 4.5,
                'zei': 4.0, 'antwoordde': 3.7, 'beschuldigde': 3.4, 'stroomopwaarts': 3.1, 'stroomafwaarts': 2.7
            },
            'es': {
                'lobo': 9.0, 'cordero': 8.3, 'agua': 6.0, 'beber': 5.5, 'río': 4.7,
                'dijo': 4.2, 'respondió': 3.9, 'acusó': 3.6, 'arriba': 3.3, 'abajo': 3.0
            }
        }
        
        return mock_freq.get(language, {})
    
    def plot_top_words(self, fable_id, languages=None, top_n=10, exclude_stopwords=True):
        """
        Create a plot showing the most frequent words for each language.
        
        Args:
            fable_id: ID of the fable to analyze
            languages: List of language codes to include
            top_n: Number of top words to display
            exclude_stopwords: Whether to exclude common stopwords
            
        Returns:
            tuple: (figure, axes) The created matplotlib figure and axes
        """
        # Load comparison data
        comparison = self.load_comparison_data(fable_id)
        if not comparison:
            # Create an empty figure with a message
            fig, ax = self.create_figure()
            ax.text(0.5, 0.5, f"No comparison data available for fable {fable_id}", 
                    ha='center', va='center', fontsize=14)
            ax.set_axis_off()
            return fig, ax
        
        # Use provided languages or get from comparison data
        if not languages:
            languages = comparison.get('languages', [])
        
        # Get word frequencies for each language
        data = []
        for lang in languages:
            # Get word frequencies
            word_freqs = self.load_word_frequency_data(fable_id, lang)
            
            # Filter out stopwords if requested
            if exclude_stopwords:
                stopwords = self.exclude_words.get(lang, set())
                word_freqs = {word: freq for word, freq in word_freqs.items() 
                              if word.lower() not in stopwords}
            
            # Get top words
            for word, freq in sorted(word_freqs.items(), key=lambda x: x[1], reverse=True)[:top_n]:
                data.append({
                    'Language': self.language_names.get(lang, lang),
                    'Word': word,
                    'Frequency (%)': freq
                })
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Handle empty data
        if df.empty:
            fig, ax = self.create_figure()
            ax.text(0.5, 0.5, "No word frequency data available", 
                    ha='center', va='center', fontsize=14)
            ax.set_axis_off()
            return fig, ax
        
        # Create figure
        fig, ax = self.create_figure(figsize=(14, 10))
        
        # Create grouped bar chart
        g = sns.catplot(
            x='Word', 
            y='Frequency (%)', 
            hue='Language',
            data=df,
            kind='bar',
            height=8,
            aspect=1.5,
            palette=self.palettes['languages'][:len(languages)]
        )
        
        # Enhance the visualization
        g.set_xticklabels(rotation=45, ha='right')
        g.set_xlabels('Word', fontsize=14)
        g.set_ylabels('Frequency (%)', fontsize=14)
        g.fig.suptitle(f'Most Frequent Words in Fable {fable_id} Across Languages', 
                        fontsize=18, y=1.02)
        g.add_legend(title='Language', frameon=True)
        
        plt.tight_layout()
        
        return g.fig, g.axes[0][0]
    
    def plot_word_frequency_heatmap(self, fable_id, languages=None, top_n=15):
        """
        Create a heatmap showing word frequencies across languages.
        
        Args:
            fable_id: ID of the fable to analyze
            languages: List of language codes to include
            top_n: Number of top words to display
            
        Returns:
            tuple: (figure, axes) The created matplotlib figure and axes
        """
        # Load comparison data
        comparison = self.load_comparison_data(fable_id)
        if not comparison:
            # Create an empty figure with a message
            fig, ax = self.create_figure()
            ax.text(0.5, 0.5, f"No comparison data available for fable {fable_id}", 
                    ha='center', va='center', fontsize=14)
            ax.set_axis_off()
            return fig, ax
        
        # Use provided languages or get from comparison data
        if not languages:
            languages = comparison.get('languages', [])
        
        # Get word frequencies for each language
        word_data = {}
        all_words = set()
        
        for lang in languages:
            # Get word frequencies
            word_freqs = self.load_word_frequency_data(fable_id, lang)
            word_data[lang] = word_freqs
            all_words.update(word_freqs.keys())
        
        # Get top words across all languages
        word_scores = {}
        for word in all_words:
            score = sum(word_data.get(lang, {}).get(word, 0) for lang in languages)
            word_scores[word] = score
        
        top_words = sorted(word_scores.keys(), key=lambda w: word_scores[w], reverse=True)[:top_n]
        
        # Create matrix for heatmap
        matrix = []
        for lang in languages:
            row = [word_data.get(lang, {}).get(word, 0) for word in top_words]
            matrix.append(row)
        
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
            xticklabels=top_words,
            yticklabels=[self.language_names.get(lang, lang) for lang in languages],
            ax=ax
        )
        
        # Enhance the visualization
        ax.set_title(f'Word Frequency Heatmap for Fable {fable_id}', fontsize=18, pad=20)
        ax.set_xlabel('Words', fontsize=14, labelpad=10)
        ax.set_ylabel('Languages', fontsize=14, labelpad=10)
        
        # Improve readability
        plt.xticks(rotation=45, ha='right')
        
        # Add colorbar title
        cbar = ax.collections[0].colorbar
        cbar.set_label('Frequency (%)', rotation=270, labelpad=20)
        
        plt.tight_layout()
        
        return fig, ax
    
    def plot_shared_vocabulary(self, fable_id, languages=None, min_freq=1.0):
        """
        Create a visualization of words shared across languages.
        
        Args:
            fable_id: ID of the fable to analyze
            languages: List of language codes to include
            min_freq: Minimum frequency for a word to be considered
            
        Returns:
            tuple: (figure, axes) The created matplotlib figure and axes
        """
        # Load comparison data
        comparison = self.load_comparison_data(fable_id)
        if not comparison:
            # Create an empty figure with a message
            fig, ax = self.create_figure()
            ax.text(0.5, 0.5, f"No comparison data available for fable {fable_id}", 
                    ha='center', va='center', fontsize=14)
            ax.set_axis_off()
            return fig, ax
        
        # Use provided languages or get from comparison data
        if not languages:
            languages = comparison.get('languages', [])
        
        # Get shared words
        word_presence = {}
        
        for lang in languages:
            # Get word frequencies
            word_freqs = self.load_word_frequency_data(fable_id, lang)
            
            # Track words that meet minimum frequency
            for word, freq in word_freqs.items():
                if freq >= min_freq:
                    if word not in word_presence:
                        word_presence[word] = set()
                    word_presence[word].add(lang)
        
        # Filter to words that appear in multiple languages
        shared_words = {}
        for word, langs in word_presence.items():
            if len(langs) > 1:
                shared_words[word] = langs
        
        # Create a matrix for visualization
        words = sorted(shared_words.keys(), key=lambda w: len(shared_words[w]), reverse=True)
        matrix = np.zeros((len(words), len(languages)))
        
        for i, word in enumerate(words):
            for j, lang in enumerate(languages):
                if lang in shared_words[word]:
                    matrix[i, j] = 1
        
        # Create figure
        fig, ax = self.create_figure(figsize=(14, 10))
        
        # Create heatmap
        sns.heatmap(
            matrix, 
            cmap='Blues',
            xticklabels=[self.language_names.get(lang, lang) for lang in languages],
            yticklabels=words,
            ax=ax,
            cbar=False
        )
        
        # Enhance the visualization
        ax.set_title(f'Shared Vocabulary Across Languages in Fable {fable_id}', fontsize=18, pad=20)
        ax.set_xlabel('Languages', fontsize=14, labelpad=10)
        ax.set_ylabel('Words', fontsize=14, labelpad=10)
        
        plt.tight_layout()
        
        return fig, ax
    
    def plot_word_distribution_comparison(self, fable_id, languages=None, words=None):
        """
        Create a visualization comparing the distribution of specific words.
        
        Args:
            fable_id: ID of the fable to analyze
            languages: List of language codes to include
            words: List of words to compare (or None to use top shared words)
            
        Returns:
            tuple: (figure, axes) The created matplotlib figure and axes
        """
        # Load comparison data
        comparison = self.load_comparison_data(fable_id)
        if not comparison:
            # Create an empty figure with a message
            fig, ax = self.create_figure()
            ax.text(0.5, 0.5, f"No comparison data available for fable {fable_id}", 
                    ha='center', va='center', fontsize=14)
            ax.set_axis_off()
            return fig, ax
        
        # Use provided languages or get from comparison data
        if not languages:
            languages = comparison.get('languages', [])
        
        # Get word frequencies for each language
        lang_data = {}
        for lang in languages:
            lang_data[lang] = self.load_word_frequency_data(fable_id, lang)
        
        # If no words specified, find the most frequent ones
        if not words:
            # Get words that appear in multiple languages
            word_scores = {}
            for lang, freqs in lang_data.items():
                for word, freq in freqs.items():
                    if word not in word_scores:
                        word_scores[word] = 0
                    word_scores[word] += freq
            
            # Take top 5 words
            words = sorted(word_scores.keys(), key=lambda w: word_scores[w], reverse=True)[:5]
        
        # Create data for plot
        plot_data = []
        for lang in languages:
            freqs = lang_data[lang]
            for word in words:
                plot_data.append({
                    'Language': self.language_names.get(lang, lang),
                    'Word': word,
                    'Frequency (%)': freqs.get(word, 0)
                })
        
        # Convert to DataFrame
        df = pd.DataFrame(plot_data)
        
        # Create figure
        fig, ax = self.create_figure(figsize=(14, 8))
        
        # Create grouped bar chart
        sns.barplot(
            x='Word', 
            y='Frequency (%)', 
            hue='Language',
            data=df,
            palette=self.palettes['languages'][:len(languages)],
            ax=ax
        )
        
        # Enhance the visualization
        ax.set_title(f'Word Distribution Comparison in Fable {fable_id}', fontsize=18, pad=20)
        ax.set_xlabel('Word', fontsize=14, labelpad=10)
        ax.set_ylabel('Frequency (%)', fontsize=14, labelpad=10)
        
        # Add a horizontal grid for readability
        ax.grid(axis='y', alpha=0.3)
        
        # Improve readability
        plt.xticks(rotation=0)
        ax.legend(title='Language', title_fontsize=12, bbox_to_anchor=(1.02, 1), loc='upper left')
        
        plt.tight_layout()
        
        return fig, ax
    
    def plot_lexical_richness_comparison(self, fable_ids=None):
        """
        Create a comparison of lexical richness metrics across fables and languages.
        
        Args:
            fable_ids: List of fable IDs to include
            
        Returns:
            tuple: (figure, axes) The created matplotlib figure and axes
        """
        # Default to first 5 fables if none specified
        if not fable_ids:
            fable_ids = [str(i) for i in range(1, 6)]
        
        # Extract type-token ratio and hapax ratio from comparison files
        data = []
        
        for fable_id in fable_ids:
            comparison = self.load_comparison_data(fable_id)
            if not comparison:
                continue
                
            # Extract lexical metrics if available
            for lang in comparison.get('languages', []):
                # These metrics would ideally come from precomputed analysis
                # For now, we'll create mock data based on available metrics
                
                # For real implementation, replace with actual metrics
                token_count = comparison.get('token_counts', {}).get(lang, 0)
                if token_count > 0:
                    # Simulate type-token ratio and hapax ratio
                    type_token_ratio = 0.65 + np.random.normal(0, 0.1)  # Simulate around 0.65 with some noise
                    hapax_ratio = 0.45 + np.random.normal(0, 0.1)  # Simulate around 0.45 with some noise
                    
                    data.append({
                        'Fable ID': fable_id,
                        'Language': self.language_names.get(lang, lang),
                        'Type-Token Ratio': type_token_ratio,
                        'Hapax Ratio': hapax_ratio
                    })
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Handle empty data
        if df.empty:
            fig, ax = self.create_figure()
            ax.text(0.5, 0.5, "No lexical richness data available", 
                    ha='center', va='center', fontsize=14)
            ax.set_axis_off()
            return fig, ax
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        # Create grouped bar charts
        sns.barplot(
            x='Fable ID', 
            y='Type-Token Ratio', 
            hue='Language',
            data=df,
            palette=self.palettes['languages'],
            ax=axes[0]
        )
        
        sns.barplot(
            x='Fable ID', 
            y='Hapax Ratio', 
            hue='Language',
            data=df,
            palette=self.palettes['languages'],
            ax=axes[1]
        )
        
        # Enhance the visualization
        axes[0].set_title('Type-Token Ratio Comparison', fontsize=16, pad=20)
        axes[0].set_xlabel('Fable ID', fontsize=12, labelpad=10)
        axes[0].set_ylabel('Type-Token Ratio', fontsize=12, labelpad=10)
        
        axes[1].set_title('Hapax Ratio Comparison', fontsize=16, pad=20)
        axes[1].set_xlabel('Fable ID', fontsize=12, labelpad=10)
        axes[1].set_ylabel('Hapax Ratio', fontsize=12, labelpad=10)
        
        # Remove legends from first plot and adjust second plot's legend
        axes[0].get_legend().remove()
        axes[1].legend(title='Language', title_fontsize=12, bbox_to_anchor=(1.02, 1), loc='upper left')
        
        # Add grids for readability
        axes[0].grid(axis='y', alpha=0.3)
        axes[1].grid(axis='y', alpha=0.3)
        
        fig.suptitle('Lexical Richness Across Fables and Languages', fontsize=18, y=1.05)
        
        plt.tight_layout()
        
        return fig, axes