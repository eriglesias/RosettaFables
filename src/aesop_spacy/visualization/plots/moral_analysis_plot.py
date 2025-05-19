# moral_analysis_plot.py
"""
Visualization components for moral analysis results.

This module provides classes for visualizing moral analysis results,
including moral themes, cross-language comparisons,
and moral similarity across fables.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import json
from ..core.figure_builder import FigureBuilder


class MoralAnalysisPlot(FigureBuilder):
    """Creates visualizations of moral analysis results."""
    
    def __init__(self, theme='default', fig_size=(12, 8), output_dir=None):
        """Initialize the moral analysis plotter with default settings."""
        super().__init__(theme=theme, fig_size=fig_size, output_dir=output_dir)
        
        # Define language names for better labels
        self.language_names = {
            'en': 'English',
            'de': 'German',
            'nl': 'Dutch',
            'es': 'Spanish',
            'grc': 'Ancient Greek'
        }
        
        # Define color schemes for different moral themes
        self.theme_colors = {
            'honesty': '#4C72B0',      # Blue
            'perseverance': '#55A868',  # Green
            'prudence': '#C44E52',     # Red
            'kindness': '#8172B3',     # Purple
            'humility': '#CCB974',     # Yellow/gold
            'gratitude': '#64B5CD',    # Light blue
            'moderation': '#B47CC7',   # Pink/purple
            'justice': '#BCB3D9'       # Lavender
        }
    
    def load_moral_data(self, language):
        """
        Load moral analysis data for a specific language.
        
        Args:
            language: Language code ('en', 'de', 'nl', 'es', 'grc')
            
        Returns:
            Dict containing the loaded data, or empty dict if not found
        """
        # Filename formats to try
        filenames = [
            f"implicit_morals_{language}.json",
            f"explicit_morals_{language}.json",
            f"moral_analysis_{language}.json"
        ]
        
        for filename in filenames:
            # Try to find the file in the analysis directory
            potential_paths = [
                Path(__file__).resolve().parents[3] / "data" / "data_handled" / "analysis" / filename,
                Path("data/data_handled/analysis") / filename,
                Path("./data/data_handled/analysis") / filename,
                Path("/Users/enriqueviv/Coding/coding_projects/data_science_projects/ds_sandbox/nlp_sandbox/aesop_spacy/data/data_handled/analysis") / filename
            ]
            
            for path in potential_paths:
                if path.exists():
                    self.logger.info(f"Found moral data at: {path}")
                    try:
                        with open(path, 'r', encoding='utf-8') as f:
                            return json.load(f)
                    except Exception as e:
                        self.logger.error(f"Error loading moral data: {e}")
                        continue
        
        self.logger.warning(f"No moral data found for language: {language}")
        return {}
    
    def load_moral_comparison_data(self):
        """
        Load cross-language moral comparison data.
        
        Returns:
            Dict containing the comparison data, or empty dict if not found
        """
        # Filename formats to try
        filenames = [
            "moral_comparison_all.json",
            "moral_comparison.json"
        ]
        
        for filename in filenames:
            # Try to find the file in the analysis directory
            potential_paths = [
                Path(__file__).resolve().parents[3] / "data" / "data_handled" / "analysis" / filename,
                Path("data/data_handled/analysis") / filename,
                Path("./data/data_handled/analysis") / filename,
                Path("/Users/enriqueviv/Coding/coding_projects/data_science_projects/ds_sandbox/nlp_sandbox/aesop_spacy/data/data_handled/analysis") / filename
            ]
            
            for path in potential_paths:
                if path.exists():
                    self.logger.info(f"Found moral comparison data at: {path}")
                    try:
                        with open(path, 'r', encoding='utf-8') as f:
                            return json.load(f)
                    except Exception as e:
                        self.logger.error(f"Error loading moral comparison data: {e}")
                        continue
        
        self.logger.warning("No moral comparison data found")
        return {}
    
    def plot_moral_themes(self, language):
        """
        Create a bar chart showing the distribution of moral themes for a language.
        
        Args:
            language: Language code ('en', 'de', 'nl', 'es', 'grc')
            
        Returns:
            tuple: (figure, axes) The created matplotlib figure and axes
        """
        # Load the moral data
        data = self.load_moral_data(language)
        
        if not data:
            # If no data, create an empty figure with a message
            fig, ax = self.create_figure()
            ax.text(0.5, 0.5, f"No moral data available for {language}", 
                    ha='center', va='center', fontsize=14)
            ax.set_axis_off()
            return fig, ax
        
        # Extract the fables that have theme information
        theme_data = []
        
        for fable in data:
            # Check if this is a list of fables or a direct themes dictionary
            if isinstance(fable, dict) and 'themes' in fable:
                categories = fable.get('themes', {}).get('categories', [])
                
                if categories:
                    for category in categories:
                        theme_data.append({
                            'Theme': category.get('name', 'Unknown'),
                            'Score': category.get('score', 0)
                        })
        
        # If no theme data found, try alternative format
        if not theme_data and isinstance(data, dict):
            for theme, info in data.items():
                if isinstance(info, dict) and 'score' in info:
                    theme_data.append({
                        'Theme': theme,
                        'Score': info['score']
                    })
        
        # If still no theme data, show message
        if not theme_data:
            fig, ax = self.create_figure()
            ax.text(0.5, 0.5, f"No moral theme data found for {language}", 
                   ha='center', va='center', fontsize=14)
            ax.set_axis_off()
            return fig, ax
        
        # Create DataFrame
        df = pd.DataFrame(theme_data)
        
        # Aggregate scores by theme
        theme_scores = df.groupby('Theme')['Score'].sum().reset_index()
        
        # Sort by score
        theme_scores = theme_scores.sort_values('Score', ascending=False)
        
        # Create the figure
        fig, ax = self.create_figure()
        
        # Create bar chart
        bars = ax.bar(
            theme_scores['Theme'], 
            theme_scores['Score'],
            color=[self.theme_colors.get(theme, '#1f77b4') for theme in theme_scores['Theme']]
        )
        
        # Add score labels to the bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{height:.2f}', ha='center', va='bottom')
        
        # Set titles and labels
        language_name = self.language_names.get(language, language)
        ax.set_title(f'Moral Themes in {language_name} Fables', 
                    fontsize=16, pad=20)
        ax.set_xlabel('Theme', fontsize=12, labelpad=10)
        ax.set_ylabel('Theme Score', fontsize=12, labelpad=10)
        
        # Rotate x tick labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add a subtle grid only on the y-axis
        ax.grid(axis='y', alpha=0.3)
        
        # Ensure layout fits everything
        plt.tight_layout()
        
        return fig, ax
    
    def plot_moral_inferences(self, language):
        """
        Create a visualization of inferred morals for a language.
        
        Args:
            language: Language code ('en', 'de', 'nl', 'es', 'grc')
            
        Returns:
            tuple: (figure, axes) The created matplotlib figure and axes
        """
        # Load the moral data
        data = self.load_moral_data(language)
        
        if not data:
            # If no data, create an empty figure with a message
            fig, ax = self.create_figure()
            ax.text(0.5, 0.5, f"No moral data available for {language}", 
                    ha='center', va='center', fontsize=14)
            ax.set_axis_off()
            return fig, ax
        
        # Extract inferred morals
        inference_data = []
        
        for fable in data:
            if isinstance(fable, dict):
                # First try to extract from moral structure in the provided JSON
                if 'inferred_morals' in fable:
                    inferred_morals = fable['inferred_morals']
                    for moral in inferred_morals:
                        inference_data.append({
                            'Moral Text': moral.get('text', 'Unknown'),
                            'Relevance': moral.get('relevance_score', 0),
                            'Source': moral.get('source', 'Unknown'),
                            'Fable ID': fable.get('fable_id', 'Unknown')
                        })
                # Try alternative location
                elif 'implicit' in fable and 'inferred_morals' in fable['implicit']:
                    inferred_morals = fable['implicit']['inferred_morals']
                    for moral in inferred_morals:
                        inference_data.append({
                            'Moral Text': moral.get('text', 'Unknown'),
                            'Relevance': moral.get('relevance_score', 0),
                            'Source': moral.get('source', 'Unknown'),
                            'Fable ID': fable.get('fable_id', 'Unknown')
                        })
        
        # If no inferred morals found, show message
        if not inference_data:
            fig, ax = self.create_figure()
            ax.text(0.5, 0.5, f"No inferred morals found for {language}", 
                   ha='center', va='center', fontsize=14)
            ax.set_axis_off()
            return fig, ax
        
        # Create DataFrame
        df = pd.DataFrame(inference_data)
        
        # Sort by relevance score
        df = df.sort_values('Relevance', ascending=False)
        
        # If there are too many morals, just show the top 10
        if len(df) > 10:
            df = df.head(10)
        
        # Create figure for a table-like visualization
        fig, ax = self.create_figure(figsize=(14, max(6, len(df) * 0.4)))
        
        # Hide axes
        ax.axis('off')
        
        # Create table
        # Only include relevant columns
        table_data = df[['Fable ID', 'Moral Text', 'Relevance', 'Source']]
        
        table = ax.table(
            cellText=table_data.values,
            colLabels=table_data.columns,
            cellLoc='center',
            loc='center',
            bbox=[0.05, 0.05, 0.9, 0.9]
        )
        
        # Customize table appearance
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        # Color header and set column widths
        header_color = '#4C72B0'
        for (i, j), cell in table.get_celld().items():
            if i == 0:  # Header row
                cell.set_facecolor(header_color)
                cell.set_text_props(color='white', fontweight='bold')
            
            # Set column widths
            if j == 0:  # Fable ID column
                cell.set_width(0.1)
            elif j == 1:  # Moral Text column
                cell.set_width(0.5)
            elif j == 2:  # Relevance column
                cell.set_width(0.15)
                                    # Color by relevance score if not header
                if i > 0:
                    relevance = float(table_data.iloc[i-1, 2])
                    # Use a color gradient based on relevance
                    color = plt.colorbar  # 
                    cell.set_facecolor(color)
            elif j == 3:  # Source column
                cell.set_width(0.25)
        
        # Set title
        language_name = self.language_names.get(language, language)
        plt.suptitle(f'Top Inferred Morals in {language_name} Fables', fontsize=16, y=0.98)
        
        # Ensure layout fits everything
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
        
        return fig, ax
    
    def plot_moral_comparison(self, fable_id=None):
        """
        Create a visualization comparing morals across languages.
        
        Args:
            fable_id: Specific fable ID to compare (None for all fables)
            
        Returns:
            tuple: (figure, axes) The created matplotlib figure and axes
        """
        # Load moral comparison data
        data = self.load_moral_comparison_data()
        
        if not data:
            # If no data, create an empty figure with a message
            fig, ax = self.create_figure()
            ax.text(0.5, 0.5, "No moral comparison data available", 
                    ha='center', va='center', fontsize=14)
            ax.set_axis_off()
            return fig, ax
        
        # Filter for specific fable if requested
        if fable_id:
            if fable_id in data:
                comparison_data = {fable_id: data[fable_id]}
            else:
                # Fable not found
                fig, ax = self.create_figure()
                ax.text(0.5, 0.5, f"Fable ID '{fable_id}' not found in comparison data", 
                       ha='center', va='center', fontsize=14)
                ax.set_axis_off()
                return fig, ax
        else:
            comparison_data = data
        
        # If there are many fables, just focus on the first one for visualization
        if len(comparison_data) > 1 and not fable_id:
            first_key = next(iter(comparison_data))
            comparison_data = {first_key: comparison_data[first_key]}
        
        # Extract moral texts by language
        moral_texts = {}
        
        for fid, fable_data in comparison_data.items():
            morals = fable_data.get('morals', {})
            for lang, moral_info in morals.items():
                final_moral = moral_info.get('final_moral', '')
                if final_moral:
                    if lang not in moral_texts:
                        moral_texts[lang] = []
                    moral_texts[lang].append({
                        'Fable ID': fid,
                        'Moral Text': final_moral
                    })
        
        # If no moral texts found, show message
        if not moral_texts:
            fig, ax = self.create_figure()
            ax.text(0.5, 0.5, "No moral texts found in comparison data", 
                   ha='center', va='center', fontsize=14)
            ax.set_axis_off()
            return fig, ax
        
        # Create figure for a table-like visualization
        fig, ax = self.create_figure(figsize=(14, max(6, sum(len(morals) for morals in moral_texts.values()) * 0.4)))
        
        # Hide axes
        ax.axis('off')
        
        # Create a list of all morals for the table
        table_data = []
        
        for lang, morals in sorted(moral_texts.items()):
            language_name = self.language_names.get(lang, lang)
            for moral in morals:
                table_data.append([
                    moral['Fable ID'],
                    language_name,
                    moral['Moral Text']
                ])
        
        # Create table
        table = ax.table(
            cellText=table_data,
            colLabels=['Fable ID', 'Language', 'Moral Text'],
            cellLoc='center',
            loc='center',
            bbox=[0.05, 0.05, 0.9, 0.9]
        )
        
        # Customize table appearance
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        # Color header and set column widths
        header_color = '#4C72B0'
        for (i, j), cell in table.get_celld().items():
            if i == 0:  # Header row
                cell.set_facecolor(header_color)
                cell.set_text_props(color='white', fontweight='bold')
            
            # Set column widths and colors
            if j == 0:  # Fable ID column
                cell.set_width(0.1)
            elif j == 1:  # Language column
                cell.set_width(0.2)
                # Color by language if not header
                if i > 0:
                    lang = table_data[i-1][1]
                    language_index = list(self.language_names.values()).index(lang) if lang in self.language_names.values() else 0
                    # Use a color from the viridis palette based on language
                    color = plt.cm._colormaps(language_index / len(self.language_names))
                    cell.set_facecolor(color)
                    cell.set_text_props(color='white')
            elif j == 2:  # Moral Text column
                cell.set_width(0.7)
        
        # Set title
        if fable_id:
            plt.suptitle(f'Moral Comparison Across Languages - Fable {fable_id}', fontsize=16, y=0.98)
        else:
            plt.suptitle('Moral Comparison Across Languages', fontsize=16, y=0.98)
        
        # Ensure layout fits everything
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
        
        return fig, ax
    
    def plot_moral_similarity_heatmap(self):
        """
        Create a heatmap visualization of moral similarity across languages.
        
        Returns:
            tuple: (figure, axes) The created matplotlib figure and axes
        """
        # Load moral comparison data
        data = self.load_moral_comparison_data()
        
        if not data:
            # If no data, create an empty figure with a message
            fig, ax = self.create_figure()
            ax.text(0.5, 0.5, "No moral comparison data available", 
                    ha='center', va='center', fontsize=14)
            ax.set_axis_off()
            return fig, ax
        
        # Extract semantic similarity data
        similarity_data = []
        
        for fable_id, fable_data in data.items():
            semantic_sim = fable_data.get('semantic_similarity', {})
            similarities = semantic_sim.get('similarities', {})
            
            # Extract language pairs and their similarity scores
            for pair, score in similarities.items():
                lang1, lang2 = pair.split('-')
                similarity_data.append({
                    'Fable ID': fable_id,
                    'Language 1': lang1,
                    'Language 2': lang2,
                    'Similarity': score
                })
        
        # If no similarity data found, show message
        if not similarity_data:
            fig, ax = self.create_figure()
            ax.text(0.5, 0.5, "No moral similarity data found", 
                   ha='center', va='center', fontsize=14)
            ax.set_axis_off()
            return fig, ax
        
        # Create DataFrame
        df = pd.DataFrame(similarity_data)
        
        # Get unique languages
        languages = sorted(set(df['Language 1']).union(set(df['Language 2'])))
        
        # Create the figure
        fig, ax = self.create_figure()
        
        # Create similarity matrix
        similarity_matrix = np.zeros((len(languages), len(languages)))
        
        # Fill with average similarity scores across all fables
        for i, lang1 in enumerate(languages):
            for j, lang2 in enumerate(languages):
                if i == j:
                    # Diagonal (same language) is 1.0
                    similarity_matrix[i, j] = 1.0
                else:
                    # Check both pair orderings
                    pair1 = f"{lang1}-{lang2}"
                    pair2 = f"{lang2}-{lang1}"
                    
                    # Get all similarity scores for these language pairs
                    scores = df[(df['Language 1'] == lang1) & (df['Language 2'] == lang2)]['Similarity'].tolist()
                    scores.extend(df[(df['Language 1'] == lang2) & (df['Language 2'] == lang1)]['Similarity'].tolist())
                    
                    if scores:
                        similarity_matrix[i, j] = np.mean(scores)
        
        # Get language names for labels
        language_names = [self.language_names.get(lang, lang) for lang in languages]
        
        # Create heatmap
        im = ax.imshow(similarity_matrix, cmap='viridis')
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Similarity Score', rotation=-90, va="bottom")
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(languages)))
        ax.set_yticks(np.arange(len(languages)))
        ax.set_xticklabels(language_names)
        ax.set_yticklabels(language_names)
        
        # Rotate x tick labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add values to cells
        for i in range(len(languages)):
            for j in range(len(languages)):
                text = ax.text(j, i, f"{similarity_matrix[i, j]:.2f}",
                              ha="center", va="center", color="w" if similarity_matrix[i, j] < 0.7 else "black")
        
        # Set title
        ax.set_title('Moral Similarity Across Languages', fontsize=16, pad=20)
        
        # Ensure layout fits everything
        plt.tight_layout()
        
        return fig, ax
    
    def plot_theme_consistency(self):
        """
        Create a bar chart showing moral theme consistency across languages.
        
        Returns:
            tuple: (figure, axes) The created matplotlib figure and axes
        """
        # Load moral comparison data
        data = self.load_moral_comparison_data()
        
        if not data:
            # If no data, create an empty figure with a message
            fig, ax = self.create_figure()
            ax.text(0.5, 0.5, "No moral comparison data available", 
                    ha='center', va='center', fontsize=14)
            ax.set_axis_off()
            return fig, ax
        
        # Extract theme consistency data
        consistency_data = []
        
        for fable_id, fable_data in data.items():
            theme_consistency = fable_data.get('theme_consistency', {})
            if theme_consistency:
                dominant_theme = theme_consistency.get('dominant_theme', 'Unknown')
                consistency_score = theme_consistency.get('consistency_score', 0)
                
                consistency_data.append({
                    'Fable ID': fable_id,
                    'Dominant Theme': dominant_theme,
                    'Consistency Score': consistency_score
                })
        
        # If no consistency data found, show message
        if not consistency_data:
            fig, ax = self.create_figure()
            ax.text(0.5, 0.5, "No theme consistency data found", 
                   ha='center', va='center', fontsize=14)
            ax.set_axis_off()
            return fig, ax
        
        # Create DataFrame
        df = pd.DataFrame(consistency_data)
        
        # Sort by consistency score
        df = df.sort_values('Consistency Score', ascending=False)
        
        # Create the figure
        fig, ax = self.create_figure()
        
        # Create bar chart
        bars = ax.bar(
            df['Fable ID'], 
            df['Consistency Score'],
            color=[self.theme_colors.get(theme, '#1f77b4') for theme in df['Dominant Theme']]
        )
        
        # Add score and theme labels to the bars
        for bar, theme in zip(bars, df['Dominant Theme']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f"{height:.2f}", ha='center', va='bottom')
            ax.text(bar.get_x() + bar.get_width()/2., height/2,
                   theme, ha='center', va='center', color='white',
                   fontweight='bold', rotation=90 if len(theme) > 10 else 0)
        
        # Set titles and labels
        ax.set_title('Moral Theme Consistency Across Languages', 
                    fontsize=16, pad=20)
        ax.set_xlabel('Fable ID', fontsize=12, labelpad=10)
        ax.set_ylabel('Consistency Score', fontsize=12, labelpad=10)
        
        # Set y-axis limit to 0-1 with a little padding
        ax.set_ylim(0, 1.1)
        
        # Create a legend for theme colors
        handles = [plt.Rectangle((0,0),1,1, color=self.theme_colors.get(theme, '#1f77b4')) 
                  for theme in set(df['Dominant Theme'])]
        labels = list(set(df['Dominant Theme']))
        ax.legend(handles, labels, title='Dominant Theme', loc='upper right')
        
        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add a subtle grid only on the y-axis
        ax.grid(axis='y', alpha=0.3)
        
        # Ensure layout fits everything
        plt.tight_layout()
        
        return fig, ax
    
    def plot_keywords_heatmap(self, language):
        """
        Create a heatmap visualizing keywords from moral analysis.
        
        Args:
            language: Language code ('en', 'de', 'nl', 'es', 'grc')
            
        Returns:
            tuple: (figure, axes) The created matplotlib figure and axes
        """
        # Load the moral data
        data = self.load_moral_data(language)
        
        if not data:
            # If no data, create an empty figure with a message
            fig, ax = self.create_figure()
            ax.text(0.5, 0.5, f"No moral data available for {language}", 
                    ha='center', va='center', fontsize=14)
            ax.set_axis_off()
            return fig, ax
        
        # Extract keywords data
        keywords_data = []
        
        for fable in data:
            if isinstance(fable, dict):
                # Try to extract keywords
                keywords = fable.get('keywords', [])
                fable_id = fable.get('fable_id', fable.get('id', 'Unknown'))
                
                # If no direct keywords, try other locations
                if not keywords and 'method' in fable and 'keywords' in fable:
                    keywords = fable['keywords']
                elif not keywords and 'implicit' in fable and 'keywords' in fable['implicit']:
                    keywords = fable['implicit']['keywords']
                
                for keyword in keywords:
                    # Handle different keyword formats
                    if isinstance(keyword, dict):
                        keywords_data.append({
                            'Fable ID': fable_id,
                            'Keyword': keyword.get('term', 'Unknown'),
                            'Score': keyword.get('score', 0)
                        })
                    elif isinstance(keyword, tuple) and len(keyword) >= 2:
                        keywords_data.append({
                            'Fable ID': fable_id,
                            'Keyword': keyword[0],
                            'Score': keyword[1]
                        })
        
        # If no keywords found, show message
        if not keywords_data:
            fig, ax = self.create_figure()
            ax.text(0.5, 0.5, f"No keywords found for {language}", 
                   ha='center', va='center', fontsize=14)
            ax.set_axis_off()
            return fig, ax
        
        # Create DataFrame
        df = pd.DataFrame(keywords_data)
        
        # Get top keywords across all fables
        top_keywords = df.groupby('Keyword')['Score'].sum().nlargest(20).index.tolist()
        
        # Get unique fable IDs
        fable_ids = sorted(df['Fable ID'].unique())
        
        # Create keyword matrix
        keyword_matrix = np.zeros((len(top_keywords), len(fable_ids)))
        
        for i, keyword in enumerate(top_keywords):
            for j, fable_id in enumerate(fable_ids):
                scores = df[(df['Keyword'] == keyword) & (df['Fable ID'] == fable_id)]['Score'].tolist()
                if scores:
                    keyword_matrix[i, j] = scores[0]
        
        # Create the figure
        fig, ax = self.create_figure(figsize=(max(8, len(fable_ids) * 0.8), max(6, len(top_keywords) * 0.4)))
        
        # Create heatmap
        im = ax.imshow(keyword_matrix, cmap='Blues')  # Changed from 'YlGnBu' to 'Blues'
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Keyword Score', rotation=-90, va="bottom")
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(fable_ids)))
        ax.set_yticks(np.arange(len(top_keywords)))
        ax.set_xticklabels(fable_ids)
        ax.set_yticklabels(top_keywords)
        
        # Rotate x tick labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Set titles and labels
        language_name = self.language_names.get(language, language)
        ax.set_title(f'Keyword Heatmap for {language_name} Fables', 
                    fontsize=16, pad=20)
        ax.set_xlabel('Fable ID', fontsize=12, labelpad=10)
        ax.set_ylabel('Keyword', fontsize=12, labelpad=10)
        
        # Ensure layout fits everything
        plt.tight_layout()
        
        return fig, ax