# entity_analysis_plot.py
"""
Visualization components for entity analysis results.

This module provides classes for visualizing entity analysis results,
including entity type distributions, cross-language comparisons,
and entity patterns within fables.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import json
from ..core.figure_builder import FigureBuilder


class EntityAnalysisPlot(FigureBuilder):
    """Creates visualizations of entity analysis results."""
    
    def __init__(self, theme='default', fig_size=(12, 8), output_dir=None):
        """Initialize the entity analysis plotter with default settings."""
        super().__init__(theme=theme, fig_size=fig_size, output_dir=output_dir)
        
        # Define language names for better labels
        self.language_names = {
            'en': 'English',
            'de': 'German',
            'nl': 'Dutch',
            'es': 'Spanish',
            'grc': 'Ancient Greek'
        }
        
        # Define color schemes for different entity types
        self.entity_colors = {
            'ANIMAL_CHAR': '#4C72B0',  # A distinctive blue
            'PERSON': '#55A868',       # Green
            'GPE': '#C44E52',          # Red
            'ORG': '#8172B3',          # Purple
            'LOC': '#CCB974',          # Yellow/gold
            'PER': '#64B5CD',          # Light blue
            'MISC': '#B47CC7',         # Pink/purple
            'DATE': '#BCB3D9',         # Lavender
            'CARDINAL': '#FF9D45',     # Orange
            'ORDINAL': '#8C8C8C',      # Gray
            'WORK_OF_ART': '#FFBE7D'   # Light orange
        }
    
    def load_entity_data(self, language):
        """
        Load entity analysis data for a specific language.
        
        Args:
            language: Language code ('en', 'de', 'nl', 'es', 'grc')
            
        Returns:
            Dict containing the loaded data, or empty dict if not found
        """
        # Filename formats to try
        filenames = [
            f"entity_{language}.json",
            f"entity_stats_{language}.json",
            f"entity_analysis_{language}.json"
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
                    self.logger.info(f"Found entity data at: {path}")
                    try:
                        with open(path, 'r', encoding='utf-8') as f:
                            return json.load(f)
                    except Exception as e:
                        self.logger.error(f"Error loading entity data: {e}")
                        continue
        
        self.logger.warning(f"No entity data found for language: {language}")
        return {}
    
    def load_all_entity_data(self):
        """
        Load entity analysis data for all available languages.
        
        Returns:
            Dict mapping language codes to their entity data
        """
        all_data = {}
        for lang in self.language_names.keys():
            data = self.load_entity_data(lang)
            if data:
                all_data[lang] = data
        
        return all_data
    
    def plot_entity_distribution(self, language):
        """
        Create a bar chart showing the distribution of entity types for a language.
        
        Args:
            language: Language code ('en', 'de', 'nl', 'es', 'grc')
            
        Returns:
            tuple: (figure, axes) The created matplotlib figure and axes
        """
        # Load the entity data
        data = self.load_entity_data(language)
        
        if not data:
            # If no data, create an empty figure with a message
            fig, ax = self.create_figure()
            ax.text(0.5, 0.5, f"No entity data available for {language}", 
                    ha='center', va='center', fontsize=14)
            ax.set_axis_off()
            return fig, ax
        
        # Prepare data for plotting
        entity_types = []
        percentages = []
        counts = []
        colors = []
        
        for entity_type, entity_info in data.items():
            entity_types.append(entity_type)
            percentages.append(entity_info.get('percentage', 0))
            counts.append(entity_info.get('count', 0))
            colors.append(self.entity_colors.get(entity_type, '#1f77b4'))  # Default to a standard color if not found
        
        # Create DataFrame
        df = pd.DataFrame({
            'Entity Type': entity_types,
            'Percentage': percentages,
            'Count': counts
        })
        
        # Sort by percentage
        df = df.sort_values('Percentage', ascending=False)
        
        # Create the figure
        fig, ax = self.create_figure()
        
        # Create bar chart
        bars = ax.bar(df['Entity Type'], df['Percentage'], 
                     color=[self.entity_colors.get(entity, '#1f77b4') for entity in df['Entity Type']])
        
        # Add count labels to the bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{int(df.loc[df["Entity Type"] == bar.get_x(), "Count"].values[0])}', 
                   ha='center', va='bottom')
        
        # Set titles and labels
        language_name = self.language_names.get(language, language)
        ax.set_title(f'Entity Type Distribution in {language_name} Fables', 
                    fontsize=16, pad=20)
        ax.set_xlabel('Entity Type', fontsize=12, labelpad=10)
        ax.set_ylabel('Percentage (%)', fontsize=12, labelpad=10)
        
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
    
    def plot_entity_comparison(self):
        """
        Create a comparative bar chart showing entity distributions across languages.
        
        Returns:
            tuple: (figure, axes) The created matplotlib figure and axes
        """
        # Load entity data for all languages
        all_data = self.load_all_entity_data()
        
        if not all_data:
            # If no data, create an empty figure with a message
            fig, ax = self.create_figure()
            ax.text(0.5, 0.5, "No entity data available for any language", 
                    ha='center', va='center', fontsize=14)
            ax.set_axis_off()
            return fig, ax
        
        # Gather all unique entity types across languages
        all_entity_types = set()
        for lang_data in all_data.values():
            all_entity_types.update(lang_data.keys())
        
        # Prepare data for plotting
        plot_data = []
        
        for lang, lang_data in all_data.items():
            language_name = self.language_names.get(lang, lang)
            
            for entity_type in all_entity_types:
                if entity_type in lang_data:
                    entity_info = lang_data[entity_type]
                    plot_data.append({
                        'Language': language_name,
                        'Entity Type': entity_type,
                        'Percentage': entity_info.get('percentage', 0),
                        'Count': entity_info.get('count', 0)
                    })
                else:
                    # Include zero values for missing entity types
                    plot_data.append({
                        'Language': language_name,
                        'Entity Type': entity_type,
                        'Percentage': 0,
                        'Count': 0
                    })
        
        # Create DataFrame
        df = pd.DataFrame(plot_data)
        
        # Create the figure
        fig, ax = self.create_figure(figsize=(14, 10))
        
        # Create grouped bar chart
        sns.barplot(
            x='Entity Type', 
            y='Percentage', 
            hue='Language', 
            data=df,
            ax=ax,
            palette='viridis'
        )
        
        # Set titles and labels
        ax.set_title('Entity Type Distribution Across Languages', 
                    fontsize=16, pad=20)
        ax.set_xlabel('Entity Type', fontsize=12, labelpad=10)
        ax.set_ylabel('Percentage (%)', fontsize=12, labelpad=10)
        
        # Rotate x tick labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Improve legend
        ax.legend(title='Language', loc='upper right')
        
        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add a subtle grid only on the y-axis
        ax.grid(axis='y', alpha=0.3)
        
        # Ensure layout fits everything
        plt.tight_layout()
        
        return fig, ax
    
    def plot_entity_heatmap(self):
        """
        Create a heatmap showing entity type distribution across languages.
        
        Returns:
            tuple: (figure, axes) The created matplotlib figure and axes
        """
        # Load entity data for all languages
        all_data = self.load_all_entity_data()
        
        if not all_data:
            # If no data, create an empty figure with a message
            fig, ax = self.create_figure()
            ax.text(0.5, 0.5, "No entity data available for any language", 
                    ha='center', va='center', fontsize=14)
            ax.set_axis_off()
            return fig, ax
        
        # Gather all unique entity types across languages
        all_entity_types = set()
        for lang_data in all_data.values():
            all_entity_types.update(lang_data.keys())
        
        # Sort entity types for consistent display
        all_entity_types = sorted(all_entity_types)
        
        # Create a matrix of percentages
        languages = sorted(all_data.keys())
        percentage_matrix = np.zeros((len(languages), len(all_entity_types)))
        
        for i, lang in enumerate(languages):
            lang_data = all_data[lang]
            for j, entity_type in enumerate(all_entity_types):
                if entity_type in lang_data:
                    percentage_matrix[i, j] = lang_data[entity_type].get('percentage', 0)
        
        # Create the figure
        fig, ax = self.create_figure(figsize=(14, 10))
        
        # Get language names for y-axis labels
        language_names = [self.language_names.get(lang, lang) for lang in languages]
        
        # Create heatmap
        sns.heatmap(
            percentage_matrix,
            annot=True,
            fmt=".1f",
            cmap="YlGnBu",
            xticklabels=all_entity_types,
            yticklabels=language_names,
            ax=ax
        )
        
        # Set titles and labels
        ax.set_title('Entity Type Distribution Heatmap Across Languages', 
                    fontsize=16, pad=20)
        ax.set_xlabel('Entity Type', fontsize=12, labelpad=10)
        ax.set_ylabel('Language', fontsize=12, labelpad=10)
        
        # Rotate x tick labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Ensure layout fits everything
        plt.tight_layout()
        
        return fig, ax
    
    def plot_entity_examples(self, language, entity_type=None):
        """
        Create a visualization of entity examples for a specific language.
        
        Args:
            language: Language code ('en', 'de', 'nl', 'es', 'grc')
            entity_type: Specific entity type to show examples for (None for all)
            
        Returns:
            tuple: (figure, axes) The created matplotlib figure and axes
        """
        # Load the entity data
        data = self.load_entity_data(language)
        
        if not data:
            # If no data, create an empty figure with a message
            fig, ax = self.create_figure()
            ax.text(0.5, 0.5, f"No entity data available for {language}", 
                    ha='center', va='center', fontsize=14)
            ax.set_axis_off()
            return fig, ax
        
        # Filter for specific entity type if provided
        if entity_type:
            if entity_type in data:
                entity_data = {entity_type: data[entity_type]}
            else:
                # Entity type not found
                fig, ax = self.create_figure()
                ax.text(0.5, 0.5, f"Entity type '{entity_type}' not found for {language}", 
                       ha='center', va='center', fontsize=14)
                ax.set_axis_off()
                return fig, ax
        else:
            entity_data = data
        
        # Prepare data for plotting
        plot_data = []
        
        for ent_type, ent_info in entity_data.items():
            examples = ent_info.get('examples', [])
            for example in examples:
                plot_data.append({
                    'Entity Type': ent_type,
                    'Example': example
                })
        
        # Create DataFrame
        df = pd.DataFrame(plot_data)
        
        # If empty, show message
        if df.empty:
            fig, ax = self.create_figure()
            ax.text(0.5, 0.5, f"No entity examples found for {language}", 
                   ha='center', va='center', fontsize=14)
            ax.set_axis_off()
            return fig, ax
        
        # Sort by entity type
        df = df.sort_values('Entity Type')
        
        # Create figure for a table-like visualization
        fig, ax = self.create_figure(figsize=(14, max(6, len(df) * 0.3)))
        
        # Hide axes
        ax.axis('off')
        
        # Create table
        table = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            cellLoc='center',
            loc='center',
            bbox=[0.1, 0.1, 0.8, 0.8]
        )
        
        # Customize table appearance
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        # Color header
        for (i, j), cell in table.get_celld().items():
            if i == 0:  # Header row
                cell.set_facecolor('#4C72B0')
                cell.set_text_props(color='white', fontweight='bold')
            elif j == 0:  # Entity type column
                entity_type = df.iloc[i-1, 0]
                cell.set_facecolor(self.entity_colors.get(entity_type, '#1f77b4'))
                cell.set_text_props(color='white')
        
        # Set title
        language_name = self.language_names.get(language, language)
        title = f'Entity Examples in {language_name} Fables'
        if entity_type:
            title += f' - {entity_type}'
        plt.suptitle(title, fontsize=16, y=0.98)
        
        # Ensure layout fits everything
        plt.tight_layout()
        
        return fig, ax
    
    def plot_top_entities(self, n=10):
        """
        Create a bar chart showing the top N most frequent entities across all languages.
        
        Args:
            n: Number of top entities to show
            
        Returns:
            tuple: (figure, axes) The created matplotlib figure and axes
        """
        # Load entity data for all languages
        all_data = self.load_all_entity_data()
        
        if not all_data:
            # If no data, create an empty figure with a message
            fig, ax = self.create_figure()
            ax.text(0.5, 0.5, "No entity data available for any language", 
                    ha='center', va='center', fontsize=14)
            ax.set_axis_off()
            return fig, ax
        
        # Count mentions across all languages
        entity_mentions = {}
        entity_types = {}
        
        for lang, lang_data in all_data.items():
            for entity_type, entity_info in lang_data.items():
                examples = entity_info.get('examples', [])
                for example in examples:
                    if example in entity_mentions:
                        entity_mentions[example] += 1
                    else:
                        entity_mentions[example] = 1
                        entity_types[example] = entity_type
        
        # Get top N entities
        top_entities = sorted(entity_mentions.items(), key=lambda x: x[1], reverse=True)[:n]
        
        # Create DataFrame
        df = pd.DataFrame([
            {'Entity': entity, 'Mentions': count, 'Type': entity_types.get(entity, 'Unknown')}
            for entity, count in top_entities
        ])
        
        # Create the figure
        fig, ax = self.create_figure()
        
        # Create bar chart
        bars = ax.bar(
            df['Entity'], 
            df['Mentions'],
            color=[self.entity_colors.get(entity_types.get(entity, 'Unknown'), '#1f77b4') 
                  for entity in df['Entity']]
        )
        
        # Add count labels to the bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{int(height)}', ha='center', va='bottom')
        
        # Set titles and labels
        ax.set_title(f'Top {n} Most Frequent Entities Across All Languages', 
                    fontsize=16, pad=20)
        ax.set_xlabel('Entity', fontsize=12, labelpad=10)
        ax.set_ylabel('Number of Mentions', fontsize=12, labelpad=10)
        
        # Rotate x tick labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Create a legend for entity types
        handles = [plt.Rectangle((0,0),1,1, color=self.entity_colors.get(etype, '#1f77b4')) 
                  for etype in set(df['Type'])]
        labels = list(set(df['Type']))
        ax.legend(handles, labels, title='Entity Type', loc='upper right')
        
        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add a subtle grid only on the y-axis
        ax.grid(axis='y', alpha=0.3)
        
        # Ensure layout fits everything
        plt.tight_layout()
        
        return fig, ax