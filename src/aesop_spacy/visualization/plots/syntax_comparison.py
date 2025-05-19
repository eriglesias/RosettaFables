# syntax_comparison.py
"""
Visualization components for syntax analysis results.

This module provides classes for visualizing syntax-related analysis results,
including dependency frequencies and parse tree shape metrics.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import json
from pathlib import Path
from ..core.figure_builder import FigureBuilder


class SyntaxAnalysisPlot(FigureBuilder):
    """Creates visualizations of syntax analysis results including dependency distributions and tree metrics."""
    
    def __init__(self, theme='default', fig_size=(12, 8), output_dir=None):
        """Initialize the syntax analysis plotter with default settings."""
        super().__init__(theme=theme, fig_size=fig_size, output_dir=output_dir)
        
        # Define mappings for dependency labels to more readable descriptions
        # These are specific to the Universal Dependencies framework (adjust for your schema)
        self.dependency_labels = {
            'nk': 'Noun Kernel',
            'sb': 'Subject',
            'ROOT': 'Root',
            'mo': 'Modifier',
            'oa': 'Accusative Object',
            'punct': 'Punctuation',
            'cd': 'Coordinating Conjunction',
            'cj': 'Conjunct',
            'da': 'Dative',
            'oc': 'Clausal Object',
            're': 'Repeated Element',
            'pnc': 'Proper Noun Component',
            'rc': 'Relative Clause',
            'pd': 'Predicate',
            'ep': 'Expletive',
            'pm': 'Morphological Particle',
            'dm': 'Discourse Marker',
            'cm': 'Comparative',
            'mnr': 'Manner',
            'cp': 'Complementizer',
            'ams': 'Measure Argument',
            'ng': 'Negation',
            'dep': 'Unspecified Dependency',
            'rs': 'Reported Speech'
        }
        
        # Define language names for better labels
        self.language_names = {
            'en': 'English',
            'de': 'German',
            'nl': 'Dutch',
            'es': 'Spanish',
            'grc': 'Ancient Greek'
        }
        
        # Create a default color palette for dependency types
        self.dep_palette = sns.color_palette('viridis', n_colors=len(self.dependency_labels))
    
    def load_syntax_data(self, fable_id, language, analysis_type):
        """
        Load syntax analysis data for a specific fable, language, and analysis type.
        
        Args:
            fable_id: ID of the fable
            language: Language code (e.g., 'en', 'de')
            analysis_type: Type of analysis ('dependency_frequencies', 'tree_shapes')
            
        Returns:
            Dict containing the loaded data, or empty dict if not found
        """
        # Correct filename format based on your actual files
        filename = f"{fable_id}_{language}_{analysis_type}.json"
        
        # Try to find the file in the syntax directory
        potential_paths = [
            # Primary path based on your screenshots
            Path(__file__).resolve().parents[3] / "data" / "data_handled" / "analysis" / "syntax" / filename,
            
            # Alternative paths if the above fails
            Path("data/data_handled/analysis/syntax") / filename,
            Path("./data/data_handled/analysis/syntax") / filename
        ]
        
        for path in potential_paths:
            if path.exists():
                self.logger.info(f"Found syntax data at: {path}")
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except Exception as e:
                    self.logger.error(f"Error loading syntax data: {e}")
                    continue
        
        self.logger.warning(f"No syntax data found for {fable_id}_{language}_{analysis_type}")
        return {}
        
    def plot_dependency_frequencies(self, fable_id, language_code, top_n=10):
        """
        Create a bar chart for dependency frequency distribution of a single language.
        
        Args:
            fable_id: ID of the fable
            language_code: Two-letter language code (e.g., 'en', 'de')
            top_n: Number of top dependency types to display
            
        Returns:
            tuple: (figure, axes) The created matplotlib figure and axes
        """
        # Load the dependency frequency data
        dep_data = self.load_syntax_data(fable_id, language_code, 'dependency_frequencies')
        
        if not dep_data or 'frequencies' not in dep_data:
            # If no data, create an empty figure with a message
            fig, ax = self.create_figure()
            ax.text(0.5, 0.5, f"No dependency frequency data available for {self.language_names.get(language_code, language_code)}", 
                    ha='center', va='center', fontsize=14)
            ax.set_axis_off()
            return fig, ax
        
        # Extract frequencies and convert to DataFrame
        frequencies = dep_data['frequencies']
        
        df = pd.DataFrame({
            'Dependency': list(frequencies.keys()),
            'Frequency (%)': list(frequencies.values())
        })
        
        # Add full descriptions for better readability
        df['Description'] = df['Dependency'].map(lambda x: self.dependency_labels.get(x, x))
        
        # Sort by frequency and take top N
        df = df.sort_values('Frequency (%)', ascending=False).head(top_n)
        
        # Create the figure
        fig, ax = self.create_figure()
        
        # Create horizontal bar chart for better readability with long labels
        bars = ax.barh(df['Description'], df['Frequency (%)'], 
                color=self.dep_palette[:len(df)])
        
        # Add percentage labels to the bars
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width + 0.5
            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.1f}%',
                   va='center', fontsize=10)
        
        # Set titles and labels
        language_name = self.language_names.get(language_code, language_code)
        ax.set_title(f'Dependency Relation Distribution in {language_name} (Fable {fable_id})', 
                    fontsize=16, pad=20)
        ax.set_xlabel('Frequency (%)', fontsize=12, labelpad=10)
        ax.set_ylabel('Dependency Type', fontsize=12, labelpad=10)
        
        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add a subtle grid only on the x-axis
        ax.grid(axis='x', alpha=0.3)
        
        # Ensure layout fits everything
        plt.tight_layout()
        
        return fig, ax
    
    def plot_dependency_comparison(self, fable_id, languages=None, top_n=8):
        """
        Create a grouped bar chart comparing dependency distributions across languages.
        
        Args:
            fable_id: ID of the fable
            languages: List of language codes to include. If None, use all available.
            top_n: Number of top dependency types to display
            
        Returns:
            tuple: (figure, axes) The created matplotlib figure and axes
        """
        # Default to all languages if none specified
        if languages is None:
            languages = ['en', 'de', 'nl', 'es', 'grc']
        
        # Load data for each language
        data_by_lang = {}
        for lang in languages:
            data = self.load_syntax_data(fable_id, lang, 'dependency_frequencies')
            if data and 'frequencies' in data:
                data_by_lang[lang] = data['frequencies']
        
        # If no data found, show a message
        if not data_by_lang:
            fig, ax = self.create_figure()
            ax.text(0.5, 0.5, "No dependency frequency data available for the requested languages", 
                    ha='center', va='center', fontsize=14)
            ax.set_axis_off()
            return fig, ax
        
        # Find the most common dependency types across all languages
        all_deps = {}
        for lang, dep_data in data_by_lang.items():
            for dep, freq in dep_data.items():
                if dep in all_deps:
                    all_deps[dep] += freq
                else:
                    all_deps[dep] = freq
        
        # Get top N most frequent overall
        top_deps = sorted(all_deps.keys(), key=lambda x: all_deps[x], reverse=True)[:top_n]
        
        # Create DataFrame for plotting
        plot_data = []
        for lang, dep_data in data_by_lang.items():
            lang_name = self.language_names.get(lang, lang)
            for dep in top_deps:
                plot_data.append({
                    'Language': lang_name,
                    'Dependency': dep,
                    'Description': self.dependency_labels.get(dep, dep),
                    'Frequency (%)': dep_data.get(dep, 0)
                })
        
        df = pd.DataFrame(plot_data)
        
        # Create the figure with a larger size for this visualization
        fig, ax = self.create_figure(figsize=(14, 8))
        
        # Create a grouped bar chart without assigning to unused variable
        sns.barplot(
            x='Description', 
            y='Frequency (%)', 
            hue='Language', 
            data=df,
            palette=self.palettes['languages'][:len(languages)],
            ax=ax
        )
        
        # Enhance the visualization
        ax.set_title(f'Dependency Relation Distribution Across Languages (Fable {fable_id})', 
                    fontsize=18, pad=20)
        ax.set_xlabel('Dependency Type', fontsize=14, labelpad=10)
        ax.set_ylabel('Frequency (%)', fontsize=14, labelpad=10)
        
        # Improve readability
        plt.xticks(rotation=45, ha='right')
        ax.legend(title='Language', title_fontsize=12, bbox_to_anchor=(1.02, 1), loc='upper left')
        
        # Add a subtle grid only on the y-axis for readability
        ax.grid(axis='y', alpha=0.3)
        
        # Ensure layout fits everything
        plt.tight_layout()
        
        return fig, ax
    
    def plot_tree_shapes(self, fable_id, language_code):
        """
        Create visualizations for tree shape metrics of a single language.
        
        Args:
            fable_id: ID of the fable
            language_code: Two-letter language code (e.g., 'en', 'de')
            
        Returns:
            tuple: (figure, axes) The created matplotlib figure and axes
        """
        # Load the tree shape data
        data = self.load_syntax_data(fable_id, language_code, 'tree_shapes')
        
        if not data:
            # If no data, create an empty figure with a message
            fig, ax = self.create_figure()
            ax.text(0.5, 0.5, f"No tree shape data available for {self.language_names.get(language_code, language_code)}", 
                    ha='center', va='center', fontsize=14)
            ax.set_axis_off()
            return fig, ax
        
        # Extract key metrics
        avg_branching = data.get('average_branching_factor', 0)
        max_branching = data.get('max_branching_factor', 0)
        avg_width_depth = data.get('average_width_depth_ratio', 0)
        non_projective = data.get('non_projective_count', 0)
        sentence_count = data.get('sentence_count', 1)
        
        # Calculate non-projective percentage
        non_projective_pct = (non_projective / sentence_count) * 100 if sentence_count > 0 else 0
        
        # Create figure with multiple subplots
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Branching factors
        ax1 = axs[0, 0]
        metrics = ['Average Branching Factor', 'Max Branching Factor']
        values = [avg_branching, max_branching]
        ax1.bar(metrics, values, color=['#3498db', '#e74c3c'])
        ax1.set_title('Tree Branching Factors', fontsize=14)
        ax1.set_ylim(0, max(max_branching * 1.2, 1))
        
        # Add value labels
        for i, v in enumerate(values):
            ax1.text(i, v + 0.1, f'{v:.2f}', ha='center', fontsize=12)
        
        # 2. Width-to-Depth Ratio
        ax2 = axs[0, 1]
        ax2.bar(['Width-to-Depth Ratio'], [avg_width_depth], color='#2ecc71')
        ax2.set_title('Average Width-to-Depth Ratio', fontsize=14)
        ax2.set_ylim(0, max(avg_width_depth * 1.2, 1))
        ax2.text(0, avg_width_depth + 0.02, f'{avg_width_depth:.2f}', ha='center', fontsize=12)
        
        # 3. Non-projective sentences
        ax3 = axs[1, 0]
        ax3.pie([non_projective, sentence_count - non_projective], 
                labels=['Non-Projective', 'Projective'],
                autopct='%1.1f%%',
                colors=['#e74c3c', '#2ecc71'],
                explode=(0.1, 0),
                shadow=True)
        ax3.set_title('Sentence Projectivity', fontsize=14)
        
        # 4. Language insights
        ax4 = axs[1, 1]
        ax4.axis('off')  # Turn off axes
        insights = data.get('language_insights', {})
        
        # Create a text box with insights
        textstr = '\n'.join([
            'Language Insights:',
            '',
            f"• {insights.get('typical_branching', 'No branching info')}",
            '',
            f"• {insights.get('typical_width_depth', 'No width-depth info')}",
            '',
            f"• Non-projective sentences: {non_projective}/{sentence_count} ({non_projective_pct:.1f}%)"
        ])
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax4.text(0.5, 0.5, textstr, transform=ax4.transAxes, fontsize=12,
                verticalalignment='center', horizontalalignment='center', bbox=props)
        
        # Overall title
        language_name = self.language_names.get(language_code, language_code)
        plt.suptitle(f'Parse Tree Shape Analysis for {language_name} (Fable {fable_id})', 
                    fontsize=16, y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
        
        return fig, axs
    
    def plot_tree_shapes_comparison(self, fable_id, languages=None):
        """
        Create comparison visualizations for tree shape metrics across languages.
        
        Args:
            fable_id: ID of the fable
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
            data = self.load_syntax_data(fable_id, lang, 'tree_shapes')
            if data:
                data_by_lang[lang] = data
        
        # If no data found, show a message
        if not data_by_lang:
            fig, ax = self.create_figure()
            ax.text(0.5, 0.5, "No tree shape data available for the requested languages", 
                    ha='center', va='center', fontsize=14)
            ax.set_axis_off()
            return fig, ax
        
        # Create a dataframe for plotting
        comparison_data = []
        
        for lang, data in data_by_lang.items():
            comparison_data.append({
                'Language': self.language_names.get(lang, lang),
                'Average Branching Factor': data.get('average_branching_factor', 0),
                'Max Branching Factor': data.get('max_branching_factor', 0),
                'Width-Depth Ratio': data.get('average_width_depth_ratio', 0),
                'Non-Projective %': (data.get('non_projective_count', 0) / data.get('sentence_count', 1)) * 100
            })
        
        df = pd.DataFrame(comparison_data)
        
        # Create figure with multiple metrics
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Average Branching Factor
        sns.barplot(x='Language', y='Average Branching Factor', data=df, ax=axs[0, 0], palette='viridis')
        axs[0, 0].set_title('Average Branching Factor by Language', fontsize=14)
        axs[0, 0].set_xlabel('')
        axs[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Max Branching Factor
        sns.barplot(x='Language', y='Max Branching Factor', data=df, ax=axs[0, 1], palette='viridis')
        axs[0, 1].set_title('Maximum Branching Factor by Language', fontsize=14)
        axs[0, 1].set_xlabel('')
        axs[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Width-Depth Ratio
        sns.barplot(x='Language', y='Width-Depth Ratio', data=df, ax=axs[1, 0], palette='viridis')
        axs[1, 0].set_title('Width-to-Depth Ratio by Language', fontsize=14)
        axs[1, 0].set_xlabel('')
        axs[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Non-Projective Percentage
        sns.barplot(x='Language', y='Non-Projective %', data=df, ax=axs[1, 1], palette='viridis')
        axs[1, 1].set_title('Non-Projective Sentences (%)', fontsize=14)
        axs[1, 1].set_xlabel('')
        axs[1, 1].tick_params(axis='x', rotation=45)
        
        # Overall title
        plt.suptitle(f'Parse Tree Shape Comparison Across Languages (Fable {fable_id})', 
                    fontsize=16, y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
        
        return fig, axs
    
    def plot_dependency_heatmap(self, fable_id, languages=None, top_n=15):
        """
        Create a heatmap visualization of dependency frequencies across languages.
        
        Args:
            fable_id: ID of the fable
            languages: List of language codes to include. If None, use all available.
            top_n: Number of top dependencies to include
            
        Returns:
            tuple: (figure, axes) The created matplotlib figure and axes
        """
        # Default to all languages if none specified
        if languages is None:
            languages = ['en', 'de', 'nl', 'es', 'grc']
        
        # Load data for each language
        data_by_lang = {}
        for lang in languages:
            data = self.load_syntax_data(fable_id, lang, 'dependency_frequencies')
            if data and 'frequencies' in data:
                data_by_lang[lang] = data['frequencies']
        
        # If no data found, show a message
        if not data_by_lang:
            fig, ax = self.create_figure()
            ax.text(0.5, 0.5, "No dependency frequency data available for the requested languages", 
                    ha='center', va='center', fontsize=14)
            ax.set_axis_off()
            return fig, ax
        
        # Find the most common dependency types across all languages
        all_deps = {}
        for lang, dep_data in data_by_lang.items():
            for dep, freq in dep_data.items():
                if dep in all_deps:
                    all_deps[dep] += freq
                else:
                    all_deps[dep] = freq
        
        # Get top N most frequent overall
        top_deps = sorted(all_deps.keys(), key=lambda x: all_deps[x], reverse=True)[:top_n]
        
        # Create a matrix for the heatmap
        matrix = []
        rows = []
        for lang, dep_data in data_by_lang.items():
            row = [dep_data.get(dep, 0) for dep in top_deps]
            matrix.append(row)
            rows.append(self.language_names.get(lang, lang))
        
        # Convert to numpy array
        heat_data = np.array(matrix)
        
        # Create column labels with better descriptions
        col_labels = [f"{dep} ({self.dependency_labels.get(dep, dep)})" for dep in top_deps]
        
        # Create figure
        fig, ax = self.create_figure(figsize=(16, 10))
        
        # Create heatmap
        sns.heatmap(
            heat_data, 
            annot=True, 
            fmt='.1f',
            cmap='viridis',
            xticklabels=col_labels,
            yticklabels=rows,
            ax=ax
        )
        
        # Enhance the visualization
        ax.set_title(f'Dependency Frequency Heatmap (Fable {fable_id})', fontsize=18, pad=20)
        
        # Improve readability
        plt.xticks(rotation=45, ha='right')
        
        # Add colorbar title
        cbar = ax.collections[0].colorbar
        cbar.set_label('Frequency (%)', rotation=270, labelpad=20)
        
        # Ensure layout fits everything
        plt.tight_layout()
        
        return fig, ax