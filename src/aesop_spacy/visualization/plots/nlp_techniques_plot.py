# nlp_techniques_plot.py
"""
Visualizations for NLP techniques analysis results including TF-IDF and topic modeling.

This module provides visualizations for term frequency analysis, topic modeling,
and cross-language NLP analysis of Aesop's fables.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from ..core.figure_builder import FigureBuilder

class NLPTechniquesPlot(FigureBuilder):
    """Visualizations for TF-IDF and topic modeling analysis results."""

    def __init__(self, theme='default', fig_size=(12, 8), output_dir=None):
        """Initialize the NLP techniques visualizer."""
        super().__init__(theme=theme, fig_size=fig_size, output_dir=output_dir)
        
        # Define language names for better labels
        self.language_names = {
            'en': 'English',
            'de': 'German',
            'nl': 'Dutch',
            'es': 'Spanish',
            'grc': 'Ancient Greek'
        }
        
        # Custom colormap for text importance
        self.importance_cmap = LinearSegmentedColormap.from_list(
            'importance', 
            ['#f7fbff', '#08306b'],  # Light blue to dark blue
            N=100
        )
        
    def plot_tfidf_top_terms(self, n_terms=20):
        """
        Create a horizontal bar chart of the most important terms across all documents.
        
        Args:
            n_terms: Number of top terms to display
            
        Returns:
            tuple: (figure, axes) The created matplotlib figure and axes
        """
        # Load TF-IDF data
        tfidf_data = self.load_analysis_data("nlp/all_tfidf.json")
        
        if not tfidf_data or 'top_terms_overall' not in tfidf_data:
            # If no data, create an empty figure with a message
            fig, ax = self.create_figure()
            ax.text(0.5, 0.5, "No TF-IDF data available", 
                   ha='center', va='center', fontsize=14)
            ax.set_axis_off()
            return fig, ax
        
        # Extract top terms
        top_terms = tfidf_data['top_terms_overall'][:n_terms]
        
        # Convert to DataFrame
        df = pd.DataFrame(top_terms)
        
        # Create figure
        fig, ax = self.create_figure(figsize=(10, n_terms * 0.4))
        
        # Create horizontal bar chart
        bars = ax.barh(df['term'], df['score'], color=self.palettes['sequential'])
        
        # Add scores to the bars
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width + 0.1
            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.2f}',
                   va='center', fontsize=9)
        
        # Set titles and labels
        ax.set_title('Most Important Terms Across All Fables (TF-IDF)', fontsize=16, pad=20)
        ax.set_xlabel('TF-IDF Score', fontsize=12, labelpad=10)
        ax.set_ylabel('Term', fontsize=12, labelpad=10)
        
        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add a subtle grid only on the x-axis
        ax.grid(axis='x', alpha=0.3)
        
        # Ensure layout fits everything
        plt.tight_layout()
        
        return fig, ax
    
    def plot_language_term_heatmap(self, n_terms=15):
        """
        Create a heatmap showing term importance across different languages.
        
        Args:
            n_terms: Number of top terms per language to include
            
        Returns:
            tuple: (figure, axes) The created matplotlib figure and axes
        """
        # Load TF-IDF data
        tfidf_data = self.load_analysis_data("nlp/all_tfidf.json")
        
        if not tfidf_data or 'document_info' not in tfidf_data:
            # If no data, create an empty figure with a message
            fig, ax = self.create_figure()
            ax.text(0.5, 0.5, "No TF-IDF data available", 
                   ha='center', va='center', fontsize=14)
            ax.set_axis_off()
            return fig, ax
        
        # Group documents by language
        language_terms = {}
        for doc in tfidf_data['document_info']:
            lang = doc.get('language')
            if lang not in language_terms:
                language_terms[lang] = {}
            
            # Add terms for this language
            for term_info in doc.get('top_terms', []):
                term = term_info.get('term')
                score = term_info.get('score', 0)
                
                if term in language_terms[lang]:
                    language_terms[lang][term] += score
                else:
                    language_terms[lang][term] = score
        
        # Get top terms for each language
        top_terms_by_lang = {}
        all_top_terms = set()
        
        for lang, terms in language_terms.items():
            # Sort terms by score and take top n
            sorted_terms = sorted(terms.items(), key=lambda x: x[1], reverse=True)[:n_terms]
            top_terms_by_lang[lang] = dict(sorted_terms)
            
            # Add to overall set of terms
            all_top_terms.update([t[0] for t in sorted_terms])
        
        # Create a matrix for the heatmap
        all_terms_list = sorted(list(all_top_terms))
        languages = sorted(language_terms.keys())
        
        # Create DataFrame for the heatmap
        heat_data = []
        for term in all_terms_list:
            row = {'Term': term}
            for lang in languages:
                lang_name = self.language_names.get(lang, lang)
                row[lang_name] = language_terms[lang].get(term, 0)
            heat_data.append(row)
        
        df = pd.DataFrame(heat_data)
        
        # Pivot for the heatmap
        pivot_df = df.set_index('Term')
        
        # Create figure
        fig, ax = self.create_figure(figsize=(12, len(all_terms_list) * 0.4))
        
        # Create heatmap
        sns.heatmap(
            pivot_df, 
            cmap=self.importance_cmap,
            annot=True, 
            fmt='.2f',
            linewidths=0.5,
            ax=ax
        )
        
        # Set titles
        ax.set_title('Term Importance Across Languages (TF-IDF)', fontsize=16, pad=20)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Ensure layout fits everything
        plt.tight_layout()
        
        return fig, ax
    
    def plot_document_term_matrix(self, language=None, n_docs=10, n_terms=15):
        """
        Create a document-term matrix visualization for a specific language.
        
        Args:
            language: Language code to filter by (None for all languages)
            n_docs: Maximum number of documents to show
            n_terms: Maximum number of terms per document
            
        Returns:
            tuple: (figure, axes) The created matplotlib figure and axes
        """
        # Load TF-IDF data
        tfidf_data = self.load_analysis_data("nlp/all_tfidf.json")
        
        if not tfidf_data or 'document_info' not in tfidf_data:
            # If no data, create an empty figure with a message
            fig, ax = self.create_figure()
            ax.text(0.5, 0.5, "No TF-IDF data available", 
                   ha='center', va='center', fontsize=14)
            ax.set_axis_off()
            return fig, ax
        
        # Filter documents by language if specified
        docs = tfidf_data['document_info']
        if language:
            docs = [doc for doc in docs if doc.get('language') == language]
            
        # Limit to n_docs
        docs = docs[:n_docs]
        
        # Extract document IDs and terms
        all_terms = set()
        for doc in docs:
            top_terms = doc.get('top_terms', [])[:n_terms]
            all_terms.update([term['term'] for term in top_terms])
        
        # Create matrix
        matrix = []
        doc_ids = []
        
        for doc in docs:
            doc_id = f"{doc.get('language')}_{doc.get('document_id')}"
            doc_ids.append(doc_id)
            
            term_scores = {term['term']: term['score'] for term in doc.get('top_terms', [])}
            row = [term_scores.get(term, 0) for term in sorted(all_terms)]
            matrix.append(row)
        
        # Convert to numpy array
        heat_data = np.array(matrix)
        
        # Create figure
        fig, ax = self.create_figure(figsize=(max(10, len(all_terms) * 0.5), len(doc_ids) * 0.5))
        
        # Create heatmap
        sns.heatmap(
            heat_data, 
            cmap=self.importance_cmap,
            annot=True, 
            fmt='.2f',
            xticklabels=sorted(all_terms),
            yticklabels=doc_ids,
            linewidths=0.5,
            ax=ax
        )
        
        # Set titles
        title = 'Document-Term Matrix'
        if language:
            title += f' ({self.language_names.get(language, language)})'
            
        ax.set_title(title, fontsize=16, pad=20)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Ensure layout fits everything
        plt.tight_layout()
        
        return fig, ax
    
    def plot_topic_term_distribution(self, topic_id=0):
        """
        Create a visualization of term weights for a specific topic.
        
        Args:
            topic_id: ID of the topic to visualize
            
        Returns:
            tuple: (figure, axes) The created matplotlib figure and axes
        """
        # Load topic modeling data
        topic_data = self.load_analysis_data("nlp/all_topic_modeling.json")
        
        if not topic_data or 'topics' not in topic_data:
            # If no data, create an empty figure with a message
            fig, ax = self.create_figure()
            ax.text(0.5, 0.5, "No topic modeling data available", 
                   ha='center', va='center', fontsize=14)
            ax.set_axis_off()
            return fig, ax
        
        # Get topics
        topics = topic_data['topics']
        
        # Validate topic_id
        if not (0 <= topic_id < len(topics)):
            topic_id = 0
        
        # Get the topic
        topic = topics[topic_id]
        
        # Extract terms and weights
        terms = [term['term'] for term in topic['top_terms']]
        weights = [term['weight'] for term in topic['top_terms']]
        
        # Create figure
        fig, ax = self.create_figure()
        
        # Create horizontal bar chart
        bars = ax.barh(terms, weights, color=self.palettes['sequential'])
        
        # Add weights to the bars
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width + 0.3
            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.2f}',
                   va='center', fontsize=9)
        
        # Set titles and labels
        ax.set_title(f'Term Distribution for Topic {topic_id}', fontsize=16, pad=20)
        ax.set_xlabel('Weight', fontsize=12, labelpad=10)
        ax.set_ylabel('Term', fontsize=12, labelpad=10)
        
        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add a subtle grid only on the x-axis
        ax.grid(axis='x', alpha=0.3)
        
        # Ensure layout fits everything
        plt.tight_layout()
        
        return fig, ax
    
    def plot_document_topic_distribution(self, doc_filter=None):
        """
        Create a visualization of topic distributions across documents.
        
        Args:
            doc_filter: Optional filter function to select which documents to include
            
        Returns:
            tuple: (figure, axes) The created matplotlib figure and axes
        """
        # Load topic modeling data
        topic_data = self.load_analysis_data("nlp/all_topic_modeling.json")
        
        if not topic_data or 'document_topics' not in topic_data:
            # If no data, create an empty figure with a message
            fig, ax = self.create_figure()
            ax.text(0.5, 0.5, "No topic modeling data available", 
                   ha='center', va='center', fontsize=14)
            ax.set_axis_off()
            return fig, ax
        
        # Get document topics
        doc_topics = topic_data['document_topics']
        
        # Apply filter if provided
        if doc_filter:
            doc_topics = [doc for doc in doc_topics if doc_filter(doc)]
        
        # Extract data for visualization
        doc_ids = []
        dominant_topics = []
        distributions = []
        
        for doc in doc_topics:
            doc_id = f"{doc.get('language')}_{doc.get('document_id')}"
            doc_ids.append(doc_id)
            
            dominant_topic = doc.get('dominant_topic')
            dominant_topics.append(dominant_topic)
            
            # Get full distribution
            distribution = doc.get('topic_distribution', [])
            distribution_dict = {d['topic_id']: d['weight'] for d in distribution}
            distributions.append(distribution_dict)
        
        # Create a DataFrame for plotting
        n_topics = len(topic_data.get('topics', []))
        data = []
        
        for i, doc_id in enumerate(doc_ids):
            for topic_id in range(n_topics):
                weight = distributions[i].get(topic_id, 0)
                data.append({
                    'Document': doc_id,
                    'Topic': f'Topic {topic_id}',
                    'Weight': weight,
                    'Dominant': topic_id == dominant_topics[i]
                })
        
        df = pd.DataFrame(data)
        
        # Create figure
        fig, ax = self.create_figure(figsize=(12, len(doc_ids) * 0.5))
        
        # Create heatmap
        pivot_df = df.pivot(index='Document', columns='Topic', values='Weight')
        
        sns.heatmap(
            pivot_df, 
            cmap=self.importance_cmap,
            annot=True, 
            fmt='.2f',
            linewidths=0.5,
            ax=ax
        )
        
        # Set titles
        ax.set_title('Topic Distribution Across Documents', fontsize=16, pad=20)
        
        # Ensure layout fits everything
        plt.tight_layout()
        
        return fig, ax
    
    def plot_language_topic_distribution(self):
        """
        Create a visualization showing which topics are dominant in each language.
        
        Returns:
            tuple: (figure, axes) The created matplotlib figure and axes
        """
        # Load topic modeling data
        topic_data = self.load_analysis_data("nlp/all_topic_modeling.json")
        
        if not topic_data or 'document_topics' not in topic_data:
            # If no data, create an empty figure with a message
            fig, ax = self.create_figure()
            ax.text(0.5, 0.5, "No topic modeling data available", 
                   ha='center', va='center', fontsize=14)
            ax.set_axis_off()
            return fig, ax
        
        # Get document topics
        doc_topics = topic_data['document_topics']
        
        # Count dominant topics by language
        lang_topic_counts = {}
        
        for doc in doc_topics:
            lang = doc.get('language')
            topic = doc.get('dominant_topic')
            
            if lang not in lang_topic_counts:
                lang_topic_counts[lang] = {}
            
            if topic not in lang_topic_counts[lang]:
                lang_topic_counts[lang][topic] = 0
                
            lang_topic_counts[lang][topic] += 1
        
        # Create data for plotting
        n_topics = len(topic_data.get('topics', []))
        data = []
        
        for lang, topic_counts in lang_topic_counts.items():
            lang_name = self.language_names.get(lang, lang)
            for topic in range(n_topics):
                count = topic_counts.get(topic, 0)
                data.append({
                    'Language': lang_name,
                    'Topic': f'Topic {topic}',
                    'Count': count
                })
        
        df = pd.DataFrame(data)
        
        # Create figure
        fig, ax = self.create_figure()
        
        # Create grouped bar chart
        sns.barplot(
            x='Language', 
            y='Count', 
            hue='Topic', 
            data=df,
            palette=self.palettes['categories'][:n_topics],
            ax=ax
        )
        
        # Set titles and labels
        ax.set_title('Dominant Topics by Language', fontsize=16, pad=20)
        ax.set_xlabel('Language', fontsize=12, labelpad=10)
        ax.set_ylabel('Number of Documents', fontsize=12, labelpad=10)
        
        # Improve the legend
        ax.legend(title='Topic', bbox_to_anchor=(1.02, 1), loc='upper left')
        
        # Ensure layout fits everything
        plt.tight_layout()
        
        return fig, ax
    
    def plot_topic_similarity_heatmap(self):
        """
        Create a visualization showing similarity between topics based on shared terms.
        
        Returns:
            tuple: (figure, axes) The created matplotlib figure and axes
        """
        # Load topic modeling data
        topic_data = self.load_analysis_data("nlp/all_topic_modeling.json")
        
        if not topic_data or 'topics' not in topic_data:
            # If no data, create an empty figure with a message
            fig, ax = self.create_figure()
            ax.text(0.5, 0.5, "No topic modeling data available", 
                   ha='center', va='center', fontsize=14)
            ax.set_axis_off()
            return fig, ax
        
        # Get topics
        topics = topic_data['topics']
        
        # Create term sets for each topic
        topic_terms = {}
        for topic in topics:
            topic_id = topic['topic_id']
            terms = {term['term'] for term in topic['top_terms']}
            topic_terms[topic_id] = terms
        
        # Calculate Jaccard similarity between topics
        n_topics = len(topics)
        similarity_matrix = np.zeros((n_topics, n_topics))
        
        for i in range(n_topics):
            for j in range(n_topics):
                if i == j:
                    similarity_matrix[i, j] = 1.0  # Self-similarity is 1
                else:
                    # Jaccard similarity: |A ∩ B| / |A ∪ B|
                    intersection = len(topic_terms[i] & topic_terms[j])
                    union = len(topic_terms[i] | topic_terms[j])
                    
                    similarity_matrix[i, j] = intersection / union if union > 0 else 0
        
        # Create figure
        fig, ax = self.create_figure()
        
        # Create heatmap
        sns.heatmap(
            similarity_matrix, 
            cmap='viridis',
            annot=True, 
            fmt='.2f',
            xticklabels=[f'Topic {i}' for i in range(n_topics)],
            yticklabels=[f'Topic {i}' for i in range(n_topics)],
            ax=ax
        )
        
        # Set titles
        ax.set_title('Topic Similarity (Jaccard Index of Shared Terms)', fontsize=16, pad=20)
        
        # Ensure layout fits everything
        plt.tight_layout()
        
        return fig, ax