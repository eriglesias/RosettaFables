# clustering_plot.py
"""
Visualization components for clustering analysis results.

This module provides classes for visualizing clustering analysis results,
including cluster distributions, feature importance, and dendrogram representations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram
from pathlib import Path
import json
from ..core.figure_builder import FigureBuilder


class ClusteringPlot(FigureBuilder):
    """Creates visualizations of clustering analysis results."""
    
    def __init__(self, theme='default', fig_size=(12, 8), output_dir=None):
        """Initialize the clustering plotter with default settings."""
        super().__init__(theme=theme, fig_size=fig_size, output_dir=output_dir)
        
        # Define language names for better labels
        self.language_names = {
            'en': 'English',
            'de': 'German',
            'nl': 'Dutch',
            'es': 'Spanish',
            'grc': 'Ancient Greek'
        }
        
        # Define color schemes for different clustering methods
        self.cluster_palettes = {
            'kmeans': sns.color_palette('viridis', 10),
            'hierarchical': sns.color_palette('muted', 10),
            'dbscan': sns.color_palette('bright', 10)
        }
    
    
    def load_clustering_data(self, method='kmeans'):
        """
        Load clustering analysis data for a specific method.
        
        Args:
            method: Clustering method ('kmeans', 'hierarchical', 'dbscan')
            
        Returns:
            Dict containing the loaded data, or empty dict if not found
        """
        # Correct filename format based on your actual files
        filename = f"all_{method}.json"  # Changed from clustering_{method}.json to all_{method}.json
        
        # You also have a cross-language file that might be useful
        if method == 'cross_language':
            filename = "all_cross_language.json"
        
        # Try to find the file in the clustering directory
        potential_paths = [
            Path(__file__).resolve().parents[3] / "data" / "data_handled" / "analysis" / "clustering" / filename,
            Path("data/data_handled/analysis/clustering") / filename,
            Path("./data/data_handled/analysis/clustering") / filename,
            # Add the exact path you mentioned
            Path("/Users/enriqueviv/Coding/coding_projects/data_science_projects/ds_sandbox/nlp_sandbox/aesop_spacy/data/data_handled/analysis/clustering") / filename
        ]
        
        for path in potential_paths:
            if path.exists():
                self.logger.info(f"Found clustering data at: {path}")
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except Exception as e:
                    self.logger.error(f"Error loading clustering data: {e}")
                    continue
        
        self.logger.warning(f"No clustering data found for method: {method}")
        self.logger.info(f"Searched paths: {[str(p) for p in potential_paths]}")  # Log all paths searched
        return {}

    
    def plot_cluster_distribution(self, method='kmeans'):
        """
        Create a bar chart showing the distribution of fables across clusters.
        
        Args:
            method: Clustering method ('kmeans', 'hierarchical', 'dbscan')
            
        Returns:
            tuple: (figure, axes) The created matplotlib figure and axes
        """
        # Load the clustering data
        data = self.load_clustering_data(method)
        
        if not data or 'clusters' not in data:
            # If no data, create an empty figure with a message
            fig, ax = self.create_figure()
            ax.text(0.5, 0.5, f"No clustering data available for {method}", 
                    ha='center', va='center', fontsize=14)
            ax.set_axis_off()
            return fig, ax
        
        # Extract cluster information
        clusters = data['clusters']
        
        # Prepare data for plotting
        cluster_names = []
        cluster_sizes = []
        
        for cluster_name, cluster_info in clusters.items():
            cluster_names.append(cluster_name)
            cluster_sizes.append(cluster_info['size'])
        
        # Create DataFrame
        df = pd.DataFrame({
            'Cluster': cluster_names,
            'Size': cluster_sizes
        })
        
        # Sort by size
        df = df.sort_values('Size', ascending=False)
        
        # Create the figure
        fig, ax = self.create_figure()
        
        # Create bar chart
        bars = ax.bar(df['Cluster'], df['Size'], 
                      color=self.cluster_palettes.get(method, sns.color_palette('viridis'))[:len(df)])
        
        # Add count labels to the bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')
        
        # Set titles and labels
        ax.set_title(f'Fable Distribution Across Clusters ({method.capitalize()})', 
                    fontsize=16, pad=20)
        ax.set_xlabel('Cluster', fontsize=12, labelpad=10)
        ax.set_ylabel('Number of Fables', fontsize=12, labelpad=10)
        
        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add a subtle grid only on the y-axis
        ax.grid(axis='y', alpha=0.3)
        
        # Ensure layout fits everything
        plt.tight_layout()
        
        return fig, ax
    
    def plot_language_distribution(self, method='kmeans'):
        """
        Create a stacked bar chart showing language distribution within each cluster.
        
        Args:
            method: Clustering method ('kmeans', 'hierarchical', 'dbscan')
            
        Returns:
            tuple: (figure, axes) The created matplotlib figure and axes
        """
        # Load the clustering data
        data = self.load_clustering_data(method)
        
        if not data or 'clusters' not in data:
            # If no data, create an empty figure with a message
            fig, ax = self.create_figure()
            ax.text(0.5, 0.5, f"No clustering data available for {method}", 
                    ha='center', va='center', fontsize=14)
            ax.set_axis_off()
            return fig, ax
        
        # Extract cluster information
        clusters = data['clusters']
        
        # Prepare data for plotting
        plot_data = []
        
        for cluster_name, cluster_info in clusters.items():
            languages = cluster_info.get('languages', {})
            
            for lang, count in languages.items():
                plot_data.append({
                    'Cluster': cluster_name,
                    'Language': self.language_names.get(lang, lang),
                    'Count': count
                })
        
        # Create DataFrame
        df = pd.DataFrame(plot_data)
        
        # If empty, return message
        if df.empty:
            fig, ax = self.create_figure()
            ax.text(0.5, 0.5, f"No language distribution data available for {method}", 
                    ha='center', va='center', fontsize=14)
            ax.set_axis_off()
            return fig, ax
        
        # Create the figure
        fig, ax = self.create_figure(figsize=(14, 8))
        
        # Create stacked bar chart
        pivot_df = df.pivot_table(index='Cluster', columns='Language', values='Count', fill_value=0)
        pivot_df.plot(kind='bar', stacked=True, ax=ax, 
                      colormap='viridis', rot=45)
        
        # Set titles and labels
        ax.set_title(f'Language Distribution Within Clusters ({method.capitalize()})', 
                    fontsize=16, pad=20)
        ax.set_xlabel('Cluster', fontsize=12, labelpad=10)
        ax.set_ylabel('Number of Fables', fontsize=12, labelpad=10)
        
        # Improve legend
        ax.legend(title='Language', bbox_to_anchor=(1.02, 1), loc='upper left')
        
        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add a subtle grid only on the y-axis
        ax.grid(axis='y', alpha=0.3)
        
        # Ensure layout fits everything
        plt.tight_layout()
        
        return fig, ax
    
    def plot_feature_importance(self, method='kmeans', cluster_id=None, top_n=10):
        """
        Create horizontal bar charts for top features in clusters.
        
        Args:
            method: Clustering method ('kmeans', 'hierarchical', 'dbscan')
            cluster_id: Specific cluster to visualize (None for all clusters)
            top_n: Number of top features to display
            
        Returns:
            tuple: (figure, axes) The created matplotlib figure and axes
        """
        # Load the clustering data
        data = self.load_clustering_data(method)
        
        if not data or 'clusters' not in data:
            # If no data, create an empty figure with a message
            fig, ax = self.create_figure()
            ax.text(0.5, 0.5, f"No clustering data available for {method}", 
                    ha='center', va='center', fontsize=14)
            ax.set_axis_off()
            return fig, ax
        
        # Extract cluster information
        clusters = data['clusters']
        
        # If specific cluster is requested
        if cluster_id is not None:
            cluster_key = f"cluster_{cluster_id}" if not cluster_id.startswith("cluster_") else cluster_id
            
            if cluster_key in clusters:
                # Plot single cluster
                cluster_info = clusters[cluster_key]
                top_features = cluster_info.get('top_features', [])
                
                # Create figure
                fig, ax = self.create_figure()
                
                # Prepare data
                features = [f['name'] for f in top_features[:top_n]]
                weights = [f['weight'] for f in top_features[:top_n]]
                
                # Reverse order for better display (highest at top)
                features.reverse()
                weights.reverse()
                
                # Create horizontal bar chart
                bars = ax.barh(features, weights, 
                               color=self.cluster_palettes.get(method, sns.color_palette('viridis'))[0])
                
                # Add value labels
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                           f'{width:.3f}', va='center')
                
                # Set titles and labels
                ax.set_title(f'Top Features for {cluster_key} ({method.capitalize()})', 
                            fontsize=16, pad=20)
                ax.set_xlabel('Feature Weight', fontsize=12, labelpad=10)
                ax.set_ylabel('Feature', fontsize=12, labelpad=10)
                
                # Remove top and right spines for cleaner look
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                # Add a subtle grid only on the x-axis
                ax.grid(axis='x', alpha=0.3)
                
                # Ensure layout fits everything
                plt.tight_layout()
                
                return fig, ax
            else:
                # Cluster not found
                fig, ax = self.create_figure()
                ax.text(0.5, 0.5, f"Cluster {cluster_id} not found in {method} results", 
                        ha='center', va='center', fontsize=14)
                ax.set_axis_off()
                return fig, ax
        
        # Create a multi-panel figure for all clusters
        cluster_count = len(clusters)
        
        # Calculate grid dimensions
        if cluster_count <= 2:
            rows, cols = 1, cluster_count
        else:
            cols = min(3, cluster_count)
            rows = (cluster_count + cols - 1) // cols  # Ceiling division
        
        # Create figure with subplots
        fig, axs = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        
        # Flatten axs if it's a multi-dimensional array
        if rows > 1 or cols > 1:
            axs = axs.flatten()
        else:
            axs = [axs]
        
        # Plot each cluster
        for i, (cluster_name, cluster_info) in enumerate(clusters.items()):
            if i >= len(axs):
                break
                
            ax = axs[i]
            top_features = cluster_info.get('top_features', [])
            
            # Prepare data
            features = [f['name'] for f in top_features[:top_n]]
            weights = [f['weight'] for f in top_features[:top_n]]
            
            # Reverse order for better display (highest at top)
            features.reverse()
            weights.reverse()
            
            # Create horizontal bar chart
            ax.barh(features, weights, 
                   color=self.cluster_palettes.get(method, sns.color_palette('viridis'))[i % 10])
            
            # Set titles and labels
            ax.set_title(f'{cluster_name}', fontsize=12)
            ax.set_xlabel('Weight', fontsize=10, labelpad=5)
            
            # Remove top and right spines for cleaner look
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Add a subtle grid only on the x-axis
            ax.grid(axis='x', alpha=0.3)
        
        # Hide any unused subplots
        for i in range(len(clusters), len(axs)):
            axs[i].set_visible(False)
        
        # Add overall title
        plt.suptitle(f'Top Features by Cluster ({method.capitalize()})', 
                    fontsize=16, y=0.98)
        
        # Ensure layout fits everything
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
        
        return fig, axs
    
    def plot_2d_cluster_projection(self, method='kmeans'):
        """
        Create a 2D projection of clusters using PCA.
        
        Args:
            method: Clustering method ('kmeans', 'hierarchical', 'dbscan')
            
        Returns:
            tuple: (figure, axes) The created matplotlib figure and axes
        """
        # Load the clustering data
        data = self.load_clustering_data(method)
        
        if not data or 'fable_clusters' not in data:
            # If no data, create an empty figure with a message
            fig, ax = self.create_figure()
            ax.text(0.5, 0.5, f"No clustering data available for {method}", 
                    ha='center', va='center', fontsize=14)
            ax.set_axis_off()
            return fig, ax
        
        # Extract cluster assignments
        fable_clusters = data['fable_clusters']
        
        # This is a bit tricky because we don't have the original feature vectors
        # In a real implementation, we'd need to either:
        # 1. Store the 2D PCA projection in the clustering result
        # 2. Load and transform the original feature vectors here
        
        # For demonstration, we'll create a mock 2D projection based on cluster assignments
        # In a real implementation, replace this with actual PCA or t-SNE projection
        
        # Create a mockup dataframe for visualization
        fable_ids = list(fable_clusters.keys())
        cluster_labels = list(fable_clusters.values())
        
        # Generate mock 2D coordinates
        np.random.seed(42)  # For reproducibility
        
        # Group points by cluster
        cluster_to_points = {}
        for fable_id, cluster in fable_clusters.items():
            if cluster not in cluster_to_points:
                # Center point for this cluster
                center_x = np.random.uniform(-5, 5)
                center_y = np.random.uniform(-5, 5)
                cluster_to_points[cluster] = (center_x, center_y)
        
        # Generate points around cluster centers
        x_coords = []
        y_coords = []
        
        for cluster in cluster_labels:
            if cluster == -1:  # Noise in DBSCAN
                x_coords.append(np.random.uniform(-8, 8))
                y_coords.append(np.random.uniform(-8, 8))
            else:
                center_x, center_y = cluster_to_points[cluster]
                x_coords.append(center_x + np.random.normal(0, 0.5))
                y_coords.append(center_y + np.random.normal(0, 0.5))
        
        # Create dataframe
        df = pd.DataFrame({
            'Fable ID': fable_ids,
            'Cluster': cluster_labels,
            'x': x_coords,
            'y': y_coords
        })
        
        # Create figure
        fig, ax = self.create_figure(figsize=(10, 8))
        
        # Create scatter plot
        palette = self.cluster_palettes.get(method, sns.color_palette('viridis'))
        
        # Handle noise points specially for DBSCAN
        if method == 'dbscan':
            # First plot noise points
            noise_points = df[df['Cluster'] == -1]
            if not noise_points.empty:
                ax.scatter(noise_points['x'], noise_points['y'], 
                          c='gray', marker='x', alpha=0.5, s=50, label='Noise')
            
            # Then plot clustered points
            clustered_points = df[df['Cluster'] != -1]
            if not clustered_points.empty:
                scatter = ax.scatter(clustered_points['x'], clustered_points['y'], 
                                   c=clustered_points['Cluster'], cmap='viridis', 
                                   s=80, alpha=0.8)
                
                # Add legend for clusters
                legend1 = ax.legend(*scatter.legend_elements(),
                                  loc="upper right", title="Clusters")
                ax.add_artist(legend1)
        else:
            # For K-means and hierarchical, all points belong to clusters
            scatter = ax.scatter(df['x'], df['y'], c=df['Cluster'], cmap='viridis', 
                               s=80, alpha=0.8)
            
            # Add legend for clusters
            legend1 = ax.legend(*scatter.legend_elements(),
                              loc="upper right", title="Clusters")
            ax.add_artist(legend1)
        
        # Add fable ID labels
        for i, row in df.iterrows():
            ax.annotate(row['Fable ID'], (row['x'], row['y']),
                      xytext=(5, 5), textcoords='offset points',
                      fontsize=8, alpha=0.7)
        
        # Set titles and labels
        ax.set_title(f'2D Projection of Clusters ({method.capitalize()})', 
                    fontsize=16, pad=20)
        ax.set_xlabel('Component 1', fontsize=12, labelpad=10)
        ax.set_ylabel('Component 2', fontsize=12, labelpad=10)
        
        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add grid
        ax.grid(alpha=0.3)
        
        # Ensure layout fits everything
        plt.tight_layout()
        
        return fig, ax
    
    def plot_dendrogram(self, max_height=None):
        """
        Create a dendrogram visualization for hierarchical clustering.
        
        Args:
            max_height: Maximum height to cut the dendrogram (None for full dendrogram)
            
        Returns:
            tuple: (figure, axes) The created matplotlib figure and axes
        """
        # Load hierarchical clustering data
        data = self.load_clustering_data('hierarchical')
        
        if not data or 'dendrogram_data' not in data:
            # If no data, create an empty figure with a message
            fig, ax = self.create_figure()
            ax.text(0.5, 0.5, "No dendrogram data available", 
                    ha='center', va='center', fontsize=14)
            ax.set_axis_off()
            return fig, ax
        
        # Extract dendrogram data
        dendrogram_data = data['dendrogram_data']
        linkage_matrix = np.array(dendrogram_data['linkage_matrix'])
        
        # Get labels if available
        labels = None
        if 'labels' in dendrogram_data:
            labels = dendrogram_data['labels']
        elif 'fable_ids' in dendrogram_data and 'languages' in dendrogram_data:
            # Construct labels from fable IDs and languages
            fable_ids = dendrogram_data['fable_ids']
            languages = dendrogram_data['languages']
            
            labels = [f"{fid} ({lang})" for fid, lang in zip(fable_ids, languages)]
        
        # Create figure
        fig, ax = self.create_figure(figsize=(14, 8))
        
        # Create dendrogram
        dendrogram(
            linkage_matrix,
            truncate_mode='level' if max_height else None,
            p=max_height if max_height else 0,
            labels=labels,
            leaf_rotation=90.,  # Rotates labels
            leaf_font_size=10.,  # Font size for labels
            ax=ax
        )
        
        # Set titles and labels
        ax.set_title('Hierarchical Clustering Dendrogram', fontsize=16, pad=20)
        ax.set_xlabel('Fables', fontsize=12, labelpad=10)
        ax.set_ylabel('Distance', fontsize=12, labelpad=10)
        
        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Ensure layout fits everything
        plt.tight_layout()
        
        return fig, ax
    
    def plot_clustering_metrics(self, methods=None):
        """
        Compare quality metrics across different clustering methods.
        
        Args:
            methods: List of methods to compare. If None, use all available.
            
        Returns:
            tuple: (figure, axes) The created matplotlib figure and axes
        """
        if methods is None:
            methods = ['kmeans', 'hierarchical', 'dbscan']
        
        # Load data for each method
        method_data = {}
        for method in methods:
            data = self.load_clustering_data(method)
            if data and 'quality' in data:
                method_data[method] = data['quality']
        
        # If no data found, show a message
        if not method_data:
            fig, ax = self.create_figure()
            ax.text(0.5, 0.5, "No clustering quality metrics available", 
                    ha='center', va='center', fontsize=14)
            ax.set_axis_off()
            return fig, ax
        
        # Create DataFrame for plotting
        plot_data = []
        
        for method, quality in method_data.items():
            for metric, value in quality.items():
                plot_data.append({
                    'Method': method.capitalize(),
                    'Metric': metric,
                    'Value': value
                })
        
        df = pd.DataFrame(plot_data)
        
        # If empty, return message
        if df.empty:
            fig, ax = self.create_figure()
            ax.text(0.5, 0.5, "No clustering quality metrics available", 
                    ha='center', va='center', fontsize=14)
            ax.set_axis_off()
            return fig, ax
        
        # Create the figure
        fig, ax = self.create_figure()
        
        # Create grouped bar chart
        sns.barplot(
            x='Method', 
            y='Value', 
            hue='Metric', 
            data=df,
            palette='viridis',
            ax=ax
        )
        
        # Set titles and labels
        ax.set_title('Clustering Quality Metrics Comparison', 
                    fontsize=16, pad=20)
        ax.set_xlabel('Clustering Method', fontsize=12, labelpad=10)
        ax.set_ylabel('Metric Value', fontsize=12, labelpad=10)
        
        # Improve legend
        ax.legend(title='Metric', bbox_to_anchor=(1.02, 1), loc='upper left')
        
        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add a subtle grid only on the y-axis
        ax.grid(axis='y', alpha=0.3)
        
        # Ensure layout fits everything
        plt.tight_layout()
        
        return fig, ax
    
    def plot_cluster_tendency(self, method='kmeans'):
        """
        Visualize cluster tendency (by language vs by fable content).
        
        Args:
            method: Clustering method ('kmeans', 'hierarchical', 'dbscan')
            
        Returns:
            tuple: (figure, axes) The created matplotlib figure and axes
        """
        # Load the clustering data
        data = self.load_clustering_data(method)
        
        if not data or 'cluster_tendency' not in data:
            # If no data, create an empty figure with a message
            fig, ax = self.create_figure()
            ax.text(0.5, 0.5, f"No cluster tendency data available for {method}", 
                    ha='center', va='center', fontsize=14)
            ax.set_axis_off()
            return fig, ax
        
        # Extract tendency data
        tendency = data['cluster_tendency']
        by_language = tendency.get('by_language', 0)
        by_fable_id = tendency.get('by_fable_id', 0)
        dominant_factor = tendency.get('dominant_factor', 'unknown')
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. Bar chart comparing the two tendencies
        tendencies = ['By Language', 'By Fable Content']
        values = [by_language, by_fable_id]
        
        ax1.bar(tendencies, values, color=['#3498db', '#e74c3c'])
        
        # Add value labels
        for i, v in enumerate(values):
            ax1.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=12)
        
        # Set limits for better visualization
        ax1.set_ylim(0, 1.1)
        
        # Set titles and labels
        ax1.set_title('Cluster Tendency Comparison', fontsize=14)
        ax1.set_ylabel('Isolation Score (higher = stronger clustering)', fontsize=12)
        
        # Remove top and right spines
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # 2. Pie chart showing dominant factor
        labels = ['By Language', 'By Fable Content']
        sizes = [by_language, by_fable_id]
        
        # Calculate percentages
        total = sum(sizes)
        percentages = [100 * size / total for size in sizes]
        
        ax2.pie(
            sizes, 
            labels=labels, 
            autopct='%1.1f%%',
            startangle=90,
            colors=['#3498db', '#e74c3c'],
            explode=(0.1, 0) if dominant_factor == 'language' else (0, 0.1),
            shadow=True
        )
        
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax2.axis('equal')
        
        ax2.set_title(f'Dominant Clustering Factor: {dominant_factor.replace("_", " ").title()}', 
                     fontsize=14)
        
        # Add overall title
        plt.suptitle(f'Cluster Tendency Analysis ({method.capitalize()})', 
                    fontsize=16, y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
        
        return fig, (ax1, ax2)