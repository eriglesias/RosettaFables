# clustering.py
"""
Implements clustering techniques for analyzing patterns across fables.

This module provides:
- K-means clustering to group similar fables
- Hierarchical clustering for exploring relationships
- Feature extraction from various fable attributes
- Visualization utilities for cluster analysis
"""

import numpy as np
import json
from pathlib import Path
import logging
from collections import defaultdict
import string
import re

# For clustering
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# For visualization support (not for direct plotting)
import matplotlib.pyplot as plt

class ClusteringAnalyzer:
    """
    Performs clustering analysis on fables to identify patterns and relationships.
    """
    
    def __init__(self, analysis_dir):
        """
        Initialize the clustering analyzer.
        
        Args:
            analysis_dir: Directory for storing/retrieving analysis results
        """
        self.analysis_dir = Path(analysis_dir)
        self.logger = logging.getLogger(__name__)
    
    def _extract_features(self, fables, feature_type='tfidf', **kwargs):
        """
        Extract features from fables for clustering.
        
        Args:
            fables: List of fable dictionaries
            feature_type: Type of features to extract ('tfidf', 'pos', 'style', 'embeddings')
            **kwargs: Additional parameters for feature extraction
            
        Returns:
            Tuple of (feature_matrix, feature_names, fable_ids)
        """
        # Create references to keep track of fables
        fable_ids = []
        fable_languages = []
        
        if feature_type == 'tfidf':
            # Extract text from fables
            texts = []
            
            for fable in fables:
                fable_id = fable.get('fable_id', 'unknown')
                language = fable.get('language', 'unknown')
                
                # Extract text (prioritize body, fallback to sentences)
                text = fable.get('body', '')
                if not text and 'sentences' in fable:
                    sentences = fable.get('sentences', [])
                    text = ' '.join(sentence.get('text', '') for sentence in sentences)
                
                if not text:
                    continue  # Skip if no text available
                
                # Normalize text
                text = text.lower()
                
                # Add to corpus
                texts.append(text)
                fable_ids.append(fable_id)
                fable_languages.append(language)
            
            # Check if we have enough texts
            if len(texts) < 2:
                raise ValueError("Not enough texts for TF-IDF feature extraction")
            
            # Create TF-IDF matrix
            max_features = kwargs.get('max_features', 200)
            ngram_range = kwargs.get('ngram_range', (1, 2))
            
            tfidf_vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range
            )
            
            feature_matrix = tfidf_vectorizer.fit_transform(texts)
            feature_names = tfidf_vectorizer.get_feature_names_out()
            
        elif feature_type == 'pos':
            # Extract POS distribution as features
            pos_distributions = []
            all_pos_tags = set()
            
            # First pass: collect all POS tags
            for fable in fables:
                if 'pos_distribution' in fable:
                    # If the fable already has a POS distribution, use it
                    all_pos_tags.update(fable['pos_distribution'].keys())
                elif 'sentences' in fable:
                    # Otherwise, extract POS tags from sentences
                    for sentence in fable['sentences']:
                        if 'pos_tags' in sentence:
                            for pos_tag in sentence['pos_tags']:
                                # Handle different POS tag formats
                                if isinstance(pos_tag, tuple) and len(pos_tag) >= 2:
                                    pos = pos_tag[1]  # (token, POS) format
                                elif isinstance(pos_tag, dict) and 'pos' in pos_tag:
                                    pos = pos_tag['pos']  # {pos: "TAG"} format
                                else:
                                    pos = str(pos_tag)  # Direct POS value
                                
                                all_pos_tags.add(pos)
            
            # Second pass: create feature vectors
            for fable in fables:
                fable_id = fable.get('fable_id', 'unknown')
                language = fable.get('language', 'unknown')
                
                # Initialize with zeros
                pos_vector = {pos: 0 for pos in all_pos_tags}
                
                if 'pos_distribution' in fable:
                    # If the fable already has a POS distribution, use it
                    for pos, freq in fable['pos_distribution'].items():
                        if pos in pos_vector:
                            pos_vector[pos] = freq
                elif 'sentences' in fable:
                    # Otherwise, extract POS distribution from sentences
                    pos_counts = defaultdict(int)
                    total_count = 0
                    
                    for sentence in fable['sentences']:
                        if 'pos_tags' in sentence:
                            for pos_tag in sentence['pos_tags']:
                                # Handle different POS tag formats
                                if isinstance(pos_tag, tuple) and len(pos_tag) >= 2:
                                    pos = pos_tag[1]  # (token, POS) format
                                elif isinstance(pos_tag, dict) and 'pos' in pos_tag:
                                    pos = pos_tag['pos']  # {pos: "TAG"} format
                                else:
                                    pos = str(pos_tag)  # Direct POS value
                                
                                pos_counts[pos] += 1
                                total_count += 1
                    
                    # Convert to percentages
                    for pos, count in pos_counts.items():
                        if pos in pos_vector:
                            pos_vector[pos] = (count / total_count * 100) if total_count > 0 else 0
                
                # Add to distributions
                pos_distributions.append(list(pos_vector.values()))
                fable_ids.append(fable_id)
                fable_languages.append(language)
            
            # Convert to numpy array
            feature_matrix = np.array(pos_distributions)
            feature_names = list(all_pos_tags)
            
        elif feature_type == 'style':
            # Extract style metrics as features
            style_features = []
            style_metric_names = []
            
            # Define style metrics to extract
            metrics_to_extract = [
                'avg_sentence_length',
                'avg_word_length',
                'ttr',
                'hapax_ratio',
                'avg_dependency_depth'
            ]
            
            # First pass: determine available metrics
            for fable in fables:
                # Check which metrics are available
                if 'style_metrics' in fable:
                    for metric in fable['style_metrics']:
                        if metric not in style_metric_names:
                            style_metric_names.append(metric)
            
            # Use predefined metrics if none found
            if not style_metric_names:
                style_metric_names = metrics_to_extract
            
            # Second pass: create feature vectors
            for fable in fables:
                fable_id = fable.get('fable_id', 'unknown')
                language = fable.get('language', 'unknown')
                
                # Initialize with zeros
                style_vector = []
                
                for metric in style_metric_names:
                    # Try to get the metric value
                    value = 0.0
                    
                    if 'style_metrics' in fable and metric in fable['style_metrics']:
                        value = fable['style_metrics'][metric]
                    elif metric == 'avg_sentence_length' and 'sentences' in fable:
                        # Calculate average sentence length
                        total_words = 0
                        for sentence in fable['sentences']:
                            text = sentence.get('text', '')
                            words = text.split()
                            total_words += len(words)
                        value = total_words / len(fable['sentences']) if fable['sentences'] else 0
                    elif metric == 'avg_word_length' and ('body' in fable or 'sentences' in fable):
                        # Calculate average word length
                        text = fable.get('body', '')
                        if not text and 'sentences' in fable:
                            text = ' '.join(sentence.get('text', '') for sentence in fable['sentences'])
                        
                        words = text.split()
                        if words:
                            value = sum(len(word) for word in words) / len(words)
                    
                    style_vector.append(float(value))
                
                # Add to feature vectors
                style_features.append(style_vector)
                fable_ids.append(fable_id)
                fable_languages.append(language)
            
            # Convert to numpy array
            feature_matrix = np.array(style_features)
            feature_names = style_metric_names
            
        elif feature_type == 'topic':
            # Extract topic distributions as features
            if 'topic_distributions' not in kwargs:
                raise ValueError("Topic distributions must be provided for topic feature extraction")
            
            topic_distributions = kwargs['topic_distributions']
            topic_features = []
            
            for fable in fables:
                fable_id = fable.get('fable_id', 'unknown')
                language = fable.get('language', 'unknown')
                
                # Get topic distribution for this fable
                if fable_id in topic_distributions:
                    topic_vector = topic_distributions[fable_id]
                    topic_features.append(topic_vector)
                    fable_ids.append(fable_id)
                    fable_languages.append(language)
            
            # Convert to numpy array
            feature_matrix = np.array(topic_features)
            feature_names = [f"topic_{i}" for i in range(feature_matrix.shape[1])]
            
        else:
            raise ValueError(f"Unsupported feature type: {feature_type}")
        
        # Scale features if needed
        if kwargs.get('scale_features', True) and feature_type != 'tfidf':
            scaler = StandardScaler()
            feature_matrix = scaler.fit_transform(feature_matrix)
        
        # Reduce dimensionality if requested
        n_components = kwargs.get('n_components', None)
        if n_components and feature_matrix.shape[1] > n_components:
            if feature_type == 'tfidf' or feature_matrix.shape[0] < feature_matrix.shape[1]:
                # Use TruncatedSVD for sparse matrices or when n_samples < n_features
                svd = TruncatedSVD(n_components=n_components)
                feature_matrix = svd.fit_transform(feature_matrix)
                # Update feature names
                feature_names = [f"component_{i}" for i in range(n_components)]
            else:
                # Use PCA for dense matrices
                pca = PCA(n_components=n_components)
                feature_matrix = pca.fit_transform(feature_matrix)
                # Update feature names
                feature_names = [f"pc_{i}" for i in range(n_components)]
        
        return feature_matrix, feature_names, fable_ids, fable_languages
    
    def kmeans_clustering(self, fables, n_clusters=3, feature_type='tfidf', **kwargs):
        """
        Perform K-means clustering on fables.
        
        Args:
            fables: List of fable dictionaries
            n_clusters: Number of clusters to create
            feature_type: Type of features to use ('tfidf', 'pos', 'style', 'topic')
            **kwargs: Additional parameters for feature extraction and clustering
            
        Returns:
            Dict with clustering results
        """
        # Extract features
        features, feature_names, fable_ids, fable_languages = self._extract_features(
            fables, feature_type, **kwargs
        )
        
        # Check if we have enough data
        if len(fable_ids) < n_clusters:
            return {
                'error': f"Not enough fables ({len(fable_ids)}) for {n_clusters} clusters"
            }
        
        # Perform K-means clustering
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=kwargs.get('random_state', 42),
            n_init=kwargs.get('n_init', 10)
        )
        
        # Get cluster labels
        cluster_labels = kmeans.fit_predict(features)
        
        # Evaluate clustering quality
        silhouette = silhouette_score(features, cluster_labels) if len(set(cluster_labels)) > 1 else 0
        calinski = calinski_harabasz_score(features, cluster_labels) if len(set(cluster_labels)) > 1 else 0
        
        # Get cluster centers
        cluster_centers = kmeans.cluster_centers_
        
        # Create cluster information
        clusters = {}
        for i in range(n_clusters):
            # Get fables in this cluster
            cluster_indices = np.where(cluster_labels == i)[0]
            cluster_fable_ids = [fable_ids[idx] for idx in cluster_indices]
            cluster_languages = [fable_languages[idx] for idx in cluster_indices]
            
            # Get the most important features for this cluster
            center = cluster_centers[i]
            
            # For sparse TF-IDF features, handle differently
            if feature_type == 'tfidf' and hasattr(features, 'toarray'):
                # Get average of TF-IDF values for this cluster
                cluster_features = features[cluster_indices].toarray()
                avg_features = np.mean(cluster_features, axis=0)
                
                # Get top features
                top_indices = avg_features.argsort()[::-1][:10]
                top_features = [
                    {
                        'name': feature_names[idx],
                        'weight': float(avg_features[idx])
                    }
                    for idx in top_indices if avg_features[idx] > 0
                ]
            else:
                # For other feature types
                top_indices = center.argsort()[::-1][:10]
                top_features = [
                    {
                        'name': feature_names[idx],
                        'weight': float(center[idx])
                    }
                    for idx in top_indices if center[idx] > 0
                ]
            
            # Count language distribution
            language_counts = defaultdict(int)
            for lang in cluster_languages:
                language_counts[lang] += 1
            
            clusters[f"cluster_{i}"] = {
                'size': len(cluster_fable_ids),
                'fable_ids': cluster_fable_ids,
                'languages': dict(language_counts),
                'top_features': top_features
            }
        
        # Combine results
        results = {
            'method': 'kmeans',
            'n_clusters': n_clusters,
            'feature_type': feature_type,
            'feature_count': len(feature_names),
            'fable_count': len(fable_ids),
            'quality': {
                'silhouette_score': silhouette,
                'calinski_harabasz_score': calinski
            },
            'clusters': clusters,
            'fable_clusters': {
                fable_id: int(cluster_labels[i])
                for i, fable_id in enumerate(fable_ids)
            }
        }
        
        return results
    
    def hierarchical_clustering(self, fables, n_clusters=None, feature_type='tfidf', **kwargs):
        """
        Perform hierarchical clustering on fables.
        
        Args:
            fables: List of fable dictionaries
            n_clusters: Number of clusters to cut the dendrogram (None for automatic determination)
            feature_type: Type of features to use ('tfidf', 'pos', 'style', 'topic')
            **kwargs: Additional parameters for feature extraction and clustering
            
        Returns:
            Dict with clustering results
        """
        # Extract features
        features, feature_names, fable_ids, fable_languages = self._extract_features(
            fables, feature_type, **kwargs
        )
        
        # Check if we have enough data
        if len(fable_ids) < 2:
            return {
                'error': f"Not enough fables ({len(fable_ids)}) for hierarchical clustering"
            }
        
        # Convert sparse matrix to dense if needed
        if hasattr(features, 'toarray'):
            features = features.toarray()
        
        # Compute linkage matrix
        linkage_method = kwargs.get('linkage_method', 'ward')
        linkage_matrix = linkage(features, method=linkage_method)
        
        # Determine optimal number of clusters if not specified
        if n_clusters is None:
            # Use the elbow method with inertia
            max_clusters = min(10, len(fable_ids) - 1)
            if max_clusters <= 1:
                max_clusters = 2
                
            inertias = []
            for k in range(1, max_clusters + 1):
                # Get cluster assignments
                cluster_labels = fcluster(linkage_matrix, k, criterion='maxclust')
                
                # Calculate inertia (sum of distances to cluster center)
                inertia = 0
                for cluster_id in range(1, k + 1):  # fcluster uses 1-indexed labels
                    cluster_indices = np.where(cluster_labels == cluster_id)[0]
                    if len(cluster_indices) > 0:
                        cluster_points = features[cluster_indices]
                        centroid = np.mean(cluster_points, axis=0)
                        inertia += np.sum((cluster_points - centroid) ** 2)
                
                inertias.append(inertia)
            
            # Find the elbow point (where inertia decreases more slowly)
            n_clusters = 2  # Default to 2 clusters
            if len(inertias) > 2:
                # Calculate the second derivative to find the elbow point
                second_derivative = np.diff(np.diff(inertias))
                elbow_index = np.argmax(second_derivative) + 2
                n_clusters = elbow_index + 1  # +1 because we started from 1
            
            # Ensure n_clusters is within bounds
            n_clusters = max(2, min(n_clusters, max_clusters))
        
        # Get cluster labels
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        
        # Convert to 0-indexed labels to match K-means
        cluster_labels = cluster_labels - 1
        
        # Evaluate clustering quality
        silhouette = silhouette_score(features, cluster_labels) if len(set(cluster_labels)) > 1 else 0
        calinski = calinski_harabasz_score(features, cluster_labels) if len(set(cluster_labels)) > 1 else 0
        
        # Create cluster information
        clusters = {}
        for i in range(n_clusters):
            # Get fables in this cluster
            cluster_indices = np.where(cluster_labels == i)[0]
            cluster_fable_ids = [fable_ids[idx] for idx in cluster_indices]
            cluster_languages = [fable_languages[idx] for idx in cluster_indices]
            
            # Get the most important features for this cluster
            cluster_features = features[cluster_indices]
            cluster_center = np.mean(cluster_features, axis=0)
            
            # Get top features
            top_indices = cluster_center.argsort()[::-1][:10]
            top_features = [
                {
                    'name': feature_names[idx],
                    'weight': float(cluster_center[idx])
                }
                for idx in top_indices if cluster_center[idx] > 0
            ]
            
            # Count language distribution
            language_counts = defaultdict(int)
            for lang in cluster_languages:
                language_counts[lang] += 1
            
            clusters[f"cluster_{i}"] = {
                'size': len(cluster_fable_ids),
                'fable_ids': cluster_fable_ids,
                'languages': dict(language_counts),
                'top_features': top_features
            }
        
        # Prepare dendrogram data
        # This will be a simplified representation that can be used to recreate the dendrogram
        dendrogram_data = {
            'linkage_matrix': linkage_matrix.tolist(),
            'fable_ids': fable_ids,
            'languages': fable_languages
        }
        
        # Combine results
        results = {
            'method': 'hierarchical',
            'linkage_method': linkage_method,
            'n_clusters': n_clusters,
            'feature_type': feature_type,
            'feature_count': len(feature_names),
            'fable_count': len(fable_ids),
            'quality': {
                'silhouette_score': silhouette,
                'calinski_harabasz_score': calinski
            },
            'clusters': clusters,
            'fable_clusters': {
                fable_id: int(cluster_labels[i])
                for i, fable_id in enumerate(fable_ids)
            },
            'dendrogram_data': dendrogram_data
        }
        
        return results
    
    def dbscan_clustering(self, fables, eps=0.5, min_samples=2, feature_type='tfidf', **kwargs):
        """
        Perform DBSCAN clustering on fables to identify clusters of varying density.
        
        Args:
            fables: List of fable dictionaries
            eps: Maximum distance between samples for neighborhood
            min_samples: Minimum number of samples in a neighborhood
            feature_type: Type of features to use ('tfidf', 'pos', 'style', 'topic')
            **kwargs: Additional parameters for feature extraction and clustering
            
        Returns:
            Dict with clustering results
        """
        # Extract features
        features, feature_names, fable_ids, fable_languages = self._extract_features(
            fables, feature_type, **kwargs
        )
        
        # Check if we have enough data
        if len(fable_ids) < min_samples:
            return {
                'error': f"Not enough fables ({len(fable_ids)}) for DBSCAN clustering"
            }
        
        # Perform DBSCAN clustering
        dbscan = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric=kwargs.get('metric', 'euclidean')
        )
        
        # Get cluster labels
        cluster_labels = dbscan.fit_predict(features)
        
        # Count the number of clusters (excluding noise points which are labeled -1)
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        
        # Evaluate clustering quality if there are proper clusters
        quality = {}
        if n_clusters > 1:
            # Filter out noise points for evaluation
            non_noise_indices = np.where(cluster_labels != -1)[0]
            if len(non_noise_indices) > n_clusters:
                non_noise_features = features[non_noise_indices]
                non_noise_labels = cluster_labels[non_noise_indices]
                
                quality = {
                    'silhouette_score': silhouette_score(non_noise_features, non_noise_labels),
                    'calinski_harabasz_score': calinski_harabasz_score(non_noise_features, non_noise_labels)
                }
        
        # Create cluster information
        clusters = {}
        for i in range(-1, n_clusters):  # Include noise cluster (-1)
            # Get fables in this cluster
            cluster_indices = np.where(cluster_labels == i)[0]
            
            if len(cluster_indices) == 0:
                continue  # Skip empty clusters
                
            cluster_fable_ids = [fable_ids[idx] for idx in cluster_indices]
            cluster_languages = [fable_languages[idx] for idx in cluster_indices]
            
            # Get the most important features for this cluster
            cluster_features = features[cluster_indices]
            if hasattr(cluster_features, 'toarray'):
                cluster_features = cluster_features.toarray()
                
            cluster_center = np.mean(cluster_features, axis=0)
            
            # Get top features
            top_indices = cluster_center.argsort()[::-1][:10]
            top_features = [
                {
                    'name': feature_names[idx],
                    'weight': float(cluster_center[idx])
                }
                for idx in top_indices if cluster_center[idx] > 0
            ]
            
            # Count language distribution
            language_counts = defaultdict(int)
            for lang in cluster_languages:
                language_counts[lang] += 1
            
            # Handle noise cluster differently
            if i == -1:
                cluster_name = 'noise'
            else:
                cluster_name = f"cluster_{i}"
                
            clusters[cluster_name] = {
                'size': len(cluster_fable_ids),
                'fable_ids': cluster_fable_ids,
                'languages': dict(language_counts),
                'top_features': top_features
            }
        
        # Combine results
        results = {
            'method': 'dbscan',
            'eps': eps,
            'min_samples': min_samples,
            'n_clusters': n_clusters,
            'noise_points': int(np.sum(cluster_labels == -1)),
            'feature_type': feature_type,
            'feature_count': len(feature_names),
            'fable_count': len(fable_ids),
            'quality': quality,
            'clusters': clusters,
            'fable_clusters': {
                fable_id: int(cluster_labels[i])
                for i, fable_id in enumerate(fable_ids)
            }
        }
        
        return results
    
    def cross_language_clustering(self, fables_by_id, feature_type='tfidf', method='kmeans', **kwargs):
        """
        Cluster different language versions of the same fables.
        
        Args:
            fables_by_id: Dict mapping fable IDs to language-specific versions
            feature_type: Type of features to use
            method: Clustering method ('kmeans', 'hierarchical', 'dbscan')
            **kwargs: Additional parameters for clustering
            
        Returns:
            Dict with cross-language clustering results
        """
        # Flatten the fables for feature extraction and clustering
        all_fables = []
        for fable_id, lang_fables in fables_by_id.items():
            for lang, fable in lang_fables.items():
                # Ensure the fable has ID and language attributes
                if 'fable_id' not in fable:
                    fable['fable_id'] = fable_id
                if 'language' not in fable:
                    fable['language'] = lang
                
                all_fables.append(fable)
        
        # Skip if no fables
        if not all_fables:
            return {'error': 'No fables provided for cross-language clustering'}
        
        # Perform clustering with the selected method
        if method == 'hierarchical':
            clustering_results = self.hierarchical_clustering(all_fables, feature_type=feature_type, **kwargs)
        elif method == 'dbscan':
            clustering_results = self.dbscan_clustering(all_fables, feature_type=feature_type, **kwargs)
        else:
            # Default to K-means, using number of fables as a hint for number of clusters
            n_clusters = kwargs.get('n_clusters', len(fables_by_id))
            clustering_results = self.kmeans_clustering(all_fables, n_clusters=n_clusters, feature_type=feature_type, **kwargs)
        
        # Analyze clustering by language and by fable
        if 'fable_clusters' in clustering_results:
            fable_clusters = clustering_results['fable_clusters']
            
            # Count fables by language in each cluster
            language_cluster_counts = defaultdict(lambda: defaultdict(int))
            fable_id_cluster_counts = defaultdict(lambda: defaultdict(int))
            
            for fable in all_fables:
                fable_id = fable.get('fable_id')
                language = fable.get('language')
                
                if fable_id in fable_clusters:
                    cluster = fable_clusters[fable_id]
                    cluster_name = f"cluster_{cluster}"
                    
                    # Count by language
                    language_cluster_counts[language][cluster_name] += 1
                    
                    # Count by fable ID
                    fable_id_cluster_counts[fable_id][cluster_name] += 1
            
            # Add analysis to results
            clustering_results['language_distribution'] = {
                lang: dict(counts) for lang, counts in language_cluster_counts.items()
            }
            
            clustering_results['fable_id_distribution'] = {
                fable_id: dict(counts) for fable_id, counts in fable_id_cluster_counts.items()
            }
            
            # Check if languages or fables cluster together
            language_isolation = {}
            for lang, counts in language_cluster_counts.items():
                total = sum(counts.values())
                if total > 0:
                    # Calculate the percentage of the most common cluster
                    most_common = max(counts.values())
                    isolation = most_common / total
                    language_isolation[lang] = isolation
            
            fable_id_isolation = {}
            for fable_id, counts in fable_id_cluster_counts.items():
                total = sum(counts.values())
                if total > 0:
                    # Calculate the percentage of the most common cluster
                    most_common = max(counts.values())
                    isolation = most_common / total
                    fable_id_isolation[fable_id] = isolation
            
            # Add to results
            clustering_results['language_isolation'] = language_isolation
            clustering_results['fable_id_isolation'] = fable_id_isolation
            
            # Calculate overall tendency to cluster by language vs by fable
            avg_language_isolation = sum(language_isolation.values()) / len(language_isolation) if language_isolation else 0
            avg_fable_isolation = sum(fable_id_isolation.values()) / len(fable_id_isolation) if fable_id_isolation else 0
            
            clustering_results['cluster_tendency'] = {
                'by_language': avg_language_isolation,
                'by_fable_id': avg_fable_isolation,
                'dominant_factor': 'language' if avg_language_isolation > avg_fable_isolation else 'fable_content'
            }
        
        return clustering_results
    
    def generate_dendrogram_data(self, clustering_result):
        """
        Generate dendrogram data for visualization.
        
        Args:
            clustering_result: Result from hierarchical_clustering
            
        Returns:
            Dict with dendrogram data for plotting
        """
        if 'dendrogram_data' not in clustering_result:
            return {'error': 'No dendrogram data available'}
        
        dendrogram_data = clustering_result['dendrogram_data']
        
        # Prepare labels for the dendrogram
        labels = []
        for i, fable_id in enumerate(dendrogram_data['fable_ids']):
            language = dendrogram_data['languages'][i]
            labels.append(f"{fable_id}_{language}")
        
        # Convert linkage matrix
        linkage_matrix = np.array(dendrogram_data['linkage_matrix'])
        
        return {
            'linkage_matrix': linkage_matrix.tolist(),
            'labels': labels
        }
    
    def optimize_clusters(self, fables, feature_type='tfidf', max_clusters=10, **kwargs):
        """
        Find the optimal number of clusters for a set of fables.
        
        Args:
            fables: List of fable dictionaries
            feature_type: Type of features to use
            max_clusters: Maximum number of clusters to consider
            **kwargs: Additional parameters for feature extraction
            
        Returns:
            Dict with optimization results
        """
        # Extract features
        features, feature_names, fable_ids, fable_languages = self._extract_features(
            fables, feature_type, **kwargs
        )
        
        # Check if we have enough data
        if len(fable_ids) < 2:
            return {
                'error': f"Not enough fables ({len(fable_ids)}) for cluster optimization"
            }
        
        # Limit max_clusters to the number of fables
        max_clusters = min(max_clusters, len(fable_ids) - 1)
        if max_clusters < 2:
            max_clusters = 2
        
        # Calculate scores for different numbers of clusters
        scores = []
        for n_clusters in range(2, max_clusters + 1):
            # Perform K-means clustering
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=kwargs.get('random_state', 42),
                n_init=kwargs.get('n_init', 10)
            )
            
            # Get cluster labels
            cluster_labels = kmeans.fit_predict(features)
            
            # Calculate evaluation metrics
            silhouette = silhouette_score(features, cluster_labels)
            calinski = calinski_harabasz_score(features, cluster_labels)
            
            # Calculate inertia (sum of squared distances to centers)
            inertia = kmeans.inertia_
            
            scores.append({
                'n_clusters': n_clusters,
                'silhouette_score': silhouette,
                'calinski_harabasz_score': calinski,
                'inertia': inertia
            })
        
        # Find optimal number of clusters
        # Using the elbow method with inertia
        inertias = [score['inertia'] for score in scores]
        optimal_k = 2  # Default
        
        if len(inertias) > 2:
            # Calculate the second derivative of inertia
            second_derivative = np.diff(np.diff(inertias))
            elbow_index = np.argmax(second_derivative)
            optimal_k = scores[elbow_index + 1]['n_clusters']
        
        # Using silhouette score (higher is better)
        silhouette_scores = [score['silhouette_score'] for score in scores]
        best_silhouette_index = np.argmax(silhouette_scores)
        best_silhouette_k = scores[best_silhouette_index]['n_clusters']
        
        # Using Calinski-Harabasz score (higher is better)
        calinski_scores = [score['calinski_harabasz_score'] for score in scores]
        best_calinski_index = np.argmax(calinski_scores)
        best_calinski_k = scores[best_calinski_index]['n_clusters']
        
        # Combine results
        results = {
            'scores': scores,
            'optimal_clusters': {
                'elbow_method': optimal_k,
                'silhouette_score': best_silhouette_k,
                'calinski_harabasz_score': best_calinski_k
            },
            'recommendation': best_silhouette_k  # Silhouette is typically most intuitive
        }
        
        return results
    
    def save_analysis(self, fable_id, analysis_type, results):
        """
        Save clustering analysis results to file.
        
        Args:
            fable_id: ID of the analyzed fable
            analysis_type: Type of analysis (e.g., 'kmeans', 'hierarchical')
            results: Analysis results to save
        """
        # Create directory if it doesn't exist
        output_dir = self.analysis_dir / 'clustering'
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create filename
        filename = f"{fable_id}_{analysis_type}.json"
        output_path = output_dir / filename
        
        # Save to JSON file
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Saved {analysis_type} clustering for fable {fable_id} to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving analysis: {e}")