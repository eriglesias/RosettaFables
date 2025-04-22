# nlp_techniques.py
"""
Implements advanced NLP techniques for analyzing fables across languages.

This module provides:
- TF-IDF analysis to identify important terms
- Topic modeling using LDA and NMF
- Word embeddings analysis and visualization
"""

import numpy as np
import json
from pathlib import Path
import logging
from collections import Counter, defaultdict
import string
import re

# For TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# For topic modeling
from sklearn.decomposition import LatentDirichletAllocation, NMF

# For word embeddings (optional, can be disabled if you don't want to use gensim)
try:
    import gensim
    from gensim.models import Word2Vec, FastText
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False

class NLPTechniques:
    """
    Advanced NLP techniques for multilingual fable analysis.
    """
    
    def __init__(self, analysis_dir):
        """
        Initialize the NLP techniques analyzer.
        
        Args:
            analysis_dir: Directory for storing/retrieving analysis results
        """
        self.analysis_dir = Path(analysis_dir)
        self.logger = logging.getLogger(__name__)
        
        # Create stopwords for different languages
        self.stopwords = {
            'en': {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'of', 'for', 
                  'with', 'by', 'as', 'is', 'was', 'were', 'be', 'been', 'being', 'this', 'that'},
            'es': {'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas', 'y', 'o', 'pero', 'en',
                  'a', 'de', 'por', 'para', 'con', 'sin', 'sobre', 'entre', 'como', 'es', 'son'},
            'de': {'der', 'die', 'das', 'ein', 'eine', 'und', 'oder', 'aber', 'in', 'auf', 'bei',
                  'zu', 'von', 'für', 'mit', 'durch', 'als', 'ist', 'sind', 'war', 'waren'},
            'nl': {'de', 'het', 'een', 'en', 'of', 'maar', 'in', 'op', 'bij', 'tot', 'van', 'voor',
                  'met', 'door', 'als', 'is', 'zijn', 'was', 'waren', 'dit', 'dat'},
            'grc': {'ὁ', 'ἡ', 'τό', 'καί', 'ἤ', 'ἀλλά', 'ἐν', 'ἐπί', 'παρά', 'πρός', 'ἀπό', 'διά',
                   'ὡς', 'εἰ', 'μή', 'οὐ', 'οὐκ', 'οὐχ', 'γάρ', 'δέ', 'τε', 'μέν'}
        }
    
    def _preprocess_text(self, text, language='en', remove_stopwords=True, min_word_length=2):
        """
        Preprocess text for NLP analysis.
        
        Args:
            text: Text to preprocess
            language: Language code
            remove_stopwords: Whether to remove stopwords
            min_word_length: Minimum word length to keep
            
        Returns:
            List of preprocessed tokens
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        translator = str.maketrans('', '', string.punctuation)
        text = text.translate(translator)
        
        # Tokenize
        tokens = text.split()
        
        # Filter by length
        tokens = [token for token in tokens if len(token) >= min_word_length]
        
        # Remove stopwords if requested
        if remove_stopwords:
            lang_stopwords = self.stopwords.get(language, set())
            tokens = [token for token in tokens if token not in lang_stopwords]
        
        return tokens
    
    def _extract_fable_texts(self, fables, preprocessed=True):
        """
        Extract text from fables for analysis.
        
        Args:
            fables: List of fable dictionaries
            preprocessed: Whether to return preprocessed tokens or raw text
            
        Returns:
            Dict mapping fable IDs to their texts or tokens
        """
        fable_texts = {}
        
        for fable in fables:
            fable_id = fable.get('fable_id')
            language = fable.get('language', 'en')
            
            # Extract text (prioritize body, fallback to sentences)
            text = fable.get('body', '')
            if not text and 'sentences' in fable:
                sentences = fable.get('sentences', [])
                text = ' '.join(sentence.get('text', '') for sentence in sentences)
            
            if not text:
                continue  # Skip if no text available
            
            if preprocessed:
                tokens = self._preprocess_text(text, language)
                fable_texts[fable_id] = tokens
            else:
                fable_texts[fable_id] = text
        
        return fable_texts
    
    def _prepare_corpus(self, fables_by_language):
        """
        Prepare a corpus for TF-IDF and topic modeling.
        
        Args:
            fables_by_language: Dict mapping language codes to fable data
            
        Returns:
            Dict with corpus information
        """
        corpus = {
            'documents': [],  # List of preprocessed documents
            'document_ids': [],  # List of document identifiers
            'languages': [],  # List of document languages
            'raw_texts': []  # List of raw documents (for reference)
        }
        
        for lang, fables in fables_by_language.items():
            if isinstance(fables, list):
                # Handle list of fables
                for fable in fables:
                    text = fable.get('body', '')
                    if not text and 'sentences' in fable:
                        sentences = fable.get('sentences', [])
                        text = ' '.join(sentence.get('text', '') for sentence in sentences)
                    
                    if not text:
                        continue  # Skip if no text available
                    
                    corpus['raw_texts'].append(text)
                    corpus['languages'].append(lang)
                    corpus['document_ids'].append(fable.get('fable_id', 'unknown'))
                    
                    # Preprocess and add to documents
                    tokens = self._preprocess_text(text, lang)
                    corpus['documents'].append(' '.join(tokens))
            else:
                # Handle single fable
                text = fables.get('body', '')
                if not text and 'sentences' in fables:
                    sentences = fables.get('sentences', [])
                    text = ' '.join(sentence.get('text', '') for sentence in sentences)
                
                if not text:
                    continue  # Skip if no text available
                
                corpus['raw_texts'].append(text)
                corpus['languages'].append(lang)
                corpus['document_ids'].append(fables.get('fable_id', 'unknown'))
                
                # Preprocess and add to documents
                tokens = self._preprocess_text(text, lang)
                corpus['documents'].append(' '.join(tokens))
        
        return corpus
    
    def tfidf_analysis(self, fables_by_language, max_features=100):
        """
        Perform TF-IDF analysis on fables.
        
        Args:
            fables_by_language: Dict mapping language codes to fable data
            max_features: Maximum number of features for TF-IDF vectorizer
            
        Returns:
            Dict with TF-IDF analysis results
        """
        # Prepare corpus
        corpus = self._prepare_corpus(fables_by_language)
        
        if not corpus['documents']:
            return {'error': 'No documents to analyze'}
        
        # Initialize TF-IDF vectorizer
        tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2)  # Use both unigrams and bigrams
        )
        
        # Fit and transform documents
        tfidf_matrix = tfidf_vectorizer.fit_transform(corpus['documents'])
        
        # Get feature names
        feature_names = tfidf_vectorizer.get_feature_names_out()
        
        # Extract top terms for each document
        results = {
            'document_info': [],
            'top_terms_overall': [],
            'tfidf_matrix_shape': tfidf_matrix.shape
        }
        
        # Process each document
        for i, doc_id in enumerate(corpus['document_ids']):
            # Get TF-IDF scores for this document
            doc_tfidf = tfidf_matrix[i].toarray()[0]
            
            # Sort terms by TF-IDF score
            sorted_indices = doc_tfidf.argsort()[::-1]
            top_indices = sorted_indices[:10]  # Get top 10 terms
            
            # Extract top terms and scores
            top_terms = [
                {
                    'term': feature_names[idx],
                    'score': float(doc_tfidf[idx])
                }
                for idx in top_indices if doc_tfidf[idx] > 0
            ]
            
            # Add document info
            results['document_info'].append({
                'document_id': doc_id,
                'language': corpus['languages'][i],
                'top_terms': top_terms
            })
        
        # Find top terms across all documents
        tfidf_sum = np.sum(tfidf_matrix.toarray(), axis=0)
        sorted_indices = tfidf_sum.argsort()[::-1]
        top_indices = sorted_indices[:20]  # Get top 20 overall terms
        
        results['top_terms_overall'] = [
            {
                'term': feature_names[idx],
                'score': float(tfidf_sum[idx])
            }
            for idx in top_indices if tfidf_sum[idx] > 0
        ]
        
        return results
    
    def topic_modeling(self, fables_by_language, n_topics=5, method='lda'):
        """
        Perform topic modeling on fables.
        
        Args:
            fables_by_language: Dict mapping language codes to fable data
            n_topics: Number of topics to extract
            method: Topic modeling method ('lda' or 'nmf')
            
        Returns:
            Dict with topic modeling results
        """
        # Prepare corpus
        corpus = self._prepare_corpus(fables_by_language)
        
        if not corpus['documents']:
            return {'error': 'No documents to analyze'}
        
        # Initialize vectorizer
        vectorizer = CountVectorizer(
            max_features=1000,
            stop_words='english'  # Can be customized for other languages
        )
        
        # Fit and transform documents
        X = vectorizer.fit_transform(corpus['documents'])
        feature_names = vectorizer.get_feature_names_out()
        
        # Initialize topic model
        if method.lower() == 'nmf':
            model = NMF(
                n_components=n_topics,
                random_state=42,
                alpha=0.1,
                l1_ratio=0.5
            )
            model_name = 'NMF'
        else:
            model = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=10,
                learning_method='online'
            )
            model_name = 'LDA'
        
        # Fit model
        model.fit(X)
        
        # Process topics
        results = {
            'model_type': model_name,
            'n_topics': n_topics,
            'topics': [],
            'document_topics': []
        }
        
        # Get top terms for each topic
        for topic_idx, topic in enumerate(model.components_):
            top_indices = topic.argsort()[:-11:-1]  # Get top 10 terms
            top_terms = [
                {
                    'term': feature_names[i],
                    'weight': float(topic[i])
                }
                for i in top_indices
            ]
            
            results['topics'].append({
                'topic_id': topic_idx,
                'top_terms': top_terms
            })
        
        # Get topic distribution for each document
        doc_topic_matrix = model.transform(X)
        
        for i, doc_id in enumerate(corpus['document_ids']):
            # Get topic distribution for this document
            topic_distribution = doc_topic_matrix[i]
            
            # Sort by weight
            sorted_topics = [
                {
                    'topic_id': topic_idx,
                    'weight': float(weight)
                }
                for topic_idx, weight in enumerate(topic_distribution)
            ]
            sorted_topics.sort(key=lambda x: x['weight'], reverse=True)
            
            # Add document info
            results['document_topics'].append({
                'document_id': doc_id,
                'language': corpus['languages'][i],
                'dominant_topic': sorted_topics[0]['topic_id'],
                'topic_distribution': sorted_topics
            })
        
        return results
    
    def word_embeddings(self, fables_by_language, embedding_size=100, window=5, min_count=1, model_type='word2vec'):
        """
        Generate and analyze word embeddings for fables.
        
        Args:
            fables_by_language: Dict mapping language codes to fable data
            embedding_size: Size of embedding vectors
            window: Context window size
            min_count: Minimum word count
            model_type: Type of embedding model ('word2vec' or 'fasttext')
            
        Returns:
            Dict with word embedding analysis results
        """
        if not GENSIM_AVAILABLE:
            return {
                'error': 'Gensim library not available. Install with "pip install gensim" to use word embeddings.'
            }
        
        # Extract and preprocess fable texts by language
        language_tokens = {}
        
        for lang, fables in fables_by_language.items():
            # Convert to list if it's a single fable
            if not isinstance(fables, list):
                fables = [fables]
            
            all_tokens = []
            
            for fable in fables:
                # Extract text
                text = fable.get('body', '')
                if not text and 'sentences' in fable:
                    sentences = fable.get('sentences', [])
                    text = ' '.join(sentence.get('text', '') for sentence in sentences)
                
                if not text:
                    continue  # Skip if no text available
                
                # Preprocess and add to tokens
                tokens = self._preprocess_text(text, lang, remove_stopwords=False)
                if tokens:
                    all_tokens.append(tokens)
            
            if all_tokens:
                language_tokens[lang] = all_tokens
        
        # Skip if no tokens
        if not language_tokens:
            return {'error': 'No texts to analyze'}
        
        results = {
            'language_models': {},
            'model_type': model_type,
            'embedding_size': embedding_size,
            'similar_words': {}
        }
        
        # Train models for each language
        for lang, token_lists in language_tokens.items():
            # Create and train model
            if model_type.lower() == 'fasttext':
                model = FastText(
                    sentences=token_lists,
                    vector_size=embedding_size,
                    window=window,
                    min_count=min_count,
                    workers=4,
                    sg=1  # Skip-gram
                )
            else:
                model = Word2Vec(
                    sentences=token_lists,
                    vector_size=embedding_size,
                    window=window,
                    min_count=min_count,
                    workers=4,
                    sg=1  # Skip-gram
                )
            
            # Extract key information
            vocabulary_size = len(model.wv.index_to_key)
            
            # Find most frequent words
            word_counts = Counter([word for tokens in token_lists for word in tokens])
            most_common = word_counts.most_common(10)
            
            # Find similar words for common terms
            similar_words = {}
            for word, _ in most_common[:5]:  # Check top 5 common words
                if word in model.wv:
                    similar = model.wv.most_similar(word, topn=5)
                    similar_words[word] = [
                        {'word': similar_word, 'similarity': float(similarity)}
                        for similar_word, similarity in similar
                    ]
            
            # Store model info
            results['language_models'][lang] = {
                'vocabulary_size': vocabulary_size,
                'most_common_words': [
                    {'word': word, 'count': count}
                    for word, count in most_common
                ]
            }
            
            results['similar_words'][lang] = similar_words
        
        return results
    
    def save_analysis(self, fable_id, analysis_type, results):
        """
        Save analysis results to file.
        
        Args:
            fable_id: ID of the analyzed fable
            analysis_type: Type of analysis (e.g., 'tfidf', 'topic_modeling')
            results: Analysis results to save
        """
        # Create directory if it doesn't exist
        output_dir = self.analysis_dir / 'nlp'
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create filename
        filename = f"{fable_id}_{analysis_type}.json"
        output_path = output_dir / filename
        
        # Save to JSON file
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Saved {analysis_type} analysis for fable {fable_id} to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving analysis: {e}")