# stats_analyzer.py
"""
Performs statistical analysis and comparison of fables across different languages.

This module provides:
- Word frequency distributions and comparisons
- Statistical tests for cross-language analysis
- Metrics for comparing different language versions of the same fable
"""

import math
import json
from collections import Counter, defaultdict
import numpy as np
from pathlib import Path
import logging
from scipy import stats
import re
import string

class StatsAnalyzer:
    """
    Performs statistical analysis and comparisons of fables across different languages.
    """
    
    def __init__(self, analysis_dir):
        """
        Initialize the statistics analyzer.
        
        Args:
            analysis_dir: Directory for storing/retrieving analysis results
        """
        self.analysis_dir = Path(analysis_dir)
        self.logger = logging.getLogger(__name__)
        
    def word_frequency(self, fable, top_n=20, exclude_stopwords=True):
        """
        Calculate word frequency distribution for a fable.
        
        Args:
            fable: The fable data dictionary
            top_n: Number of most frequent words to return
            exclude_stopwords: Whether to exclude common stopwords
            
        Returns:
            Dict with word frequency statistics
        """
        # Extract text (prioritize body, fallback to sentences)
        full_text = fable.get('body', '')
        if not full_text and 'sentences' in fable:
            sentences = fable.get('sentences', [])
            full_text = ' '.join(sentence.get('text', '') for sentence in sentences)
        
        # Get language for stopwords
        language = fable.get('language', 'en')
        
        # Normalize text (lowercase, remove punctuation)
        translator = str.maketrans('', '', string.punctuation)
        normalized_text = full_text.lower().translate(translator)
        
        # Tokenize
        tokens = [token for token in normalized_text.split() if token]
        
        # Get stopwords for the language
        stopwords = self._get_stopwords(language)
        
        # Filter out stopwords if requested
        if exclude_stopwords:
            tokens = [token for token in tokens if token not in stopwords]
        
        # Count word frequencies
        word_counts = Counter(tokens)
        total_words = len(tokens)
        unique_words = len(word_counts)
        
        # Calculate frequency metrics
        relative_frequencies = {
            word: count / total_words for word, count in word_counts.items()
        }
        
        # Get top N most frequent words
        most_common = word_counts.most_common(top_n)
        
        # Calculate Zipf's law statistics
        zipf_values = []
        for i, (word, count) in enumerate(word_counts.most_common(), 1):
            zipf_values.append({
                'rank': i,
                'word': word,
                'count': count,
                'expected_zipf': total_words / i  # Theoretical Zipf value
            })
        
        # Calculate frequency distribution statistics
        hapax_legomena = sum(1 for word, count in word_counts.items() if count == 1)
        dis_legomena = sum(1 for word, count in word_counts.items() if count == 2)
        
        return {
            'total_word_count': total_words,
            'unique_word_count': unique_words,
            'top_words': [
                {'word': word, 'count': count, 'relative_freq': relative_frequencies[word]}
                for word, count in most_common
            ],
            'hapax_legomena_count': hapax_legomena,
            'hapax_legomena_ratio': hapax_legomena / unique_words if unique_words else 0,
            'dis_legomena_count': dis_legomena,
            'entropy': self._calculate_entropy(relative_frequencies.values()),
            'zipf_data': zipf_values[:top_n]  # Limit to top N for conciseness
        }
    
    def _get_stopwords(self, language):
        """
        Get stopwords for the specified language.
        
        Args:
            language: Language code
            
        Returns:
            Set of stopwords
        """
        # Common stopwords for different languages
        stopwords = {
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
        
        return stopwords.get(language, set())
    
    def _calculate_entropy(self, probabilities):
        """
        Calculate Shannon entropy of a probability distribution.
        
        Args:
            probabilities: Iterable of probability values
            
        Returns:
            Entropy value
        """
        return -sum(p * math.log2(p) if p > 0 else 0 for p in probabilities)
    
    def compare_word_usage(self, fables_by_language):
        """
        Compare word usage patterns across different language versions of a fable.
        
        Args:
            fables_by_language: Dict mapping language codes to fable data
            
        Returns:
            Dict with comparison results
        """
        results = {
            'languages': list(fables_by_language.keys()),
            'word_counts': {},
            'frequent_words': {},
            'hapax_comparison': {},
            'entropy_comparison': {},
            'correlation': {}
        }
        
        # Analyze each language
        word_freqs = {}
        for lang, fable in fables_by_language.items():
            # Get word frequency analysis
            analysis = self.word_frequency(fable)
            
            results['word_counts'][lang] = {
                'total': analysis['total_word_count'],
                'unique': analysis['unique_word_count'],
                'ratio': analysis['unique_word_count'] / analysis['total_word_count'] if analysis['total_word_count'] else 0
            }
            
            results['frequent_words'][lang] = analysis['top_words'][:10]  # Top 10 words
            
            results['hapax_comparison'][lang] = {
                'count': analysis['hapax_legomena_count'],
                'ratio': analysis['hapax_legomena_ratio']
            }
            
            results['entropy_comparison'][lang] = analysis['entropy']
            
            # Store word frequencies for correlation analysis
            word_freqs[lang] = {item['word']: item['relative_freq'] for item in analysis['top_words']}
        
        # Compute cosine similarity between languages (for languages with enough common words)
        for lang1 in results['languages']:
            for lang2 in results['languages']:
                if lang1 >= lang2:  # Only compute for unique pairs
                    continue
                
                similarity = self._calculate_cosine_similarity(word_freqs[lang1], word_freqs[lang2])
                
                if f"{lang1}_{lang2}" not in results['correlation']:
                    results['correlation'][f"{lang1}_{lang2}"] = {}
                
                results['correlation'][f"{lang1}_{lang2}"]["cosine_similarity"] = similarity
        
        return results
    
    def _calculate_cosine_similarity(self, freq1, freq2):
        """
        Calculate cosine similarity between two word frequency dictionaries.
        
        Args:
            freq1: First word frequency dictionary
            freq2: Second word frequency dictionary
            
        Returns:
            Cosine similarity value
        """
        # Get all unique words
        all_words = set(freq1.keys()) | set(freq2.keys())
        
        # Create vectors with zeroes for missing words
        vec1 = [freq1.get(word, 0) for word in all_words]
        vec2 = [freq2.get(word, 0) for word in all_words]
        
        # Calculate dot product
        dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
        
        # Calculate magnitudes
        mag1 = math.sqrt(sum(v1 * v1 for v1 in vec1))
        mag2 = math.sqrt(sum(v2 * v2 for v2 in vec2))
        
        # Calculate cosine similarity
        if mag1 > 0 and mag2 > 0:
            return dot_product / (mag1 * mag2)
        else:
            return 0
    
    def chi_square_test(self, fables_by_language, feature='pos'):
        """
        Perform chi-square test to compare distributions across languages.
        
        Args:
            fables_by_language: Dict mapping language codes to fable data
            feature: The feature to compare ('pos', 'entity_type', etc.)
            
        Returns:
            Dict with chi-square test results
        """
        results = {
            'languages': list(fables_by_language.keys()),
            'feature': feature,
            'contingency_table': {},
            'chi_square_results': {},
            'significant_differences': {}
        }
        
        # Extract feature distributions
        distributions = {}
        
        for lang, fable in fables_by_language.items():
            if feature == 'pos':
                # Extract POS counts
                pos_counts = defaultdict(int)
                total_count = 0
                
                # Get tokens and their POS tags
                if 'sentences' in fable:
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
                distributions[lang] = {
                    pos: (count / total_count * 100) if total_count > 0 else 0
                    for pos, count in pos_counts.items()
                }
            
            elif feature == 'entity_type':
                # Extract named entity counts
                entity_counts = defaultdict(int)
                total_count = 0
                
                if 'entities' in fable:
                    for entity in fable['entities']:
                        entity_type = entity.get('type', 'UNKNOWN')
                        entity_counts[entity_type] += 1
                        total_count += 1
                
                # Convert to percentages
                distributions[lang] = {
                    entity_type: (count / total_count * 100) if total_count > 0 else 0
                    for entity_type, count in entity_counts.items()
                }
            
            # Add more feature types here as needed
        
        # Create contingency table
        all_categories = set()
        for dist in distributions.values():
            all_categories.update(dist.keys())
        
        contingency_table = {}
        for lang, dist in distributions.items():
            row = {}
            for category in all_categories:
                row[category] = dist.get(category, 0)
            contingency_table[lang] = row
        
        results['contingency_table'] = contingency_table
        
        # Perform chi-square test for each pair of languages
        for i, lang1 in enumerate(results['languages']):
            for lang2 in results['languages'][i+1:]:
                # Get distributions for the two languages
                dist1 = contingency_table[lang1]
                dist2 = contingency_table[lang2]
                
                # Create observed frequencies arrays
                categories = sorted(set(dist1.keys()) | set(dist2.keys()))
                obs1 = [dist1.get(cat, 0) for cat in categories]
                obs2 = [dist2.get(cat, 0) for cat in categories]
                
                # Only perform test if we have enough non-zero values
                if sum(1 for v in obs1 if v > 0) >= 2 and sum(1 for v in obs2 if v > 0) >= 2:
                    # Convert to numpy arrays
                    observed = np.array([obs1, obs2])
                    
                    # Perform chi-square test
                    try:
                        chi2, p_value, dof, expected = stats.chi2_contingency(observed)
                        
                        results['chi_square_results'][f"{lang1}_{lang2}"] = {
                            'chi_square': chi2,
                            'p_value': p_value,
                            'degrees_of_freedom': dof,
                            'is_significant': p_value < 0.05
                        }
                        
                        # Find significant differences in categories
                        if p_value < 0.05:
                            significant_diffs = []
                            for j, cat in enumerate(categories):
                                # Check if category has a significant contribution
                                obs_value = observed[0][j]
                                exp_value = expected[0][j]
                                
                                if obs_value > 0 and exp_value > 0:
                                    # Calculate contribution to chi-square
                                    contribution = ((obs_value - exp_value) ** 2) / exp_value
                                    
                                    # If this category contributes significantly
                                    if contribution > 3.84:  # Critical value for df=1, p=0.05
                                        significant_diffs.append({
                                            'category': cat,
                                            'language1_value': dist1.get(cat, 0),
                                            'language2_value': dist2.get(cat, 0),
                                            'difference': dist1.get(cat, 0) - dist2.get(cat, 0),
                                            'contribution': contribution
                                        })
                            
                            significant_diffs.sort(key=lambda x: abs(x['difference']), reverse=True)
                            results['significant_differences'][f"{lang1}_{lang2}"] = significant_diffs
                    
                    except Exception as e:
                        results['chi_square_results'][f"{lang1}_{lang2}"] = {
                            'error': str(e)
                        }
        
        return results
    
    def pos_variation(self, fable):
        """
        Analyze POS tag variation and distribution in a fable.
        
        Args:
            fable: The fable data dictionary
            
        Returns:
            Dict with POS variation analysis
        """
        # Extract POS tags
        pos_counts = defaultdict(int)
        pos_sequences = []
        total_pos = 0
        
        # Iterate through sentences and collect POS information
        if 'sentences' in fable:
            for sentence in fable['sentences']:
                if 'pos_tags' in sentence:
                    sentence_pos = []
                    
                    for pos_tag in sentence['pos_tags']:
                        # Handle different POS tag formats
                        if isinstance(pos_tag, tuple) and len(pos_tag) >= 2:
                            pos = pos_tag[1]  # (token, POS) format
                        elif isinstance(pos_tag, dict) and 'pos' in pos_tag:
                            pos = pos_tag['pos']  # {pos: "TAG"} format
                        else:
                            pos = str(pos_tag)  # Direct POS value
                            
                        pos_counts[pos] += 1
                        sentence_pos.append(pos)
                        total_pos += 1
                    
                    pos_sequences.append(sentence_pos)
        
        # If no POS data found, return empty result
        if total_pos == 0:
            return {
                'error': 'No POS data found in the fable'
            }
        
        # Calculate POS distribution
        pos_distribution = {
            pos: (count / total_pos * 100)
            for pos, count in sorted(pos_counts.items(), key=lambda x: x[1], reverse=True)
        }
        
        # Calculate POS sequence patterns (bigrams)
        bigram_counts = defaultdict(int)
        for sequence in pos_sequences:
            for i in range(len(sequence) - 1):
                bigram = (sequence[i], sequence[i+1])
                bigram_counts[bigram] += 1
        
        # Get most common bigrams
        total_bigrams = sum(bigram_counts.values())
        common_bigrams = []
        
        for bigram, count in sorted(bigram_counts.items(), key=lambda x: x[1], reverse=True)[:15]:
            common_bigrams.append({
                'sequence': f"{bigram[0]}-{bigram[1]}",
                'count': count,
                'percentage': (count / total_bigrams * 100) if total_bigrams > 0 else 0
            })
        
        # Calculate POS diversity metrics
        pos_diversity = len(pos_counts) / total_pos if total_pos > 0 else 0
        pos_entropy = self._calculate_entropy([count/total_pos for count in pos_counts.values()])
        
        return {
            'total_pos_tags': total_pos,
            'unique_pos_tags': len(pos_counts),
            'pos_distribution': pos_distribution,
            'most_common_bigrams': common_bigrams,
            'pos_diversity': pos_diversity,
            'pos_entropy': pos_entropy
        }
    
    def compare_lexical_diversity(self, fables_by_language):
        """
        Compare lexical diversity metrics across language versions.
        
        Args:
            fables_by_language: Dict mapping language codes to fable data
            
        Returns:
            Dict with comparison results
        """
        results = {
            'languages': list(fables_by_language.keys()),
            'type_token_ratio': {},
            'hapax_ratio': {},
            'entropy': {},
            'word_lengths': {}
        }
        
        for lang, fable in fables_by_language.items():
            # Extract text
            full_text = fable.get('body', '')
            if not full_text and 'sentences' in fable:
                sentences = fable.get('sentences', [])
                full_text = ' '.join(sentence.get('text', '') for sentence in sentences)
            
            # Tokenize
            translator = str.maketrans('', '', string.punctuation)
            normalized_text = full_text.lower().translate(translator)
            tokens = [token for token in normalized_text.split() if token]
            
            # Calculate metrics
            token_count = len(tokens)
            type_count = len(set(tokens))
            ttr = type_count / token_count if token_count > 0 else 0
            
            # Word frequency
            word_counts = Counter(tokens)
            
            # Hapax legomena
            hapax_count = sum(1 for count in word_counts.values() if count == 1)
            hapax_ratio = hapax_count / token_count if token_count > 0 else 0
            
            # Entropy
            word_probs = [count / token_count for count in word_counts.values()]
            entropy = self._calculate_entropy(word_probs)
            
            # Word lengths
            word_lengths = [len(token) for token in tokens]
            avg_word_length = sum(word_lengths) / len(word_lengths) if word_lengths else 0
            
            # Store results
            results['type_token_ratio'][lang] = ttr
            results['hapax_ratio'][lang] = hapax_ratio
            results['entropy'][lang] = entropy
            results['word_lengths'][lang] = {
                'average': avg_word_length,
                'min': min(word_lengths) if word_lengths else 0,
                'max': max(word_lengths) if word_lengths else 0,
                'distribution': Counter(word_lengths)
            }
        
        return results
    
    def save_analysis(self, fable_id, analysis_type, results):
        """
        Save analysis results to file.
        
        Args:
            fable_id: ID of the analyzed fable
            analysis_type: Type of analysis (e.g., 'word_frequency', 'chi_square')
            results: Analysis results to save
        """
        # Create directory if it doesn't exist
        output_dir = self.analysis_dir / 'stats'
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