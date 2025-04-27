"""
Provides cross-language comparison for fables across different dimensions.

This module enables:
- Structural comparisons (length, sentences, etc.)
- Content comparisons (themes, entities, etc.)
- Linguistic comparisons (syntax, word usage, etc.)
- Visualizable difference data
"""

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List, Tuple

class ComparisonAnalyzer:
    """Analyzes and compares fables across different languages."""
    
    def __init__(self, analysis_dir: Path):
        """
        Initialize the comparison analyzer with the analysis directory.
        
        Args:
            analysis_dir: Directory for storing/retrieving analysis results
        """
        self.analysis_dir = analysis_dir
        self.logger = logging.getLogger(__name__)
        
    
    def find_common_fable_ids(self, fables_by_language: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        """
        Find fable IDs that appear in multiple languages.
        
        Args:
            fables_by_language: Dictionary mapping language codes to fable lists
            
        Returns:
            List of fable IDs that appear in multiple languages
        """
        # First, build a mapping of fable IDs to the languages they appear in
        fable_id_languages = {}
        
        for lang, fables in fables_by_language.items():
            for fable in fables:
                if isinstance(fable, dict) and 'id' in fable:
                    fable_id = fable['id']
                    if fable_id not in fable_id_languages:
                        fable_id_languages[fable_id] = []
                    fable_id_languages[fable_id].append(lang)
        
        # Find IDs that appear in at least two languages
        common_ids = [
            fable_id for fable_id, langs in fable_id_languages.items()
            if len(langs) >= 2
        ]
        
        self.logger.info("Found %d fables with content in multiple languages", len(common_ids))
        return common_ids
    
    def compare_fable(self, fable_id: str, fables_by_id: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Compare the same fable across different languages.
        
        Args:
            fable_id: ID of the fable to compare
            fables_by_id: Dictionary mapping fable IDs to language-specific fable data
            
        Returns:
            Comparison data dictionary
        """
        if fable_id not in fables_by_id:
            self.logger.warning("Fable ID %s not found in the dataset", fable_id)
            return None
        
        lang_fables = fables_by_id[fable_id]
        if not isinstance(lang_fables, dict):
            self.logger.error("Expected dictionary for fable %s but got %s", 
                             fable_id, type(lang_fables).__name__)
            return None
        
        result = {
            'fable_id': fable_id,
            'languages': list(lang_fables.keys()),
            'token_counts': {},  # Initialize token counts dictionary
            'common_entities': [],
            'sentiment_scores': {},
            'complexity_scores': {}
        }
        
        # Calculate token counts for each language
        for lang, fable in lang_fables.items():
            # Get content for tokenization
            content = fable.get('content', '')
            if not content and 'text' in fable:  # Try alternative field name
                content = fable.get('text', '')
            
            # Calculate token count - using a simple whitespace split as fallback
            tokens = []
            token_count = 0
            
            # First try to get tokens from the 'tokens' field
            if 'tokens' in fable:
                tokens = fable['tokens']
                if isinstance(tokens, list):
                    # Different token formats
                    if tokens:
                        if isinstance(tokens[0], str):
                            token_count = len(tokens)
                        elif isinstance(tokens[0], dict) and 'text' in tokens[0]:
                            token_count = len(tokens)
                        elif isinstance(tokens[0], list) and tokens[0]:
                            token_count = len(tokens)
            
            # If tokens not available, try to tokenize the content
            if token_count == 0 and isinstance(content, str):
                # Simple tokenization by whitespace
                tokens = content.split()
                token_count = len(tokens)
            
            # Save token count
            result['token_counts'][lang] = token_count
        
        # Extract basic metrics and content analysis
        basic_metrics = {
            'title': {},
            'word_counts': {},
            'character_counts': {},
            'sentence_counts': {},
            'entity_counts': {},
            'avg_sentence_length': {}
        }
        
        content = {
            'pos_distribution': {},
            'has_moral': {},
            'moral_length': {},
            'shared_entities': [],
            'unique_entities': {}
        }
        
        # Extract language-specific information
        all_entities = {}
        all_tokens = {}
        
        for lang, fable in lang_fables.items():
            # Basic metrics
            basic_metrics['title'][lang] = fable.get('title', '')
            
            # Calculate word count (based on tokenized text)
            tokens = fable.get('tokens', [])
            if tokens:
                # Filter out punctuation tokens
                word_tokens = []
                if isinstance(tokens[0], str):
                    word_tokens = [t for t in tokens if t and not all(c in '.,;:!?"-()[]{}' for c in t)]
                elif isinstance(tokens[0], dict):
                    word_tokens = [t for t in tokens if 'text' in t and not all(c in '.,;:!?"-()[]{}' for c in t['text'])]
                elif isinstance(tokens[0], list) and len(tokens[0]) > 0:
                    word_tokens = [t[0] for t in tokens if t and not all(c in '.,;:!?"-()[]{}' for c in t[0])]
                    
                basic_metrics['word_counts'][lang] = len(word_tokens)
                all_tokens[lang] = word_tokens
                
                # Character count (without whitespace)
                body = fable.get('body', '')
                if not body:
                    # Try to reconstruct from sentences
                    sentences = fable.get('sentences', [])
                    body = ' '.join(s.get('text', '') for s in sentences)
                    
                char_count = sum(1 for c in body if c.strip())
                basic_metrics['character_counts'][lang] = char_count
            
            # Sentence metrics
            sentences = fable.get('sentences', [])
            basic_metrics['sentence_counts'][lang] = len(sentences)
            
            if sentences and basic_metrics['word_counts'].get(lang, 0) > 0:
                avg_len = basic_metrics['word_counts'][lang] / len(sentences)
                basic_metrics['avg_sentence_length'][lang] = avg_len
            
            # Entity analysis
            entity_list = []
            if 'entities' in fable:
                for entity in fable['entities']:
                    # Handle different entity formats
                    entity_text = None
                    entity_type = None
                    
                    if isinstance(entity, list) and len(entity) >= 2:
                        entity_text = entity[0]
                        entity_type = entity[1]
                    elif isinstance(entity, tuple) and len(entity) >= 2:
                        entity_text = entity[0]
                        entity_type = entity[1]
                    elif isinstance(entity, dict):
                        entity_text = entity.get('text')
                        entity_type = entity.get('label')
                    
                    if entity_text and entity_type:
                        entity_list.append({
                            'text': entity_text,
                            'type': entity_type
                        })
                
                basic_metrics['entity_counts'][lang] = len(entity_list)
                all_entities[lang] = entity_list
            
            # Calculate POS distribution
            pos_counts = {}
            for token_pos in fable.get('pos_tags', []):
                if isinstance(token_pos, list) and len(token_pos) >= 2:
                    pos = token_pos[1]  # Only use the POS tag, ignore token
                    pos_counts[pos] = pos_counts.get(pos, 0) + 1
                
            total_tokens = sum(pos_counts.values())
            if total_tokens > 0:
                content['pos_distribution'][lang] = {
                    pos: count / total_tokens * 100 
                    for pos, count in pos_counts.items()
                }
            
            # Check moral
            moral = fable.get('moral', {})
            if isinstance(moral, dict):
                has_moral = bool(moral.get('text', ''))
                content['has_moral'][lang] = has_moral
                
                if has_moral:
                    moral_text = moral.get('text', '')
                    content['moral_length'][lang] = len(moral_text.split())
        
        # Add basic metrics to result
        result['basic_metrics'] = basic_metrics
        result['content'] = content
        
        # Find shared entities across languages
        if all_entities:
            # Create a normalized entity set for each language
            norm_entities = {}
            for lang, entities in all_entities.items():
                norm_entities[lang] = {e['text'].lower(): e for e in entities}
            
            # Find shared entities (appearing in at least two languages)
            shared_entities = set()
            for lang1, entities1 in norm_entities.items():
                for lang2, entities2 in norm_entities.items():
                    if lang1 >= lang2:  # Skip self-comparison and duplicates
                        continue
                        
                    for entity1 in entities1:
                        if entity1 in entities2:
                            shared_entities.add(entity1)
            
            # Format shared entities
            result['content']['shared_entities'] = list(shared_entities)
            
            # Find unique entities for each language
            for lang, entities in norm_entities.items():
                unique = [
                    {'text': entities[e]['text'], 'type': entities[e]['type']}
                    for e in entities 
                    if e not in shared_entities
                ]
                if unique:
                    result['content']['unique_entities'][lang] = unique
        
        # Calculate similarities between languages
        similarity = {
            'structural_similarity': {},
            'content_similarity': {},
            'lexical_similarity': {}
        }
        
        language_pairs = self._get_language_pairs(result['languages'])
        for lang1, lang2 in language_pairs:
            pair_key = f"{lang1}-{lang2}"
            
            # Calculate structural similarity based on relative metrics
            struct_sim = self._calculate_structural_similarity(
                lang_fables[lang1],
                lang_fables[lang2]
            )
            similarity['structural_similarity'][pair_key] = struct_sim
            
            # Calculate content-based similarity
            content_sim = self._calculate_content_similarity(
                lang_fables[lang1],
                lang_fables[lang2]
            )
            similarity['content_similarity'][pair_key] = content_sim
            
            # Calculate lexical similarity (word frequencies and distributions)
            if lang1 in all_tokens and lang2 in all_tokens:
                lex_sim = self._calculate_lexical_similarity(
                    all_tokens[lang1],
                    all_tokens[lang2]
                )
                similarity['lexical_similarity'][pair_key] = lex_sim
        
        # Add similarity to result
        result['similarity'] = similarity
        
        # Save the comparison results to file
        self.output_dir = self.analysis_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_file = self.output_dir / f"comparison_{fable_id}.json"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            self.logger.debug("Saved comparison results for fable %s", fable_id)
        except Exception as e:
            self.logger.error("Error saving comparison for fable %s: %s", fable_id, e)
        
        return result
    
    def _calculate_structural_similarity(self, fable1: Dict[str, Any], fable2: Dict[str, Any]) -> float:
        """
        Calculate structural similarity between two fables.
        
        Args:
            fable1: First fable dictionary
            fable2: Second fable dictionary
            
        Returns:
            Similarity score (0-1)
        """
        # Compare sentence counts
        sent_count1 = len(fable1.get('sentences', []))
        sent_count2 = len(fable2.get('sentences', []))
        
        if sent_count1 == 0 or sent_count2 == 0:
            return 0.0
            
        # Calculate relative difference
        sent_ratio = min(sent_count1, sent_count2) / max(sent_count1, sent_count2)
        
        # Compare word counts
        tokens1 = fable1.get('tokens', [])
        tokens2 = fable2.get('tokens', [])
        
        word_count1 = 0
        word_count2 = 0
        
        # Count actual words (not punctuation)
        if tokens1:
            if isinstance(tokens1[0], str):
                word_count1 = sum(1 for t in tokens1 if t and not all(c in '.,;:!?"-()[]{}' for c in t))
            elif isinstance(tokens1[0], dict) and 'text' in tokens1[0]:
                word_count1 = sum(1 for t in tokens1 if not all(c in '.,;:!?"-()[]{}' for c in t['text']))
            elif isinstance(tokens1[0], list) and tokens1[0]:
                word_count1 = sum(1 for t in tokens1 if t and not all(c in '.,;:!?"-()[]{}' for c in t[0]))
        
        if tokens2:
            if isinstance(tokens2[0], str):
                word_count2 = sum(1 for t in tokens2 if t and not all(c in '.,;:!?"-()[]{}' for c in t))
            elif isinstance(tokens2[0], dict) and 'text' in tokens2[0]:
                word_count2 = sum(1 for t in tokens2 if not all(c in '.,;:!?"-()[]{}' for c in t['text']))
            elif isinstance(tokens2[0], list) and tokens2[0]:
                word_count2 = sum(1 for t in tokens2 if t and not all(c in '.,;:!?"-()[]{}' for c in t[0]))
        
        word_ratio = 1.0
        if word_count1 > 0 and word_count2 > 0:
            word_ratio = min(word_count1, word_count2) / max(word_count1, word_count2)
        
        # Compare average sentence length
        avg_len1 = word_count1 / sent_count1 if sent_count1 > 0 else 0
        avg_len2 = word_count2 / sent_count2 if sent_count2 > 0 else 0
        
        len_ratio = 1.0
        if avg_len1 > 0 and avg_len2 > 0:
            len_ratio = min(avg_len1, avg_len2) / max(avg_len1, avg_len2)
        
        # Calculate overall structural similarity (weighted average)
        similarity = 0.4 * sent_ratio + 0.4 * word_ratio + 0.2 * len_ratio
        
        return similarity
    
    def _calculate_content_similarity(self, fable1: Dict[str, Any], fable2: Dict[str, Any]) -> float:
        """
        Calculate content-based similarity between two fables.
        
        Args:
            fable1: First fable dictionary
            fable2: Second fable dictionary
            
        Returns:
            Similarity score (0-1)
        """
        # Compare entity overlap
        entities1 = []
        entities2 = []
        
        if 'entities' in fable1:
            for entity in fable1['entities']:
                if isinstance(entity, list) and len(entity) >= 1:
                    entities1.append(entity[0].lower())
                elif isinstance(entity, tuple) and len(entity) >= 1:
                    entities1.append(entity[0].lower())
                elif isinstance(entity, dict) and 'text' in entity:
                    entities1.append(entity['text'].lower())
        
        if 'entities' in fable2:
            for entity in fable2['entities']:
                if isinstance(entity, list) and len(entity) >= 1:
                    entities2.append(entity[0].lower())
                elif isinstance(entity, tuple) and len(entity) >= 1:
                    entities2.append(entity[0].lower())
                elif isinstance(entity, dict) and 'text' in entity:
                    entities2.append(entity['text'].lower())
        
        # Calculate entity overlap (Jaccard similarity)
        entity_sim = 0.0
        if entities1 and entities2:
            set1 = set(entities1)
            set2 = set(entities2)
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            if union > 0:
                entity_sim = intersection / union
        
        # Compare POS distribution similarity
        pos_sim = 0.0
        pos_tags1 = Counter()
        pos_tags2 = Counter()
        
        for token_pos in fable1.get('pos_tags', []):
            if isinstance(token_pos, list) and len(token_pos) >= 2:
                pos_tags1[token_pos[1]] += 1
        
        for token_pos in fable2.get('pos_tags', []):
            if isinstance(token_pos, list) and len(token_pos) >= 2:
                pos_tags2[token_pos[1]] += 1
        
        # Calculate cosine similarity for POS distribution
        if pos_tags1 and pos_tags2:
            all_tags = set(pos_tags1.keys()).union(pos_tags2.keys())
            vec1 = [pos_tags1.get(tag, 0) for tag in all_tags]
            vec2 = [pos_tags2.get(tag, 0) for tag in all_tags]
            
            # Calculate dot product
            dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
            
            # Calculate magnitudes
            mag1 = sum(v * v for v in vec1) ** 0.5
            mag2 = sum(v * v for v in vec2) ** 0.5
            
            if mag1 > 0 and mag2 > 0:
                pos_sim = dot_product / (mag1 * mag2)
        
        # Compare moral presence and length
        moral_sim = 0.0
        moral1 = fable1.get('moral', {})
        moral2 = fable2.get('moral', {})
        
        has_moral1 = False
        has_moral2 = False
        moral_len1 = 0
        moral_len2 = 0
        
        if isinstance(moral1, dict) and 'text' in moral1:
            has_moral1 = bool(moral1['text'])
            if has_moral1:
                moral_len1 = len(moral1['text'].split())
        
        if isinstance(moral2, dict) and 'text' in moral2:
            has_moral2 = bool(moral2['text'])
            if has_moral2:
                moral_len2 = len(moral2['text'].split())
        
        # Compare moral existence and length
        if has_moral1 and has_moral2:
            moral_sim = 1.0
            # Also factor in relative length
            if moral_len1 > 0 and moral_len2 > 0:
                len_ratio = min(moral_len1, moral_len2) / max(moral_len1, moral_len2)
                moral_sim = 0.7 + 0.3 * len_ratio  # Base 0.7 for having a moral at all
        elif not has_moral1 and not has_moral2:
            moral_sim = 1.0  # Both lack a moral
        else:
            moral_sim = 0.0  # One has a moral, one doesn't
        
        # Calculate overall content similarity (weighted average)
        # Entity overlap is most important for content similarity
        similarity = 0.5 * entity_sim + 0.3 * pos_sim + 0.2 * moral_sim
        
        return similarity
    
    def _calculate_lexical_similarity(self, tokens1: List[str], tokens2: List[str]) -> float:
        """
        Calculate lexical similarity based on word frequencies.
        
        Args:
            tokens1: List of tokens from first fable
            tokens2: List of tokens from second fable
            
        Returns:
            Similarity score (0-1)
        """
        # Calculate word frequency distributions
        freq1 = Counter(t.lower() for t in tokens1)
        freq2 = Counter(t.lower() for t in tokens2)
        
        # Calculate relative frequencies
        total1 = sum(freq1.values())
        total2 = sum(freq2.values())
        
        if total1 == 0 or total2 == 0:
            return 0.0
        
        rel_freq1 = {word: count / total1 for word, count in freq1.items()}
        rel_freq2 = {word: count / total2 for word, count in freq2.items()}
        
        # Get all unique words
        all_words = set(rel_freq1.keys()).union(rel_freq2.keys())
        
        # Calculate frequency vector similarity (cosine similarity)
        vec1 = [rel_freq1.get(word, 0) for word in all_words]
        vec2 = [rel_freq2.get(word, 0) for word in all_words]
        
        # Calculate dot product
        dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
        
        # Calculate magnitudes
        mag1 = sum(v * v for v in vec1) ** 0.5
        mag2 = sum(v * v for v in vec2) ** 0.5
        
        # Calculate cosine similarity
        if mag1 > 0 and mag2 > 0:
            return dot_product / (mag1 * mag2)
        else:
            return 0.0
    
    def _get_language_pairs(self, languages: List[str]) -> List[Tuple[str, str]]:
        """
        Generate all unique pairs of languages.
        
        Args:
            languages: List of language codes
            
        Returns:
            List of language code pairs
        """
        pairs = []
        for i in range(len(languages)):
            for j in range(i+1, len(languages)):
                pairs.append((languages[i], languages[j]))
        return pairs
    
    def save_comparison_results(self, comparison: Dict[str, Any], fable_id: str):
        """
        Save comparison results to file.
        
        Args:
            comparison: Comparison results to save
            fable_id: ID of the fable being compared
        """
        # Create directory if it doesn't exist
        output_dir = self.analysis_dir / 'comparisons'
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create filename
        filename = f"{fable_id}_comparison.json"
        output_path = output_dir / filename
        
        # Save to JSON file
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(comparison, f, ensure_ascii=False, indent=2)
            self.logger.info("Saved comparison for fable %s to %s", 
                            fable_id, output_path)
        except Exception as e:
            self.logger.error("Error saving comparison: %s", e)