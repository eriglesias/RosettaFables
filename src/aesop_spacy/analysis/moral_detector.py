# moral_detector.py
"""
Analyzes morals in fables across different languages.

This module provides:
- Detection of explicit morals (tagged or formulaic endings)
- Inference of implicit morals using NLP techniques
- Classification of moral themes and categories
- Cross-language moral comparison
"""

from pathlib import Path
import re
import json
import logging
from collections import Counter

# For topic modeling and keyword extraction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.corpus import stopwords

class MoralDetector:
    """Analyzes explicit and implicit morals in multilingual fables."""
    
    def __init__(self, analysis_dir):
        """
        Initialize the moral detector.
        
        Args:
            analysis_dir: Directory for storing/retrieving analysis results
        """
        self.analysis_dir = Path(analysis_dir)
        self.logger = logging.getLogger(__name__)
        
        # Set up NLTK resources
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        
        # Language-specific moral indicators
        self.moral_indicators = {
            'en': ['moral', 'lesson', 'teaches us', 'shows that', 'reminds us'],
            'de': ['Moral', 'Lehre', 'zeigt uns', 'erinnert uns'],
            'es': ['moraleja', 'enseña', 'muestra que', 'nos recuerda'],
            'nl': ['moraal', 'les', 'leert ons', 'toont dat'],
            'grc': ['ὁ λόγος δηλοῖ', 'ὁ μῦθος δηλοῖ']
        }
        
        # Common moral themes/categories
        self.moral_categories = {
            'prudence': ['caution', 'careful', 'think', 'consider', 'wisdom', 'plan'],
            'honesty': ['truth', 'honest', 'lie', 'deceive', 'integrity'],
            'perseverance': ['persist', 'effort', 'continue', 'try', 'overcome'],
            'kindness': ['kind', 'help', 'assist', 'care', 'compassion'],
            'humility': ['humble', 'pride', 'arrogance', 'modest'],
            'gratitude': ['grateful', 'thank', 'appreciate', 'recognition'],
            'moderation': ['moderate', 'excess', 'enough', 'content'],
            'justice': ['fair', 'justice', 'punish', 'reward', 'deserve']
        }
        
    def detect_explicit_moral(self, fable):
        """
        Detect explicitly stated morals in a fable.
        
        Looks for:
        1. <moral> tags in the original text
        2. Common moral-indicating phrases in each language
        
        Returns:
            Dict with moral text and metadata
        """
        # Initialize results
        results = {
            'has_explicit_moral': False,
            'moral_text': None,
            'moral_location': None,  # start, end or None
            'detection_method': None,
            'confidence': 0.0
        }
        
        # Get language and text
        language = fable.get('language', 'en')
        body = fable.get('body', '')
        
        # Method 1: Look for moral tag in the original text
        moral_tag = fable.get('moral', None)
        moral_type = fable.get('moral_type', None)
        
        if moral_tag and moral_tag.strip():
            results['has_explicit_moral'] = True
            results['moral_text'] = moral_tag.strip()
            results['detection_method'] = 'xml_tag'
            results['confidence'] = 1.0
            
            # If we have information about moral type (explicit/implicit)
            if moral_type:
                results['moral_type'] = moral_type
                
            return results
            
        # Method 2: Look for common moral-indicating phrases
        if body:
            sentences = self._extract_sentences(body, language)
            
            # Check the last 3 sentences for moral indicators
            indicators = self.moral_indicators.get(language, self.moral_indicators['en'])
            
            for i, sentence in enumerate(sentences[-3:]):
                sentence_lower = sentence.lower()
                
                for indicator in indicators:
                    if indicator.lower() in sentence_lower:
                        results['has_explicit_moral'] = True
                        results['moral_text'] = sentence.strip()
                        results['moral_location'] = 'end'
                        results['detection_method'] = 'indicator_phrase'
                        results['confidence'] = 0.8
                        return results
        
        # No explicit moral found
        return results
    
    def infer_implicit_moral(self, fable, explicit_moral_results=None):
        """
        Infer implicit morals when none are explicitly stated.
        
        Uses:
        1. Keyword extraction for identifying key themes
        2. Topic modeling for latent themes
        3. Character relationship analysis
        
        Returns:
            Dict with inferred moral(s) and confidence scores
        """
        # Skip if we already have an explicit moral with high confidence
        if explicit_moral_results and explicit_moral_results.get('has_explicit_moral', False):
            if explicit_moral_results.get('confidence', 0) > 0.7:
                return {
                    'has_inferred_moral': False,
                    'inferred_morals': [],
                    'method': 'skipped_due_to_explicit_moral'
                }
        
        # Get language and text
        language = fable.get('language', 'en')
        body = fable.get('body', '')
        
        if not body:
            return {
                'has_inferred_moral': False,
                'inferred_morals': [],
                'method': None
            }
            
        # Step 1: Extract keywords and important terms
        keywords = self._extract_keywords(body, language)
        
        # Step 2: Apply topic modeling
        topics = self._apply_topic_modeling(body, language)
        
        # Step 3: Analyze character relationships and actions
        characters = self._extract_characters(fable)
        character_actions = self._analyze_character_actions(fable, characters)
        
        # Step 4: Generate potential morals from the combined analysis
        potential_morals = self._generate_potential_morals(
            keywords, topics, character_actions, language
        )
        
        # Step 5: Rank and filter the potential morals
        ranked_morals = self._rank_morals(potential_morals, body, language)
        
        results = {
            'has_inferred_moral': len(ranked_morals) > 0,
            'inferred_morals': ranked_morals,
            'method': 'combined_nlp_inference',
            'keywords': keywords[:10],  # Include top keywords for reference
            'topics': topics
        }
        
        return results
    
    def classify_moral_theme(self, moral_text, language='en'):
        """
        Classify the moral into predefined categories.
        
        Returns:
            Dict with theme categories and confidence scores
        """
        if not moral_text:
            return {
                'categories': [],
                'dominant_category': None
            }
            
        # Convert moral text to lowercase for matching
        moral_lower = moral_text.lower()
        
        # Score each category based on keyword presence
        category_scores = {}
        
        for category, keywords in self.moral_categories.items():
            score = 0
            for keyword in keywords:
                if keyword.lower() in moral_lower:
                    score += 1
            
            # Only include categories with matches
            if score > 0:
                category_scores[category] = score
        
        # Sort categories by score
        sorted_categories = sorted(
            category_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Format the results
        categories = [
            {'name': cat, 'score': score}
            for cat, score in sorted_categories
        ]
        
        dominant_category = categories[0]['name'] if categories else None
        
        return {
            'categories': categories,
            'dominant_category': dominant_category
        }
    
    def calculate_moral_similarity(self, morals_by_language):
        """
        Calculate semantic similarity between morals in different languages.
        
        Uses multilingual sentence embeddings to compare meaning across languages.
        
        Args:
            morals_by_language: Dict mapping language codes to moral texts
            
        Returns:
            Dict with pairwise similarity scores
        """
        # Import here to avoid dependencies if not using this function
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except ImportError:
            return {
                'error': 'sentence-transformers package not installed',
                'install_command': 'pip install sentence-transformers'
            }
        
        # Filter out empty morals
        valid_morals = {
            lang: moral for lang, moral in morals_by_language.items()
            if moral and isinstance(moral, str)
        }
        
        if len(valid_morals) < 2:
            return {
                'error': 'Need at least two valid morals to compare',
                'valid_morals_count': len(valid_morals)
            }
        
        try:
            # Load multilingual model
            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            
            # Encode all morals
            embeddings = {}
            for lang, moral in valid_morals.items():
                embedding = model.encode(moral, convert_to_numpy=True)
                embeddings[lang] = embedding
            
            # Calculate pairwise similarities
            similarities = {}
            languages = list(embeddings.keys())
            
            for i, lang1 in enumerate(languages):
                for lang2 in languages[i+1:]:
                    emb1 = embeddings[lang1]
                    emb2 = embeddings[lang2]
                    
                    # Cosine similarity
                    sim_score = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                    
                    pair_key = f"{lang1}-{lang2}"
                    similarities[pair_key] = float(sim_score)
            
            return {
                'similarities': similarities,
                'method': 'multilingual_embeddings',
                'model': 'paraphrase-multilingual-MiniLM-L12-v2'
            }
        
        except Exception as e:
            return {
                'error': f'Error calculating similarities: {str(e)}'
            }
    
    def compare_morals(self, fables_by_id):
        """
        Compare morals across different language versions of the same fable.
        
        Args:
            fables_by_id: Dict mapping fable IDs to language-specific fables
            
        Returns:
            Dict with comparison results
        """
        comparison = {}
        
        for fable_id, lang_fables in fables_by_id.items():
            # Store results for each language version
            moral_comparison = {
                'languages': list(lang_fables.keys()),
                'morals': {},
                'theme_consistency': None,
                'semantic_similarity': {}
            }
            
            # Extract morals for each language
            for lang, fable in lang_fables.items():
                # Detect explicit moral
                explicit_results = self.detect_explicit_moral(fable)
                
                # Infer implicit moral if needed
                implicit_results = {}
                if not explicit_results.get('has_explicit_moral'):
                    implicit_results = self.infer_implicit_moral(fable)
                
                # Determine the final moral text
                moral_text = None
                if explicit_results.get('has_explicit_moral'):
                    moral_text = explicit_results.get('moral_text')
                elif implicit_results.get('has_inferred_moral'):
                    # Use the top inferred moral
                    inferred = implicit_results.get('inferred_morals', [])
                    if inferred:
                        moral_text = inferred[0].get('text')
                
                # Classify the moral if we have one
                theme_results = {}
                if moral_text:
                    theme_results = self.classify_moral_theme(moral_text, lang)
                
                # Store all results for this language
                moral_comparison['morals'][lang] = {
                    'explicit': explicit_results,
                    'implicit': implicit_results,
                    'final_moral': moral_text,
                    'themes': theme_results
                }
            
            # Analyze theme consistency across languages
            all_themes = []
            for lang, results in moral_comparison['morals'].items():
                dominant = results.get('themes', {}).get('dominant_category')
                if dominant:
                    all_themes.append(dominant)
            
            # Calculate theme consistency
            if all_themes:
                theme_counter = Counter(all_themes)
                most_common = theme_counter.most_common(1)[0]
                consistency = most_common[1] / len(all_themes)
                
                moral_comparison['theme_consistency'] = {
                    'dominant_theme': most_common[0],
                    'consistency_score': consistency
                }
            
            # Calculate semantic similarity between morals
            moral_texts = {}
            for lang, results in moral_comparison['morals'].items():
                moral_text = results.get('final_moral')
                if moral_text:
                    moral_texts[lang] = moral_text

            # Only calculate similarity if we have morals in multiple languages
            if len(moral_texts) >= 2:
                similarity_results = self.calculate_moral_similarity(moral_texts)
                moral_comparison['semantic_similarity'] = similarity_results
            
            comparison[fable_id] = moral_comparison
        
        return comparison
    
    def _extract_sentences(self, text, language):
        """Extract sentences from text with language-specific handling."""
        # Simple sentence splitting for demonstration
        # In a real implementation, use language-specific sentence tokenizers
        return re.split(r'[.!?]+', text)
    
    def _extract_keywords(self, text, language):
        """Extract important keywords from text."""
        # Simple TF-IDF based keyword extraction
        try:
            # Get stopwords for the language if available
            stop_words = set(stopwords.words(self._map_language_code(language)))
        except:
            # Fallback to empty set if language not supported
            stop_words = set()
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=20,
            stop_words=stop_words if stop_words else None
        )
        
        # Apply vectorizer to text
        try:
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            
            # Get scores for each term
            scores = zip(
                feature_names,
                tfidf_matrix.toarray()[0]
            )
            
            # Sort by score (descending)
            sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
            
            return [
                {'term': term, 'score': score}
                for term, score in sorted_scores
            ]
        except:
            # Fallback if vectorization fails
            return []
    
    def _apply_topic_modeling(self, text, language):
        """Apply topic modeling to extract latent themes."""
        # Simplified topic modeling
        try:
            # Get stopwords for the language if available
            stop_words = set(stopwords.words(self._map_language_code(language)))
        except:
            # Fallback to empty set if language not supported
            stop_words = set()
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=50,
            stop_words=stop_words if stop_words else None
        )
        
        # Apply vectorizer to text
        try:
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            
            # Apply LDA
            lda = LatentDirichletAllocation(
                n_components=3,  # Extract 3 topics
                random_state=42
            )
            
            lda.fit(tfidf_matrix)
            
            # Extract top words for each topic
            topics = []
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[:-10 - 1:-1]
                top_words = [feature_names[i] for i in top_words_idx]
                
                topics.append({
                    'id': topic_idx,
                    'top_words': top_words
                })
            
            return topics
        except:
            # Fallback if topic modeling fails
            return []
    
    def _extract_characters(self, fable):
        """Extract main characters from the fable."""
        # In a real implementation, use NER to extract characters
        # For demonstration, we'll use a simple approach
        
        # Get named entities if available
        entities = fable.get('entities', [])
        if entities:
            # Filter for PERSON, ANIMAL, or equivalent entities
            person_entities = [
                e.get('text') 
                for e in entities 
                if e.get('label') in ['PERSON', 'ANIMAL', 'ORG']
            ]
            
            if person_entities:
                # Count frequency and return top entities
                counter = Counter(person_entities)
                return [
                    {'name': name, 'count': count}
                    for name, count in counter.most_common(5)
                ]
        
        # Fallback: look for capitalized words that might be characters
        body = fable.get('body', '')
        words = re.findall(r'\b[A-Z][a-z]+\b', body)
        
        if words:
            counter = Counter(words)
            return [
                {'name': name, 'count': count}
                for name, count in counter.most_common(5)
            ]
        
        return []
    
    def _analyze_character_actions(self, fable, characters):
        """Analyze actions associated with each character."""
        # This would require detailed parsing in a real implementation
        # For demonstration, we'll return a simplified structure
        
        actions = {}
        body = fable.get('body', '')
        
        for character in characters:
            name = character.get('name', '')
            if name:
                # Find sentences containing this character
                sentences = [
                    s for s in re.split(r'[.!?]+', body)
                    if name in s
                ]
                
                # Simplified "action" extraction
                actions[name] = {
                    'sentence_count': len(sentences),
                    'sample_sentences': sentences[:2]  # First two sentences
                }
        
        return actions
    
    def _generate_potential_morals(self, keywords, topics, character_actions, language):
        """Generate potential moral statements from analysis results."""
        # In a real implementation, this would use templates or generative models
        # For demonstration, we'll use a basic approach
        
        potential_morals = []
        
        # 1. Keyword-based potential morals
        if keywords:
            top_keywords = [kw['term'] for kw in keywords[:5]]
            
            # Simple template-based generation
            templates = {
                'en': [
                    "The moral is to {verb} {object}.",
                    "One should always {verb} {object}.",
                    "It's important to {verb} when dealing with {object}."
                ],
                'de': [
                    "Die Moral ist, {object} zu {verb}.",
                    "Man sollte immer {object} {verb}."
                ],
                'es': [
                    "La moraleja es {verb} {object}.",
                    "Uno siempre debe {verb} {object}."
                ],
                'nl': [
                    "De moraal is om {object} te {verb}.",
                    "Men moet altijd {object} {verb}."
                ]
            }
            
            # Use English templates as fallback
            lang_templates = templates.get(language, templates['en'])
            
            # Generate simple moral statements
            for i in range(min(3, len(top_keywords))):
                template = lang_templates[i % len(lang_templates)]
                moral = template.format(
                    verb=top_keywords[i],
                    object=top_keywords[(i+1) % len(top_keywords)]
                )
                
                potential_morals.append({
                    'text': moral,
                    'source': 'keyword_template',
                    'keywords': top_keywords[:2]
                })
        
        # 2. Topic-based potential morals
        if topics:
            for topic in topics:
                top_words = topic.get('top_words', [])[:3]
                if top_words:
                    moral = f"The moral concerns {', '.join(top_words)}."
                    
                    potential_morals.append({
                        'text': moral,
                        'source': 'topic_modeling',
                        'topic_words': top_words
                    })
        
        # 3. Character-action based morals
        if character_actions:
            for character, actions in character_actions.items():
                sample = actions.get('sample_sentences', [])
                if sample:
                    moral = f"The story of {character} teaches us about consequences of our actions."
                    
                    potential_morals.append({
                        'text': moral,
                        'source': 'character_action',
                        'character': character
                    })
        
        return potential_morals
    
    def _rank_morals(self, potential_morals, body, language):
        """Rank and filter potential morals based on relevance to the fable."""
        # In a real implementation, this would use semantic similarity
        # For demonstration, we'll use a simple word overlap approach
        
        ranked_morals = []
        body_words = set(re.findall(r'\b\w+\b', body.lower()))
        
        for moral in potential_morals:
            moral_text = moral.get('text', '')
            moral_words = set(re.findall(r'\b\w+\b', moral_text.lower()))
            
            # Calculate simple overlap score
            overlap = len(moral_words.intersection(body_words))
            score = overlap / len(moral_words) if moral_words else 0
            
            # Add to ranked list
            ranked_morals.append({
                'text': moral_text,
                'relevance_score': score,
                'source': moral.get('source'),
                'metadata': {k: v for k, v in moral.items() if k not in ['text', 'source']}
            })
        
        # Sort by relevance score
        ranked_morals.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Return top results
        return ranked_morals[:3]  # Top 3 morals
    
    def _map_language_code(self, language):
        """Map ISO language codes to NLTK language codes."""
        mapping = {
            'en': 'english',
            'de': 'german',
            'es': 'spanish',
            'nl': 'dutch',
            'grc': 'english'  # Fallback for Ancient Greek
        }
        return mapping.get(language, 'english')