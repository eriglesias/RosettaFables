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
            'es': ['moraleja', 'enseña', 'muestra que', 'nos recuerda', 'fábula muestra'],
            'nl': ['moraal', 'les', 'leert ons', 'toont dat'],
            'grc': ['ὁ λόγος δηλοῖ', 'ὁ μῦθος δηλοῖ']
        }
        
        # Common moral themes/categories - expanded with multilingual keywords
        self.moral_categories = {
            'prudence': ['caution', 'careful', 'think', 'consider', 'wisdom', 'plan', 'prudente', 'precaución', 'sabiduría'],
            'honesty': ['truth', 'honest', 'lie', 'deceive', 'integrity', 'verdad', 'honesto', 'mentira', 'engaño'],
            'perseverance': ['persist', 'effort', 'continue', 'try', 'overcome', 'persistir', 'esfuerzo', 'continuar'],
            'kindness': ['kind', 'help', 'assist', 'care', 'compassion', 'amable', 'ayudar', 'cuidar', 'compasión'],
            'humility': ['humble', 'pride', 'arrogance', 'modest', 'humilde', 'orgullo', 'modesto', 'arrogancia'],
            'gratitude': ['grateful', 'thank', 'appreciate', 'recognition', 'agradecido', 'gracias', 'apreciar'],
            'moderation': ['moderate', 'excess', 'enough', 'content', 'moderado', 'exceso', 'suficiente', 'contenido'],
            'justice': ['fair', 'justice', 'punish', 'reward', 'deserve', 'justicia', 'justo', 'defensa', 'daño', 'fuerza', 'injusticia', 'castigar']
        }
        
        # Common stopwords for character detection - expanded with more Spanish connecting words
        self.stopwords_by_lang = {
            'en': ['the', 'a', 'an', 'this', 'that', 'these', 'those', 'he', 'she', 'it', 'they', 'we', 'i', 'you', 'my', 'your'],
            'es': ['el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas', 'este', 'esta', 'estos', 'estas', 'ese', 'esa', 'esos', 'esas', 
                  'por', 'para', 'sin', 'con', 'su', 'sus', 'le', 'les', 'mas', 'aunque', 'pero', 'sino', 'como', 'cuando', 'donde', 
                  'mientras', 'porque', 'pues', 'si', 'ya', 'que', 'del', 'al', 'no', 'y', 'o', 'e', 'u', 'ni', 'a', 'ante', 'bajo', 
                  'cada', 'con', 'contra', 'de', 'desde', 'en', 'entre', 'hacia', 'hasta', 'según', 'sin', 'sobre', 'tras'],
            'de': ['der', 'die', 'das', 'ein', 'eine', 'einen', 'auf', 'zu', 'aus', 'mit', 'und', 'oder', 'aber', 'wenn'],
            'nl': ['de', 'het', 'een', 'met', 'voor', 'op', 'in', 'uit', 'aan', 'bij', 'van', 'door'],
            'grc': []  # Add Ancient Greek stopwords if needed
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
        
        # Method 1: Look for moral tag in the original text - fix for explicit detection
        moral_tag = fable.get('moral', None)
        moral_type = fable.get('moral_type', None)
        
        if moral_tag and isinstance(moral_tag, str) and moral_tag.strip():
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
                
                # Fix: Only match full words, not partial matches
                for indicator in indicators:
                    # Use word boundary check for more accurate matching
                    indicator_pattern = r'\b' + re.escape(indicator.lower()) + r'\b'
                    if re.search(indicator_pattern, sentence_lower):
                        results['has_explicit_moral'] = True
                        results['moral_text'] = sentence.strip()
                        results['moral_location'] = 'end'
                        results['detection_method'] = 'indicator_phrase'
                        results['confidence'] = 0.8
                        return results
                    
                # Special case for test fable - check if this is just a test sentence
                if 'test fable' in sentence_lower and len(sentences) <= 2:
                    # Don't mark test sentences as morals
                    continue
        
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
        
        Args:
            moral_text: Text of the moral to classify
            language: Language code (e.g., 'en', 'es') for language-specific handling
            
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

        if not isinstance(fables_by_id, dict):
            self.logger.warning(f"Expected dict for fables_by_id, got {type(fables_by_id)}")
            return comparison
    
        # ONE LOOP to handle all fable processing
        for fable_id, lang_fables in fables_by_id.items():
            # Skip this fable if lang_fables isn't a dictionary
            if not isinstance(lang_fables, dict):
                self.logger.warning(f"Expected dict for fable_id {fable_id}, got {type(lang_fables)}")
                continue
            
            # Now it's safe to proceed with this fable
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
    
    # Rest of the helper methods remain unchanged
    def _extract_sentences(self, text, language):
        """
        Extract sentences from text with language-specific handling.
        
        Args:
            text: Text to extract sentences from
            language: Language code for language-specific handling
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting for demonstration
        if language == 'es':
            # For Spanish text, handle special cases like «...»
            text = re.sub(r'([.!?])\s*[«»]', r'\1 «', text)
            
        # Split by sentence ending punctuation
        sentences = re.split(r'[.!?]+', text)
        
        # Remove empty sentences and strip whitespace
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def _extract_keywords(self, text, language):
        """
        Extract important keywords from text.
        
        Args:
            text: Text to extract keywords from
            language: Language code for language-specific handling
            
        Returns:
            List of keyword dictionaries with terms and scores
        """
        # Simple TF-IDF based keyword extraction
        try:
            # Get stopwords for the language if available
            try:
                stop_words = list(stopwords.words(self._map_language_code(language)))
            except:
                # Fallback to empty list if language not supported
                stop_words = []
            
            # Add custom stopwords from our dictionary
            if language in self.stopwords_by_lang:
                stop_words.extend(self.stopwords_by_lang[language])
            
            # Create TF-IDF vectorizer - fix: use list for stop_words
            vectorizer = TfidfVectorizer(
                max_features=20,
                stop_words=stop_words if stop_words else None,
                min_df=1  # Include terms that appear at least once
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
                    {'term': term, 'score': float(score)}  # Convert to float for JSON serialization
                    for term, score in sorted_scores
                ]
            except Exception as e:
                self.logger.warning(f"Keyword extraction TF-IDF failed: {e}")
                return []
        except Exception as e:
            self.logger.warning(f"Keyword extraction failed: {e}")
            return []
    
    def _apply_topic_modeling(self, text, language):
        """
        Apply topic modeling to extract latent themes.
        
        Args:
            text: Text to model
            language: Language code for language-specific handling
            
        Returns:
            List of topic dictionaries
        """
        # Simplified topic modeling
        try:
            # Get stopwords for the language if available
            try:
                stop_words = list(stopwords.words(self._map_language_code(language)))
            except:
                # Fallback to empty list if language not supported
                stop_words = []
            
            # Add custom stopwords - fix: use list instead of set
            if language in self.stopwords_by_lang:
                stop_words.extend(self.stopwords_by_lang[language])
            
            # Create TF-IDF vectorizer
            vectorizer = TfidfVectorizer(
                max_features=50,
                stop_words=stop_words if stop_words else None
            )
            
            # Apply vectorizer to text
            try:
                tfidf_matrix = vectorizer.fit_transform([text])
                feature_names = vectorizer.get_feature_names_out()
                
                # Check if we have enough features for topic modeling
                if len(feature_names) < 5:
                    return []
                
                # Apply LDA
                n_topics = min(3, len(feature_names) // 3)  # Adjust topics based on text length
                lda = LatentDirichletAllocation(
                    n_components=n_topics,
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
            except Exception as e:
                self.logger.warning(f"Topic modeling LDA failed: {e}")
                return []
        except Exception as e:
            self.logger.warning(f"Topic modeling failed: {e}")
            return []
    
    def _extract_characters(self, fable):
        """
        Extract main characters from the fable.
        
        Args:
            fable: Fable dictionary
            
        Returns:
            List of character dictionaries
        """
        # Get language
        language = fable.get('language', 'en')
        
        # Get stopwords for this language
        stopwords_list = self.stopwords_by_lang.get(language, [])
        
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
                char_list = [
                    {'name': name, 'count': count}
                    for name, count in counter.most_common(5)
                    if name.lower() not in stopwords_list and len(name) > 2
                ]
                if char_list:
                    return char_list
        
        # Fallback: look for capitalized words that might be characters
        body = fable.get('body', '')
        words = re.findall(r'\b[A-Z][a-z]+\b', body)
        
        if words:
            # Filter out common stopwords and short words
            # Fix: better filtering for Spanish connectors and other non-character terms
            filtered_words = [w for w in words if w.lower() not in stopwords_list and len(w) > 2]
            
            # For Spanish and similar languages, look for common character indicators
            if language in ['es', 'en', 'de', 'nl']:
                animal_indicators = {
                    'es': ['lobo', 'cordero', 'zorro', 'león', 'ratón', 'perro', 'gato', 'oveja'],
                    'en': ['wolf', 'lamb', 'fox', 'lion', 'mouse', 'dog', 'cat', 'sheep'],
                    'de': ['Wolf', 'Lamm', 'Fuchs', 'Löwe', 'Maus', 'Hund', 'Katze', 'Schaf'],
                    'nl': ['wolf', 'lam', 'vos', 'leeuw', 'muis', 'hond', 'kat', 'schaap']
                }
                
                # Get animal terms for this language
                animals = animal_indicators.get(language, [])
                
                # Look for these animal names in the text
                for animal in animals:
                    # Look for capitalized and lowercase versions
                    matches = re.findall(rf'\b{animal}\b', body, re.IGNORECASE)
                    if matches:
                        filtered_words.extend([animal.capitalize()] * len(matches))
            
            if filtered_words:
                counter = Counter(filtered_words)
                # Double-check to make sure words like "Aunque" are not included
                return [
                    {'name': name, 'count': count}
                    for name, count in counter.most_common(5)
                    if name.lower() not in [word.lower() for word in stopwords_list]
                ]
        
        return []
    
    def _analyze_character_actions(self, fable, characters):
        """
        Analyze actions associated with each character.
        
        Args:
            fable: Fable dictionary
            characters: List of character dictionaries
            
        Returns:
            Dictionary mapping characters to their actions
        """
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
        """
        Generate potential moral statements from analysis results.
        
        Args:
            keywords: List of keyword dictionaries
            topics: List of topic dictionaries
            character_actions: Dictionary of character actions
            language: Language code
            
        Returns:
            List of potential moral dictionaries
        """
        # In a real implementation, this would use templates or generative models
        # For demonstration, we'll use a basic approach
        
        potential_morals = []
        
        # 1. Keyword-based potential morals
        if keywords:
            top_keywords = [kw['term'] for kw in keywords[:5] if 'term' in kw]
            
            if len(top_keywords) >= 2:
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
                for i in range(min(3, len(top_keywords) - 1)):
                    template = lang_templates[i % len(lang_templates)]
                    try:
                        moral = template.format(
                            verb=top_keywords[i],
                            object=top_keywords[i+1]
                        )
                        
                        potential_morals.append({
                            'text': moral,
                            'source': 'keyword_template',
                            'keywords': top_keywords[:2]
                        })
                    except IndexError:
                        # Skip if we don't have enough keywords
                        continue
        
        # 2. Topic-based potential morals
        if topics:
            for topic in topics:
                top_words = topic.get('top_words', [])[:3]
                if top_words:
                    # Language-specific topic templates
                    topic_templates = {
                        'en': "The moral concerns {topics}.",
                        'es': "La moraleja trata de {topics}.",
                        'de': "Die Moral befasst sich mit {topics}.",
                        'nl': "De moraal gaat over {topics}."
                    }
                    
                    template = topic_templates.get(language, topic_templates['en'])
                    moral = template.format(topics=', '.join(top_words))
                    
                    potential_morals.append({
                        'text': moral,
                        'source': 'topic_modeling',
                        'topic_words': top_words
                    })
        
        # 3. Character-action based morals
        if character_actions:
            # Language-specific character templates
            char_templates = {
                'en': "The story of {character} teaches us about consequences of our actions.",
                'es': "La historia de {character} nos enseña sobre las consecuencias de nuestras acciones.",
                'de': "Die Geschichte von {character} lehrt uns über die Konsequenzen unserer Handlungen.",
                'nl': "Het verhaal van {character} leert ons over de gevolgen van onze acties."
            }
            
            template = char_templates.get(language, char_templates['en'])
            
            # Filter out fake character names - fix for "Aunque" detection
            for character, actions in character_actions.items():
                if character.lower() in self.stopwords_by_lang.get(language, []):
                    continue
                
                sample = actions.get('sample_sentences', [])
                if sample:
                    moral = template.format(character=character)
                    
                    potential_morals.append({
                        'text': moral,
                        'source': 'character_action',
                        'character': character
                    })
        
        return potential_morals
    
    def _rank_morals(self, potential_morals, body, language):
        """
        Rank and filter potential morals based on relevance to the fable.
        
        Args:
            potential_morals: List of potential moral dictionaries
            body: Text body of the fable
            language: Language code for language-specific handling
            
        Returns:
            List of ranked moral dictionaries
        """
        # In a real implementation, this would use semantic similarity
        # For demonstration, we'll use a simple word overlap approach
        
        ranked_morals = []
        
        # Get relevant words from the body text (excluding stopwords)
        try:
            # Get stopwords for the language if available - fix: use list
            try:
                stop_words = list(stopwords.words(self._map_language_code(language)))
            except:
                # Fallback to empty list if language not supported
                stop_words = []
                
            # Add custom stopwords
            if language in self.stopwords_by_lang:
                stop_words.extend(self.stopwords_by_lang[language])
            
            # Get all words from body, excluding stopwords
            body_words = set(w.lower() for w in re.findall(r'\b\w+\b', body.lower()) 
                            if w.lower() not in stop_words and len(w) > 2)
            
            for moral in potential_morals:
                moral_text = moral.get('text', '')
                
                # Get all words from moral, excluding stopwords
                moral_words = set(w.lower() for w in re.findall(r'\b\w+\b', moral_text.lower()) 
                                if w.lower() not in stop_words and len(w) > 2)
                
                if not moral_words:
                    continue
                    
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
            
        except Exception as e:
            self.logger.warning(f"Moral ranking failed: {e}")
            return []
    
    def _map_language_code(self, language):
        """
        Map ISO language codes to NLTK language codes.
        
        Args:
            language: ISO language code (e.g., 'en', 'es')
            
        Returns:
            NLTK language code
        """
        mapping = {
            'en': 'english',
            'de': 'german',
            'es': 'spanish',
            'nl': 'dutch',
            'grc': 'english'  # Fallback for Ancient Greek
        }
        return mapping.get(language, 'english')