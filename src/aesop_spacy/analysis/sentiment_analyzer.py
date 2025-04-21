# sentiment_analyzer.py
"""
Analyzes sentiment and emotion in fables across languages.

This module provides:
- Sentiment classification (positive/negative/neutral)
- Emotion detection (joy, anger, fear, etc.)
- Cross-language sentiment comparison
- Correlation between sentiment and moral type
"""

import logging
from typing import Dict, List, Any
import re
import statistics

class SentimentAnalyzer:
    """Analyzes sentiment and emotions in multilingual fables."""
    
    def __init__(self, transformer_manager=None):
        """
        Initialize the sentiment analyzer.
        
        Args:
            transformer_manager: TransformerManager instance for model access
        """
        self.logger = logging.getLogger(__name__)
        self.transformer_manager = transformer_manager
        
        # Define emotion categories and their related keywords
        self.emotion_categories = {
            'joy': ['happy', 'joy', 'delight', 'pleased', 'glad', 'cheerful'],
            'fear': ['afraid', 'fear', 'terror', 'scared', 'frightened', 'dread'],
            'anger': ['angry', 'rage', 'fury', 'outraged', 'annoyed', 'mad'],
            'sadness': ['sad', 'sorrow', 'grief', 'unhappy', 'miserable', 'depressed'],
            'surprise': ['surprised', 'amazed', 'astonished', 'shocked', 'startled'],
            'disgust': ['disgust', 'revolted', 'horrified', 'repulsed', 'appalled']
        }
        
        # Multilingual keywords for emotions - could be expanded
        self.multilingual_emotion_keywords = {
            'es': {
                'joy': ['feliz', 'alegría', 'contento', 'gozo', 'alegre', 'felicidad'],
                'fear': ['miedo', 'temor', 'terror', 'asustado', 'espantado'],
                'anger': ['enfadado', 'rabia', 'furia', 'indignado', 'enojado'],
                'sadness': ['triste', 'tristeza', 'pena', 'infeliz', 'miserable'],
                'surprise': ['sorprendido', 'asombrado', 'atónito', 'estupefacto'],
                'disgust': ['asco', 'repugnancia', 'repulsión', 'repugnante']
            },
            'de': {
                'joy': ['glücklich', 'freude', 'froh', 'fröhlich', 'heiter'],
                'fear': ['angst', 'furcht', 'schrecken', 'erschrocken', 'fürchten'],
                'anger': ['wütend', 'zorn', 'ärger', 'empört', 'verärgert'],
                'sadness': ['traurig', 'trauer', 'kummer', 'betrübt', 'unglücklich'],
                'surprise': ['überrascht', 'erstaunt', 'verblüfft', 'verblüfft'],
                'disgust': ['ekel', 'abscheu', 'angewidert', 'empörung']
            },
            'nl': {
                'joy': ['blij', 'vrolijk', 'vreugde', 'gelukkig', 'opgewekt'],
                'fear': ['angst', 'vrees', 'bang', 'schrik', 'verschrikt'],
                'anger': ['boos', 'woedend', 'kwaad', 'toorn', 'razend'],
                'sadness': ['verdrietig', 'droevig', 'somber', 'triest', 'ongelukkig'],
                'surprise': ['verrast', 'verbaasd', 'versteld', 'geschokt'],
                'disgust': ['walging', 'afkeer', 'weerzin', 'walgelijk']
            }
        }
    
    def analyze_sentiment(self, fable: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze sentiment and emotions in a fable.
        
        Args:
            fable: Fable dictionary with text and metadata
            
        Returns:
            Dict with sentiment and emotion analysis results
        """
        # Initialize results
        results = {
            'sentiment': {
                'overall': None,
                'title': None,
                'body': None,
                'moral': None
            },
            'emotions': {},
            'analysis_method': 'transformer',
            'language': fable.get('language', 'en')
        }
        
        # Get language and text
        language = fable.get('language', 'en')
        title = fable.get('title', '')
        body = fable.get('body', '')
        moral = fable.get('moral', '')
        
        # Check if we have a transformer manager
        if self.transformer_manager is None:
            self.logger.warning("No transformer manager provided. Using keyword-based analysis.")
            results['analysis_method'] = 'keyword'
            
            # Fallback to keyword-based emotion detection
            results['emotions'] = self._detect_emotions_keyword(body, language)
            
            return results
        
        # Analyze parts of the fable
        if title:
            results['sentiment']['title'] = self.transformer_manager.classify_sentiment(title)
        
        if body:
            results['sentiment']['body'] = self.transformer_manager.classify_sentiment(body)
            
            # Also analyze emotions in the body
            try:
                results['emotions'] = self._detect_emotions_transformer(body, language)
            except (RuntimeError, ValueError) as e:
                self.logger.warning("Error detecting emotions with transformer: %s", e)
                # Fall back to keyword-based analysis
                results['emotions'] = self._detect_emotions_keyword(body, language)
        
        if moral:
            results['sentiment']['moral'] = self.transformer_manager.classify_sentiment(moral)
        
        # Calculate overall sentiment (weighted average of body and moral)
        # Fix the unsubscriptable issue with proper checks
        body_result = results['sentiment'].get('body', {})
        body_score = body_result.get('score') if isinstance(body_result, dict) else None
        
        moral_result = results['sentiment'].get('moral', {})
        moral_score = moral_result.get('score') if isinstance(moral_result, dict) else None
            
        if body_score is not None:
            if moral_score is not None:
                # Weight body 70%, moral 30%
                overall_score = (body_score * 0.7) + (moral_score * 0.3)
            else:
                overall_score = body_score
                
            # Map the overall score to a label
            if overall_score > 0.66:
                overall_label = "positive"
            elif overall_score < 0.33:
                overall_label = "negative"
            else:
                overall_label = "neutral"
                
            results['sentiment']['overall'] = {
                'label': overall_label,
                'score': overall_score
            }
        
        return results
    
    def _detect_emotions_transformer(self, text: str, language: str) -> Dict[str, float]:
        """
        Detect emotions in text using transformer models.
        
        Args:
            text: Text to analyze
            language: Language code
            
        Returns:
            Dict mapping emotion categories to intensity scores
        """
        try:
            # This requires a specialized emotion classification model
            # For illustration - in production you'd use a real emotion model
            
            # Example emotion models:
            # - j-hartmann/emotion-english-distilroberta-base (English only)
            # - joeddav/xlm-roberta-large-xnli (multilingual, needs fine-tuning)
            
            # Split text into paragraphs to handle long inputs
            paragraphs = text.split('\n')
            paragraphs = [p for p in paragraphs if p.strip()]
            
            # Process each paragraph
            emotion_scores = {emotion: 0.0 for emotion in self.emotion_categories}
            paragraph_count = 0
            
            for paragraph in paragraphs:
                if not paragraph.strip():
                    continue
                    
                # For now, we'll simulate emotion detection based on sentiment
                # In a real implementation, you'd use a dedicated emotion model
                sentiment = self.transformer_manager.classify_sentiment(paragraph)
                
                # Map sentiment to emotions (very simplified)
                if sentiment['label'] == 'positive' or sentiment['label'] == 'very positive':
                    emotion_scores['joy'] += sentiment['score']
                elif sentiment['label'] == 'negative' or sentiment['label'] == 'very negative':
                    # Distribute between negative emotions
                    neg_score = sentiment['score']
                    emotion_scores['sadness'] += neg_score * 0.4
                    emotion_scores['anger'] += neg_score * 0.3
                    emotion_scores['fear'] += neg_score * 0.3
                
                paragraph_count += 1
            
            # Average the scores
            if paragraph_count > 0:
                for emotion in emotion_scores:
                    emotion_scores[emotion] /= paragraph_count
            
            # Supplement with keyword analysis for better coverage
            keyword_emotions = self._detect_emotions_keyword(text, language)
            
            # Merge the results (70% transformer, 30% keyword)
            for emotion in emotion_scores:
                if emotion in keyword_emotions:
                    emotion_scores[emotion] = (emotion_scores[emotion] * 0.7) + (keyword_emotions[emotion] * 0.3)
            
            return emotion_scores
            
        except RuntimeError as e:
            self.logger.error("Runtime error detecting emotions with transformer: %s", e)
            # Fall back to keyword-based approach
            return self._detect_emotions_keyword(text, language)
        except ValueError as e:
            self.logger.error("Value error detecting emotions with transformer: %s", e)
            return self._detect_emotions_keyword(text, language)
    
    def _detect_emotions_keyword(self, text: str, language: str) -> Dict[str, float]:
        """
        Detect emotions in text using keyword matching.
        
        Args:
            text: Text to analyze
            language: Language code
            
        Returns:
            Dict mapping emotion categories to intensity scores
        """
        # Initialize scores
        emotion_scores = {emotion: 0.0 for emotion in self.emotion_categories}
        
        # Get text and normalize
        text_lower = text.lower()
        
        # Get the appropriate keyword dictionary based on language
        if language in self.multilingual_emotion_keywords:
            keywords = self.multilingual_emotion_keywords[language]
        else:
            # Fallback to English keywords
            keywords = {emotion: words for emotion, words in self.emotion_categories.items()}
        
        # Count keyword occurrences
        total_matches = 0
        for emotion, words in keywords.items():
            count = 0
            for word in words:
                # Look for whole words with word boundaries
                pattern = r'\b' + re.escape(word) + r'\b'
                matches = re.findall(pattern, text_lower)
                count += len(matches)
            
            emotion_scores[emotion] = count
            total_matches += count
        
        # Normalize scores to 0-1 range
        if total_matches > 0:
            for emotion in emotion_scores:
                emotion_scores[emotion] /= total_matches
        else:
            # If no matches, distribute evenly with low scores
            for emotion in emotion_scores:
                emotion_scores[emotion] = 0.1
        
        return emotion_scores
    
    def compare_sentiment_across_languages(self, fables_by_id: Dict[str, Dict[str, Dict]]) -> Dict[str, Any]:
        """
        Compare sentiment and emotions across different language versions of the same fable.
        
        Args:
            fables_by_id: Dict mapping fable IDs to language-specific fables
            
        Returns:
            Dict with comparison results
        """
        comparison = {}
        
        for fable_id, lang_fables in fables_by_id.items():
            # Store results for each language version
            sentiment_comparison = {
                'languages': list(lang_fables.keys()),
                'sentiment': {},
                'emotions': {},
                'consistency': {
                    'sentiment': None,
                    'dominant_emotion': None
                }
            }
            
            # Analyze each language version
            all_sentiments = []
            all_emotions = {}
            
            for lang, fable in lang_fables.items():
                # Analyze sentiment and emotions
                analysis = self.analyze_sentiment(fable)
                
                # Store results
                sentiment_comparison['sentiment'][lang] = analysis['sentiment']
                sentiment_comparison['emotions'][lang] = analysis['emotions']
                
                # Collect data for consistency analysis
                overall_sentiment = analysis['sentiment'].get('overall', {})
                if overall_sentiment and isinstance(overall_sentiment, dict) and 'label' in overall_sentiment:
                    all_sentiments.append(overall_sentiment['label'])
                
                # Collect emotion data
                for emotion, score in analysis['emotions'].items():
                    if emotion not in all_emotions:
                        all_emotions[emotion] = []
                    all_emotions[emotion].append(score)
            
            # Calculate sentiment consistency
            if all_sentiments:
                from collections import Counter
                sentiment_counter = Counter(all_sentiments)
                if sentiment_counter:  # Make sure it's not empty
                    most_common = sentiment_counter.most_common(1)[0]
                    consistency = most_common[1] / len(all_sentiments)
                    
                    sentiment_comparison['consistency']['sentiment'] = {
                        'dominant_sentiment': most_common[0],
                        'consistency_score': consistency
                    }
            
            # Calculate dominant emotion
            if all_emotions:
                # Average each emotion's score across languages
                avg_emotions = {}
                for emotion, scores in all_emotions.items():
                    if scores:  # Make sure we're not calculating mean of an empty sequence
                        avg_emotions[emotion] = statistics.mean(scores)
                
                # Find the dominant emotion
                if avg_emotions:
                    dominant_emotion = max(avg_emotions.items(), key=lambda x: x[1])
                    
                    sentiment_comparison['consistency']['dominant_emotion'] = {
                        'emotion': dominant_emotion[0],
                        'average_score': dominant_emotion[1]
                    }
            
            comparison[fable_id] = sentiment_comparison
        
        return comparison
    
    def correlate_sentiment_with_moral_type(self, fables: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze correlation between sentiment and moral type.
        
        Args:
            fables: List of fable dictionaries with sentiment and moral type info
            
        Returns:
            Dict with correlation analysis
        """
        # Initialize results
        results = {
            'explicit_morals': {
                'positive': 0,
                'neutral': 0,
                'negative': 0,
                'total': 0
            },
            'implicit_morals': {
                'positive': 0,
                'neutral': 0,
                'negative': 0,
                'total': 0
            },
            'correlation': None
        }
        
        # Count fables by moral type and sentiment
        for fable in fables:
            # Extract values with proper checks
            moral_type = fable.get('moral_type', 'implicit')
            
            # Safely extract sentiment
            sentiment_data = fable.get('sentiment', {})
            if isinstance(sentiment_data, dict):
                overall_sentiment = sentiment_data.get('overall', {})
                if isinstance(overall_sentiment, dict):
                    sentiment = overall_sentiment.get('label', 'neutral')
                else:
                    sentiment = 'neutral'
            else:
                sentiment = 'neutral'
            
            # Increment counters
            if moral_type == 'explicit':
                results['explicit_morals'][sentiment] += 1
                results['explicit_morals']['total'] += 1
            else:
                results['implicit_morals'][sentiment] += 1
                results['implicit_morals']['total'] += 1
        
        # Calculate percentages
        for moral_type in ['explicit_morals', 'implicit_morals']:
            total = results[moral_type]['total']
            if total > 0:
                for sentiment in ['positive', 'neutral', 'negative']:
                    count = results[moral_type][sentiment]
                    results[moral_type][f'{sentiment}_percent'] = (count / total) * 100
        
        # Calculate simple correlation
        if results['explicit_morals']['total'] > 0 and results['implicit_morals']['total'] > 0:
            # Compare the distribution of sentiment between explicit and implicit morals
            # This is a very simplified correlation measure
            explicit_pos_pct = results['explicit_morals'].get('positive_percent', 0)
            implicit_pos_pct = results['implicit_morals'].get('positive_percent', 0)
            
            explicit_neg_pct = results['explicit_morals'].get('negative_percent', 0)
            implicit_neg_pct = results['implicit_morals'].get('negative_percent', 0)
            
            # Calculate difference in distribution
            pos_diff = abs(explicit_pos_pct - implicit_pos_pct)
            neg_diff = abs(explicit_neg_pct - implicit_neg_pct)
            
            # Average difference (lower means more similar distribution)
            avg_diff = (pos_diff + neg_diff) / 2
            
            # Convert to a correlation score (0-1, higher means more similar)
            correlation = max(0, 1 - (avg_diff / 100))
            
            results['correlation'] = {
                'score': correlation,
                'interpretation': 'strong' if correlation > 0.7 else 'moderate' if correlation > 0.4 else 'weak'
            }
        
        return results