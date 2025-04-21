# test_sentiment_analyzer.py

import unittest
from unittest.mock import MagicMock, patch
import logging

# Import the modules to test
from src.aesop_spacy.analysis.sentiment_analyzer import SentimentAnalyzer
from src.aesop_spacy.models.transformer_manager import TransformerManager

class TestSentimentAnalyzer(unittest.TestCase):
    """Test cases for the SentimentAnalyzer class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures, to be run once before all tests."""
        # Configure logging
        logging.basicConfig(
            filename='test_sentiment_analyzer.log',
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Mock the transformer manager for faster testing
        cls.mock_transformer_manager = cls._create_mock_transformer_manager()
        
        # Initialize the sentiment analyzer with the mock transformer
        cls.sentiment_analyzer = SentimentAnalyzer(cls.mock_transformer_manager)
        
        # Load test fables
        cls.test_fables = cls._load_test_fables()
        
        # Organize fables by ID
        cls.fables_by_id = {}
        for fable in cls.test_fables:
            fable_id = fable.get('fable_id')
            language = fable.get('language')
            
            if fable_id not in cls.fables_by_id:
                cls.fables_by_id[fable_id] = {}
                
            cls.fables_by_id[fable_id][language] = fable
    
    @staticmethod
    def _create_mock_transformer_manager():
        """Create a mock TransformerManager that returns predictable results."""
        mock_manager = MagicMock(spec=TransformerManager)
        
        # Configure the classify_sentiment method
        def mock_classify_sentiment(text, model_name=None):
            # Simple sentiment detection based on keywords
            text_lower = text.lower()
            
            if "happy" in text_lower or "luxe" in text_lower or "delicious" in text_lower:
                return {
                    'label': 'positive',
                    'score': 0.75,
                    'confidence': 0.8
                }
            elif "fear" in text_lower or "scared" in text_lower or "doodsbang" in text_lower:
                return {
                    'label': 'negative',
                    'score': 0.2,
                    'confidence': 0.7
                }
            else:
                return {
                    'label': 'neutral',
                    'score': 0.5,
                    'confidence': 0.6
                }
        
        mock_manager.classify_sentiment.side_effect = mock_classify_sentiment
        
        # Mock similarity calculation
        mock_manager.calculate_similarity.return_value = 0.7
        
        return mock_manager
    
    @staticmethod
    def _load_test_fables():
        """Load test fables."""
        fables = []
        
        # Dutch version
        fables.append({
            'fable_id': '2',
            'title': 'De Stadsmuis en de Veldmuis',
            'language': 'nl',
            'body': """Een Stadsmuis ging op bezoek bij een familielid, welke in het veld woonde, en bleef daar eten. De Veldmuis serveerde een maaltijd van tarwe, wortels en eikels, samen met wat koud water om erbij te drinken. De Stadsmuis at maar heel weinig en nam slechts een hapje van dit en een hapje van dat. Het was heel duidelijk dat ze het eenvoudige eten niet lustte en er alleen maar aan knabbelde om niet onbeleefd te zijn. Na de maaltijd begon de Stadsmuis te spreken over haar luxe leven in de stad, terwijl de Veldmuis aandachtig luisterde. Daarna gingen ze naar bed in een gezellig nestje onder de grond en sliepen rustig en ongestoord tot de volgende morgen. Terwijl ze sliep droomde de Veldmuis dat ze een Stadsmuis was en dat ze genoot van alle luxe en genoegens waarover de Stadsmuis verteld had. De volgende morgen vroeg de Stadsmuis aan de Veldmuis of ze graag mee ging naar de stad. De Veldmuis was blij en zei ja. Toen ze in de stad waren gingen ze binnen in een mooi, groot huis. In de eetkamer stond een tafel met daarop de overschotjes van een rijkelijk feestmaal. Er waren snoepjes en gelatinepudding, taartjes, heerlijke kazen, en nog veel andere zaken die muizen zo graag eten. Maar toen de Veldmuis aan een taartje wou knabbelen hoorden ze een Kat luid miauwen en krabben aan de deur. Doodsbang vluchten de muizen naar een schuilplaats en daar bleven ze lange tijd heel stil liggen; ze durfden zelfs amper ademhalen.Toen ze zich tenslotte terug naar de tafel waagden zwaaide plots de deur open. Er kwamen dienstboden binnen om de tafel af te ruimen, op de voet gevolgd door de hond van het huis. In paniek vluchtten de muisjes terug naar hun schuilplaats, welke ze veilig bereikten. Van zodra de dienstboden en de hond de kamer hadden verlaten nam de Veldmuis haar paraplu en haar handtas en zei tegen de Stadsmuis: "Je hebt meer luxe en lekkernijen dan ik heb, maar toch heb ik liever mijn eenvoudig eten en mijn nederig leventje op het platteland. En vooral de vrede en de veiligheid die erbij horen." """,
            'moral': 'Een eenvoudig leven met rust en zekerheid is meer waard dan rijkdom temidden van angst en onzekerheid.',
            'moral_type': 'explicit'
        })
        
        # English version
        fables.append({
            'fable_id': '2',
            'title': 'The City Mouse and the Country Mouse',
            'language': 'en',
            'body': """A city mouse once happened to pay a visit to the house of a country mouse where he was served a humble meal of acorns. The city mouse finished his business in the country, and by means of insistent invitations he persuaded the country mouse to come pay him a visit. The city mouse then brought the country mouse into a room that was overflowing with food. As they were feasting on various delicacies, a butler opened the door. The city mouse quickly concealed himself in a familiar mouse-hole, but the poor country mouse was not acquainted with the house and frantically scurried around the floorboards, frightened out of his wits. When the butler had taken what he needed, he closed the door behind him. The city mouse then urged the country mouse to sit back down to dinner. The country mouse refused and said, 'How could I possibly do that? Oh, how scared I am! Do you think that the man is going to come back?' This was all that the terrified mouse was able to say. The city mouse insisted, 'My dear fellow, you could never find such delicious food as this anywhere else in the world.' 'Acorns are enough for me,' the country mouse maintained, 'so long as I am secure in my freedom!'""",
            'moral': 'It is better to live in self-sufficient poverty than to be tormented by the worries of wealth.',
            'moral_type': 'explicit'
        })
        
        # Add more languages as needed
        return fables
    
    def test_analyze_sentiment_basic(self):
        """Test basic sentiment analysis functionality."""
        # Get the Dutch fable
        dutch_fable = next(f for f in self.test_fables if f['language'] == 'nl')
        
        # Analyze sentiment
        result = self.sentiment_analyzer.analyze_sentiment(dutch_fable)
        
        # Assertions
        self.assertIsNotNone(result)
        self.assertIn('sentiment', result)
        self.assertIn('emotions', result)
        self.assertEqual(result['language'], 'nl')
        
        # Check for body and moral sentiment
        self.assertIn('body', result['sentiment'])
        self.assertIn('moral', result['sentiment'])
    
    def test_emotion_detection(self):
        """Test emotion detection in fables."""
        # Create test text with clear emotions
        joy_text = "I am happy and joyful about the delicious food and luxury!"
        fear_text = "I am scared and afraid, filled with terror and dread!"
        
        # Test emotion detection
        joy_emotions = self.sentiment_analyzer._detect_emotions_keyword(joy_text, 'en')
        fear_emotions = self.sentiment_analyzer._detect_emotions_keyword(fear_text, 'en')
        
        # Check if joy is the dominant emotion in joy_text
        joy_score = joy_emotions.get('joy', 0)
        joy_max = max(joy_emotions.values())
        self.assertEqual(joy_score, joy_max, "Joy should be the dominant emotion in joy_text")
        
        # Check if fear is the dominant emotion in fear_text
        fear_score = fear_emotions.get('fear', 0)
        fear_max = max(fear_emotions.values())
        self.assertEqual(fear_score, fear_max, "Fear should be the dominant emotion in fear_text")
    
    def test_analyze_cross_language(self):
        """Test cross-language sentiment analysis."""
        # Analyze sentiment across languages
        comparison = self.sentiment_analyzer.compare_sentiment_across_languages(self.fables_by_id)
        
        # Assertions
        self.assertIn('2', comparison, "Fable ID 2 should be present in results")
        
        fable_comparison = comparison['2']
        self.assertIn('languages', fable_comparison)
        self.assertIn('sentiment', fable_comparison)
        self.assertIn('emotions', fable_comparison)
        self.assertIn('consistency', fable_comparison)
        
        # Check that we have data for all languages
        for language in self.fables_by_id['2'].keys():
            self.assertIn(language, fable_comparison['sentiment'])
    
    def test_moral_correlation(self):
        """Test correlation between sentiment and moral type."""
        # First, add sentiment data to test fables
        for fable in self.test_fables:
            # Add mock sentiment data
            sentiment_result = self.sentiment_analyzer.analyze_sentiment(fable)
            fable['sentiment'] = sentiment_result['sentiment']
        
        # Analyze correlation
        correlation = self.sentiment_analyzer.correlate_sentiment_with_moral_type(self.test_fables)
        
        # Basic assertions
        self.assertIn('explicit_morals', correlation)
        self.assertIn('implicit_morals', correlation)
        
        # Our test fables all have explicit morals
        self.assertTrue(correlation['explicit_morals']['total'] > 0)
    
    @patch('src.aesop_spacy.analysis.sentiment_analyzer.SentimentAnalyzer._detect_emotions_transformer')
    def test_emotion_fallback(self, mock_transformer):
        """Test fallback to keyword-based emotions if transformer method fails."""
        # Make the transformer method fail
        mock_transformer.side_effect = RuntimeError("Simulated error")
        
        # Get a fable
        fable = self.test_fables[0]
        
        # Analyze sentiment - should fall back to keyword detection
        result = self.sentiment_analyzer.analyze_sentiment(fable)
        
        # We should still get emotions despite the transformer error
        self.assertIn('emotions', result)
        self.assertTrue(len(result['emotions']) > 0)
        
        # Verify the mock was called
        mock_transformer.assert_called_once()


if __name__ == '__main__':
    unittest.main()