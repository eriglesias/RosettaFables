# test_moral_detector.py
"""
Tests for the moral_detector.py module.
"""

import unittest
from pathlib import Path

# Import the MoralDetector class
from aesop_spacy.analysis.moral_detector import MoralDetector

class TestMoralDetector(unittest.TestCase):
    """Tests for the MoralDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test outputs
        self.test_dir = Path("test_output")
        self.test_dir.mkdir(exist_ok=True)
        
        # Initialize the detector
        self.detector = MoralDetector(self.test_dir)
        
        # Define test fables
        self.spanish_fable = {
            'fable_id': '1',
            'title': 'El lobo y el cordero',
            'language': 'es',
            'source': "Aesop's Original Collection",
            'version': '1',
            'body': """Un lobo que vio a un cordero beber en un río quiso devorarlo con un pretexto razonable. Por eso, aunque el lobo estaba situado río arriba, le acusó de haber removido el agua y no dejarle beber. El cordero le dijo que bebía con la punta del hocico y que además no era posible, estando él río abajo, remover el agua de arriba; mas el lobo, al fracasar en ese pretexto, dijo: «El año pasado injuriaste a mi padre». Sin embargo, el cordero dijo que ni siquiera tenía un año de vida, a lo que el lobo replicó: «Aunque tengas abundantes justificaciones, no voy a dejar de devorarte».""",
            'moral': "La fábula muestra que no tiene fuerza una defensa justa con quienes tienen la voluntad de hacer daño.",
            'sentences': [
                {'text': 'Un lobo que vio a un cordero beber en un río quiso devorarlo con un pretexto razonable.'},
                {'text': 'Por eso, aunque el lobo estaba situado río arriba, le acusó de haber removido el agua y no dejarle beber.'},
                {'text': 'El cordero le dijo que bebía con la punta del hocico y que además no era posible, estando él río abajo, remover el agua de arriba;'},
                {'text': 'mas el lobo, al fracasar en ese pretexto, dijo: «El año pasado injuriaste a mi padre».'},
                {'text': 'Sin embargo, el cordero dijo que ni siquiera tenía un año de vida, a lo que el lobo replicó: «Aunque tengas abundantes justificaciones, no voy a dejar de devorarte».'}
            ]
        }
        
        self.english_fable = {
            'fable_id': '1',
            'title': 'The Wolf and the Lamb',
            'language': 'en',
            'source': 'Aesop Laura Gibbs',
            'version': '1',
            'body': """A wolf once saw a lamb who had wandered away from the flock. He did not want to rush upon the lamb and seize him violently. Instead, he sought a reasonable complaint to justify his hatred. 'You insulted me last year, when you were small', said the wolf. The lamb replied, 'How could I have insulted you last year? I'm not even a year old.' The wolf continued, 'Well, are you not cropping the grass of this field which belongs to me?' The lamb said, 'No, I haven't eaten any grass; I have not even begun to graze.' Finally the wolf exclaimed, 'But didn't you drink from the fountain which I drink from?' The lamb answered, 'It is my mother's breast that gives me my drink.' The wolf then seized the lamb and as he chewed he said, 'You are not going to make this wolf go without his dinner, even if you are able to easily refute every one of my charges!""",
            'moral': "This fable's moral is implicit in the narrative",
            'moral_type': 'implicit',
            'sentences': [
                {'text': 'A wolf once saw a lamb who had wandered away from the flock.'},
                {'text': 'He did not want to rush upon the lamb and seize him violently.'},
                {'text': 'Instead, he sought a reasonable complaint to justify his hatred.'},
                {'text': "'You insulted me last year, when you were small', said the wolf."},
                {'text': "The lamb replied, 'How could I have insulted you last year? I'm not even a year old.'"},
                {'text': "The wolf continued, 'Well, are you not cropping the grass of this field which belongs to me?'"},
                {'text': "The lamb said, 'No, I haven't eaten any grass; I have not even begun to graze.'"},
                {'text': "Finally the wolf exclaimed, 'But didn't you drink from the fountain which I drink from?'"},
                {'text': "The lamb answered, 'It is my mother's breast that gives me my drink.'"},
                {'text': "The wolf then seized the lamb and as he chewed he said, 'You are not going to make this wolf go without his dinner, even if you are able to easily refute every one of my charges!'"}
            ]
        }
        
        # Create a fable with no moral for testing
        self.no_moral_fable = {
            'fable_id': '2',
            'title': 'Test Fable Without Moral',
            'language': 'en',
            'body': 'This is a test fable with no explicit moral.',
            'sentences': [
                {'text': 'This is a test fable with no explicit moral.'}
            ]
        }
        
        # Create a collection of fables for testing compare_morals
        self.fables_by_id = {
            '1': {
                'en': self.english_fable,
                'es': self.spanish_fable
            }
        }
    
    def test_detect_explicit_moral_with_tag(self):
        """Test detection of explicit morals with moral tag."""
        print("\n----- Testing explicit moral detection with tag -----")
        
        # Test Spanish fable with explicit moral
        results = self.detector.detect_explicit_moral(self.spanish_fable)
        
        print(f"Found explicit moral: {results['has_explicit_moral']}")
        print(f"Moral text: {results['moral_text']}")
        print(f"Detection method: {results['detection_method']}")
        print(f"Confidence: {results['confidence']}")
        
        self.assertTrue(results['has_explicit_moral'])
    
    def test_detect_explicit_moral_with_type(self):
        """Test detection with moral_type attribute."""
        print("\n----- Testing explicit moral detection with type -----")
        
        # Test English fable with implicit moral
        results = self.detector.detect_explicit_moral(self.english_fable)
        
        print(f"Found explicit moral: {results['has_explicit_moral']}")
        print(f"Moral text: {results['moral_text']}")
        if 'moral_type' in results:
            print(f"Moral type: {results['moral_type']}")
        print(f"Detection method: {results['detection_method']}")
        
        self.assertTrue(results['has_explicit_moral'])
    
    def test_detect_explicit_moral_none(self):
        """Test detection when no explicit moral exists."""
        print("\n----- Testing explicit moral detection with no moral -----")
        
        results = self.detector.detect_explicit_moral(self.no_moral_fable)
        
        print(f"Found explicit moral: {results['has_explicit_moral']}")
        print(f"Moral text: {results['moral_text']}")
        
        self.assertFalse(results['has_explicit_moral'])
    
    def test_infer_implicit_moral(self):
        """Test inference of implicit morals."""
        print("\n----- Testing implicit moral inference -----")
        
        # Create a modified fable without the moral tag
        fable_no_tag = self.spanish_fable.copy()
        del fable_no_tag['moral']
        
        # Test inference
        results = self.detector.infer_implicit_moral(fable_no_tag)
        
        print(f"Has inferred moral: {results['has_inferred_moral']}")
        print(f"Method used: {results['method']}")
        
        if results['has_inferred_moral'] and results['inferred_morals']:
            print("\nTop inferred morals:")
            for i, moral in enumerate(results['inferred_morals']):
                print(f"{i+1}. {moral['text']} (score: {moral['relevance_score']:.2f})")
                
        print("\nTop keywords:")
        for kw in results['keywords'][:5]:
            print(f"- {kw['term']} (score: {kw['score']:.2f})")
            
        print("\nExtracted topics:")
        for topic in results['topics']:
            print(f"Topic {topic['id']+1}: {', '.join(topic['top_words'][:5])}")
            
        self.assertTrue(results['has_inferred_moral'])
    
    def test_infer_implicit_moral_skip_if_explicit(self):
        """Test that inference is skipped when we have explicit moral."""
        print("\n----- Testing implicit moral skipping when explicit exists -----")
        
        # First get explicit moral
        explicit_results = self.detector.detect_explicit_moral(self.spanish_fable)
        
        # Then try to infer implicit
        results = self.detector.infer_implicit_moral(
            self.spanish_fable, explicit_results
        )
        
        print(f"Has inferred moral: {results['has_inferred_moral']}")
        print(f"Method: {results['method']}")
        
        self.assertFalse(results['has_inferred_moral'])
    
    def test_classify_moral_theme(self):
        """Test classification of moral themes."""
        print("\n----- Testing moral theme classification -----")
        
        # Test with a clear justice-themed moral
        moral_text = "La fábula muestra que no tiene fuerza una defensa justa con quienes tienen la voluntad de hacer daño."
        results = self.detector.classify_moral_theme(moral_text, 'es')
        
        print(f"Dominant category: {results['dominant_category']}")
        print("\nAll categories:")
        for cat in results['categories']:
            print(f"- {cat['name']} (score: {cat['score']})")
        
        self.assertIsNotNone(results['dominant_category'])
    
    def test_compare_morals(self):
        """Test comparison of morals across languages."""
        print("\n----- Testing cross-language moral comparison -----")
        
        comparison = self.detector.compare_morals(self.fables_by_id)
        
        # Get results for fable #1
        fable_comparison = comparison['1']
        
        print(f"Languages analyzed: {', '.join(fable_comparison['languages'])}")
        
        print("\nMorals by language:")
        for lang, moral_info in fable_comparison['morals'].items():
            print(f"\n{lang.upper()} moral: {moral_info['final_moral']}")
            
            if 'themes' in moral_info and 'dominant_category' in moral_info['themes']:
                print(f"Dominant theme: {moral_info['themes']['dominant_category']}")
        
        if 'theme_consistency' in fable_comparison and fable_comparison['theme_consistency']:
            consistency = fable_comparison['theme_consistency']
            print(f"\nTheme consistency across languages:")
            print(f"Dominant theme: {consistency.get('dominant_theme')}")
            print(f"Consistency score: {consistency.get('consistency_score', 0):.2f}")
        
        if 'semantic_similarity' in fable_comparison:
            sim = fable_comparison['semantic_similarity']
            print("\nSemantic similarity between languages:")
            if 'similarities' in sim:
                for pair, score in sim['similarities'].items():
                    print(f"{pair}: {score:.4f}")
            elif 'error' in sim:
                print(f"Error: {sim['error']}")
        
        self.assertIn('en', fable_comparison['languages'])
        self.assertIn('es', fable_comparison['languages'])
    
    def test_semantic_similarity(self):
        """Test semantic similarity calculation (if available)."""
        print("\n----- Testing semantic similarity -----")
        
        # Test with two similar morals
        morals = {
            'en': "The strong will do what they can regardless of justice.",
            'es': "La fábula muestra que no tiene fuerza una defensa justa con quienes tienen la voluntad de hacer daño."
        }
        
        results = self.detector.calculate_moral_similarity(morals)
        
        if 'error' in results:
            print(f"Error: {results['error']}")
            if 'install_command' in results:
                print(f"Installation command: {results['install_command']}")
        elif 'similarities' in results:
            print(f"Model used: {results.get('model', 'unknown')}")
            print("\nSimilarity scores:")
            for pair, score in results['similarities'].items():
                print(f"{pair}: {score:.4f}")
                
                # Print interpretation
                if score > 0.7:
                    interpretation = "Very similar meaning"
                elif score > 0.5:
                    interpretation = "Moderately similar meaning"
                elif score > 0.3:
                    interpretation = "Somewhat similar meaning"
                else:
                    interpretation = "Different meaning"
                print(f"  Interpretation: {interpretation}")
        
        # No assertion because this test depends on external library


if __name__ == '__main__':
    unittest.main(verbosity=2)