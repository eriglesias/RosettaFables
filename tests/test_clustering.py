#test_clustering.py
"""does something"""
import unittest
from pathlib import Path
import logging
import sys


project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the modules to test
try:
    from src.aesop_spacy.analysis.clustering import ClusteringAnalyzer
    SKLEARN_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    SKLEARN_AVAILABLE = False
    
@unittest.skipIf(not SKLEARN_AVAILABLE, "Module import failed or scikit-learn not available")
class TestClusteringAnalyzer(unittest.TestCase):
    """Test cases for ClusteringAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path('test_output')
        self.test_dir.mkdir(exist_ok=True)
        self.analyzer = ClusteringAnalyzer(self.test_dir)
        
        # Configure logging
        logging.basicConfig(
            filename='test_clustering.log',
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create sample fables for testing
        self.sample_fables = [
            {
                'fable_id': '1',
                'language': 'en',
                'body': 'The fox saw a crow sitting on a tree with a piece of cheese in its beak. The fox wanted the cheese, so he said to the crow, "What a beautiful voice you have! Won\'t you sing for me?" The crow was flattered and opened his beak to sing, dropping the cheese. The fox quickly grabbed the cheese and said, "That\'s what happens when you believe everything you hear."',
                'sentences': [
                    {'text': 'The fox saw a crow sitting on a tree with a piece of cheese in its beak.', 
                     'pos_tags': [('The', 'DET'), ('fox', 'NOUN'), ('saw', 'VERB'), ('a', 'DET'), 
                                  ('crow', 'NOUN'), ('sitting', 'VERB'), ('on', 'ADP'), ('a', 'DET'), 
                                  ('tree', 'NOUN'), ('with', 'ADP'), ('a', 'DET'), ('piece', 'NOUN'), 
                                  ('of', 'ADP'), ('cheese', 'NOUN'), ('in', 'ADP'), ('its', 'PRON'), 
                                  ('beak', 'NOUN'), ('.', 'PUNCT')]}
                ]
            },
            {
                'fable_id': '1',
                'language': 'es',
                'body': 'El zorro vio un cuervo sentado en un árbol con un trozo de queso en el pico. El zorro quería el queso, así que le dijo al cuervo: "¡Qué hermosa voz tienes! ¿No cantarías para mí?" El cuervo se sintió halagado y abrió el pico para cantar, dejando caer el queso. El zorro rápidamente agarró el queso y dijo: "Esto es lo que sucede cuando crees todo lo que escuchas."',
                'sentences': [
                    {'text': 'El zorro vio un cuervo sentado en un árbol con un trozo de queso en el pico.',
                     'pos_tags': [('El', 'DET'), ('zorro', 'NOUN'), ('vio', 'VERB'), ('un', 'DET'), 
                                  ('cuervo', 'NOUN'), ('sentado', 'VERB'), ('en', 'ADP'), ('un', 'DET'), 
                                  ('árbol', 'NOUN'), ('con', 'ADP'), ('un', 'DET'), ('trozo', 'NOUN'), 
                                  ('de', 'ADP'), ('queso', 'NOUN'), ('en', 'ADP'), ('el', 'DET'), 
                                  ('pico', 'NOUN'), ('.', 'PUNCT')]}
                ]
            },
            {
                'fable_id': '2',
                'language': 'en',
                'body': 'A town mouse visited his cousin in the country. The country mouse served a simple meal of grains and nuts. The town mouse said, "Come with me to the city for much better food." In the city, they found delicious leftovers in a grand dining room. But just as they started to eat, a cat appeared. They ran for their lives. The country mouse said, "I prefer my simple food in peace to your rich food in fear."',
                'sentences': [
                    {'text': 'A town mouse visited his cousin in the country.',
                     'pos_tags': [('A', 'DET'), ('town', 'NOUN'), ('mouse', 'NOUN'), ('visited', 'VERB'), 
                                  ('his', 'PRON'), ('cousin', 'NOUN'), ('in', 'ADP'), ('the', 'DET'), 
                                  ('country', 'NOUN'), ('.', 'PUNCT')]}
                ]
            },
            {
                'fable_id': '2',
                'language': 'es',
                'body': 'Un ratón de ciudad visitó a su primo en el campo. El ratón de campo sirvió una comida simple de granos y nueces. El ratón de ciudad dijo: "Ven conmigo a la ciudad para una comida mucho mejor." En la ciudad, encontraron deliciosas sobras en un gran comedor. Pero justo cuando comenzaron a comer, apareció un gato. Corrieron por sus vidas. El ratón de campo dijo: "Prefiero mi comida simple en paz a tu comida rica con miedo."',
                'sentences': [
                    {'text': 'Un ratón de ciudad visitó a su primo en el campo.',
                     'pos_tags': [('Un', 'DET'), ('ratón', 'NOUN'), ('de', 'ADP'), ('ciudad', 'NOUN'), 
                                  ('visitó', 'VERB'), ('a', 'ADP'), ('su', 'PRON'), ('primo', 'NOUN'), 
                                  ('en', 'ADP'), ('el', 'DET'), ('campo', 'NOUN'), ('.', 'PUNCT')]}
                ]
            }
        ]
        
        # Create fables by ID for cross-language testing
        self.fables_by_id = {
            '1': {
                'en': self.sample_fables[0],
                'es': self.sample_fables[1]
            },
            '2': {
                'en': self.sample_fables[2],
                'es': self.sample_fables[3]
            }
        }
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove test directory if needed
        # If you want to keep test outputs for inspection, comment this out
        import shutil
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_kmeans_clustering(self):
        """Test K-means clustering."""
        # Test with TF-IDF features
        result = self.analyzer.kmeans_clustering(
            self.sample_fables, n_clusters=2, feature_type='tfidf'
        )
        
        # Basic validation of the result structure
        self.assertIn('method', result)
        self.assertEqual(result['method'], 'kmeans')
        self.assertIn('n_clusters', result)
        self.assertEqual(result['n_clusters'], 2)
        self.assertIn('clusters', result)
        self.assertIn('fable_clusters', result)
        
        # Test with POS features
        result = self.analyzer.kmeans_clustering(
            self.sample_fables, n_clusters=2, feature_type='pos'
        )
        
        self.assertEqual(result['method'], 'kmeans')
        self.assertEqual(result['feature_type'], 'pos')
    
    def test_hierarchical_clustering(self):
        """Test hierarchical clustering."""
        result = self.analyzer.hierarchical_clustering(
            self.sample_fables, n_clusters=2, feature_type='tfidf'
        )
        
        # Basic validation
        self.assertIn('method', result)
        self.assertEqual(result['method'], 'hierarchical')
        self.assertIn('n_clusters', result)
        self.assertIn('clusters', result)
        self.assertIn('dendrogram_data', result)
        
        # Check dendrogram data
        dendrogram_data = result['dendrogram_data']
        self.assertIn('linkage_matrix', dendrogram_data)
        self.assertIn('fable_ids', dendrogram_data)
    
    def test_dbscan_clustering(self):
        """Test DBSCAN clustering."""
        result = self.analyzer.dbscan_clustering(
            self.sample_fables, eps=1.0, min_samples=2, feature_type='tfidf'
        )
        
        # Basic validation
        self.assertIn('method', result)
        self.assertEqual(result['method'], 'dbscan')
        self.assertIn('clusters', result)
    
    def test_cross_language_clustering(self):
        """Test cross-language clustering."""
        result = self.analyzer.cross_language_clustering(
            self.fables_by_id, feature_type='tfidf', method='kmeans'
        )
        
        # Basic validation
        self.assertIn('method', result)
        self.assertIn('clusters', result)
        self.assertIn('fable_clusters', result)
        
        # Check the special cross-language analysis
        self.assertIn('language_distribution', result)
        self.assertIn('fable_id_distribution', result)
        self.assertIn('cluster_tendency', result)
    
    def test_optimize_clusters(self):
        """Test cluster optimization."""
        result = self.analyzer.optimize_clusters(
            self.sample_fables, feature_type='tfidf', max_clusters=3
        )
        
        # Basic validation
        self.assertIn('scores', result)
        self.assertIn('optimal_clusters', result)
        self.assertIn('recommendation', result)
    

if __name__ == '__main__':
    unittest.main()