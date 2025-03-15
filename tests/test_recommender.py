import unittest
import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recommender.content_based import ContentBasedRecommender
from recommender.utils import cosine_similarity

class TestContentBasedRecommender(unittest.TestCase):
    """Test cases for ContentBasedRecommender class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a small test dataset
        self.test_data = pd.DataFrame({
            'Movie': ['Movie A', 'Movie B', 'Movie C', 'Movie D', 'Movie E'],
            'Action': [1, 0, 1, 0, 1],
            'Comedy': [0, 1, 1, 0, 0],
            'Drama': [0, 1, 0, 1, 0],
            'Horror': [0, 0, 0, 1, 1],
            'SciFi': [1, 0, 0, 0, 1],
            'Num_Attr': [2, 2, 2, 2, 3]
        })
        
        # Write test data to a temporary CSV file
        self.test_file = 'tests/test_data.csv'
        os.makedirs(os.path.dirname(self.test_file), exist_ok=True)
        self.test_data.to_csv(self.test_file, index=False)
        
        # Create a recommender instance
        self.recommender = ContentBasedRecommender()
        
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove temporary file
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
    
    def test_load_data(self):
        """Test loading data from a CSV file."""
        self.recommender.load_data(self.test_file)
        
        # Check that data was loaded correctly
        self.assertEqual(len(self.recommender.data), 5)
        self.assertEqual(len(self.recommender.attribute_columns), 5)
        self.assertEqual(self.recommender.item_column, 'Movie')
    
    def test_add_user_ratings(self):
        """Test adding user ratings."""
        self.recommender.load_data(self.test_file)
        self.recommender.add_user_ratings('user1', {0: 1, 1: -1, 2: 0, 3: 0, 4: 1})
        
        # Check that ratings were added correctly
        self.assertIn('user1', self.recommender.user_ratings)
        self.assertEqual(len(self.recommender.user_ratings['user1']), 5)
        self.assertEqual(self.recommender.user_ratings['user1'][0], 1)
        self.assertEqual(self.recommender.user_ratings['user1'][1], -1)
        
        # Check that user profile was built
        self.assertIn('user1_unweighted', self.recommender.user_profiles)
    
    def test_user_profile_creation(self):
        """Test user profile creation."""
        self.recommender.load_data(self.test_file)
        self.recommender.add_user_ratings('user1', {0: 1, 1: -1, 4: 1})
        
        # Check unweighted profile
        unweighted_profile = self.recommender.user_profiles['user1_unweighted']
        self.assertEqual(len(unweighted_profile), 5)  # 5 attributes
        
        # Action: +2 (from Movie A and Movie E)
        # Comedy: -1 (from Movie B)
        # Drama: -1 (from Movie B)
        # Horror: +1 (from Movie E)
        # SciFi: +2 (from Movie A and Movie E)
        self.assertEqual(unweighted_profile[0], 2)  # Action
        self.assertEqual(unweighted_profile[1], -1)  # Comedy
        self.assertEqual(unweighted_profile[2], -1)  # Drama
        self.assertEqual(unweighted_profile[3], 1)   # Horror
        self.assertEqual(unweighted_profile[4], 2)   # SciFi
        
        # Test weighted profile creation
        weighted_profile = self.recommender._build_user_profile('user1', weighted=True)
        self.assertEqual(len(weighted_profile), 5)  # 5 attributes
        
        # Weights should be different when normalized by sqrt(Num_Attr)
        self.assertNotEqual(unweighted_profile[0], weighted_profile[0])
    
    def test_predictions(self):
        """Test prediction generation."""
        self.recommender.load_data(self.test_file)
        self.recommender.add_user_ratings('user1', {0: 1, 1: -1})
        
        # Get predictions
        predictions = self.recommender.generate_predictions('user1')
        
        # Check that predictions were generated for all items
        self.assertEqual(len(predictions), 5)
        
        # Since user1 likes Movie A (Action + SciFi) and dislikes Movie B (Comedy + Drama),
        # they should prefer Movie E (Action + Horror + SciFi) over Movie D (Drama + Horror)
        movie_e_index = predictions[predictions['item'] == 'Movie E'].index[0]
        movie_d_index = predictions[predictions['item'] == 'Movie D'].index[0]
        self.assertGreater(predictions.loc[movie_e_index, 'prediction'], 
                           predictions.loc[movie_d_index, 'prediction'])
    
    def test_recommendations(self):
        """Test recommendation generation."""
        self.recommender.load_data(self.test_file)
        self.recommender.add_user_ratings('user1', {0: 1, 1: -1})
        
        # Get top 2 recommendations
        recommendations = self.recommender.get_recommendations('user1', top_n=2)
        
        # Check that 2 recommendations were returned
        self.assertEqual(len(recommendations), 2)
        
        # Check that recommended items have rating 0 (unrated)
        self.assertTrue((recommendations['rating'] == 0).all())
        
        # First recommendation should be Movie E (most similar to Movie A)
        self.assertEqual(recommendations.iloc[0]['item'], 'Movie E')
    
    def test_weighted_vs_unweighted(self):
        """Test comparison between weighted and unweighted approaches."""
        self.recommender.load_data(self.test_file)
        self.recommender.add_user_ratings('user1', {0: 1, 1: -1})
        
        # Compare approaches
        comparison = self.recommender.compare_approaches('user1')
        
        # Check that comparison has 2 rows (one for each approach)
        self.assertEqual(len(comparison), 2)
        
        # Check that approaches are correctly labeled
        self.assertEqual(comparison.iloc[0]['Approach'], 'Unweighted')
        self.assertEqual(comparison.iloc[1]['Approach'], 'Weighted')

if __name__ == '__main__':
    unittest.main()
