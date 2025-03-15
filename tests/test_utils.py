import unittest
import numpy as np
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recommender.utils import normalize_vector, cosine_similarity, precision_at_k, recall_at_k

class TestUtils(unittest.TestCase):
    """Test cases for utility functions."""
    
    def test_normalize_vector(self):
        """Test vector normalization."""
        # Test with non-zero vector
        vec = np.array([3, 4])
        normalized = normalize_vector(vec)
        self.assertAlmostEqual(np.linalg.norm(normalized), 1.0)
        self.assertAlmostEqual(normalized[0], 0.6, places=1)
        self.assertAlmostEqual(normalized[1], 0.8, places=1)
        
        # Test with zero vector
        zero_vec = np.array([0, 0])
        normalized_zero = normalize_vector(zero_vec)
        np.testing.assert_array_equal(normalized_zero, zero_vec)
    
    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        # Identical vectors
        vec1 = np.array([1, 2, 3])
        self.assertAlmostEqual(cosine_similarity(vec1, vec1), 1.0)
        
        # Orthogonal vectors
        vec2 = np.array([0, 1, 0])
        vec3 = np.array([0, 0, 1])
        self.assertAlmostEqual(cosine_similarity(vec2, vec3), 0.0)
        
        # Opposite vectors
        vec4 = np.array([1, 1])
        vec5 = np.array([-1, -1])
        self.assertAlmostEqual(cosine_similarity(vec4, vec5), -1.0)
    
    def test_precision_at_k(self):
        """Test precision@k calculation."""
        recommended = [1, 2, 3, 4, 5]
        relevant = [1, 3, 5, 7, 9]
        
        # 2 out of top 3 are relevant
        self.assertAlmostEqual(precision_at_k(recommended, relevant, 3), 2/3)
        
        # 3 out of top 5 are relevant
        self.assertAlmostEqual(precision_at_k(recommended, relevant, 5), 3/5)
        
        # Handle k=0 case
        self.assertAlmostEqual(precision_at_k(recommended, relevant, 0), 0)
    
    def test_recall_at_k(self):
        """Test recall@k calculation."""
        recommended = [1, 2, 3, 4, 5]
        relevant = [1, 3, 5, 7, 9]
        
        # 2 out of 5 relevant items are in top 3
        self.assertAlmostEqual(recall_at_k(recommended, relevant, 3), 2/5)
        
        # 3 out of 5 relevant items are in top 5
        self.assertAlmostEqual(recall_at_k(recommended, relevant, 5), 3/5)
        
        # Handle empty relevant items case
        self.assertAlmostEqual(recall_at_k(recommended, [], 5), 0)

if __name__ == '__main__':
    unittest.main()
