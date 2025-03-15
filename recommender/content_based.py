# recommender/content_based.py

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union


class ContentBasedRecommender:
    """
    A content-based recommender system that uses item attributes to generate personalized recommendations.
    """
    
    def __init__(self):
        """Initialize the recommender system."""
        self.data = None
        self.user_ratings = {}
        self.user_profiles = {}
        self.attribute_columns = []
        self.item_column = ""
        
    def load_data(self, filepath: str, item_column: str = "Movie", 
                 attribute_columns: List[str] = None) -> None:
        """
        Load data from a CSV file.
        
        Args:
            filepath: Path to the CSV file
            item_column: Name of the column containing item names
            attribute_columns: List of column names containing item attributes
        """
        self.data = pd.read_csv(filepath)
        self.item_column = item_column
        
        if attribute_columns is None:
            # By default, use all numeric columns except 'Num_Attr' as attribute columns
            self.attribute_columns = [col for col in self.data.select_dtypes(include=['int64', 'float64']).columns 
                                     if col != 'Num_Attr']
        else:
            self.attribute_columns = attribute_columns
            
        print(f"Loaded {len(self.data)} items with {len(self.attribute_columns)} attributes.")
    
    def add_user_ratings(self, user_id: str, ratings: Dict[int, int]) -> None:
        """
        Add ratings for a user.
        
        Args:
            user_id: Unique identifier for the user
            ratings: Dictionary mapping item indices to ratings (1 for like, -1 for dislike, 0 for not rated)
        """
        if self.data is None:
            raise ValueError("Data must be loaded before adding user ratings.")
            
        # Create a ratings vector for the user
        user_ratings = np.zeros(len(self.data))
        for idx, rating in ratings.items():
            user_ratings[idx] = rating
            
        self.user_ratings[user_id] = user_ratings
        
        # Build the user profile
        self._build_user_profile(user_id)
        
        print(f"Added ratings for user {user_id} ({len(ratings)} items rated).")
    
    def _build_user_profile(self, user_id: str, weighted: bool = False) -> np.ndarray:
        """
        Build a user profile based on their ratings.
        
        Args:
            user_id: Unique identifier for the user
            weighted: Whether to use weighted attributes
        """
        if user_id not in self.user_ratings:
            raise ValueError(f"No ratings found for user {user_id}.")
            
        user_ratings = self.user_ratings[user_id].reshape(-1, 1)
        
        if weighted:
            # Get weighted item attributes
            item_attributes = self._get_weighted_attributes()
        else:
            # Get unweighted item attributes
            item_attributes = self.data[self.attribute_columns].values
            
        # Calculate user profile
        user_profile = (item_attributes * user_ratings).sum(axis=0)
        
        # Store the profile
        self.user_profiles[f"{user_id}_{'weighted' if weighted else 'unweighted'}"] = user_profile
        
        return user_profile
    
    def _get_weighted_attributes(self) -> np.ndarray:
        """
        Get weighted item attributes by dividing each attribute by the square root of the number of attributes.
        
        Returns:
            Numpy array of weighted attributes
        """
        weighted_data = self.data.copy()
        
        # Calculate the square root of the number of attributes for each item
        if 'Num_Attr' in self.data.columns:
            sqrt_attrs = np.sqrt(self.data['Num_Attr'])
        else:
            # If Num_Attr is not available, count non-zero attributes
            sqrt_attrs = np.sqrt((self.data[self.attribute_columns] > 0).sum(axis=1))
            
        # Divide each attribute by the square root of the number of attributes
        for col in self.attribute_columns:
            weighted_data[col] = weighted_data[col] / sqrt_attrs
            
        return weighted_data[self.attribute_columns].values
    
    def generate_predictions(self, user_id: str, weighted: bool = False) -> pd.DataFrame:
        """
        Generate prediction scores for all items for a user.
        
        Args:
            user_id: Unique identifier for the user
            weighted: Whether to use weighted attributes
            
        Returns:
            DataFrame with items and their prediction scores
        """
        profile_key = f"{user_id}_{'weighted' if weighted else 'unweighted'}"
        
        if profile_key not in self.user_profiles:
            self._build_user_profile(user_id, weighted)
            
        user_profile = self.user_profiles[profile_key]
        
        if weighted:
            item_attributes = self._get_weighted_attributes()
        else:
            item_attributes = self.data[self.attribute_columns].values
            
        # Calculate prediction scores
        pred_scores = (item_attributes * user_profile).sum(axis=1)
        
        # Create a DataFrame with items and their prediction scores
        predictions = pd.DataFrame({
            'item_index': range(len(self.data)),
            'item': self.data[self.item_column],
            'rating': self.user_ratings[user_id],
            'prediction': pred_scores
        })
        
        return predictions
    
    def get_recommendations(self, user_id: str, top_n: int = 5, weighted: bool = False,
                           exclude_rated: bool = True) -> pd.DataFrame:
        """
        Get top N recommendations for a user.
        
        Args:
            user_id: Unique identifier for the user
            top_n: Number of recommendations to return
            weighted: Whether to use weighted attributes
            exclude_rated: Whether to exclude items the user has already rated
            
        Returns:
            DataFrame with top N recommended items
        """
        predictions = self.generate_predictions(user_id, weighted)
        
        if exclude_rated:
            # Filter out items the user has already rated
            recommendations = predictions[predictions['rating'] == 0]
        else:
            recommendations = predictions
            
        # Sort by prediction score (descending)
        recommendations = recommendations.sort_values('prediction', ascending=False)
        
        return recommendations.head(top_n)
    
    def get_best_unrated_item(self, user_id: str, weighted: bool = False) -> Tuple[str, float]:
        """
        Get the best unrated item for a user.
        
        Args:
            user_id: Unique identifier for the user
            weighted: Whether to use weighted attributes
            
        Returns:
            Tuple of (item_name, prediction_score)
        """
        recommendations = self.get_recommendations(user_id, top_n=1, weighted=weighted)
        
        if len(recommendations) == 0:
            return None, None
            
        item = recommendations.iloc[0]['item']
        score = recommendations.iloc[0]['prediction']
        
        return item, score
    
    def count_disliked_items(self, user_id: str, weighted: bool = False) -> int:
        """
        Count the number of items predicted to be disliked by the user.
        
        Args:
            user_id: Unique identifier for the user
            weighted: Whether to use weighted attributes
            
        Returns:
            Number of items with negative prediction scores
        """
        predictions = self.generate_predictions(user_id, weighted)
        return (predictions['prediction'] < 0).sum()
    
    def compare_approaches(self, user_id: str) -> pd.DataFrame:
        """
        Compare weighted and unweighted approaches for a user.
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            DataFrame comparing the two approaches
        """
        # Generate predictions for both approaches
        unweighted_preds = self.generate_predictions(user_id, weighted=False)
        weighted_preds = self.generate_predictions(user_id, weighted=True)
        
        # Get best unrated item for both approaches
        unweighted_best, unweighted_score = self.get_best_unrated_item(user_id, weighted=False)
        weighted_best, weighted_score = self.get_best_unrated_item(user_id, weighted=True)
        
        # Count disliked items for both approaches
        unweighted_disliked = self.count_disliked_items(user_id, weighted=False)
        weighted_disliked = self.count_disliked_items(user_id, weighted=True)
        
        comparison = {
            'Approach': ['Unweighted', 'Weighted'],
            'Best Unrated Item': [unweighted_best, weighted_best],
            'Best Item Score': [unweighted_score, weighted_score],
            'Disliked Items Count': [unweighted_disliked, weighted_disliked]
        }
        
        return pd.DataFrame(comparison)


# Example usage
if __name__ == "__main__":
    # Initialize recommender
    recommender = ContentBasedRecommender()
    
    # Load data
    recommender.load_data('data/movies.csv')
    
    # Add user ratings
    recommender.add_user_ratings('John', {0: 1, 1: -1, 5: 1, 15: 1, 18: -1})
    recommender.add_user_ratings('Joan', {0: -1, 1: 1, 3: 1, 11: -1, 16: 1})
    
    # Generate recommendations
    john_recommendations = recommender.get_recommendations('John', top_n=5)
    print("\nTop 5 recommendations for John:")
    print(john_recommendations[['item', 'prediction']])
    
    # Compare approaches
    print("\nComparison of approaches for John:")
    print(recommender.compare_approaches('John'))
