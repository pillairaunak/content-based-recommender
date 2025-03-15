# recommender/base.py

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, List, Optional

class BaseRecommender(ABC):
    """Abstract base class for recommender systems."""
    
    @abstractmethod
    def load_data(self, filepath: str) -> None:
        """Load data from a file."""
        pass
    
    @abstractmethod
    def add_user_ratings(self, user_id: str, ratings: Dict[int, int]) -> None:
        """Add ratings for a user."""
        pass
    
    @abstractmethod
    def generate_predictions(self, user_id: str) -> pd.DataFrame:
        """Generate prediction scores for all items for a user."""
        pass
    
    @abstractmethod
    def get_recommendations(self, user_id: str, top_n: int = 5) -> pd.DataFrame:
        """Get top N recommendations for a user."""
        pass
