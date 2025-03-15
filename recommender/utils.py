# recommender/utils.py

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict

def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """
    Normalize a vector to unit length.
    
    Args:
        vector: Input vector
        
    Returns:
        Normalized vector
    """
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity value
    """
    # Normalize vectors
    vec1_normalized = normalize_vector(vec1)
    vec2_normalized = normalize_vector(vec2)
    
    # Calculate dot product
    return np.dot(vec1_normalized, vec2_normalized)

def precision_at_k(recommended_items: List[int], relevant_items: List[int], k: int) -> float:
    """
    Calculate precision@k.
    
    Args:
        recommended_items: List of recommended item indices
        relevant_items: List of relevant (true positive) item indices
        k: Number of recommendations to consider
        
    Returns:
        Precision@k value
    """
    # Get top k recommendations
    top_k = recommended_items[:k]
    
    # Count number of relevant items in top k
    num_relevant = len(set(top_k) & set(relevant_items))
    
    return num_relevant / k if k > 0 else 0

def recall_at_k(recommended_items: List[int], relevant_items: List[int], k: int) -> float:
    """
    Calculate recall@k.
    
    Args:
        recommended_items: List of recommended item indices
        relevant_items: List of relevant (true positive) item indices
        k: Number of recommendations to consider
        
    Returns:
        Recall@k value
    """
    # Get top k recommendations
    top_k = recommended_items[:k]
    
    # Count number of relevant items in top k
    num_relevant = len(set(top_k) & set(relevant_items))
    
    return num_relevant / len(relevant_items) if len(relevant_items) > 0 else 0
