# recommender/evaluation.py

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from .utils import precision_at_k, recall_at_k

def evaluate_recommendations(
    recommendations: pd.DataFrame,
    test_ratings: Dict[int, int],
    k_values: List[int] = [1, 3, 5, 10]
) -> Dict[str, float]:
    """
    Evaluate recommendation performance using various metrics.
    
    Args:
        recommendations: DataFrame with recommended items
        test_ratings: Dictionary mapping item indices to ratings
        k_values: List of k values for precision@k and recall@k
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Extract recommended item indices
    rec_indices = recommendations['item_index'].tolist()
    
    # Get relevant item indices (positive ratings in test set)
    relevant_indices = [idx for idx, rating in test_ratings.items() if rating > 0]
    
    # Calculate metrics
    metrics = {}
    
    # Precision and recall at different k values
    for k in k_values:
        if k <= len(rec_indices):
            metrics[f'precision@{k}'] = precision_at_k(rec_indices, relevant_indices, k)
            metrics[f'recall@{k}'] = recall_at_k(rec_indices, relevant_indices, k)
    
    # Mean reciprocal rank (MRR)
    mrr = 0
    for i, idx in enumerate(rec_indices):
        if idx in relevant_indices:
            mrr = 1.0 / (i + 1)
            break
    metrics['mrr'] = mrr
    
    # Hit rate
    metrics['hit_rate'] = 1.0 if any(idx in relevant_indices for idx in rec_indices) else 0.0
    
    return metrics

def cross_validate(
    recommender,
    ratings: Dict[str, Dict[int, int]],
    n_folds: int = 5,
    test_ratio: float = 0.2
) -> pd.DataFrame:
    """
    Perform cross-validation to evaluate recommender performance.
    
    Args:
        recommender: Recommender system instance
        ratings: Dictionary mapping user IDs to their ratings
        n_folds: Number of folds for cross-validation
        test_ratio: Ratio of ratings to use for testing
        
    Returns:
        DataFrame with evaluation results
    """
    results = []
    
    for user_id, user_ratings in ratings.items():
        # Convert ratings to list of (item_idx, rating) tuples
        rating_items = list(user_ratings.items())
        
        for fold in range(n_folds):
            # Shuffle ratings
            np.random.shuffle(rating_items)
            
            # Split into train and test
            split_idx = int(len(rating_items) * (1 - test_ratio))
            train_ratings = dict(rating_items[:split_idx])
            test_ratings = dict(rating_items[split_idx:])
            
            # Skip fold if no positive ratings in test set
            if not any(rating > 0 for rating in test_ratings.values()):
                continue
                
            # Add training ratings to recommender
            recommender.add_user_ratings(f"{user_id}_{fold}", train_ratings)
            
            # Generate recommendations
            recommendations = recommender.get_recommendations(f"{user_id}_{fold}", top_n=10)
            
            # Evaluate recommendations
            metrics = evaluate_recommendations(recommendations, test_ratings)
            
            # Add fold results
            fold_result = {
                'user_id': user_id,
                'fold': fold,
                'train_size': len(train_ratings),
                'test_size': len(test_ratings),
                'weighted': False,
                **metrics
            }
            results.append(fold_result)
            
            # Repeat with weighted approach
            weighted_recommendations = recommender.get_recommendations(
                f"{user_id}_{fold}", top_n=10, weighted=True
            )
            weighted_metrics = evaluate_recommendations(weighted_recommendations, test_ratings)
            weighted_fold_result = {
                'user_id': user_id,
                'fold': fold,
                'train_size': len(train_ratings),
                'test_size': len(test_ratings),
                'weighted': True,
                **weighted_metrics
            }
            results.append(weighted_fold_result)
    
    return pd.DataFrame(results)

def compare_models(eval_results: pd.DataFrame) -> pd.DataFrame:
    """
    Compare weighted and unweighted models based on evaluation results.
    
    Args:
        eval_results: DataFrame with evaluation results
        
    Returns:
        DataFrame with model comparison
    """
    # Group by weighted flag and calculate mean metrics
    grouped = eval_results.groupby('weighted')
    
    # Get metrics columns (excluding user_id, fold, train_size, test_size, weighted)
    metric_cols = [col for col in eval_results.columns 
                  if col not in ['user_id', 'fold', 'train_size', 'test_size', 'weighted']]
    
    # Calculate mean and std for each metric
    comparison = grouped[metric_cols].agg(['mean', 'std']).reset_index()
    
    return comparison
