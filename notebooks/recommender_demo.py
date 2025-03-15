# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Content-Based Recommender System Demo
#
# This notebook demonstrates how to use the content-based recommender system to generate personalized recommendations based on item attributes.

# +
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path to import recommender module
sys.path.append('..')
from recommender.content_based import ContentBasedRecommender
# -

# ## 1. Loading and Exploring the Dataset

# +
# Load the dataset
movies_path = '../data/movies.csv'
movies = pd.read_csv(movies_path)

# Display the first few rows
print("Movies dataset preview:")
movies.head()
# -

# Explore dataset statistics
print(f"Dataset shape: {movies.shape}")
print(f"\nNumber of unique movies: {movies[movies.columns[0]].nunique()}")
print(f"\nGenre columns: {movies.columns[1:11].tolist()}")
print(f"\nSummary statistics:")
movies.describe()

# Visualize the distribution of genres in the dataset
plt.figure(figsize=(12, 6))
genre_counts = movies.iloc[:, 1:11].sum()
sns.barplot(x=genre_counts.index, y=genre_counts.values)
plt.title('Distribution of Genres in the Dataset')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Distribution of number of attributes per movie
plt.figure(figsize=(10, 6))
sns.countplot(x='Num_Attr', data=movies)
plt.title('Distribution of Number of Attributes per Movie')
plt.xlabel('Number of Attributes')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# ## 2. Setting Up the Recommender

# Initialize the recommender system
recommender = ContentBasedRecommender()

# Load the data
recommender.load_data(movies_path)

# ## 3. Adding User Ratings

# Define ratings for John and Joan
john_ratings = {0: 1, 1: -1, 5: 1, 15: 1, 18: -1}  # Movie indices with ratings (1=like, -1=dislike)
joan_ratings = {0: -1, 1: 1, 3: 1, 11: -1, 16: 1}

# Add user ratings
recommender.add_user_ratings('John', john_ratings)
recommender.add_user_ratings('Joan', joan_ratings)

# Display movies with their ratings for John
john_rated_movies = pd.DataFrame({
    'Movie': movies.iloc[list(john_ratings.keys()), 0].values,
    'Rating': list(john_ratings.values())
})
print("John's rated movies:")
john_rated_movies

# Display movies with their ratings for Joan
joan_rated_movies = pd.DataFrame({
    'Movie': movies.iloc[list(joan_ratings.keys()), 0].values,
    'Rating': list(joan_ratings.values())
})
print("Joan's rated movies:")
joan_rated_movies

# ## 4. Building User Profiles

# Get John's unweighted profile
john_unweighted_profile = recommender._build_user_profile('John', weighted=False)
print("John's unweighted profile:")
pd.DataFrame(john_unweighted_profile.reshape(1, -1), columns=movies.columns[1:11])

# Get John's weighted profile
john_weighted_profile = recommender._build_user_profile('John', weighted=True)
print("John's weighted profile:")
pd.DataFrame(john_weighted_profile.reshape(1, -1), columns=movies.columns[1:11])

# Compare John's weighted and unweighted profiles visually
plt.figure(figsize=(12, 6))
indices = np.arange(len(movies.columns[1:11]))
width = 0.35

plt.bar(indices - width/2, john_unweighted_profile, width, label='Unweighted')
plt.bar(indices + width/2, john_weighted_profile, width, label='Weighted')

plt.xlabel('Genre')
plt.ylabel('Preference Score')
plt.title('John\'s Profile: Weighted vs Unweighted')
plt.xticks(indices, movies.columns[1:11], rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# ## 5. Generating Recommendations

# Generate predictions for John (unweighted)
john_preds_unweighted = recommender.generate_predictions('John', weighted=False)
print("John's unweighted predictions:")
john_preds_unweighted.sort_values('prediction', ascending=False).head(10)

# Generate predictions for John (weighted)
john_preds_weighted = recommender.generate_predictions('John', weighted=True)
print("John's weighted predictions:")
john_preds_weighted.sort_values('prediction', ascending=False).head(10)

# Get top 5 recommendations for John (unweighted)
john_recs_unweighted = recommender.get_recommendations('John', top_n=5, weighted=False)
print("Top 5 unweighted recommendations for John:")
john_recs_unweighted[['item', 'prediction']]

# Get top 5 recommendations for John (weighted)
john_recs_weighted = recommender.get_recommendations('John', top_n=5, weighted=True)
print("Top 5 weighted recommendations for John:")
john_recs_weighted[['item', 'prediction']]

# Generate predictions for Joan (unweighted)
joan_preds_unweighted = recommender.generate_predictions('Joan', weighted=False)
print("Joan's unweighted predictions:")
joan_preds_unweighted.sort_values('prediction', ascending=False).head(10)

# Generate predictions for Joan (weighted)
joan_preds_weighted = recommender.generate_predictions('Joan', weighted=True)
print("Joan's weighted predictions:")
joan_preds_weighted.sort_values('prediction', ascending=False).head(10)

# ## 6. Comparing Weighted and Unweighted Approaches

# Compare approaches for John
john_comparison = recommender.compare_approaches('John')
print("Comparison of approaches for John:")
john_comparison

# Compare approaches for Joan
joan_comparison = recommender.compare_approaches('Joan')
print("Comparison of approaches for Joan:")
joan_comparison

# Visualize prediction differences between weighted and unweighted approaches
plt.figure(figsize=(12, 8))
movie_indices = john_preds_unweighted['item_index'].values[:10]
movie_names = john_preds_unweighted['item'].values[:10]

x = np.arange(len(movie_indices))
width = 0.35

unweighted_scores = john_preds_unweighted.sort_values('prediction', ascending=False)['prediction'].values[:10]
weighted_scores = john_preds_weighted.loc[john_preds_unweighted.sort_values('prediction', ascending=False).index[:10], 'prediction'].values

plt.bar(x - width/2, unweighted_scores, width, label='Unweighted')
plt.bar(x + width/2, weighted_scores, width, label='Weighted')

plt.xlabel('Movie')
plt.ylabel('Prediction Score')
plt.title('Prediction Scores: Weighted vs Unweighted (John)')
plt.xticks(x, movie_names, rotation=90)
plt.legend()
plt.tight_layout()
plt.show()

# ## 7. Discussion and Insights

# +
# What we've learned:

"""
Key Insights:

1. Unweighted vs Weighted Profiles:
   - Unweighted profiles can be dominated by movies with many genres
   - Weighted profiles normalize the influence of each movie based on its attribute density

2. Recommendation Changes:
   - For John, the top recommendation stays the same in both approaches, suggesting a strong preference
   - For Joan, the weighted approach changes the ranking of recommendations slightly

3. When to Use Each Approach:
   - Unweighted: When all items have similar numbers of attributes or when absolute attribute matching is important
   - Weighted: When items have varying numbers of attributes and you want fair comparison

4. Real-world Applications:
   - Content-based recommenders work well for domains with rich item metadata
   - They can address the "cold start" problem for new users or items better than collaborative filtering

5. Limitations:
   - Content-based systems only recommend items similar to what users have already liked
   - They lack serendipity - the chance discovery of new, unexpected items
   - They require good quality metadata for items
"""
# -

# ## 8. Next Steps

"""
Future Improvements:

1. Feature Engineering:
   - Incorporate more sophisticated attribute weighting (TF-IDF)
   - Add text-based features through NLP techniques

2. Evaluation:
   - Implement offline evaluation metrics (precision, recall, NDCG)
   - Design A/B tests for online evaluation

3. Hybrid Approaches:
   - Combine content-based with collaborative filtering
   - Use content-based as a fallback for cold-start scenarios

4. Scalability:
   - Optimize for larger datasets
   - Implement feature reduction techniques
"""
