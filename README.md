# Content-Based Recommender System

A Python implementation of a content-based recommendation system that analyzes item attributes to generate personalized recommendations.

## Project Overview

This project demonstrates how to build a content-based recommender system from scratch. Content-based filtering is a recommendation technique that uses item features to suggest new items similar to what a user has liked in the past. Unlike collaborative filtering, which relies on user-user similarities, content-based filtering focuses on item attributes and user preferences for those attributes.

## Features

- Basic content-based recommendation using item attributes
- User profile generation based on liked/disliked items
- Weighted attribute approach to handle varying attribute densities
- Performance comparison between weighted and unweighted approaches
- Prediction generation for unrated items

## Installation

```bash
git clone https://github.com/yourusername/content-based-recommender.git
cd content-based-recommender
pip install -r requirements.txt
```

## Dataset

The project includes a sample movies dataset (`data/movies.csv`) that contains:
- Movie titles
- Genre attributes (Adventure, Animation, Children, Comedy, etc.)
- Number of attributes per movie

You can replace this with your own dataset following the same structure, or adjust the code to work with different attribute schemas.

## Usage

### Basic Recommendation Workflow

```python
from recommender import ContentBasedRecommender

# Initialize the recommender
recommender = ContentBasedRecommender()

# Load data
recommender.load_data('data/movies.csv')

# Add user ratings
recommender.add_user_ratings('John', {0: 1, 1: -1, 5: 1, 15: 1, 18: -1})  # Movie indices with ratings

# Generate recommendations
recommendations = recommender.get_recommendations('John', top_n=5)
print(recommendations)
```

### Weighted vs Unweighted Comparison

```python
# Generate unweighted recommendations
unweighted_recommendations = recommender.get_recommendations('John', weighted=False)

# Generate weighted recommendations
weighted_recommendations = recommender.get_recommendations('John', weighted=True)

# Compare the results
recommender.compare_approaches(unweighted_recommendations, weighted_recommendations)
```

## Implementation Details

### User Profile Creation

The system creates a user profile by:
1. Identifying items a user has rated
2. Extracting the attributes of those items
3. Weighting attributes by user ratings (positive for liked items, negative for disliked)
4. Aggregating these weighted attributes into a user preference vector

### Prediction Generation

For unrated items, the system:
1. Takes the dot product of the user's profile vector with each item's attribute vector
2. Normalizes the results (optional)
3. Returns items with the highest scores as recommendations

### Weighted Approach

To address the bias from items with varying numbers of attributes, the weighted approach:
1. Normalizes each item's attribute vector by dividing by the square root of the number of attributes
2. This ensures that items with many attributes don't have disproportionate influence on user profiles

## Performance Considerations

- **Unweighted Model**: Simple and computationally efficient. Best when all items have similar numbers of attributes.
- **Weighted Model**: Slightly more complex but provides more balanced recommendations when items have varying attribute densities.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
