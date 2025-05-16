import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack


# ---------- Content-Based Recommendation ----------
def get_content_based_recommendations(movie_id):
    # Load movie data
    movies_df = pd.read_csv('project.csv')

    # Fill NaN values
    movies_df.fillna('', inplace=True)

    # Combine overview, cast, and genre into a single column
    movies_df['combined_features'] = movies_df['overview'] + ' ' + movies_df['cast'] + ' ' + movies_df['genre']

    # Vectorize combined features
    count_vectorizer = CountVectorizer(stop_words='english')
    count_matrix = count_vectorizer.fit_transform(movies_df['combined_features'])

    # Compute cosine similarity
    cosine_sim = cosine_similarity(count_matrix)

    # Get index of the movie
    if movie_id not in movies_df['movie_id'].values:
        return pd.DataFrame()  # Return empty if movie ID not found

    movie_index = movies_df[movies_df['movie_id'] == movie_id].index[0]

    # Get top 4 similar movies
    similar_indices = cosine_sim[movie_index].argsort()[::-1][1:5]
    similar_movies = movies_df.iloc[similar_indices]

    return similar_movies[['movie_id', 'movie_name', 'year', 'genre', 'language']]


# ---------- Custom Exception ----------
class RecommendationNotFoundError(Exception):
    pass


# ---------- Collaborative Filtering Recommendation ----------
def get_collaborative_filtering_recommendations(user_id):
    # Load ratings and movies
    ratings_df = pd.read_csv('ratings.csv')
    movies_df = pd.read_csv('project.csv')

    # Validate user ID
    if int(user_id) not in ratings_df['user_id'].unique():
        raise RecommendationNotFoundError(f"User with ID {user_id} not found")

    # Create user-movie matrix
    user_movie_matrix = ratings_df.pivot_table(index='user_id', columns='movie_id', values='rating').fillna(0)

    # Fit Nearest Neighbors model
    knn = NearestNeighbors(n_neighbors=5, metric='cosine')
    knn.fit(user_movie_matrix)

    # Convert user vector to DataFrame with same columns
    user_vector = pd.DataFrame([user_movie_matrix.loc[int(user_id)]], columns=user_movie_matrix.columns)

    # Find similar users
    distances, indices = knn.kneighbors(user_vector, n_neighbors=5)
    similar_user_ids = user_movie_matrix.index[indices.flatten()].tolist()

    # Movies the user has already rated
    user_watched = user_movie_matrix.loc[int(user_id)]
    watched_movies = user_watched[user_watched > 0].index.tolist()

    # Mean rating from similar users
    similar_users_movies = user_movie_matrix.loc[similar_user_ids]
    movie_scores = similar_users_movies.mean(axis=0).sort_values(ascending=False)

    # Exclude already watched
    movie_scores = movie_scores[~movie_scores.index.isin(watched_movies)]

    if movie_scores.empty:
        raise RecommendationNotFoundError(f"No recommendations found for user {user_id}")

    # Get top 4 recommended movie IDs
    top_movie_ids = movie_scores.index[:4].tolist()

    # Merge with movie data
    recommended_movies = pd.merge(pd.DataFrame({'movie_id': top_movie_ids}), movies_df, on='movie_id')

    return recommended_movies[['movie_id', 'movie_name', 'year', 'genre', 'language']]
