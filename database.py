import pandas as pd
import os

def get_movies():
    # Load the CSV file into a DataFrame
    movies_df = pd.read_csv('project.csv')
    
    # Handle NaN values safely (avoids chained assignment warnings)
    movies_df['movie_id'] = movies_df['movie_id'].fillna("Unknown")
    movies_df['movie_name'] = movies_df['movie_name'].fillna("Untitled")
    movies_df['year'] = movies_df['year'].fillna("Unknown")
    movies_df['genre'] = movies_df['genre'].fillna("Unknown")
    movies_df['language'] = movies_df['language'].fillna("Unknown")

    # Convert the DataFrame to a list of dictionaries
    movies = movies_df[['movie_id', 'movie_name', 'year', 'genre', 'language']].to_dict(orient='records')
    
    return movies

def get_user_ratings():
    # Load user ratings from the CSV
    ratings_df = pd.read_csv('ratings.csv')
    return ratings_df

def add_user_rating(user_id, movie_id, rating):
    # Define the path for the ratings CSV file
    ratings_file_path = 'ratings.csv'

    # Load existing ratings or create a new DataFrame if the file doesn't exist
    if os.path.exists(ratings_file_path):
        ratings_df = pd.read_csv(ratings_file_path)
    else:
        ratings_df = pd.DataFrame(columns=['user_id', 'movie_id', 'rating'])

    # Check if the user already rated the movie
    user_rating_exists = ratings_df[
        (ratings_df['user_id'] == user_id) & (ratings_df['movie_id'] == movie_id)
    ]

    if not user_rating_exists.empty:
        # Update the existing rating
        ratings_df.loc[
            (ratings_df['user_id'] == user_id) & (ratings_df['movie_id'] == movie_id),
            'rating'
        ] = rating
    else:
        # Append a new rating
        new_rating = pd.DataFrame({
            'user_id': [user_id],
            'movie_id': [movie_id],
            'rating': [rating]
        })
        ratings_df = pd.concat([ratings_df, new_rating], ignore_index=True)

    # Save the updated DataFrame
    ratings_df.to_csv(ratings_file_path, index=False)
