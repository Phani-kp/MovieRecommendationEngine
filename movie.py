import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_movie_data():
    """Load sample movie data."""
    data = {
        'title': [
            'The Matrix', 'The Godfather', 'The Dark Knight', 'Pulp Fiction', 'Inception',
            'Fight Club', 'Forrest Gump', 'Interstellar', 'The Lord of the Rings', 'The Social Network'
        ],
        'genre': [
            'Action Sci-Fi', 'Crime Drama', 'Action Crime Drama', 'Crime Drama', 'Action Sci-Fi Thriller',
            'Drama', 'Drama Romance', 'Adventure Drama Sci-Fi', 'Adventure Drama Fantasy', 'Biography Drama'
        ]
    }
    return pd.DataFrame(data)

def recommend_movies(title, movie_data):
    """Recommend movies based on the given title using genre similarity."""
    count_vectorizer = CountVectorizer()
    genre_matrix = count_vectorizer.fit_transform(movie_data['genre'])

    # Compute cosine similarity between movies
    cosine_sim = cosine_similarity(genre_matrix, genre_matrix)

    # Get index of the given movie title
    if title not in movie_data['title'].values:
        return ["Movie not found in the dataset."]

    idx = movie_data[movie_data['title'] == title].index[0]
    similarity_scores = list(enumerate(cosine_sim[idx]))

    # Sort movies based on similarity scores
    sorted_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Get the titles of the top 5 similar movies
    recommendations = [movie_data['title'][i[0]] for i in sorted_movies[1:6]]

    return recommendations

if __name__ == "__main__":
    # Load movie data
    movies = load_movie_data()

    # Example usage
    movie_to_search = "Inception"
    print(f"Recommendations for '{movie_to_search}':")
    recommendations = recommend_movies(movie_to_search, movies)

    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
