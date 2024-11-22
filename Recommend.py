import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load movie data
movies = pd.read_csv("movie.csv")  # Ensure movie.csv contains 'title', 'genre', 'overview'

# Load pre-trained models using joblib instead of pickle
tfidf_genres = joblib.load("tfidf_genres.pkl")
tfidf_overview = joblib.load("tfidf_overview.pkl")
kmeans_model = joblib.load("kmeans_model.pkl")

# Function to combine the loaded TF-IDF matrices (if applicable)
def calculate_combined_tfidf(movies, tfidf_genres, tfidf_overview):
    # Concatenate genres and overview into a single combined feature
    combined_features = movies['genres'].fillna("") + " " + movies['overview'].fillna("")
    
    # Apply the same TF-IDF vectorizer to the combined features
    combined_matrix = tfidf_genres.transform(combined_features)  # Use genres TF-IDF
    return combined_matrix

# Create the combined matrix using the pre-loaded TF-IDF vectorizer
tfidf_matrix = calculate_combined_tfidf(movies, tfidf_genres, tfidf_overview)

# Function to Recommend Movies
def recommend_movies(movie_title, num_recommendations=5):
    if movie_title not in movies['title'].values:
        return []
    
    # Find index of the selected movie
    movie_idx = movies[movies['title'] == movie_title].index[0]
    
    # Calculate cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix[movie_idx], tfidf_matrix).flatten()
    
    # Get top 'num_recommendations' similar movies
    similar_indices = cosine_sim.argsort()[-(num_recommendations + 1):-1][::-1]
    recommended_movies = movies.iloc[similar_indices][['title', 'tagline', 'genres', 'production_companies', 'keywords', 'overview']]
    
    # Filter out movies with NaN in 'genres' or 'overview' columns
    recommended_movies = recommended_movies.dropna(subset=['genres', 'overview'])
    
    return recommended_movies

# Rest of the Streamlit UI code for displaying recommendations remains the same...


# Custom CSS for Stylish UI
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(to right, #ff7e5f, #feb47b);
        font-family: Arial, sans-serif;
    }
    .title {
        font-size: 3em;
        color: #000; /* Changed to black */
        text-align: center;
        padding: 20px;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.2);
    }
    .subtitle {
        font-size: 1.2em;
        color: #000; /* Changed to black */
        text-align: center;
        margin-top: -10px;
    }
    .card {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    }
    .recommendations {
        font-size: 1.1em;
        font-weight: bold;
        color: #000; /* Changed to black */
    }
    .movie-title {
        color: #000; /* Changed to black */
        font-size: 1.5em;
        font-weight: bold;
    }
    .movie-genre {
        color: #000; /* Changed to black */
        font-size: 1.1em;
        font-style: italic;
    }
    .movie-description {
        color: #000; /* Changed to black */
        font-size: 0.9em;
        margin-top: 5px;
    }
    </style>
    """, unsafe_allow_html=True
)

# Title and Subtitle
st.markdown('<h1 class="title">🎥 Movie Recommendation System</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Discover movies tailored to your preferences</p>', unsafe_allow_html=True)
st.markdown('[📊 Visualize Movie Data](visualization.py)', unsafe_allow_html=True)


# Search Bar
search_query = st.text_input("🔍 Search for a movie you like", placeholder="Type a movie name here...")

if search_query:
    # Display matching movie options
    matching_movies = movies[movies['title'].str.contains(search_query, case=False, na=False)]
    
    if not matching_movies.empty:
        selected_movie = st.selectbox("🎞️ Select a movie:", matching_movies['title'])
        
        if selected_movie:
            # Extract the genre and overview for the selected movie
            selected_movie_data = movies[movies['title'] == selected_movie].iloc[0]
            st.markdown(
                f"""
                <div class="card">
                    <p class="movie-title">🎬 {selected_movie}</p>
                    <p class="movie-genre">Genre: {selected_movie_data['genres']}</p>
                    <p class="movie-description">Overview: {selected_movie_data['overview']}</p>
                </div>
                """, unsafe_allow_html=True
            )
            
            # Show recommendations
            recommendations = recommend_movies(selected_movie)
            
            if not recommendations.empty:
                st.markdown('<h3 class="recommendations">✨ Movies Recommended:  "{}":</h3>'.format(selected_movie), unsafe_allow_html=True)
                for _, row in recommendations.iterrows():
                    st.markdown(f"""
                        <div class="card">
                            <p class="movie-title">📽️ {row['title']}</p>
                            <p class="movie-genre">Genre: {row['genres']}</p>
                            <p class="movie-description">{row['overview']}</p>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No recommendations found. Try another movie!")
    else:
        st.error("No movies match your search query. Please refine your search!")
else:
    st.info("Start typing in the search box to find a movie.")
