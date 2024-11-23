import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Load data
movies = pd.read_csv("movie.csv")

# Load the pre-trained KMeans model
with open("kmeans_model.pkl", "rb") as file:
    kmeans_model = pickle.load(file)

# Initialize Lemmatizer and Stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Preprocessing Function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters, numbers, and punctuation
    text = re.sub(r"[^a-z\s]", "", text)
    # Remove stopwords
    words = [word for word in text.split() if word not in stop_words]
    # Lemmatize words
    words = [lemmatizer.lemmatize(word) for word in words]
    # Join back into a single string
    return " ".join(words)

# Apply preprocessing to the descriptions
movies['cleaned_description'] = movies['description'].apply(preprocess_text)

# Build TF-IDF Vectorizer and transform descriptions
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(movies['cleaned_description'])

# Function to Recommend Movies based on KMeans clusters
def recommend_movies_kmeans(movie_title, num_recommendations=5):
    if movie_title not in movies['title'].values:
        return []
    # Get the cluster of the input movie
    movie_idx = movies[movies['title'] == movie_title].index[0]
    movie_cluster = kmeans_model.predict(tfidf_matrix[movie_idx])[0]
    
    # Find all movies in the same cluster
    cluster_indices = [i for i, label in enumerate(kmeans_model.labels_) if label == movie_cluster]
    cluster_movies = movies.iloc[cluster_indices]
    
    # Exclude the input movie from recommendations
    cluster_movies = cluster_movies[cluster_movies['title'] != movie_title]
    
    # Return top N movies from the cluster
    return cluster_movies['title'].head(num_recommendations).tolist()

# Custom CSS for Stylish UI with Black Font
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

# Search Bar
search_query = st.text_input("🔍 Search for a movie you like", placeholder="Type a movie name here...")

if search_query:
    # Display matching movie options
    matching_movies = movies[movies['title'].str.contains(search_query, case=False, na=False)]
    
    if not matching_movies.empty:
        selected_movie = st.selectbox("🎞️ Select a movie:", matching_movies['title'])
        
        if selected_movie:
            # Show recommendations
            recommendations = recommend_movies_kmeans(selected_movie)
            
            if recommendations:
                st.markdown('<h3 class="recommendations">✨ Movies similar to "{}":</h3>'.format(selected_movie), unsafe_allow_html=True)
                for rec in recommendations:
                    with st.container():
                        rec_description = movies[movies['title'] == rec]['description'].values[0]
                        st.markdown(f"""
                            <div class="card">
                                <p class="movie-title">📽️ {rec}</p>
                                <p class="movie-description">{rec_description}</p>
                            </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("No recommendations found. Try another movie!")
    else:
        st.error("No movies match your search query. Please refine your search!")
else:
    st.info("Start typing in the search box to find a movie.")
