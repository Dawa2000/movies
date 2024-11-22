import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Title for the app
st.title("🎥 Movie Dataset Visualization")

# Load the default dataset
default_file = "movie_d.csv"
if os.path.exists(default_file):
    df = pd.read_csv(default_file)
    st.info(f"Using the default dataset: `{default_file}`.")
else:
    st.error(f"The default dataset `{default_file}` is not found in the local directory. Please ensure the file exists.")
    st.stop()

# Display the first few rows of the dataset
st.subheader("Dataset Preview")
st.write(df.head())

# Display dataset summary
st.subheader("Dataset Summary")
st.write(df.describe())

# Visualization 1: Trend of Movie Releases Over Time
if "release_date" in df.columns:
    st.subheader("1. Trend of Movie Releases Over Time")
    df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
    # Limit the year to 2024 or earlier
    df = df[df['release_year'] <= 2024]
    movies_per_year = df['release_year'].value_counts().sort_index()

    # Ensure the year axis starts from a relevant year range
    movies_per_year = movies_per_year[movies_per_year.index <= 2024]

    fig, ax = plt.subplots(figsize=(10, 6))
    movies_per_year.plot(kind='line', ax=ax, color='blue', marker='o')
    ax.set_xlim(movies_per_year.index.min(), 2024)  # Set x-axis limit to 2024
    ax.set_title("Number of Movies Released Per Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Movies")
    st.pyplot(fig)
else:
    st.warning("Column 'release_date' not found in the dataset!")

# # Visualization 2: Revenue vs. Popularity
# if "revenue" in df.columns and "popularity" in df.columns:
#     st.subheader("2. Revenue vs. Popularity")
#     fig, ax = plt.subplots()
#     sns.scatterplot(data=df, x='popularity', y='revenue', color='green', ax=ax)
#     ax.set_title("Revenue vs. Popularity")
#     ax.set_xlabel("Popularity")
#     ax.set_ylabel("Revenue")
#     st.pyplot(fig)
# else:
#     st.warning("Columns 'revenue' and/or 'popularity' not found in the dataset!")

# Visualization 3: Histogram of Vote Average
if "vote_average" in df.columns:
    st.subheader("2. Histogram of Vote Average")
    fig, ax = plt.subplots()
    sns.histplot(df['vote_average'], bins=30, kde=True, color='skyblue', ax=ax)
    ax.set_title("Distribution of Vote Average")
    st.pyplot(fig)
else:
    st.warning("Column 'vote_average' not found in the dataset!")

# Visualization 4: Scatter Plot - Budget vs Revenue
if "budget" in df.columns and "revenue" in df.columns:
    st.subheader("3. Scatter Plot: Budget vs Revenue")
    fig, ax = plt.subplots()
    sns.scatterplot(
        data=df,
        x='budget', y='revenue',
        hue='vote_average' if 'vote_average' in df.columns else None,
        size='popularity' if 'popularity' in df.columns else None,
        sizes=(20, 200),
        palette="cool",
        ax=ax
    )
    ax.set_title("Scatter Plot: Budget vs Revenue")
    st.pyplot(fig)
else:
    st.warning("Columns 'budget' and/or 'revenue' not found in the dataset!")

# Visualization 5: Top 10 Most Frequent Genres
if "genres" in df.columns:
    st.subheader("4. Top 10 Most Frequent Genres")
    
    # Clean and preprocess the 'genres' column
    df['genres'] = df['genres'].str.lower().str.strip()  # Convert to lowercase and strip spaces
    genre_counts = df['genres'].str.split(',').explode().str.strip().value_counts().head(10)  # Explode and count
    
    # Create the barplot
    fig, ax = plt.subplots()
    sns.barplot(x=genre_counts.values, y=genre_counts.index, ax=ax, palette='viridis')
    ax.set_title("Top 10 Genres")
    ax.set_xlabel("Count")
    ax.set_ylabel("Genre")
    st.pyplot(fig)
else:
    st.warning("Column 'genres' not found in the dataset!")

# Visualization 6: Correlation Heatmap
numeric_cols = ['vote_average', 'revenue', 'runtime', 'budget', 'popularity']
valid_numeric_cols = [col for col in numeric_cols if col in df.columns]
if valid_numeric_cols:
    st.subheader("5. Correlation Heatmap")
    correlation_matrix = df[valid_numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)
else:
    st.warning("No numeric columns available for correlation heatmap!")

st.write("Explore the visualizations to better understand the dataset!")
