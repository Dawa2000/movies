import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import concat_ws, col, coalesce, lit, monotonically_increasing_id
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import joblib

# Set environment variables for Spark and Java
os.environ['JAVA_HOME'] = '/opt/bitnami/java'
os.environ['SPARK_HOME'] = '/opt/bitnami/spark'

# Create a Spark session
spark = SparkSession.builder.appName("Movie Recommendation System").getOrCreate()

# Step 1: Load Dataset
data_path = 'hdfs://namenode:8020/input/movie_dataset.csv'  # Path to the dataset on HDFS
df = spark.read.csv(data_path, header=True, inferSchema=True)

# Step 2: Preprocess Data
# Replace null values with empty strings for relevant text columns
text_columns = ['title', 'tagline', 'genres', 'production_companies', 'keywords', 'overview']
for col_name in text_columns:
    df = df.withColumn(col_name, coalesce(col(col_name), lit('')))

# Combine text columns into a single column
df = df.withColumn("combined_text", concat_ws(" ", *text_columns))

# Add a unique identifier column to the DataFrame
df = df.withColumn("row_id", monotonically_increasing_id())

# Step 3: Extract Combined Text for TF-IDF
# Collect the combined text column into a list
combined_text_data = df.select("combined_text").rdd.flatMap(lambda x: x).collect()

# Step 4: Apply TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(combined_text_data)

# Save the TF-IDF vectorizer for future use
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

# Convert the TF-IDF matrix to a dense format for scaling and clustering
tfidf_features = tfidf_matrix.toarray()

# Step 5: Train k-Means Clustering Model
# Initialize and train the k-Means model
num_clusters = 5
kmeans = KMeans(
    n_clusters=num_clusters,     # Number of clusters
    init='k-means++',            # Initialization method ('k-means++' is better for convergence)
    n_init=15,                   # Number of times the algorithm will run with different initial centroids
    max_iter=500,                # Maximum number of iterations for a single run
    tol=1e-4,                    # Tolerance for convergence
    random_state=42              # Random seed for reproducibility
)
kmeans.fit(tfidf_features)

# Save the k-Means model
joblib.dump(kmeans, 'kmeans_model.pkl')

# Step 6: Assign Movies to Clusters
# Predict cluster labels for each movie
cluster_labels = kmeans.labels_

# Verify that the number of cluster labels matches the number of rows in the DataFrame
num_rows = df.count()
num_labels = len(cluster_labels)

if num_rows != num_labels:
    raise ValueError(f"Mismatch: DataFrame rows ({num_rows}) and cluster labels ({num_labels}) do not match.")

# Convert cluster labels to a Pandas DataFrame
import pandas as pd
from pyspark.sql import Row

cluster_labels_df = pd.DataFrame({'row_id': range(len(cluster_labels)), 'cluster': cluster_labels})

# Convert the Pandas DataFrame to a Spark DataFrame
cluster_labels_spark = spark.createDataFrame(
    [Row(row_id=int(i), cluster=int(cluster_labels[i])) for i in range(len(cluster_labels))]
)

# Join the cluster labels with the original Spark DataFrame using the row_id column
clustered_df = df.join(cluster_labels_spark, on="row_id", how="inner")

# Save the clustered DataFrame to HDFS
# clustered_data_path = 'hdfs://namenode:8020/output/clustered_movies.csv'
# clustered_df.write.csv(clustered_data_path, header=True)

# Step 7: Evaluate the k-Means Model
from sklearn.metrics import silhouette_score

# Compute Silhouette Score
silhouette_avg = silhouette_score(tfidf_features, kmeans.labels_)
print(f"Silhouette Score: {silhouette_avg:.4f}")



# Stop the Spark session
spark.stop()