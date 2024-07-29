import os
import pandas as pd
import kaggle
import zipfile
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy

# Ensure the directory exists for Kaggle API key
kaggle_dir = os.path.expanduser('~/.kaggle')
os.makedirs(kaggle_dir, exist_ok=True)

# Define the path to your kaggle.json file
kaggle_json_path = os.path.join(kaggle_dir, 'kaggle.json')

# Uncomment and update the path if you need to move your kaggle.json file
os.rename('/Users/anurag/Documents/GitHub/RecommendationEngine/.kaggle/kaggle.json', kaggle_json_path)

# Define dataset and file path
dataset = 'snap/amazon-fine-food-reviews'
zip_file = 'amazon-fine-food-reviews.zip'

# Create a directory for the data
data_dir = 'data'
os.makedirs(data_dir, exist_ok=True)

# Download the dataset
kaggle.api.dataset_download_files(dataset, path=data_dir, unzip=False)

# Unzip the dataset
with zipfile.ZipFile(os.path.join(data_dir, zip_file), 'r') as zip_ref:
    zip_ref.extractall(data_dir)

# Load the dataset into a DataFrame
data_path = os.path.join(data_dir, 'Reviews.csv')
df = pd.read_csv(data_path)

# Data Preprocessing
df = df[['UserId', 'ProductId', 'Score', 'Text']]
df = df.dropna()
df = df.reset_index(drop=True)

# Text Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)  # Limit number of features
tfidf_matrix = vectorizer.fit_transform(df['Text'])

# Convert to sparse matrix format
sparse_matrix = csr_matrix(tfidf_matrix)

# Dimensionality Reduction
svd = TruncatedSVD(n_components=100)  # Adjust the number of components as needed
reduced_matrix = svd.fit_transform(sparse_matrix)

# Collaborative Filtering with SVD
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['UserId', 'ProductId', 'Score']], reader)

svd_model = SVD()
trainset, testset = train_test_split(data, test_size=0.25)
svd_model.fit(trainset)
predictions = svd_model.test(testset)
print("RMSE: ", accuracy.rmse(predictions))

# Approximate Nearest Neighbors for Content-Based Filtering
nn = NearestNeighbors(n_neighbors=10, algorithm='auto', n_jobs=-1)
nn.fit(reduced_matrix)


def recommend_products(product_id, top_n=10):
    idx = df[df['ProductId'] == product_id].index[0]
    query_vector = reduced_matrix[idx, :].reshape(1, -1)  # Reshape to 2D array
    distances, indices = nn.kneighbors(query_vector)
    product_indices = indices.flatten()[1:top_n + 1]  # Skip the first one as it's the same item
    return df['ProductId'].iloc[product_indices]


# Hybrid Model
def hybrid_recommendation(user_id, product_id, top_n=10):
    # Get collaborative filtering prediction
    pred_collab = svd_model.predict(user_id, product_id).est

    # Get content-based recommendations
    content_recommendations = recommend_products(product_id, top_n)

    # Adjust recommendations based on collaborative filtering score
    recommendations = {
        'collaborative_prediction': pred_collab,
        'content_based_recommendations': content_recommendations
    }

    return recommendations


# Example testing
test_user_id = 'A3SGXH7AUHU8GW'  # Replace with a UserId from the dataset
test_product_id = 'B001E4KFG0'  # Replace with a ProductId from the dataset

# Get hybrid recommendations
hybrid_recommendations = hybrid_recommendation(test_user_id, test_product_id, top_n=10)

print(f'Hybrid recommendations for user {test_user_id} and product {test_product_id}:')
print(f'Collaborative Filtering Prediction Score: {hybrid_recommendations["collaborative_prediction"]}')
print('Content-Based Recommendations:')
print(hybrid_recommendations['content_based_recommendations'])
