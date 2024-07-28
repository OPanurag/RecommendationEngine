# -*- coding: utf-8 -*-
"""Code.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1uO6lgyO8t2Z5FlBPJNSQV7t_1sXC1wQm
"""

import os

import kaggle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader
from surprise import accuracy
from surprise.model_selection import cross_validate, train_test_split

kaggle_dir = 'kaggle'
os.makedirs(kaggle_dir, exist_ok=True)

# Move the kaggle.json to the directory
# Make sure you have 'kaggle.json' in the 'kaggle' directory or adjust the path as needed
kaggle_json_path = os.path.join(kaggle_dir, 'kaggle.json')
os.rename('path_to_your_kaggle_json/kaggle.json', kaggle_json_path)

# Set Kaggle API credentials
os.environ['KAGGLE_CONFIG_DIR'] = kaggle_dir

# Define dataset and file path
dataset = 'snap/amazon-fine-food-reviews'
zip_file = 'amazon-fine-food-reviews.zip'

# Download the dataset
kaggle.api.dataset_download_files(dataset, path='data/', unzip=False)

# Unzip the dataset
with zipfile.ZipFile(f'data/{zip_file}', 'r') as zip_ref:
    zip_ref.extractall('data/')

import pandas as pd

# Load the dataset into a DataFrame
data_path = 'data/Reviews.csv'
df = pd.read_csv(data_path)

# Preview the data
df.head()

# Data Preprocessing
df = df[['UserId', 'ProductId', 'Score', 'Text']]
df = df.dropna()
df = df.reset_index(drop=True)

# Text Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['Text'])

# Collaborative Filtering with SVD
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['UserId', 'ProductId', 'Score']], reader)

svd = SVD()
cross_validate(svd, data, measures=['RMSE'], cv=5, verbose=True)

# Content-Based Filtering
cosine_sim = cosine_similarity(X, X)


def recommend_products(product_id, top_n=10):
    idx = df[df['ProductId'] == product_id].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n + 1]
    product_indices = [i[0] for i in sim_scores]
    return df['ProductId'].iloc[product_indices]


# Hybrid Model
def hybrid_recommendation(user_id, product_id, top_n=10):
    pred_collab = svd.predict(user_id, product_id).est
    recommendations = recommend_products(product_id, top_n)
    return recommendations


# Evaluation
trainset, testset = train_test_split(data, test_size=0.25)
svd.fit(trainset)
predictions = svd.test(testset)
print("RMSE: ", accuracy.rmse(predictions))

# Example testing
test_user_id = 'A3SGXH7AUHU8GW'  # Replace with a UserId from dataset
test_product_id = 'B001E4KFG0'  # Replace with a ProductId from dataset

# Get hybrid recommendations
hybrid_recommendations = hybrid_recommendation(test_user_id, test_product_id, top_n=10)

print(f'Hybrid recommendations for user {test_user_id} and product {test_product_id}:')
print(hybrid_recommendations)
