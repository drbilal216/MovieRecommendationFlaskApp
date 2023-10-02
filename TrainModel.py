## 0. Load Dataset
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import pickle

## loaded file as pandas dataframe
ratings = pd.read_csv('ml-latest-small/ratings.csv')
movies = pd.read_csv('ml-latest-small/movies.csv')
links = pd.read_csv('ml-latest-small/links.csv')
tags = pd.read_csv('ml-latest-small/tags.csv')

## 1. Clean the data-set and Merging datasets needed
merged_dataset = pd.merge(movies, ratings, how='inner', on='movieId')

## 2. Select features to be focused on
# removing extra columns
merged_dataset2 = merged_dataset.drop(['timestamp'],axis=1)

## 3. Find correlation of the features and training ML Model
####################################
# pivot and create user-movie matrix
user_to_movie_df = merged_dataset2.pivot_table(index='userId',columns='title',values='rating').fillna(0)
#saving all movie names
movies_list = user_to_movie_df.columns
## transform matrix to scipy sparse matrix for ML model input
user_to_movie_sparse_df = csr_matrix(user_to_movie_df.values)

# using KNN model, Training Machine Learning Model for user-movie
knn_user_model = NearestNeighbors(metric='cosine', algorithm='brute')
knn_user_model.fit(user_to_movie_sparse_df)
####################################
# pivot and create movie-user matrix
movie_to_user_df = merged_dataset2.pivot_table(index='title',columns='userId',values='rating').fillna(0)
# Creating a dictionary with movie name as key and its index from the list as value:
movie_dict = {movie : index for index, movie in enumerate(movies_list)}
## transform matrix to scipy sparse matrix, will use this for our knn model input
movie_to_user_sparse_df = csr_matrix(movie_to_user_df.values)

# using KNN model, Training Machine Learning Model for movie-user
knn_movie_model = NearestNeighbors(metric='cosine', algorithm='brute')
knn_movie_model.fit(movie_to_user_sparse_df)
####################################

# Saving knn_user_model in binary mode 
knnPickle = open('knn_user_model', 'wb')
pickle.dump(knn_user_model, knnPickle)  
# close the file
knnPickle.close()
#################
# Saving knn_model_user_movie in binary mode 
knnPickle = open('knn_movie_model', 'wb')
pickle.dump(knn_movie_model, knnPickle)  
# close the file
knnPickle.close()
#################

## Saving dataframe for later use

user_to_movie_df.to_csv('user_to_movie_df.csv', index = False)
movie_to_user_df.to_csv('movie_to_user_df.csv', index = False)
merged_dataset2.to_csv('merged_dataset2.csv', index = False)

print("Data loaded, Model trained and saved as |knn_user_model, knn_movie_model| Sucessfully")